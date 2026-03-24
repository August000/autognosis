import asyncio
import logging
import os

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000  # Hz – required by OpenAI Realtime API
CHANNELS = 1
DTYPE = "int16"  # PCM16
CHUNK_DURATION_MS = 200
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 4800 samples


def _find_working_input_device() -> int | None:
    """Return the index of the first input device that appears to be active.

    Prefers the system default if it works; otherwise iterates through
    available input devices and picks the first one that can open a short
    test stream without errors (skipping virtual/aggregate devices that
    tend to return silence).
    """
    devices = sd.query_devices()

    # Try default first
    try:
        default_idx = sd.default.device[0]  # input index
        if default_idx is not None:
            info = sd.query_devices(default_idx)
            if info["max_input_channels"] >= 1:
                # Quick probe: open a tiny stream and read a few frames
                if _device_has_signal(default_idx):
                    return default_idx
                logger.info(
                    "Default device [%d] '%s' returned silence, trying others…",
                    default_idx,
                    info["name"],
                )
    except Exception as e:
        logger.debug("Could not probe default device: %s", e)

    # Walk all input devices and pick the first with a real signal
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] < 1:
            continue
        if idx == (sd.default.device[0] if sd.default.device[0] is not None else -1):
            continue  # already tried
        try:
            if _device_has_signal(idx):
                logger.info("Auto-selected input device [%d] '%s'", idx, dev["name"])
                return idx
        except Exception as e:
            logger.debug("Device [%d] '%s' probe failed: %s", idx, dev["name"], e)

    # Fallback: just return the default and hope for the best
    logger.warning("No device with signal found; falling back to system default")
    return None


def _device_has_signal(device_idx: int, duration_s: float = 0.3) -> bool:
    """Record a short clip from *device_idx* and return True if peak > 0."""
    try:
        frames = int(SAMPLE_RATE * duration_s)
        recording = sd.rec(
            frames,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            device=device_idx,
        )
        sd.wait()
        peak = int(np.max(np.abs(recording)))
        logger.debug("Probe device [%d]: peak=%d", device_idx, peak)
        return peak > 0
    except Exception as e:
        logger.debug("Probe device [%d] error: %s", device_idx, e)
        return False


def _resolve_device() -> int | None:
    """Determine which input device to use.

    Priority:
      1. AUDIO_INPUT_DEVICE env var (integer index or device name substring)
      2. Auto-detection via _find_working_input_device()
    """
    env_val = os.getenv("AUDIO_INPUT_DEVICE", "").strip()
    if env_val:
        # Try as integer index
        try:
            idx = int(env_val)
            info = sd.query_devices(idx)
            if info["max_input_channels"] >= 1:
                logger.info("Using device from AUDIO_INPUT_DEVICE: [%d] '%s'", idx, info["name"])
                return idx
        except (ValueError, sd.PortAudioError):
            pass
        # Try as name substring
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] >= 1 and env_val.lower() in dev["name"].lower():
                logger.info("Using device matching '%s': [%d] '%s'", env_val, idx, dev["name"])
                return idx
        logger.warning("AUDIO_INPUT_DEVICE='%s' not found; auto-detecting…", env_val)

    return _find_working_input_device()


async def mic_stream(
    queue: asyncio.Queue[bytes],
    stop_event: asyncio.Event,
    mute_flag: asyncio.Event | None = None,
) -> None:
    """Capture microphone audio and feed PCM16 chunks into *queue*.

    Args:
        queue: asyncio.Queue to put raw PCM16 byte chunks into.
        stop_event: Set this event to stop capturing.
        mute_flag: When this event is *set*, audio is silenced (zeros sent)
                   to prevent echo during TTS playback.
    """
    # Resolve device (env var → auto-detect)
    device = await asyncio.to_thread(_resolve_device)
    device_info = sd.query_devices(device) if device is not None else sd.query_devices(sd.default.device[0])
    print(f"🎤 Microphone: {device_info['name']}")

    loop = asyncio.get_running_loop()

    _chunk_count = 0

    def _audio_callback(indata: np.ndarray, frames: int, time_info, status):
        nonlocal _chunk_count
        if status:
            logger.warning("sounddevice status: %s", status)
        # Explicit copy – indata references a temporary C buffer
        data = indata.copy()
        pcm_bytes = data.tobytes()
        if mute_flag and mute_flag.is_set():
            # Send silence instead of real mic data
            pcm_bytes = b"\x00" * len(pcm_bytes)
        # Log audio level periodically to verify mic is working
        _chunk_count += 1
        if _chunk_count % 25 == 1:  # every ~5 seconds
            peak = np.max(np.abs(data))
            logger.debug("Mic chunk #%d: peak=%d, bytes=%d", _chunk_count, peak, len(pcm_bytes))
        try:
            loop.call_soon_threadsafe(queue.put_nowait, pcm_bytes)
        except asyncio.QueueFull:
            pass  # drop frame rather than block audio thread

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=CHUNK_SAMPLES,
        device=device,
        callback=_audio_callback,
    )

    with stream:
        logger.info(
            "Microphone open – device [%s] %s, %d Hz, %d ch, %d ms chunks",
            device,
            device_info["name"],
            SAMPLE_RATE,
            CHANNELS,
            CHUNK_DURATION_MS,
        )
        # Block until stop_event is set
        while not stop_event.is_set():
            await asyncio.sleep(0.1)

    logger.info("Microphone closed")
