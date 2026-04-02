import asyncio
import logging
import os

from elevenlabs.client import ElevenLabs
from elevenlabs import stream as el_stream

logger = logging.getLogger(__name__)

# Default voice – can be overridden via env or constructor
DEFAULT_VOICE_ID = "gJx1vCzNCD1EQHT212Ls"
DEFAULT_MODEL = "eleven_turbo_v2_5"


class TTSPlayer:
    """Streams text → ElevenLabs TTS → speaker playback via mpv."""

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        model_id: str = DEFAULT_MODEL,
    ):
        self.client = ElevenLabs(api_key=api_key or os.getenv("ELEVENLABS_API_KEY", ""))
        self.voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID)
        self.model_id = model_id

    async def speak(self, text: str) -> None:
        """Convert *text* to speech and play it through speakers.

        This wraps the synchronous ElevenLabs SDK in asyncio.to_thread()
        so the event loop stays free while audio is playing.
        """
        if not text.strip():
            return
        logger.info("TTS speaking: %s", text[:80])
        await asyncio.to_thread(self._blocking_speak, text)

    def _blocking_speak(self, text: str) -> None:
        """Synchronous: stream TTS audio and play via mpv."""
        audio_iter = self.client.text_to_speech.stream(
            voice_id=self.voice_id,
            text=text,
            model_id=self.model_id,
            output_format="mp3_44100_128",
        )
        el_stream(audio_iter)
