"""FastAPI WebSocket router for browser-based voice conversations.

Bridges the browser ↔ server ↔ OpenAI Realtime API:
- Mic audio is captured server-side (sounddevice)
- Transcripts and status events are streamed to the browser via WebSocket
- TTS playback happens server-side (ElevenLabs → speakers)
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from voice.session import VoiceSession

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice"])

# Single active voice session (one at a time)
_active_session: VoiceSession | None = None
_session_task: asyncio.Task | None = None


class BrowserVoiceSession(VoiceSession):
    """VoiceSession subclass that forwards transcript events to a WebSocket."""

    def __init__(self, ws: WebSocket, user_id: str = "default_user"):
        super().__init__(user_id=user_id)
        self._ws = ws

    async def _send_event(self, event_type: str, data: dict | None = None) -> None:
        """Send a JSON event to the browser."""
        payload = {"type": event_type}
        if data:
            payload.update(data)
        try:
            await self._ws.send_json(payload)
        except Exception:
            pass  # WebSocket may be closed

    async def _event_handler(self) -> None:
        """Override to also push transcript events to the browser."""
        async for event in self.realtime.receive_events():
            if self.stop_event.is_set():
                break

            etype = event.get("type", "")
            logger.debug("Event: %s", etype)

            if etype in ("session.created", "session.updated"):
                logger.info("Server %s", etype)

            # ── User speech ───────────────────────────────────────
            elif etype == "input_audio_buffer.speech_started":
                await self._send_event("speech_started")

            elif etype == "input_audio_buffer.speech_stopped":
                await self._send_event("speech_stopped")

            # ── User transcript ───────────────────────────────────
            elif etype in (
                "conversation.item.input_audio_transcription.completed",
                "conversation.item.input_audio_transcription.done",
            ):
                self._user_transcript = event.get("transcript", "").strip()
                if self._user_transcript:
                    print(f"  You: {self._user_transcript}")
                    await self._send_event("user_transcript", {"text": self._user_transcript})

            # ── Assistant text deltas ─────────────────────────────
            elif etype in ("response.output_text.delta", "response.text.delta"):
                delta = event.get("delta", "")
                self._assistant_text += delta
                self._text_buffer += delta
                self._flush_sentences(force=False)
                await self._send_event("assistant_delta", {"delta": delta})

            elif etype in ("response.output_text.done", "response.text.done"):
                self._assistant_text = event.get("text", self._assistant_text)

            # ── Response fully complete ───────────────────────────
            elif etype == "response.done":
                self._flush_sentences(force=True)
                self._tts_queue.put_nowait(None)

                assistant_text = self._assistant_text.strip()
                if assistant_text:
                    print(f"  Solace: {assistant_text}")
                    await self._send_event("assistant_done", {"text": assistant_text})
                    asyncio.create_task(
                        self._store_memory(self._user_transcript, assistant_text)
                    )
                    asyncio.create_task(self._refresh_instructions())

                self._user_transcript = ""
                self._assistant_text = ""
                self._text_buffer = ""

            # ── Errors ────────────────────────────────────────────
            elif etype == "error":
                err = event.get("error", {})
                msg = err.get("message", str(event))
                logger.error("Realtime API error: %s", msg)
                await self._send_event("error", {"message": msg})

    async def run(self) -> None:
        """Start the voice conversation (no signal handlers in web context)."""
        instructions = await self._build_instructions()
        self.realtime.instructions = instructions

        await self.realtime.connect()
        await self._send_event("connected")
        print("\n🎙️  Voice mode active via browser.\n")

        from voice.audio_capture import mic_stream

        tasks = [
            asyncio.create_task(mic_stream(self.audio_queue, self.stop_event, self.mute_flag)),
            asyncio.create_task(self._mic_sender()),
            asyncio.create_task(self._event_handler()),
            asyncio.create_task(self._tts_speaker()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.realtime.close()
            await self._send_event("disconnected")
            print("\n👋 Voice mode ended.\n")


@router.websocket("/ws/voice")
async def voice_websocket(ws: WebSocket):
    """WebSocket endpoint for voice conversations.

    The browser connects here to start/stop voice mode.
    Audio is captured and played back server-side.
    Transcripts are streamed to the browser for display.
    """
    global _active_session, _session_task

    await ws.accept()

    # Only allow one voice session at a time
    if _active_session is not None:
        await ws.send_json({"type": "error", "message": "Voice session already active"})
        await ws.close()
        return

    session = BrowserVoiceSession(ws)
    _active_session = session

    # Run the voice session as a background task
    _session_task = asyncio.create_task(session.run())

    try:
        # Keep the WebSocket alive; listen for control messages
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "stop":
                break
    except WebSocketDisconnect:
        logger.info("Voice WebSocket disconnected")
    except Exception as e:
        logger.error("Voice WebSocket error: %s", e)
    finally:
        # Shut down the voice session
        session.stop_event.set()
        if _session_task and not _session_task.done():
            _session_task.cancel()
            try:
                await _session_task
            except (asyncio.CancelledError, Exception):
                pass
        _active_session = None
        _session_task = None
