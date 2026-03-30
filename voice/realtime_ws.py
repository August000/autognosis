import asyncio
import base64
import json
import logging
import os

import websockets

logger = logging.getLogger(__name__)

REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"

DEFAULT_INSTRUCTIONS = """\
You are 
"""


class RealtimeClient:
    """Manages a WebSocket connection to the OpenAI Realtime API.

    Handles:
    - Session configuration (text-only output, server VAD, input transcription)
    - Sending base64-encoded PCM16 audio chunks
    - Yielding parsed server events as an async iterator
    """

    def __init__(self, api_key: str | None = None, instructions: str = DEFAULT_INSTRUCTIONS):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.instructions = instructions
        self._ws: websockets.WebSocketClientProtocol | None = None

    async def connect(self) -> None:
        """Open the WebSocket and send the initial session.update."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        self._ws = await websockets.connect(
            REALTIME_URL,
            additional_headers=headers,
            max_size=None,
        )
        logger.info("Connected to OpenAI Realtime API")

        # Configure the session: text-only output, semantic VAD, input transcription
        await self._send(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": "gpt-realtime",
                    "output_modalities": ["text"],
                    "instructions": self.instructions,
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "turn_detection": {"type": "semantic_vad"},
                            "transcription": {
                                "model": "gpt-4o-mini-transcribe",
                            },
                        },
                    },
                },
            }
        )
        logger.info("Session configured (text-only, server VAD, input transcription)")

    async def send_audio(self, pcm16_chunk: bytes) -> None:
        """Send a PCM16 audio chunk (raw bytes) to the input buffer."""
        b64 = base64.b64encode(pcm16_chunk).decode("ascii")
        await self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": b64,
            }
        )

    async def update_instructions(self, instructions: str) -> None:
        """Update session instructions mid-conversation (e.g. with memory context)."""
        self.instructions = instructions
        await self._send(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": "gpt-realtime",
                    "instructions": instructions,
                },
            }
        )

    async def receive_events(self):
        """Async generator that yields parsed JSON events from the server."""
        if self._ws is None:
            raise RuntimeError("Not connected – call connect() first")
        try:
            async for raw in self._ws:
                event = json.loads(raw)
                yield event
        except websockets.ConnectionClosed as e:
            logger.warning("WebSocket closed: %s", e)
        except asyncio.CancelledError:
            logger.info("Event receiver cancelled")

    async def close(self) -> None:
        """Gracefully close the WebSocket."""
        if self._ws:
            await self._ws.close()
            self._ws = None
            logger.info("WebSocket closed")

    # ── private ──────────────────────────────────────────────────────────

    async def _send(self, payload: dict) -> None:
        if self._ws is None:
            raise RuntimeError("Not connected – call connect() first")
        await self._ws.send(json.dumps(payload))
