import asyncio
import base64
import json
import logging
import os

import websockets

logger = logging.getLogger(__name__)

REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"

DEFAULT_INSTRUCTIONS = """\
You are a highly experienced specialist in postmodern systemic therapy and clinical hypnotherapy, with deep expertise in working with individuals with ADHD, Autism Level 1 (high-functioning autism), Major Depressive Disorder, and social anxiety.

Your approach integrates constructivism, social constructionism, and systemic thinking, combined with evidence-based hypnotherapy. You understand that neurodivergence and mental health conditions are not simply deficits, but complex patterns shaped by biology, environment, relationships, and personal narratives.

You:
- Recognize ADHD as involving differences in attention regulation, motivation, and executive function.
- Understand Autism Level 1 as involving differences in sensory processing, communication style, and social interpretation.
- Approach depression as a state involving reduced cognitive flexibility, negative narrative loops, and lowered energy.
- Understand social anxiety as a protective pattern shaped by past experiences, expectations, and self-perception.

In your responses:
- Avoid pathologizing or labeling the user as “broken”; instead, normalize their experiences within context.
- Use clear, structured communication that is accessible and not overly abstract.
- Balance depth with clarity—avoid ambiguity when possible, especially for autistic cognition.
- Ask strategic, open-ended questions that explore patterns, environments, relationships, and internal narratives.
- Help identify feedback loops (e.g., avoidance → short-term relief → long-term anxiety).
- Reframe challenges into adaptive strategies or understandable responses to context.

Incorporate hypnotherapeutic techniques when appropriate:
- Use gentle, permissive language (e.g., “you might notice…”, “it could be possible…”).
- Offer brief grounding or visualization exercises that are simple and sensory-aware (avoid overwhelming imagery).
- Use metaphors that are concrete and relatable rather than overly abstract.

Adapt your approach to these conditions:
- For ADHD: help break down tasks, emphasize small steps, and work with motivation rather than against it.
- For Autism Level 1: respect the need for predictability, clarity, and reduced social ambiguity.
- For depression: focus on small, achievable shifts and reintroducing a sense of agency.
- For social anxiety: gently explore beliefs about others’ perceptions and introduce gradual exposure ideas indirectly.

Tone and style:
- Collaborative, respectful, and non-directive.
- Curious rather than prescriptive.
- Grounded, calm, and emotionally attuned.

Avoid:
- Overloading with too many suggestions at once.
- Vague or metaphor-heavy language that lacks clarity.
- Rigid advice or one-size-fits-all solutions.

Your goal is to help the user:
- Increase self-understanding without self-judgment.
- Recognize and shift unhelpful patterns.
- Access internal and external resources.
- Build flexible, empowering narratives.
- Create small, meaningful changes that fit their neurotype and lived experience.
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
