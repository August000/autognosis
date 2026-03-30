import asyncio
import json
import logging
import os
import re
import signal

from dotenv import load_dotenv

from memory.client import mem
from voice.audio_capture import mic_stream
from voice.realtime_ws import RealtimeClient, DEFAULT_INSTRUCTIONS
from voice.tts import TTSPlayer

load_dotenv()
logger = logging.getLogger(__name__)

# Regex to split on sentence boundaries while keeping the delimiter
_SENTENCE_RE = re.compile(r'(?<=[.!?…])\s+')


class VoiceSession:
    """Orchestrates a live voice conversation.

    Flow:
        Mic → OpenAI Realtime (STT + LLM text) → ElevenLabs TTS → Speakers
        mem0 provides persistent memory across conversations.
    """

    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.realtime = RealtimeClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            instructions=DEFAULT_INSTRUCTIONS,
        )
        self.tts = TTSPlayer()

        # Async primitives
        self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)
        self.stop_event = asyncio.Event()
        self.mute_flag = asyncio.Event()  # set → mic is muted

        # Conversation state
        self._user_transcript = ""
        self._assistant_text = ""

        # Sentence-level streaming: as deltas arrive, we buffer text and
        # push complete sentences into this queue for the TTS speaker task.
        self._tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._text_buffer = ""  # partial sentence accumulator

    # ── public ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start the voice conversation loop."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown)

        # Load initial memory context
        instructions = await self._build_instructions()
        self.realtime.instructions = instructions

        await self.realtime.connect()
        print("\n🎙️  Solace is listening. Speak naturally. Press Ctrl-C to quit.\n")

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
            print("\n👋 Goodbye!\n")

    # ── private tasks ────────────────────────────────────────────────────

    async def _mic_sender(self) -> None:
        """Read PCM16 chunks from the queue and send them to OpenAI."""
        while not self.stop_event.is_set():
            try:
                chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.5)
                await self.realtime.send_audio(chunk)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _tts_speaker(self) -> None:
        """Consume sentences from the TTS queue and speak them sequentially.

        Mutes the mic during playback to prevent echo feedback.
        """
        while not self.stop_event.is_set():
            try:
                sentence = await asyncio.wait_for(self._tts_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if sentence is None:
                # End-of-response marker
                continue

            self.mute_flag.set()
            try:
                await self.tts.speak(sentence)
            finally:
                self.mute_flag.clear()

    def _flush_sentences(self, force: bool = False) -> None:
        """Extract complete sentences from the text buffer and enqueue them for TTS.

        If force=True (end of response), flush whatever remains.
        """
        if not self._text_buffer.strip():
            return

        parts = _SENTENCE_RE.split(self._text_buffer)

        if force:
            # Send everything remaining
            for part in parts:
                text = part.strip()
                if text:
                    self._tts_queue.put_nowait(text)
            self._text_buffer = ""
        else:
            # Only send complete sentences; keep the last fragment buffered
            for part in parts[:-1]:
                text = part.strip()
                if text:
                    self._tts_queue.put_nowait(text)
            self._text_buffer = parts[-1] if parts else ""

    async def _event_handler(self) -> None:
        """Process events from the OpenAI Realtime WebSocket."""
        async for event in self.realtime.receive_events():
            if self.stop_event.is_set():
                break

            etype = event.get("type", "")
            logger.debug("Event: %s", etype)

            if etype in ("session.created", "session.updated"):
                logger.info("Server %s: %s", etype, json.dumps(event, indent=2)[:2000])

            # ── User speech ───────────────────────────────────────────
            elif etype == "input_audio_buffer.speech_started":
                logger.debug("User started speaking")

            elif etype == "input_audio_buffer.speech_stopped":
                logger.debug("User stopped speaking")

            # ── User transcript ───────────────────────────────────────
            elif etype in (
                "conversation.item.input_audio_transcription.completed",
                "conversation.item.input_audio_transcription.done",
            ):
                self._user_transcript = event.get("transcript", "").strip()
                if self._user_transcript:
                    print(f"  You: {self._user_transcript}")

            # ── Assistant text deltas (stream to TTS sentence-by-sentence) ──
            elif etype in ("response.output_text.delta", "response.text.delta"):
                delta = event.get("delta", "")
                self._assistant_text += delta
                self._text_buffer += delta
                # Flush any complete sentences immediately
                self._flush_sentences(force=False)

            elif etype in ("response.output_text.done", "response.text.done"):
                self._assistant_text = event.get("text", self._assistant_text)

            # ── Response fully complete ───────────────────────────────
            elif etype == "response.done":
                # Flush any remaining partial sentence
                self._flush_sentences(force=True)
                # Send end-of-response marker
                self._tts_queue.put_nowait(None)

                assistant_text = self._assistant_text.strip()
                if assistant_text:
                    print(f"  Solace: {assistant_text}")

                    # Store memory in background
                    asyncio.create_task(
                        self._store_memory(self._user_transcript, assistant_text)
                    )
                    # Refresh instructions with fresh memory for next turn
                    asyncio.create_task(self._refresh_instructions())
                else:
                    logger.warning("response.done but no assistant text accumulated")

                # Reset for next turn
                self._user_transcript = ""
                self._assistant_text = ""
                self._text_buffer = ""

            # ── Errors ────────────────────────────────────────────────
            elif etype == "error":
                err = event.get("error", {})
                logger.error("Realtime API error: %s", err.get("message", event))

    # ── memory helpers ───────────────────────────────────────────────────

    async def _build_instructions(self, query: str | None = None) -> str:
        search_query = query or f"conversation context for user {self.user_id}"

        search_results = {}
        all_results = {}
        try:
            search_results, all_results = await asyncio.gather(
                asyncio.to_thread(
                    mem.search, search_query, user_id=self.user_id, limit=5
                ),
                asyncio.to_thread(mem.get_all, user_id=self.user_id),
            )
        except Exception as e:
            logger.warning("Memory retrieval failed: %s", e)

        memories = []
        vector_results = (
            search_results.get("results", [])
            if isinstance(search_results, dict)
            else search_results if isinstance(search_results, list) else []
        )
        for r in vector_results:
            text = r.get("memory", "") if isinstance(r, dict) else str(r)
            if text:
                memories.append(f"- {text}")

        relations = []
        graph_relations = (
            all_results.get("relations", [])
            if isinstance(all_results, dict)
            else []
        )
        for rel in graph_relations:
            if isinstance(rel, dict):
                src = rel.get("source", "")
                relationship = rel.get("relationship", "").replace("_", " ")
                tgt = rel.get("target", "")
                if src and tgt:
                    relations.append(f"- {src} {relationship} {tgt}")

        context_parts = []
        if memories:
            context_parts.append(
                "## Relevant memories about this user:\n" + "\n".join(memories)
            )
        if relations:
            context_parts.append(
                "## Known facts (knowledge graph):\n" + "\n".join(relations)
            )

        if context_parts:
            return f"{DEFAULT_INSTRUCTIONS}\n\n" + "\n\n".join(context_parts)
        return DEFAULT_INSTRUCTIONS

    async def _refresh_instructions(self) -> None:
        try:
            query = self._user_transcript or f"user {self.user_id}"
            instructions = await self._build_instructions(query)
            await self.realtime.update_instructions(instructions)
        except Exception as e:
            logger.warning("Failed to refresh instructions: %s", e)

    @staticmethod
    def _sanitize_for_cypher(text: str) -> str:
        """Strip characters that break mem0's Cypher query generation in Memgraph."""
        import re
        # Remove characters known to break unparameterised Cypher strings:
        #   /  \  '  "  `  { }  and control chars
        text = re.sub(r"[/\\'\"`{}]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    async def _store_memory(self, user_text: str, assistant_text: str) -> None:
        if not user_text:
            return
        try:
            safe_user = self._sanitize_for_cypher(user_text)
            safe_asst = self._sanitize_for_cypher(assistant_text)
            result = await asyncio.to_thread(
                mem.add,
                [
                    {"role": "user", "content": safe_user},
                    {"role": "assistant", "content": safe_asst},
                ],
                user_id=self.user_id,
            )
            stored = result.get("results", []) if isinstance(result, dict) else result
            logger.info("Memory stored: %s", stored)
        except Exception as e:
            logger.error("Failed to store memory: %s", e)

    # ── shutdown ─────────────────────────────────────────────────────────

    def _handle_shutdown(self) -> None:
        print("\n⏹️  Shutting down...")
        self.stop_event.set()
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()
