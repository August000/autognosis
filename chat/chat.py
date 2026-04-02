"""Chat service — memory-augmented conversation with selectable LLM models."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from uuid import UUID

import asyncpg
from openai import OpenAI

from db import chat_repo

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = [
    # Frontier / flagship (best reasoning + coding)
    "gpt-5.4-2026-03-05",
    "gpt-5.4-mini-2026-03-17",
    "gpt-5.4-nano-2026-03-17",

    # General-purpose high quality (cheaper than GPT-5)
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",

    # Multimodal + real-time (audio / vision / voice apps)
    "gpt-4o",
    "gpt-4o-mini",
]

SYSTEM_PROMPT_TEMPLATE = """You are Solace, a thoughtful assistant that helps the user explore and understand their personal knowledge graph — a living map of their thoughts, memories, and connections.

You have access to the user's memories and knowledge graph relations. Use them to provide deeply personalized, insightful responses. When relevant, reference specific memories or connections from their graph.

Be concise, warm, and intellectually curious. Help the user see patterns, connections, and insights they might have missed.

{memory_context}"""


class ChatService:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self._openai = OpenAI()

    def _get_mem_client(self):
        from memory.client import mem
        return mem

    async def _search_memories(self, query: str, user_id: str = "augusto") -> list[dict]:
        """Search mem0 for relevant memories."""
        mem = self._get_mem_client()
        try:
            results = await asyncio.to_thread(mem.search, query, user_id=user_id, limit=5)
            raw = results.get("results", []) if isinstance(results, dict) else results if isinstance(results, list) else []
            return [{"text": r.get("memory", "") if isinstance(r, dict) else str(r)} for r in raw if r]
        except Exception as e:
            logger.warning("Memory search failed: %s", e)
            return []

    async def _get_graph_relations(self, user_id: str = "augusto") -> list[dict]:
        """Get all graph relations from mem0."""
        mem = self._get_mem_client()
        try:
            result = await asyncio.to_thread(mem.get_all, user_id=user_id)
            raw = result.get("results", []) if isinstance(result, dict) else result if isinstance(result, list) else []
            return [{"text": r.get("memory", "") if isinstance(r, dict) else str(r)} for r in raw[:20] if r]
        except Exception as e:
            logger.warning("Graph relations fetch failed: %s", e)
            return []

    def _build_system_prompt(self, memories: list[dict], relations: list[dict]) -> str:
        """Build system prompt with memory context."""
        parts = []
        if memories:
            parts.append("Relevant memories:\n" + "\n".join(f"- {m['text']}" for m in memories if m.get("text")))
        if relations:
            parts.append("Knowledge graph relations:\n" + "\n".join(f"- {r['text']}" for r in relations if r.get("text")))

        context = "\n\n".join(parts) if parts else "No specific memories found for this query."
        return SYSTEM_PROMPT_TEMPLATE.format(memory_context=context)

    async def send_message(
        self,
        conversation_id: UUID,
        user_message: str,
        model: str = "gpt-5.4-2026-03-05",
        user_id: str = "augusto",
    ) -> AsyncGenerator[str, None]:
        """Process a user message and stream the assistant response."""
        if model not in SUPPORTED_MODELS:
            model = "gpt-4o"

        # Store user message
        await chat_repo.add_message(self.pool, conversation_id, "user", user_message)

        # Search memories for context
        memories, relations = await asyncio.gather(
            self._search_memories(user_message, user_id),
            self._get_graph_relations(user_id),
        )

        memory_context = [m for m in memories if m.get("text")]
        system_prompt = self._build_system_prompt(memories, relations)

        # Fetch conversation history
        history = await chat_repo.get_messages(self.pool, conversation_id, limit=50)

        # Build messages for LLM
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Stream response
        full_response = ""
        try:
            response = await asyncio.to_thread(
                self._openai.chat.completions.create,
                model=model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    full_response += delta.content
                    yield delta.content
        except Exception as e:
            logger.error("Chat LLM failed: %s", e)
            error_msg = "I encountered an error processing your message. Please try again."
            full_response = error_msg
            yield error_msg

        # Store assistant response
        await chat_repo.add_message(
            self.pool, conversation_id, "assistant", full_response,
            model=model, memory_context=memory_context,
        )

        # Auto-generate title from first message
        conv = await chat_repo.get_conversation(self.pool, conversation_id)
        if conv and not conv.get("title"):
            title = user_message[:60] + ("..." if len(user_message) > 60 else "")
            await chat_repo.update_conversation_title(self.pool, conversation_id, title)
