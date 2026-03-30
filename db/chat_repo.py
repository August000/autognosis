"""Chat conversation and message CRUD operations."""

from __future__ import annotations

import json
from uuid import UUID

import asyncpg


async def create_conversation(pool: asyncpg.Pool, user_id: str = "augusto", model: str = "gpt-4o-mini") -> dict:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO conversations (user_id, model) VALUES ($1, $2) RETURNING *",
            user_id, model,
        )
    return _conv_to_dict(row)


async def list_conversations(pool: asyncpg.Pool, user_id: str = "augusto", limit: int = 50) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM conversations WHERE user_id = $1 ORDER BY updated_at DESC LIMIT $2",
            user_id, limit,
        )
    return [_conv_to_dict(r) for r in rows]


async def get_conversation(pool: asyncpg.Pool, conversation_id: UUID) -> dict | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM conversations WHERE id = $1", conversation_id)
    return _conv_to_dict(row) if row else None


async def get_messages(pool: asyncpg.Pool, conversation_id: UUID, limit: int = 100) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM messages WHERE conversation_id = $1 ORDER BY created_at ASC LIMIT $2",
            conversation_id, limit,
        )
    return [_msg_to_dict(r) for r in rows]


async def add_message(
    pool: asyncpg.Pool,
    conversation_id: UUID,
    role: str,
    content: str,
    model: str | None = None,
    memory_context: list | None = None,
) -> dict:
    ctx_json = json.dumps(memory_context) if memory_context else None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO messages (conversation_id, role, content, model, memory_context)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            RETURNING *
            """,
            conversation_id, role, content, model, ctx_json,
        )
        # Update conversation's updated_at
        await conn.execute(
            "UPDATE conversations SET updated_at = NOW() WHERE id = $1",
            conversation_id,
        )
    return _msg_to_dict(row)


async def update_conversation_title(pool: asyncpg.Pool, conversation_id: UUID, title: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE conversations SET title = $1, updated_at = NOW() WHERE id = $2",
            title, conversation_id,
        )


async def delete_conversation(pool: asyncpg.Pool, conversation_id: UUID) -> bool:
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM conversations WHERE id = $1", conversation_id)
    return result == "DELETE 1"


# ── Helpers ─────────────────────────────────────────────────────────────


def _conv_to_dict(row: asyncpg.Record) -> dict:
    return {
        "id": str(row["id"]),
        "user_id": row["user_id"],
        "title": row["title"],
        "model": row["model"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }


def _msg_to_dict(row: asyncpg.Record) -> dict:
    mc = row["memory_context"]
    if isinstance(mc, str):
        mc = json.loads(mc)
    return {
        "id": str(row["id"]),
        "conversation_id": str(row["conversation_id"]),
        "role": row["role"],
        "content": row["content"],
        "model": row["model"],
        "memory_context": mc,
        "created_at": row["created_at"].isoformat(),
    }
