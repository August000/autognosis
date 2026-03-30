"""Database schema DDL and initialization."""

from __future__ import annotations

import asyncpg

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS enrichment_cache (
    id              SERIAL PRIMARY KEY,
    cache_key       TEXT NOT NULL UNIQUE,
    node_descriptions  JSONB NOT NULL DEFAULT '{}',
    edge_labels        JSONB NOT NULL DEFAULT '{}',
    node_groups        JSONB NOT NULL DEFAULT '{}',
    cluster_names      JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS api_cache (
    id              SERIAL PRIMARY KEY,
    cache_type      TEXT NOT NULL,
    cache_key       TEXT NOT NULL,
    response_json   JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ,
    UNIQUE(cache_type, cache_key)
);

CREATE INDEX IF NOT EXISTS idx_api_cache_lookup ON api_cache(cache_type, cache_key);
CREATE INDEX IF NOT EXISTS idx_api_cache_expiry ON api_cache(expires_at) WHERE expires_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS conversations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT NOT NULL DEFAULT 'augusto',
    title           TEXT,
    model           TEXT NOT NULL DEFAULT 'gpt-4o-mini',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content         TEXT NOT NULL,
    model           TEXT,
    memory_context  JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at DESC);
"""


async def init_db(pool: asyncpg.Pool) -> None:
    """Create all tables if they don't exist."""
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)
