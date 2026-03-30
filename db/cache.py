"""Postgres-backed cache for LLM enrichments and API responses."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import asyncpg


# ── Enrichment cache ────────────────────────────────────────────────────


async def get_enrichment_cache(pool: asyncpg.Pool) -> dict | None:
    """Load the latest enrichment cache. Returns dict with same shape as the old in-memory cache, or None."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT node_descriptions, edge_labels, node_groups, cluster_names "
            "FROM enrichment_cache WHERE cache_key = 'latest'"
        )
    if not row:
        return None
    return {
        "node_descriptions": json.loads(row["node_descriptions"]) if isinstance(row["node_descriptions"], str) else dict(row["node_descriptions"]),
        "edge_labels": json.loads(row["edge_labels"]) if isinstance(row["edge_labels"], str) else dict(row["edge_labels"]),
        "node_groups": json.loads(row["node_groups"]) if isinstance(row["node_groups"], str) else dict(row["node_groups"]),
        "cluster_names": json.loads(row["cluster_names"]) if isinstance(row["cluster_names"], str) else dict(row["cluster_names"]),
    }


async def set_enrichment_cache(pool: asyncpg.Pool, data: dict) -> None:
    """Upsert the enrichment cache."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO enrichment_cache (cache_key, node_descriptions, edge_labels, node_groups, cluster_names, updated_at)
            VALUES ('latest', $1::jsonb, $2::jsonb, $3::jsonb, $4::jsonb, NOW())
            ON CONFLICT (cache_key) DO UPDATE SET
                node_descriptions = EXCLUDED.node_descriptions,
                edge_labels = EXCLUDED.edge_labels,
                node_groups = EXCLUDED.node_groups,
                cluster_names = EXCLUDED.cluster_names,
                updated_at = NOW()
            """,
            json.dumps(data["node_descriptions"]),
            json.dumps(data["edge_labels"]),
            json.dumps(data["node_groups"]),
            json.dumps(data["cluster_names"]),
        )


async def invalidate_enrichment_cache(pool: asyncpg.Pool) -> None:
    """Delete the enrichment cache row, forcing re-enrichment on next request."""
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM enrichment_cache WHERE cache_key = 'latest'")


# ── API response cache ──────────────────────────────────────────────────


async def get_api_cache(pool: asyncpg.Pool, cache_type: str, key: str) -> dict | None:
    """Fetch a cached API response, respecting TTL. Returns the response dict or None."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT response_json FROM api_cache
            WHERE cache_type = $1 AND cache_key = $2
              AND (expires_at IS NULL OR expires_at > NOW())
            """,
            cache_type, key,
        )
    if not row:
        return None
    val = row["response_json"]
    return json.loads(val) if isinstance(val, str) else dict(val)


async def set_api_cache(pool: asyncpg.Pool, cache_type: str, key: str, data: dict, ttl_hours: int = 24) -> None:
    """Upsert an API cache entry with TTL."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO api_cache (cache_type, cache_key, response_json, expires_at)
            VALUES ($1, $2, $3::jsonb, NOW() + make_interval(hours => $4))
            ON CONFLICT (cache_type, cache_key) DO UPDATE SET
                response_json = EXCLUDED.response_json,
                expires_at = EXCLUDED.expires_at,
                created_at = NOW()
            """,
            cache_type, key, json.dumps(data), ttl_hours,
        )
