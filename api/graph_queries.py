"""Memgraph query logic, LLM enrichment, and topology transformations."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections import defaultdict

from neo4j import GraphDatabase
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BOLT_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
BOLT_USER = os.getenv("MEMGRAPH_USER", "memgraph")
BOLT_PASS = os.getenv("MEMGRAPH_PASS", "memgraph")


def _get_driver():
    return GraphDatabase.driver(BOLT_URI, auth=(BOLT_USER, BOLT_PASS))


# ── Raw graph extraction ─────────────────────────────────────────────────


def get_raw_graph() -> dict:
    """Fetch all nodes and relationships from Memgraph."""
    driver = _get_driver()
    nodes_map: dict[str, dict] = {}
    links: list[dict] = []

    with driver.session() as session:
        result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m")
        for record in result:
            n = record["n"]
            m = record["m"]
            r = record["r"]

            for node in (n, m):
                nid = str(node.element_id)
                if nid not in nodes_map:
                    props = dict(node)
                    label = props.get("name", props.get("id", nid))
                    nodes_map[nid] = {
                        "id": nid,
                        "label": str(label),
                        "properties": props,
                    }

            links.append(
                {
                    "source": str(n.element_id),
                    "target": str(m.element_id),
                    "type": r.type,
                }
            )

    driver.close()

    # Also grab isolated nodes
    driver2 = _get_driver()
    with driver2.session() as session:
        result = session.run("MATCH (n) WHERE NOT (n)--() RETURN n")
        for record in result:
            node = record["n"]
            nid = str(node.element_id)
            if nid not in nodes_map:
                props = dict(node)
                label = props.get("name", props.get("id", nid))
                nodes_map[nid] = {
                    "id": nid,
                    "label": str(label),
                    "properties": props,
                }
    driver2.close()

    return {"nodes": list(nodes_map.values()), "links": links}


# ── LLM Batch Enrichment ─────────────────────────────────────────────────

BATCH_SIZE = 20


async def _batch_describe_nodes(client: OpenAI, nodes: list[dict], links: list[dict]) -> dict[int, str]:
    """Generate descriptions for nodes in batches. Returns {node_index: description}."""
    # Build neighbor context for each node
    neighbors: dict[str, list[str]] = defaultdict(list)
    node_map = {n["id"]: n["label"] for n in nodes}
    for link in links:
        src_label = node_map.get(link["source"], "?")
        tgt_label = node_map.get(link["target"], "?")
        rel = link.get("type", "relates_to").replace("_", " ")
        neighbors[link["source"]].append(f"{rel} {tgt_label}")
        neighbors[link["target"]].append(f"{src_label} {rel} this")

    idx_to_desc: dict[int, str] = {}
    batches = [nodes[i:i + BATCH_SIZE] for i in range(0, len(nodes), BATCH_SIZE)]

    async def process_batch(batch: list[dict], batch_idx: int) -> None:
        items = []
        for i, n in enumerate(batch):
            ctx = neighbors.get(n["id"], [])[:5]
            ctx_str = "; ".join(ctx) if ctx else "isolated concept"
            items.append({"idx": batch_idx * BATCH_SIZE + i, "name": n["label"], "context": ctx_str})

        prompt = (
            "You are analyzing a knowledge graph of someone's thoughts and memories.\n"
            "For each concept below, write a 1-sentence description (10-20 words) of what this "
            "concept likely represents in this person's thinking. Be introspective and human.\n"
            "Use only ASCII characters in your response.\n\n"
            "Return ONLY a JSON object with a 'results' array of objects with 'idx' (integer) and 'description' fields.\n"
            f"You MUST return exactly {len(batch)} results, one for each concept.\n\n"
            f"Concepts:\n{json.dumps(items, ensure_ascii=True)}"
        )

        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.6,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            parsed = json.loads(text)
            results = parsed.get("results", parsed.get("items", parsed.get("concepts", [])))
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict) and "idx" in item and "description" in item:
                        idx_to_desc[item["idx"]] = item["description"]
        except Exception as e:
            logger.warning("Batch node description failed: %s", e)

    await asyncio.gather(*[process_batch(b, i) for i, b in enumerate(batches)])
    return idx_to_desc


async def _batch_label_edges(client: OpenAI, links: list[dict], node_map: dict[str, str]) -> dict[int, str]:
    """Generate descriptive labels for edges in batches. Returns {link_index: label}."""
    idx_to_label: dict[int, str] = {}
    batches = [links[i:i + BATCH_SIZE] for i in range(0, len(links), BATCH_SIZE)]

    async def process_batch(batch: list[dict], batch_idx: int) -> None:
        items = []
        for i, link in enumerate(batch):
            src = node_map.get(link["source"], "?")
            tgt = node_map.get(link["target"], "?")
            rel = link.get("type", "relates_to").replace("_", " ")
            items.append({"idx": batch_idx * BATCH_SIZE + i, "source": src, "target": tgt, "relationship": rel})

        prompt = (
            "You are analyzing connections in someone's knowledge graph, a map of their thoughts.\n"
            "For each connection below, write a short phrase (3-7 words) that describes HOW these "
            "two ideas connect in this person's mind. Be descriptive and human, not mechanical.\n"
            "Use only ASCII characters in your response.\n\n"
            "Return ONLY a JSON object with a 'results' array of objects with 'idx' (integer) and 'label' fields.\n\n"
            f"Connections:\n{json.dumps(items, ensure_ascii=True)}"
        )

        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            parsed = json.loads(text)
            results = parsed.get("results", parsed.get("items", parsed.get("connections", [])))
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict) and "idx" in item and "label" in item:
                        idx_to_label[item["idx"]] = item["label"]
        except Exception as e:
            logger.warning("Batch edge labeling failed: %s", e)

    await asyncio.gather(*[process_batch(b, i) for i, b in enumerate(batches)])
    return idx_to_label


def _compute_clusters(nodes: list[dict], links: list[dict]) -> tuple[dict[str, int], list[list[str]]]:
    """Compute connected components. Returns (node_id -> group_id, list of components)."""
    adj: dict[str, set[str]] = defaultdict(set)
    for link in links:
        adj[link["source"]].add(link["target"])
        adj[link["target"]].add(link["source"])

    visited: set[str] = set()
    components: list[list[str]] = []
    node_ids = {n["id"] for n in nodes}

    for nid in node_ids:
        if nid in visited:
            continue
        component: list[str] = []
        queue = [nid]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(component)

    # Sort components by size descending so biggest cluster is group 0
    components.sort(key=len, reverse=True)

    node_group: dict[str, int] = {}
    for group_id, component in enumerate(components):
        for nid in component:
            node_group[nid] = group_id

    return node_group, components


async def _name_clusters(client: OpenAI, components: list[list[str]], node_map: dict[str, str]) -> dict[int, str]:
    """Use LLM to generate a short topic name for each cluster."""
    cluster_names: dict[int, str] = {}

    # Only name clusters with 2+ nodes
    items = []
    for i, comp in enumerate(components):
        labels = [node_map.get(nid, "?") for nid in comp[:15]]
        if len(comp) < 2:
            cluster_names[i] = labels[0] if labels else ""
            continue
        items.append({"idx": i, "concepts": labels})

    if not items:
        return cluster_names

    prompt = (
        "You are analyzing clusters of concepts in someone's knowledge graph.\n"
        "For each cluster below, write a 1-3 word topic name that captures the theme.\n"
        "Be concise and descriptive. Use only ASCII characters.\n\n"
        "Return ONLY a JSON object with a 'results' array of objects with 'idx' (integer) and 'name' fields.\n\n"
        f"Clusters:\n{json.dumps(items, ensure_ascii=True)}"
    )

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.5,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        parsed = json.loads(text)
        results = parsed.get("results", [])
        for item in results:
            if isinstance(item, dict) and "idx" in item and "name" in item:
                cluster_names[item["idx"]] = item["name"]
    except Exception as e:
        logger.warning("Cluster naming failed: %s", e)

    return cluster_names


async def enrich_graph(graph: dict, pool=None) -> dict:
    """Add LLM-generated descriptions, labels, clusters, and cluster names.

    Results are cached in Postgres (if pool provided) so enrichments persist across restarts.
    """
    # Try loading from Postgres cache
    if pool:
        from db.cache import get_enrichment_cache
        cached = await get_enrichment_cache(pool)
        if cached is not None:
            node_descs = cached["node_descriptions"]
            edge_labels = cached["edge_labels"]
            node_groups = cached["node_groups"]
            cluster_names = cached["cluster_names"]

            for node in graph["nodes"]:
                node["description"] = node_descs.get(node["id"], "")
                node["group"] = node_groups.get(node["id"], 0)

            for link in graph["links"]:
                key = f"{link['source']}|{link['target']}"
                link["label"] = edge_labels.get(key, link.get("type", "").replace("_", " "))

            graph["clusterNames"] = cluster_names
            logger.info("Loaded enrichment from Postgres cache")
            return graph

    logger.info("Enriching graph with LLM descriptions (first load)...")
    client = OpenAI()
    node_map = {n["id"]: n["label"] for n in graph["nodes"]}

    # Compute clusters
    node_groups, components = _compute_clusters(graph["nodes"], graph["links"])

    # Run node descriptions, edge labels, and cluster naming in parallel
    idx_node_descs, idx_edge_labels, cluster_names = await asyncio.gather(
        _batch_describe_nodes(client, graph["nodes"], graph["links"]),
        _batch_label_edges(client, graph["links"], node_map),
        _name_clusters(client, components, node_map),
    )

    # Convert index-based results to id/key-based for stable caching
    node_descs: dict[str, str] = {}
    for i, node in enumerate(graph["nodes"]):
        node_descs[node["id"]] = idx_node_descs.get(i, "")

    edge_labels: dict[str, str] = {}
    for i, link in enumerate(graph["links"]):
        key = f"{link['source']}|{link['target']}"
        edge_labels[key] = idx_edge_labels.get(i, link.get("type", "").replace("_", " "))

    cache_data = {
        "node_descriptions": node_descs,
        "edge_labels": edge_labels,
        "node_groups": node_groups,
        "cluster_names": cluster_names,
    }

    # Persist to Postgres
    if pool:
        from db.cache import set_enrichment_cache
        await set_enrichment_cache(pool, cache_data)
        logger.info("Saved enrichment to Postgres cache")

    for node in graph["nodes"]:
        node["description"] = node_descs.get(node["id"], "")
        node["group"] = node_groups.get(node["id"], 0)

    for link in graph["links"]:
        key = f"{link['source']}|{link['target']}"
        link["label"] = edge_labels.get(key, link.get("type", "").replace("_", " "))

    graph["clusterNames"] = cluster_names

    logger.info("Enrichment complete: %d nodes, %d edges, %d clusters",
                len(node_descs), len(edge_labels), len(cluster_names))
    return graph


# ── Memory search (Qdrant via mem0) ──────────────────────────────────────


def get_node_memories(node_label: str, user_id: str = "augusto") -> list[dict]:
    """Search Qdrant for memories related to a node concept."""
    from memory.client import mem

    try:
        results = mem.search(node_label, user_id=user_id, limit=3)
        memories = []
        raw = results.get("results", []) if isinstance(results, dict) else results if isinstance(results, list) else []
        for r in raw:
            text = r.get("memory", "") if isinstance(r, dict) else str(r)
            if text:
                memories.append({"text": text})
        return memories
    except Exception as e:
        logger.warning("Memory search failed for '%s': %s", node_label, e)
        return []


# ── Search ───────────────────────────────────────────────────────────────


def search_nodes(query: str, graph: dict) -> list[dict]:
    """Fuzzy search nodes by label. Returns best matches sorted by relevance."""
    query_lower = query.lower().strip()
    if not query_lower:
        return []

    scored = []
    for node in graph["nodes"]:
        label = (node.get("label") or "").lower()
        if not label:
            continue
        # Exact match
        if label == query_lower:
            scored.append((0, node))
        # Starts with
        elif label.startswith(query_lower):
            scored.append((1, node))
        # Contains
        elif query_lower in label:
            scored.append((2, node))
        # Any query word appears in label
        elif any(w in label for w in query_lower.split()):
            scored.append((3, node))

    scored.sort(key=lambda x: x[0])
    return [s[1] for s in scored[:10]]


# ── Suggest related concepts for leaf nodes ──────────────────────────────


async def suggest_related(node_label: str, existing_labels: list[str], pool=None) -> list[dict]:
    """Use LLM to suggest concepts related to a leaf node using first principles."""
    # Check Postgres cache
    if pool:
        from db.cache import get_api_cache
        cached = await get_api_cache(pool, "node_suggest", node_label)
        if cached is not None:
            return cached.get("suggestions", [])

    existing_set = {l.lower() for l in existing_labels}

    client = OpenAI()
    prompt = (
        "You are helping someone explore their knowledge graph by suggesting related concepts.\n"
        f"The concept is: '{node_label}'\n"
        f"Concepts already in their graph: {json.dumps(existing_labels[:50])}\n\n"
        "Using first principles thinking, suggest 3-5 related concepts that would deepen "
        "understanding. For each suggestion:\n"
        "- Break the concept down to its fundamental components\n"
        "- Suggest concepts that reveal WHY this thing exists, HOW it works, or WHAT it connects to\n"
        "- Prefer foundational concepts over surface-level associations\n"
        "- Do NOT suggest concepts already in the graph\n\n"
        "Return a JSON object with a 'suggestions' array of objects with 'label' and 'reason' fields.\n"
        "The reason should explain the first-principles connection (10-15 words)."
    )

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        parsed = json.loads(text)
        results = parsed.get("suggestions", [])
        suggestions = [
            s for s in results
            if isinstance(s, dict) and s.get("label", "").lower() not in existing_set
        ][:5]

        # Cache in Postgres
        if pool and suggestions:
            from db.cache import set_api_cache
            await set_api_cache(pool, "node_suggest", node_label, {"suggestions": suggestions}, ttl_hours=24)

        return suggestions
    except Exception as e:
        logger.warning("Suggest related failed for '%s': %s", node_label, e)
        return []


# ── Personal insight (AI-inferred meaning) ─────────────────────────────


def _is_self_node(node_label: str, user_id: str) -> bool:
    label = (node_label or "").strip().lower()
    user = (user_id or "").strip().lower()
    return label in {user, "me", "myself", "self"}


def _trim_texts(items: list[str], limit: int = 10, max_chars: int = 300) -> list[str]:
    return [item[:max_chars] for item in items if item][:limit]


def _label_tokens(node_label: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", (node_label or "").lower()) if len(token) >= 3]


def _matches_concept_text(text: str, node_label: str) -> bool:
    text_lower = (text or "").lower()
    label_lower = (node_label or "").strip().lower()
    if not text_lower or not label_lower:
        return False
    if label_lower in text_lower:
        return True

    tokens = _label_tokens(node_label)
    if not tokens:
        return False

    matches = sum(1 for token in tokens if token in text_lower)
    if len(tokens) == 1:
        return matches == 1
    return matches >= max(1, len(tokens) - 1)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    unique = []
    for item in items:
        normalized = item.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(item)
    return unique


def _build_evidence_preview(evidence: dict, limit: int = 3) -> dict[str, list[str]]:
    return {
        "memories": evidence["memory_texts"][:limit],
        "messages": evidence["message_texts"][:limit],
        "relations": evidence["relation_texts"][:limit],
    }


async def collect_insight_evidence(node_label: str, pool=None, user_id: str = "augusto") -> dict:
    """Collect evidence used to generate a personal insight."""
    label_lower = (node_label or "").lower()
    is_self = _is_self_node(node_label, user_id)

    memory_texts: list[str] = []
    message_texts: list[str] = []
    relation_texts: list[str] = []

    try:
        from memory.client import mem
    except Exception as e:
        logger.warning("Memory client unavailable for insight '%s': %s", node_label, e)
        mem = None

    if is_self:
        if mem is not None:
            try:
                result = await asyncio.to_thread(mem.get_all, user_id=user_id)
                raw = result.get("results", []) if isinstance(result, dict) else result if isinstance(result, list) else []
                all_texts = []
                for r in raw:
                    text = r.get("memory", "") if isinstance(r, dict) else str(r)
                    if text:
                        all_texts.append(text)
                memory_texts = _trim_texts(all_texts, limit=12, max_chars=400)
                relation_texts = _trim_texts(all_texts, limit=12, max_chars=220)
            except Exception as e:
                logger.warning("Self insight memory fetch failed for '%s': %s", node_label, e)

        if pool:
            from db.chat_repo import list_recent_user_messages
            message_texts = _trim_texts(
                await list_recent_user_messages(pool, user_id=user_id, limit=15),
                limit=12,
                max_chars=300,
            )
    else:
        memories = await asyncio.to_thread(get_node_memories, node_label, user_id)
        memory_texts = _trim_texts([m["text"] for m in memories if m.get("text")], limit=10, max_chars=400)

        if pool:
            from db.chat_repo import list_recent_user_messages, search_user_messages
            direct_messages = await search_user_messages(pool, node_label, user_id=user_id, limit=15)
            recent_messages = await list_recent_user_messages(pool, user_id=user_id, limit=40)
            fuzzy_messages = [msg for msg in recent_messages if _matches_concept_text(msg, node_label)]
            message_texts = _trim_texts(
                _dedupe_preserve_order(direct_messages + fuzzy_messages),
                limit=10,
                max_chars=300,
            )

        if mem is not None:
            try:
                result = await asyncio.to_thread(mem.get_all, user_id=user_id)
                raw = result.get("results", []) if isinstance(result, dict) else result if isinstance(result, list) else []
                matches = []
                for r in raw:
                    text = r.get("memory", "") if isinstance(r, dict) else str(r)
                    if text and _matches_concept_text(text, node_label):
                        matches.append(text)
                relation_texts = _trim_texts(matches, limit=10, max_chars=220)
            except Exception as e:
                logger.warning("Graph relations fetch failed for insight '%s': %s", node_label, e)

    counts = {
        "memories": len(memory_texts),
        "messages": len(message_texts),
        "relations": len(relation_texts),
    }

    evidence = {
        "node_label": node_label,
        "is_self": is_self,
        "memory_texts": memory_texts,
        "message_texts": message_texts,
        "relation_texts": relation_texts,
        "counts": counts,
        "source_count": sum(counts.values()),
    }
    logger.info(
        "Insight evidence for '%s': self=%s memories=%d messages=%d relations=%d",
        node_label,
        is_self,
        counts["memories"],
        counts["messages"],
        counts["relations"],
    )
    return evidence


async def generate_personal_insight(evidence: dict) -> dict:
    """Generate a personal insight from collected evidence."""
    node_label = evidence["node_label"]
    source_count = evidence["source_count"]

    if source_count == 0:
        return {
            "status": "no_sources",
            "insight": None,
            "source_count": 0,
            "strategy": "self" if evidence["is_self"] else "concept",
            "counts": evidence["counts"],
            "reason": "no_evidence",
            "evidence_preview": _build_evidence_preview(evidence),
        }

    context_parts = []
    if evidence["memory_texts"]:
        context_parts.append("Memories:\n" + "\n".join(f"- {t}" for t in evidence["memory_texts"]))
    if evidence["message_texts"]:
        context_parts.append("Things they've said:\n" + "\n".join(f"- {t}" for t in evidence["message_texts"]))
    if evidence["relation_texts"]:
        context_parts.append("Graph relations:\n" + "\n".join(f"- {t}" for t in evidence["relation_texts"]))

    context = "\n\n".join(context_parts)

    if evidence["is_self"]:
        prompt = (
            f"You are analysing someone's personal knowledge graph to understand who they are. "
            f"Their name is \"{node_label}\".\n\n"
            f"Here is personal evidence drawn from their memories, graph, and conversations:\n\n"
            f"{context}\n\n"
            "Write a concise self-reflection in second person (2-4 sentences) about who they seem to be right now. "
            "Focus on recurring themes, tensions, values, motivations, and emotional patterns. "
            "Be warm, specific, and grounded in the evidence. Do not mention that you are using data."
        )
    else:
        prompt = (
            f"You are analysing someone's personal knowledge graph to understand how they relate to the concept: \"{node_label}\".\n\n"
            f"Here is everything you know about how this person has talked about or interacted with \"{node_label}\":\n\n"
            f"{context}\n\n"
            "Based on this, write a concise personal insight (2-4 sentences) about what this concept means to them. "
            "How do they relate to it? What patterns do you see? Are there tensions or recurring themes?\n"
            "Write in second person (\"You tend to...\", \"You see...\"). "
            "Be warm, thoughtful, and specific. Do NOT be generic or repeat dictionary definitions."
        )

    try:
        client = OpenAI()
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.7,
        )
        insight_text = response.choices[0].message.content.strip()
        return {
            "status": "ok",
            "insight": insight_text,
            "source_count": source_count,
            "strategy": "self" if evidence["is_self"] else "concept",
            "counts": evidence["counts"],
            "reason": "generated",
            "evidence_preview": _build_evidence_preview(evidence),
        }
    except Exception as e:
        logger.warning("Personal insight generation failed for '%s': %s", node_label, e)
        return {
            "status": "error",
            "insight": None,
            "source_count": source_count,
            "strategy": "self" if evidence["is_self"] else "concept",
            "counts": evidence["counts"],
            "reason": "llm_error",
            "evidence_preview": _build_evidence_preview(evidence),
        }


async def get_personal_insight(node_label: str, pool=None, user_id: str = "augusto") -> dict | None:
    """Synthesise what a concept means to the user based on their conversations and memories."""

    # 1. Check cache first
    if pool:
        from db.cache import get_api_cache
        cached = await get_api_cache(pool, "personal_insight", node_label)
        if cached is not None:
            return cached

    evidence = await collect_insight_evidence(node_label, pool=pool, user_id=user_id)
    result = await generate_personal_insight(evidence)

    if pool and result.get("status") == "ok":
        from db.cache import set_api_cache
        await set_api_cache(pool, "personal_insight", node_label, result, ttl_hours=24)

    return result


# ── Detailed knowledge panel ─────────────────────────────────────────────


async def get_concept_detail(node_label: str, neighbors: list[str], pool=None) -> dict:
    """Fetch a Wikipedia article for a concept. Returns summary + full article."""
    # Check Postgres cache
    if pool:
        from db.cache import get_api_cache
        cached = await get_api_cache(pool, "node_detail", node_label)
        if cached is not None:
            return cached

    import wikipediaapi

    def _fetch():
        wiki = wikipediaapi.Wikipedia(
            user_agent="Solace/1.0 (Knowledge Graph Visualizer)",
            language="en",
        )
        page = wiki.page(node_label)
        if not page.exists():
            return None

        title = page.title
        summary = page.summary
        full_article = page.text

        return {
            "summary": summary,
            "full_article": full_article,
            "title": title,
            "url": page.fullurl,
        }

    try:
        result = await asyncio.to_thread(_fetch)
        if not result:
            return {"summary": "", "full_article": "", "title": node_label, "url": ""}

        # Cache in Postgres
        if pool:
            from db.cache import set_api_cache
            await set_api_cache(pool, "node_detail", node_label, result, ttl_hours=72)

        return result
    except Exception as e:
        logger.warning("Wikipedia fetch failed for '%s': %s", node_label, e)
        return {"summary": "", "full_article": "", "title": node_label, "url": ""}


# ── Materialize ghost node into Memgraph ─────────────────────────────────


async def materialize_ghost_node(label: str, parent_id: str, reason: str, pool=None) -> dict:
    """Create a new node in Memgraph and connect it to the parent node."""

    def _create():
        driver = _get_driver()
        new_node = None

        with driver.session() as session:
            result = session.run(
                "MATCH (parent) WHERE id(parent) = toInteger($parent_id) "
                "CREATE (n {name: $label})-[r:SUGGESTED_BY]->(parent) "
                "RETURN n, r",
                parent_id=parent_id,
                label=label,
            )
            record = result.single()
            if record:
                node = record["n"]
                nid = str(node.element_id)
                new_node = {
                    "id": nid,
                    "label": label,
                    "properties": dict(node),
                    "description": reason,
                }

        driver.close()
        return new_node

    new_node = await asyncio.to_thread(_create)

    # Invalidate enrichment cache so the new node gets descriptions & clustering
    if new_node and pool:
        from db.cache import invalidate_enrichment_cache
        await invalidate_enrichment_cache(pool)

    return new_node


# ── Topology: Centralized ────────────────────────────────────────────────


def to_centralized(graph: dict) -> dict:
    """Restructure around the highest-degree node as the single core."""
    nodes = graph["nodes"]
    links = graph["links"]

    if not nodes:
        return {"nodes": [], "links": []}

    degree: dict[str, int] = defaultdict(int)
    for link in links:
        degree[link["source"]] += 1
        degree[link["target"]] += 1

    core_id = max(degree, key=degree.get) if degree else nodes[0]["id"]

    out_nodes = []
    for n in nodes:
        out_nodes.append({
            **n,
            "isCore": n["id"] == core_id,
            "group": 0,
        })

    connected_to_core = {core_id}
    for link in links:
        if link["source"] == core_id:
            connected_to_core.add(link["target"])
        if link["target"] == core_id:
            connected_to_core.add(link["source"])

    out_links = list(links)
    for n in nodes:
        if n["id"] not in connected_to_core and n["id"] != core_id:
            out_links.append({
                "source": core_id,
                "target": n["id"],
                "type": "centralized_link",
                "label": "connected through core",
            })

    return {"nodes": out_nodes, "links": out_links, "coreId": core_id}


# ── Topology: Decentralized ──────────────────────────────────────────────


def to_decentralized(graph: dict) -> dict:
    """Cluster nodes into themed hubs. Groups already assigned by enrich_graph."""
    nodes = graph["nodes"]
    links = graph["links"]

    if not nodes:
        return {"nodes": [], "links": []}

    # Groups already assigned by enrich_graph; find hub of each group
    groups: dict[int, list[dict]] = defaultdict(list)
    for n in nodes:
        groups[n.get("group", 0)].append(n)

    hub_nodes: set[str] = set()
    degree: dict[str, int] = defaultdict(int)
    for link in links:
        degree[link["source"]] += 1
        degree[link["target"]] += 1

    for group_id, members in groups.items():
        hub = max(members, key=lambda n: degree.get(n["id"], 0))
        hub_nodes.add(hub["id"])

    out_nodes = [{**n, "isHub": n["id"] in hub_nodes} for n in nodes]

    return {
        "nodes": out_nodes,
        "links": links,
        "clusters": len(groups),
        "clusterNames": graph.get("clusterNames", {}),
    }
