"""Memgraph query logic and topology transformations."""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict

from neo4j import GraphDatabase
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

BOLT_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
BOLT_USER = os.getenv("MEMGRAPH_USER", "memgraph")
BOLT_PASS = os.getenv("MEMGRAPH_PASS", "memgraph")


def _get_driver():
    return GraphDatabase.driver(BOLT_URI, auth=(BOLT_USER, BOLT_PASS))


# ── Raw graph extraction ─────────────────────────────────────────────────


def get_raw_graph() -> dict:
    """Fetch all nodes and relationships from Memgraph.

    Returns {nodes: [{id, label, properties}], links: [{source, target, type}]}
    """
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
        result = session.run(
            "MATCH (n) WHERE NOT (n)--() RETURN n"
        )
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


# ── Topology: Centralized ────────────────────────────────────────────────


def to_centralized(graph: dict) -> dict:
    """Restructure around the highest-degree node as the single core.

    All nodes connect to the core; original edges kept but core is flagged.
    """
    nodes = graph["nodes"]
    links = graph["links"]

    if not nodes:
        return {"nodes": [], "links": []}

    # Count degree per node
    degree: dict[str, int] = defaultdict(int)
    for link in links:
        degree[link["source"]] += 1
        degree[link["target"]] += 1

    # Find hub (highest degree)
    core_id = max(degree, key=degree.get) if degree else nodes[0]["id"]

    # Mark nodes
    out_nodes = []
    for n in nodes:
        out_nodes.append({
            **n,
            "isCore": n["id"] == core_id,
            "group": 0,
        })

    # Keep original links + add core→node links for any unconnected nodes
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
            })

    return {"nodes": out_nodes, "links": out_links, "coreId": core_id}


# ── Topology: Decentralized ──────────────────────────────────────────────


def to_decentralized(graph: dict) -> dict:
    """Cluster nodes into themed hubs using connected components."""
    nodes = graph["nodes"]
    links = graph["links"]

    if not nodes:
        return {"nodes": [], "links": []}

    # Build adjacency for connected components
    adj: dict[str, set[str]] = defaultdict(set)
    for link in links:
        adj[link["source"]].add(link["target"])
        adj[link["target"]].add(link["source"])

    # BFS to find connected components
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

    # Assign group IDs
    node_group: dict[str, int] = {}
    hub_nodes: list[str] = []
    for group_id, component in enumerate(components):
        # Find hub of each component (highest local degree)
        local_degree: dict[str, int] = defaultdict(int)
        for link in links:
            if link["source"] in component and link["target"] in component:
                local_degree[link["source"]] += 1
                local_degree[link["target"]] += 1

        hub_id = max(component, key=lambda x: local_degree.get(x, 0))
        hub_nodes.append(hub_id)

        for nid in component:
            node_group[nid] = group_id

    out_nodes = []
    for n in nodes:
        out_nodes.append({
            **n,
            "group": node_group.get(n["id"], 0),
            "isHub": n["id"] in hub_nodes,
        })

    return {
        "nodes": out_nodes,
        "links": links,
        "clusters": len(components),
    }


# ── Topology: Distributed ────────────────────────────────────────────────


async def to_distributed(graph: dict) -> dict:
    """Add LLM-generated edge labels describing how ideas connect."""
    nodes = graph["nodes"]
    links = graph["links"]

    if not links:
        return {"nodes": [{"group": 0, **n} for n in nodes], "links": []}

    # Build node lookup
    node_map = {n["id"]: n for n in nodes}

    client = OpenAI()

    async def label_edge(link: dict) -> dict:
        src = node_map.get(link["source"], {})
        tgt = node_map.get(link["target"], {})
        src_label = src.get("label", link["source"])
        tgt_label = tgt.get("label", link["target"])
        rel_type = link.get("type", "relates_to")

        prompt = (
            f"In 3-5 words, describe how '{src_label}' connects to '{tgt_label}' "
            f"given their relationship type is '{rel_type}'. "
            f"Be poetic and concise. Return only the label, nothing else."
        )

        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.7,
            )
            label = response.choices[0].message.content.strip()
        except Exception:
            label = rel_type.replace("_", " ")

        return {**link, "label": label}

    # Label all edges concurrently
    labeled_links = await asyncio.gather(*[label_edge(l) for l in links])

    out_nodes = [{"group": 0, **n} for n in nodes]

    return {"nodes": out_nodes, "links": list(labeled_links)}
