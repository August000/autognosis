"""FastAPI server for the Solace graph visualization."""

from __future__ import annotations

import asyncio
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.graph_queries import (
    get_raw_graph,
    enrich_graph,
    get_node_memories,
    search_nodes,
    suggest_related,
    get_concept_detail,
    materialize_ghost_node,
    to_centralized,
    to_decentralized,
)

app = FastAPI(title="Solace Graph Visualizer")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# Cache the latest enriched graph for search/suggest operations
_latest_graph: dict | None = None


class Topology(str, Enum):
    raw = "raw"
    centralized = "centralized"
    decentralized = "decentralized"
    distributed = "distributed"


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/graph")
async def graph(topology: Topology = Query(default=Topology.raw)):
    """Return enriched graph data shaped for the requested topology."""
    global _latest_graph
    raw = get_raw_graph()
    enriched = await enrich_graph(raw)
    _latest_graph = enriched

    if topology == Topology.centralized:
        return to_centralized(enriched)
    elif topology == Topology.decentralized:
        return to_decentralized(enriched)
    elif topology == Topology.distributed:
        for n in enriched["nodes"]:
            n.setdefault("group", 0)
        return enriched
    else:
        for n in enriched["nodes"]:
            n.setdefault("group", 0)
        return enriched


@app.get("/api/search")
async def search(q: str = Query(default="")):
    """Fuzzy search nodes in the current graph."""
    if not _latest_graph:
        return {"results": []}
    results = search_nodes(q, _latest_graph)
    return {
        "results": [
            {"id": n["id"], "label": n.get("label", ""), "description": n.get("description", "")}
            for n in results
        ]
    }


@app.get("/api/node/{node_id}/memories")
async def node_memories(node_id: str, label: str = ""):
    """Return vector-search memories related to a node."""
    if not label:
        return {"memories": []}
    memories = await asyncio.to_thread(get_node_memories, label)
    return {"memories": memories}


@app.get("/api/node/{node_id}/suggest")
async def node_suggest(node_id: str, label: str = ""):
    """Suggest related concepts for a leaf node using first principles."""
    if not label or not _latest_graph:
        return {"suggestions": []}
    existing = [n.get("label", "") for n in _latest_graph.get("nodes", [])]
    suggestions = await suggest_related(label, existing)
    return {"suggestions": suggestions}


@app.get("/api/node/{node_id}/detail")
async def node_detail(node_id: str, label: str = "", neighbors: str = ""):
    """Get first-principles knowledge panel for a concept."""
    if not label:
        return {"detail": ""}
    neighbor_list = [n.strip() for n in neighbors.split(",") if n.strip()] if neighbors else []
    detail = await get_concept_detail(label, neighbor_list)
    return {"detail": detail}


@app.post("/api/node/{parent_id}/materialize")
async def materialize(parent_id: str, label: str = Query(...), reason: str = Query(default="")):
    """Materialize a ghost node into Memgraph, connected to parent."""
    result = await asyncio.to_thread(materialize_ghost_node, label, parent_id, reason)
    if result:
        return {"node": result, "success": True}
    return {"node": None, "success": False}


# Serve static assets
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
