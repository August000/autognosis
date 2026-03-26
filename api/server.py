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
    to_centralized,
    to_decentralized,
)

app = FastAPI(title="Solace Graph Visualizer")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


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
    raw = get_raw_graph()

    # Enrich all nodes with descriptions and all edges with labels (cached after first call)
    enriched = await enrich_graph(raw)

    if topology == Topology.centralized:
        return to_centralized(enriched)
    elif topology == Topology.decentralized:
        return to_decentralized(enriched)
    elif topology == Topology.distributed:
        # Distributed uses the same enriched graph, just with different force layout
        for n in enriched["nodes"]:
            n.setdefault("group", 0)
        return enriched
    else:
        for n in enriched["nodes"]:
            n.setdefault("group", 0)
        return enriched


@app.get("/api/node/{node_id}/memories")
async def node_memories(node_id: str, label: str = ""):
    """Return vector-search memories related to a node."""
    if not label:
        return {"memories": []}
    memories = await asyncio.to_thread(get_node_memories, label)
    return {"memories": memories}


# Serve static assets
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
