"""FastAPI server for the Solace graph visualization."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.graph_queries import get_raw_graph, to_centralized, to_decentralized, to_distributed

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
    """Return graph data shaped for the requested topology."""
    raw = get_raw_graph()

    if topology == Topology.centralized:
        return to_centralized(raw)
    elif topology == Topology.decentralized:
        return to_decentralized(raw)
    elif topology == Topology.distributed:
        return await to_distributed(raw)
    else:
        # Raw: just add default group
        for n in raw["nodes"]:
            n["group"] = 0
        return raw


# Serve static assets (CSS, JS if we ever add them)
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
