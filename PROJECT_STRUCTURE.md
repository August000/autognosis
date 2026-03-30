# Solace — Project Structure

## Overview

Solace is a knowledge graph visualization and voice interaction system. It stores a user's thoughts, memories, and concepts in a graph database (Memgraph) with vector search (Qdrant via mem0), enriches the graph with LLM-generated descriptions, and renders it as an interactive 3D force-directed graph in the browser.

## Architecture

```
User Voice Input
      |
  VoiceSession (WebSocket + OpenAI Realtime)
      |
  mem0 (Memory Client)
      |
  +---+---+
  |       |
Memgraph  Qdrant
(graph)   (vectors)
  |       |
  +---+---+
      |
  FastAPI Server (/api/*)
      |
  3D Force Graph (browser)
```

## Directory Structure

```
Solace/
  main.py              # CLI entry point — launches VoiceSession
  pyproject.toml       # Python project config (uv, dependencies)
  .env                 # Environment variables (API keys, URIs)

  voice/               # Voice interaction pipeline
    session.py         # VoiceSession: orchestrates mic capture, OpenAI realtime WS, TTS
    audio_capture.py   # Microphone input capture via sounddevice
    realtime_ws.py     # WebSocket client for OpenAI Realtime API
    tts.py             # Text-to-speech output via ElevenLabs

  memory/              # Memory storage client
    client.py          # mem0 Memory instance configured for Memgraph + Qdrant

  chat/                # Standalone chat test
    chat.py            # Simple OpenAI streaming test script

  api/                 # FastAPI web server
    server.py          # HTTP endpoints: graph, search, suggest, detail, memories, materialize
    graph_queries.py   # Memgraph queries, LLM enrichment (descriptions, edge labels, clusters)

  web/                 # Frontend (single-page, no build step)
    index.html         # Full application: HTML + CSS + JS (3D force graph)
    three.min.js       # Three.js (3D rendering engine)
    three-spritetext.min.js  # SpriteText for 3D text labels
    3d-force-graph.min.js    # ForceGraph3D library
    fonts/             # Gellix font family (woff2)

  Gellix Font/         # Source font files (ttf, woff, woff2, eot)
```

## Dependencies

### Python (pyproject.toml)
- **openai** — LLM calls (GPT-4o-mini for enrichment, realtime API for voice)
- **mem0ai[graph]** — Memory framework with graph store support
- **neo4j** — Bolt driver for Memgraph (Cypher queries)
- **qdrant-client** — Vector database for memory search
- **fastapi + uvicorn** — Web API server
- **elevenlabs** — Text-to-speech
- **sounddevice** — Microphone audio capture
- **websockets** — WebSocket client for OpenAI Realtime
- **python-dotenv** — .env file loading
- **numpy** — Audio buffer processing
- **langchain-memgraph** — Memgraph integration (used by mem0)

### Frontend (vendored, no npm)
- **Three.js** — 3D WebGL rendering
- **three-spritetext** — Text sprites in 3D space
- **ForceGraph3D** — Force-directed 3D graph built on Three.js + d3-force-3d

## External Services

| Service   | Purpose                    | Default URI            |
|-----------|----------------------------|------------------------|
| Memgraph  | Graph database (Cypher)    | bolt://localhost:7687  |
| Qdrant    | Vector search              | localhost:6333         |
| OpenAI    | LLM + embeddings + realtime| api.openai.com         |
| ElevenLabs| Text-to-speech             | api.elevenlabs.io      |

## How the Code Works

### 1. Voice Pipeline (`main.py` -> `voice/`)
The CLI entry point creates a `VoiceSession` for user "augusto". The session captures microphone audio (`audio_capture.py`), streams it to OpenAI's Realtime API over WebSocket (`realtime_ws.py`), processes responses, stores memories via mem0, and speaks responses back via ElevenLabs TTS (`tts.py`).

### 2. Memory Storage (`memory/client.py`)
A single `mem0.Memory` instance configured with:
- **Graph store**: Memgraph (stores concept nodes and relationships)
- **Vector store**: Qdrant (stores embeddings for semantic search)
- **LLM/Embedder**: OpenAI GPT-4o-mini / text-embedding-3-small

### 3. API Server (`api/server.py` + `api/graph_queries.py`)
FastAPI app serving:
- `GET /` — Serves the frontend
- `GET /api/graph?topology=` — Fetches raw graph from Memgraph, enriches with LLM (cached), applies topology transforms
- `GET /api/search?q=` — Fuzzy search over graph nodes
- `GET /api/node/{id}/suggest?label=` — LLM suggests related concepts (first-principles)
- `GET /api/node/{id}/detail?label=&neighbors=` — LLM generates knowledge panel
- `GET /api/node/{id}/memories?label=` — Vector search for related memories
- `POST /api/node/{id}/materialize?label=&reason=` — Creates a new node in Memgraph

**Enrichment pipeline** (first request only, then cached):
1. Fetches all nodes and relationships from Memgraph
2. Computes connected components (clusters)
3. Batches nodes to GPT-4o-mini for 1-sentence descriptions
4. Batches edges to GPT-4o-mini for descriptive labels
5. Names each cluster with a 1-3 word topic

**Topology transforms**:
- **Raw** — Full graph as-is
- **Centralized** — Highest-degree node becomes hub; all others link through it
- **Decentralized** — Nodes cluster into themed groups with per-group hubs
- **Distributed** — Equal mesh with labeled edges

### 4. Frontend (`web/index.html`)
Single HTML file containing all CSS and JS. Key features:

- **3D Force Graph**: Renders nodes as canvas sprites ("● label" — tiny white bullet to the left of the cluster-colored label text). Links are solid lines with midpoint edge labels. Ghost links are hidden (zero width, transparent).
- **Brain-shaped layout**: Custom force (`brainShapeForce`) constrains nodes to an ellipsoid shape.
- **Cluster labels**: Large faded SpriteText names float above each cluster centroid.
- **Node selection**: Clicking a node highlights it and its neighbors, flies the camera to it, shows an info panel with connections, knowledge detail, suggestions (pink), and related memories.
- **Suggestions panel**: Pink-styled items in the detail panel showing LLM-suggested related concepts. Clicking a suggestion materializes it as a real node in Memgraph.
- **Search**: Fuzzy search bar (shortcut: `/`) with dropdown results; clicking navigates to the node.
- **Topology switcher**: Buttons to switch between raw/centralized/decentralized/distributed views.
- **Zoom controls**: Orbit controls with configurable speed and distance bounds.

### Data Flow (node click)
1. User clicks node -> `onNodeClick` -> `selectAndFocusNode`
2. Highlights node + neighbors, dims everything else
3. Camera flies to node
4. Info panel shows: title, description, connections (navigable buttons)
5. Parallel fetches: `/api/node/{id}/detail`, `/api/node/{id}/memories`, `/api/node/{id}/suggest`
6. Detail, memories, and suggestions populate in the panel as they arrive

### Known fixes applied
- **Node re-selection**: Background click deselects cleanly and rebuilds node objects so subsequent clicks register properly.
- **Solid lines only**: No pulsing/animated links; ghost links hidden with zero width and transparent color.
- **Cypher injection safety**: `VoiceSession._sanitize_for_cypher()` strips `/`, `\`, quotes, backticks, and braces from text before passing to mem0, preventing Memgraph parse errors.
