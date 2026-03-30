// ── Solace — Shapes of Thought ──────────────────────────────────────────

const DESCRIPTIONS = {
  raw: 'Full graph -- every concept and connection, enriched with meaning',
  centralized: 'One core idea connecting all others -- hub and spoke',
  decentralized: 'Ideas cluster into themed hubs -- communities of thought',
  distributed: 'Equal mesh -- every edge labeled with how ideas connect',
};

// Muted cluster palette
const CLUSTER_COLORS = [
  '#8CB4D5', '#D4956A', '#7DB889', '#C47D7D', '#B8A062',
  '#9B8EC4', '#5DAFAF', '#CC8E6F', '#82A6C4', '#A3C47D',
  '#CB85A4', '#7DBDBD', '#B0A36D', '#9595C4', '#D49090',
];

let currentGraph = null;
let currentTopology = 'raw';
let highlightNodes = new Set();
let highlightLinks = new Set();
let selectedNode = null;
let clusterLabelObjects = [];

// ── Initialize 3D Graph ──────────────────────────
const Graph = ForceGraph3D()(document.getElementById('graph-container'))
  .backgroundColor('#050507')
  .showNavInfo(false)
  .enableNavigationControls(true)
  .nodeLabel(node => {
    if (node._isGhost) return node.label + ' (click to explore)';
    if (node.description) {
      return '<div style="background:rgba(10,10,12,0.95);padding:8px 12px;border-radius:8px;border:1px solid rgba(255,255,255,0.1);max-width:250px;font-size:12px;line-height:1.4;color:#ccc;font-family:Gellix,sans-serif">'
        + '<div style="font-weight:600;color:#eee;margin-bottom:4px;">' + node.label + '</div>'
        + '<div style="color:#888;font-style:italic;">' + node.description + '</div></div>';
    }
    return node.label || '';
  })
  .nodeColor(() => '#ffffff')
  .nodeOpacity(0.9)
  .linkColor(link => {
    if (link._isGhost) return 'rgba(0,0,0,0)';
    if (highlightLinks.has(link)) return '#ffffff';
    if (selectedNode) return 'rgba(255,255,255,0.04)';
    return 'rgba(255,255,255,0.18)';
  })
  .linkWidth(link => {
    if (link._isGhost) return 0;
    if (highlightLinks.has(link)) return 0.3;
    if (selectedNode) return 0.15;
    return 0.4;
  })
  .linkDirectionalParticles(0)
  .nodeVal(0)
  .nodeThreeObjectExtend(false)
  .nodeThreeObject(node => {
    const label = node.label || '';
    const textHeight = node._isGhost ? 1.5
      : node.isCore ? 3.2 : node.isHub ? 2.6 : 1.8;

    let color;
    if (node._isGhost) {
      color = 'rgba(255,255,255,0.25)';
    } else {
      const clusterColor = CLUSTER_COLORS[(node.group || 0) % CLUSTER_COLORS.length];
      color = highlightNodes.has(node.id) ? '#ffffff' : clusterColor;
    }

    const fontSize = 64;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const fontWeight = node.isCore ? 'bold' : 'normal';
    ctx.font = fontWeight + ' ' + fontSize + 'px Gellix, sans-serif';

    const labelMetrics = ctx.measureText(label);
    const bulletRadius = fontSize * 0.08;
    const gapPx = fontSize * 0.25;
    const padL = 4;
    const w = Math.ceil(padL + bulletRadius * 2 + gapPx + labelMetrics.width + 10);
    const h = Math.ceil(fontSize * 1.4);
    canvas.width = w;
    canvas.height = h;

    ctx.font = fontWeight + ' ' + fontSize + 'px Gellix, sans-serif';
    ctx.textBaseline = 'middle';

    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(padL + bulletRadius, h / 2, bulletRadius, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = color;
    ctx.fillText(label, padL + bulletRadius * 2 + gapPx, h / 2);

    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    const spriteMat = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthWrite: false
    });
    const sprite = new THREE.Sprite(spriteMat);

    const aspect = w / h;
    sprite.scale.set(textHeight * aspect, textHeight, 1);

    return sprite;
  })
  .linkThreeObjectExtend(true)
  .linkThreeObject(link => {
    const text = link.label || link.type || '';
    if (!text || link._isGhost) return null;
    const sprite = new SpriteText(text);
    sprite.fontFace = 'Gellix, sans-serif';
    sprite.color = highlightLinks.has(link) ? 'rgba(255,255,255,0.7)' : 'rgba(255,255,255,0.12)';
    sprite.textHeight = 0.9;
    sprite.fontWeight = '400';
    return sprite;
  })
  .linkPositionUpdate((sprite, { start, end }) => {
    if (!sprite) return;
    Object.assign(sprite.position, {
      x: (start.x + end.x) / 2,
      y: (start.y + end.y) / 2,
      z: (start.z + end.z) / 2,
    });
  })
  .onNodeClick(onNodeClick)
  .onBackgroundClick(() => {
    highlightNodes.clear();
    highlightLinks.clear();
    selectedNode = null;
    hideInfoPanel();
    Graph.linkColor(Graph.linkColor())
         .linkWidth(Graph.linkWidth())
         .linkThreeObject(Graph.linkThreeObject())
         .nodeThreeObject(Graph.nodeThreeObject());
  });

// Fix zoom: configure orbit controls
const controls = Graph.controls();
if (controls) {
  controls.zoomSpeed = 1.5;
  controls.rotateSpeed = 0.8;
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.minDistance = 30;
  controls.maxDistance = 800;
}

function refreshVisuals() {
  Graph.linkColor(Graph.linkColor())
       .linkWidth(Graph.linkWidth());
  Graph.linkThreeObject(Graph.linkThreeObject());
  Graph.nodeThreeObject(Graph.nodeThreeObject());
}

// ── Node click ──────────────────────────────────
function onNodeClick(node) {
  if (!node) return;
  selectAndFocusNode(node, true);
}

function selectAndFocusNode(node, flyTo) {
  selectedNode = node;
  highlightNodes.clear();
  highlightLinks.clear();
  highlightNodes.add(node.id);

  const neighbors = [];
  currentGraph.links.forEach(link => {
    const srcId = typeof link.source === 'object' ? link.source.id : link.source;
    const tgtId = typeof link.target === 'object' ? link.target.id : link.target;
    if (srcId === node.id && !link._isGhost) {
      highlightNodes.add(tgtId);
      highlightLinks.add(link);
      const tgtNode = currentGraph.nodes.find(n => n.id === tgtId);
      neighbors.push({ id: tgtId, label: tgtNode ? tgtNode.label : tgtId, edgeLabel: link.label || link.type || '', direction: 'out' });
    }
    if (tgtId === node.id && !link._isGhost) {
      highlightNodes.add(srcId);
      highlightLinks.add(link);
      const srcNode = currentGraph.nodes.find(n => n.id === srcId);
      neighbors.push({ id: srcId, label: srcNode ? srcNode.label : srcId, edgeLabel: link.label || link.type || '', direction: 'in' });
    }
  });

  showInfoPanel(node, neighbors);
  fetchMemories(node);
  fetchDetail(node, neighbors);
  fetchSuggestions(node);
  refreshVisuals();

  if (flyTo) {
    const distance = 120;
    const dist = Math.hypot(node.x || 0, node.y || 0, node.z || 0) || 1;
    const ratio = 1 + distance / dist;
    Graph.cameraPosition(
      { x: (node.x || 0) * ratio, y: (node.y || 0) * ratio, z: (node.z || 0) * ratio },
      { x: node.x || 0, y: node.y || 0, z: node.z || 0 },
      1200
    );
  }
}


async function fetchSuggestions(parentNode) {
  const section = document.getElementById('suggestions-section');
  const content = document.getElementById('suggestions-content');
  section.style.display = 'block';
  content.innerHTML = '<div class="suggestions-loading">exploring related concepts...</div>';

  try {
    const resp = await fetch(
      '/api/node/' + encodeURIComponent(parentNode.id)
      + '/suggest?label=' + encodeURIComponent(parentNode.label || '')
    );
    const data = await resp.json();

    if (selectedNode !== parentNode) return;

    if (!data.suggestions || data.suggestions.length === 0) {
      section.style.display = 'none';
      return;
    }

    content.innerHTML = '';
    data.suggestions.forEach(s => {
      const div = document.createElement('div');
      div.className = 'suggestion-item';
      div.innerHTML =
        '<div><div class="sug-label">' + esc(s.label) + '</div>'
        + (s.reason ? '<div class="sug-reason">' + esc(s.reason) + '</div>' : '')
        + '</div>';
      div.addEventListener('click', () => materializeSuggestion(parentNode, s));
      content.appendChild(div);
    });
  } catch (err) {
    section.style.display = 'none';
    console.error('Suggestions failed:', err);
  }
}

async function materializeSuggestion(parentNode, suggestion) {
  try {
    const resp = await fetch(
      '/api/node/' + encodeURIComponent(parentNode.id)
      + '/materialize?label=' + encodeURIComponent(suggestion.label)
      + '&reason=' + encodeURIComponent(suggestion.reason || ''),
      { method: 'POST' }
    );
    const data = await resp.json();
    if (data.success) {
      loadGraph(currentTopology);
    }
  } catch (err) {
    console.error('Materialize failed:', err);
  }
}


// ── Cluster labels in 3D ────────────────────────
function renderClusterLabels(data) {
  clusterLabelObjects.forEach(obj => Graph.scene().remove(obj));
  clusterLabelObjects = [];

  const clusterNames = data.clusterNames || {};
  if (!clusterNames || Object.keys(clusterNames).length === 0) return;

  setTimeout(() => {
    const nodes = Graph.graphData().nodes;
    const groups = {};

    nodes.forEach(n => {
      if (n._isGhost) return;
      const g = n.group || 0;
      if (!groups[g]) groups[g] = { xs: [], ys: [], zs: [] };
      groups[g].xs.push(n.x || 0);
      groups[g].ys.push(n.y || 0);
      groups[g].zs.push(n.z || 0);
    });

    Object.keys(groups).forEach(g => {
      const name = clusterNames[g] || clusterNames[parseInt(g)];
      if (!name) return;
      const grp = groups[g];
      if (grp.xs.length < 2) return;

      const cx = grp.xs.reduce((a, b) => a + b, 0) / grp.xs.length;
      const cy = grp.ys.reduce((a, b) => a + b, 0) / grp.ys.length;
      const cz = grp.zs.reduce((a, b) => a + b, 0) / grp.zs.length;

      const color = CLUSTER_COLORS[parseInt(g) % CLUSTER_COLORS.length];
      const sprite = new SpriteText(name.toUpperCase());
      sprite.fontFace = 'Gellix, sans-serif';
      sprite.color = color;
      sprite.textHeight = 8;
      sprite.fontWeight = 'bold';
      sprite.material.opacity = 0.12;
      sprite.position.set(cx, cy + 30, cz);

      Graph.scene().add(sprite);
      clusterLabelObjects.push(sprite);
    });
  }, 3000);
}

// ── Search ──────────────────────────────────────
const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');
let searchTimeout = null;

searchInput.addEventListener('input', () => {
  clearTimeout(searchTimeout);
  const q = searchInput.value.trim();
  if (!q) { searchResults.classList.remove('visible'); return; }
  searchTimeout = setTimeout(() => doSearch(q), 200);
});
searchInput.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') { searchInput.blur(); searchResults.classList.remove('visible'); }
});
document.addEventListener('click', (e) => {
  if (!e.target.closest('.search-wrapper')) searchResults.classList.remove('visible');
});
document.addEventListener('keydown', (e) => {
  if (e.key === '/' && document.activeElement !== searchInput) { e.preventDefault(); searchInput.focus(); }
});

async function doSearch(q) {
  try {
    const resp = await fetch('/api/search?q=' + encodeURIComponent(q));
    const data = await resp.json();
    searchResults.innerHTML = '';
    if (data.results && data.results.length > 0) {
      data.results.forEach(r => {
        const div = document.createElement('div');
        div.className = 'search-result-item';
        div.innerHTML = '<div class="sr-label">' + esc(r.label) + '</div>'
          + (r.description ? '<div class="sr-desc">' + esc(r.description) + '</div>' : '');
        div.addEventListener('click', () => {
          searchResults.classList.remove('visible');
          searchInput.value = r.label;
          searchInput.blur();
          const node = currentGraph.nodes.find(n => n.id === r.id);
          if (node) selectAndFocusNode(node, true);
        });
        searchResults.appendChild(div);
      });
      searchResults.classList.add('visible');
    } else {
      searchResults.classList.remove('visible');
    }
  } catch (err) { console.error('Search failed:', err); }
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

// ── Brain-shaped force layout ────────────────
function brainShapeForce(alpha) {
  const nodes = Graph.graphData().nodes;
  const RX = 180;
  const RY = 130;
  const RZ = 140;
  const k = alpha * 0.02;

  nodes.forEach(n => {
    if (n._isGhost) return;
    const x = n.x || 0;
    const y = n.y || 0;
    const z = n.z || 0;

    const r = Math.sqrt((x*x)/(RX*RX) + (y*y)/(RY*RY) + (z*z)/(RZ*RZ));
    if (r < 0.01) return;

    let targetR;
    if (r > 1) {
      targetR = 1 / r;
    } else {
      targetR = 1;
    }

    if (r > 1) {
      const factor = k * (1 - targetR);
      n.vx = (n.vx || 0) - x * factor;
      n.vy = (n.vy || 0) - y * factor;
      n.vz = (n.vz || 0) - z * factor;
    }
  });
}

// ── Load graph ──────────────────────────────────
async function loadGraph(topology) {
  currentTopology = topology;
  const loading = document.getElementById('loading');
  const loadingText = document.getElementById('loading-text');
  loading.classList.remove('hidden');
  loadingText.textContent = !currentGraph ? 'enriching thoughts with meaning...' : 'reshaping topology...';

  try {
    const resp = await fetch('/api/graph?topology=' + topology);
    const data = await resp.json();
    currentGraph = data;

    Graph.d3Force('brain', brainShapeForce);

    if (topology === 'centralized' && data.coreId) {
      Graph.d3Force('center', null);
      Graph.d3Force('charge').strength(-120);
      Graph.d3Force('cluster', null);
    } else if (topology === 'decentralized') {
      Graph.d3Force('charge').strength(-80);
      Graph.d3Force('cluster', alpha => {
        const nodes = Graph.graphData().nodes;
        const centroids = {}, counts = {};
        nodes.forEach(n => {
          if (n._isGhost) return;
          const g = n.group || 0;
          if (!centroids[g]) { centroids[g] = { x: 0, y: 0, z: 0 }; counts[g] = 0; }
          centroids[g].x += n.x || 0; centroids[g].y += n.y || 0; centroids[g].z += n.z || 0;
          counts[g]++;
        });
        Object.keys(centroids).forEach(g => {
          centroids[g].x /= counts[g]; centroids[g].y /= counts[g]; centroids[g].z /= counts[g];
        });
        const k = alpha * 0.3;
        nodes.forEach(n => {
          if (n._isGhost) return;
          const c = centroids[n.group || 0];
          if (!c) return;
          n.vx = (n.vx || 0) + (c.x - (n.x || 0)) * k;
          n.vy = (n.vy || 0) + (c.y - (n.y || 0)) * k;
          n.vz = (n.vz || 0) + (c.z - (n.z || 0)) * k;
        });
      });
    } else if (topology === 'distributed') {
      Graph.d3Force('charge').strength(-150);
      Graph.d3Force('cluster', null);
    } else {
      Graph.d3Force('charge').strength(-60);
      Graph.d3Force('cluster', null);
    }

    Graph.graphData(data);
    updateStats(data);
    renderClusterLabels(data);

    highlightNodes.clear();
    highlightLinks.clear();
    selectedNode = null;
    hideInfoPanel();

  } catch (err) {
    console.error('Failed to load graph:', err);
  } finally {
    setTimeout(() => loading.classList.add('hidden'), 300);
  }
}

// ── Info panel ──────────────────────────────────
function showInfoPanel(node, neighbors) {
  const panel = document.getElementById('info-panel');
  document.getElementById('info-title').textContent = node.label || node.id;

  const descEl = document.getElementById('info-description');
  if (node.description) { descEl.textContent = node.description; descEl.style.display = 'block'; }
  else { descEl.style.display = 'none'; }

  document.getElementById('info-connections').textContent =
    neighbors.length + ' connection' + (neighbors.length !== 1 ? 's' : '');

  const list = document.getElementById('info-list');
  list.innerHTML = '';
  neighbors.forEach(n => {
    const li = document.createElement('li');
    const btn = document.createElement('button');
    btn.className = 'nav-link';
    btn.innerHTML =
      '<span class="nav-arrow">' + (n.direction === 'out' ? '&rarr;' : '&larr;') + '</span>'
      + '<span class="nav-edge">' + esc(n.edgeLabel) + '</span>'
      + '<span class="nav-name">' + esc(n.label) + '</span>'
      + '<span class="nav-goto">go &rarr;</span>';
    btn.addEventListener('click', () => {
      const target = currentGraph.nodes.find(nd => nd.id === n.id);
      if (target) selectAndFocusNode(target, true);
    });
    li.appendChild(btn);
    list.appendChild(li);
  });

  document.getElementById('detail-section').style.display = 'none';
  document.getElementById('detail-content').innerHTML = '';
  document.getElementById('suggestions-section').style.display = 'none';
  document.getElementById('suggestions-content').innerHTML = '';
  document.getElementById('memories-section').style.display = 'none';
  document.getElementById('memories-content').innerHTML = '';

  panel.classList.add('visible');
}

function hideInfoPanel() {
  document.getElementById('info-panel').classList.remove('visible');
}

// ── Detail + Memories ───────────────────────────
async function fetchDetail(node, neighbors) {
  const section = document.getElementById('detail-section');
  const content = document.getElementById('detail-content');
  section.style.display = 'block';
  content.innerHTML = '<div class="detail-loading">thinking from first principles...</div>';
  try {
    const nl = neighbors.map(n => n.label).join(',');
    const resp = await fetch('/api/node/' + encodeURIComponent(node.id)
      + '/detail?label=' + encodeURIComponent(node.label || '') + '&neighbors=' + encodeURIComponent(nl));
    const data = await resp.json();
    if (data.detail) content.textContent = data.detail;
    else section.style.display = 'none';
  } catch (err) { section.style.display = 'none'; }
}

async function fetchMemories(node) {
  const section = document.getElementById('memories-section');
  const content = document.getElementById('memories-content');
  section.style.display = 'block';
  content.innerHTML = '<div class="memories-loading">searching memories...</div>';
  try {
    const resp = await fetch('/api/node/' + encodeURIComponent(node.id) + '/memories?label=' + encodeURIComponent(node.label || ''));
    const data = await resp.json();
    if (data.memories && data.memories.length > 0) {
      content.innerHTML = '';
      data.memories.forEach(m => {
        const div = document.createElement('div');
        div.className = 'memory-item';
        div.textContent = m.text;
        content.appendChild(div);
      });
    } else {
      content.innerHTML = '<div class="memories-loading">no memories found</div>';
    }
  } catch (err) {
    content.innerHTML = '<div class="memories-loading">could not load memories</div>';
  }
}

// ── Stats ───────────────────────────────────────
function updateStats(data) {
  const stats = document.getElementById('stats');
  const nc = data.nodes ? data.nodes.length : 0;
  const lc = data.links ? data.links.length : 0;
  const cl = data.clusters ? data.clusters + ' clusters' : '';
  stats.innerHTML = nc + ' nodes &middot; ' + lc + ' edges' + (cl ? ' &middot; ' + cl : '');
}

// ── Topology buttons ────────────────────────────
document.querySelectorAll('.topo-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.topo-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const topology = btn.dataset.topology;
    const desc = document.getElementById('topo-desc');
    desc.textContent = DESCRIPTIONS[topology] || '';
    desc.classList.add('visible');
    setTimeout(() => desc.classList.remove('visible'), 3000);
    loadGraph(topology);
  });
});


// ═══════════════════════════════════════════════════════════════════════════
// ── Chat Panel ─────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

const chatPanel = document.getElementById('chat-panel');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');
const chatToggle = document.getElementById('chat-toggle');
const chatModelSelect = document.getElementById('chat-model-select');
const chatNewBtn = document.getElementById('chat-new-btn');
const chatHistoryBtn = document.getElementById('chat-history-btn');
const chatConversations = document.getElementById('chat-conversations');

let chatConversationId = null;
let chatIsStreaming = false;

// Toggle minimize/expand
chatToggle.addEventListener('click', (e) => {
  e.stopPropagation();
  chatPanel.classList.toggle('minimized');
  chatToggle.textContent = chatPanel.classList.contains('minimized') ? '+' : '_';
});

// New conversation
chatNewBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  createNewConversation();
});

// History toggle
chatHistoryBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  chatConversations.classList.toggle('visible');
  if (chatConversations.classList.contains('visible')) {
    loadConversationsList();
  }
});

// Send message
chatSend.addEventListener('click', () => sendChatMessage());
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
  }
});

// Auto-resize textarea
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 80) + 'px';
});

async function createNewConversation() {
  const model = chatModelSelect.value;
  try {
    const resp = await fetch('/api/chat/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    });
    const data = await resp.json();
    chatConversationId = data.id;
    chatMessages.innerHTML = '<div id="chat-empty">ask anything about your knowledge graph...</div>';
    chatConversations.classList.remove('visible');
  } catch (err) {
    console.error('Failed to create conversation:', err);
  }
}

async function loadConversationsList() {
  try {
    const resp = await fetch('/api/chat/conversations');
    const data = await resp.json();
    chatConversations.innerHTML = '';

    if (!data.conversations || data.conversations.length === 0) {
      chatConversations.innerHTML = '<div style="padding:12px 14px;font-size:11px;color:#555;">no conversations yet</div>';
      return;
    }

    data.conversations.forEach(conv => {
      const div = document.createElement('div');
      div.className = 'chat-conv-item' + (conv.id === chatConversationId ? ' active' : '');

      const title = document.createElement('span');
      title.className = 'chat-conv-title';
      title.textContent = conv.title || 'New conversation';
      div.appendChild(title);

      const delBtn = document.createElement('button');
      delBtn.className = 'chat-conv-delete';
      delBtn.textContent = 'x';
      delBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        await fetch('/api/chat/conversations/' + conv.id, { method: 'DELETE' });
        if (conv.id === chatConversationId) {
          chatConversationId = null;
          chatMessages.innerHTML = '<div id="chat-empty">ask anything about your knowledge graph...</div>';
        }
        loadConversationsList();
      });
      div.appendChild(delBtn);

      div.addEventListener('click', () => loadConversation(conv.id));
      chatConversations.appendChild(div);
    });
  } catch (err) {
    console.error('Failed to load conversations:', err);
  }
}

async function loadConversation(id) {
  try {
    const resp = await fetch('/api/chat/conversations/' + id + '/messages');
    const data = await resp.json();
    chatConversationId = id;
    chatMessages.innerHTML = '';

    if (data.conversation && data.conversation.model) {
      chatModelSelect.value = data.conversation.model;
    }

    if (!data.messages || data.messages.length === 0) {
      chatMessages.innerHTML = '<div id="chat-empty">ask anything about your knowledge graph...</div>';
    } else {
      data.messages.forEach(msg => {
        if (msg.role === 'user' || msg.role === 'assistant') {
          appendChatMessage(msg.role, msg.content);
        }
      });
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    chatConversations.classList.remove('visible');
  } catch (err) {
    console.error('Failed to load conversation:', err);
  }
}

function appendChatMessage(role, content) {
  const empty = document.getElementById('chat-empty');
  if (empty) empty.remove();

  const div = document.createElement('div');
  div.className = 'chat-msg ' + role;
  div.textContent = content;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return div;
}

async function sendChatMessage() {
  const content = chatInput.value.trim();
  if (!content || chatIsStreaming) return;

  // Auto-create conversation if needed
  if (!chatConversationId) {
    await createNewConversation();
  }

  chatIsStreaming = true;
  chatSend.disabled = true;
  chatInput.value = '';
  chatInput.style.height = 'auto';

  // Show user message
  appendChatMessage('user', content);

  // Show typing indicator
  const typingDiv = document.createElement('div');
  typingDiv.className = 'chat-msg assistant typing';
  typingDiv.textContent = 'thinking...';
  chatMessages.appendChild(typingDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  try {
    const model = chatModelSelect.value;
    const resp = await fetch('/api/chat/conversations/' + chatConversationId + '/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content, model }),
    });

    // Remove typing indicator
    typingDiv.remove();

    // Create assistant message div for streaming
    const assistantDiv = appendChatMessage('assistant', '');

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const data = JSON.parse(line.slice(6));
          if (data.chunk) {
            assistantDiv.textContent += data.chunk;
            chatMessages.scrollTop = chatMessages.scrollHeight;
          }
        } catch (e) { /* ignore parse errors for incomplete chunks */ }
      }
    }
  } catch (err) {
    typingDiv.remove();
    appendChatMessage('assistant', 'Failed to get response. Please try again.');
    console.error('Chat failed:', err);
  } finally {
    chatIsStreaming = false;
    chatSend.disabled = false;
    chatInput.focus();
  }
}


// ── Init ────────────────────────────────────────
loadGraph('raw');
