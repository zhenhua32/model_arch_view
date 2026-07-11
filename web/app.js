const state = {
  models: [],
  activeModelId: null,
  payload: null,
  selectedNodeId: null,
  llmHierarchyMode: "summary",
  refreshTimer: null,
};

const ui = {
  modelCount: document.getElementById("model-count"),
  modelSearch: document.getElementById("model-search"),
  modelList: document.getElementById("model-list"),
  controlsForm: document.getElementById("controls-form"),
  refreshButton: document.getElementById("refresh-button"),
  hierarchyToolbar: document.getElementById("hierarchy-toolbar"),
  exportJsonButton: document.getElementById("export-json-button"),
  exportSvgButton: document.getElementById("export-svg-button"),
  statusBar: document.getElementById("status-bar"),
  summaryPanel: document.getElementById("model-summary"),
  warningsPanel: document.getElementById("warnings-panel"),
  graphCanvas: document.getElementById("graph-canvas"),
  graphBoard: document.getElementById("graph-board"),
  edgeLayer: document.getElementById("edge-layer"),
  detailPanel: document.getElementById("detail-panel"),
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setStatus(text, isError = false) {
  ui.statusBar.textContent = text;
  ui.statusBar.style.color = isError ? "#9b4d34" : "";
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  return response.json();
}

function updateExportButtons() {
  const disabled = !state.payload;
  ui.exportJsonButton.disabled = disabled;
  ui.exportSvgButton.disabled = disabled;
}

function extractLlmLayerCount(payload = state.payload) {
  const summaryItem = payload?.model?.summary?.find?.((item) => item.label === "层数");
  const numeric = Number.parseInt(summaryItem?.value ?? "", 10);
  return Number.isFinite(numeric) ? numeric : null;
}

function getHierarchyModeLabel(mode, payload = state.payload) {
  if (mode === "summary") {
    return "汇总";
  }
  if (mode === "block") {
    return "单层 Block（Q/K/V/RoPE/Score/Causal/Window/Softmax/Output）";
  }
  if (mode === "repeat") {
    const layerCount = extractLlmLayerCount(payload);
    return layerCount ? `重复 ${layerCount} 层摘要` : "重复 N 层摘要";
  }
  return mode;
}

function getHierarchyMode(payload = state.payload) {
  if (payload?.model?.type !== "llm") {
    return "all";
  }
  return state.llmHierarchyMode;
}

function isVisibleInHierarchy(item, mode) {
  if (mode === "all") {
    return true;
  }
  if (!Array.isArray(item?.viewModes) || !item.viewModes.length) {
    return true;
  }
  return item.viewModes.includes(mode);
}

function getVisibleGraph(payload = state.payload) {
  const graph = payload?.graph || { nodes: [], edges: [], lanes: [] };
  const mode = getHierarchyMode(payload);
  const nodes = (graph.nodes || []).filter((node) => isVisibleInHierarchy(node, mode));
  const visibleIds = new Set(nodes.map((node) => node.id));
  const edges = (graph.edges || []).filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target) && isVisibleInHierarchy(edge, mode));
  return {
    lanes: graph.lanes || [],
    nodes,
    edges,
  };
}

function syncSelectedNode(payload = state.payload) {
  const visibleGraph = getVisibleGraph(payload);
  const visibleIds = new Set((visibleGraph.nodes || []).map((node) => node.id));
  if (visibleIds.has(state.selectedNodeId)) {
    return;
  }

  const fullNodes = payload?.graph?.nodes || [];
  const hiddenNode = fullNodes.find((node) => node.id === state.selectedNodeId);
  if (hiddenNode?.parentId && visibleIds.has(hiddenNode.parentId)) {
    state.selectedNodeId = hiddenNode.parentId;
    return;
  }

  if (hiddenNode?.parentId) {
    const siblingNode = fullNodes.find((node) => node.parentId === hiddenNode.parentId && visibleIds.has(node.id));
    if (siblingNode) {
      state.selectedNodeId = siblingNode.id;
      return;
    }
  }

  const visibleChild = fullNodes.find((node) => node.parentId === state.selectedNodeId && visibleIds.has(node.id));
  if (visibleChild) {
    state.selectedNodeId = visibleChild.id;
    return;
  }

  if (visibleIds.has(payload?.selectedNodeId)) {
    state.selectedNodeId = payload.selectedNodeId;
    return;
  }

  state.selectedNodeId = visibleGraph.nodes?.[0]?.id || null;
}

function renderHierarchyToolbar() {
  const isLlm = state.payload?.model?.type === "llm";
  ui.hierarchyToolbar.hidden = !isLlm;
  if (!isLlm) {
    return;
  }

  ui.hierarchyToolbar.querySelectorAll("[data-hierarchy-mode]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.hierarchyMode === state.llmHierarchyMode);
  });
}

function buildQueryFromControls() {
  const formData = new FormData(ui.controlsForm);
  const params = new URLSearchParams();
  for (const [key, value] of formData.entries()) {
    if (String(value).trim() !== "") {
      params.set(key, String(value));
    }
  }
  return params.toString();
}

function buildGraphIndex(graph) {
  const nodeById = new Map((graph?.nodes || []).map((node) => [node.id, node]));
  const outgoingNeighbors = new Map();
  const incomingNeighbors = new Map();

  (graph?.nodes || []).forEach((node) => {
    outgoingNeighbors.set(node.id, []);
    incomingNeighbors.set(node.id, []);
  });

  (graph?.edges || []).forEach((edge) => {
    if (!outgoingNeighbors.has(edge.source)) {
      outgoingNeighbors.set(edge.source, []);
    }
    if (!incomingNeighbors.has(edge.target)) {
      incomingNeighbors.set(edge.target, []);
    }
    outgoingNeighbors.get(edge.source).push(edge.target);
    incomingNeighbors.get(edge.target).push(edge.source);
  });

  return { nodeById, outgoingNeighbors, incomingNeighbors };
}

function collectReachable(startId, neighbors) {
  const visited = new Set();
  const queue = [...(neighbors.get(startId) || [])];
  while (queue.length) {
    const current = queue.shift();
    if (visited.has(current)) {
      continue;
    }
    visited.add(current);
    for (const next of neighbors.get(current) || []) {
      if (!visited.has(next)) {
        queue.push(next);
      }
    }
  }
  return visited;
}

function buildGraphRelations(payload) {
  const graph = getVisibleGraph(payload);
  const index = buildGraphIndex(graph);
  const selectedId = state.selectedNodeId || payload?.selectedNodeId || graph.nodes?.[0]?.id || null;
  const upstream = selectedId ? collectReachable(selectedId, index.incomingNeighbors) : new Set();
  const downstream = selectedId ? collectReachable(selectedId, index.outgoingNeighbors) : new Set();
  const related = new Set([selectedId, ...upstream, ...downstream].filter(Boolean));

  function nodeClass(nodeId) {
    if (!selectedId) {
      return "";
    }
    if (nodeId === selectedId) {
      return "selected";
    }
    if (upstream.has(nodeId)) {
      return "upstream";
    }
    if (downstream.has(nodeId)) {
      return "downstream";
    }
    return "dimmed";
  }

  function edgeClass(edge) {
    if (!selectedId) {
      return "related";
    }
    if (edge.target === selectedId || upstream.has(edge.target)) {
      return "upstream";
    }
    if (edge.source === selectedId || downstream.has(edge.source)) {
      return "downstream";
    }
    if (related.has(edge.source) && related.has(edge.target)) {
      return "related";
    }
    return "dimmed";
  }

  return {
    index,
    selectedId,
    upstream,
    downstream,
    related,
    nodeClass,
    edgeClass,
  };
}

function computePrimaryPath(graph) {
  const index = buildGraphIndex(graph);
  const memo = new Map();

  function longestFrom(nodeId, visiting = new Set()) {
    if (memo.has(nodeId)) {
      return memo.get(nodeId);
    }
    if (visiting.has(nodeId)) {
      return [nodeId];
    }

    visiting.add(nodeId);
    let best = [nodeId];
    for (const nextId of index.outgoingNeighbors.get(nodeId) || []) {
      const candidate = [nodeId, ...longestFrom(nextId, visiting)];
      if (candidate.length > best.length) {
        best = candidate;
      }
    }
    visiting.delete(nodeId);
    memo.set(nodeId, best);
    return best;
  }

  const sources = (graph?.nodes || [])
    .filter((node) => (index.incomingNeighbors.get(node.id) || []).length === 0)
    .map((node) => node.id);
  const candidates = sources.length ? sources : (graph?.nodes || []).map((node) => node.id);

  let bestPath = [];
  for (const nodeId of candidates) {
    const candidate = longestFrom(nodeId, new Set());
    if (candidate.length > bestPath.length) {
      bestPath = candidate;
    }
  }

  return bestPath.map((nodeId) => index.nodeById.get(nodeId)).filter(Boolean);
}

function renderChipPairs(items, className = "flow-chip") {
  return items
    .map(
      (item) => `
        <span class="${className}">
          <span class="flow-chip-label">${escapeHtml(item.label)}</span>
          <span class="flow-chip-value">${escapeHtml(item.value)}</span>
        </span>
      `
    )
    .join("");
}

function renderModelList() {
  const keyword = ui.modelSearch.value.trim().toLowerCase();
  const models = state.models.filter((model) => {
    if (!keyword) {
      return true;
    }
    return [model.name, model.architecture, model.type].some((value) => String(value).toLowerCase().includes(keyword));
  });

  ui.modelCount.textContent = String(models.length);
  if (!models.length) {
    ui.modelList.innerHTML = '<div class="empty-state">没有匹配的模型。</div>';
    return;
  }

  ui.modelList.innerHTML = models
    .map((model) => {
      const active = model.id === state.activeModelId ? "active" : "";
      return `
        <button class="model-card ${active}" type="button" data-model-id="${escapeHtml(model.id)}">
          <h3 class="model-card-title">${escapeHtml(model.name)}</h3>
          <div class="tag-row">
            <span class="tag">${escapeHtml(model.type)}</span>
            <span class="tag">${escapeHtml(model.architecture)}</span>
          </div>
          <p class="model-card-meta">${escapeHtml(model.fileCount)} files</p>
        </button>
      `;
    })
    .join("");
}

async function loadModels() {
  setStatus("正在读取 model_configs 目录...");
  const data = await fetchJson("/api/models");
  state.models = data.models || [];
  renderModelList();

  if (!state.models.length) {
    setStatus("未找到模型目录。", true);
    state.payload = null;
    updateExportButtons();
    ui.summaryPanel.innerHTML = '<div class="empty-state">当前没有可展示的模型目录。</div>';
    return;
  }

  const currentExists = state.models.some((model) => model.id === state.activeModelId);
  if (!currentExists) {
    state.activeModelId = state.models[0].id;
  }
  await loadModelPayload();
}

async function loadModelPayload() {
  if (!state.activeModelId) {
    return;
  }

  const query = buildQueryFromControls();
  const suffix = query ? `?${query}` : "";
  setStatus(`正在分析 ${state.activeModelId} ...`);

  try {
    const previousSelected = state.selectedNodeId;
    const payload = await fetchJson(`/api/models/${encodeURIComponent(state.activeModelId)}${suffix}`);
    state.payload = payload;
    state.selectedNodeId = (payload.graph?.nodes || []).some((node) => node.id === previousSelected) ? previousSelected : payload.selectedNodeId;
    syncSelectedNode(payload);
    updateExportButtons();
    renderHierarchyToolbar();
    renderModelList();
    renderSummary();
    renderControls();
    renderWarnings();
    renderGraph();
    renderDetails();
    setStatus(`已加载 ${payload.model.name}，当前展示 ${payload.model.type} 图结构。`);
  } catch (error) {
    console.error(error);
    state.payload = null;
    updateExportButtons();
    setStatus(`加载失败：${error.message}`, true);
    ui.summaryPanel.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
    ui.graphBoard.innerHTML = "";
    ui.edgeLayer.innerHTML = "";
    ui.detailPanel.innerHTML = '<div class="empty-state">无法读取模型详情。</div>';
  }
}

function renderSummary() {
  const payload = state.payload;
  if (!payload) {
    return;
  }

  const graph = getVisibleGraph(payload);
  const metrics = payload.model.summary
    .map(
      (item) => `
        <div class="metric-card">
          <div class="metric-label">${escapeHtml(item.label)}</div>
          <div class="metric-value">${escapeHtml(item.value)}</div>
        </div>
      `
    )
    .join("");
  const sources = (payload.model.sources || [])
    .map((source) => `<span class="source-chip">${escapeHtml(source)}</span>`)
    .join("");

  const graphStats = renderChipPairs([
    { label: "节点", value: graph.nodes.length },
    { label: "连线", value: graph.edges.length },
    { label: "泳道", value: graph.lanes.length },
    { label: "警告", value: payload.warnings?.length || 0 },
  ]);

  const primaryPath = computePrimaryPath(graph);
  const primaryPathMarkup = primaryPath.length
    ? primaryPath
        .map((node) => `<span class="flow-chip path-chip"><span class="flow-chip-value">${escapeHtml(node.label)}</span></span>`)
        .join("")
    : '<span class="subtle">当前图没有可提取的主链路。</span>';

  const parameterEntries = Object.entries(payload.parameters || {}).filter(([, value]) => value !== undefined && value !== null && value !== "" && value !== 0);
  const parameterMarkup = parameterEntries.length
    ? renderChipPairs(parameterEntries.map(([label, value]) => ({ label, value })), "flow-chip param-chip")
    : '<span class="subtle">当前模型没有额外运行参数。</span>';
  const hierarchyMarkup = payload.model.type === "llm"
    ? `
      <div class="detail-section">
        <h3>LLM 层级视图</h3>
        <div class="flow-strip">${renderChipPairs([{ label: "当前视图", value: getHierarchyModeLabel(state.llmHierarchyMode, payload) }])}</div>
      </div>
    `
    : "";

  ui.summaryPanel.innerHTML = `
    <div class="summary-top">
      <div class="tag-row">
        <span class="tag">${escapeHtml(payload.model.type)}</span>
        <span class="tag">${escapeHtml(payload.model.architecture)}</span>
      </div>
      <h2>${escapeHtml(payload.model.name)}</h2>
      <p class="headline">${escapeHtml(payload.model.headline)}</p>
    </div>
    <div class="summary-grid">${metrics}</div>
    <div class="detail-section">
      <h3>图谱摘要</h3>
      <div class="flow-strip">${graphStats}</div>
    </div>
    <div class="detail-section">
      <h3>主链路</h3>
      <div class="flow-strip">${primaryPathMarkup}</div>
    </div>
    <div class="detail-section">
      <h3>当前参数</h3>
      <div class="flow-strip">${parameterMarkup}</div>
    </div>
    ${hierarchyMarkup}
    <div class="detail-section">
      <h3>来源配置</h3>
      <div class="source-list">${sources || '<span class="subtle">未检测到配置文件。</span>'}</div>
    </div>
  `;
}

function renderControls() {
  const payload = state.payload;
  if (!payload) {
    return;
  }

  if (!(payload.controls || []).length) {
    ui.controlsForm.innerHTML = '<div class="empty-state">当前模型无需额外运行参数。</div>';
    return;
  }

  ui.controlsForm.innerHTML = payload.controls
    .map(
      (control) => `
        <label class="control-group">
          <span>${escapeHtml(control.label)}</span>
          <input
            class="control-input"
            name="${escapeHtml(control.name)}"
            type="${escapeHtml(control.type || "number")}" 
            value="${escapeHtml(control.value)}"
            min="${escapeHtml(control.min ?? "")}" 
            max="${escapeHtml(control.max ?? "")}" 
            step="${escapeHtml(control.step ?? 1)}"
          />
          <span class="control-meta">${escapeHtml(control.help || "")}</span>
        </label>
      `
    )
    .join("");
}

function renderWarnings() {
  const warnings = state.payload?.warnings || [];
  if (!warnings.length) {
    ui.warningsPanel.innerHTML = "";
    return;
  }

  ui.warningsPanel.innerHTML = warnings
    .map((warning) => `<div class="warning-card">${escapeHtml(warning)}</div>`)
    .join("");
}

function groupNodesByLane(nodes, lanes) {
  const grouped = new Map();
  lanes.forEach((lane) => grouped.set(lane.id, []));
  nodes.forEach((node) => {
    if (!grouped.has(node.lane)) {
      grouped.set(node.lane, []);
    }
    grouped.get(node.lane).push(node);
  });
  grouped.forEach((laneNodes) => laneNodes.sort((a, b) => a.order - b.order));
  return grouped;
}

function buildEdgeGeometry(sourceRect, targetRect, canvasRect) {
  const sourceCx = sourceRect.left - canvasRect.left + sourceRect.width / 2;
  const sourceCy = sourceRect.top - canvasRect.top + sourceRect.height / 2;
  const targetCx = targetRect.left - canvasRect.left + targetRect.width / 2;
  const targetCy = targetRect.top - canvasRect.top + targetRect.height / 2;
  const dx = targetCx - sourceCx;
  const dy = targetCy - sourceCy;
  const horizontal = Math.abs(dx) > Math.abs(dy);

  if (horizontal) {
    const goingRight = dx > 0;
    const startX = goingRight ? sourceRect.right - canvasRect.left : sourceRect.left - canvasRect.left;
    const endX = goingRight ? targetRect.left - canvasRect.left : targetRect.right - canvasRect.left;
    const startY = sourceCy;
    const endY = targetCy;
    const curve = Math.max(42, Math.abs(endX - startX) * 0.38);
    const cp1x = startX + (goingRight ? curve : -curve);
    const cp2x = endX + (goingRight ? -curve : curve);
    const path = `M ${startX} ${startY} C ${cp1x} ${startY}, ${cp2x} ${endY}, ${endX} ${endY}`;
    return { startX, startY, endX, endY, path, labelX: (startX + endX) / 2, labelY: (startY + endY) / 2 - 8 };
  }

  const goingDown = dy > 0;
  const startX = sourceCx;
  const startY = goingDown ? sourceRect.bottom - canvasRect.top : sourceRect.top - canvasRect.top;
  const endX = targetCx;
  const endY = goingDown ? targetRect.top - canvasRect.top : targetRect.bottom - canvasRect.top;
  const curve = Math.max(42, Math.abs(endY - startY) * 0.38);
  const cp1y = startY + (goingDown ? curve : -curve);
  const cp2y = endY + (goingDown ? -curve : curve);
  const path = `M ${startX} ${startY} C ${startX} ${cp1y}, ${endX} ${cp2y}, ${endX} ${endY}`;
  return { startX, startY, endX, endY, path, labelX: (startX + endX) / 2, labelY: (startY + endY) / 2 - 8 };
}

function renderGraph() {
  const payload = state.payload;
  if (!payload) {
    return;
  }

  const visibleGraph = getVisibleGraph(payload);
  const relations = buildGraphRelations(payload);
  const grouped = groupNodesByLane(visibleGraph.nodes || [], visibleGraph.lanes || []);
  ui.graphBoard.innerHTML = (visibleGraph.lanes || [])
    .map((lane) => {
      const nodes = grouped.get(lane.id) || [];
      const nodeMarkup = nodes
        .map((node) => {
          const relationClass = relations.nodeClass(node.id);
          const classes = ["graph-node", relationClass];
          if (node.id === state.selectedNodeId) {
            classes.push("active");
          }
          const incomingCount = relations.index.incomingNeighbors.get(node.id)?.length || 0;
          const outgoingCount = relations.index.outgoingNeighbors.get(node.id)?.length || 0;
          return `
            <article class="${classes.join(" ")}" data-node-id="${escapeHtml(node.id)}" data-accent="${escapeHtml(node.accent || "core")}" ${node.parentId ? `data-parent-id="${escapeHtml(node.parentId)}"` : ""}>
              <div>
                <h3 class="node-title">${escapeHtml(node.label)}</h3>
                <div class="node-subtitle">${escapeHtml(node.subtitle || "")}</div>
              </div>
              <div class="badge-row">
                ${(node.badges || []).map((badge) => `<span class="badge">${escapeHtml(badge)}</span>`).join("")}
              </div>
              ${(node.microFlow || []).length ? `<div class="micro-flow">${node.microFlow.map((item) => `<span class="micro-chip">${escapeHtml(item)}</span>`).join("")}</div>` : ""}
              <div class="shape-block">
                <div><span class="shape-label">IN</span> ${escapeHtml(node.inputShape || "-")}</div>
                <div><span class="shape-label">OUT</span> ${escapeHtml(node.outputShape || "-")}</div>
              </div>
              <div class="node-io-hint">${incomingCount} in / ${outgoingCount} out</div>
            </article>
          `;
        })
        .join("");

      return `
        <section class="lane">
          <div class="lane-title">${escapeHtml(lane.label)}</div>
          <div class="lane-body">${nodeMarkup || '<div class="empty-state">无节点</div>'}</div>
        </section>
      `;
    })
    .join("");

  requestAnimationFrame(drawEdges);
}

function drawEdges() {
  const payload = state.payload;
  if (!payload) {
    return;
  }

  const visibleGraph = getVisibleGraph(payload);
  const relations = buildGraphRelations(payload);
  const canvasRect = ui.graphCanvas.getBoundingClientRect();
  const nodes = new Map();
  ui.graphBoard.querySelectorAll(".graph-node").forEach((element) => {
    nodes.set(element.dataset.nodeId, element);
  });

  const width = Math.max(ui.graphBoard.scrollWidth + 24, ui.graphBoard.clientWidth);
  const height = Math.max(ui.graphBoard.scrollHeight + 24, ui.graphBoard.clientHeight);
  ui.edgeLayer.setAttribute("viewBox", `0 0 ${width} ${height}`);
  ui.edgeLayer.setAttribute("width", String(width));
  ui.edgeLayer.setAttribute("height", String(height));

  let markup = `
    <defs>
      <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto-start-reverse">
        <path d="M0,0 L12,6 L0,12 z" fill="rgba(76, 93, 108, 0.56)"></path>
      </marker>
    </defs>
  `;

  (visibleGraph.edges || []).forEach((edge) => {
    const source = nodes.get(edge.source);
    const target = nodes.get(edge.target);
    if (!source || !target) {
      return;
    }

    const geometry = buildEdgeGeometry(source.getBoundingClientRect(), target.getBoundingClientRect(), canvasRect);
    const labelText = escapeHtml(edge.label || "");
    const labelWidth = Math.max(56, labelText.length * 7.4);
    const relationClass = relations.edgeClass(edge);

    markup += `
      <path class="edge-path ${relationClass}" d="${geometry.path}" marker-end="url(#arrow)"></path>
      <rect class="edge-label-bg ${relationClass}" x="${geometry.labelX - labelWidth / 2}" y="${geometry.labelY - 12}" rx="10" ry="10" width="${labelWidth}" height="24"></rect>
      <text class="edge-label ${relationClass}" x="${geometry.labelX}" y="${geometry.labelY + 4}" text-anchor="middle">${labelText}</text>
    `;
  });

  ui.edgeLayer.innerHTML = markup;
}

function findSelectedNode() {
  const visibleGraph = getVisibleGraph(state.payload);
  return (visibleGraph.nodes || []).find((node) => node.id === state.selectedNodeId) || visibleGraph.nodes?.[0] || null;
}

function renderJumpList(ids, index) {
  if (!ids.length) {
    return '<span class="subtle">无</span>';
  }

  return ids
    .map((nodeId) => {
      const node = index.nodeById.get(nodeId);
      if (!node) {
        return "";
      }
      return `<button class="jump-chip" type="button" data-jump-node="${escapeHtml(node.id)}">${escapeHtml(node.label)}</button>`;
    })
    .join("");
}

function renderDetails() {
  const payload = state.payload;
  const node = findSelectedNode();
  if (!payload || !node) {
    ui.detailPanel.innerHTML = '<div class="empty-state">选择节点后在这里查看细节。</div>';
    return;
  }

  const relations = buildGraphRelations(payload);
  const laneLabel = (payload.graph.lanes || []).find((lane) => lane.id === node.lane)?.label || node.lane;
  const detailRows = (items) => `
    <div class="kv-list">
      ${items
        .map(
          (item) => `
            <div class="kv-row">
              <div class="kv-label">${escapeHtml(item.label)}</div>
              <div class="kv-value">${escapeHtml(item.value)}</div>
            </div>
          `
        )
        .join("")}
    </div>
  `;

  const sections = (node.sections || [])
    .map(
      (block) => `
        <section class="detail-section">
          <h3>${escapeHtml(block.title)}</h3>
          ${detailRows(block.items || [])}
        </section>
      `
    )
    .join("");

  const directIncoming = relations.index.incomingNeighbors.get(node.id) || [];
  const directOutgoing = relations.index.outgoingNeighbors.get(node.id) || [];

  ui.detailPanel.innerHTML = `
    <div class="tag-row">
      <span class="tag">${escapeHtml(payload.model.type)}</span>
      <span class="tag">${escapeHtml(node.accent || "node")}</span>
      <span class="tag">${escapeHtml(laneLabel)}</span>
    </div>
    <h2 class="detail-title">${escapeHtml(node.label)}</h2>
    <p class="detail-intro">${escapeHtml(node.description || "")}</p>
    <section class="detail-section">
      <h3>Shape</h3>
      ${detailRows([
        { label: "输入", value: node.inputShape || "-" },
        { label: "输出", value: node.outputShape || "-" },
      ])}
    </section>
    <section class="detail-section">
      <h3>图关系</h3>
      <div class="flow-strip">
        ${renderChipPairs([
          { label: "直接上游", value: directIncoming.length },
          { label: "直接下游", value: directOutgoing.length },
          { label: "可达上游", value: relations.upstream.size },
          { label: "可达下游", value: relations.downstream.size },
        ])}
      </div>
      <div class="jump-block">
        <div class="kv-label">直接上游</div>
        <div class="jump-list">${renderJumpList(directIncoming, relations.index)}</div>
      </div>
      <div class="jump-block">
        <div class="kv-label">直接下游</div>
        <div class="jump-list">${renderJumpList(directOutgoing, relations.index)}</div>
      </div>
    </section>
    <section class="detail-section">
      <h3>关键字段</h3>
      ${detailRows([
        { label: "lane", value: laneLabel },
        { label: "order", value: node.order },
        ...(node.details || []),
      ])}
    </section>
    ${sections}
  `;
}

function scheduleRefresh() {
  clearTimeout(state.refreshTimer);
  state.refreshTimer = setTimeout(() => {
    loadModelPayload();
  }, 220);
}

function downloadBlob(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function sanitizeFileName(value) {
  return String(value).replace(/[\\/:*?"<>|]+/g, "-");
}

function truncateText(value, maxLength = 46) {
  const text = String(value || "");
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength - 1)}...`;
}

function accentColor(accent) {
  if (accent === "vision") {
    return "#cc7d21";
  }
  if (accent === "scheduler" || accent === "latent") {
    return "#5180a4";
  }
  if (accent === "output" || accent === "head" || accent === "decode") {
    return "#b65c45";
  }
  return "#0d6a64";
}

function relationColor(kind) {
  if (kind === "upstream") {
    return "#b58332";
  }
  if (kind === "downstream") {
    return "#0d6a64";
  }
  if (kind === "related") {
    return "#5a768f";
  }
  return "rgba(76, 93, 108, 0.24)";
}

function exportCurrentPayload() {
  if (!state.payload) {
    return;
  }
  const exported = {
    ...state.payload,
    currentHierarchyMode: getHierarchyMode(state.payload),
    visibleGraph: getVisibleGraph(state.payload),
  };
  downloadBlob(`${sanitizeFileName(state.payload.model.id)}.json`, `${JSON.stringify(exported, null, 2)}\n`, "application/json;charset=utf-8");
}

function exportCurrentSvg() {
  if (!state.payload) {
    return;
  }

  const payload = state.payload;
  const relations = buildGraphRelations(payload);
  const visibleGraph = getVisibleGraph(payload);
  const canvasRect = ui.graphCanvas.getBoundingClientRect();
  const width = Math.max(ui.graphBoard.scrollWidth + 24, ui.graphBoard.clientWidth);
  const height = Math.max(ui.graphBoard.scrollHeight + 24, ui.graphBoard.clientHeight);
  const nodeElements = new Map();
  ui.graphBoard.querySelectorAll(".graph-node").forEach((element) => {
    nodeElements.set(element.dataset.nodeId, element);
  });

  const edgeMarkup = (visibleGraph.edges || [])
    .map((edge) => {
      const source = nodeElements.get(edge.source);
      const target = nodeElements.get(edge.target);
      if (!source || !target) {
        return "";
      }

      const geometry = buildEdgeGeometry(source.getBoundingClientRect(), target.getBoundingClientRect(), canvasRect);
      const label = truncateText(edge.label || "", 24);
      const labelWidth = Math.max(56, label.length * 7.2);
      const color = relationColor(relations.edgeClass(edge));
      return `
        <path d="${geometry.path}" fill="none" stroke="${color}" stroke-width="2.4" marker-end="url(#arrow)" />
        <rect x="${geometry.labelX - labelWidth / 2}" y="${geometry.labelY - 12}" width="${labelWidth}" height="24" rx="10" ry="10" fill="rgba(255,252,246,0.94)" stroke="rgba(33,25,16,0.08)" />
        <text x="${geometry.labelX}" y="${geometry.labelY + 4}" fill="#655d51" font-family="Aptos, Microsoft YaHei UI, sans-serif" font-size="12" text-anchor="middle">${escapeHtml(label)}</text>
      `;
    })
    .join("");

  const nodeMarkup = (visibleGraph.nodes || [])
    .map((node) => {
      const element = nodeElements.get(node.id);
      if (!element) {
        return "";
      }

      const rect = element.getBoundingClientRect();
      const x = rect.left - canvasRect.left;
      const y = rect.top - canvasRect.top;
      const widthPx = rect.width;
      const heightPx = rect.height;
      const accent = accentColor(node.accent || "core");
      const relation = relations.nodeClass(node.id);
      const opacity = relation === "dimmed" ? 0.34 : 1;
      const stroke = node.id === relations.selectedId ? "#0d6a64" : "rgba(31,26,20,0.08)";
      const title = truncateText(node.label, 26);
      const subtitle = truncateText(node.subtitle || "", 38);
      const input = truncateText(`IN ${node.inputShape || "-"}`, 38);
      const output = truncateText(`OUT ${node.outputShape || "-"}`, 38);
      return `
        <g opacity="${opacity}">
          <rect x="${x}" y="${y}" width="${widthPx}" height="${heightPx}" rx="22" ry="22" fill="rgba(255,248,235,0.98)" stroke="${stroke}" />
          <rect x="${x}" y="${y}" width="6" height="${heightPx}" rx="6" ry="6" fill="${accent}" />
          <text x="${x + 18}" y="${y + 28}" fill="#1d1f1c" font-family="Bahnschrift, Aptos Display, Microsoft YaHei UI, sans-serif" font-size="16">${escapeHtml(title)}</text>
          <text x="${x + 18}" y="${y + 48}" fill="#655d51" font-family="Aptos, Microsoft YaHei UI, sans-serif" font-size="12">${escapeHtml(subtitle)}</text>
          <rect x="${x + 16}" y="${y + heightPx - 54}" width="${Math.max(120, widthPx - 32)}" height="38" rx="12" ry="12" fill="rgba(246,242,232,0.9)" stroke="rgba(33,25,16,0.08)" />
          <text x="${x + 28}" y="${y + heightPx - 31}" fill="#655d51" font-family="Cascadia Code, Consolas, monospace" font-size="11">${escapeHtml(input)}</text>
          <text x="${x + 28}" y="${y + heightPx - 17}" fill="#655d51" font-family="Cascadia Code, Consolas, monospace" font-size="11">${escapeHtml(output)}</text>
        </g>
      `;
    })
    .join("");

  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f7f0de" />
      <stop offset="100%" stop-color="#e9dfc7" />
    </linearGradient>
    <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto-start-reverse">
      <path d="M0,0 L12,6 L0,12 z" fill="rgba(76,93,108,0.56)" />
    </marker>
  </defs>
  <rect width="100%" height="100%" fill="url(#bg)" />
  ${edgeMarkup}
  ${nodeMarkup}
</svg>`;

  downloadBlob(`${sanitizeFileName(payload.model.id)}.svg`, svg, "image/svg+xml;charset=utf-8");
}

ui.modelSearch.addEventListener("input", renderModelList);

ui.modelList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-model-id]");
  if (!button) {
    return;
  }
  state.activeModelId = button.dataset.modelId;
  loadModelPayload();
});

ui.controlsForm.addEventListener("input", scheduleRefresh);
ui.refreshButton.addEventListener("click", () => loadModelPayload());
ui.hierarchyToolbar.addEventListener("click", (event) => {
  const button = event.target.closest("[data-hierarchy-mode]");
  if (!button || !state.payload || state.payload.model.type !== "llm") {
    return;
  }
  state.llmHierarchyMode = button.dataset.hierarchyMode;
  syncSelectedNode(state.payload);
  renderHierarchyToolbar();
  renderSummary();
  renderGraph();
  renderDetails();
  setStatus(`已切换为 ${getHierarchyModeLabel(state.llmHierarchyMode)} 层级视图。`);
});
ui.exportJsonButton.addEventListener("click", exportCurrentPayload);
ui.exportSvgButton.addEventListener("click", exportCurrentSvg);

ui.graphBoard.addEventListener("click", (event) => {
  const nodeElement = event.target.closest("[data-node-id]");
  if (!nodeElement) {
    return;
  }
  state.selectedNodeId = nodeElement.dataset.nodeId;
  renderGraph();
  renderDetails();
});

ui.detailPanel.addEventListener("click", (event) => {
  const jumpButton = event.target.closest("[data-jump-node]");
  if (!jumpButton) {
    return;
  }
  state.selectedNodeId = jumpButton.dataset.jumpNode;
  renderGraph();
  renderDetails();
});

window.addEventListener("resize", () => {
  if (state.payload) {
    requestAnimationFrame(drawEdges);
  }
});

updateExportButtons();
renderHierarchyToolbar();
loadModels().catch((error) => {
  console.error(error);
  setStatus(`初始化失败：${error.message}`, true);
});