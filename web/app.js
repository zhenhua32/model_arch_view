const state = {
  models: [],
  activeModelId: null,
  payload: null,
  compareModelId: null,
  comparePayload: null,
  compareController: null,
  selectedNodeId: null,
  llmHierarchyMode: "summary",
  refreshTimer: null,
  fetchController: null,
  zoom: { scale: 1, x: 0, y: 0 },
  pan: { dragging: false, startX: 0, startY: 0, scrollLeft: 0, scrollTop: 0 },
  collapsedParents: new Set(),
  collapsedLanes: new Set(),
  centerView: "graph",
};

const ui = {
  modelCount: document.getElementById("model-count"),
  modelSearch: document.getElementById("model-search"),
  modelList: document.getElementById("model-list"),
  controlsForm: document.getElementById("controls-form"),
  refreshButton: document.getElementById("refresh-button"),
  hierarchyToolbar: document.getElementById("hierarchy-toolbar"),
  exportAuditButton: document.getElementById("export-audit-button"),
  exportJsonButton: document.getElementById("export-json-button"),
  exportSvgButton: document.getElementById("export-svg-button"),
  statusBar: document.getElementById("status-bar"),
  summaryPanel: document.getElementById("model-summary"),
  compareSection: document.getElementById("compare-section"),
  compareSelect: document.getElementById("compare-select"),
  compareBody: document.getElementById("compare-body"),
  compareClear: document.getElementById("compare-clear"),
  warningsPanel: document.getElementById("warnings-panel"),
  graphScroll: document.getElementById("graph-scroll"),
  graphCanvas: document.getElementById("graph-canvas"),
  graphBoard: document.getElementById("graph-board"),
  edgeLayer: document.getElementById("edge-layer"),
  detailPanel: document.getElementById("detail-panel"),
  zoomIn: document.getElementById("zoom-in"),
  zoomOut: document.getElementById("zoom-out"),
  zoomReset: document.getElementById("zoom-reset"),
  zoomLevel: document.getElementById("zoom-level"),
  nodeSearch: document.getElementById("node-search"),
  viewTabs: document.querySelector(".view-tabs"),
  graphView: document.getElementById("graph-view"),
  detailsView: document.getElementById("details-view"),
  auditView: document.getElementById("audit-view"),
  auditPanel: document.getElementById("audit-panel"),
  compareView: document.getElementById("compare-view"),
  viewStage: document.getElementById("view-stage"),
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

async function fetchJson(url, signal) {
  const response = await fetch(url, { cache: "no-store", signal });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  return response.json();
}

function updateExportButtons() {
  const disabled = !state.payload;
  ui.exportAuditButton.disabled = disabled;
  ui.exportJsonButton.disabled = disabled;
  ui.exportSvgButton.disabled = disabled;
}

function extractRepeatCount(payload = state.payload) {
  const modelType = payload?.model?.type;
  const label = modelType === "diffusers" ? "推理步数" : modelType === "multimodal" ? "主干层数" : "层数";
  const summaryItem = payload?.model?.summary?.find?.((item) => item.label === label);
  const numeric = Number.parseInt(summaryItem?.value ?? "", 10);
  return Number.isFinite(numeric) ? numeric : null;
}

function getHierarchyModeLabel(mode, payload = state.payload) {
  const modelType = payload?.model?.type;
  if (mode === "summary") {
    return "汇总";
  }
  if (mode === "block") {
    if (modelType === "diffusers") {
      return "Transformer 内部（Patchify/Attn/Cross-Attn/FFN/AdaLN）";
    }
    if (modelType === "multimodal") {
      return "编码器内部（Vision Attn/Cross-Modal/FFN）";
    }
    return "单层 Block（Q/K/V/RoPE/Score/Causal/Window/Softmax/Output）";
  }
  if (mode === "repeat") {
    const count = extractRepeatCount(payload);
    const unit = modelType === "diffusers" ? "步" : "层";
    return count ? `重复 ${count} ${unit}摘要` : `重复 N ${unit}摘要`;
  }
  return mode;
}

const HIERARCHY_TYPES = new Set(["llm", "diffusers", "multimodal"]);
const HIERARCHY_MODES = ["summary", "block", "repeat"];

function getAvailableHierarchyModes(payload = state.payload) {
  if (!HIERARCHY_TYPES.has(payload?.model?.type)) {
    return new Set(["all"]);
  }
  const nodes = payload?.graph?.nodes || [];
  return new Set(HIERARCHY_MODES.filter((mode) => nodes.some((node) => isVisibleInHierarchy(node, mode))));
}

function getHierarchyMode(payload = state.payload) {
  if (!HIERARCHY_TYPES.has(payload?.model?.type)) {
    return "all";
  }
  const availableModes = getAvailableHierarchyModes(payload);
  if (availableModes.has(state.llmHierarchyMode)) {
    return state.llmHierarchyMode;
  }
  if (availableModes.has("summary")) {
    return "summary";
  }
  return availableModes.values().next().value || "all";
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
  let nodes = (graph.nodes || []).filter((node) => isVisibleInHierarchy(node, mode));
  nodes = nodes.filter((node) => {
    if (!node.parentId) return true;
    return !state.collapsedParents.has(node.parentId);
  });
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
  const modelType = state.payload?.model?.type;
  const availableModes = getAvailableHierarchyModes();
  const hasHierarchy = state.centerView === "graph" && HIERARCHY_TYPES.has(modelType) && availableModes.size > 1;
  ui.hierarchyToolbar.hidden = !hasHierarchy;
  if (!hasHierarchy) {
    return;
  }

  const activeMode = getHierarchyMode();
  const blockLabel = modelType === "diffusers" ? "Transformer 内部" : modelType === "multimodal" ? "编码器内部" : "Q/K/V/Causal";
  const repeatLabel = modelType === "diffusers" ? "重复 N 步" : "重复 N 层";

  ui.hierarchyToolbar.querySelectorAll("[data-hierarchy-mode]").forEach((button) => {
    const supported = availableModes.has(button.dataset.hierarchyMode);
    button.hidden = !supported;
    button.disabled = !supported;
    button.classList.toggle("is-active", button.dataset.hierarchyMode === activeMode);
    if (button.dataset.hierarchyMode === "block") {
      button.textContent = blockLabel;
    }
    if (button.dataset.hierarchyMode === "repeat") {
      button.textContent = repeatLabel;
    }
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
    ui.auditPanel.innerHTML = '<div class="empty-state">当前没有可审计的模型目录。</div>';
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

  if (state.fetchController) {
    state.fetchController.abort();
  }
  state.fetchController = new AbortController();

  const query = buildQueryFromControls();
  const suffix = query ? `?${query}` : "";
  setStatus(`正在分析 ${state.activeModelId} ...`);

  try {
    const previousSelected = state.selectedNodeId;
    const payload = await fetchJson(`/api/models/${encodeURIComponent(state.activeModelId)}${suffix}`, state.fetchController.signal);
    state.payload = payload;
    state.selectedNodeId = (payload.graph?.nodes || []).some((node) => node.id === previousSelected) ? previousSelected : payload.selectedNodeId;
    syncSelectedNode(payload);
    updateExportButtons();
    renderHierarchyToolbar();
    renderModelList();
    renderSummary();
    renderAudit();
    renderControls();
    renderWarnings();
    renderGraph();
    renderDetails();
    renderCompareSelect();
    loadComparePayload();
    setStatus(`已加载 ${payload.model.name}，当前展示 ${payload.model.type} 图结构。`);
  } catch (error) {
    if (error.name === "AbortError") {
      return;
    }
    console.error(error);
    state.payload = null;
    updateExportButtons();
    renderHierarchyToolbar();
    renderWarnings();
    ui.controlsForm.innerHTML = '<div class="empty-state">无法加载运行参数。</div>';
    setStatus(`加载失败：${error.message}`, true);
    ui.summaryPanel.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
    ui.auditPanel.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
    ui.graphBoard.innerHTML = "";
    ui.edgeLayer.innerHTML = "";
    ui.detailPanel.innerHTML = '<div class="empty-state">无法读取模型详情。</div>';
  }
}

const BREAKDOWN_META = [
  { key: "attention", label: "注意力", color: "#378ADD" },
  { key: "routed_experts", label: "路由专家", color: "#1D9E75" },
  { key: "shared_experts", label: "共享专家", color: "#EF9F27" },
  { key: "dense_ffn", label: "稠密 FFN", color: "#7F77DD" },
  { key: "embedding", label: "嵌入 / 输出头", color: "#888780" },
  { key: "mtp", label: "MTP 辅助层", color: "#D4537E" },
];

const MEM_META = [
  { key: "weights_bytes", label: "权重", color: "#378ADD" },
  { key: "kv_bytes", label: "KV cache", color: "#1D9E75" },
  { key: "activation_bytes", label: "激活(粗估)", color: "#EF9F27" },
];

function formatB(v) {
  return `${(v / 1e9).toFixed(2)}B`;
}

function formatGB(bytes) {
  const gb = bytes / 1024 ** 3;
  if (gb >= 1) return `${gb.toFixed(1)} GiB`;
  return `${(bytes / 1024 ** 2).toFixed(0)} MiB`;
}

function formatTps(v) {
  return v >= 1 ? v.toFixed(0) : v.toFixed(1);
}

function formatCount(v) {
  return Number(v || 0).toLocaleString("en-US");
}

function formulaRow(k, expr, val) {
  return `<div class="formula-row"><span class="formula-k">${escapeHtml(k)}</span><span class="formula-eq">${escapeHtml(expr)}</span><span class="formula-v">${escapeHtml(val)}</span></div>`;
}

function renderStackedBar(items, total) {
  if (!total) return "";
  const segments = items
    .filter((it) => it.value > 0)
    .map(
      (it) =>
        `<span class="stack-seg" style="width:${(it.value / total * 100).toFixed(2)}%;background:${it.color}" title="${escapeHtml(it.label)} ${(it.value / total * 100).toFixed(1)}%"></span>`
    )
    .join("");
  return `<div class="stack-bar">${segments}</div>`;
}

function renderBreakdownSection(metrics) {
  const bd = metrics.breakdown || {};
  const total = metrics.total_params || 0;
  if (!total) return "";
  const items = BREAKDOWN_META.map((m) => ({ ...m, value: bd[m.key] || 0 }));
  const bar = renderStackedBar(items, total);
  const legend = items
    .filter((it) => it.value > 0)
    .sort((a, b) => b.value - a.value)
    .map(
      (it) => `
        <div class="legend-row">
          <span class="legend-dot" style="background:${it.color}"></span>
          <span class="legend-name">${escapeHtml(it.label)}</span>
          <span class="legend-val">${formatB(it.value)}</span>
          <span class="legend-pct">${(it.value / total * 100).toFixed(1)}%</span>
        </div>`
    )
    .join("");
  const activeBar = metrics.is_moe
    ? `<div class="active-row">
         <span class="active-label">激活参数</span>
         <div class="stack-bar thin"><span class="stack-seg" style="width:${(metrics.active_params / total * 100).toFixed(2)}%;background:#1D9E75"></span></div>
         <span class="active-val">${formatB(metrics.active_params)} · ${(metrics.active_params / total * 100).toFixed(1)}%</span>
       </div>`
    : "";

  const ft = metrics.formula_terms || {};
  const attnNote = ft.is_deepseek_v4
    ? `DeepSeek V4 = q 低秩 + 单头共享 K=V + grouped-o 低秩；再按每层 compress ratio 加 CSA/HCA compressor、indexer 与 mHC 参数。`
    : ft.is_mla
      ? `注意力(层) = hidden×q_lora + q_lora×heads×qk_head_dim + hidden×(kv_lora+qk_rope) + kv_lora×heads×(qk_nope+v_head) + heads×v_head×hidden`
      : `注意力(层) = hidden×q_dim + hidden×kv_dim×2 + q_dim×hidden，其中 q_dim=heads×head_dim，kv_dim=kv_heads×head_dim`;
  const attnExpr = ft.is_deepseek_v4
    ? `${ft.num_layers} 层按压缩率逐层求和（平均 ${formatCount(ft.attn_per_layer)}/层）`
    : `${formatCount(ft.attn_per_layer)} × ${ft.num_layers} 层`;
  const rows = [formulaRow("注意力", attnExpr, formatB(bd.attention))];
  if (metrics.is_moe) {
    rows.push(formulaRow("路由专家", `${ft.n_moe_layers} 层 × ${ft.num_experts} 专家 × ${formatCount(ft.expert_per)}`, formatB(bd.routed_experts)));
    if (bd.shared_experts > 0) {
      rows.push(formulaRow("共享专家", `${ft.n_moe_layers} 层 × ${ft.n_shared_experts} 共享 × ${formatCount(ft.shared_expert_per)}`, formatB(bd.shared_experts)));
    }
  }
  if (bd.dense_ffn > 0) {
    rows.push(formulaRow("稠密 FFN", `${ft.n_dense_layers} 层 × ${formatCount(ft.dense_ffn_per)}`, formatB(bd.dense_ffn)));
  }
  if (bd.mtp > 0) {
    rows.push(formulaRow("MTP", `${ft.mtp_layers} 个辅助预测层`, formatB(bd.mtp)));
  }
  const embedExpr = ft.has_output_head
    ? `token embedding + ${ft.tie_word_embeddings ? "共享 LM Head" : "独立 LM Head"}`
    : "token + position + token-type embedding（无 LM Head）";
  rows.push(formulaRow("嵌入/输出头", embedExpr, formatB(bd.embedding)));
  rows.push(formulaRow("总计", `注意力 + 路由 + 共享 + 稠密 + 嵌入 + MTP`, formatB(total)));
  const formula = `
    <details class="formula-box">
      <summary>计算公式</summary>
      <div class="formula-note">${escapeHtml(attnNote)}</div>
      <div class="formula-list">${rows.join("")}</div>
    </details>`;

  return `
    <div class="detail-section">
      <h3>参数构成 <span class="subtle">共 ${formatB(total)}</span></h3>
      ${bar}
      <div class="legend">${legend}</div>
      ${activeBar}
      ${formula}
    </div>`;
}

function renderMemorySection(metrics) {
  const mem = metrics.memory;
  if (!mem) return "";
  const items = MEM_META.map((m) => ({ ...m, value: mem[m.key] || 0 }));
  const bar = renderStackedBar(items, mem.total_bytes);
  const legend = items
    .filter((it) => it.value > 0)
    .map(
      (it) => `
        <div class="legend-row">
          <span class="legend-dot" style="background:${it.color}"></span>
          <span class="legend-name">${escapeHtml(it.label)}</span>
          <span class="legend-val">${formatGB(it.value)}</span>
        </div>`
    )
    .join("");
  const gpus = (mem.gpu_fit || [])
    .map((g) => {
      const cls = g.count <= 1 ? "gpu-ok" : g.count <= 4 ? "gpu-mid" : "gpu-heavy";
      return `<div class="gpu-card ${cls}"><span class="gpu-name">${escapeHtml(g.name)}</span><span class="gpu-vram">${g.mem_gb} GB</span><span class="gpu-count">×${g.count}</span></div>`;
    })
    .join("");
  const ft = metrics.formula_terms || {};
  const bpp = mem.bytes_per_param;
  const checkpointBpp = mem.checkpoint_bytes_per_param;
  const weightExpr = mem.weight_source === "checkpoint"
    ? `safetensors index 的实际 checkpoint 字节数（有效 ${checkpointBpp != null ? checkpointBpp.toFixed(3) : "?"} B/参数，包含量化元数据与辅助层）`
    : `总参数 × ${bpp}B/参数 = ${formatB(metrics.total_params)} × ${bpp}`;
  const memRows = [
    formulaRow("权重", weightExpr, formatGB(mem.weights_bytes)),
    formulaRow("KV cache", `按 full/sliding 层分别应用上下文上限 × batch ${mem.batch}`, formatGB(mem.kv_bytes)),
    formulaRow("激活", `单层 workspace 复用：batch×seq×hidden×${mem.activation_factor}×2B`, formatGB(mem.activation_bytes)),
    formulaRow("总需求", `权重 + KV + 激活`, `${mem.total_gb.toFixed(1)} GiB`),
  ];
  const gpuNote = mem.gpu_fit && mem.gpu_fit.length
    ? `GPU 卡数容量下界 = ⌈总需求 / (显存 × 0.9)⌉；未校验并行切分约束和通信 buffer。`
    : "";
  const formula = `
    <details class="formula-box">
      <summary>计算公式</summary>
      <div class="formula-list">${memRows.join("")}</div>
      <div class="formula-note">${escapeHtml(gpuNote)}</div>
    </details>`;
  return `
    <div class="detail-section">
      <h3>显存 &amp; 成本 <span class="est-badge">粗估 · ${escapeHtml(mem.weight_format || mem.precision)}</span></h3>
      <div class="mem-total">总需求 ≈ ${mem.total_gb.toFixed(1)} GiB <span class="subtle">(batch ${mem.batch} · seq ${mem.seq_len})</span></div>
      ${bar}
      <div class="legend">${legend}</div>
      <div class="gpu-grid">${gpus}</div>
      ${formula}
    </div>`;
}

function renderThroughputSection(metrics) {
  const rows = metrics.throughput || [];
  if (!rows.length) return "";
  const tt = metrics.throughput_terms || {};
  const act = tt.active_params || metrics.active_params || 0;
  const firstRow = rows[0] || {};
  const bpp = firstRow.active_bytes_per_param || tt.bytes_per_param || (metrics.memory ? metrics.memory.bytes_per_param : 2);
  const weightFormat = tt.uses_native_profile ? tt.weight_format : tt.precision || "bf16";
  const computePrecision = firstRow.compute_precision || tt.precision || "bf16";
  const seq = tt.seq_len || 2048;
  const mfu = tt.mfu != null ? tt.mfu : 0.40;

  const body = rows
    .map(
      (r) => `
        <div class="tp-row">
          <span class="tp-name">${escapeHtml(r.name)}${r.gpu_count > 1 ? ` ×${r.gpu_count}` : ""}</span>
          <span class="tp-tps">${formatTps(r.decode_tps)} tok/s</span>
          <span class="tp-ttft">${r.ttft_ms.toFixed(0)} ms</span>
          <span class="tp-bound">${escapeHtml(r.bound)}</span>
        </div>`
    )
    .join("");

  const gpuRows = rows
    .map((r) => {
      return `<div class="tp-sub">
          <span class="tp-sub-name">${escapeHtml(r.name)}${r.gpu_count > 1 ? ` ×${r.gpu_count}` : ""}</span>
          <span class="tp-sub-formula">算力上限 = ${r.per_gpu_tops} ${escapeHtml(r.compute_precision || computePrecision)} TOPS × ${r.gpu_count} 卡 × MFU ${mfu} / ${formatCount(r.decode_flops)} FLOPs/token ≈ ${formatTps(r.compute_tps)} tok/s</span>
          <span class="tp-sub-formula">带宽上限：权重 ${formatGB(r.active_weight_bytes)}/batch ${r.batch} + KV 读取 ${formatGB(r.kv_read_bytes)} ≈ ${formatTps(r.bandwidth_tps)} tok/s</span>
          <span class="tp-sub-bound">→ decode = ${formatTps(r.decode_tps)} tok/s（${escapeHtml(r.bound)}瓶颈）</span>
        </div>`;
    })
    .join("");

  const formula = `
    <details class="formula-box">
      <summary>计算公式</summary>
      <div class="formula-list">
        ${formulaRow("激活参数", "active", formatCount(act))}
        ${formulaRow("激活权重格式", weightFormat, `${bpp.toFixed(3)} B/active 参数`)}
        ${formulaRow("有效算力", `peak = ${escapeHtml(computePrecision)} 原生 TOPS × 10¹² × MFU`, mfu)}
        ${formulaRow("显存带宽", "bw = bw_GB/s × 10⁹", "")}
        ${formulaRow("算力上限", "peak / (线性 FLOPs + 当前上下文 QK/AV FLOPs)", "")}
        ${formulaRow("带宽上限", "bw / (active 权重字节 / batch + 当前上下文 KV 读取)", "")}
        ${formulaRow("decode 吞吐", "min(算力上限, 带宽上限)", "")}
        ${formulaRow("首 token 延迟", `batch ${tt.batch || 1} × (线性项 + causal attention 二次项) / peak`, "")}
      </div>
      <div class="formula-note">各符号：<b>激活</b> = 每 token 实际参与的线性层参数；<b>MFU ${mfu}</b> = 理论峰值利用率；<b>B/参数</b> = 权重存储精度。多卡行按显存容量下界聚合理想算力与带宽，不含跨卡通信损失；decode 同时计入当前上下文的 QK/AV FLOPs 与 KV cache 读取。</div>
      <div class="tp-sub-grid">${gpuRows}</div>
    </details>`;

  return `
    <div class="detail-section">
      <h3>吞吐 / 延迟 <span class="est-badge">理论上限</span></h3>
      <div class="tp-head">
        <span class="tp-name">GPU</span><span class="tp-tps">decode</span><span class="tp-ttft">首 token</span><span class="tp-bound">瓶颈</span>
      </div>
      ${body}
      ${formula}
    </div>`;
}

function renderAnalysisSections(payload) {
  const metrics = payload.metrics;
  if (!metrics) return "";
  return renderBreakdownSection(metrics) + renderMemorySection(metrics) + renderThroughputSection(metrics);
}

// ---- A/B comparison ----

function metricsHaveData(payload) {
  return Boolean(payload && payload.metrics && payload.metrics.total_params);
}

function pickThroughput(metrics, gpuName) {
  const rows = metrics.throughput || [];
  const hit = rows.find((r) => r.name === gpuName) || rows[rows.length - 1];
  return hit || null;
}

const COMPARE_ROWS = [
  { label: "总参数量", get: (m) => m.total_params, fmt: formatB, lowerBetter: true },
  { label: "激活参数", get: (m) => m.active_params, fmt: formatB, lowerBetter: true },
  { label: "每 token FLOPs", get: (m) => m.gflops_per_token * 1e9, fmt: (v) => `${(v / 1e9).toFixed(2)} G`, lowerBetter: true },
  { label: "KV / 1k tok", get: (m) => m.kv_cache_mb_per_1k, fmt: (v) => `${v.toFixed(1)} MiB`, lowerBetter: true },
  { label: "显存需求", get: (m) => (m.memory ? m.memory.total_gb : 0), fmt: (v) => `${v.toFixed(1)} GiB`, lowerBetter: true },
  {
    label: "H100 decode",
    get: (m) => { const t = pickThroughput(m, "H100 80G"); return t ? t.decode_tps : 0; },
    fmt: (v) => `${formatTps(v)} tok/s`,
    lowerBetter: false,
  },
];

function renderDeltaBadge(base, other, lowerBetter) {
  if (!base || !other) return `<span class="cmp-delta same">—</span>`;
  const ratio = (other - base) / base;
  if (Math.abs(ratio) < 0.005) return `<span class="cmp-delta same">≈</span>`;
  const up = other > base;
  const good = lowerBetter ? !up : up;
  const arrow = up ? "▲" : "▼";
  const pct = `${up ? "+" : ""}${(ratio * 100).toFixed(0)}%`;
  return `<span class="cmp-delta ${good ? "good" : "bad"}">${arrow} ${pct}</span>`;
}

function renderCompareTable() {
  const base = state.payload;
  const other = state.comparePayload;
  if (!metricsHaveData(base) || !metricsHaveData(other)) {
    ui.compareBody.innerHTML = "";
    return;
  }
  const ma = base.metrics;
  const mb = other.metrics;
  const rows = COMPARE_ROWS.map((row) => {
    const va = row.get(ma) || 0;
    const vb = row.get(mb) || 0;
    return `
      <div class="cmp-row">
        <span class="cmp-metric">${escapeHtml(row.label)}</span>
        <span class="cmp-a">${row.fmt(va)}</span>
        <span class="cmp-b">${row.fmt(vb)} ${renderDeltaBadge(va, vb, row.lowerBetter)}</span>
      </div>`;
  }).join("");
  ui.compareBody.innerHTML = `
    <div class="cmp-table">
      <div class="cmp-head">
        <span class="cmp-metric">指标</span>
        <span class="cmp-a" title="${escapeHtml(base.model.name)}">A · ${escapeHtml(truncateText(base.model.name, 14))}</span>
        <span class="cmp-b" title="${escapeHtml(other.model.name)}">B · ${escapeHtml(truncateText(other.model.name, 14))}</span>
      </div>
      ${rows}
      <p class="cmp-note">B 列箭头相对 A 变化，绿色=更优（资源更省或吞吐更高）。</p>
    </div>`;
}

function renderCompareSelect() {
  if (!ui.compareSelect) return;
  const showable = metricsHaveData(state.payload);
  ui.compareSection.hidden = !showable;
  if (!showable) {
    state.compareModelId = null;
    state.comparePayload = null;
    return;
  }
  const candidates = state.models.filter(
    (m) => m.type === "llm" && m.id !== state.activeModelId
  );
  if (state.compareModelId && !candidates.some((m) => m.id === state.compareModelId)) {
    state.compareModelId = null;
    state.comparePayload = null;
  }
  const options = [`<option value="">（选择对比模型…）</option>`].concat(
    candidates.map(
      (m) =>
        `<option value="${escapeHtml(m.id)}"${m.id === state.compareModelId ? " selected" : ""}>${escapeHtml(m.name)}</option>`
    )
  );
  ui.compareSelect.innerHTML = options.join("");
  ui.compareClear.hidden = !state.compareModelId;
}

async function loadComparePayload() {
  if (!state.compareModelId) {
    state.comparePayload = null;
    renderCompareTable();
    return;
  }
  if (state.compareController) {
    state.compareController.abort();
  }
  state.compareController = new AbortController();
  const query = buildQueryFromControls();
  const suffix = query ? `?${query}` : "";
  try {
    const payload = await fetchJson(
      `/api/models/${encodeURIComponent(state.compareModelId)}${suffix}`,
      state.compareController.signal
    );
    state.comparePayload = payload;
    renderCompareTable();
  } catch (error) {
    if (error.name === "AbortError") return;
    console.error(error);
    state.comparePayload = null;
    ui.compareBody.innerHTML = `<div class="empty-state">对比模型加载失败：${escapeHtml(error.message)}</div>`;
  }
}

function renderSummary() {
  const payload = state.payload;
  if (!payload) {
    return;
  }

  const graph = getVisibleGraph(payload);
  // Clean spec-table layout: two-col rows (label | value).
  const metrics = payload.model.summary
    .map(
      (item) => `
        <div class="spec-row">
          <span class="spec-label">${escapeHtml(item.label)}</span>
          <span class="spec-value">${escapeHtml(item.value)}</span>
        </div>`
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
    <div class="spec-table">${metrics}</div>
    ${renderAnalysisSections(payload)}
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

function auditConfidenceLabel(level) {
  if (level === "high") return "高可信";
  if (level === "medium") return "中等可信";
  return "低可信";
}

function auditCheckpointLabel(status) {
  const labels = {
    matched: "高度吻合",
    close: "基本吻合",
    divergent: "偏差较大",
    informational: "仅供参考",
    unavailable: "无真值",
  };
  return labels[status] || status || "未知";
}

function auditSeverityLabel(severity) {
  if (severity === "error") return "错误";
  if (severity === "warning") return "警告";
  return "信息";
}

function formatAuditResult(value, unit) {
  if (value === undefined || value === null || value === "") return "—";
  const numeric = Number(value);
  if (unit === "parameters" && Number.isFinite(numeric)) return formatB(numeric);
  if (unit === "bytes" && Number.isFinite(numeric)) return formatGB(numeric);
  if (unit === "GFLOPs/token" && Number.isFinite(numeric)) return `${numeric.toFixed(2)} GFLOPs/token`;
  return String(value);
}

function renderAuditInputs(inputs) {
  if (!inputs?.length) return '<div class="audit-empty-inline">该证据不依赖统一配置字段。</div>';
  const originLabels = { direct: "直接", inherited: "继承", derived: "推导", runtime: "运行参数" };
  return `
    <div class="audit-input-table">
      <div class="audit-input-head"><span>字段</span><span>值</span><span>来源</span></div>
      ${inputs
        .map(
          (input) => `
            <div class="audit-input-row">
              <code>${escapeHtml(input.name)}</code>
              <code>${escapeHtml(input.value ?? "—")}</code>
              <span><span class="origin-badge origin-${escapeHtml(input.origin || "derived")}">${escapeHtml(originLabels[input.origin] || input.origin || "推导")}</span> ${escapeHtml(input.source || "derived")}</span>
            </div>`
        )
        .join("")}
    </div>`;
}

function renderAuditEvidence(evidence, index) {
  const result = formatAuditResult(evidence.result, evidence.unit);
  const assumptions = (evidence.assumptions || [])
    .filter(Boolean)
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("");
  const sources = (evidence.sourceFiles || [])
    .map((source) => `<span class="source-chip">${escapeHtml(source)}</span>`)
    .join("");
  return `
    <details class="audit-evidence"${index === 0 ? " open" : ""}>
      <summary>
        <span><span class="audit-category">${escapeHtml(evidence.category || "calculation")}</span>${escapeHtml(evidence.label)}</span>
        <strong>${escapeHtml(result)}</strong>
      </summary>
      <div class="audit-evidence-body">
        <div class="audit-formula"><span>公式</span><code>${escapeHtml(evidence.formula || "—")}</code></div>
        ${renderAuditInputs(evidence.inputs || [])}
        ${assumptions ? `<div class="audit-assumptions"><h4>假设与边界</h4><ul>${assumptions}</ul></div>` : ""}
        ${sources ? `<div class="source-list">${sources}</div>` : ""}
      </div>
    </details>`;
}

function renderAudit() {
  const payload = state.payload;
  const audit = payload?.audit;
  if (!audit) {
    ui.auditPanel.innerHTML = '<div class="empty-state">当前 payload 没有审计数据。</div>';
    return;
  }

  const confidence = audit.confidence || {};
  const checkpoint = audit.checkpoint || {};
  const diagnostics = audit.diagnostics || [];
  const config = audit.config || {};
  const evidence = audit.evidence || [];
  const diagnosticCounts = diagnostics.reduce(
    (counts, item) => ({ ...counts, [item.severity]: (counts[item.severity] || 0) + 1 }),
    { error: 0, warning: 0, info: 0 }
  );
  const confidenceReasons = (confidence.reasons || []).map((reason) => `<li>${escapeHtml(reason)}</li>`).join("");
  const checkpointError = checkpoint.relativeError == null ? "—" : `${(checkpoint.relativeError * 100).toFixed(3)}%`;
  const checkpointActual = checkpoint.bytes ? formatGB(checkpoint.bytes) : "—";
  const checkpointExpected = checkpoint.estimatedBytes ? formatGB(checkpoint.estimatedBytes) : "—";
  const checkpointParams = checkpoint.estimatedParameters ? formatB(checkpoint.estimatedParameters) : "—";
  const lineage = (config.lineage || [])
    .map(
      (item) => `<span class="lineage-chip"><strong>${escapeHtml(item.role)}</strong>${escapeHtml(item.modelId)}${item.configFile ? ` · ${escapeHtml(item.configFile)}` : ""}</span>`
    )
    .join("");
  const diagnosticMarkup = diagnostics.length
    ? diagnostics
        .map(
          (item) => `
            <article class="diagnostic-card diagnostic-${escapeHtml(item.severity)}">
              <div class="diagnostic-head"><span>${escapeHtml(auditSeverityLabel(item.severity))}</span><code>${escapeHtml(item.code)}</code></div>
              <h4>${escapeHtml(item.title)}</h4>
              <p>${escapeHtml(item.message)}</p>
              ${(item.fields || []).length ? `<div class="diagnostic-fields">${item.fields.map((field) => `<code>${escapeHtml(field)}</code>`).join("")}</div>` : ""}
            </article>`
        )
        .join("")
    : '<div class="audit-ok-state">未发现配置冲突或计算不变量异常。</div>';
  const evidenceMarkup = evidence.length
    ? evidence.map(renderAuditEvidence).join("")
    : '<div class="empty-state">当前模型没有可展示的计算证据。</div>';

  ui.auditPanel.innerHTML = `
    <div class="audit-title-row">
      <div>
        <div class="tag-row"><span class="tag">Audit Schema v${escapeHtml(audit.schemaVersion || 1)}</span><span class="tag">${escapeHtml(payload.model.type)}</span></div>
        <h2>计算审计 · ${escapeHtml(payload.model.name)}</h2>
        <p>追踪配置字段、公式、估算假设与 checkpoint 校验结果。</p>
      </div>
      <div class="audit-score audit-score-${escapeHtml(confidence.level || "low")}">
        <strong>${escapeHtml(confidence.score ?? 0)}</strong>
        <span>${escapeHtml(auditConfidenceLabel(confidence.level))}</span>
      </div>
    </div>
    <div class="audit-confidence-track"><span style="width:${Math.max(0, Math.min(100, confidence.score || 0))}%"></span></div>
    ${confidenceReasons ? `<ul class="audit-reasons">${confidenceReasons}</ul>` : ""}

    <div class="audit-overview-grid">
      <section class="audit-card">
        <div class="audit-card-head"><h3>Checkpoint 校验</h3><span class="checkpoint-status checkpoint-${escapeHtml(checkpoint.status || "unavailable")}">${escapeHtml(auditCheckpointLabel(checkpoint.status))}</span></div>
        <div class="audit-kv-grid">
          <span>实际体积</span><code>${escapeHtml(checkpointActual)}</code>
          <span>公式估算</span><code>${escapeHtml(checkpointExpected)}</code>
          <span>相对偏差</span><code>${escapeHtml(checkpointError)}</code>
          <span>真值参数代理</span><code>${escapeHtml(checkpointParams)}</code>
          <span>权重条目</span><code>${escapeHtml(checkpoint.weightMapEntries || 0)}</code>
          <span>精度</span><code>${escapeHtml(checkpoint.precision || "—")}</code>
        </div>
        <p class="audit-note">${escapeHtml(checkpoint.method || "—")}</p>
      </section>
      <section class="audit-card">
        <div class="audit-card-head"><h3>配置诊断</h3><span class="diagnostic-counts">${diagnosticCounts.error} 错误 · ${diagnosticCounts.warning} 警告 · ${diagnosticCounts.info} 信息</span></div>
        <div class="lineage-list">${lineage || '<span class="subtle">无配置继承链。</span>'}</div>
        <div class="audit-kv-grid compact">
          <span>架构目录</span><code>${escapeHtml(config.resolvedArchitectureDir || "—")}</code>
          <span>本地配置</span><code>${escapeHtml(config.directConfigFile || "—")}</code>
          <span>基础模型</span><code>${escapeHtml(config.baseReference || "—")}</code>
        </div>
      </section>
    </div>

    <section class="audit-section">
      <div class="audit-section-head"><h3>诊断项</h3><span>${diagnostics.length}</span></div>
      <div class="diagnostic-list">${diagnosticMarkup}</div>
    </section>
    <section class="audit-section">
      <div class="audit-section-head"><h3>计算证据链</h3><span>${evidence.length}</span></div>
      <div class="audit-evidence-list">${evidenceMarkup}</div>
    </section>
    <section class="audit-section">
      <div class="audit-section-head"><h3>核心字段来源</h3><span>${(config.fields || []).length}</span></div>
      ${renderAuditInputs(config.fields || [])}
      <div class="source-list">${(config.sourceFiles || []).map((source) => `<span class="source-chip">${escapeHtml(source)}</span>`).join("")}</div>
    </section>
  `;
}

function renderControls() {
  const payload = state.payload;
  if (!payload) {
    return;
  }

  const controls = payload.controls || [];
  if (!controls.length) {
    ui.controlsForm.innerHTML = '<div class="empty-state">当前模型无需额外运行参数。</div>';
    delete ui.controlsForm.dataset.structure;
    return;
  }

  const existingFields = Array.from(ui.controlsForm.querySelectorAll("[name]"));
  const existingNames = existingFields.map((el) => el.name);
  const newNames = controls.map((c) => c.name);
  const structureSignature = JSON.stringify(
    controls.map((control) => [
      control.name,
      control.type || "number",
      control.label || "",
      control.help || "",
      control.options || [],
    ])
  );
  const sameStructure =
    ui.controlsForm.dataset.structure === structureSignature &&
    existingNames.length === newNames.length &&
    existingNames.every((name, i) => name === newNames[i]);

  if (sameStructure) {
    controls.forEach((control) => {
      const input = ui.controlsForm.querySelector(`input[name="${control.name}"]`);
      if (input) {
        input.value = String(control.value ?? "");
        input.min = control.min ?? "";
        input.max = control.max ?? "";
        input.step = control.step ?? 1;
      }
      const select = ui.controlsForm.querySelector(`select[name="${control.name}"]`);
      if (select && control.value != null) {
        select.value = String(control.value);
      }
    });
    return;
  }

  ui.controlsForm.innerHTML = controls
    .map((control) => {
      const field =
        control.type === "select"
          ? `<select class="control-input" name="${escapeHtml(control.name)}">${(control.options || [])
              .map(
                (opt) =>
                  `<option value="${escapeHtml(opt)}"${String(opt) === String(control.value) ? " selected" : ""}>${escapeHtml(opt)}</option>`
              )
              .join("")}</select>`
          : `<input
            class="control-input"
            name="${escapeHtml(control.name)}"
            type="${escapeHtml(control.type || "number")}" 
            value="${escapeHtml(control.value ?? "")}"
            min="${escapeHtml(control.min ?? "")}" 
            max="${escapeHtml(control.max ?? "")}" 
            step="${escapeHtml(control.step ?? 1)}"
          />`;
      return `
        <label class="control-group">
          <span>${escapeHtml(control.label)}</span>
          ${field}
          <span class="control-meta">${escapeHtml(control.help || "")}</span>
        </label>
      `;
    })
    .join("");
  ui.controlsForm.dataset.structure = structureSignature;
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
        <section class="lane${state.collapsedLanes.has(lane.id) ? " collapsed" : ""}" data-lane-id="${escapeHtml(lane.id)}">
          <div class="lane-title" data-toggle-lane="${escapeHtml(lane.id)}">${escapeHtml(lane.label)} <span class="lane-toggle">${state.collapsedLanes.has(lane.id) ? "▸" : "▾"}</span></div>
          <div class="lane-body">${nodeMarkup || '<div class="empty-state">无节点</div>'}</div>
        </section>
      `;
    })
    .join("");

  requestAnimationFrame(() => {
    drawEdges();
    applyNodeSearch();
  });
}

function applyZoom() {
  const { scale, x, y } = state.zoom;
  ui.graphCanvas.style.transform = `scale(${scale}) translate(${x}px, ${y}px)`;
  ui.zoomLevel.textContent = `${Math.round(scale * 100)}%`;
  requestAnimationFrame(drawEdges);
}

function resetZoom() {
  state.zoom = { scale: 1, x: 0, y: 0 };
  applyZoom();
}

function zoomBy(delta, centerX, centerY) {
  const oldScale = state.zoom.scale;
  const newScale = Math.max(0.3, Math.min(3, oldScale * delta));
  if (newScale === oldScale) {
    return;
  }
  if (centerX !== undefined && centerY !== undefined) {
    const scrollRect = ui.graphScroll.getBoundingClientRect();
    const mx = centerX - scrollRect.left + ui.graphScroll.scrollLeft;
    const my = centerY - scrollRect.top + ui.graphScroll.scrollTop;
    state.zoom.x = mx - (mx - state.zoom.x) * (newScale / oldScale) - mx;
    state.zoom.y = my - (my - state.zoom.y) * (newScale / oldScale) - my;
  }
  state.zoom.scale = newScale;
  applyZoom();
}

function getNodeOffset(element, container) {
  let left = 0;
  let top = 0;
  let el = element;
  while (el && el !== container) {
    left += el.offsetLeft;
    top += el.offsetTop;
    el = el.offsetParent;
  }
  return {
    left,
    top,
    width: element.offsetWidth,
    height: element.offsetHeight,
    right: left + element.offsetWidth,
    bottom: top + element.offsetHeight,
  };
}

function isMeasurableElement(element) {
  return Boolean(element && element.offsetWidth > 0 && element.offsetHeight > 0);
}

function buildEdgeAnchorIndex(visibleGraph) {
  const nodeById = new Map((visibleGraph.nodes || []).map((node) => [node.id, node]));
  const nodeElements = new Map();
  const collapsedLaneElements = new Map();

  ui.graphBoard.querySelectorAll(".graph-node").forEach((element) => {
    nodeElements.set(element.dataset.nodeId, element);
  });
  ui.graphBoard.querySelectorAll(".lane.collapsed").forEach((laneElement) => {
    const laneTitle = laneElement.querySelector(".lane-title");
    if (isMeasurableElement(laneTitle)) {
      collapsedLaneElements.set(laneElement.dataset.laneId, laneTitle);
    }
  });

  function resolve(nodeId) {
    const nodeElement = nodeElements.get(nodeId);
    if (isMeasurableElement(nodeElement)) {
      return { element: nodeElement, key: `node:${nodeId}` };
    }

    const node = nodeById.get(nodeId);
    const laneElement = node ? collapsedLaneElements.get(node.lane) : null;
    if (isMeasurableElement(laneElement)) {
      return { element: laneElement, key: `lane:${node.lane}` };
    }
    return null;
  }

  return { nodeElements, collapsedLaneElements, resolve };
}

function buildDrawableEdges(visibleGraph, relations, anchors) {
  const relationPriority = { dimmed: 0, related: 1, upstream: 2, downstream: 2 };
  const grouped = new Map();

  (visibleGraph.edges || []).forEach((edge) => {
    const source = anchors.resolve(edge.source);
    const target = anchors.resolve(edge.target);
    if (!source || !target || source.key === target.key) {
      return;
    }

    const groupKey = `${source.key}\u0000${target.key}`;
    const relationClass = relations.edgeClass(edge);
    let group = grouped.get(groupKey);
    if (!group) {
      group = { source, target, labels: [], relationClass };
      grouped.set(groupKey, group);
    } else if ((relationPriority[relationClass] ?? 0) > (relationPriority[group.relationClass] ?? 0)) {
      group.relationClass = relationClass;
    }

    const label = String(edge.label || "");
    if (label && !group.labels.includes(label)) {
      group.labels.push(label);
    }
  });

  return Array.from(grouped.values()).map((group) => ({
    ...group,
    label: group.labels.length > 1 ? `${group.labels[0]} +${group.labels.length - 1}` : group.labels[0] || "",
  }));
}

function updateNodeSelection() {
  const payload = state.payload;
  if (!payload) {
    return;
  }

  const relations = buildGraphRelations(payload);
  ui.graphBoard.querySelectorAll(".graph-node").forEach((element) => {
    const nodeId = element.dataset.nodeId;
    element.classList.remove("selected", "upstream", "downstream", "dimmed", "active");
    const relationClass = relations.nodeClass(nodeId);
    if (relationClass) {
      element.classList.add(relationClass);
    }
    if (nodeId === state.selectedNodeId) {
      element.classList.add("active");
    }
  });
  requestAnimationFrame(drawEdges);
}

function drawEdges() {
  const payload = state.payload;
  if (!payload) {
    return;
  }

  const visibleGraph = getVisibleGraph(payload);
  const relations = buildGraphRelations(payload);
  const canvasRect = { left: 0, top: 0 };
  const anchors = buildEdgeAnchorIndex(visibleGraph);
  const drawableEdges = buildDrawableEdges(visibleGraph, relations, anchors);

  const width = Math.max(ui.graphBoard.scrollWidth + 24, ui.graphBoard.clientWidth);
  const height = Math.max(ui.graphBoard.scrollHeight + 24, ui.graphBoard.clientHeight);
  ui.edgeLayer.setAttribute("viewBox", `0 0 ${width} ${height}`);
  ui.edgeLayer.setAttribute("width", String(width));
  ui.edgeLayer.setAttribute("height", String(height));

  let markup = `
    <defs>
      <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto-start-reverse">
        <path d="M0,0 L12,6 L0,12 z" fill="#94a3b8"></path>
      </marker>
    </defs>
  `;

  drawableEdges.forEach((edge) => {
    const geometry = buildEdgeGeometry(
      getNodeOffset(edge.source.element, ui.graphCanvas),
      getNodeOffset(edge.target.element, ui.graphCanvas),
      canvasRect
    );
    const labelText = escapeHtml(edge.label);
    const labelWidth = Math.max(56, edge.label.length * 7.4);

    markup += `
      <path class="edge-path ${edge.relationClass}" data-source-anchor="${escapeHtml(edge.source.key)}" data-target-anchor="${escapeHtml(edge.target.key)}" d="${geometry.path}" marker-end="url(#arrow)"></path>
      <rect class="edge-label-bg ${edge.relationClass}" x="${geometry.labelX - labelWidth / 2}" y="${geometry.labelY - 12}" rx="10" ry="10" width="${labelWidth}" height="24"></rect>
      <text class="edge-label ${edge.relationClass}" x="${geometry.labelX}" y="${geometry.labelY + 4}" text-anchor="middle">${labelText}</text>
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
    return "#d97706";
  }
  if (accent === "scheduler" || accent === "latent") {
    return "#0ea5e9";
  }
  if (accent === "output" || accent === "head" || accent === "decode") {
    return "#e11d48";
  }
  return "#4f46e5";
}

function relationColor(kind) {
  if (kind === "upstream") {
    return "#d97706";
  }
  if (kind === "downstream") {
    return "#4f46e5";
  }
  if (kind === "related") {
    return "#94a3b8";
  }
  return "#cbd5e1";
}

function markdownCell(value) {
  return String(value ?? "—").replaceAll("|", "\\|").replace(/[\r\n]+/g, " ");
}

function buildAuditMarkdown(payload) {
  const audit = payload.audit || {};
  const confidence = audit.confidence || {};
  const checkpoint = audit.checkpoint || {};
  const config = audit.config || {};
  const diagnostics = audit.diagnostics || [];
  const evidence = audit.evidence || [];
  const lines = [
    `# 模型计算审计：${payload.model.name}`,
    "",
    `- 模型 ID：\`${payload.model.id}\``,
    `- 架构：\`${payload.model.architecture}\``,
    `- 类型：\`${payload.model.type}\``,
    `- 可信度：**${confidence.score ?? 0}/100 · ${auditConfidenceLabel(confidence.level)}**`,
    `- 报告生成：${new Date().toISOString()}`,
    "",
    "## 可信度依据",
    "",
    ...(confidence.reasons || []).map((reason) => `- ${reason}`),
    "",
    "## Checkpoint 校验",
    "",
    "| 项目 | 结果 |",
    "|---|---:|",
    `| 状态 | ${auditCheckpointLabel(checkpoint.status)} |`,
    `| 实际体积 | ${checkpoint.bytes ? formatGB(checkpoint.bytes) : "—"} |`,
    `| 公式估算 | ${checkpoint.estimatedBytes ? formatGB(checkpoint.estimatedBytes) : "—"} |`,
    `| 相对偏差 | ${checkpoint.relativeError == null ? "—" : `${(checkpoint.relativeError * 100).toFixed(3)}%`} |`,
    `| 精度 | ${checkpoint.precision || "—"} |`,
    `| 权重条目 | ${checkpoint.weightMapEntries || 0} |`,
    "",
    checkpoint.method || "未提供校验说明。",
    "",
    "## 配置诊断",
    "",
  ];
  if (diagnostics.length) {
    diagnostics.forEach((item) => {
      lines.push(`- **[${auditSeverityLabel(item.severity)}] ${item.title}**（\`${item.code}\`）：${item.message}`);
    });
  } else {
    lines.push("- 未发现配置冲突或计算不变量异常。");
  }
  lines.push("", "## 配置继承", "");
  (config.lineage || []).forEach((item) => {
    lines.push(`- ${item.role}: \`${item.modelId}\`${item.configFile ? ` · \`${item.configFile}\`` : ""}`);
  });
  lines.push("", "## 核心字段来源", "", "| 字段 | 值 | 来源 | 类型 |", "|---|---:|---|---|");
  (config.fields || []).forEach((input) => {
    lines.push(`| \`${markdownCell(input.name)}\` | ${markdownCell(input.value)} | \`${markdownCell(input.source)}\` | ${markdownCell(input.origin)} |`);
  });
  lines.push("", "## 计算证据链", "");
  evidence.forEach((item, index) => {
    lines.push(`### ${index + 1}. ${item.label}`, "");
    lines.push(`- 结果：**${formatAuditResult(item.result, item.unit)}**`);
    lines.push(`- 公式：\`${String(item.formula || "—").replaceAll("`", "'")}\``);
    if (item.inputs?.length) {
      lines.push("- 输入：");
      item.inputs.forEach((input) => {
        lines.push(`  - \`${input.name}\` = \`${markdownCell(input.value)}\`，来源 \`${markdownCell(input.source)}\`（${input.origin}）`);
      });
    }
    if (item.assumptions?.length) {
      lines.push("- 假设：");
      item.assumptions.forEach((assumption) => lines.push(`  - ${assumption}`));
    }
    lines.push("");
  });
  lines.push("## 当前运行参数", "", "```json", JSON.stringify(payload.parameters || {}, null, 2), "```", "");
  return `${lines.join("\n")}\n`;
}

function exportAuditReport() {
  if (!state.payload) return;
  const filename = `${sanitizeFileName(state.payload.model.id)}-audit.md`;
  downloadBlob(filename, buildAuditMarkdown(state.payload), "text/markdown;charset=utf-8");
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
  const canvasRect = { left: 0, top: 0 };
  const width = Math.max(ui.graphBoard.scrollWidth + 24, ui.graphBoard.clientWidth);
  const height = Math.max(ui.graphBoard.scrollHeight + 24, ui.graphBoard.clientHeight);
  const anchors = buildEdgeAnchorIndex(visibleGraph);
  const drawableEdges = buildDrawableEdges(visibleGraph, relations, anchors);
  const laneById = new Map((visibleGraph.lanes || []).map((lane) => [lane.id, lane]));

  const edgeMarkup = drawableEdges
    .map((edge) => {
      const geometry = buildEdgeGeometry(
        getNodeOffset(edge.source.element, ui.graphCanvas),
        getNodeOffset(edge.target.element, ui.graphCanvas),
        canvasRect
      );
      const label = truncateText(edge.label || "", 24);
      const labelWidth = Math.max(56, label.length * 7.2);
      const color = relationColor(edge.relationClass);
      return `
        <path d="${geometry.path}" fill="none" stroke="${color}" stroke-width="2.4" marker-end="url(#arrow)" />
        <rect x="${geometry.labelX - labelWidth / 2}" y="${geometry.labelY - 12}" width="${labelWidth}" height="24" rx="10" ry="10" fill="rgba(255,255,255,0.92)" stroke="rgba(15,23,42,0.08)" />
        <text x="${geometry.labelX}" y="${geometry.labelY + 4}" fill="#64748b" font-family="Aptos, Microsoft YaHei UI, sans-serif" font-size="12" text-anchor="middle">${escapeHtml(label)}</text>
      `;
    })
    .join("");

  const collapsedLaneMarkup = Array.from(anchors.collapsedLaneElements.entries())
    .map(([laneId, element]) => {
      const lane = laneById.get(laneId);
      const rect = getNodeOffset(element, ui.graphCanvas);
      return `
        <g>
          <rect x="${rect.left}" y="${rect.top}" width="${rect.width}" height="${rect.height}" rx="8" ry="8" fill="rgba(255,255,255,0.98)" stroke="rgba(15,23,42,0.12)" />
          <text x="${rect.left + rect.width / 2}" y="${rect.top + rect.height / 2 + 4}" fill="#0f172a" font-family="Aptos, Microsoft YaHei UI, sans-serif" font-size="12" font-weight="600" text-anchor="middle">${escapeHtml(lane?.label || laneId)}</text>
        </g>
      `;
    })
    .join("");

  const nodeMarkup = (visibleGraph.nodes || [])
    .map((node) => {
      const element = anchors.nodeElements.get(node.id);
      if (!isMeasurableElement(element)) {
        return "";
      }

      const rect = getNodeOffset(element, ui.graphCanvas);
      const x = rect.left;
      const y = rect.top;
      const widthPx = rect.width;
      const heightPx = rect.height;
      const accent = accentColor(node.accent || "core");
      const relation = relations.nodeClass(node.id);
      const opacity = relation === "dimmed" ? 0.34 : 1;
      const stroke = node.id === relations.selectedId ? "#4f46e5" : "rgba(15,23,42,0.08)";
      const title = truncateText(node.label, 26);
      const subtitle = truncateText(node.subtitle || "", 38);
      const input = truncateText(`IN ${node.inputShape || "-"}`, 38);
      const output = truncateText(`OUT ${node.outputShape || "-"}`, 38);
      return `
        <g opacity="${opacity}">
          <rect x="${x}" y="${y}" width="${widthPx}" height="${heightPx}" rx="12" ry="12" fill="rgba(255,255,255,0.98)" stroke="${stroke}" />
          <rect x="${x}" y="${y}" width="6" height="${heightPx}" rx="6" ry="6" fill="${accent}" />
          <text x="${x + 18}" y="${y + 28}" fill="#0f172a" font-family="Bahnschrift, Aptos Display, Microsoft YaHei UI, sans-serif" font-size="16">${escapeHtml(title)}</text>
          <text x="${x + 18}" y="${y + 48}" fill="#64748b" font-family="Aptos, Microsoft YaHei UI, sans-serif" font-size="12">${escapeHtml(subtitle)}</text>
          <rect x="${x + 16}" y="${y + heightPx - 54}" width="${Math.max(120, widthPx - 32)}" height="38" rx="8" ry="8" fill="rgba(241,245,249,0.9)" stroke="rgba(226,232,240,0.6)" />
          <text x="${x + 28}" y="${y + heightPx - 31}" fill="#64748b" font-family="Cascadia Code, Consolas, monospace" font-size="11">${escapeHtml(input)}</text>
          <text x="${x + 28}" y="${y + heightPx - 17}" fill="#64748b" font-family="Cascadia Code, Consolas, monospace" font-size="11">${escapeHtml(output)}</text>
        </g>
      `;
    })
    .join("");

  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f8fafc" />
      <stop offset="100%" stop-color="#f1f5f9" />
    </linearGradient>
    <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto-start-reverse">
      <path d="M0,0 L12,6 L0,12 z" fill="#94a3b8" />
    </marker>
  </defs>
  <rect width="100%" height="100%" fill="url(#bg)" />
  ${edgeMarkup}
  ${collapsedLaneMarkup}
  ${nodeMarkup}
</svg>`;

  downloadBlob(`${sanitizeFileName(payload.model.id)}.svg`, svg, "image/svg+xml;charset=utf-8");
}

function setCenterView(view) {
  state.centerView = view;
  ui.viewTabs.querySelectorAll(".view-tab").forEach((tab) => {
    const active = tab.dataset.view === view;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  });
  const graphOnly = view === "graph";
  ui.graphView.hidden = !graphOnly;
  ui.detailsView.hidden = view !== "details";
  ui.auditView.hidden = view !== "audit";
  ui.compareView.hidden = view !== "compare";
  ui.nodeSearch.hidden = !graphOnly;
  if (graphOnly) {
    renderHierarchyToolbar();
    if (state.payload) renderGraph();
  } else {
    ui.hierarchyToolbar.hidden = true;
  }
}

ui.viewTabs.addEventListener("click", (event) => {
  const tab = event.target.closest(".view-tab");
  if (!tab) return;
  setCenterView(tab.dataset.view);
});

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
ui.controlsForm.addEventListener("change", scheduleRefresh);
ui.refreshButton.addEventListener("click", () => loadModelPayload());

ui.compareSelect.addEventListener("change", () => {
  state.compareModelId = ui.compareSelect.value || null;
  ui.compareClear.hidden = !state.compareModelId;
  loadComparePayload();
});
ui.compareClear.addEventListener("click", () => {
  state.compareModelId = null;
  state.comparePayload = null;
  ui.compareClear.hidden = true;
  if (ui.compareSelect) ui.compareSelect.value = "";
  renderCompareTable();
});
ui.hierarchyToolbar.addEventListener("click", (event) => {
  const button = event.target.closest("[data-hierarchy-mode]");
  if (!button || !state.payload || !HIERARCHY_TYPES.has(state.payload.model.type)) {
    return;
  }
  if (!getAvailableHierarchyModes().has(button.dataset.hierarchyMode)) {
    return;
  }
  clearTimeout(state.refreshTimer);
  state.llmHierarchyMode = button.dataset.hierarchyMode;
  syncSelectedNode(state.payload);
  renderHierarchyToolbar();
  renderSummary();
  renderGraph();
  renderDetails();
  setStatus(`已切换为 ${getHierarchyModeLabel(state.llmHierarchyMode)} 层级视图。`);
});
ui.exportAuditButton.addEventListener("click", exportAuditReport);
ui.exportJsonButton.addEventListener("click", exportCurrentPayload);
ui.exportSvgButton.addEventListener("click", exportCurrentSvg);

ui.graphBoard.addEventListener("click", (event) => {
  const laneToggle = event.target.closest("[data-toggle-lane]");
  if (laneToggle) {
    const laneId = laneToggle.dataset.toggleLane;
    if (state.collapsedLanes.has(laneId)) {
      state.collapsedLanes.delete(laneId);
    } else {
      state.collapsedLanes.add(laneId);
    }
    renderGraph();
    return;
  }

  const nodeElement = event.target.closest("[data-node-id]");
  if (!nodeElement) {
    return;
  }
  state.selectedNodeId = nodeElement.dataset.nodeId;
  updateNodeSelection();
  renderDetails();
});

ui.detailPanel.addEventListener("click", (event) => {
  const jumpButton = event.target.closest("[data-jump-node]");
  if (!jumpButton) {
    return;
  }
  state.selectedNodeId = jumpButton.dataset.jumpNode;
  updateNodeSelection();
  renderDetails();
  const targetElement = ui.graphBoard.querySelector(`[data-node-id="${state.selectedNodeId}"]`);
  if (targetElement) {
    targetElement.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
  }
});

let resizeTimer = null;
window.addEventListener("resize", () => {
  if (!state.payload) {
    return;
  }
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    requestAnimationFrame(drawEdges);
  }, 150);
});

// --- Node Search ---

let nodeSearchTimer = null;
ui.nodeSearch.addEventListener("input", () => {
  clearTimeout(nodeSearchTimer);
  nodeSearchTimer = setTimeout(applyNodeSearch, 120);
});

function applyNodeSearch() {
  const keyword = ui.nodeSearch.value.trim().toLowerCase();
  ui.graphBoard.querySelectorAll(".graph-node").forEach((element) => {
    element.classList.remove("matched", "search-dimmed");
    if (!keyword) {
      return;
    }
    const nodeId = element.dataset.nodeId || "";
    const title = element.querySelector(".node-title")?.textContent?.toLowerCase() || "";
    const subtitle = element.querySelector(".node-subtitle")?.textContent?.toLowerCase() || "";
    const matched = title.includes(keyword) || subtitle.includes(keyword) || nodeId.toLowerCase().includes(keyword);
    if (matched) {
      element.classList.add("matched");
    } else {
      element.classList.add("search-dimmed");
    }
  });
}

// --- Zoom & Pan ---

ui.graphScroll.addEventListener("wheel", (event) => {
  if (event.ctrlKey || event.metaKey) {
    event.preventDefault();
    zoomBy(event.deltaY < 0 ? 1.1 : 0.9, event.clientX, event.clientY);
  }
}, { passive: false });

ui.graphScroll.addEventListener("mousedown", (event) => {
  if (event.button !== 0) {
    return;
  }
  const isNode = event.target.closest("[data-node-id]");
  if (isNode) {
    return;
  }
  state.pan.dragging = true;
  state.pan.startX = event.clientX;
  state.pan.startY = event.clientY;
  state.pan.scrollLeft = ui.graphScroll.scrollLeft;
  state.pan.scrollTop = ui.graphScroll.scrollTop;
  ui.graphScroll.classList.add("is-panning");
});

window.addEventListener("mousemove", (event) => {
  if (!state.pan.dragging) {
    return;
  }
  const dx = event.clientX - state.pan.startX;
  const dy = event.clientY - state.pan.startY;
  ui.graphScroll.scrollLeft = state.pan.scrollLeft - dx;
  ui.graphScroll.scrollTop = state.pan.scrollTop - dy;
});

window.addEventListener("mouseup", () => {
  if (state.pan.dragging) {
    state.pan.dragging = false;
    ui.graphScroll.classList.remove("is-panning");
  }
});

ui.zoomIn.addEventListener("click", () => zoomBy(1.2));
ui.zoomOut.addEventListener("click", () => zoomBy(0.83));
ui.zoomReset.addEventListener("click", resetZoom);

updateExportButtons();
renderHierarchyToolbar();
loadModels().catch((error) => {
  console.error(error);
  setStatus(`初始化失败：${error.message}`, true);
});
