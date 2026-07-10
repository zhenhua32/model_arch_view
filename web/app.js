const state = {
  models: [],
  activeModelId: null,
  payload: null,
  selectedNodeId: null,
  refreshTimer: null,
};

const ui = {
  modelCount: document.getElementById("model-count"),
  modelSearch: document.getElementById("model-search"),
  modelList: document.getElementById("model-list"),
  controlsForm: document.getElementById("controls-form"),
  refreshButton: document.getElementById("refresh-button"),
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

async function loadModels() {
  setStatus("正在读取 model_configs 目录...");
  const data = await fetchJson("/api/models");
  state.models = data.models || [];
  renderModelList();

  if (!state.models.length) {
    setStatus("未找到模型目录。", true);
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
    const payload = await fetchJson(`/api/models/${encodeURIComponent(state.activeModelId)}${suffix}`);
    state.payload = payload;
    state.selectedNodeId = payload.selectedNodeId;
    renderModelList();
    renderSummary();
    renderControls();
    renderWarnings();
    renderGraph();
    renderDetails();
    setStatus(`已加载 ${payload.model.name}，当前展示 ${payload.model.type} 图结构。`);
  } catch (error) {
    console.error(error);
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

  ui.summaryPanel.innerHTML = `
    <div class="tag-row">
      <span class="tag">${escapeHtml(payload.model.type)}</span>
      <span class="tag">${escapeHtml(payload.model.architecture)}</span>
    </div>
    <h2>${escapeHtml(payload.model.name)}</h2>
    <p class="headline">${escapeHtml(payload.model.headline)}</p>
    <div class="summary-grid">${metrics}</div>
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

function renderGraph() {
  const payload = state.payload;
  if (!payload) {
    return;
  }

  const grouped = groupNodesByLane(payload.graph.nodes || [], payload.graph.lanes || []);
  ui.graphBoard.innerHTML = (payload.graph.lanes || [])
    .map((lane) => {
      const nodes = grouped.get(lane.id) || [];
      const nodeMarkup = nodes
        .map(
          (node) => `
            <article class="graph-node ${node.id === state.selectedNodeId ? "active" : ""}" data-node-id="${escapeHtml(node.id)}" data-accent="${escapeHtml(node.accent || "core")}">
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
            </article>
          `
        )
        .join("");

      return `
        <section class="lane">
          <div class="lane-title">${escapeHtml(lane.label)}</div>
          ${nodeMarkup || '<div class="empty-state">无节点</div>'}
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

  (payload.graph.edges || []).forEach((edge) => {
    const source = nodes.get(edge.source);
    const target = nodes.get(edge.target);
    if (!source || !target) {
      return;
    }

    const sourceRect = source.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();

    const startX = sourceRect.right - canvasRect.left;
    const startY = sourceRect.top - canvasRect.top + sourceRect.height / 2;
    const endX = targetRect.left - canvasRect.left;
    const endY = targetRect.top - canvasRect.top + targetRect.height / 2;
    const curve = Math.max(42, Math.abs(endX - startX) * 0.38);
    const path = `M ${startX} ${startY} C ${startX + curve} ${startY}, ${endX - curve} ${endY}, ${endX} ${endY}`;
    const labelX = (startX + endX) / 2;
    const labelY = (startY + endY) / 2 - 8;
    const labelText = escapeHtml(edge.label || "");
    const labelWidth = Math.max(48, labelText.length * 7.2);

    markup += `
      <path class="edge-path" d="${path}" marker-end="url(#arrow)"></path>
      <rect class="edge-label-bg" x="${labelX - labelWidth / 2}" y="${labelY - 12}" rx="10" ry="10" width="${labelWidth}" height="24"></rect>
      <text class="edge-label" x="${labelX}" y="${labelY + 4}" text-anchor="middle">${labelText}</text>
    `;
  });

  ui.edgeLayer.innerHTML = markup;
}

function findSelectedNode() {
  return (state.payload?.graph?.nodes || []).find((node) => node.id === state.selectedNodeId) || state.payload?.graph?.nodes?.[0] || null;
}

function renderDetails() {
  const payload = state.payload;
  const node = findSelectedNode();
  if (!payload || !node) {
    ui.detailPanel.innerHTML = '<div class="empty-state">选择节点后在这里查看细节。</div>';
    return;
  }

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
    .map((block) => `
      <section class="detail-section">
        <h3>${escapeHtml(block.title)}</h3>
        ${detailRows(block.items || [])}
      </section>
    `)
    .join("");

  ui.detailPanel.innerHTML = `
    <div class="tag-row">
      <span class="tag">${escapeHtml(payload.model.type)}</span>
      <span class="tag">${escapeHtml(node.accent || "node")}</span>
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
      <h3>关键字段</h3>
      ${detailRows(node.details || [])}
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

ui.graphBoard.addEventListener("click", (event) => {
  const nodeElement = event.target.closest("[data-node-id]");
  if (!nodeElement) {
    return;
  }
  state.selectedNodeId = nodeElement.dataset.nodeId;
  renderGraph();
  renderDetails();
});

window.addEventListener("resize", () => {
  if (state.payload) {
    requestAnimationFrame(drawEdges);
  }
});

loadModels().catch((error) => {
  console.error(error);
  setStatus(`初始化失败：${error.message}`, true);
});