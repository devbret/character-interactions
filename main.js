const width = 2000;
const height = 1000;
const K_MAX = 6;

let svgRoot, container, zoom;
let linksLayer, nodesLayer, labelsLayer;
let nodesSel = null,
  linksSel = null,
  labelsSel = null,
  simulation = null;
let allNodes = [],
  allLinks = [];
let aliasMap = new Map();
let degreeMap, rScale, colorScale, curver;
let maxWeight = 1;
let visibleNeighbors = new Map();

let inputEl, filterToggleEl;
let nodeLimitInputEl, applyNodeLimitBtn;
let darkModeToggleEl;
let currentNodeLimit = 15;
let tip;

const Facet = { seed: "", k: 0 };
let facetNodeKeep = new Set();

let khopMenuEl, kInputEl, applyKHopBtn, clearKHopBtn, closeKHopBtn;
let egoViewBtn, egoOverlayEl, egoCloseBtn, egoDownloadBtn;
let khopSeedId = null;
let lastEgo = null;

const debounce = (fn, ms = 200) => {
  let t;
  return (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
};

const idOf = (endpoint) =>
  typeof endpoint === "object" ? endpoint.id : endpoint;
const linkKey = (l) => `${idOf(l.source)}→${idOf(l.target)}`;
const cssVar = (name) =>
  getComputedStyle(document.body).getPropertyValue(name).trim();

fetch("character_interactions.json")
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then(init)
  .catch((error) => {
    console.error("Error loading JSON data:", error);
    showLoadError();
  });

function showLoadError() {
  const el = document.createElement("div");
  el.id = "loadError";
  el.innerHTML =
    "<b>Could not load <code>character_interactions.json</code>.</b>" +
    "<p>Generate it first by placing <code>.txt</code> files in the <code>input</code> directory and running <code>python3 app.py</code>, then reload this page.</p>" +
    "<p>Make sure you are viewing this page through a local server (<code>python3 -m http.server</code>), not as a <code>file://</code> URL.</p>";
  document.body.appendChild(el);
}

function buildLinks(data) {
  if (data.edges) {
    return data.edges.map((e) => ({
      source: e.source,
      target: e.target,
      value: e.weight,
      confidence: e.confidence,
      evidence: e.evidence || {},
    }));
  }
  const links = [];
  (data.matrix || []).forEach((row, i) => {
    row.forEach((value, j) => {
      if (j > i && value > 0) {
        links.push({
          source: data.characters[i],
          target: data.characters[j],
          value,
          evidence: {},
        });
      }
    });
  });
  return links;
}

function init(data) {
  const baseNodes = data.nodes || (data.characters || []).map((id) => ({ id }));
  allNodes = baseNodes.map((n) => ({ ...n }));
  allLinks = buildLinks(data);
  aliasMap = new Map(Object.entries(data.aliases || {}));

  maxWeight = d3.max(allLinks, (d) => d.value) || 1;

  degreeMap = new Map(allNodes.map((n) => [n.id, 0]));
  allLinks.forEach((l) => {
    degreeMap.set(idOf(l.source), (degreeMap.get(idOf(l.source)) || 0) + 1);
    degreeMap.set(idOf(l.target), (degreeMap.get(idOf(l.target)) || 0) + 1);
  });

  const degExtent = d3.extent(allNodes, (d) => degreeMap.get(d.id));
  rScale = d3
    .scaleSqrt()
    .domain([degExtent[0] || 1, degExtent[1] || 1])
    .range([6, 22]);
  colorScale = d3
    .scaleSequential(d3.interpolateTurbo)
    .domain([degExtent[0] || 0, degExtent[1] || 1]);
  curver = d3
    .scaleLinear()
    .domain(d3.extent(allNodes, (d) => d.id.length))
    .range([0.45, 0.85]);

  setupGraph();
  setupUI();
  initDarkMode();
  applyFacets();
}

function setupGraph() {
  svgRoot = d3
    .select("body")
    .append("svg")
    .attr("width", "100%")
    .attr("height", "100%")
    .attr("viewBox", [0, 0, width, height])
    .attr("preserveAspectRatio", "xMidYMid meet");

  container = svgRoot.append("g");

  zoom = d3
    .zoom()
    .on("zoom", (event) => container.attr("transform", event.transform));
  svgRoot.call(zoom);

  svgRoot
    .append("defs")
    .append("marker")
    .attr("id", "arrow")
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 18)
    .attr("refY", 0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-5L10,0L0,5")
    .attr("fill", "#9aa0a6")
    .attr("opacity", 0.6);

  linksLayer = container.append("g").attr("class", "links");
  nodesLayer = container.append("g").attr("class", "nodes");
  labelsLayer = container.append("g").attr("class", "labels");

  tip = d3
    .select("body")
    .append("div")
    .attr("id", "tip")
    .style("position", "fixed")
    .style("pointer-events", "none")
    .style("opacity", 0)
    .style("padding", "6px 8px")
    .style("border-radius", "6px")
    .style("box-shadow", "0 2px 8px rgba(0,0,0,.08)")
    .style("font", "12px system-ui");

  window.addEventListener("resize", () => fitToScreen(60));
}

function setupUI() {
  inputEl = document.getElementById("searchInput");
  filterToggleEl = document.getElementById("filterToggle");
  nodeLimitInputEl = document.getElementById("nodeLimitInput");
  applyNodeLimitBtn = document.getElementById("applyNodeLimitBtn");
  darkModeToggleEl = document.getElementById("darkModeToggle");

  const clearBtn = document.getElementById("clearBtn");
  const fitBtn = document.getElementById("fitBtn");

  egoViewBtn = document.getElementById("egoViewBtn");
  egoOverlayEl = document.getElementById("egoOverlay");
  egoCloseBtn = document.getElementById("egoCloseBtn");
  egoDownloadBtn = document.getElementById("egoDownloadBtn");

  khopMenuEl = document.getElementById("khopMenu");
  kInputEl = document.getElementById("kHop");
  applyKHopBtn = document.getElementById("applyKHop");
  clearKHopBtn = document.getElementById("clearKHop");
  closeKHopBtn = document.getElementById("closeKHop");

  readNodeLimit();

  applyNodeLimitBtn.addEventListener("click", () => {
    readNodeLimit();
    applyFacets();
  });

  nodeLimitInputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      readNodeLimit();
      applyFacets();
    }
  });

  darkModeToggleEl.addEventListener("change", () => {
    setDarkMode(darkModeToggleEl.checked);
  });

  egoViewBtn.addEventListener("click", () => {
    if (!khopSeedId) return;
    const kVal = Math.max(1, Math.min(K_MAX, +kInputEl.value || 1));
    kInputEl.value = String(kVal);
    showEgoOverlay(khopSeedId, kVal);
  });

  egoCloseBtn.addEventListener("click", hideEgoOverlay);
  egoDownloadBtn.addEventListener("click", () => downloadEgoSVG());

  const handleSearch = debounce(() => reapplySearchFromUI(), 120);

  inputEl.addEventListener("input", handleSearch);
  filterToggleEl.addEventListener("change", () => handleSearch());

  clearBtn.addEventListener("click", () => {
    inputEl.value = "";
    filterToggleEl.checked = false;
    clearSearch();
  });

  fitBtn.addEventListener("click", () => fitToScreen(60));

  applyKHopBtn.addEventListener("click", () => {
    Facet.k = Math.max(0, Math.min(K_MAX, +kInputEl.value || 0));
    if (khopSeedId) Facet.seed = khopSeedId;
    applyFacets();
  });

  clearKHopBtn.addEventListener("click", () => {
    Facet.k = 0;
    Facet.seed = "";
    kInputEl.value = "0";
    applyFacets();
  });

  closeKHopBtn.addEventListener("click", hideKHopMenu);

  document.addEventListener("click", (e) => {
    if (!khopMenuEl.style.display || khopMenuEl.style.display === "none") {
      return;
    }
    const inside = khopMenuEl.contains(e.target);
    const clickedNode = e.target && e.target.tagName === "circle";
    if (!inside && !clickedNode) hideKHopMenu();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (egoOverlayEl.style.display === "block") {
      hideEgoOverlay();
    } else {
      hideKHopMenu();
    }
  });
}

function readNodeLimit() {
  currentNodeLimit = Math.max(1, Math.round(+nodeLimitInputEl.value) || 15);
  nodeLimitInputEl.value = String(currentNodeLimit);
}

function initDarkMode() {
  const saved = localStorage.getItem("characterGraphDarkMode");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const enabled = saved === null ? prefersDark : saved === "true";
  darkModeToggleEl.checked = enabled;
  setDarkMode(enabled, false);
}

function setDarkMode(enabled, persist = true) {
  document.body.classList.toggle("dark-mode", enabled);
  if (persist) {
    localStorage.setItem("characterGraphDarkMode", String(enabled));
  }
  updateGraphTheme();
}

function updateGraphTheme() {
  if (svgRoot) svgRoot.style("background", cssVar("--bg"));
  if (linksSel) linksSel.attr("stroke", cssVar("--link-stroke"));
  if (nodesSel) nodesSel.attr("stroke", cssVar("--node-stroke"));
  if (labelsSel) {
    labelsSel
      .attr("fill", cssVar("--label-fill"))
      .attr("stroke", cssVar("--label-stroke"));
  }
  if (tip) {
    tip
      .style("background", cssVar("--tooltip-bg"))
      .style("border", `1px solid ${cssVar("--tooltip-border")}`)
      .style("color", cssVar("--tooltip-text"));
  }
  if (lastEgo && egoOverlayEl && egoOverlayEl.style.display === "block") {
    renderEgoRadial(lastEgo.seedId, lastEgo.k);
  }
}

function renderGraph(nodes, links) {
  if (simulation) simulation.stop();

  visibleNeighbors = new Map(nodes.map((n) => [n.id, new Set()]));
  links.forEach((l) => {
    visibleNeighbors.get(idOf(l.source))?.add(idOf(l.target));
    visibleNeighbors.get(idOf(l.target))?.add(idOf(l.source));
  });

  linksSel = linksLayer.selectAll("path").data(links, linkKey);
  linksSel.exit().remove();
  linksSel = linksSel
    .enter()
    .append("path")
    .attr("class", "link")
    .attr("fill", "none")
    .attr("stroke-opacity", 0.35)
    .attr("marker-end", "url(#arrow)")
    .on("mouseover", onLinkMouseOver)
    .on("mousemove", moveTip)
    .on("mouseleave", onLinkMouseLeave)
    .merge(linksSel)
    .attr("stroke", cssVar("--link-stroke"))
    .attr("stroke-width", (d) => 1 + 2 * (d.value / maxWeight));

  nodesSel = nodesLayer.selectAll("circle").data(nodes, (d) => d.id);
  nodesSel.exit().remove();
  nodesSel = nodesSel
    .enter()
    .append("circle")
    .attr("class", "node")
    .call(drag())
    .on("mouseover", onNodeMouseOver)
    .on("mousemove", moveTip)
    .on("mouseleave", onNodeMouseLeave)
    .on("click", onNodeClick)
    .merge(nodesSel)
    .attr("r", (d) => rScale(degreeMap.get(d.id)))
    .attr("fill", (d) => colorScale(degreeMap.get(d.id)))
    .attr("stroke", cssVar("--node-stroke"))
    .attr("stroke-width", 1.2);

  labelsSel = labelsLayer.selectAll("text").data(nodes, (d) => d.id);
  labelsSel.exit().remove();
  labelsSel = labelsSel
    .enter()
    .append("text")
    .attr("class", "label")
    .attr("font-size", 10)
    .attr("stroke-width", 3)
    .attr("paint-order", "stroke")
    .text((d) => d.id)
    .merge(labelsSel)
    .attr("fill", cssVar("--label-fill"))
    .attr("stroke", cssVar("--label-stroke"));

  simulation = d3
    .forceSimulation(nodes)
    .force(
      "link",
      d3
        .forceLink(links)
        .id((d) => d.id)
        .distance((d) => 120 + 240 * (1 - d.value / maxWeight) * 1.33)
        .strength((d) => 0.2 + 0.6 * (d.value / maxWeight)),
    )
    .force(
      "charge",
      d3.forceManyBody().strength(-200).theta(0.9).distanceMax(930),
    )
    .force(
      "collide",
      d3
        .forceCollide()
        .radius((d) => rScale(degreeMap.get(d.id)) + 9)
        .strength(0.9),
    )
    .force("center", d3.forceCenter(width / 2, height / 2))
    .on("tick", ticked);

  simulation.on("end.fit", () => {
    fitToScreen(60);
    simulation.on("end.fit", null);
  });
}

function arcPath(d) {
  const x1 = d.source.x,
    y1 = d.source.y,
    x2 = d.target.x,
    y2 = d.target.y;
  const dx = x2 - x1,
    dy = y2 - y1;
  const dr = Math.hypot(dx, dy) * curver(d.source.id.length);
  return `M${x1},${y1}A${dr},${dr} 0 0,1 ${x2},${y2}`;
}

function ticked() {
  linksSel.attr("d", arcPath);
  nodesSel.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
  labelsSel.attr("x", (d) => d.x + 6).attr("y", (d) => d.y + 3);
}

function drag() {
  return d3
    .drag()
    .on("start", (event, d) => {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    })
    .on("drag", (event, d) => {
      d.fx = event.x;
      d.fy = event.y;
    })
    .on("end", (event, d) => {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    });
}

function nodeTooltipHTML(d) {
  const deg = degreeMap.get(d.id) || 0;
  const aliases = aliasMap.get(d.id) || [];
  let html = `<b>${d.id}</b>`;
  if (aliases.length) html += `<br><i>aka ${aliases.join(", ")}</i>`;
  if (d.mentions != null) {
    html += `<br>Mentions: ${d.mentions}`;
    if (d.coref_mentions) html += ` (${d.named_mentions} named)`;
    if (d.scenes != null) html += `<br>Scenes: ${d.scenes}`;
    if (d.weighted_degree != null)
      html += `<br>Weighted degree: ${d.weighted_degree}`;
  }
  html += `<br>Connections: ${deg}`;
  return html;
}

function linkTooltipHTML(d) {
  let html = `<b>${idOf(d.source)} - ${idOf(d.target)}</b><br>Weight: ${d.value}`;
  if (d.confidence != null) html += `<br>Confidence: ${d.confidence}`;
  const parts = Object.entries(d.evidence || {})
    .sort((a, b) => b[1] - a[1])
    .map(([key, count]) => `${key.replace(/_/g, " ")}: ${count}`);
  if (parts.length) html += `<br><i>${parts.join("<br>")}</i>`;
  return html;
}

function moveTip(event) {
  tip
    .style("left", event.clientX + 12 + "px")
    .style("top", event.clientY + 12 + "px");
}

function onNodeMouseOver(event, d) {
  const isNeighbor = (n) =>
    n.id === d.id || visibleNeighbors.get(d.id)?.has(n.id);
  nodesSel.classed("dim", (n) => !isNeighbor(n));
  labelsSel.classed("dim", (n) => !isNeighbor(n));
  linksSel.classed(
    "dim",
    (l) => idOf(l.source) !== d.id && idOf(l.target) !== d.id,
  );
  labelsSel.filter((n) => n.id === d.id).raise();
  tip.style("opacity", 1).html(nodeTooltipHTML(d));
}

function onNodeMouseLeave() {
  tip.style("opacity", 0);
  nodesSel.classed("dim", false);
  labelsSel.classed("dim", false);
  linksSel.classed("dim", false);
  reapplySearchFromUI();
}

function onNodeClick(event, d) {
  if (event.defaultPrevented) return;
  khopSeedId = d.id;
  showKHopMenu(event.clientX, event.clientY, d.id);
}

function onLinkMouseOver(event, d) {
  d3.select(event.currentTarget).classed("hover", true);
  tip.style("opacity", 1).html(linkTooltipHTML(d));
}

function onLinkMouseLeave(event) {
  d3.select(event.currentTarget).classed("hover", false);
  tip.style("opacity", 0);
}

function fitToScreen(pad = 40) {
  if (!container) return;
  const bbox = container.node().getBBox();
  if (!bbox.width || !bbox.height) return;

  const cx = bbox.x + bbox.width / 2;
  const cy = bbox.y + bbox.height / 2;
  const k = Math.min(
    width / (bbox.width + pad * 2),
    height / (bbox.height + pad * 2),
    1,
  );

  const transform = d3.zoomIdentity
    .translate(width / 2, height / 2)
    .scale(k)
    .translate(-cx, -cy);

  svgRoot.transition().duration(300).call(zoom.transform, transform);
}

function buildAdjacency() {
  const adj = new Map(allNodes.map((n) => [n.id, new Set()]));
  for (const l of allLinks) {
    adj.get(idOf(l.source))?.add(idOf(l.target));
    adj.get(idOf(l.target))?.add(idOf(l.source));
  }
  return adj;
}

function getTopNodeIdsByDegree(limit) {
  return new Set(
    [...allNodes]
      .sort((a, b) => {
        const degDiff = (degreeMap.get(b.id) || 0) - (degreeMap.get(a.id) || 0);
        if (degDiff !== 0) return degDiff;
        return a.id.localeCompare(b.id);
      })
      .slice(0, Math.max(1, limit))
      .map((n) => n.id),
  );
}

function kHopSet(seedId, k, adj) {
  if (!seedId || k <= 0) return null;
  const seen = new Set([seedId]);
  let frontier = [seedId];

  for (let step = 0; step < k; step++) {
    const next = [];
    for (const v of frontier) {
      for (const u of adj.get(v) || []) {
        if (!seen.has(u)) {
          seen.add(u);
          next.push(u);
        }
      }
    }
    if (!next.length) break;
    frontier = next;
  }

  return seen;
}

function kHopBFS(seedId, k, adj) {
  const dist = new Map();
  dist.set(seedId, 0);
  let frontier = [seedId];

  for (let step = 1; step <= k; step++) {
    const next = [];
    for (const v of frontier) {
      for (const u of adj.get(v) || []) {
        if (!dist.has(u)) {
          dist.set(u, step);
          next.push(u);
        }
      }
    }
    if (!next.length) break;
    frontier = next;
  }

  return dist;
}

function showKHopMenu(clientX, clientY, nodeId) {
  if (!khopMenuEl) return;

  document.getElementById("khopTitle").textContent = `k-hop from “${nodeId}”`;
  if (Facet.seed === nodeId) kInputEl.value = String(Facet.k || 0);

  const pad = 8;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const rect = { w: 260, h: 140 };

  let left = clientX + 12;
  let top = clientY + 12;

  if (left + rect.w + pad > vw) left = vw - rect.w - pad;
  if (top + rect.h + pad > vh) top = vh - rect.h - pad;

  khopMenuEl.style.left = `${left}px`;
  khopMenuEl.style.top = `${top}px`;
  khopMenuEl.style.display = "block";
}

function hideKHopMenu() {
  if (khopMenuEl) khopMenuEl.style.display = "none";
}

function computeVisibleIds() {
  if (Facet.seed && Facet.k > 0) {
    const kSet = kHopSet(Facet.seed, Facet.k, buildAdjacency());
    if (kSet) return kSet;
  }
  return getTopNodeIdsByDegree(currentNodeLimit);
}

const applyFacets = debounce(() => {
  facetNodeKeep = computeVisibleIds();

  const nodes = allNodes.filter((n) => facetNodeKeep.has(n.id));
  const links = allLinks.filter(
    (l) =>
      facetNodeKeep.has(idOf(l.source)) && facetNodeKeep.has(idOf(l.target)),
  );

  renderGraph(nodes, links);
  reapplySearchFromUI();
}, 50);

function nodeMatches(n, q) {
  if (n.id.toLowerCase().includes(q)) return true;
  const aliases = aliasMap.get(n.id) || [];
  return aliases.some((a) => a.toLowerCase().includes(q));
}

function reapplySearchFromUI() {
  const q = (inputEl?.value || "").trim().toLowerCase();
  if (!q) {
    clearSearch();
    return;
  }
  const filterMode = !!filterToggleEl?.checked;
  filterMode ? applyFilter(q) : applyHighlight(q);
}

function applyHighlight(q) {
  if (!nodesSel) return;
  const matches = new Set(
    allNodes.filter((n) => nodeMatches(n, q)).map((n) => n.id),
  );

  nodesSel
    .classed("hidden", false)
    .classed("match", (d) => matches.has(d.id))
    .classed("dim", (d) => !matches.has(d.id));

  labelsSel
    .classed("hidden", false)
    .classed("match", (d) => matches.has(d.id))
    .classed("dim", (d) => !matches.has(d.id));

  linksSel
    .classed("hidden", false)
    .classed(
      "match",
      (l) => matches.has(idOf(l.source)) || matches.has(idOf(l.target)),
    )
    .classed(
      "dim",
      (l) => !(matches.has(idOf(l.source)) || matches.has(idOf(l.target))),
    );
}

function applyFilter(q) {
  if (!nodesSel) return;
  const matches = new Set(
    allNodes.filter((n) => nodeMatches(n, q)).map((n) => n.id),
  );

  nodesSel
    .classed("hidden", (d) => !matches.has(d.id))
    .classed("match", (d) => matches.has(d.id))
    .classed("dim", false);

  labelsSel
    .classed("hidden", (d) => !matches.has(d.id))
    .classed("match", (d) => matches.has(d.id))
    .classed("dim", false);

  linksSel
    .classed(
      "hidden",
      (l) => !(matches.has(idOf(l.source)) || matches.has(idOf(l.target))),
    )
    .classed(
      "match",
      (l) => matches.has(idOf(l.source)) || matches.has(idOf(l.target)),
    )
    .classed("dim", false);

  fitToScreen(60);
}

function clearSearch() {
  if (!nodesSel) return;
  nodesSel.classed("match dim hidden", false);
  labelsSel.classed("match dim hidden", false);
  linksSel.classed("match dim hidden", false);
}

function hideEgoOverlay() {
  if (!egoOverlayEl) return;
  egoOverlayEl.style.display = "none";
  lastEgo = null;
  d3.select("#egoCanvas").selectAll("*").remove();
}

function showEgoOverlay(seedId, k) {
  if (!seedId || !egoOverlayEl) return;
  egoOverlayEl.style.display = "block";
  lastEgo = { seedId, k };
  const count = renderEgoRadial(seedId, k);
  const noun = count === 1 ? "node" : "nodes";
  document.getElementById("egoTitle").textContent =
    `Ego network for “${seedId}” (k=${k}, ${count} ${noun})`;
}

function downloadEgoSVG() {
  const srcSvg = d3.select("#egoCanvas").select("svg").node();
  if (!srcSvg) return;

  const svg = srcSvg.cloneNode(true);

  const style = document.createElement("style");
  style.textContent =
    'text { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }';
  svg.insertBefore(style, svg.firstChild);

  const serializer = new XMLSerializer();
  const src = serializer.serializeToString(svg);
  const blob = new Blob([src], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "ego-network.svg";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function renderEgoRadial(seedId, k) {
  d3.select("#egoCanvas").selectAll("*").remove();

  const adj = buildAdjacency();
  const dist = kHopBFS(seedId, k, adj);

  if (!dist.has(seedId)) {
    dist.set(seedId, 0);
  }

  const nodesSub = allNodes.filter((n) => dist.has(n.id));
  const nodeSet = new Set(nodesSub.map((n) => n.id));

  const linksSub = allLinks.filter(
    (l) => nodeSet.has(idOf(l.source)) && nodeSet.has(idOf(l.target)),
  );

  const wrap = document.getElementById("egoCard");
  const W = wrap.clientWidth || window.innerWidth * 0.92;
  const H = (wrap.clientHeight || window.innerHeight * 0.9) - 44;

  const svg = d3
    .select("#egoCanvas")
    .append("svg")
    .attr("width", W)
    .attr("height", H);

  const g = svg.append("g");

  const zoomEgo = d3
    .zoom()
    .on("zoom", (ev) => g.attr("transform", ev.transform));
  svg.call(zoomEgo);

  const cx = W / 2;
  const cy = H / 2;
  const rings = Math.max(1, k);
  const ringGap = (0.42 * Math.min(W, H)) / rings;

  const byRing = Array.from({ length: k + 1 }, () => []);
  nodesSub.forEach((n) => byRing[dist.get(n.id) || 0].push(n));

  const subDeg = new Map(nodesSub.map((n) => [n.id, 0]));
  linksSub.forEach((l) => {
    subDeg.set(idOf(l.source), (subDeg.get(idOf(l.source)) || 0) + 1);
    subDeg.set(idOf(l.target), (subDeg.get(idOf(l.target)) || 0) + 1);
  });

  byRing.forEach((arr) =>
    arr.sort((a, b) => (subDeg.get(b.id) || 0) - (subDeg.get(a.id) || 0)),
  );

  const pos = new Map();
  byRing.forEach((arr, r) => {
    const R = r === 0 ? 0 : r * ringGap;
    const n = Math.max(1, arr.length);
    const angleStep = (2 * Math.PI) / n;

    arr.forEach((node, i) => {
      let x, y;
      if (r === 0) {
        x = cx;
        y = cy;
      } else {
        const a = i * angleStep - Math.PI / 2;
        x = cx + R * Math.cos(a);
        y = cy + R * Math.sin(a);
      }
      pos.set(node.id, { x, y });
    });
  });

  const ringStroke = cssVar("--ring-stroke");
  const linkStroke = cssVar("--link-stroke");
  const labelFill = cssVar("--label-fill");
  const seedFill = cssVar("--ego-seed-fill");
  const nodeFill = cssVar("--ego-node-fill");
  const nodeStroke = cssVar("--node-stroke");

  const ringsLayer = g.append("g");
  for (let r = 1; r <= k; r++) {
    ringsLayer
      .append("circle")
      .attr("cx", cx)
      .attr("cy", cy)
      .attr("r", r * ringGap)
      .attr("fill", "none")
      .attr("stroke", ringStroke)
      .attr("stroke-dasharray", "3 4");
  }

  const egoLinksLayer = g.append("g");
  egoLinksLayer
    .selectAll("line")
    .data(linksSub)
    .enter()
    .append("line")
    .attr("x1", (d) => pos.get(idOf(d.source)).x)
    .attr("y1", (d) => pos.get(idOf(d.source)).y)
    .attr("x2", (d) => pos.get(idOf(d.target)).x)
    .attr("y2", (d) => pos.get(idOf(d.target)).y)
    .attr("stroke", linkStroke)
    .attr("stroke-opacity", 0.35)
    .attr("stroke-width", (d) => 1 + 2 * ((d.value || 1) / (maxWeight || 1)));

  const egoNodesLayer = g.append("g");
  const circles = egoNodesLayer
    .selectAll("circle")
    .data(nodesSub)
    .enter()
    .append("circle")
    .attr("cx", (d) => pos.get(d.id).x)
    .attr("cy", (d) => pos.get(d.id).y)
    .attr("r", (d) =>
      dist.get(d.id) === 0 ? 12 : 6 + Math.min(10, subDeg.get(d.id) || 0),
    )
    .attr("fill", (d) => (dist.get(d.id) === 0 ? seedFill : nodeFill))
    .attr("stroke", nodeStroke)
    .attr("stroke-width", 2);

  const egoLabelsLayer = g.append("g");
  egoLabelsLayer
    .selectAll("text")
    .data(nodesSub)
    .enter()
    .append("text")
    .attr("x", (d) => pos.get(d.id).x + (dist.get(d.id) === 0 ? 0 : 10))
    .attr("y", (d) => pos.get(d.id).y + 4)
    .attr("text-anchor", (d) => (dist.get(d.id) === 0 ? "middle" : "start"))
    .attr("font-size", (d) => (dist.get(d.id) === 0 ? 13 : 10))
    .attr("fill", labelFill)
    .text((d) => d.id);

  circles.filter((d) => dist.get(d.id) === 0).raise();

  return nodesSub.length;
}
