let root,
  container,
  zoom,
  width = 2000,
  height = 1000;
let nodesSel, linksSel, labelsSel, simulation;
let allNodes = [],
  allLinks = [];
let degreeMap,
  rScale,
  maxWeight = 1;

let inputEl, filterToggleEl;
let nodeLimitInputEl, applyNodeLimitBtn;
let darkModeToggleEl;
let currentNodeLimit = 15;
let tip;

const Facet = { degMin: 0, seed: "", k: 0 };
let facetNodeKeep = new Set();
let facetLinkKeep = new Set();

const debounce = (fn, ms = 200) => {
  let t;
  return (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
};

const linkKey = (l) => {
  const s = typeof l.source === "object" ? l.source.id : l.source;
  const t = typeof l.target === "object" ? l.target.id : l.target;
  return `${s}→${t}`;
};

function reapplySearchFromUI() {
  const q = (inputEl?.value || "").trim().toLowerCase();
  if (!q) {
    clearSearch();
    return;
  }
  const filterMode = !!filterToggleEl?.checked;
  filterMode ? applyFilter(q) : applyHighlight(q);
}

let khopMenuEl, kInputEl, applyKHopBtn, clearKHopBtn, closeKHopBtn;
let egoViewBtn, egoOverlayEl, egoCanvasEl, egoCloseBtn, egoDownloadBtn;
let khopSeedId = null;

fetch("character_interactions.json")
  .then((response) => response.json())
  .then((data) => {
    ({ root, container, zoom, nodesSel, linksSel, labelsSel, simulation } =
      createGraph(data.characters, data.matrix));

    simulation.on("end", () => fitToScreen(60));
    window.addEventListener("resize", () => fitToScreen(60));

    inputEl = document.getElementById("searchInput");
    filterToggleEl = document.getElementById("filterToggle");
    nodeLimitInputEl = document.getElementById("nodeLimitInput");
    applyNodeLimitBtn = document.getElementById("applyNodeLimitBtn");
    darkModeToggleEl = document.getElementById("darkModeToggle");

    const clearBtn = document.getElementById("clearBtn");
    const fitBtn = document.getElementById("fitBtn");

    egoViewBtn = document.getElementById("egoViewBtn");
    egoOverlayEl = document.getElementById("egoOverlay");
    egoCanvasEl = document.getElementById("egoCanvas");
    egoCloseBtn = document.getElementById("egoCloseBtn");
    egoDownloadBtn = document.getElementById("egoDownloadBtn");

    khopMenuEl = document.getElementById("khopMenu");
    kInputEl = document.getElementById("kHop");
    applyKHopBtn = document.getElementById("applyKHop");
    clearKHopBtn = document.getElementById("clearKHop");
    closeKHopBtn = document.getElementById("closeKHop");

    currentNodeLimit = Math.max(1, +nodeLimitInputEl.value || 15);
    nodeLimitInputEl.value = String(currentNodeLimit);

    initDarkMode();

    applyNodeLimitBtn.addEventListener("click", () => {
      currentNodeLimit = Math.max(1, +nodeLimitInputEl.value || 15);
      nodeLimitInputEl.value = String(currentNodeLimit);
      applyFacets();
    });

    nodeLimitInputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        currentNodeLimit = Math.max(1, +nodeLimitInputEl.value || 15);
        nodeLimitInputEl.value = String(currentNodeLimit);
        applyFacets();
      }
    });

    darkModeToggleEl.addEventListener("change", () => {
      setDarkMode(darkModeToggleEl.checked);
    });

    egoViewBtn.addEventListener("click", () => {
      const kVal = Math.max(0, Math.min(8, +kInputEl.value || 0));
      if (!khopSeedId || kVal === 0) {
        kInputEl.value = "1";
      }
      showEgoOverlay(khopSeedId, Math.max(1, kVal));
    });

    egoCloseBtn.addEventListener("click", hideEgoOverlay);
    egoDownloadBtn.addEventListener("click", downloadEgoSVG);

    const handleSearch = debounce(() => {
      const q = (inputEl.value || "").trim().toLowerCase();
      const filterMode = filterToggleEl.checked;
      if (!q) {
        clearSearch();
        return;
      }
      filterMode ? applyFilter(q) : applyHighlight(q);
    }, 120);

    inputEl.addEventListener("input", handleSearch);
    filterToggleEl.addEventListener("change", () => handleSearch());

    clearBtn.addEventListener("click", () => {
      inputEl.value = "";
      filterToggleEl.checked = false;
      clearSearch();
    });

    fitBtn.addEventListener("click", () => fitToScreen(60));

    applyKHopBtn.addEventListener("click", () => {
      Facet.k = Math.max(0, Math.min(6, +kInputEl.value || 0));
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

    applyFacets();
  })
  .catch((error) => console.error("Error loading JSON data:", error));

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
  if (!root || !nodesSel || !linksSel || !labelsSel) return;

  const styles = getComputedStyle(document.body);
  const bg = styles.getPropertyValue("--bg").trim();
  const labelFill = styles.getPropertyValue("--label-fill").trim();
  const labelStroke = styles.getPropertyValue("--label-stroke").trim();
  const linkStroke = styles.getPropertyValue("--link-stroke").trim();
  const tooltipBg = styles.getPropertyValue("--tooltip-bg").trim();
  const tooltipBorder = styles.getPropertyValue("--tooltip-border").trim();
  const tooltipText = styles.getPropertyValue("--tooltip-text").trim();

  root.style("background", bg);

  linksSel.attr("stroke", linkStroke);
  labelsSel.attr("fill", labelFill).attr("stroke", labelStroke);

  if (tip) {
    tip
      .style("background", tooltipBg)
      .style("border", `1px solid ${tooltipBorder}`)
      .style("color", tooltipText);
  }
}

function createGraph(characters, matrix) {
  root = d3
    .select("body")
    .append("svg")
    .attr("width", "100%")
    .attr("height", "100%")
    .attr("viewBox", [0, 0, width, height])
    .attr("preserveAspectRatio", "xMidYMid meet");

  container = root.append("g");

  zoom = d3
    .zoom()
    .on("zoom", (event) => container.attr("transform", event.transform));
  root.call(zoom);

  const nodes = characters.map((id) => ({ id }));
  const links = [];
  matrix.forEach((row, i) => {
    row.forEach((value, j) => {
      if (value > 0) {
        links.push({ source: characters[i], target: characters[j], value });
      }
    });
  });

  allNodes = nodes;
  allLinks = links;

  maxWeight = d3.max(links, (d) => d.value) || 1;

  degreeMap = new Map(nodes.map((n) => [n.id, 0]));
  links.forEach((l) => {
    degreeMap.set(l.source, (degreeMap.get(l.source) || 0) + 1);
    degreeMap.set(l.target, (degreeMap.get(l.target) || 0) + 1);
  });

  const degExtent = d3.extent(nodes, (d) => degreeMap.get(d.id));
  rScale = d3
    .scaleSqrt()
    .domain([degExtent[0] || 1, degExtent[1] || 1])
    .range([6, 22]);

  const color = d3
    .scaleSequential(d3.interpolateTurbo)
    .domain([degExtent[0] || 0, degExtent[1] || 1]);

  const linksLayer = container.append("g").attr("class", "links");
  const nodesLayer = container.append("g").attr("class", "nodes");
  const labelsLayer = container.append("g").attr("class", "labels");

  root
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

  const linkPath = linksLayer
    .selectAll("path")
    .data(links)
    .enter()
    .append("path")
    .attr("class", "link")
    .attr("fill", "none")
    .attr(
      "stroke",
      getComputedStyle(document.body).getPropertyValue("--link-stroke").trim(),
    )
    .attr("stroke-opacity", 0.35)
    .attr("stroke-width", (d) => 1 + 2 * (d.value / maxWeight))
    .attr("marker-end", "url(#arrow)");

  const node = nodesLayer
    .selectAll("circle")
    .data(nodes)
    .enter()
    .append("circle")
    .attr("class", "node")
    .attr("r", (d) => rScale(degreeMap.get(d.id)))
    .attr("fill", (d) => color(degreeMap.get(d.id)))
    .attr("stroke", "white")
    .attr("stroke-width", 1.2)
    .call(drag());

  const labels = labelsLayer
    .selectAll("text")
    .data(nodes)
    .enter()
    .append("text")
    .attr("class", "label")
    .text((d) => d.id)
    .attr("font-size", 10)
    .attr(
      "fill",
      getComputedStyle(document.body).getPropertyValue("--label-fill").trim(),
    )
    .attr(
      "stroke",
      getComputedStyle(document.body).getPropertyValue("--label-stroke").trim(),
    )
    .attr("stroke-width", 3)
    .attr("paint-order", "stroke")
    .style("pointer-events", "none");

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

  updateGraphTheme();

  const neighbors = new Map(nodes.map((n) => [n.id, new Set()]));
  links.forEach((l) => {
    neighbors.get(l.source).add(l.target);
    neighbors.get(l.target).add(l.source);
  });

  const isNeighbor = (a, b) => a.id === b.id || neighbors.get(a.id)?.has(b.id);

  node
    .on("mouseover", function (event, d) {
      node.classed("dim", (n) => !isNeighbor(d, n));
      labels.classed("dim", (n) => !isNeighbor(d, n));
      linkPath.classed(
        "dim",
        (l) => l.source.id !== d.id && l.target.id !== d.id,
      );
      labels.filter((n) => n.id === d.id).raise();
      tip.style("opacity", 1);
    })
    .on("mousemove", (event, d) => {
      const deg = degreeMap.get(d.id);
      tip
        .style("left", event.clientX + 12 + "px")
        .style("top", event.clientY + 12 + "px")
        .html(`<b>${d.id}</b><br>Connections: ${deg}`);
    })
    .on("mouseleave", () => {
      tip.style("opacity", 0);
    })
    .on("mouseout", () => {
      node.classed("dim", false);
      labels.classed("dim", false);
      linkPath.classed("dim", false);
      reapplySearchFromUI();
    })
    .on("click", function (event, d) {
      if (event.defaultPrevented) return;
      khopSeedId = d.id;
      showKHopMenu(event.clientX, event.clientY, d.id);
    });

  const curver = d3
    .scaleLinear()
    .domain(d3.extent(nodes, (d) => d.id.length))
    .range([0.45, 0.85]);

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
    .force("center", d3.forceCenter(width / 2, height / 2));

  simulation.on("tick", () => {
    linkPath.attr("d", arcPath);
    node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
    labels.attr("x", (d) => d.x + 6).attr("y", (d) => d.y + 3);
  });

  nodesSel = node;
  linksSel = linkPath;
  labelsSel = labels;

  return { root, container, zoom, nodesSel, linksSel, labelsSel, simulation };
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

  root.transition().duration(300).call(zoom.transform, transform);
}

function buildAdjacency() {
  const adj = new Map(allNodes.map((n) => [n.id, new Set()]));
  for (const l of allLinks) {
    const s = typeof l.source === "object" ? l.source.id : l.source;
    const t = typeof l.target === "object" ? l.target.id : l.target;
    adj.get(s)?.add(t);
    adj.get(t)?.add(s);
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

const applyFacets = debounce(() => {
  const topNodeKeep = getTopNodeIdsByDegree(currentNodeLimit);

  const degKeep = new Set(
    allNodes
      .filter(
        (n) =>
          topNodeKeep.has(n.id) && (degreeMap.get(n.id) || 0) >= Facet.degMin,
      )
      .map((n) => n.id),
  );

  let kKeep = null;
  if (Facet.seed && Facet.k > 0) {
    kKeep = kHopSet(Facet.seed, Facet.k, buildAdjacency());
  }

  facetNodeKeep = new Set();
  for (const n of allNodes) {
    const keepTop = topNodeKeep.has(n.id);
    const keepDeg = degKeep.has(n.id);
    const keepK = kKeep ? kKeep.has(n.id) : true;
    if (keepTop && keepDeg && keepK) {
      facetNodeKeep.add(n.id);
    }
  }

  facetLinkKeep = new Set();
  for (const l of allLinks) {
    const s = typeof l.source === "object" ? l.source.id : l.source;
    const t = typeof l.target === "object" ? l.target.id : l.target;
    if (facetNodeKeep.has(s) && facetNodeKeep.has(t)) {
      facetLinkKeep.add(`${s}→${t}`);
    }
  }

  nodesSel.classed("hidden", (d) => !facetNodeKeep.has(d.id));
  labelsSel.classed("hidden", (d) => !facetNodeKeep.has(d.id));
  linksSel.classed("hidden", (l) => !facetLinkKeep.has(linkKey(l)));

  reapplySearchFromUI();
  fitToScreen(60);
}, 50);

function applyHighlight(q) {
  const matches = new Set(
    allNodes.filter((n) => n.id.toLowerCase().includes(q)).map((n) => n.id),
  );

  nodesSel
    .classed("match", (d) => facetNodeKeep.has(d.id) && matches.has(d.id))
    .classed("dim", (d) => facetNodeKeep.has(d.id) && !matches.has(d.id));

  labelsSel
    .classed("match", (d) => facetNodeKeep.has(d.id) && matches.has(d.id))
    .classed("dim", (d) => facetNodeKeep.has(d.id) && !matches.has(d.id));

  linksSel
    .classed(
      "match",
      (l) =>
        facetLinkKeep.has(linkKey(l)) &&
        (matches.has(l.source.id) || matches.has(l.target.id)),
    )
    .classed(
      "dim",
      (l) =>
        facetLinkKeep.has(linkKey(l)) &&
        !(matches.has(l.source.id) || matches.has(l.target.id)),
    );
}

function applyFilter(q) {
  const matches = new Set(
    allNodes.filter((n) => n.id.toLowerCase().includes(q)).map((n) => n.id),
  );

  nodesSel
    .classed("hidden", (d) => !(facetNodeKeep.has(d.id) && matches.has(d.id)))
    .classed("match", (d) => facetNodeKeep.has(d.id) && matches.has(d.id))
    .classed("dim", false);

  labelsSel
    .classed("hidden", (d) => !(facetNodeKeep.has(d.id) && matches.has(d.id)))
    .classed("match", (d) => facetNodeKeep.has(d.id) && matches.has(d.id))
    .classed("dim", false);

  const visibleMatched = new Set(
    allNodes
      .filter((n) => facetNodeKeep.has(n.id) && matches.has(n.id))
      .map((n) => n.id),
  );

  linksSel
    .classed("hidden", (l) => {
      if (!facetLinkKeep.has(linkKey(l))) return true;
      const s = l.source.id;
      const t = l.target.id;
      return !(visibleMatched.has(s) || visibleMatched.has(t));
    })
    .classed(
      "match",
      (l) =>
        facetLinkKeep.has(linkKey(l)) &&
        (visibleMatched.has(l.source.id) || visibleMatched.has(l.target.id)),
    )
    .classed("dim", false);

  fitToScreen(60);
}

function clearSearch() {
  nodesSel.classed("match dim hidden", false);
  labelsSel.classed("match dim hidden", false);
  linksSel.classed("match dim hidden", false);

  nodesSel.classed("hidden", (d) => !facetNodeKeep.has(d.id));
  labelsSel.classed("hidden", (d) => !facetNodeKeep.has(d.id));
  linksSel.classed("hidden", (l) => !facetLinkKeep.has(linkKey(l)));
}

function hideEgoOverlay() {
  if (!egoOverlayEl) return;
  egoOverlayEl.style.display = "none";
  d3.select("#egoCanvas").selectAll("*").remove();
}

function showEgoOverlay(seedId, k) {
  if (!seedId || !egoOverlayEl) return;
  egoOverlayEl.style.display = "block";
  const count = renderEgoRadial(seedId, k);
  const noun = count === 1 ? "node" : "nodes";
  document.getElementById("egoTitle").textContent =
    `Ego network for “${seedId}” (k=${k}, ${count} ${noun})`;
}

function downloadEgoSVG(scale = 1) {
  const srcSvg = d3.select("#egoCanvas").select("svg").node();
  if (!srcSvg) return;

  const svg = srcSvg.cloneNode(true);
  const $svg = d3.select(svg);

  $svg.selectAll("text").each(function () {
    const el = this;
    const attrFS = el.getAttribute("font-size");
    const base = attrFS
      ? parseFloat(attrFS)
      : parseFloat(getComputedStyle(el).fontSize) || 12;
    el.setAttribute("font-size", (base * scale).toFixed(2));

    const sw = el.getAttribute("stroke-width");
    if (sw) {
      el.setAttribute("stroke-width", (parseFloat(sw) * scale).toFixed(2));
    }
  });

  const style = document.createElement("style");
  style.textContent = `
    text { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
  `;
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

  const nodesSub = allNodes.filter(
    (n) => dist.has(n.id) && facetNodeKeep.has(n.id),
  );
  const nodeSet = new Set(nodesSub.map((n) => n.id));

  const linksSub = allLinks.filter((l) => {
    const s = typeof l.source === "object" ? l.source.id : l.source;
    const t = typeof l.target === "object" ? l.target.id : l.target;
    return nodeSet.has(s) && nodeSet.has(t);
  });

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
  const ringScale = 9.3;
  const ringGap = (ringScale * Math.min(W, H)) / (2 * (rings + 1));

  const byRing = Array.from({ length: k + 1 }, () => []);
  nodesSub.forEach((n) => byRing[dist.get(n.id) || 0].push(n));

  const subDeg = new Map(nodesSub.map((n) => [n.id, 0]));
  linksSub.forEach((l) => {
    const s = typeof l.source === "object" ? l.source.id : l.source;
    const t = typeof l.target === "object" ? l.target.id : l.target;
    subDeg.set(s, (subDeg.get(s) || 0) + 1);
    subDeg.set(t, (subDeg.get(t) || 0) + 1);
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

  const styles = getComputedStyle(document.body);
  const ringStroke = styles.getPropertyValue("--ring-stroke").trim();
  const linkStroke = styles.getPropertyValue("--link-stroke").trim();
  const labelFill = styles.getPropertyValue("--label-fill").trim();

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

  const linksLayer = g.append("g");
  linksLayer
    .selectAll("line")
    .data(linksSub)
    .enter()
    .append("line")
    .attr(
      "x1",
      (d) => pos.get(typeof d.source === "object" ? d.source.id : d.source).x,
    )
    .attr(
      "y1",
      (d) => pos.get(typeof d.source === "object" ? d.source.id : d.source).y,
    )
    .attr(
      "x2",
      (d) => pos.get(typeof d.target === "object" ? d.target.id : d.target).x,
    )
    .attr(
      "y2",
      (d) => pos.get(typeof d.target === "object" ? d.target.id : d.target).y,
    )
    .attr("stroke", linkStroke)
    .attr("stroke-opacity", 0.35)
    .attr("stroke-width", (d) => 1 + 2 * ((d.value || 1) / (maxWeight || 1)));

  const nodesLayer = g.append("g");
  const circles = nodesLayer
    .selectAll("circle")
    .data(nodesSub)
    .enter()
    .append("circle")
    .attr("cx", (d) => pos.get(d.id).x)
    .attr("cy", (d) => pos.get(d.id).y)
    .attr("r", (d) =>
      dist.get(d.id) === 0 ? 12 : 6 + Math.min(10, subDeg.get(d.id) || 0),
    )
    .attr("fill", (d) => (dist.get(d.id) === 0 ? "#111827" : "#2563eb"))
    .attr("stroke", "#fff")
    .attr("stroke-width", 2);

  const labelsLayer = g.append("g");
  labelsLayer
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
