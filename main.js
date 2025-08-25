fetch("character_interactions.json")
  .then((response) => response.json())
  .then((data) => {
    createGraph(data.characters, data.matrix);
  })
  .catch((error) => console.error("Error loading JSON data:", error));

function createGraph(characters, matrix) {
  const width = 2000,
    height = 1000;

  // Root SVG
  const root = d3
    .select("body")
    .append("svg")
    .attr("width", "100%")
    .attr("height", "100%")
    .attr("viewBox", [0, 0, width, height])
    .attr("preserveAspectRatio", "xMidYMid meet");

  const container = root.append("g");

  // --- Zoom
  const zoom = d3.zoom().on("zoom", (event) => {
    container.attr("transform", event.transform);
  });
  root.call(zoom);

  // --- Build nodes/links from matrix
  const nodes = characters.map((id) => ({ id }));
  const links = [];
  matrix.forEach((row, i) => {
    row.forEach((value, j) => {
      if (value > 0) {
        links.push({ source: characters[i], target: characters[j], value });
      }
    });
  });

  const maxWeight = d3.max(links, (d) => d.value) || 1;

  // --- Degree
  const degree = new Map(nodes.map((n) => [n.id, 0]));
  links.forEach((l) => {
    degree.set(l.source, (degree.get(l.source) || 0) + 1);
    degree.set(l.target, (degree.get(l.target) || 0) + 1);
  });

  const degExtent = d3.extent(nodes, (d) => degree.get(d.id));
  const rScale = d3
    .scaleSqrt()
    .domain([degExtent[0] || 1, degExtent[1] || 1])
    .range([6, 22]);
  const color = d3
    .scaleSequential(d3.interpolateTurbo)
    .domain([degExtent[0] || 0, degExtent[1] || 1]);

  // --- Adjacency map
  const neighbors = new Map();
  nodes.forEach((n) => neighbors.set(n.id, new Set()));
  links.forEach((l) => {
    neighbors.get(l.source).add(l.target);
    neighbors.get(l.target).add(l.source);
  });

  // --- Layers
  const linksLayer = container.append("g").attr("class", "links");
  const nodesLayer = container.append("g").attr("class", "nodes");
  const labelsLayer = container.append("g").attr("class", "labels");

  // --- Arrowheads
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

  // --- Links as curved paths
  const linkPath = linksLayer
    .selectAll("path")
    .data(links)
    .enter()
    .append("path")
    .attr("fill", "none")
    .attr("stroke", "#9aa0a6")
    .attr("stroke-opacity", 0.35)
    .attr("stroke-width", (d) => 1 + 2 * (d.value / maxWeight))
    .attr("marker-end", "url(#arrow)");

  // --- Nodes
  const node = nodesLayer
    .selectAll("circle")
    .data(nodes)
    .enter()
    .append("circle")
    .attr("r", (d) => rScale(degree.get(d.id)))
    .attr("fill", (d) => color(degree.get(d.id)))
    .attr("stroke", "white")
    .attr("stroke-width", 1.2)
    .call(drag(simulationBuilder));

  // --- Labels
  const labels = labelsLayer
    .selectAll("text")
    .data(nodes)
    .enter()
    .append("text")
    .text((d) => d.id)
    .attr("font-size", 10)
    .attr("fill", "#222")
    .attr("stroke", "white")
    .attr("stroke-width", 3)
    .attr("paint-order", "stroke")
    .style("pointer-events", "none");

  // --- Tooltips
  const tip = d3
    .select("body")
    .append("div")
    .attr("id", "tip")
    .style("position", "fixed")
    .style("pointer-events", "none")
    .style("opacity", 0)
    .style("background", "#fff")
    .style("border", "1px solid #ddd")
    .style("padding", "6px 8px")
    .style("border-radius", "6px")
    .style("box-shadow", "0 2px 8px rgba(0,0,0,.08)")
    .style("font", "12px system-ui");

  // --- Simulation
  const simulation = simulationBuilder(nodes, links);

  // --- Hover focus + tooltip
  function isNeighbor(a, b) {
    if (a.id === b.id) return true;
    return neighbors.get(a.id)?.has(b.id);
  }

  node
    .on("mouseover", function (event, d) {
      node.attr("opacity", (n) => (isNeighbor(d, n) ? 1 : 0.15));
      labels.attr("opacity", (n) => (isNeighbor(d, n) ? 1 : 0.1));
      linkPath.attr("opacity", (l) =>
        l.source.id === d.id || l.target.id === d.id ? 0.9 : 0.1
      );
      labels.filter((n) => n.id === d.id).raise();
      tip.style("opacity", 1);
    })
    .on("mousemove", (event, d) => {
      const deg = degree.get(d.id);
      tip
        .style("left", event.clientX + 12 + "px")
        .style("top", event.clientY + 12 + "px")
        .html(`<b>${d.id}</b><br>Connections: ${deg}`);
    })
    .on("mouseleave", () => {
      tip.style("opacity", 0);
    })
    .on("mouseout", () => {
      node.attr("opacity", 1);
      labels.attr("opacity", 1);
      linkPath.attr("opacity", 0.35);
    });

  // --- Bundling-lite curvature
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

  // --- Tick updates
  simulation.on("tick", () => {
    linkPath.attr("d", arcPath);
    node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
    labels.attr("x", (d) => d.x + 6).attr("y", (d) => d.y + 3);
  });

  // --- Force/drag helpers
  function simulationBuilder(nodes, links) {
    return d3
      .forceSimulation(nodes)
      .force(
        "link",
        d3
          .forceLink(links)
          .id((d) => d.id)
          .distance((d) => 120 + 240 * (1 - d.value / maxWeight) * 1.33)
          .strength((d) => 0.2 + 0.6 * (d.value / maxWeight))
      )
      .force(
        "charge",
        d3.forceManyBody().strength(-200).theta(0.9).distanceMax(930)
      )
      .force(
        "collide",
        d3
          .forceCollide()
          .radius((d) => rScale(degree.get(d.id)) + 9)
          .strength(0.9)
      )
      .force("center", d3.forceCenter(width / 2, height / 2));
  }

  function drag(simBuilder) {
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
}
