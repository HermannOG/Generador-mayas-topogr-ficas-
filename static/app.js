"use strict";

// ── State ──────────────────────────────────────────────────────────────────
const tiles = [{ gpxFile: null, gpxName: "Empty Tile", routeCoords: [] }];
let activeTileIdx = 0;
let leafletMap = null;
let routeLayer = null;
let nextTileId  = 1;

// ── DOM references ─────────────────────────────────────────────────────────
const dropZone    = document.getElementById("dropZone");
const dropInner   = document.getElementById("dropInner");
const mapContainer = document.getElementById("mapContainer");
const fileInput   = document.getElementById("fileInput");
const trailDrop   = document.getElementById("trailDrop");
const trailInput  = document.getElementById("trailInput");
const tileBar     = document.getElementById("tileBar");
const tileAdd     = document.getElementById("tileAdd");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingTitle   = document.getElementById("loadingTitle");
const loadingSub     = document.getElementById("loadingSub");

// ── Drop zone: main ────────────────────────────────────────────────────────
dropZone.addEventListener("click", (e) => {
  if (e.target === dropZone || e.target === dropInner ||
      dropInner.contains(e.target)) {
    fileInput.click();
  }
});

["dragover", "dragenter"].forEach(evt =>
  dropZone.addEventListener(evt, e => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  })
);

["dragleave", "dragend"].forEach(evt =>
  dropZone.addEventListener(evt, () => dropZone.classList.remove("drag-over"))
);

dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) loadGPX(file);
});

fileInput.addEventListener("change", e => {
  const file = e.target.files[0];
  if (file) loadGPX(file);
  fileInput.value = "";
});

// ── Drop zone: trail (secondary GPX) ──────────────────────────────────────
trailDrop.addEventListener("click", () => trailInput.click());
trailDrop.addEventListener("dragover", e => { e.preventDefault(); trailDrop.style.background = "#e8f8fc"; });
trailDrop.addEventListener("dragleave", () => { trailDrop.style.background = ""; });
trailDrop.addEventListener("drop", e => {
  e.preventDefault();
  trailDrop.style.background = "";
  const file = e.dataTransfer.files[0];
  if (file) loadGPX(file);
});
trailInput.addEventListener("change", e => {
  const file = e.target.files[0];
  if (file) loadGPX(file);
  trailInput.value = "";
});

// ── Load GPX ───────────────────────────────────────────────────────────────
function loadGPX(file) {
  tiles[activeTileIdx].gpxFile = file;
  tiles[activeTileIdx].gpxName = file.name.replace(/\.gpx$/i, "");

  const reader = new FileReader();
  reader.onload = e => {
    try {
      const parser = new DOMParser();
      const doc = parser.parseFromString(e.target.result, "text/xml");
      const coords = extractCoords(doc);
      tiles[activeTileIdx].routeCoords = coords;
      showMap(coords);
      updateTileUI(activeTileIdx);
    } catch (err) {
      alert("Error leyendo el archivo GPX: " + err.message);
    }
  };
  reader.readAsText(file);
}

function extractCoords(doc) {
  const tags = ["trkpt", "rtept", "wpt"];
  let pts = [];
  for (const tag of tags) {
    const nodes = doc.querySelectorAll(tag);
    if (nodes.length > 0) {
      pts = Array.from(nodes).map(n => [
        parseFloat(n.getAttribute("lat")),
        parseFloat(n.getAttribute("lon")),
      ]);
      break;
    }
  }
  return pts;
}

// ── Map display ────────────────────────────────────────────────────────────
function showMap(coords) {
  if (!coords.length) return;

  // Switch from drop inner to map container
  dropInner.style.display = "none";
  mapContainer.style.display = "block";

  if (!leafletMap) {
    leafletMap = L.map("mapContainer", { attributionControl: false, zoomControl: true });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
    }).addTo(leafletMap);
  }

  if (routeLayer) {
    leafletMap.removeLayer(routeLayer);
    routeLayer = null;
  }

  routeLayer = L.polyline(coords, { color: "#e74c3c", weight: 3, opacity: 0.9 }).addTo(leafletMap);
  leafletMap.fitBounds(routeLayer.getBounds(), { padding: [24, 24] });

  // Add start/end markers
  if (coords.length > 1) {
    L.circleMarker(coords[0], { radius: 7, color: "#2ecc71", fillColor: "#2ecc71", fillOpacity: 0.9 })
      .bindTooltip("Start").addTo(leafletMap);
    L.circleMarker(coords[coords.length - 1], { radius: 7, color: "#e74c3c", fillColor: "#e74c3c", fillOpacity: 0.9 })
      .bindTooltip("End").addTo(leafletMap);
  }

  // Force map resize
  setTimeout(() => leafletMap.invalidateSize(), 50);
}

// ── Tile management ────────────────────────────────────────────────────────
function updateTileUI(idx) {
  const tileEl = document.getElementById(`tile${idx}`);
  if (!tileEl) return;
  const label = tileEl.querySelector(".tile-label");
  if (label) label.textContent = tiles[idx].gpxName || "Tile " + (idx + 1);
  tileEl.classList.add("active");
}

tileAdd.addEventListener("click", () => {
  const idx = nextTileId++;
  tiles.push({ gpxFile: null, gpxName: "Empty Tile", routeCoords: [] });

  const el = document.createElement("div");
  el.className = "tile tile-empty";
  el.id = `tile${idx}`;
  el.dataset.idx = idx;
  el.innerHTML = `<span class="tile-label">Empty Tile</span>`;
  el.addEventListener("click", () => setActiveTile(idx));
  tileBar.insertBefore(el, tileAdd);

  setActiveTile(idx);
});

function setActiveTile(idx) {
  activeTileIdx = idx;
  document.querySelectorAll(".tile-empty").forEach(t => t.classList.remove("active"));
  const el = document.getElementById(`tile${idx}`);
  if (el) el.classList.add("active");

  // Show map or drop zone for this tile
  const t = tiles[idx];
  if (t && t.routeCoords.length > 0) {
    showMap(t.routeCoords);
  } else {
    dropInner.style.display = "flex";
    mapContainer.style.display = "none";
  }
}

document.getElementById("tile0").addEventListener("click", () => setActiveTile(0));

// ── Sliders ────────────────────────────────────────────────────────────────
function bindSlider(sliderId, valId) {
  const s = document.getElementById(sliderId);
  const v = document.getElementById(valId);
  if (!s || !v) return;
  v.textContent = s.value;
  s.addEventListener("input", () => { v.textContent = s.value; });
}

bindSlider("heightScale",   "heightScaleVal");
bindSlider("trailWidth",    "trailWidthVal");
bindSlider("trailHeight",   "trailHeightVal");
bindSlider("baseThickness", "baseThicknessVal");
bindSlider("trailBorder",   "trailBorderVal");
bindSlider("baseSize",      "baseSizeVal");

// ── Shape buttons ──────────────────────────────────────────────────────────
document.querySelectorAll(".shape-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".shape-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
  });
});

// ── Color pickers ──────────────────────────────────────────────────────────
function bindColorBar(barId, inputId) {
  const bar = document.getElementById(barId);
  const inp = document.getElementById(inputId);
  if (!bar || !inp) return;
  bar.style.background = inp.value;
  bar.addEventListener("click", () => inp.click());
  inp.addEventListener("input", () => { bar.style.background = inp.value; });
}

bindColorBar("waterColorBar", "waterColor");
bindColorBar("landColorBar",  "landColor");
bindColorBar("rockColorBar",  "rockColor");
bindColorBar("trailColorBar", "trailColor");

// ── Collect settings ───────────────────────────────────────────────────────
function collectSettings() {
  const centerOn = document.querySelector("input[name='centerOn']:checked")?.value ?? "fit";
  const shape    = document.querySelector(".shape-btn.active")?.dataset.shape ?? "square";

  return {
    include_seas:      document.getElementById("incSeas")?.checked   ?? true,
    include_lakes:     document.getElementById("incLakes")?.checked  ?? true,
    include_rivers:    document.getElementById("incRivers")?.checked ?? false,
    print_separately:  document.getElementById("printSep")?.checked  ?? false,
    include_buildings: document.getElementById("incBuildings")?.checked ?? false,
    water_color:       document.getElementById("waterColor")?.value  ?? "#0055ff",
    land_color:        document.getElementById("landColor")?.value   ?? "#00cc00",
    rock_color:        document.getElementById("rockColor")?.value   ?? "#aaaaaa",
    trail_color:       document.getElementById("trailColor")?.value  ?? "#ff0000",
    tree_line:         parseFloat(document.getElementById("treeLine")?.value   ?? 1250),
    height_scale:      parseFloat(document.getElementById("heightScale")?.value ?? 1),
    trail_width:       parseFloat(document.getElementById("trailWidth")?.value  ?? 1),
    trail_height:      parseFloat(document.getElementById("trailHeight")?.value ?? 1),
    use_gpx_elevation: document.getElementById("useGPXEle")?.checked ?? false,
    center_on:         centerOn,
    shape:             shape,
    base_thickness:    parseFloat(document.getElementById("baseThickness")?.value ?? 5),
    trail_border:      parseFloat(document.getElementById("trailBorder")?.value  ?? 0.25),
    base_size:         parseFloat(document.getElementById("baseSize")?.value     ?? 100),
    high_resolution:   document.getElementById("highRes")?.checked   ?? false,
    cache_id:          tiles[activeTileIdx]?.cacheId ?? null,
  };
}

// ── Preview 3D ─────────────────────────────────────────────────────────────
document.getElementById("previewBtn").addEventListener("click", async () => {
  const tile = tiles[activeTileIdx];
  if (!tile?.gpxFile) {
    alert("Por favor, carga un archivo GPX primero.");
    return;
  }

  const btn = document.getElementById("previewBtn");
  btn.disabled = true;
  btn.textContent = "⏳ Cargando elevaciones…";

  try {
    const fd = new FormData();
    fd.append("gpx_file", tile.gpxFile);
    fd.append("settings", JSON.stringify(collectSettings()));

    const resp = await fetch("/api/preview-mesh", { method: "POST", body: fd });
    if (!resp.ok) {
      let msg = `Error ${resp.status}`;
      try { msg = (await resp.json()).detail ?? msg; } catch (_) {}
      throw new Error(msg);
    }

    const data = await resp.json();

    // Cache the elevation ID so Download can skip re-fetching
    tile.cacheId = data.cache_id;

    // Update stats bar
    const stats = document.getElementById("previewStats");
    if (stats) {
      stats.textContent =
        `Elevación: ${data.ele_min.toFixed(0)} m – ${data.ele_max.toFixed(0)} m  ·  ${data.gpx_count} puntos GPX`;
    }

    window.Preview3D.render(data);
    window.Preview3D.show();

  } catch (err) {
    alert("Error en la vista previa:\n" + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "🏔️ Preview 3D";
  }
});

// ── Download from inside the preview modal ─────────────────────────────────
document.getElementById("downloadFromPreview").addEventListener("click", () => {
  window.Preview3D.hide();
  generateSTL();
});

// ── Generate & download ────────────────────────────────────────────────────
async function generateSTL() {
  const tile = tiles[activeTileIdx];
  if (!tile?.gpxFile) {
    alert("Por favor, carga un archivo GPX primero.");
    return;
  }

  const settings = collectSettings();

  loadingOverlay.hidden = false;
  loadingTitle.textContent = "Generando tu mapa 3D…";
  loadingSub.textContent   = "Descargando datos de elevación (puede tardar 30–120 s)";

  try {
    const fd = new FormData();
    fd.append("gpx_file", tile.gpxFile);
    fd.append("settings", JSON.stringify(settings));

    const resp = await fetch("/api/generate", { method: "POST", body: fd });

    if (!resp.ok) {
      let msg = `Error ${resp.status}`;
      try { const j = await resp.json(); msg = j.detail || msg; } catch (_) {}
      throw new Error(msg);
    }

    const blob = await resp.blob();
    const cd   = resp.headers.get("Content-Disposition") ?? "";
    const match = cd.match(/filename="?([^";]+)"?/);
    const filename = match ? match[1] : (settings.print_separately ? "map3d.zip" : "map3d.stl");

    const url = URL.createObjectURL(blob);
    const a   = Object.assign(document.createElement("a"), { href: url, download: filename });
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

  } catch (err) {
    alert("Error generando el modelo:\n" + err.message);
  } finally {
    loadingOverlay.hidden = true;
  }
}

document.getElementById("downloadBtn").addEventListener("click", generateSTL);
document.getElementById("updateBtn").addEventListener("click", generateSTL);
document.getElementById("applyAllBtn").addEventListener("click", () => {
  // In a future update: apply current settings to all tiles
  alert("Configuración guardada. Se aplicará a todos los archivos al generar.");
});
