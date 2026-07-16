"use strict";

/**
 * Generator page logic — vanilla-JS port of the topotrail.com React
 * component ($S in their bundle), which is the source of truth:
 *   - per-tile settings + GPX file list (multiple trails per tile)
 *   - dropping a GPX starts generation immediately (POST /api/upload, NDJSON)
 *   - fileHash reuse: files upload once, re-generations send hashes only
 *   - inline OBJ viewer in the main pane, image.png thumbnails on tiles
 *   - Download fetches the ready-made ZIP for the tile's job path
 */

const API_BASE = "/api";

// Site defaults (verbatim from the topotrail.com bundle)
const DEFAULT_SETTINGS = {
  waterColor: "#0084ff",
  landColor: "#00FF00",
  trackColor: "#FC5200",
  rockColor: "#BDBDBD",
  treeLine: 1250,
  heightScale: 1,
  trailWidth: 1,
  trailHeight: 1,
  useHeightFromGpx: false,
  shape: "hexagon",
  distanceTrackToBorder: 0,   // extra margin beyond the built-in 5% minimum
  baseThickness: 5,
  includeSeas: true,
  includeLakes: true,
  includeRivers: false,
  base_size: 100,
  center: "normal",
  buildings: false,
  building_scale: 1,
  buildingsColor: "#777777",
  higherResolution: false,
  singleColor: false,
  singleColor_gap: 0.5,
  // TopoTrail extensions (not on topotrail.com): hexagon border with text
  baseColor: "#FFFFFF",
  textColor: "#000000",
  borderLabels: ["", "", "", "", "", ""],
  // order: top, upper-right, lower-right, bottom, lower-left, upper-left
};

// Panel inputs whose element id === settings key
const COLOR_KEYS = ["waterColor", "landColor", "trackColor", "rockColor",
                    "buildingsColor", "baseColor", "textColor"];
const CHECKBOX_KEYS = ["useHeightFromGpx", "higherResolution", "includeSeas",
                       "includeLakes", "includeRivers", "buildings"];

// Settings the site parses as numbers on change
const NUMERIC_KEYS = [
  "treeLine", "heightScale", "trailWidth", "trailHeight", "shapeWidth",
  "shapeHeight", "distanceTrackToBorder", "baseThickness", "base_size",
  "building_scale",
];

// ── State ──────────────────────────────────────────────────────────────────
let tiles = [makeTile(0)];
let activeTileId = 0;
let nextTileId = 1;
let cacheKey = 0;

function makeTile(id) {
  return { id, path: null, settings: { ...DEFAULT_SETTINGS }, fileHash: [], file: [] };
}

function activeTile() {
  return tiles.find(t => t.id === activeTileId);
}

// ── DOM references ─────────────────────────────────────────────────────────
const dropzone        = document.getElementById("dropzone");
const fileInput       = document.getElementById("fileInput");
const addTrailDrop    = document.getElementById("addTrailDropzone");
const addTrailInput   = document.getElementById("addTrailInput");
const tilesContainer  = document.getElementById("tilesContainer");
const viewerCanvas    = document.getElementById("viewerCanvas");
const loadingOverlay  = document.getElementById("loadingOverlay");
const loadingMessage  = document.getElementById("loadingMessage");
const errorMessage    = document.getElementById("errorMessage");
const errorText      = document.getElementById("errorText");
const uploadedList    = document.getElementById("uploadedFilesList");

// ── Error banner ───────────────────────────────────────────────────────────
function showError(msg) {
  errorText.textContent = msg;
  errorMessage.hidden = false;
}
document.getElementById("errorClose").addEventListener("click", () => {
  errorMessage.hidden = true;
});

// ── Settings panel ↔ active tile ───────────────────────────────────────────
function setSetting(key, value) {
  const tile = activeTile();
  let v = value;
  if (NUMERIC_KEYS.includes(key)) {
    v = String(value).includes(".") ? parseFloat(value) : parseInt(value);
    if (isNaN(v)) v = 0;
  }
  tile.settings = { ...tile.settings, [key]: v };
  syncPanel(tile.settings);
}

function syncPanel(s) {
  for (const id of COLOR_KEYS) {
    document.getElementById(id).value = s[id];
  }
  for (const id of CHECKBOX_KEYS) {
    document.getElementById(id).checked = !!s[id];
  }
  document.querySelectorAll(".border-label").forEach(inp => {
    inp.value = (s.borderLabels ?? [])[Number(inp.dataset.side)] ?? "";
  });

  // Paired range+number inputs
  document.querySelectorAll("[data-setting]").forEach(inp => {
    inp.value = s[inp.dataset.setting];
  });

  // Center radios
  document.querySelectorAll("input[name='center']").forEach(r => {
    r.checked = r.value === s.center;
  });

  // Shape buttons
  document.querySelectorAll(".shape-button").forEach(b => {
    b.classList.toggle("selected", b.dataset.shape === s.shape);
  });

  // Conditional buildings extras (site renders these only when enabled)
  document.getElementById("buildingsExtra").hidden = !s.buildings;
}

function bindPanel() {
  for (const id of COLOR_KEYS) {
    document.getElementById(id).addEventListener("input", e => setSetting(id, e.target.value));
  }
  for (const id of CHECKBOX_KEYS) {
    document.getElementById(id).addEventListener("change", e => setSetting(id, e.target.checked));
  }
  document.querySelectorAll(".border-label").forEach(inp => {
    inp.addEventListener("input", e => {
      const tile = activeTile();
      const labels = [...(tile.settings.borderLabels ?? ["", "", "", "", "", ""])];
      labels[Number(inp.dataset.side)] = e.target.value;
      tile.settings = { ...tile.settings, borderLabels: labels };
    });
  });
  document.querySelectorAll("[data-setting]").forEach(inp => {
    inp.addEventListener("input", e => setSetting(inp.dataset.setting, e.target.value));
  });
  document.querySelectorAll("input[name='center']").forEach(r => {
    r.addEventListener("change", () => setSetting("center", r.value));
  });
  document.querySelectorAll(".shape-button").forEach(b => {
    b.addEventListener("click", () => setSetting("shape", b.dataset.shape));
  });
}

// ── Tiles ──────────────────────────────────────────────────────────────────
function tileName(tile) {
  return tile.file.length > 0 ? tile.file[0].name : "Unnamed Trail";
}

function renderTiles() {
  tilesContainer.innerHTML = "";
  for (const tile of tiles) {
    const el = document.createElement("div");
    el.className = "tile" + (tile.id === activeTileId ? " selected" : "");
    const content = document.createElement("div");
    content.className = "tile-content";
    if (tile.path) {
      const img = document.createElement("img");
      img.className = "tile-preview";
      img.src = `${API_BASE}/public/${tile.path}/image.png?${cacheKey}`;
      img.alt = "Preview";
      content.appendChild(img);
    } else {
      const empty = document.createElement("div");
      empty.className = "tile-empty";
      empty.textContent = "Empty Tile";
      content.appendChild(empty);
    }
    const hover = document.createElement("div");
    hover.className = "tile-hover";
    hover.textContent = tileName(tile);
    content.appendChild(hover);
    el.appendChild(content);
    el.addEventListener("click", () => selectTile(tile.id));
    tilesContainer.appendChild(el);
  }

  const add = document.createElement("div");
  add.className = "tile add-tile";
  add.innerHTML = `<div class="plus-icon">+</div>`;
  add.addEventListener("click", () => {
    const tile = makeTile(nextTileId++);
    tiles.push(tile);
    selectTile(tile.id);
  });
  tilesContainer.appendChild(add);
}

function selectTile(id) {
  activeTileId = id;
  syncPanel(activeTile().settings);
  renderTiles();
  refreshViewer();
}

// ── Main pane: dropzone ↔ inline 3D viewer ─────────────────────────────────
async function refreshViewer() {
  const tile = activeTile();
  if (tile.path) {
    dropzone.style.display = "none";
    viewerCanvas.hidden = false;
    try {
      await window.Preview3D.renderJob({
        base:        `${API_BASE}/public/${tile.path}`,
        trailAmount: Math.max(1, tile.file.length),
        settings:    tile.settings,
        cacheKey,
      });
    } catch (err) {
      showError("Could not load the 3D preview: " + err.message);
    }
  } else {
    viewerCanvas.hidden = true;
    dropzone.style.display = "flex";
    window.Preview3D?.clear();
  }
}

// ── Upload / generate (site's M function) ──────────────────────────────────
async function upload(newFiles, settingsOverride = null) {
  const tile = activeTile();

  loadingOverlay.hidden = false;
  loadingMessage.textContent = "Sending data to Server";

  // Merge new files into the tile (dedupe by name), like the site does
  tile.file = [
    ...tile.file,
    ...newFiles.filter(nf => !tile.file.some(f => f.name === nf.name)),
  ];

  const fd = new FormData();
  if (tile.file.length > tile.fileHash.length || tile.fileHash.length === 0) {
    tile.file.forEach(f => fd.append("file", f));
  } else {
    fd.append("fileHash", JSON.stringify(tile.fileHash));
  }
  fd.append("settings", JSON.stringify(settingsOverride || tile.settings));

  try {
    const resp = await fetch(`${API_BASE}/upload`, { method: "POST", body: fd });
    if (!resp.ok || !resp.body) {
      showError("Error uploading the file. " + resp.status);
      return;
    }

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line) continue;
        try {
          const msg = JSON.parse(line);
          if (msg.type === "info") {
            loadingMessage.textContent = msg.message;
          } else if (msg.type === "path") {
            tile.path = msg.path;
            tile.fileHash = msg.fileHash;
          } else if (msg.type === "error") {
            showError(msg.message);
          } else {
            showError("Server Error");
          }
        } catch (err) {
          showError("Server Error");
          console.error("Error parsing a JSON chunk:", err, line);
        }
      }
    }
  } catch (err) {
    showError("Error sending the request: " + err);
  } finally {
    loadingOverlay.hidden = true;
    loadingMessage.textContent = "";
    cacheKey += 1;
    clearUploadedList();
    renderTiles();
    refreshViewer();
  }
}

// ── Dropzones ──────────────────────────────────────────────────────────────
function bindDropzone(zone, input, onFiles) {
  zone.addEventListener("click", () => input.click());
  ["dragover", "dragenter"].forEach(evt =>
    zone.addEventListener(evt, e => { e.preventDefault(); zone.classList.add("active"); })
  );
  ["dragleave", "dragend"].forEach(evt =>
    zone.addEventListener(evt, () => zone.classList.remove("active"))
  );
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("active");
    const files = Array.from(e.dataTransfer.files).filter(f => /\.gpx$/i.test(f.name));
    if (files.length) onFiles(files);
  });
  input.addEventListener("change", e => {
    const files = Array.from(e.target.files);
    if (files.length) onFiles(files);
    input.value = "";
  });
}

// Main dropzone: dropping files starts generation immediately (site behavior)
bindDropzone(dropzone, fileInput, files => upload(files));

// "Add trail" dropzone: appends files to the tile; generation happens on Update
bindDropzone(addTrailDrop, addTrailInput, files => {
  const tile = activeTile();
  tile.file = [
    ...tile.file,
    ...files.filter(nf => !tile.file.some(f => f.name === nf.name)),
  ];
  const ul = uploadedList.querySelector("ul");
  for (const f of files) {
    const li = document.createElement("li");
    li.textContent = f.name;
    ul.appendChild(li);
  }
  uploadedList.hidden = false;
});

function clearUploadedList() {
  uploadedList.querySelector("ul").innerHTML = "";
  uploadedList.hidden = true;
}

// ── Buttons ────────────────────────────────────────────────────────────────
document.getElementById("updateBtn").addEventListener("click", async () => {
  const tile = activeTile();
  if (tile.file.length === 0 && tile.fileHash.length === 0) {
    showError("Please load a GPX file first.");
    return;
  }
  await upload(tile.file);
});

document.getElementById("downloadBtn").addEventListener("click", async () => {
  const tile = activeTile();
  if (!tile.path) {
    showError("Please load a GPX file and generate the map first.");
    return;
  }
  let name = tile.file[0]?.name || "unnamed";
  if (name.endsWith(".gpx")) name = name.slice(0, -4);

  try {
    const resp = await fetch(`${API_BASE}/download/${tile.path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!resp.ok) throw new Error("Download failed");

    const blob = await resp.blob();
    const url  = window.URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url;
    a.download = `${name}.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    showError("Error downloading: " + err.message);
  }
});

// Copies the active tile's settings to every tile (site behavior)
document.getElementById("applyAllBtn").addEventListener("click", () => {
  const s = { ...activeTile().settings };
  tiles = tiles.map(t => ({ ...t, settings: { ...s } }));
});

// ── Boot ───────────────────────────────────────────────────────────────────
bindPanel();
syncPanel(activeTile().settings);
renderTiles();
