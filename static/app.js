"use strict";

/**
 * Generator page logic — vanilla-JS port of the topotrail.com React
 * component ($S in their bundle), which is the source of truth:
 *   - per-tile settings + GPX file list (multiple trails per tile)
 *   - dropping a GPX starts generation immediately (POST /api/upload, NDJSON)
 *   - fileHash reuse: files upload once, re-generations send hashes only
 *   - inline STL viewer in the main pane, image.png thumbnails on tiles
 *   - Download fetches the ready-made ZIP for the tile's job path
 *
 * Structure: a small explicit store (`state`) + action functions that
 * mutate it and then call targeted render functions. No framework.
 */

const API_BASE = "/api";

// Settings schema follows topotrail.com (camelCase) with TopoTrail
// extensions. Colours are derived from the Strava-orange trail (#FC5200):
// hues rotated, lightness/saturation kept in the same family.
// This local copy is the fallback; /api/settings-schema (if available)
// is merged over it on boot.
const LOCAL_DEFAULT_SETTINGS = {
  waterColor: "#306BA6",   // complementary blue
  landColor: "#327B4B",    // forest green
  trackColor: "#FC5200",
  rockColor: "#9A877E",    // same hue as the trail, desaturated
  snowColor: "#F3EFED",    // near-white, warm cast
  snowLevel: 0,            // 0 none → 1 everything under snow
  forestLevel: 0,          // 0 mapped forests only → 1 fully grown
  heightScale: 1,
  standardizeHeight: false,   // lock total model height to 45 mm
  trailWidth: 1,
  trailHeight: 1,
  useHeightFromGpx: false,
  shape: "hexagon",
  distanceTrackToBorder: 0,   // extra margin beyond the built-in 5% minimum
  baseThickness: 15,   // mm
  includeSeas: true,
  includeLakes: true,
  includeRivers: false,
  base_size: 108,   // hexagon width, point to point (mm)
  center: "normal",
  buildings: false,
  building_scale: 1,
  buildingsColor: "#777777",
  smooth: true,            // smooth vector zone boundaries (no pixel blocks)
  printResolution: 0.2,   // mm per mesh cell: 0.1 / 0.2 / 0.4 / 0.8
  singleColor: false,
  singleColor_gap: 0.5,
  // TopoTrail extensions (not on topotrail.com): hexagon border with text
  baseColor: "#FFFFFF",
  textColor: "#000000",
  borderLabels: ["", "", "", "", "", ""],
  // order: top, upper-right, lower-right, bottom, lower-left, upper-left
};

let DEFAULT_SETTINGS = { ...LOCAL_DEFAULT_SETTINGS };

// Panel inputs whose element id === settings key.
// COLOR_KEYS are pure material properties: changing them never requires a
// regeneration (the viewer retints live); every other key affects geometry.
const COLOR_KEYS = ["waterColor", "landColor", "rockColor", "snowColor",
                    "trackColor", "buildingsColor", "baseColor", "textColor"];
const CHECKBOX_KEYS = ["useHeightFromGpx", "standardizeHeight", "smooth", "includeSeas",
                       "includeLakes", "includeRivers", "buildings"];

// Settings the site parses as numbers on change
const NUMERIC_KEYS = [
  "snowLevel", "forestLevel", "heightScale", "trailWidth", "trailHeight", "shapeWidth",
  "shapeHeight", "distanceTrackToBorder", "baseThickness", "base_size", "printResolution",
  "building_scale",
];

const STORAGE_KEY = "topotrail.tiles.v1";

// ── State store ────────────────────────────────────────────────────────────
// One source of truth. Actions mutate it, then call the render functions
// that depend on the changed slice. Tile objects keep a stable identity for
// the lifetime of the session (in-flight uploads hold references to them).
const state = {
  tiles: [],        // [tile]
  activeTileId: 0,
  nextTileId: 1,
};

function makeTile(id) {
  return {
    id,
    // path = job with 3D meshes; imagePath = latest job with a 2D map
    // (a terrain-only run has imagePath but no meshes)
    path: null,
    imagePath: null,
    settings: { ...DEFAULT_SETTINGS },
    // Trails on this tile. `file` is the in-memory File (never persisted);
    // `hash` is the server-side cache key (persisted, survives reloads).
    files: [],        // [{ name, hash: string|null, file: File|null }]
    dirty: false,     // geometry settings changed since last successful 3D run
    modelVersion: 0,  // bumped only when the 3D assets actually change
    // ── transient (never persisted) ──
    generating: false,
    stage: null,          // streamed progress message
    stageImagePath: null, // early 2D image job path from the stream
    abort: null,          // AbortController for the in-flight fetch
  };
}

function activeTile() {
  return state.tiles.find(t => t.id === state.activeTileId);
}

function tileName(tile) {
  return tile.files.length > 0 ? tile.files[0].name : "Unnamed Trail";
}

// ── Persistence ────────────────────────────────────────────────────────────
// File objects can't be persisted; we keep names + server fileHashes, and
// regenerate via the fileHash path after a reload (the backend supports it).
function persist() {
  try {
    const data = {
      v: 1,
      activeTileId: state.activeTileId,
      nextTileId: state.nextTileId,
      tiles: state.tiles.map(t => ({
        id: t.id,
        path: t.path,
        imagePath: t.imagePath,
        settings: t.settings,
        dirty: t.dirty,
        modelVersion: t.modelVersion,
        files: t.files.map(f => ({ name: f.name, hash: f.hash })),
      })),
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch (err) {
    // Quota exceeded / storage disabled — the app still works, just
    // without persistence.
    console.warn("Could not persist state:", err);
  }
}

function restore() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return false;
    const data = JSON.parse(raw);
    if (!data || data.v !== 1 || !Array.isArray(data.tiles) || data.tiles.length === 0) {
      return false;
    }
    state.tiles = data.tiles.map(t => {
      const tile = makeTile(Number.isInteger(t.id) ? t.id : 0);
      tile.path = typeof t.path === "string" ? t.path : null;
      tile.imagePath = typeof t.imagePath === "string" ? t.imagePath : null;
      tile.settings = { ...DEFAULT_SETTINGS, ...(t.settings && typeof t.settings === "object" ? t.settings : {}) };
      tile.dirty = !!t.dirty;
      tile.modelVersion = Number.isInteger(t.modelVersion) ? t.modelVersion : 0;
      tile.files = Array.isArray(t.files)
        ? t.files
            .filter(f => f && typeof f.name === "string")
            .map(f => ({ name: f.name, hash: typeof f.hash === "string" ? f.hash : null, file: null }))
        : [];
      return tile;
    });
    state.activeTileId = state.tiles.some(t => t.id === data.activeTileId)
      ? data.activeTileId
      : state.tiles[0].id;
    state.nextTileId = Math.max(
      Number.isInteger(data.nextTileId) ? data.nextTileId : 1,
      ...state.tiles.map(t => t.id + 1),
    );
    return true;
  } catch (err) {
    console.warn("Could not restore saved state:", err);
    return false;
  }
}

// ── DOM references ─────────────────────────────────────────────────────────
const dropzone        = document.getElementById("dropzone");
const fileInput       = document.getElementById("fileInput");
const addTrailDrop    = document.getElementById("addTrailDropzone");
const addTrailInput   = document.getElementById("addTrailInput");
const tilesContainer  = document.getElementById("tilesContainer");
const viewerCanvas    = document.getElementById("viewerCanvas");
const mapPreview      = document.getElementById("mapPreview");
const errorMessage    = document.getElementById("errorMessage");
const errorText       = document.getElementById("errorText");
const uploadedList    = document.getElementById("uploadedFilesList");
const progressStrip   = document.getElementById("progressStrip");
const progressStage   = document.getElementById("progressStage");
const progressImg     = document.getElementById("progressMapPreview");
const cancelBtn       = document.getElementById("cancelBtn");
const terrainBtn      = document.getElementById("terrainBtn");
const updateBtn       = document.getElementById("updateBtn");
const downloadBtn     = document.getElementById("downloadBtn");
const dirtyHint       = document.getElementById("dirtyHint");

// ── Error banner ───────────────────────────────────────────────────────────
function showError(msg) {
  errorText.textContent = msg;
  errorMessage.hidden = false;
}

// Map raw HTTP failures to friendly messages (raw detail goes to console)
function friendlyHttpError(status) {
  if (status === 410) return "The cached GPX expired — please re-add the file.";
  if (status >= 500) return "Generation failed on the server. Please try again.";
  return `Upload failed (HTTP ${status}).`;
}

// ── Render: settings panel ─────────────────────────────────────────────────
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

  // Resolution radios
  document.querySelectorAll("input[name='printResolution']").forEach(r => {
    r.checked = Number(r.value) === Number(s.printResolution);
  });

  // Shape buttons
  document.querySelectorAll(".shape-button").forEach(b => {
    b.classList.toggle("selected", b.dataset.shape === s.shape);
    b.setAttribute("aria-pressed", String(b.dataset.shape === s.shape));
  });

  // Conditional buildings extras (site renders these only when enabled)
  document.getElementById("buildingsExtra").hidden = !s.buildings;
}

// ── Render: tiles row ──────────────────────────────────────────────────────
function renderTiles() {
  tilesContainer.innerHTML = "";
  for (const tile of state.tiles) {
    const el = document.createElement("div");
    el.className = "tile" + (tile.id === state.activeTileId ? " selected" : "");
    el.setAttribute("role", "button");
    el.tabIndex = 0;
    el.setAttribute("aria-label", `Select tile: ${tileName(tile)}`);

    const content = document.createElement("div");
    content.className = "tile-content";
    const thumbPath = tile.imagePath ?? tile.path;
    if (thumbPath) {
      const img = document.createElement("img");
      img.className = "tile-preview";
      img.src = `${API_BASE}/public/${thumbPath}/image.png?v=${tile.modelVersion}`;
      img.alt = `Preview of ${tileName(tile)}`;
      content.appendChild(img);
    } else {
      const empty = document.createElement("div");
      empty.className = "tile-empty";
      empty.textContent = "Empty Tile";
      content.appendChild(empty);
    }

    if (tile.generating) {
      const badge = document.createElement("div");
      badge.className = "tile-badge";
      badge.setAttribute("aria-label", "Generating");
      badge.title = "Generating…";
      content.appendChild(badge);
    }

    const del = document.createElement("button");
    del.type = "button";
    del.className = "tile-delete";
    del.textContent = "✕";
    del.setAttribute("aria-label", `Delete tile: ${tileName(tile)}`);
    del.title = "Delete tile";
    del.addEventListener("click", e => {
      e.stopPropagation();
      deleteTile(tile.id);
    });
    content.appendChild(del);

    const hover = document.createElement("div");
    hover.className = "tile-hover";
    hover.textContent = tileName(tile);
    content.appendChild(hover);
    el.appendChild(content);

    el.addEventListener("click", () => selectTile(tile.id));
    el.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectTile(tile.id);
      }
    });
    tilesContainer.appendChild(el);
  }

  const add = document.createElement("button");
  add.type = "button";
  add.className = "tile add-tile";
  add.setAttribute("aria-label", "Add a new tile");
  add.innerHTML = `<div class="plus-icon" aria-hidden="true">+</div>`;
  add.addEventListener("click", addTile);
  tilesContainer.appendChild(add);
}

// ── Render: trail file list (settings panel "Add trail" section) ──────────
function renderFilesList() {
  const tile = activeTile();
  const ul = uploadedList.querySelector("ul");
  ul.innerHTML = "";
  uploadedList.hidden = tile.files.length === 0;
  tile.files.forEach((entry, index) => {
    const li = document.createElement("li");
    li.className = "uploaded-file-row";
    const name = document.createElement("span");
    name.className = "uploaded-file-name";
    name.textContent = entry.name;
    li.appendChild(name);
    if (!entry.file && !entry.hash) {
      const note = document.createElement("span");
      note.className = "uploaded-file-note";
      note.textContent = "(re-add needed)";
      li.appendChild(note);
    }
    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "uploaded-file-remove";
    rm.textContent = "✕";
    rm.setAttribute("aria-label", `Remove ${entry.name}`);
    rm.title = "Remove file";
    rm.addEventListener("click", () => removeFile(tile, index));
    li.appendChild(rm);
    ul.appendChild(li);
  });
}

// ── Render: action buttons + dirty state ───────────────────────────────────
function renderButtons() {
  const tile = activeTile();
  const busy = tile.generating;

  terrainBtn.disabled = busy;
  updateBtn.disabled = busy;
  downloadBtn.disabled = !tile.path;

  const showDirty = !!tile.dirty && !!tile.path;
  updateBtn.classList.toggle("dirty", showDirty && !busy);
  downloadBtn.classList.toggle("stale", showDirty);
  downloadBtn.title = showDirty
    ? "Settings changed since this model was generated — the download is the previous version."
    : "Download the generated model as a ZIP of STL files";
  dirtyHint.hidden = !showDirty;
}

// ── Render: progress strip (non-blocking, per active tile) ────────────────
function renderProgress() {
  const tile = activeTile();
  progressStrip.hidden = !tile.generating;
  if (!tile.generating) return;

  progressStage.textContent = tile.stage || "Working…";
  if (tile.stageImagePath && progressImg.dataset.path !== tile.stageImagePath) {
    // The 2D map is ready before the 3D build — show it right away
    progressImg.dataset.path = tile.stageImagePath;
    progressImg.src = `${API_BASE}/public/${tile.stageImagePath}/image.png?${Date.now()}`;
    progressImg.hidden = false;
  } else if (!tile.stageImagePath) {
    progressImg.hidden = true;
    progressImg.removeAttribute("src");
    delete progressImg.dataset.path;
  }
}

// ── Render: main pane (dropzone ↔ inline 3D viewer + minimap) ─────────────
// Tracks the last (path, modelVersion) actually handed to the 3D viewer so
// a terrain-only run — or a plain tile re-select — never re-downloads and
// rebuilds unchanged zone STLs.
let lastRendered = { path: null, version: -1 };

function whenPreview3DReady() {
  if (window.Preview3D) return Promise.resolve();
  return new Promise(resolve =>
    window.addEventListener("preview3d-ready", resolve, { once: true }));
}

async function refreshViewer() {
  const tile = activeTile();

  // Minimap: latest 2D map (terrain-only runs included). Its colours are
  // baked server-side, so it only updates on regeneration — acceptable.
  const imagePath = tile.imagePath ?? tile.path;
  mapPreview.hidden = !imagePath;
  if (imagePath) {
    mapPreview.src = `${API_BASE}/public/${imagePath}/image.png?v=${tile.modelVersion}`;
  }

  if (tile.path) {
    dropzone.style.display = "none";
    viewerCanvas.hidden = false;

    if (lastRendered.path === tile.path && lastRendered.version === tile.modelVersion) {
      // Same 3D job already on screen — skip the expensive STL rebuild,
      // but make sure the (cheap) material colours match current settings.
      window.Preview3D?.applyColors?.(tile.settings);
      return;
    }

    try {
      await whenPreview3DReady();
      const committed = await window.Preview3D.renderJob({
        base:        `${API_BASE}/public/${tile.path}`,
        trailAmount: Math.max(1, tile.files.length),
        settings:    tile.settings,
        cacheKey:    tile.modelVersion,
      });
      if (committed) {
        lastRendered = { path: tile.path, version: tile.modelVersion };
      }
    } catch (err) {
      console.error("3D preview failed:", err);
      showError("Could not load the 3D preview: " + err.message);
    }
  } else {
    viewerCanvas.hidden = true;
    dropzone.style.display = "flex";
    window.Preview3D?.clear();
    lastRendered = { path: null, version: -1 };
  }
}

function renderAll() {
  syncPanel(activeTile().settings);
  renderTiles();
  renderFilesList();
  renderButtons();
  renderProgress();
  refreshViewer();
}

// ── Actions ────────────────────────────────────────────────────────────────
function selectTile(id) {
  if (!state.tiles.some(t => t.id === id)) return;
  state.activeTileId = id;
  persist();
  renderAll();
}

function addTile() {
  const tile = makeTile(state.nextTileId++);
  state.tiles.push(tile);
  selectTile(tile.id);
}

function deleteTile(id) {
  const tile = state.tiles.find(t => t.id === id);
  if (!tile) return;
  if (tile.path || tile.imagePath) {
    const ok = window.confirm(`Delete "${tileName(tile)}"? Its generated model will be removed from this page.`);
    if (!ok) return;
  }
  tile.abort?.abort();
  state.tiles = state.tiles.filter(t => t.id !== id);
  if (state.tiles.length === 0) {
    // Always keep at least one tile: deleting the last one resets it.
    state.tiles = [makeTile(state.nextTileId++)];
  }
  if (state.activeTileId === id) {
    state.activeTileId = state.tiles[0].id;
  }
  persist();
  renderAll();
}

function resetAll() {
  const ok = window.confirm("Reset everything? All tiles, settings and generated models on this page will be cleared.");
  if (!ok) return;
  for (const t of state.tiles) t.abort?.abort();
  try { localStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
  state.tiles = [makeTile(0)];
  state.activeTileId = 0;
  state.nextTileId = 1;
  renderAll();
}

function markDirty(tile) {
  // Only meaningful once a 3D model exists to be out of date.
  if (tile.path && !tile.dirty) {
    tile.dirty = true;
  }
}

function setSetting(key, value) {
  const tile = activeTile();
  let v = value;
  if (NUMERIC_KEYS.includes(key)) {
    v = String(value).includes(".") ? parseFloat(value) : parseInt(value);
    if (isNaN(v)) v = 0;
  }
  tile.settings = { ...tile.settings, [key]: v };

  if (COLOR_KEYS.includes(key)) {
    // Colours are live material updates — never dirty, never regenerate.
    if (tile.path) window.Preview3D?.applyColors?.(tile.settings);
  } else {
    markDirty(tile);
  }

  persist();
  syncPanel(tile.settings);
  renderButtons();
}

// True when any geometry-affecting (non-colour) setting differs
function geometryDiffers(a, b) {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const k of keys) {
    if (COLOR_KEYS.includes(k)) continue;
    if (JSON.stringify(a[k]) !== JSON.stringify(b[k])) return true;
  }
  return false;
}

function applySettingsToAll() {
  const src = activeTile();
  for (const t of state.tiles) {
    if (t === src) continue;
    if (geometryDiffers(t.settings, src.settings)) markDirty(t);
    t.settings = { ...src.settings, borderLabels: [...(src.settings.borderLabels ?? [])] };
  }
  persist();
  renderButtons();
}

// Merge files into the tile. Re-adding a name that lost its File object
// (page reload, or an expired hash) re-attaches the File and clears the
// stale hash so it re-uploads.
function addFiles(tile, files) {
  let changed = false;
  for (const f of files) {
    const existing = tile.files.find(e => e.name === f.name);
    if (existing) {
      if (!existing.file) {
        existing.file = f;
        existing.hash = null;
        changed = true;
      }
    } else {
      tile.files.push({ name: f.name, hash: null, file: f });
      changed = true;
    }
  }
  if (changed) markDirty(tile);
  persist();
  renderFilesList();
  renderTiles();
  renderButtons();
  return changed;
}

function removeFile(tile, index) {
  if (index < 0 || index >= tile.files.length) return;
  tile.files.splice(index, 1);
  markDirty(tile);
  persist();
  renderFilesList();
  renderTiles();
  renderButtons();
}

// ── Generation (site's M function) ────────────────────────────────────────
// terrainOnly=true hits /api/preview: just the fast 2D map, no 3D meshes.
async function generate(tile, { terrainOnly = false } = {}) {
  if (tile.generating) {
    showError("A generation is already running for this tile.");
    return;
  }
  if (tile.files.length === 0) {
    showError("Please load a GPX file first.");
    return;
  }

  // Decide payload: either every trail by server-side hash, or every trail
  // as a File. Mixing isn't part of the API contract.
  const fd = new FormData();
  let sentFiles = false;
  if (tile.files.every(e => e.hash)) {
    fd.append("fileHash", JSON.stringify(tile.files.map(e => e.hash)));
  } else if (tile.files.every(e => e.file)) {
    tile.files.forEach(e => fd.append("file", e.file));
    sentFiles = true;
  } else {
    const missing = tile.files.filter(e => !e.file).map(e => e.name).join(", ");
    showError(`Please re-add these GPX files before generating with new ones: ${missing}`);
    return;
  }
  fd.append("settings", JSON.stringify(tile.settings));

  tile.generating = true;
  tile.stage = "Sending data to Server";
  tile.stageImagePath = null;
  const controller = new AbortController();
  tile.abort = controller;
  renderTiles();
  renderButtons();
  renderProgress();

  const isActive = () => tile.id === state.activeTileId;
  let got3d = false;

  try {
    const endpoint = terrainOnly ? "/preview" : "/upload";
    const resp = await fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      body: fd,
      signal: controller.signal,
    });
    if (!resp.ok || !resp.body) {
      console.error("Upload request failed:", resp.status, resp.statusText);
      showError(friendlyHttpError(resp.status));
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
        let msg;
        try {
          msg = JSON.parse(line);
        } catch (err) {
          console.error("Error parsing a JSON chunk:", err, line);
          continue;
        }
        if (msg.type === "info") {
          tile.stage = msg.message;
          if (isActive()) renderProgress();
        } else if (msg.type === "image") {
          tile.stageImagePath = msg.path;
          if (isActive()) renderProgress();
        } else if (msg.type === "path") {
          if (!terrainOnly) {
            tile.path = msg.path;
            tile.modelVersion += 1;   // 3D assets actually changed
            got3d = true;
          }
          tile.imagePath = msg.path;
          if (Array.isArray(msg.fileHash)) {
            // Hashes come back in the order the files were sent, which is
            // the order of tile.files (both for file and fileHash sends).
            msg.fileHash.forEach((h, i) => {
              if (tile.files[i]) tile.files[i].hash = h;
            });
          }
        } else if (msg.type === "error") {
          showError(msg.message);
        } else {
          console.warn("Ignoring unknown NDJSON message type:", msg);
        }
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      // User hit Cancel: we just stop listening to the stream. The server
      // keeps working and finishes the job server-side — that's fine, we
      // simply won't pick up its result.
    } else {
      console.error("Generation request failed:", err);
      showError("Could not reach the server. Please check your connection and try again.");
    }
  } finally {
    tile.generating = false;
    tile.abort = null;
    tile.stage = null;
    tile.stageImagePath = null;
    if (got3d) tile.dirty = false;   // model now matches settings
    persist();
    renderTiles();
    renderButtons();
    renderProgress();
    if (isActive()) {
      renderFilesList();
      refreshViewer();
    }
  }
}

// ── Dropzones ──────────────────────────────────────────────────────────────
function bindDropzone(zone, input, onFiles) {
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });
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

// ── Buttons ────────────────────────────────────────────────────────────────
function bindButtons() {
  document.getElementById("errorClose").addEventListener("click", () => {
    errorMessage.hidden = true;
  });

  // Full 3D generation
  updateBtn.addEventListener("click", () => generate(activeTile()));

  // Fast 2D terrain preview only — no 3D model is built
  terrainBtn.addEventListener("click", () => generate(activeTile(), { terrainOnly: true }));

  cancelBtn.addEventListener("click", () => {
    activeTile().abort?.abort();
  });

  downloadBtn.addEventListener("click", async () => {
    const tile = activeTile();
    if (!tile.path) {
      showError("Please load a GPX file and generate the map first.");
      return;
    }
    let name = tile.files[0]?.name || "unnamed";
    if (name.endsWith(".gpx")) name = name.slice(0, -4);

    try {
      const resp = await fetch(`${API_BASE}/download/${tile.path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

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
      console.error("Download failed:", err);
      showError("Could not download the model. Please try again.");
    }
  });

  // Copies the active tile's settings to every tile (site behavior)
  document.getElementById("applyAllBtn").addEventListener("click", applySettingsToAll);

  document.getElementById("resetAllBtn").addEventListener("click", resetAll);
}

// ── Settings panel bindings ────────────────────────────────────────────────
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
      markDirty(tile);   // border text is baked into geometry
      persist();
      renderButtons();
    });
  });
  document.querySelectorAll("[data-setting]").forEach(inp => {
    inp.addEventListener("input", e => setSetting(inp.dataset.setting, e.target.value));
  });
  document.querySelectorAll("input[name='center']").forEach(r => {
    r.addEventListener("change", () => setSetting("center", r.value));
  });
  document.querySelectorAll("input[name='printResolution']").forEach(r => {
    r.addEventListener("change", () => setSetting("printResolution", r.value));
  });
  document.querySelectorAll(".shape-button").forEach(b => {
    b.addEventListener("click", () => setSetting("shape", b.dataset.shape));
  });
}

// ── Boot ───────────────────────────────────────────────────────────────────
async function fetchSettingsSchema() {
  // The backend may expose GET /api/settings-schema with its
  // DEFAULT_SETTINGS. Use it when available; the local copy is always the
  // fallback (the endpoint can 404 — never block boot on it).
  try {
    const resp = await fetch(`${API_BASE}/settings-schema`);
    if (!resp.ok) return;
    const schema = await resp.json();
    if (schema && typeof schema === "object" && !Array.isArray(schema)) {
      DEFAULT_SETTINGS = { ...LOCAL_DEFAULT_SETTINGS, ...schema };
    }
  } catch (err) {
    console.warn("settings-schema unavailable, using local defaults:", err);
  }
}

async function boot() {
  bindPanel();
  bindButtons();

  // Main dropzone: dropping files starts generation immediately (site behavior)
  bindDropzone(dropzone, fileInput, files => {
    const tile = activeTile();
    if (tile.generating) {
      showError("A generation is already running for this tile.");
      return;
    }
    addFiles(tile, files);
    generate(tile);
  });

  // "Add trail" dropzone: appends files; generation happens on Generate 3D
  bindDropzone(addTrailDrop, addTrailInput, files => {
    addFiles(activeTile(), files);
  });

  await fetchSettingsSchema();

  if (!restore()) {
    state.tiles = [makeTile(0)];
    state.activeTileId = 0;
    state.nextTileId = 1;
  }
  renderAll();
}

boot();
