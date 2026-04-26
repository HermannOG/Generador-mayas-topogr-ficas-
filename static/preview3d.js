/**
 * preview3d.js – Three.js 3D terrain preview
 * Colors terrain with user-configured land/rock colors, white background.
 * Exposes: window.Preview3D = { render, show, hide, getData }
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ── Scene state ───────────────────────────────────────────────────────────────
let renderer, scene, camera, controls;
let animId = null;
let currentData = null;
let wireframeActive = false;

// ── Init ──────────────────────────────────────────────────────────────────────
function initScene() {
  const canvas = document.getElementById("previewCanvas");

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf0f5fa);   // light grey-white, matches app bg

  camera = new THREE.PerspectiveCamera(45, 1, 0.1, 2000);
  camera.position.set(0, -90, 70);
  camera.lookAt(0, 0, 0);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 10;
  controls.maxDistance = 500;
  controls.target.set(0, 0, 8);
  controls.update();

  addLights();
}

function addLights() {
  // Soft fill from above
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));

  // Main sun – angled from NW for good topographic shading
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(-80, -30, 60);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far  = 600;
  sun.shadow.camera.left = sun.shadow.camera.bottom = -120;
  sun.shadow.camera.right = sun.shadow.camera.top  =  120;
  scene.add(sun);

  // Subtle back-fill to lift shadows slightly
  const fill = new THREE.DirectionalLight(0xd0e8ff, 0.3);
  fill.position.set(60, 60, 40);
  scene.add(fill);
}

// ── Resize ────────────────────────────────────────────────────────────────────
function resizeRenderer() {
  const canvas = renderer.domElement;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

// ── Color helpers ─────────────────────────────────────────────────────────────
function hexToRGB01(hex) {
  const n = parseInt(hex.replace("#", ""), 16);
  return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
}

function smoothstep(lo, hi, x) {
  const t = Math.max(0, Math.min(1, (x - lo) / Math.max(hi - lo, 1e-6)));
  return t * t * (3 - 2 * t);
}

/**
 * Map elevation → RGB using the user's land/rock colour setting.
 *  - below tree line           → land colour
 *  - above tree line           → rock colour
 *  - smooth blend ±5 % around tree-line fraction
 */
function elevationColor(ele, ele_min, ele_max, treeLine, landHex, rockHex) {
  const range = Math.max(ele_max - ele_min, 1);
  const t     = (ele     - ele_min) / range;
  const tT    = (treeLine - ele_min) / range;

  const blend = smoothstep(tT - 0.04, tT + 0.04, t);

  const land = hexToRGB01(landHex);
  const rock = hexToRGB01(rockHex);
  return [
    land[0] + (rock[0] - land[0]) * blend,
    land[1] + (rock[1] - land[1]) * blend,
    land[2] + (rock[2] - land[2]) * blend,
  ];
}

// ── Build terrain mesh ────────────────────────────────────────────────────────
function buildTerrain(data) {
  const rows = data.grid.length;
  const cols = data.grid[0].length;
  const { ele_min, ele_max, lat_bounds, lon_bounds } = data;
  const treeLine  = data.settings?.tree_line  ?? 1250;
  const landColor = data.settings?.land_color  ?? "#00cc00";
  const rockColor = data.settings?.rock_color  ?? "#aaaaaa";

  // Keep physical aspect ratio
  const latRange = lat_bounds[1] - lat_bounds[0];
  const lonRange = lon_bounds[1] - lon_bounds[0];
  const latMid   = (lat_bounds[0] + lat_bounds[1]) / 2;
  const cosLat   = Math.cos((latMid * Math.PI) / 180);
  const aspect   = (latRange * 111000) / (lonRange * 111000 * cosLat);

  const W    = 100;
  const H    = W * aspect;
  const MAX_Z = 28;

  const positions = new Float32Array(rows * cols * 3);
  const colorsArr = new Float32Array(rows * cols * 3);
  const indices   = [];

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const idx3 = (i * cols + j) * 3;
      const ele  = data.grid[i][j];

      positions[idx3]     = (j / (cols - 1)) * W - W / 2;
      positions[idx3 + 1] = (1 - i / (rows - 1)) * H - H / 2;   // N at top
      positions[idx3 + 2] = ((ele - ele_min) / Math.max(ele_max - ele_min, 1)) * MAX_Z;

      const [r, g, b] = elevationColor(ele, ele_min, ele_max, treeLine, landColor, rockColor);
      colorsArr[idx3]     = r;
      colorsArr[idx3 + 1] = g;
      colorsArr[idx3 + 2] = b;
    }
  }

  for (let i = 0; i < rows - 1; i++) {
    for (let j = 0; j < cols - 1; j++) {
      const a = i * cols + j,       b = i * cols + j + 1;
      const c = (i + 1) * cols + j, d = (i + 1) * cols + j + 1;
      indices.push(a, c, d,  a, d, b);
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute("color",    new THREE.Float32BufferAttribute(colorsArr, 3));
  geom.setIndex(indices);
  geom.computeVertexNormals();

  const mat  = new THREE.MeshLambertMaterial({ vertexColors: true });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.receiveShadow = true;
  mesh.castShadow    = true;

  // Base plate – same colour as land (matches the reference image)
  const [lr, lg, lb] = hexToRGB01(landColor);
  const baseGeom = new THREE.BoxGeometry(W, H, 1.0);
  const baseMat  = new THREE.MeshLambertMaterial({
    color: new THREE.Color(lr, lg, lb),
  });
  const base = new THREE.Mesh(baseGeom, baseMat);
  base.position.z = -0.5;

  const group = new THREE.Group();
  group.add(mesh, base);
  return { group, W, H, MAX_Z };
}

// ── GPX route line ────────────────────────────────────────────────────────────
function buildRoute(data, W, H, MAX_Z) {
  if (!data.gpx_points || data.gpx_points.length < 2) return null;

  const rows = data.grid.length, cols = data.grid[0].length;
  const { ele_min, ele_max, lat_bounds, lon_bounds } = data;
  const eleRange = Math.max(ele_max - ele_min, 1);
  const latRange = lat_bounds[1] - lat_bounds[0];
  const lonRange = lon_bounds[1] - lon_bounds[0];

  const pts = [];
  for (const [lat, lon] of data.gpx_points) {
    const fx = (lon - lon_bounds[0]) / lonRange;
    const fy = (lat - lat_bounds[0]) / latRange;
    const gi = Math.min(rows-1, Math.max(0, Math.round((1-fy) * (rows-1))));
    const gj = Math.min(cols-1, Math.max(0, Math.round(fx    * (cols-1))));
    const z  = ((data.grid[gi][gj] - ele_min) / eleRange) * MAX_Z + 0.9;
    pts.push(new THREE.Vector3(fx*W - W/2, fy*H - H/2, z));
  }

  const color      = data.settings?.trail_color ?? "#ff0000";
  const routeGeom  = new THREE.BufferGeometry().setFromPoints(pts);
  const routeMat   = new THREE.LineBasicMaterial({ color, linewidth: 2 });
  return new THREE.Line(routeGeom, routeMat);
}

// ── Start / end markers ───────────────────────────────────────────────────────
function buildMarkers(data, W, H, MAX_Z) {
  const pts = data.gpx_points;
  if (!pts || pts.length < 1) return null;

  const rows = data.grid.length, cols = data.grid[0].length;
  const { ele_min, ele_max, lat_bounds, lon_bounds } = data;
  const eleRange = Math.max(ele_max - ele_min, 1);
  const latRange = lat_bounds[1] - lat_bounds[0];
  const lonRange = lon_bounds[1] - lon_bounds[0];

  function sphere(pt, color) {
    const fx = (pt[1] - lon_bounds[0]) / lonRange;
    const fy = (pt[0] - lat_bounds[0]) / latRange;
    const gi = Math.min(rows-1, Math.max(0, Math.round((1-fy)*(rows-1))));
    const gj = Math.min(cols-1, Math.max(0, Math.round(fx*(cols-1))));
    const z  = ((data.grid[gi][gj] - ele_min) / eleRange) * MAX_Z + 2;
    const m  = new THREE.Mesh(
      new THREE.SphereGeometry(1.4, 14, 10),
      new THREE.MeshLambertMaterial({ color })
    );
    m.position.set(fx*W - W/2, fy*H - H/2, z);
    return m;
  }

  const g = new THREE.Group();
  g.add(sphere(pts[0],            0x22dd55));   // start: green
  g.add(sphere(pts[pts.length-1], 0xff2222));   // end:   red
  return g;
}

// ── Render preview data ───────────────────────────────────────────────────────
function renderPreview(data) {
  // Keep lights, clear geometry
  for (const child of [...scene.children]) {
    if (!child.isLight) scene.remove(child);
  }
  wireframeActive = false;
  document.getElementById("wireframeToggle")?.classList.remove("active");

  const { group, W, H, MAX_Z } = buildTerrain(data);
  scene.add(group);

  const route = buildRoute(data, W, H, MAX_Z);
  if (route) scene.add(route);

  const markers = buildMarkers(data, W, H, MAX_Z);
  if (markers) scene.add(markers);

  // Position camera to see the whole model at ~30° elevation angle
  const diag = Math.sqrt(W*W + H*H);
  camera.position.set(0, -diag * 0.7, diag * 0.5);
  controls.target.set(0, 0, MAX_Z * 0.25);
  controls.update();

  currentData = data;
}

// ── Animation loop ────────────────────────────────────────────────────────────
function startAnimate() {
  if (animId) return;
  (function loop() {
    animId = requestAnimationFrame(loop);
    resizeRenderer();
    controls.update();
    renderer.render(scene, camera);
  })();
}

function stopAnimate() {
  if (animId) { cancelAnimationFrame(animId); animId = null; }
}

// ── Wireframe toggle ──────────────────────────────────────────────────────────
function toggleWireframe() {
  wireframeActive = !wireframeActive;
  scene.traverse(obj => {
    if (obj.isMesh && obj.geometry?.attributes?.color) {
      obj.material.wireframe = wireframeActive;
    }
  });
  document.getElementById("wireframeToggle")?.classList.toggle("active", wireframeActive);
}

// ── Show / hide ───────────────────────────────────────────────────────────────
function showModal() {
  document.getElementById("previewModal").hidden = false;
  setTimeout(() => { resizeRenderer(); startAnimate(); }, 50);
}

function hideModal() {
  stopAnimate();
  document.getElementById("previewModal").hidden = true;
}

// ── Boot ──────────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  initScene();

  document.getElementById("closePreview")
    .addEventListener("click", hideModal);

  document.getElementById("wireframeToggle")
    .addEventListener("click", toggleWireframe);

  // Click on dark overlay also closes
  document.getElementById("previewBg")
    .addEventListener("click", hideModal);

  window.addEventListener("keydown", e => {
    if (e.key === "Escape") hideModal();
  });
});

// ── Public API ────────────────────────────────────────────────────────────────
window.Preview3D = {
  render:  renderPreview,
  show:    showModal,
  hide:    hideModal,
  getData: () => currentData,
};
