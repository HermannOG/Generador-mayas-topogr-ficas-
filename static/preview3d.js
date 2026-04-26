/**
 * preview3d.js – Three.js 3D terrain preview module
 * Exposes window.Preview3D = { render, show, hide, getData }
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ── Scene state ───────────────────────────────────────────────────────────
let renderer, scene, camera, controls;
let animId = null;
let currentData = null;

// ── Init (called once on DOMContentLoaded) ────────────────────────────────
function initScene() {
  const canvas = document.getElementById("previewCanvas");

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x87ceeb);
  scene.fog = new THREE.FogExp2(0x87ceeb, 0.006);

  camera = new THREE.PerspectiveCamera(45, 1, 0.1, 2000);
  camera.position.set(0, -90, 70);
  camera.lookAt(0, 0, 0);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 10;
  controls.maxDistance = 400;
  controls.target.set(0, 0, 8);
  controls.update();

  addLights();
}

function addLights() {
  const ambient = new THREE.AmbientLight(0xffffff, 0.55);
  scene.add(ambient);

  const sun = new THREE.DirectionalLight(0xfff4e0, 1.3);
  sun.position.set(-60, -40, 100);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  scene.add(sun);

  const fill = new THREE.DirectionalLight(0xb0d4ff, 0.4);
  fill.position.set(60, 60, 30);
  scene.add(fill);
}

// ── Resize ────────────────────────────────────────────────────────────────
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

// ── Terrain color by elevation ────────────────────────────────────────────
function elevationColor(ele, ele_min, ele_max, treeLine) {
  const range = Math.max(ele_max - ele_min, 1);
  const t = (ele - ele_min) / range;
  const tT = Math.max(0, Math.min(1, (treeLine - ele_min) / range));

  const lerp = (a, b, s) => a + (b - a) * Math.max(0, Math.min(1, s));
  const lerpRGB = (c1, c2, s) => [
    lerp(c1[0], c2[0], s),
    lerp(c1[1], c2[1], s),
    lerp(c1[2], c2[2], s),
  ];

  // Palette: deep-green → mid-green → brown → rock-grey → snow
  const deepGreen = [0.22, 0.55, 0.20];
  const midGreen  = [0.30, 0.62, 0.25];
  const brown     = [0.55, 0.43, 0.25];
  const rock      = [0.55, 0.53, 0.52];
  const snow      = [0.93, 0.95, 0.98];

  let c;
  if (t < tT * 0.4)      c = lerpRGB(deepGreen, midGreen, t / Math.max(tT * 0.4, 0.001));
  else if (t < tT)       c = lerpRGB(midGreen,  brown,   (t - tT * 0.4) / Math.max(tT * 0.6, 0.001));
  else if (t < tT + 0.2) c = lerpRGB(brown,     rock,    (t - tT) / 0.2);
  else                   c = lerpRGB(rock,       snow,    (t - tT - 0.2) / Math.max(1 - tT - 0.2, 0.1));

  return c;
}

// ── Build terrain mesh ────────────────────────────────────────────────────
function buildTerrain(data) {
  const rows = data.grid.length;
  const cols = data.grid[0].length;
  const { ele_min, ele_max, lat_bounds, lon_bounds } = data;
  const treeLine = data.settings?.tree_line ?? 1250;

  const latRange = lat_bounds[1] - lat_bounds[0];
  const lonRange = lon_bounds[1] - lon_bounds[0];
  const latMid = (lat_bounds[0] + lat_bounds[1]) / 2;
  const cosLat = Math.cos((latMid * Math.PI) / 180);
  const aspect = (latRange * 111000) / (lonRange * 111000 * cosLat);

  const W = 100;
  const H = W * aspect;
  const MAX_Z = 28;

  const positions = new Float32Array(rows * cols * 3);
  const colorsArr = new Float32Array(rows * cols * 3);
  const indices = [];

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const idx3 = (i * cols + j) * 3;
      const ele = data.grid[i][j];

      positions[idx3]     = (j / (cols - 1)) * W - W / 2;
      positions[idx3 + 1] = (1 - i / (rows - 1)) * H - H / 2;   // flip Y: south at bottom
      positions[idx3 + 2] = ((ele - ele_min) / Math.max(ele_max - ele_min, 1)) * MAX_Z;

      const [r, g, b] = elevationColor(ele, ele_min, ele_max, treeLine);
      colorsArr[idx3]     = r;
      colorsArr[idx3 + 1] = g;
      colorsArr[idx3 + 2] = b;
    }
  }

  for (let i = 0; i < rows - 1; i++) {
    for (let j = 0; j < cols - 1; j++) {
      const a = i * cols + j,     b = i * cols + j + 1;
      const c = (i + 1) * cols + j, d = (i + 1) * cols + j + 1;
      indices.push(a, c, d,  a, d, b);
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute("color",    new THREE.Float32BufferAttribute(colorsArr, 3));
  geom.setIndex(indices);
  geom.computeVertexNormals();

  const mat = new THREE.MeshLambertMaterial({ vertexColors: true });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.receiveShadow = true;
  mesh.castShadow = true;

  // Solid base plate
  const baseGeom = new THREE.BoxGeometry(W, H, 0.8);
  const baseMat  = new THREE.MeshLambertMaterial({ color: 0x6b5030 });
  const base     = new THREE.Mesh(baseGeom, baseMat);
  base.position.z = -0.4;

  const group = new THREE.Group();
  group.add(mesh, base);
  return { group, W, H, MAX_Z };
}

// ── Build GPX route line ──────────────────────────────────────────────────
function buildRoute(data, W, H, MAX_Z) {
  if (!data.gpx_points || data.gpx_points.length < 2) return null;

  const rows = data.grid.length;
  const cols = data.grid[0].length;
  const { ele_min, ele_max, lat_bounds, lon_bounds } = data;
  const eleRange = Math.max(ele_max - ele_min, 1);

  const latRange = lat_bounds[1] - lat_bounds[0];
  const lonRange = lon_bounds[1] - lon_bounds[0];

  const pts = [];
  for (const [lat, lon] of data.gpx_points) {
    const fx = (lon - lon_bounds[0]) / lonRange;
    const fy = (lat - lat_bounds[0]) / latRange;

    // Sample elevation from grid at nearest cell
    const gi = Math.min(rows - 1, Math.max(0, Math.round((1 - fy) * (rows - 1))));
    const gj = Math.min(cols - 1, Math.max(0, Math.round(fx * (cols - 1))));
    const ele = data.grid[gi][gj];

    const x = fx * W - W / 2;
    const y = fy * H - H / 2;
    const z = ((ele - ele_min) / eleRange) * MAX_Z + 0.8;   // small lift above terrain

    pts.push(new THREE.Vector3(x, y, z));
  }

  const geom = new THREE.BufferGeometry().setFromPoints(pts);
  const color = data.settings?.trail_color ?? "#ff3030";
  const mat   = new THREE.LineBasicMaterial({ color, linewidth: 2 });
  return new THREE.Line(geom, mat);
}

// ── Start/end markers ─────────────────────────────────────────────────────
function buildMarkers(data, W, H, MAX_Z) {
  const pts = data.gpx_points;
  if (!pts || pts.length < 1) return null;

  const rows = data.grid.length, cols = data.grid[0].length;
  const { ele_min, ele_max, lat_bounds, lon_bounds } = data;
  const eleRange = Math.max(ele_max - ele_min, 1);
  const latRange = lat_bounds[1] - lat_bounds[0];
  const lonRange = lon_bounds[1] - lon_bounds[0];

  function markerAt(pt, color) {
    const fx = (pt[1] - lon_bounds[0]) / lonRange;
    const fy = (pt[0] - lat_bounds[0]) / latRange;
    const gi = Math.min(rows-1, Math.max(0, Math.round((1-fy)*(rows-1))));
    const gj = Math.min(cols-1, Math.max(0, Math.round(fx*(cols-1))));
    const z  = ((data.grid[gi][gj] - ele_min) / eleRange) * MAX_Z + 2;
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(1.2, 12, 8),
      new THREE.MeshLambertMaterial({ color })
    );
    sphere.position.set(fx*W - W/2, fy*H - H/2, z);
    return sphere;
  }

  const group = new THREE.Group();
  group.add(markerAt(pts[0], 0x22cc55));             // start: green
  group.add(markerAt(pts[pts.length - 1], 0xff3333)); // end: red
  return group;
}

// ── Render preview data ───────────────────────────────────────────────────
function renderPreview(data) {
  // Clear scene (keep lights)
  for (const child of [...scene.children]) {
    if (!child.isLight) scene.remove(child);
  }

  const { group, W, H, MAX_Z } = buildTerrain(data);
  scene.add(group);

  const route = buildRoute(data, W, H, MAX_Z);
  if (route) scene.add(route);

  const markers = buildMarkers(data, W, H, MAX_Z);
  if (markers) scene.add(markers);

  // Re-position camera to fit model
  const diagXY = Math.sqrt(W * W + H * H);
  camera.position.set(0, -diagXY * 0.65, diagXY * 0.55);
  controls.target.set(0, 0, MAX_Z * 0.3);
  controls.update();

  currentData = data;
}

// ── Animation loop ────────────────────────────────────────────────────────
function startAnimate() {
  if (animId) return;
  function loop() {
    animId = requestAnimationFrame(loop);
    resizeRenderer();
    controls.update();
    renderer.render(scene, camera);
  }
  loop();
}

function stopAnimate() {
  if (animId) { cancelAnimationFrame(animId); animId = null; }
}

// ── Wireframe toggle ──────────────────────────────────────────────────────
function toggleWireframe() {
  scene.traverse(obj => {
    if (obj.isMesh && obj.geometry?.attributes?.color) {
      obj.material.wireframe = !obj.material.wireframe;
    }
  });
}

// ── Show / hide modal ─────────────────────────────────────────────────────
function showModal() {
  document.getElementById("previewModal").hidden = false;
  setTimeout(() => {
    resizeRenderer();
    startAnimate();
  }, 50);
}

function hideModal() {
  stopAnimate();
  document.getElementById("previewModal").hidden = true;
}

// ── Boot ──────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  initScene();

  document.getElementById("closePreview")
    .addEventListener("click", hideModal);

  document.getElementById("wireframeToggle")
    .addEventListener("click", toggleWireframe);

  window.addEventListener("keydown", e => {
    if (e.key === "Escape") hideModal();
  });
});

// ── Public API ────────────────────────────────────────────────────────────
window.Preview3D = {
  render:  renderPreview,
  show:    showModal,
  hide:    hideModal,
  getData: () => currentData,
};
