/**
 * preview3d.js – renders the real STL mesh geometry in the browser.
 * Uses Three.js STLLoader so the preview looks exactly like the 3D-printed model.
 * Exposes: window.Preview3D = { render, show, hide, getData }
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader }     from "three/addons/loaders/STLLoader.js";

// ── Scene state ───────────────────────────────────────────────────────────────
let renderer, scene, camera, controls;
let animId = null;
let currentData = null;
let wireframeActive = false;
let geoGroup = null;          // Group that holds the model meshes (easy to clear)

const loader = new STLLoader();

// ── Init ──────────────────────────────────────────────────────────────────────
function initScene() {
  const canvas = document.getElementById("previewCanvas");

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf0f5fa);

  camera = new THREE.PerspectiveCamera(45, 1, 0.1, 2000);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance   = 5;
  controls.maxDistance   = 600;

  // Soft ambient + angled sun for good topographic shading
  scene.add(new THREE.AmbientLight(0xffffff, 0.50));

  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(-80, -30, 60);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = sun.shadow.camera.bottom = -150;
  sun.shadow.camera.right = sun.shadow.camera.top   =  150;
  scene.add(sun);

  const fill = new THREE.DirectionalLight(0xd0e4ff, 0.25);
  fill.position.set(60, 60, 40);
  scene.add(fill);
}

// ── Resize ────────────────────────────────────────────────────────────────────
function resizeRenderer() {
  const c = renderer.domElement;
  const w = c.clientWidth, h = c.clientHeight;
  if (c.width !== w || c.height !== h) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function b64toArrayBuffer(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const arr = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return buf;
}

function makeMesh(b64, colorHex) {
  if (!b64) return null;
  const geom = loader.parse(b64toArrayBuffer(b64));
  if (!geom || geom.attributes.position.count === 0) return null;
  geom.computeVertexNormals();
  const mat = new THREE.MeshPhongMaterial({
    color:     new THREE.Color(colorHex),
    shininess: 12,
    specular:  new THREE.Color(0x111111),
    side:      THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.castShadow    = true;
  mesh.receiveShadow = true;
  return mesh;
}

// ── Render real STL geometry ──────────────────────────────────────────────────
function renderPreview(data) {
  // Remove previous model
  if (geoGroup) { scene.remove(geoGroup); geoGroup = null; }
  wireframeActive = false;
  document.getElementById("wireframeToggle")?.classList.remove("active");

  const s = data.settings ?? {};
  const landColor  = s.land_color  ?? "#00cc00";
  const rockColor  = s.rock_color  ?? "#aaaaaa";
  const trailColor = s.trail_color ?? "#ff0000";

  const comps = data.components ?? {};
  const landMesh  = makeMesh(comps.land,  landColor);
  const rockMesh  = makeMesh(comps.rock,  rockColor);
  const trailMesh = makeMesh(comps.trail, trailColor);

  if (!landMesh && !rockMesh) {
    console.warn("Preview: no mesh data received");
    return;
  }

  geoGroup = new THREE.Group();
  if (landMesh)  geoGroup.add(landMesh);
  if (rockMesh)  geoGroup.add(rockMesh);
  if (trailMesh) geoGroup.add(trailMesh);

  // Centre the model at origin
  const bbox = new THREE.Box3().setFromObject(geoGroup);
  const center = new THREE.Vector3();
  bbox.getCenter(center);
  geoGroup.position.set(-center.x, -center.y, -bbox.min.z);  // Z base on ground

  scene.add(geoGroup);

  // Fit camera to model
  const size = new THREE.Vector3();
  bbox.getSize(size);
  const diag = size.length();

  camera.position.set(
    diag * -0.3,
    diag * -0.75,
    diag *  0.55,
  );
  controls.target.set(0, 0, size.z * 0.35);
  controls.update();

  currentData = data;
}

// ── Wireframe ─────────────────────────────────────────────────────────────────
function toggleWireframe() {
  if (!geoGroup) return;
  wireframeActive = !wireframeActive;
  geoGroup.traverse(obj => {
    if (obj.isMesh) obj.material.wireframe = wireframeActive;
  });
  document.getElementById("wireframeToggle")?.classList.toggle("active", wireframeActive);
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
