/**
 * preview3d.js – inline 3D viewer for the main pane.
 * Loads one binary STL per colour zone (zone_<name>.stl) plus trail{i}.stl,
 * all produced by /api/upload. Terrain zones get welded vertices and smooth
 * normals; the base slab and text keep crisp flat shading.
 * Exposes: window.Preview3D = { renderJob, clear, applyColors }
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader }     from "three/addons/loaders/STLLoader.js";
import { mergeVertices } from "three/addons/utils/BufferGeometryUtils.js";

// Zones drawn back-to-front conceptually; missing files are skipped
const ZONE_NAMES = ["base", "rock", "forest", "water", "snow", "buildings", "text"];
const SMOOTH_ZONES = new Set(["rock", "forest", "water", "snow", "trail"]);

let renderer, scene, camera, controls;
let animId = null;
let geoGroup = null;
let renderSeq = 0;   // guards against interleaved renderJob calls

const stlLoader = new STLLoader();

function initScene() {
  const canvas = document.getElementById("viewerCanvas");

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

function resizeRenderer() {
  const c = renderer.domElement;
  const w = c.clientWidth, h = c.clientHeight;
  if (w > 0 && h > 0 && (c.width !== w || c.height !== h)) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

function makeMaterial(colorHex) {
  return new THREE.MeshPhongMaterial({
    color:     new THREE.Color(colorHex),
    shininess: 12,
    specular:  new THREE.Color(0x111111),
    side:      THREE.DoubleSide,
  });
}

// Free GPU resources held by every mesh in a group. STL geometries and
// their materials are only ours — nothing is shared — so a straight
// dispose() of both is safe. Without this, repeated regenerations leak
// GPU memory (the WebGL buffers survive garbage collection).
function disposeGroup(group) {
  if (!group) return;
  group.traverse(obj => {
    if (obj.isMesh) {
      obj.geometry?.dispose();
      if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
      else obj.material?.dispose();
    }
  });
}

async function loadZone(url, colorHex, smooth, zone) {
  const resp = await fetch(url);
  if (!resp.ok) return null;
  let geometry = stlLoader.parse(await resp.arrayBuffer());
  if (smooth) {
    // STL is a triangle soup: weld shared vertices so normals average
    // across faces — otherwise the terrain renders faceted/blocky
    geometry = mergeVertices(geometry, 1e-4);
  }
  geometry.computeVertexNormals();
  const mesh = new THREE.Mesh(geometry, makeMaterial(colorHex));
  mesh.castShadow    = true;
  mesh.receiveShadow = true;
  mesh.userData.zone = zone;   // lets applyColors() retarget by zone later
  return mesh;
}

function zoneColorMap(settings) {
  return {
    rock:      settings.rockColor      ?? "#9A877E",
    forest:    settings.landColor      ?? "#327B4B",
    water:     settings.waterColor     ?? "#306BA6",
    snow:      settings.snowColor      ?? "#F3EFED",
    buildings: settings.buildingsColor ?? "#777777",
    base:      settings.baseColor      ?? "#FFFFFF",
    text:      settings.textColor      ?? "#000000",
    trail:     settings.trackColor     ?? "#FC5200",
  };
}

/**
 * Load and display the STL set for a job. Returns true if the new model was
 * committed to the scene, false if this call was superseded by a newer
 * renderJob before its downloads finished (caller should not record it as
 * "rendered" in that case).
 */
async function renderJob({ base, trailAmount = 1, settings = {}, cacheKey = 0 }) {
  const seq = ++renderSeq;

  const colors = zoneColorMap(settings);

  const jobs = ZONE_NAMES.map(name =>
    loadZone(`${base}/zone_${name}.stl?v=${cacheKey}`,
             colors[name], SMOOTH_ZONES.has(name), name));
  for (let i = 0; i < trailAmount; i++) {
    jobs.push(loadZone(`${base}/trail${i}.stl?v=${cacheKey}`, colors.trail, true, "trail"));
  }

  const meshes = (await Promise.all(jobs)).filter(Boolean);

  const newGroup = new THREE.Group();
  for (const mesh of meshes) newGroup.add(mesh);

  if (seq !== renderSeq) {
    // A newer renderJob started while we were downloading — drop this one.
    disposeGroup(newGroup);
    return false;
  }

  if (newGroup.children.length === 0) {
    throw new Error("no model files found");
  }

  // Swap in the new model only now, so the previous one stays visible
  // while the replacement downloads.
  if (geoGroup) {
    scene.remove(geoGroup);
    disposeGroup(geoGroup);
  }
  geoGroup = newGroup;

  // Centre the model at origin, base on the ground plane
  const bbox = new THREE.Box3().setFromObject(geoGroup);
  const center = new THREE.Vector3();
  bbox.getCenter(center);
  geoGroup.position.set(-center.x, -center.y, -bbox.min.z);

  scene.add(geoGroup);

  // Fit camera to model
  const size = new THREE.Vector3();
  bbox.getSize(size);
  const diag = size.length();

  camera.position.set(diag * -0.3, diag * -0.75, diag * 0.55);
  controls.target.set(0, 0, size.z * 0.35);
  controls.update();

  startAnimate();
  return true;
}

/**
 * Live-update material colours on the currently displayed model.
 * Colours are pure material state — no re-download or rebuild needed.
 */
function applyColors(settings = {}) {
  if (!geoGroup) return;
  const colors = zoneColorMap(settings);
  geoGroup.traverse(obj => {
    if (obj.isMesh && obj.userData.zone && colors[obj.userData.zone]) {
      obj.material.color.set(colors[obj.userData.zone]);
    }
  });
}

function clear() {
  renderSeq += 1;   // invalidate any in-flight renderJob
  if (geoGroup) {
    scene.remove(geoGroup);
    disposeGroup(geoGroup);
    geoGroup = null;
  }
  stopAnimate();
}

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

// The module may finish loading (CDN import) after DOMContentLoaded fired.
if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", initScene);
} else {
  initScene();
}

window.Preview3D = { renderJob, clear, applyColors };
window.dispatchEvent(new Event("preview3d-ready"));
