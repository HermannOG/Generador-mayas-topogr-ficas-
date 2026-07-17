/**
 * preview3d.js – inline 3D viewer for the main pane.
 * Loads one binary STL per colour zone (zone_<name>.stl) plus trail{i}.stl,
 * all produced by /api/upload. Terrain zones get welded vertices and smooth
 * normals; the base slab and text keep crisp flat shading.
 * Exposes: window.Preview3D = { renderJob, clear }
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

async function loadZone(url, colorHex, smooth) {
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
  return mesh;
}

async function renderJob({ base, trailAmount = 1, settings = {}, cacheKey = 0 }) {
  if (geoGroup) { scene.remove(geoGroup); geoGroup = null; }

  const zoneColor = {
    rock:      settings.rockColor      ?? "#9A877E",
    forest:    settings.landColor      ?? "#327B4B",
    water:     settings.waterColor     ?? "#306BA6",
    snow:      settings.snowColor      ?? "#F3EFED",
    buildings: settings.buildingsColor ?? "#777777",
    base:      settings.baseColor      ?? "#FFFFFF",
    text:      settings.textColor      ?? "#000000",
  };
  const trackColor = settings.trackColor ?? "#FC5200";

  geoGroup = new THREE.Group();

  const jobs = ZONE_NAMES.map(name =>
    loadZone(`${base}/zone_${name}.stl?v=${cacheKey}`,
             zoneColor[name], SMOOTH_ZONES.has(name)));
  for (let i = 0; i < trailAmount; i++) {
    jobs.push(loadZone(`${base}/trail${i}.stl?v=${cacheKey}`, trackColor, true));
  }
  for (const mesh of await Promise.all(jobs)) {
    if (mesh) geoGroup.add(mesh);
  }
  if (geoGroup.children.length === 0) {
    throw new Error("no model files found");
  }

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
}

function clear() {
  if (geoGroup) { scene.remove(geoGroup); geoGroup = null; }
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

window.addEventListener("DOMContentLoaded", initScene);

window.Preview3D = { renderJob, clear };
