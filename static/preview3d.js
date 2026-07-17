/**
 * preview3d.js – inline 3D viewer for the main pane (mirrors the
 * topotrail.com viewer, which shows the generated model directly in the
 * top-left area). Loads the OBJ assets produced by /api/upload:
 * terrain.obj (one object per colour zone) + trail{i}.obj per GPX file.
 * Exposes: window.Preview3D = { renderJob, clear }
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { OBJLoader }     from "three/addons/loaders/OBJLoader.js";
import { mergeVertices } from "three/addons/utils/BufferGeometryUtils.js";

// Terrain zones get welded vertices + smooth normals; base/text keep
// crisp flat shading (walls and letters should stay sharp)
const SMOOTH_ZONES = new Set(["forest", "rock", "snow", "water", "trail"]);

let renderer, scene, camera, controls;
let animId = null;
let geoGroup = null;

const loader = new OBJLoader();

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

function applyColors(objRoot, colorByName, fallback, smoothFallback = false) {
  objRoot.traverse(obj => {
    if (!obj.isMesh) return;
    const colorHex = colorByName[obj.name] ?? fallback;
    obj.material = makeMaterial(colorHex);
    if (SMOOTH_ZONES.has(obj.name) || smoothFallback) {
      // OBJ triangles are a soup: weld shared vertices so normals average
      // across faces — otherwise the terrain renders faceted/blocky
      obj.geometry = mergeVertices(obj.geometry, 1e-4);
    }
    obj.geometry.computeVertexNormals();
    obj.castShadow    = true;
    obj.receiveShadow = true;
  });
}

async function renderJob({ base, trailAmount = 1, settings = {}, cacheKey = 0 }) {
  if (geoGroup) { scene.remove(geoGroup); geoGroup = null; }

  const forestColor    = settings.landColor      ?? "#327B4B";
  const rockColor      = settings.rockColor      ?? "#9A877E";
  const snowColor      = settings.snowColor      ?? "#F3EFED";
  const trackColor     = settings.trackColor     ?? "#FC5200";
  const waterColor     = settings.waterColor     ?? "#306BA6";
  const buildingsColor = settings.buildingsColor ?? "#777777";
  const baseColor      = settings.baseColor      ?? "#FFFFFF";
  const textColor      = settings.textColor      ?? "#000000";

  geoGroup = new THREE.Group();

  const terrain = await loader.loadAsync(`${base}/terrain.obj?v=${cacheKey}`);
  applyColors(terrain, {
    forest:    forestColor,
    rock:      rockColor,
    snow:      snowColor,
    water:     waterColor,
    buildings: buildingsColor,
    base:      baseColor,
    text:      textColor,
  }, rockColor);
  geoGroup.add(terrain);

  for (let i = 0; i < trailAmount; i++) {
    try {
      const trail = await loader.loadAsync(`${base}/trail${i}.obj?v=${cacheKey}`);
      applyColors(trail, {}, trackColor);
      geoGroup.add(trail);
    } catch (_) {
      // Trail file may be empty/missing when a GPX had no usable points
    }
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
