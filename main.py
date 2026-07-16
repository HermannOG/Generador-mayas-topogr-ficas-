"""
Topo Trail Generator – FastAPI backend

API contract aligned with topotrail.com:

  POST /api/upload            multipart: file* (GPX, repeatable) OR fileHash
                              (JSON list of hashes of already-uploaded files),
                              plus settings (JSON, camelCase schema).
                              Streams NDJSON progress lines:
                                {"type":"info","message": "..."}
                                {"type":"path","path": <jobId>, "fileHash":[...]}
                                {"type":"error","message": "..."}
                              The server generates ALL assets up front.

  GET  /api/public/{job}/…    generated assets for the browser viewer:
                              terrain.obj, trail{i}.obj, image.png, meta.json
                              (plus the printable STLs).

  POST /api/download/{job}    body {"name": str} → ZIP with printable STLs.
"""

import asyncio
import hashlib
import io
import json
import shutil
import time
import uuid as uuid_module
import zipfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import numpy as np

from src.buildings import fetch_buildings
from src.gpx_handler import get_gpx_bounds, is_swim, parse_gpx_with_type
from src.mesh_generator import MeshGenerator
from src.render_preview import render_preview_png
from src.terrain import fetch_elevation_grid
from src.water import fetch_map_features

GENERATED_DIR = Path("generated")
GPX_STORE     = GENERATED_DIR / "gpx"
JOB_TTL_S     = 24 * 3600

# Settings schema follows the topotrail.com front-end (camelCase), with
# TopoTrail extensions. Colours default to a retro national-park palette.
DEFAULT_SETTINGS = {
    "waterColor":            "#4A7A8C",   # dusty teal
    "landColor":             "#667C4E",   # forest olive
    "trackColor":            "#FC5200",
    "rockColor":             "#8C7A6B",   # warm taupe
    "sandColor":             "#D9BE8C",   # tan
    "snowColor":             "#EFEBE2",   # cream
    "snowLevel":             0,           # 0 none → 1 everything under snow
    "heightScale":           1,
    "trailWidth":            1,
    "trailHeight":           1,
    "useHeightFromGpx":      False,
    "shape":                 "hexagon",
    "distanceTrackToBorder": 0,
    "baseThickness":         5,
    "includeSeas":           True,
    "includeLakes":          True,
    "includeRivers":         False,
    "base_size":             100,
    "center":                "normal",
    "buildings":             False,
    "building_scale":        1,
    "buildingsColor":        "#777777",
    "higherResolution":      False,
    "singleColor":           False,
    "singleColor_gap":       0.5,
    # TopoTrail extensions (not on topotrail.com): hexagon border with text
    "baseColor":             "#FFFFFF",
    "textColor":             "#000000",
    "borderLabels":          ["", "", "", "", "", ""],
    # order: top, upper-right, lower-right, bottom, lower-left, upper-left
}

# Legacy snake_case names (old front-end) still accepted as fallbacks
LEGACY_KEYS = {
    "waterColor":            "water_color",
    "landColor":             "land_color",
    "trackColor":            "trail_color",
    "rockColor":             "rock_color",
    "heightScale":           "height_scale",
    "trailWidth":            "trail_width",
    "trailHeight":           "trail_height",
    "useHeightFromGpx":      "use_gpx_elevation",
    "distanceTrackToBorder": "trail_border",
    "baseThickness":         "base_thickness",
    "includeSeas":           "include_seas",
    "includeLakes":          "include_lakes",
    "includeRivers":         "include_rivers",
    "center":                "center_on",
    "buildings":             "include_buildings",
    "singleColor":           "print_separately",
}


def normalize_settings(raw: dict) -> dict:
    cfg = dict(DEFAULT_SETTINGS)
    for key in DEFAULT_SETTINGS:
        if key in raw:
            cfg[key] = raw[key]
        elif LEGACY_KEYS.get(key) in raw:
            cfg[key] = raw[LEGACY_KEYS[key]]
    return cfg


def _resolve_center(cfg, points, bounds):
    """Return (lat_c, lon_c) based on the `center` setting."""
    center = cfg.get("center", "normal")
    lat_c = bounds["lat_center"]
    lon_c = bounds["lon_center"]
    if center == "start" and points:
        lat_c, lon_c = points[0][0], points[0][1]
    elif center == "end" and points:
        lat_c, lon_c = points[-1][0], points[-1][1]
    elif center in ("highestPoint", "highest") and points:
        hp = max(points, key=lambda p: p[2])
        lat_c, lon_c = hp[0], hp[1]
    elif center in ("furthestAway", "furthest") and points:
        s = points[0]
        fp = max(points, key=lambda p: (p[0] - s[0]) ** 2 + (p[1] - s[1]) ** 2)
        lat_c, lon_c = fp[0], fp[1]
    return lat_c, lon_c


def _cleanup_old_jobs():
    if not GENERATED_DIR.exists():
        return
    now = time.time()
    for d in GENERATED_DIR.iterdir():
        if d.is_dir() and d.name != "gpx" and now - d.stat().st_mtime > JOB_TTL_S:
            shutil.rmtree(d, ignore_errors=True)


def _generate_job(job_dir: Path, trails: list, cfg: dict, progress):
    """
    Blocking generation pipeline: elevation → water/buildings → meshes → assets.
    trails: list of {"points": [(lat, lon, ele)...], "flat": bool} per GPX file
            (flat=True for swims — the ribbon stays at one height).
    progress(msg): callback for user-facing status messages.
    """
    all_points = [p for t in trails for p in t["points"]]
    bounds = get_gpx_bounds(all_points)
    if not bounds:
        raise ValueError("Could not compute bounds from the GPX files")

    lat_c, lon_c = _resolve_center(cfg, all_points, bounds)

    # Border ring with text labels: hexagon models only (for now)
    labels = [str(x) for x in (cfg.get("borderLabels") or [])][:6]
    has_labels = cfg["shape"] == "hexagon" and any(x.strip() for x in labels)
    base_size = float(cfg["base_size"])

    height_scale = max(0.1, float(cfg["heightScale"]))
    gen = MeshGenerator({
        "target_size_mm":     base_size,
        "base_thickness_mm":  float(cfg["baseThickness"]),
        "max_ele_height_mm":  20.0 * height_scale,
        "building_height_mm": 2.0 * max(0.1, float(cfg["building_scale"])),
        "route_width_mm":     float(cfg["trailWidth"]),
        "route_height_mm":    float(cfg["trailHeight"]),
        "shape":              cfg["shape"],
        "border_mm":          max(6.0, 0.08 * base_size) if has_labels else 0.0,
        "border_labels":      labels,
    })

    # Zoom out until the whole route fits INSIDE the model shape (hexagon
    # corners cut into the bounding box) with 5% clearance by default;
    # the "Distance Trail to Border" slider (0–1) adds up to +50% more.
    knob = min(1.0, max(0.0, float(cfg["distanceTrackToBorder"])))
    margin_frac = 0.05 + 0.5 * knob
    size_km = gen.fit_size_km(all_points, lat_c, lon_c, bounds["span_km"], margin_frac)

    resolution = 300 if cfg["higherResolution"] else 200

    progress("Downloading elevation data")
    grid, lat_b, lon_b = fetch_elevation_grid(lat_c, lon_c, size_km, resolution)

    progress("Downloading map features (water, forests)")
    water_data, forest_data = None, None
    try:
        all_water, forest_data = fetch_map_features(lat_b[0], lat_b[1], lon_b[0], lon_b[1])
        water_data = [
            w for w in all_water
            if (w["type"] == "lake"  and cfg["includeLakes"])
            or (w["type"] == "sea"   and cfg["includeSeas"])
            or (w["type"] == "river" and cfg["includeRivers"])
        ]
    except Exception:
        import traceback
        traceback.print_exc()   # degrade to a plain terrain model, but say why

    buildings_data = None
    if cfg["buildings"]:
        progress("Downloading buildings")
        try:
            buildings_data = fetch_buildings(lat_b[0], lat_b[1], lon_b[0], lon_b[1])
        except Exception:
            buildings_data = None

    progress("Generating 3D mesh")

    # For open seas/bays, SRTM returns ~0 m — use elevation threshold
    detect_ocean_m = 0.5 if cfg["includeSeas"] else None

    zones = gen.generate_zone_tris(
        grid, lat_b, lon_b,
        buildings=buildings_data, water=water_data, forests=forest_data,
        snow_level=float(cfg["snowLevel"]), detect_ocean_m=detect_ocean_m,
    )
    use_gpx_ele = bool(cfg["useHeightFromGpx"])
    trail_tris = [
        gen.route_tris(t["points"], flat=t["flat"], use_gpx_ele=use_gpx_ele)
        for t in trails
    ]
    border = gen.border_tris()        # border slab + raised text labels
    zones["base"] = zones["base"] + border["base"]   # walls/bottom + slab
    zones["text"] = border["text"]

    progress("Writing model files")
    job_dir.mkdir(parents=True, exist_ok=True)

    # Viewer assets: terrain.obj (zone groups) + one trail{i}.obj per GPX file
    (job_dir / "terrain.obj").write_bytes(gen.to_obj_bytes(zones))
    for i, tris in enumerate(trail_tris):
        (job_dir / f"trail{i}.obj").write_bytes(gen.to_obj_bytes({"trail": tris}))

    # Printable STLs: combined map + separate terrain/trails for multi-filament
    terrain_tris = [t for tris in zones.values() for t in tris]
    all_trail_tris = [t for tris in trail_tris for t in tris]
    (job_dir / "map.stl").write_bytes(gen.to_stl_bytes(terrain_tris + all_trail_tris))
    (job_dir / "terrain.stl").write_bytes(gen.to_stl_bytes(terrain_tris))
    for i, tris in enumerate(trail_tris):
        (job_dir / f"trail{i}.stl").write_bytes(gen.to_stl_bytes(tris))

    # Tile thumbnail
    render_preview_png(
        grid, gen.last_zone_map, MeshGenerator.ZONES, lat_b, lon_b,
        [t["points"] for t in trails],
        {
            "sand":   cfg["sandColor"],
            "forest": cfg["landColor"],
            "rock":   cfg["rockColor"],
            "water":  cfg["waterColor"],
            "snow":   cfg["snowColor"],
            "track":  cfg["trackColor"],
        },
        job_dir / "image.png",
    )

    (job_dir / "meta.json").write_text(json.dumps({
        "settings":    cfg,
        "trailAmount": len(trails),
        "eleMin":      float(grid.min()),
        "eleMax":      float(grid.max()),
        "gpxCount":    len(all_points),
    }))


app = FastAPI(title="Topo Trail Generator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    return FileResponse("static/index.html")


# ── Upload / generate endpoint ────────────────────────────────────────────────
@app.post("/api/upload")
async def upload(
    file: Optional[List[UploadFile]] = File(None),
    fileHash: Optional[str] = Form(None),
    settings: str = Form("{}"),
):
    try:
        cfg = normalize_settings(json.loads(settings))
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid settings JSON")

    # Resolve GPX contents: new uploads and/or previously stored hashes
    GPX_STORE.mkdir(parents=True, exist_ok=True)
    contents: list[bytes] = []
    hashes:   list[str]   = []

    if file:
        for up in file:
            data = await up.read()
            h = hashlib.sha256(data).hexdigest()
            (GPX_STORE / f"{h}.gpx").write_bytes(data)
            contents.append(data)
            hashes.append(h)

    if not contents and fileHash:
        try:
            requested = json.loads(fileHash)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid fileHash JSON")
        for h in requested:
            p = GPX_STORE / f"{Path(str(h)).name}.gpx"
            if not p.exists():
                raise HTTPException(410, f"Cached GPX no longer available: {h}")
            contents.append(p.read_bytes())
            hashes.append(str(h))

    if not contents:
        raise HTTPException(400, "No GPX file or fileHash provided")

    async def stream():
        def line(obj):
            return json.dumps(obj) + "\n"

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def progress(msg):
            loop.call_soon_threadsafe(queue.put_nowait, msg)

        yield line({"type": "info", "message": "Parsing GPX files"})
        try:
            trails = []
            for data in contents:
                pts, activity = parse_gpx_with_type(data)
                if not pts:
                    continue
                flat = is_swim(activity)
                if flat:
                    # Swims are flat water: GPS elevation is noise, use median
                    med = float(np.median([p[2] for p in pts]))
                    pts = [(p[0], p[1], med) for p in pts]
                trails.append({"points": pts, "flat": flat})
            if not trails:
                yield line({"type": "error", "message": "No points found in the GPX files"})
                return

            _cleanup_old_jobs()
            job_id = uuid_module.uuid4().hex[:12]
            job_dir = GENERATED_DIR / job_id

            task = asyncio.create_task(
                asyncio.to_thread(_generate_job, job_dir, trails, cfg, progress)
            )
            while not task.done():
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield line({"type": "info", "message": msg})
                except asyncio.TimeoutError:
                    pass
            task.result()  # re-raise generation errors

            while not queue.empty():
                yield line({"type": "info", "message": queue.get_nowait()})
            yield line({"type": "path", "path": job_id, "fileHash": hashes})

        except Exception as exc:
            import traceback
            traceback.print_exc()
            yield line({"type": "error", "message": f"Error generating map: {exc}"})

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ── Download endpoint ─────────────────────────────────────────────────────────
@app.post("/api/download/{job_id}")
async def download(job_id: str, body: dict):
    job_dir = GENERATED_DIR / Path(job_id).name
    meta_path = job_dir / "meta.json"
    if not job_dir.is_dir() or not meta_path.exists():
        raise HTTPException(404, "Unknown or expired job")

    meta = json.loads(meta_path.read_text())
    name = str(body.get("name") or "map3d").strip() or "map3d"

    # singleColor → separate terrain + trail STLs for filament swaps;
    # otherwise a single combined STL.
    if meta["settings"].get("singleColor"):
        files = [("terrain.stl", f"{name}_terrain.stl")] + [
            (f"trail{i}.stl", f"{name}_trail{i}.stl")
            for i in range(meta.get("trailAmount", 1))
        ]
    else:
        files = [("map.stl", f"{name}.stl")]

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in files:
            p = job_dir / src
            if p.exists():
                zf.write(p, arcname)
    zip_buf.seek(0)

    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}.zip"'},
    )


# Generated assets (terrain.obj, trail{i}.obj, image.png, meta.json, STLs)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/api/public", StaticFiles(directory=GENERATED_DIR), name="public")

# Static files – mounted last so API routes take priority
app.mount("/static", StaticFiles(directory="static"), name="static")
