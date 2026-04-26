"""
Topo Trail Generator – FastAPI backend
"""

import io
import json
import math
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.buildings import fetch_buildings
from src.gpx_handler import get_gpx_bounds, parse_gpx
from src.mesh_generator import MeshGenerator
from src.terrain import fetch_elevation_grid
from src.water import fetch_water_bodies

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


@app.post("/api/generate")
async def generate(
    gpx_file: UploadFile = File(...),
    settings: str = Form("{}"),
):
    cfg = json.loads(settings)

    # ── Parse GPX ────────────────────────────────────────────────────────
    content = await gpx_file.read()
    try:
        points = parse_gpx(content)
    except Exception as exc:
        raise HTTPException(400, f"GPX inválido: {exc}")

    if not points:
        raise HTTPException(400, "No se encontraron puntos en el archivo GPX")

    bounds = get_gpx_bounds(points)
    if not bounds:
        raise HTTPException(400, "No se pueden calcular los límites del GPX")

    # ── Center selection ──────────────────────────────────────────────────
    center_on = cfg.get("center_on", "fit")
    lat_c = bounds["lat_center"]
    lon_c = bounds["lon_center"]

    if center_on == "start" and points:
        lat_c, lon_c = points[0][0], points[0][1]
    elif center_on == "end" and points:
        lat_c, lon_c = points[-1][0], points[-1][1]
    elif center_on == "highest" and points:
        hp = max(points, key=lambda p: p[2])
        lat_c, lon_c = hp[0], hp[1]
    elif center_on == "furthest" and points:
        s = points[0]
        fp = max(points, key=lambda p: (p[0] - s[0]) ** 2 + (p[1] - s[1]) ** 2)
        lat_c, lon_c = fp[0], fp[1]

    size_km = bounds["size_km"]
    resolution = 80 if cfg.get("high_resolution", False) else 50

    # ── Fetch elevation ───────────────────────────────────────────────────
    try:
        grid, lat_b, lon_b = fetch_elevation_grid(lat_c, lon_c, size_km, resolution)
    except Exception as exc:
        raise HTTPException(500, f"Error descargando elevación: {exc}")

    # ── Optional: water features ──────────────────────────────────────────
    water_data = None
    if cfg.get("include_lakes") or cfg.get("include_rivers") or cfg.get("include_seas"):
        try:
            all_water = fetch_water_bodies(lat_b[0], lat_b[1], lon_b[0], lon_b[1])
            water_data = [
                w for w in all_water
                if (
                    (w["type"] == "lake" and cfg.get("include_lakes", True))
                    or (w["type"] == "sea" and cfg.get("include_seas", True))
                    or (w["type"] == "river" and cfg.get("include_rivers", False))
                )
            ]
        except Exception:
            water_data = None

    # ── Optional: buildings ───────────────────────────────────────────────
    buildings_data = None
    if cfg.get("include_buildings"):
        try:
            buildings_data = fetch_buildings(lat_b[0], lat_b[1], lon_b[0], lon_b[1])
        except Exception:
            buildings_data = None

    # ── Mesh config ───────────────────────────────────────────────────────
    height_scale = max(0.1, float(cfg.get("height_scale", 1.0)))
    mesh_cfg = {
        "target_size_mm":    float(cfg.get("base_size", 100)),
        "base_thickness_mm": float(cfg.get("base_thickness", 5.0)),
        "max_ele_height_mm": 20.0 * height_scale,
        "building_height_mm": 2.0,
        "route_width_mm":    float(cfg.get("trail_width", 1.0)),
        "route_height_mm":   float(cfg.get("trail_height", 1.0)),
        "shape":             cfg.get("shape", "square"),
        "tree_line_m":       float(cfg.get("tree_line", 1250)),
    }

    gen = MeshGenerator(mesh_cfg)

    # ── Generate ──────────────────────────────────────────────────────────
    try:
        if cfg.get("print_separately"):
            terrain_bytes = gen.generate_bytes(
                grid, lat_b, lon_b,
                buildings=buildings_data, water=water_data, gpx_points=None,
            )
            trail_bytes = gen.generate_trail_only_bytes(grid, lat_b, lon_b, gpx_points=points)

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("terrain.stl", terrain_bytes)
                zf.writestr("trail.stl", trail_bytes)
            zip_buf.seek(0)

            return StreamingResponse(
                zip_buf,
                media_type="application/zip",
                headers={"Content-Disposition": 'attachment; filename="map3d.zip"'},
            )
        else:
            stl_bytes = gen.generate_bytes(
                grid, lat_b, lon_b,
                buildings=buildings_data, water=water_data, gpx_points=points,
            )
            return StreamingResponse(
                io.BytesIO(stl_bytes),
                media_type="application/octet-stream",
                headers={"Content-Disposition": 'attachment; filename="map3d.stl"'},
            )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Error generando malla: {exc}")


# Static files – mounted last so API routes take priority
app.mount("/static", StaticFiles(directory="static"), name="static")
