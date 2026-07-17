"""
Generation pipeline: GPX trails + settings -> preview image + printable STLs.

This is the blocking (thread-pool) side of the API: main.py owns the routes
and streaming plumbing, this module owns the actual work.
"""

import json
import shutil
import time
from pathlib import Path

from src.buildings import fetch_buildings
from src.gpx_handler import get_gpx_bounds
from src.mesh.solids import concat_tris
from src.mesh_generator import MeshGenerator
from src.render_preview import render_preview_png
from src.terrain import fetch_elevation_grid
from src.water import fetch_map_features

GENERATED_DIR = Path("generated")
GPX_STORE = GENERATED_DIR / "gpx"
JOB_TTL_S = 24 * 3600

# The browser viewer gets a capped mesh density (0.4 mm cells -> 0.2 here
# gives one cell per 0.2 mm) so it stays responsive; print STLs go finer.
VIEW_CELL_MM = 0.2
MAX_PRINT_CELLS = 1100


def resolve_center(cfg, points, bounds):
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


def cleanup_old_jobs():
    if not GENERATED_DIR.exists():
        return
    now = time.time()
    for d in GENERATED_DIR.iterdir():
        if d.is_dir() and d.name != "gpx" and now - d.stat().st_mtime > JOB_TTL_S:
            shutil.rmtree(d, ignore_errors=True)


def _mesh_config(cfg, cell_mm, base_size, has_labels, labels):
    return {
        "target_size_mm":     base_size,
        "base_thickness_mm":  float(cfg["baseThickness"]),
        "height_scale":       max(0.0, float(cfg["heightScale"])),
        "standardize_height": bool(cfg["standardizeHeight"]),
        "min_feature_mm":     2.0 * cell_mm,
        "smooth_zones":       bool(cfg["smooth"]),
        "building_height_mm": 2.0 * max(0.1, float(cfg["building_scale"])),
        "building_scale":     max(0.1, float(cfg["building_scale"])),
        "route_width_mm":     max(float(cfg["trailWidth"]), cell_mm),
        "route_height_mm":    float(cfg["trailHeight"]),
        "shape":              cfg["shape"],
        "border_mm":          max(6.0, 0.08 * base_size) if has_labels else 0.0,
        "border_labels":      labels,
    }


def generate_job(job_dir: Path, trails: list, cfg: dict, progress,
                 terrain_only: bool = False):
    """
    Blocking generation pipeline: elevation -> water/buildings -> meshes -> assets.
    trails: list of {"points": [(lat, lon, ele)...], "flat": bool} per GPX file
            (flat=True for swims — the ribbon stays at one height).
    progress(msg): callback for user-facing status messages; dicts are
            forwarded verbatim to the NDJSON stream (e.g. the early
            {"type": "image"} event once the 2D map preview is ready).
    terrain_only: stop after the 2D map preview (image.png + meta.json) —
            used by the fast "Generate Terrain" button; no meshes are built
            and only the view-density elevation grid is fetched.
    """
    all_points = [p for t in trails for p in t["points"]]
    bounds = get_gpx_bounds(all_points)
    if not bounds:
        raise ValueError("Could not compute bounds from the GPX files")

    lat_c, lon_c = resolve_center(cfg, all_points, bounds)

    # Border ring with text labels: hexagon models only (for now)
    labels = [str(x) for x in (cfg.get("borderLabels") or [])][:6]
    has_labels = cfg["shape"] == "hexagon" and any(x.strip() for x in labels)
    base_size = float(cfg["base_size"])

    # Physical print fidelity: mm per mesh cell (0.1 / 0.2 / 0.4 / 0.8)
    cell_mm = min(0.8, max(0.1, float(cfg["printResolution"])))
    mesh_cfg = _mesh_config(cfg, cell_mm, base_size, has_labels, labels)
    gen = MeshGenerator(mesh_cfg)

    # Zoom out until the whole route fits INSIDE the model shape (hexagon
    # corners cut into the bounding box) with 5% clearance by default;
    # the "Distance Trail to Border" slider (0-1) adds up to +50% more.
    knob = min(1.0, max(0.0, float(cfg["distanceTrackToBorder"])))
    margin_frac = 0.05 + 0.5 * knob
    size_km = gen.fit_size_km(all_points, lat_c, lon_c, bounds["span_km"], margin_frac)

    # Mesh density from the chosen fidelity: one cell per cell_mm of model.
    # The browser viewer gets a capped copy so it stays responsive; the
    # printable STLs use the full density — the slicer decides the rest.
    n_print = min(MAX_PRINT_CELLS, round(base_size / cell_mm) + 1)
    n_view = min(n_print, round(base_size / VIEW_CELL_MM) + 1)

    progress("Downloading elevation data")
    if terrain_only:
        # Fast path: only the view-density grid is ever needed
        grid_view, lat_b, lon_b = fetch_elevation_grid(lat_c, lon_c, size_km, n_view)
        grid = grid_view
    else:
        grid, lat_b, lon_b = fetch_elevation_grid(lat_c, lon_c, size_km, n_print)
        grid_view = grid if n_view == n_print else \
            fetch_elevation_grid(lat_c, lon_c, size_km, n_view)[0]

    progress("Downloading map features (water, forests)")
    water_data, forest_data = None, None
    try:
        all_water, forest_data = fetch_map_features(lat_b[0], lat_b[1], lon_b[0], lon_b[1])
        water_data = [
            w for w in all_water
            if (w["type"] == "lake" and cfg["includeLakes"])
            or (w["type"] == "sea" and cfg["includeSeas"])
            or (w["type"] == "river" and cfg["includeRivers"])
        ]
    except Exception:
        import traceback
        traceback.print_exc()   # degrade to a plain terrain model, but say why

    buildings_data = None
    if cfg["buildings"] and not terrain_only:
        progress("Downloading buildings")
        try:
            buildings_data = fetch_buildings(lat_b[0], lat_b[1], lon_b[0], lon_b[1])
        except Exception:
            buildings_data = None

    # For open seas/bays, SRTM returns ~0 m — use elevation threshold
    detect_ocean_m = 0.5 if cfg["includeSeas"] else None

    zone_kwargs = dict(
        water=water_data, forests=forest_data,
        forest_level=float(cfg["forestLevel"]),
        snow_level=float(cfg["snowLevel"]), detect_ocean_m=detect_ocean_m,
    )

    gen_view = MeshGenerator(mesh_cfg)
    if terrain_only:
        # No triangulation: stop after setup + water carving + zone map
        gen_view.prepare(grid_view, lat_b, lon_b, **zone_kwargs)
        zones_view = None
    else:
        progress("Generating 3D mesh")
        zones_view = gen_view.generate_zone_tris(
            grid_view, lat_b, lon_b, buildings=buildings_data, **zone_kwargs)

    # 2D map preview first — the client can show it while the 3D build runs
    job_dir.mkdir(parents=True, exist_ok=True)
    render_preview_png(
        grid_view, gen_view.last_zone_map, MeshGenerator.ZONES, lat_b, lon_b,
        [t["points"] for t in trails],
        {
            "forest": cfg["landColor"],
            "rock":   cfg["rockColor"],
            "water":  cfg["waterColor"],
            "snow":   cfg["snowColor"],
            "track":  cfg["trackColor"],
        },
        job_dir / "image.png",
    )
    progress({"type": "image", "path": job_dir.name})

    if terrain_only:
        (job_dir / "meta.json").write_text(json.dumps({
            "settings":    cfg,
            "terrainOnly": True,
            "trailAmount": len(trails),
            "eleMin":      float(grid.min()),
            "eleMax":      float(grid.max()),
            "gpxCount":    len(all_points),
        }))
        return

    use_gpx_ele = bool(cfg["useHeightFromGpx"])

    # Viewer assets: one binary STL per colour zone at the view density
    border_view = gen_view.border_tris()
    zones_view["base"] = concat_tris([zones_view["base"], border_view["base"]])
    zones_view["text"] = border_view["text"]
    for name, tris in zones_view.items():
        if len(tris):
            (job_dir / f"zone_{name}.stl").write_bytes(gen_view.to_stl_bytes(tris))

    # Printable STLs at full fidelity
    if n_print != n_view:
        progress("Building print-quality mesh")
        zones = gen.generate_zone_tris(grid, lat_b, lon_b,
                                       buildings=buildings_data, **zone_kwargs)
        printer = gen
    else:
        zones = zones_view
        printer = gen_view
    trail_tris = [
        printer.route_tris(t["points"], flat=t["flat"], use_gpx_ele=use_gpx_ele)
        for t in trails
    ]
    if n_print != n_view:
        border = printer.border_tris()
        zones["base"] = concat_tris([zones["base"], border["base"]])
        zones["text"] = border["text"]

    progress("Writing model files")
    terrain_tris = concat_tris(list(zones.values()))
    all_trail_tris = concat_tris(trail_tris)
    (job_dir / "map.stl").write_bytes(
        printer.to_stl_bytes(concat_tris([terrain_tris, all_trail_tris])))
    (job_dir / "terrain.stl").write_bytes(printer.to_stl_bytes(terrain_tris))
    for i, tris in enumerate(trail_tris):
        (job_dir / f"trail{i}.stl").write_bytes(printer.to_stl_bytes(tris))

    (job_dir / "meta.json").write_text(json.dumps({
        "settings":    cfg,
        "trailAmount": len(trails),
        "eleMin":      float(grid.min()),
        "eleMax":      float(grid.max()),
        "gpxCount":    len(all_points),
    }))
