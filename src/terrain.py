"""
Elevation data fetching.

Primary source: AWS Terrain Tiles (Mapzen "terrarium" tiles on the public
elevation-tiles-prod S3 bucket) — free, no API key, up to ~10 m detail in
the US. Falls back to OpenTopoData SRTM 90m point queries if tiles fail.
"""

import io
import math
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from scipy.ndimage import gaussian_filter

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TILE_CACHE_DIR = Path("generated") / "tiles"
TILE_SIZE = 256
MAX_ZOOM = 15
MIN_ZOOM = 8


def _bounds(lat_center, lon_center, size_km):
    lat_delta = (size_km / 2.0) / 111.0
    lon_delta = (size_km / 2.0) / (111.0 * math.cos(math.radians(lat_center)))
    return (
        (lat_center - lat_delta, lat_center + lat_delta),
        (lon_center - lon_delta, lon_center + lon_delta),
    )


def fetch_elevation_grid(lat_center, lon_center, size_km, resolution=200):
    """
    Fetch a resolution×resolution elevation grid.

    Returns:
        elevation_grid : ndarray (resolution, resolution)  [meters]
                         row 0 = north (lat_max), col 0 = west (lon_min)
        lat_bounds     : (lat_min, lat_max)
        lon_bounds     : (lon_min, lon_max)
    """
    resolution = max(10, min(resolution, 500))
    try:
        return _fetch_from_tiles(lat_center, lon_center, size_km, resolution)
    except Exception:
        import traceback
        traceback.print_exc()
        return _fetch_from_opentopodata(lat_center, lon_center, size_km,
                                        min(resolution, 100))


# ── AWS Terrain Tiles (terrarium) ────────────────────────────────────────────

def _lonlat_to_tilef(lat, lon, z):
    """Fractional web-mercator tile coordinates."""
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y


def _fetch_tile(z, x, y, session):
    """Fetch one terrarium tile (with on-disk cache) → float32 (256,256) meters."""
    n = 2 ** z
    x %= n
    if y < 0 or y >= n:
        return np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)

    cache_path = TILE_CACHE_DIR / str(z) / str(x) / f"{y}.png"
    if cache_path.exists():
        data = cache_path.read_bytes()
    else:
        resp = session.get(TILE_URL.format(z=z, x=x, y=y), timeout=30)
        resp.raise_for_status()
        data = resp.content
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)

    img = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.asarray(img, dtype=np.float32)
    # terrarium encoding: ele = R*256 + G + B/256 − 32768
    return rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0 - 32768.0


def _fetch_from_tiles(lat_center, lon_center, size_km, resolution):
    (lat_min, lat_max), (lon_min, lon_max) = _bounds(lat_center, lon_center, size_km)

    # Zoom where one tile pixel ≈ one grid cell
    target_m_per_px = size_km * 1000.0 / resolution
    m_per_px_z0 = 156543.03 * math.cos(math.radians(lat_center))
    z = math.ceil(math.log2(m_per_px_z0 / (target_m_per_px * TILE_SIZE)) + 8)
    z = max(MIN_ZOOM, min(MAX_ZOOM, z))

    x0f, y0f = _lonlat_to_tilef(lat_max, lon_min, z)   # top-left
    x1f, y1f = _lonlat_to_tilef(lat_min, lon_max, z)   # bottom-right
    tx0, ty0 = int(math.floor(x0f)), int(math.floor(y0f))
    tx1, ty1 = int(math.floor(x1f)), int(math.floor(y1f))

    session = requests.Session()
    rows_px = (ty1 - ty0 + 1) * TILE_SIZE
    cols_px = (tx1 - tx0 + 1) * TILE_SIZE
    mosaic = np.zeros((rows_px, cols_px), dtype=np.float32)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            tile = _fetch_tile(z, tx, ty, session)
            r0 = (ty - ty0) * TILE_SIZE
            c0 = (tx - tx0) * TILE_SIZE
            mosaic[r0:r0 + TILE_SIZE, c0:c0 + TILE_SIZE] = tile

    # Sample the mosaic at grid points (bilinear)
    lats = np.linspace(lat_max, lat_min, resolution)   # north → south
    lons = np.linspace(lon_min, lon_max, resolution)   # west  → east
    lon_g, lat_g = np.meshgrid(lons, lats)

    n = 2 ** z
    px = ((lon_g + 180.0) / 360.0 * n - tx0) * TILE_SIZE
    lat_r = np.radians(lat_g)
    py = ((1.0 - np.arcsinh(np.tan(lat_r)) / math.pi) / 2.0 * n - ty0) * TILE_SIZE

    px = np.clip(px, 0, cols_px - 1.001)
    py = np.clip(py, 0, rows_px - 1.001)
    x0 = px.astype(int); y0 = py.astype(int)
    fx = px - x0;        fy = py - y0

    grid = (
        mosaic[y0,     x0    ] * (1 - fx) * (1 - fy)
        + mosaic[y0,     x0 + 1] * fx       * (1 - fy)
        + mosaic[y0 + 1, x0    ] * (1 - fx) * fy
        + mosaic[y0 + 1, x0 + 1] * fx       * fy
    ).astype(float)

    grid = gaussian_filter(grid, sigma=1.0)
    return grid, (lat_min, lat_max), (lon_min, lon_max)


# ── Fallback: OpenTopoData SRTM 90m point queries ────────────────────────────

def _fetch_from_opentopodata(lat_center, lon_center, size_km, resolution=50):
    (lat_min, lat_max), (lon_min, lon_max) = _bounds(lat_center, lon_center, size_km)

    lats = np.linspace(lat_max, lat_min, resolution)   # north → south
    lons = np.linspace(lon_min, lon_max, resolution)   # west  → east
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    locations = list(zip(lat_grid.flatten().tolist(), lon_grid.flatten().tolist()))
    elevations = []
    batch_size = 100

    for i in range(0, len(locations), batch_size):
        batch = locations[i : i + batch_size]
        loc_str = "|".join(f"{lat},{lon}" for lat, lon in batch)
        fetched = False

        for attempt in range(3):
            try:
                resp = requests.get(
                    "https://api.opentopodata.org/v1/srtm90m",
                    params={"locations": loc_str},
                    timeout=30,
                )
                data = resp.json()
                if data.get("status") == "OK":
                    elevations.extend(
                        r["elevation"] if r["elevation"] is not None else 0.0
                        for r in data["results"]
                    )
                    fetched = True
                    break
            except Exception:
                if attempt < 2:
                    time.sleep(1 + attempt)

        if not fetched:
            elevations.extend([0.0] * len(batch))

        if i + batch_size < len(locations):
            time.sleep(1.1)   # respect public API rate limit (1 req/s)

    grid = np.array(elevations, dtype=float).reshape(resolution, resolution)
    grid = gaussian_filter(grid, sigma=0.5)   # slight smoothing
    return grid, (lat_min, lat_max), (lon_min, lon_max)
