"""Zone classification: rock/forest ground, procedural snow, frozen lakes.

Pure stage: frame + grids + masks in, per-cell zone map out.
"""

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

from src.mesh.raster import mask_from_feature

# Zone ids used in the per-cell zone map (thumbnail + classification)
ZONES = ("rock", "forest", "water", "snow")
Z_ROCK, Z_FOREST, Z_WATER, Z_SNOW = range(4)

# Slope used by forest growth as "too steep for trees" (rise/run)
ROCK_SLOPE = 0.45


def build_zone_map(frame, grid, water_mask, forests, forest_level, snow_level,
                   lake_masks=()):
    """Per-cell zone ids: rock/forest ground, then snow, then water."""
    rows, cols = grid.shape

    cell_m = max((frame.lat_max - frame.lat_min) * 111_000 / max(rows - 1, 1), 1.0)
    gy, gx = np.gradient(grid, cell_m)
    slope = np.hypot(gx, gy)
    zone = np.full((rows, cols), Z_ROCK, dtype=np.uint8)

    # Forests: real OSM outlines as the baseline, grown procedurally
    forest_mask = np.zeros((rows, cols), dtype=bool)
    for f in forests or []:
        forest_mask |= mask_from_feature(frame, f, rows, cols)
    forest_mask = grow_forest_mask(frame, grid, forest_mask, water_mask,
                                   forest_level, slope)
    zone[forest_mask] = Z_FOREST

    snow_mask = build_snow_mask(frame, grid, water_mask, snow_level, gx, gy, slope)
    if snow_mask is not None:
        zone[snow_mask] = Z_SNOW
    if water_mask is not None:
        zone[water_mask] = Z_WATER
    # Frozen lakes: a lake whose entire shore is snowed-in reads as snow
    # too (it stays flat — which is exactly what a frozen lake looks like)
    if snow_mask is not None:
        for mask in lake_masks or []:
            ring = binary_dilation(mask) & ~mask
            if ring.any() and snow_mask[ring].mean() > 0.9:
                zone[mask] = Z_SNOW
    return zone


def grow_forest_mask(frame, grid, base_mask, water_mask, forest_level, slope):
    """
    Grow forests procedurally from the OSM baseline. forest_level 0..1:
    0 keeps exactly the mapped forests; 1 forests everything growable.
    New trees appear in the most likely places first:
      + next to existing forest (patches spread outward)
      + lower, gentler, moister (hollows) ground
      + a deterministic noise field seeds detached patches and keeps
        edges ragged rather than evenly cut off
    Trees never grow near the peaks: in mountainous terrain a local tree
    line caps growth a little above the highest mapped forest.
    """
    level = min(1.0, max(0.0, float(forest_level)))
    if level <= 0.0:
        return base_mask

    ele_min = float(grid.min())
    ele_range = max(float(grid.max()) - ele_min, 1.0)

    # Local tree line: only meaningful in mountainous terrain
    cap = np.inf
    if ele_range > 700.0:
        cap = ele_min + 0.8 * ele_range
        if base_mask.any():
            cap = max(cap, float(np.quantile(grid[base_mask], 0.99)) + 0.05 * ele_range)

    growable = (grid <= cap) & (slope < 2 * ROCK_SLOPE) & ~base_mask
    if water_mask is not None:
        growable &= ~water_mask
    if not growable.any():
        return base_mask

    def z(x):
        sd = float(np.std(x))
        return (x - float(np.mean(x))) / (sd if sd > 1e-9 else 1.0)

    # Distance from existing forest: spreading beats sprouting
    if base_mask.any():
        proximity = -distance_transform_edt(~base_mask)
    else:
        proximity = np.zeros(grid.shape)

    hollows = gaussian_filter(grid, sigma=3.0) - grid
    rng = np.random.default_rng(
        abs(hash(("forest", round(frame.lat_min, 4), round(frame.lon_min, 4)))) % 2**32)
    noise = gaussian_filter(rng.standard_normal(grid.shape), sigma=2.5)

    score = (1.5 * np.tanh(z(proximity)) - 1.0 * z(grid) + 0.4 * np.tanh(z(hollows))
             - 0.5 * np.clip(z(slope), 0.0, None) + 0.9 * noise)

    # level = fraction of the growable area that gets trees
    thr = float(np.quantile(score[growable], 1.0 - level))
    return base_mask | (growable & (score >= thr))


def build_snow_mask(frame, grid, water_mask, snow_level, gx, gy, slope):
    """
    Procedural snow cover. snow_level 0..1 sets roughly the fraction of
    (non-water) terrain under snow, distributed realistically:
      + higher terrain first (snow line drags down as the slider rises)
      + shaded slopes keep snow longer (north-facing in the northern
        hemisphere, south-facing below the equator)
      + hollows and gullies hold drifts
      + patchy noise at the melt edge (deterministic per location)
      - the steepest cliff faces shed their snow
    """
    s = min(1.0, max(0.0, float(snow_level)))
    if s <= 0.0:
        return None
    not_water = ~water_mask if water_mask is not None else np.ones(grid.shape, bool)
    if s >= 1.0:
        return not_water

    def z(x):
        sd = float(np.std(x))
        return (x - float(np.mean(x))) / (sd if sd > 1e-9 else 1.0)

    ele_n = (grid - grid.min()) / max(grid.max() - grid.min(), 1.0)

    # gy is the north->south row gradient: positive on north-facing slopes
    lat_mid = (frame.lat_min + frame.lat_max) / 2.0
    shade = gy if lat_mid >= 0 else -gy

    # Hollows: locally below the smoothed surface
    hollows = gaussian_filter(grid, sigma=3.0) - grid

    rng = np.random.default_rng(
        abs(hash((round(frame.lat_min, 4), round(frame.lon_min, 4)))) % 2**32)
    noise = gaussian_filter(rng.standard_normal(grid.shape), sigma=2.0)

    score = (3.0 * z(ele_n) + 0.8 * np.tanh(z(shade)) + 0.5 * np.tanh(z(hollows))
             + 0.5 * noise - 0.6 * np.clip(z(slope), 0.0, None))

    # Threshold at the requested coverage over non-water terrain
    thr = float(np.quantile(score[not_water], 1.0 - s))
    return (score >= thr) & not_water
