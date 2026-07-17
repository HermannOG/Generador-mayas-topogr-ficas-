"""Water carving: apply lakes/seas/rivers to the terrain grid.

Pure stage: grid + features + frame in, (water_mask, adjusted_grid,
lake_masks) out. No triangulation happens here, so the terrain-only fast
path can stop after this stage plus zone classification.
"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from src.mesh.raster import mask_from_feature, rasterize_xy_line, rasterize_xy_poly


def carve_water(frame, grid, water_features, min_feature_mm, detect_ocean_m=None):
    """
    Apply water bodies to the terrain.
    Returns (merged_mask | None, adjusted_grid, lake_masks).

    - Lakes/reservoirs: flattened to the median elevation under each body
      (each lake sits at its own level).
    - Seas (polygons or detect_ocean_m cells): flattened to the map minimum.
    - Rivers: line buffered to their real width; elevation forced to the
      running minimum along the way order (OSM maps rivers downstream),
      so rivers only ever go downhill.
    """
    rows, cols = grid.shape
    out = grid.copy()
    merged = np.zeros((rows, cols), dtype=bool)
    ele_min = float(np.nanmin(grid))

    lakes, seas, rivers = [], [], []
    for w in water_features or []:
        if w["type"] == "river":
            line = w.get("line") or []
            # Rivers render at their REAL width — ones too narrow to
            # print at the chosen resolution are simply not shown.
            width_mm = w.get("width_m", 10.0) * frame.sxy
            if len(line) >= 2 and width_mm >= min_feature_mm:
                rivers.append((line, width_mm))
        elif len(w.get("coords", [])) >= 3:
            (seas if w["type"] == "sea" else lakes).append(w)

    # Lakes: each body flat at the level of the LAND at its polygon edge,
    # so the water always meets the shore without a wall. The DEM inside
    # a lake polygon cannot be trusted: reservoirs may show the dry basin
    # floor or a lower water stage than the mapped (full-pool) outline —
    # only the terrain just outside the outline is reliable.
    lake_masks = []
    for lake in lakes:
        mask = mask_from_feature(frame, lake, rows, cols)
        if mask.any():
            rim = binary_dilation(mask) & ~mask
            if not rim.any():
                rim = mask & ~binary_erosion(mask)
            level_cells = rim if rim.any() else mask
            out[mask] = float(np.median(grid[level_cells]))
            merged |= mask
            lake_masks.append(mask)

    # Seas: flatten to the map minimum
    sea_mask = np.zeros((rows, cols), dtype=bool)
    for sea in seas:
        sea_mask |= mask_from_feature(frame, sea, rows, cols)
    if detect_ocean_m is not None:
        sea_mask |= grid <= detect_ocean_m
    if sea_mask.any():
        out[sea_mask] = ele_min
        merged |= sea_mask

    # Rivers: monotonically decreasing along flow direction.
    # Pre-compute each river's ribbon cells + nearest-vertex mapping once,
    # then carve in two passes against the CURRENT grid so junctions with
    # other rivers/lakes stay consistent (a min-only op, so it converges).
    cell_mm = frame.model_w / max(cols - 1, 1)
    prepared = []
    for line, width_mm in rivers:
        raw_xy = np.array([frame.ll_to_xy(lat, lon) for lat, lon in line])
        if len(raw_xy) < 2:
            continue
        ls = LineString(raw_xy)
        # Resample at half-cell spacing: the carve must "see" every cell it
        # crosses, or dips (e.g. from tributaries) can't propagate downstream.
        n_samples = int(np.clip(ls.length / (cell_mm * 0.5), len(raw_xy), 4000))
        dists = np.linspace(0.0, ls.length, max(2, n_samples))
        pts_xy = np.array([[p.x, p.y] for p in (ls.interpolate(d) for d in dists)])

        # OSM maps rivers downstream by convention, but not always —
        # flip ways whose terrain clearly rises along their point order.
        ele0 = _grid_ele_at_xy(frame, grid, pts_xy[:, 0], pts_xy[:, 1])
        k = max(1, len(ele0) // 10)
        if np.mean(ele0[-k:]) > np.mean(ele0[:k]) + 5.0:
            pts_xy = pts_xy[::-1]

        # Ribbon polygon can be thinner than a grid cell — draw the
        # polyline too so every cell under the river is covered.
        radius_mm = width_mm / 2.0
        ribbon = ls.buffer(radius_mm)
        width_px = max(1, round(2 * radius_mm / cell_mm))
        mask = (rasterize_xy_poly(frame, ribbon, rows, cols)
                | rasterize_xy_line(frame, pts_xy, rows, cols, width_px))
        if not mask.any():
            continue
        cells = np.argwhere(mask)
        xs = cells[:, 1] / (cols - 1) * frame.model_w
        ys = (1.0 - cells[:, 0] / (rows - 1)) * frame.model_h
        _, nearest = cKDTree(pts_xy).query(np.column_stack([xs, ys]))
        prepared.append((pts_xy, cells, nearest))
        merged |= mask

    for _ in range(2 if prepared else 0):
        for pts_xy, cells, nearest in prepared:
            line_ele = _grid_ele_at_xy(frame, out, pts_xy[:, 0], pts_xy[:, 1])
            running_min = np.minimum.accumulate(line_ele)
            # Broad ribbon cells: value of their nearest sample
            out[cells[:, 0], cells[:, 1]] = np.minimum(
                out[cells[:, 0], cells[:, 1]], running_min[nearest]
            )
            # Cells the line actually crosses (possibly several times, e.g.
            # zigzags at cell scale): take the MIN over all their samples
            sj = np.clip(np.round(pts_xy[:, 0] / frame.model_w * (cols - 1)).astype(int),
                         0, cols - 1)
            si = np.clip(np.round((1.0 - pts_xy[:, 1] / frame.model_h) * (rows - 1)).astype(int),
                         0, rows - 1)
            np.minimum.at(out, (si, sj), running_min)

    if not merged.any():
        return None, out, lake_masks
    return merged, out, lake_masks


def _grid_ele_at_xy(frame, grid, xs, ys):
    """Sample grid elevation at model-xy points (nearest cell)."""
    rows, cols = grid.shape
    j = np.clip(np.round(np.asarray(xs) / frame.model_w * (cols - 1)).astype(int),
                0, cols - 1)
    i = np.clip(np.round((1.0 - np.asarray(ys) / frame.model_h) * (rows - 1)).astype(int),
                0, rows - 1)
    return grid[i, j]
