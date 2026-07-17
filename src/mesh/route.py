"""GPX route ribbon: a raised prism following the trail across the terrain."""

import math

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from src.mesh.solids import EMPTY_TRIS, prism_tris


def route_flat_z(frame, gpx_points, water_mask):
    """
    Constant ribbon height for flat activities (swims): the water surface
    height under the path when it crosses mapped water, else the median
    terrain height along the path.
    """
    zs, zs_water = [], []
    rows, cols = frame.rows, frame.cols
    for p in gpx_points[::max(1, len(gpx_points) // 300)]:
        if not frame.in_bounds(p[0], p[1]):
            continue
        z = frame.z_at(p[0], p[1])
        zs.append(z)
        if water_mask is not None:
            x, y = frame.ll_to_xy(p[0], p[1])
            j = min(cols - 1, max(0, int(x / frame.model_w * (cols - 1))))
            i = min(rows - 1, max(0, int((1.0 - y / frame.model_h) * (rows - 1))))
            if water_mask[i, j]:
                zs_water.append(z)
    if zs_water:
        return float(np.median(zs_water)) + 0.15   # float on the surface
    if zs:
        return float(np.median(zs))
    return None


def build_route(frame, points, route_width_mm, route_height_mm,
                flat_z=None, use_gpx_ele=False):
    """Trail ribbon triangles as an (N,3,3) ndarray."""
    hw = route_width_mm / 2.0
    raise_ = route_height_mm
    MIN_D = hw * 0.5   # minimum spacing to remove GPS jitter

    # Convert to model 2D, skip out-of-bounds, decimate
    pts2d = []
    for p in points:
        lat, lon = p[0], p[1]
        if not frame.in_bounds(lat, lon):
            continue
        x, y = frame.ll_to_xy(lat, lon)
        if pts2d and math.hypot(x - pts2d[-1][0], y - pts2d[-1][1]) < MIN_D:
            continue
        pts2d.append((x, y))

    if len(pts2d) < 2:
        return EMPTY_TRIS

    def z_terrain(xs, ys):
        return frame.z_at_xy(xs, ys)

    if flat_z is not None:
        def z_top(xs, ys):
            return np.full(np.shape(xs), flat_z + raise_)

        def z_bottom(xs, ys):
            return np.full(np.shape(xs), flat_z)
    elif use_gpx_ele:
        # Height from the GPX file's own elevation (nearest track point).
        # GPS altitude is noisy — smooth it with a moving average first,
        # otherwise the ribbon jumps step to step.
        in_pts = [(frame.ll_to_xy(p[0], p[1]), p[2]) for p in points
                  if frame.in_bounds(p[0], p[1])]
        kd = cKDTree([xy for xy, _ in in_pts])
        eles = np.array([e for _, e in in_pts], dtype=float)
        win = max(3, min(31, len(eles) // 50) | 1)
        kernel = np.ones(win) / win
        eles = np.convolve(np.pad(eles, win // 2, mode="edge"), kernel, "valid")

        def z_gpx(xs, ys):
            idx = kd.query(np.column_stack([np.ravel(xs), np.ravel(ys)]))[1]
            return frame.ele_to_z(eles[idx])

        def z_top(xs, ys):
            return z_gpx(xs, ys) + raise_

        def z_bottom(xs, ys):
            # Anchor into the terrain where the GPX dips below it
            return np.minimum(z_gpx(xs, ys), z_terrain(xs, ys))
    else:
        def z_top(xs, ys):
            return z_terrain(xs, ys) + raise_

        z_bottom = z_terrain

    # Simplify before buffering to remove GPS-noise zigzags
    path = LineString(pts2d).simplify(hw * 0.5, preserve_topology=True)

    # Buffer the 2D path -> smooth ribbon (round joins, flat end caps),
    # clipped to the model shape so the trail never spills over the edge
    ribbon = path.buffer(hw, cap_style=2, join_style=1, resolution=8)

    # Out-and-back passes that don't retrace exactly leave a lumpy double
    # line; morphological closing merges any touching/nearby passes into
    # one clean combined ribbon, then the outline is relaxed.
    m = hw * 1.5
    ribbon = ribbon.buffer(m).buffer(-m).simplify(hw * 0.2)

    ribbon = ribbon.intersection(frame.shape_poly)
    if ribbon.is_empty:
        return EMPTY_TRIS

    # Densify the ribbon outline so the top surface actually FOLLOWS the
    # terrain: with sparse vertices the long triangles bridge over hills
    # and dip underground in valleys.
    cell_mm = frame.model_w / max(frame.cols - 1, 1)
    ribbon = ribbon.segmentize(max(cell_mm * 0.5, hw * 0.4))

    return prism_tris(ribbon, z_top, z_bottom)
