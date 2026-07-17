"""Building solids: extruded OSM footprints with earcut roofs.

Roof heights honour OSM tags when present: "height" in metres (units
stripped) or building:levels x 3 m, scaled to model mm and clamped to a
printable range. Untagged buildings fall back to the default height.
"""

import numpy as np
from shapely.geometry import Polygon

from src.mesh.solids import concat_tris, top_tris_from_polys

# Printable clamp for tag-derived building heights (mm, x building scale)
MIN_BUILDING_MM = 0.5
MAX_BUILDING_MM = 10.0


def parse_building_height_m(bld):
    """
    Best-effort real-world height in metres from OSM tags, or None.
    Accepts a pre-parsed "height_m", a "height" tag ("12", "12 m", "12m"),
    or "levels" / "building:levels" x 3 m. Robust to junk tags.
    """
    h = bld.get("height_m")
    if isinstance(h, (int, float)) and np.isfinite(h) and h > 0:
        return float(h)

    def _num(val):
        try:
            s = str(val).strip().split()[0].rstrip("mM")
            v = float(s)
            return v if np.isfinite(v) and v > 0 else None
        except (ValueError, IndexError):
            return None

    for key in ("height", "height_tag"):
        v = _num(bld.get(key))
        if v is not None:
            return v
    levels = _num(bld.get("levels", bld.get("building:levels")))
    if levels is not None and levels < 200:   # junk guard
        return levels * 3.0
    return None


def build_buildings(frame, buildings, default_height_mm, building_scale=1.0):
    """All building solids as one (N,3,3) ndarray."""
    scale = max(0.1, float(building_scale))
    parts = []
    for bld in buildings or []:
        coords = bld.get("coords", [])
        if len(coords) < 3:
            continue
        lats = np.array([c[0] for c in coords], dtype=np.float64)
        lons = np.array([c[1] for c in coords], dtype=np.float64)
        if (lats.max() < frame.lat_min or lats.min() > frame.lat_max or
                lons.max() < frame.lon_min or lons.min() > frame.lon_max):
            continue

        xs, ys = frame.ll_to_xy(lats, lons)
        base_z = float(np.mean(frame.z_at_many(lats[:4], lons[:4])))

        height_m = parse_building_height_m(bld)
        if height_m is not None:
            h_mm = float(np.clip(height_m * frame.sz,
                                 MIN_BUILDING_MM * scale, MAX_BUILDING_MM * scale))
        else:
            h_mm = default_height_mm
        top_z = base_z + h_mm

        n = len(xs)
        # Walls around the footprint ring
        p0 = np.column_stack([xs, ys])
        p1 = np.roll(p0, -1, axis=0)
        b0 = np.column_stack([p0, np.full(n, base_z)])
        b1 = np.column_stack([p1, np.full(n, base_z)])
        t0 = np.column_stack([p0, np.full(n, top_z)])
        t1 = np.column_stack([p1, np.full(n, top_z)])
        parts.append(np.concatenate([
            np.stack([b0, t1, b1], axis=1),
            np.stack([b0, t0, t1], axis=1),
        ]))

        # Roof: earcut triangulation — a naive centroid fan breaks on
        # concave footprints (L/U-shaped buildings)
        try:
            poly = Polygon(p0)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty:
                parts.append(top_tris_from_polys(
                    poly, lambda vx, vy, z=top_z: np.full(np.shape(vx), z)))
        except Exception:
            pass   # degenerate footprint: keep the walls, skip the roof

    return concat_tris(parts)
