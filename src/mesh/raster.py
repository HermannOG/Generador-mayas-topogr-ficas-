"""PIL-based rasterisation helpers (fast replacement for per-cell contains).

Pure functions of an explicit ModelFrame; all masks are bool ndarrays with
row 0 = north (lat_max).
"""

import numpy as np
from PIL import Image, ImageDraw


def rasterize_ll_polys(frame, polys, rows, cols):
    """Rasterise lat/lon rings onto the grid -> bool mask (row 0 = lat_max)."""
    img = Image.new("1", (cols, rows), 0)
    d = ImageDraw.Draw(img)
    lat_rng = max(frame.lat_max - frame.lat_min, 1e-12)
    lon_rng = max(frame.lon_max - frame.lon_min, 1e-12)
    for coords in polys:
        pts = [
            (
                (lon - frame.lon_min) / lon_rng * (cols - 1),
                (frame.lat_max - lat) / lat_rng * (rows - 1),
            )
            for lat, lon in coords
        ]
        if len(pts) >= 3:
            d.polygon(pts, fill=1)
    return np.array(img, dtype=bool)


def rasterize_xy_poly(frame, poly, rows, cols):
    """
    Rasterise a shapely (multi)polygon in model-xy space -> bool mask,
    RESPECTING interior holes (e.g. a snow patch inside a forest region).
    Parts are drawn largest-first so islands inside another part's hole
    are re-filled after the hole is punched.
    """
    img = Image.new("1", (cols, rows), 0)
    d = ImageDraw.Draw(img)

    def to_px(ring):
        return [
            (x / frame.model_w * (cols - 1), (1.0 - y / frame.model_h) * (rows - 1))
            for x, y in ring.coords
        ]

    geoms = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
    geoms = [g for g in geoms if not g.is_empty and g.geom_type == "Polygon"]
    geoms.sort(key=lambda g: g.area, reverse=True)
    for g in geoms:
        pts = to_px(g.exterior)
        if len(pts) >= 3:
            d.polygon(pts, fill=1)
        for hole in g.interiors:
            hpts = to_px(hole)
            if len(hpts) >= 3:
                d.polygon(hpts, fill=0)
    return np.array(img, dtype=bool)


def mask_from_feature(frame, feature, rows, cols):
    """Mask of a polygon feature: outer ring minus its hole rings
    (islands in lakes, clearings in forests)."""
    mask = rasterize_ll_polys(frame, [feature["coords"]], rows, cols)
    holes = feature.get("holes") or []
    if mask.any() and holes:
        mask &= ~rasterize_ll_polys(frame, holes, rows, cols)
    return mask


def rasterize_xy_line(frame, pts_xy, rows, cols, width_px=1):
    """Rasterise a polyline in model-xy space -> bool mask (every cell hit)."""
    img = Image.new("1", (cols, rows), 0)
    d = ImageDraw.Draw(img)
    pts = [
        (x / frame.model_w * (cols - 1), (1.0 - y / frame.model_h) * (rows - 1))
        for x, y in pts_xy
    ]
    if len(pts) >= 2:
        d.line(pts, fill=1, width=max(1, int(width_px)))
    return np.array(img, dtype=bool)
