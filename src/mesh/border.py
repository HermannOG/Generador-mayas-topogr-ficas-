"""Border ring: base slab around the terrain plus raised text labels
(osifont), one label per hexagon side.
"""

import math

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from src.mesh.solids import EMPTY_TRIS, concat_tris, prism_tris, top_tris_from_polys

# Label slots in UI order -> hexagon side mid-angle (flat-bottom hexagon)
SIDE_ANGLES_DEG = [90, 30, 330, 270, 210, 150]
# order: top, upper-right, lower-right, bottom, lower-left, upper-left


def build_border(frame, border_labels, text_height_mm=0.8):
    """
    Base slab (full outer shape, z 0 -> base thickness) plus raised text
    on the border ring, one label per hexagon side.
    Returns {"base": (N,3,3), "text": (N,3,3)}.
    """
    if frame.border_mm <= 0 or frame.outer_poly is None:
        return {"base": EMPTY_TRIS, "text": EMPTY_TRIS}

    base_mm = frame.base_mm

    # Slab: top face is only the visible RING (the terrain provides the
    # surface inside — a full top face would be coplanar with edge-level
    # water and z-fight), plus bottom at 0 and outer walls. The terrain's
    # walls end exactly on the ring's inner edge, closing the solid.
    ring_poly = frame.outer_poly.difference(frame.shape_poly)
    parts = [top_tris_from_polys(ring_poly, lambda xs, ys: np.full(np.shape(xs), base_mm))]

    coords = np.asarray(frame.outer_poly.exterior.coords[:-1], dtype=np.float64)
    c = frame.outer_poly.centroid
    n = len(coords)
    a = np.column_stack([coords, np.zeros(n)])                     # (ax, ay, 0)
    b = np.column_stack([np.roll(coords, -1, axis=0), np.zeros(n)])  # (bx, by, 0)
    a_top = a.copy()
    a_top[:, 2] = base_mm
    b_top = b.copy()
    b_top[:, 2] = base_mm
    cpt = np.broadcast_to([c.x, c.y, 0.0], (n, 3))
    parts.append(np.concatenate([
        np.stack([cpt, b, a], axis=1),        # bottom fan
        np.stack([a, b_top, b], axis=1),      # outer wall
        np.stack([a, a_top, b_top], axis=1),
    ]))
    base = concat_tris(parts)

    if frame.shape != "hexagon":
        return {"base": base, "text": EMPTY_TRIS}

    text_parts = []
    cx, cy = frame.model_w / 2, frame.model_h / 2
    R = min(cx, cy)                   # outer hexagon circumradius = side length
    apothem = R * math.cos(math.pi / 6)
    band_center = apothem - 0.433 * frame.border_mm  # radial middle of the ring

    for label, ang in zip(border_labels, SIDE_ANGLES_DEG, strict=False):
        label = (label or "").strip()
        if not label:
            continue
        th = math.radians(ang)
        u = (math.cos(th), math.sin(th))        # outward normal of this side
        inward = (-u[0], -u[1])
        # Keep text upright: its "up" must point skyward on the model
        up = inward if inward[1] > 0 else u
        dirv = (up[1], -up[0])                  # reading direction (left->right)

        glyphs = text_polygons(label, frame.border_mm * 0.5)
        if glyphs is None or glyphs.is_empty:
            continue
        minx, miny, maxx, maxy = glyphs.bounds
        gx, gy = (minx + maxx) / 2, (miny + maxy) / 2
        sc = min(1.0, (R * 0.9) / max(maxx - minx, 1e-6))

        px = cx + band_center * u[0]
        py = cy + band_center * u[1]
        # local (x,y) -> world: P + (x-gx)*sc*dirv + (y-gy)*sc*up
        m_a, m_b = dirv[0] * sc, up[0] * sc
        m_d, m_e = dirv[1] * sc, up[1] * sc
        placed = affine_transform(
            glyphs, [m_a, m_b, m_d, m_e,
                     px - m_a * gx - m_b * gy, py - m_d * gx - m_e * gy])
        top_z = base_mm + text_height_mm
        text_parts.append(prism_tris(
            placed,
            lambda xs, ys, z=top_z: np.full(np.shape(xs), z),
            lambda xs, ys, z=base_mm: np.full(np.shape(xs), z)))

    return {"base": base, "text": concat_tris(text_parts) if text_parts else EMPTY_TRIS}


def text_polygons(text, size_mm):
    """Text -> shapely geometry (even-odd fill handles letter holes)."""
    from pathlib import Path as _P

    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    osifont = _P("assets/fonts/osifont.ttf")
    prop = (FontProperties(fname=str(osifont)) if osifont.exists()
            else FontProperties(family="DejaVu Sans", weight="bold"))
    tp = TextPath((0, 0), text, size=size_mm, prop=prop)
    result = None
    for arr in tp.to_polygons():
        if len(arr) < 3:
            continue
        p = Polygon(arr)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty:
            continue
        result = p if result is None else result.symmetric_difference(p)
    return result
