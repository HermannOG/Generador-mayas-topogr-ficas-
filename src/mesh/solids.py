"""Shared solid-building helpers: earcut top faces, boundary edges, prisms.

Triangles flow through the mesh package as (N, 3, 3) float64 ndarrays
(triangle, vertex, xyz). z functions are VECTORISED: they take (xs, ys)
ndarrays and return an ndarray of z values.
"""

import mapbox_earcut as earcut
import numpy as np

EMPTY_TRIS = np.zeros((0, 3, 3), dtype=np.float64)


def as_tris(x):
    """Coerce a triangle list/array to an (N,3,3) float64 ndarray."""
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return EMPTY_TRIS
    return a.reshape(-1, 3, 3)


def concat_tris(parts):
    parts = [as_tris(p) for p in parts]
    parts = [p for p in parts if len(p)]
    if not parts:
        return EMPTY_TRIS
    return np.concatenate(parts, axis=0)


def fix_winding_ccw(tris):
    """Flip triangles whose 2D (xy) winding is clockwise, in place."""
    if not len(tris):
        return tris
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    cz = ((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
          - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0]))
    flip = cz < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return tris


def top_tris_from_polys(geom, z_fn):
    """
    Triangulate a (multi)polygon -> CCW top-face tris, z from z_fn(xs, ys).
    Uses earcut, which handles holes (letter glyphs, islands) and never
    drops slivers — a Delaunay+filter approach leaves holes in trail
    ribbons and mangles border text.
    """
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    out = []
    for poly in polys:
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        rings = [list(poly.exterior.coords[:-1])]
        rings += [list(r.coords[:-1]) for r in poly.interiors]
        verts = np.array([p for ring in rings for p in ring], dtype=np.float64)
        if len(verts) < 3:
            continue
        ring_ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
        idx = np.asarray(earcut.triangulate_float64(verts, ring_ends), dtype=np.int64)
        if len(idx) < 3:
            continue
        zs = np.asarray(z_fn(verts[:, 0], verts[:, 1]), dtype=np.float64)
        zs = np.broadcast_to(zs, (len(verts),))
        v3 = np.column_stack([verts, zs])
        tris = v3[idx].reshape(-1, 3, 3)
        out.append(fix_winding_ccw(tris))
    return concat_tris(out)


def boundary_edges(top_tris):
    """Directed edges that appear in exactly one triangle of a surface.

    Returns an (E, 2, 3) ndarray of edge endpoint pairs (in the winding
    direction of the triangle that owns them).
    """
    tris = as_tris(top_tris)
    if not len(tris):
        return np.zeros((0, 2, 3), dtype=np.float64)
    PREC = 3

    edge_cnt, edge_dir = {}, {}
    r = np.round(tris, PREC)
    for t in range(len(tris)):
        for k in range(3):
            a = tuple(r[t, k])
            b = tuple(r[t, (k + 1) % 3])
            key = (min(a, b), max(a, b))
            edge_cnt[key] = edge_cnt.get(key, 0) + 1
            edge_dir[key] = (tris[t, k], tris[t, (k + 1) % 3])
    edges = [edge_dir[k] for k, c in edge_cnt.items() if c == 1]
    if not edges:
        return np.zeros((0, 2, 3), dtype=np.float64)
    return np.stack([np.stack(e) for e in edges])


def prism_tris(geom, z_top_fn, z_bottom_fn):
    """
    Closed solid from a (multi)polygon footprint: top face at z_top_fn,
    bottom face (reversed winding) at z_bottom_fn, side walls between.
    Used for trail ribbons and border text. z functions are vectorised.
    """
    top = top_tris_from_polys(geom, z_top_fn)
    if not len(top):
        return EMPTY_TRIS

    # Bottom face: same footprint, z from z_bottom_fn, reversed winding
    flat = top.reshape(-1, 3)
    zb = np.broadcast_to(
        np.asarray(z_bottom_fn(flat[:, 0], flat[:, 1]), dtype=np.float64),
        (len(flat),)).reshape(-1, 3)
    bottom = top.copy()
    bottom[:, :, 2] = zb
    bottom = bottom[:, [0, 2, 1]]

    # Side walls between the top boundary and its projection at z_bottom
    edges = boundary_edges(top)
    walls = np.zeros((0, 3, 3), dtype=np.float64)
    if len(edges):
        t1, t2 = edges[:, 0], edges[:, 1]
        ends = edges.reshape(-1, 3)
        zbe = np.broadcast_to(
            np.asarray(z_bottom_fn(ends[:, 0], ends[:, 1]), dtype=np.float64),
            (len(ends),)).reshape(-1, 2)
        b1 = np.column_stack([t1[:, 0], t1[:, 1], zbe[:, 0]])
        b2 = np.column_stack([t2[:, 0], t2[:, 1], zbe[:, 1]])
        walls = np.concatenate([
            np.stack([t2, t1, b1], axis=1),   # outward-facing (right of t1->t2)
            np.stack([t2, b1, b2], axis=1),
        ])

    return concat_tris([top, bottom, walls])
