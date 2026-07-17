"""Terrain surface construction: lattice, exact region clipping, smooth zone
regions, crust/body cut walls, and the bottom face.
"""

import math

import numpy as np
from contourpy import contour_generator
from scipy.ndimage import binary_dilation, gaussian_filter
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from src.mesh.raster import rasterize_xy_poly
from src.mesh.solids import EMPTY_TRIS, boundary_edges, concat_tris
from src.mesh.zones import Z_FOREST, Z_ROCK, Z_SNOW, Z_WATER

# Cut-wall crust: how deep the surface colour extends down the sides
CRUST_MM = 1.2


def lattice(frame, grid):
    """Vertex lattice: model xy + z for every grid node."""
    rows, cols = grid.shape
    xs = np.linspace(0.0, frame.model_w, cols)
    ys = np.linspace(frame.model_h, 0.0, rows)   # row 0 = north = max y
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X, Y, frame.ele_to_z(grid)], axis=-1)


def region_surface(frame, V, poly, claimed=None):
    """
    Surface triangles clipped EXACTLY to a region polygon. Rasterisation
    alone is half-a-cell sloppy — a STRICT mask (region shrunk by a cell)
    marks guaranteed-inside quads, a LOOSE mask (region grown) marks
    candidates for exact shapely clipping, so cuts land precisely on the
    region outline (model shape or a smooth zone boundary).
    Returns (fast_tris, slow_tris): slow tris are near a boundary and
    must take part in wall/edge accounting; fast tris never can.
    """
    rows, cols = V.shape[:2]
    cell_mm = frame.model_w / max(cols - 1, 1)
    strict = rasterize_xy_poly(frame, poly.buffer(-1.6 * cell_mm), rows, cols)
    loose = rasterize_xy_poly(frame, poly.buffer(+1.6 * cell_mm), rows, cols)

    n_strict = (strict[:-1, :-1].astype(np.int8) + strict[:-1, 1:] +
                strict[1:, 1:] + strict[1:, :-1])
    n_loose = (loose[:-1, :-1].astype(np.int8) + loose[:-1, 1:] +
               loose[1:, 1:] + loose[1:, :-1])
    full = n_strict == 4
    if claimed is not None:
        # Raster masks are half-a-pixel sloppy: never let two regions
        # both claim the same full quad (z-fighting checkerboards)
        full &= ~claimed
        claimed |= full
    partial = ~full & (n_loose > 0)

    # Only quads near a boundary need slow edge bookkeeping; the grid
    # edge itself is a boundary too (square shapes fill the whole grid).
    padded = np.ones((full.shape[0] + 2, full.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = ~full
    near_boundary = binary_dilation(padded, iterations=2)[1:-1, 1:-1]
    fast_full = full & ~near_boundary
    slow_full = full & near_boundary

    # Fast path: bulk-emit interior full quads
    fi, fj = np.nonzero(fast_full)
    if len(fi):
        v00 = V[fi, fj]
        v01 = V[fi, fj + 1]
        v10 = V[fi + 1, fj]
        v11 = V[fi + 1, fj + 1]
        fast_tris = np.concatenate([
            np.stack([v00, v11, v01], axis=1),
            np.stack([v00, v10, v11], axis=1),
        ])
    else:
        fast_tris = EMPTY_TRIS

    # Slow full quads: same bulk emit, kept separate for edge accounting
    si, sj = np.nonzero(slow_full)
    if len(si):
        v00 = V[si, sj]
        v01 = V[si, sj + 1]
        v10 = V[si + 1, sj]
        v11 = V[si + 1, sj + 1]
        slow_parts = [np.concatenate([
            np.stack([v00, v11, v01], axis=1),
            np.stack([v00, v10, v11], axis=1),
        ])]
    else:
        slow_parts = []

    # Spatial index over the region's parts keeps per-quad clipping cheap
    parts = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
    tree = STRtree(parts)

    pi_, pj_ = np.nonzero(partial)
    clip_polys = []
    for i, j in zip(pi_.tolist(), pj_.tolist(), strict=True):
        quad_poly = Polygon([
            (V[i, j][0], V[i, j][1]),
            (V[i, j + 1][0], V[i, j + 1][1]),
            (V[i + 1, j + 1][0], V[i + 1, j + 1][1]),
            (V[i + 1, j][0], V[i + 1, j][1]),
        ])
        for pi in tree.query(quad_poly):
            clipped = parts[pi].intersection(quad_poly)
            if clipped.is_empty:
                continue
            geoms = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
            for geom in geoms:
                if (geom.geom_type != "Polygon" or geom.is_empty
                        or geom.area < 1e-4):   # skip hairline seam slivers
                    continue
                pts2d = list(geom.exterior.coords[:-1])
                if len(pts2d) >= 3:
                    clip_polys.append(np.asarray(pts2d, dtype=np.float64))

    if clip_polys:
        # Batch the elevation lookups for ALL clipped polygon vertices at once
        all_pts = np.concatenate(clip_polys)
        lat, lon = frame.xy_to_ll(all_pts[:, 0], all_pts[:, 1])
        all_z = frame.z_at_many(lat, lon)
        off = 0
        for pts2d in clip_polys:
            n = len(pts2d)
            verts = np.column_stack([pts2d, all_z[off:off + n]])
            off += n
            # Fan-triangulate with per-triangle winding fix
            v0 = np.broadcast_to(verts[0], (n - 2, 3))
            v1 = verts[1:-1]
            v2 = verts[2:]
            cz = ((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
                  - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0]))
            tris = np.stack([v0, v1, v2], axis=1)
            flip = cz < 0
            tris[flip] = tris[flip][:, [0, 2, 1]]
            slow_parts.append(tris)

    slow_tris = concat_tris(slow_parts) if slow_parts else EMPTY_TRIS
    return fast_tris, slow_tris


def smooth_field_polys(frame, mask, rows, cols):
    """
    Vector outline of a raster mask as SMOOTH polygons in model xy.
    The mask is softened, contour-traced at sub-cell precision, and the
    marching-squares corners are rounded off — procedural smooth lines
    instead of pixel staircases.
    """
    if not mask.any():
        return None
    field = gaussian_filter(mask.astype(np.float32), 1.0)
    padded = np.pad(field, 1, constant_values=0.0)
    cell = frame.model_w / max(cols - 1, 1)

    geom = None
    for arr in contour_generator(z=padded).lines(0.45):
        if len(arr) < 4:
            continue
        xs = (arr[:, 0] - 1) / (cols - 1) * frame.model_w
        ys = (1.0 - (arr[:, 1] - 1) / (rows - 1)) * frame.model_h
        p = Polygon(np.column_stack([xs, ys]))
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty:
            continue
        geom = p if geom is None else geom.symmetric_difference(p)
    if geom is None or geom.is_empty:
        return None
    r = 1.0 * cell
    geom = (geom.buffer(r, join_style=1).buffer(-r, join_style=1)
            .simplify(0.2 * cell))
    return None if geom.is_empty else geom


def zone_regions(frame, zone_map):
    """
    Partition the model shape into smooth vector regions per zone,
    precedence water > snow > forest, remainder = rock.
    """
    rows, cols = zone_map.shape
    regions = []
    occupied = None
    for zid in (Z_WATER, Z_SNOW, Z_FOREST):
        g = smooth_field_polys(frame, zone_map == zid, rows, cols)
        if g is None:
            continue
        g = g.intersection(frame.shape_poly)
        if occupied is not None:
            g = g.difference(occupied)
        if g.is_empty:
            continue
        regions.append((zid, g))
        occupied = g if occupied is None else occupied.union(g)
    rock = (frame.shape_poly.difference(occupied)
            if occupied is not None else frame.shape_poly)
    if not rock.is_empty:
        regions.append((Z_ROCK, rock))
    return regions


def build_terrain(frame, elevation_grid, zone_map=None):
    """
    Build the terrain surface(s) plus cut walls and bottom.
    zone_map given (smooth mode): the surface is cut along smooth vector
    zone boundaries and returned per zone id. zone_map None: one surface
    under key None, classified per cell by the caller.
    Returns (surfaces: dict of (N,3,3) arrays, crust, body, bottom).
    """
    V = lattice(frame, elevation_grid)

    if zone_map is not None:
        region_list = zone_regions(frame, zone_map)
    else:
        region_list = [(None, frame.shape_poly)]

    surfaces = {}
    all_slow = []
    claimed = np.zeros((V.shape[0] - 1, V.shape[1] - 1), dtype=bool)
    for zid, poly in region_list:
        fast, slow = region_surface(frame, V, poly, claimed)
        surfaces[zid] = concat_tris([surfaces.get(zid, EMPTY_TRIS), fast, slow])
        all_slow.append(slow)

    # With a border, the terrain sits ON the base slab: walls stop at the
    # slab top and the slab provides the bottom face.
    floor_z = frame.base_mm if frame.border_mm > 0 else 0.0

    # -- Walls: a thin CRUST band under the surface takes the surface
    # zone colour (cutting through snow shows a white line, through a
    # lake a blue line), and the BODY below reads as rock — like a
    # geological cross-section.
    edges = boundary_edges(concat_tris(all_slow))
    crust = EMPTY_TRIS
    body = EMPTY_TRIS
    bot_pts = np.zeros((0, 2))
    if len(edges):
        t1, t2 = edges[:, 0], edges[:, 1]
        c1 = t1.copy()
        c1[:, 2] = np.maximum(t1[:, 2] - CRUST_MM, floor_z)
        c2 = t2.copy()
        c2[:, 2] = np.maximum(t2[:, 2] - CRUST_MM, floor_z)
        crust = np.concatenate([
            np.stack([t2, t1, c1], axis=1),   # outward-facing (right of t1->t2)
            np.stack([t2, c1, c2], axis=1),
        ])
        keep = (c1[:, 2] > floor_z + 1e-9) | (c2[:, 2] > floor_z + 1e-9)
        if keep.any():
            b1 = c1[keep].copy()
            b1[:, 2] = floor_z
            b2 = c2[keep].copy()
            b2[:, 2] = floor_z
            body = np.concatenate([
                np.stack([c2[keep], c1[keep], b1], axis=1),
                np.stack([c2[keep], b1, b2], axis=1),
            ])
        bot_pts = edges[:, :, :2].reshape(-1, 2)

    # -- Bottom face: sort boundary projection by angle, fan-tri --------
    bottom = EMPTY_TRIS
    if len(bot_pts) and frame.border_mm <= 0:
        cx, cy = frame.model_w / 2, frame.model_h / 2
        seen, uniq = set(), []
        for p in bot_pts:
            pk = (round(p[0], 3), round(p[1], 3))
            if pk not in seen:
                seen.add(pk)
                uniq.append((p[0], p[1]))
        # Secondary radius key: points exactly collinear with the centre
        # (angle ties) get a deterministic order regardless of edge order
        uniq.sort(key=lambda p: (math.atan2(p[1] - cy, p[0] - cx),
                                 (p[0] - cx) ** 2 + (p[1] - cy) ** 2))
        n = len(uniq)
        if n >= 3:
            u = np.asarray(uniq)
            p0 = np.column_stack([u, np.zeros(n)])
            p1 = np.roll(p0, -1, axis=0)
            cpt = np.broadcast_to([cx, cy, 0.0], (n, 3))
            bottom = np.stack([cpt, p1, p0], axis=1)  # -Z normal

    return surfaces, crust, body, bottom
