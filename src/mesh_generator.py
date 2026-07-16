"""
3D mesh generator for topographic maps.

Model-space coordinate system (millimetres):
  X → East   (longitude direction)
  Y → North  (latitude direction)
  Z → Up     (elevation)
"""

import math
import struct

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import triangulate as shp_triangulate

RIVER_WIDTH_M = 30.0     # river ribbon half-width on the ground (metres)
RIVER_MIN_MM  = 0.5      # minimum printed river width on the model (mm)


class MeshGenerator:

    def __init__(self, config=None):
        cfg = config or {}
        self.target_size_mm  = cfg.get("target_size_mm",    100.0)
        self.base_mm         = cfg.get("base_thickness_mm",   5.0)
        self.max_ele_mm      = cfg.get("max_ele_height_mm",  20.0)
        self.building_h_mm   = cfg.get("building_height_mm",  2.0)
        self.route_width_mm  = cfg.get("route_width_mm",      1.0)
        self.route_height_mm = cfg.get("route_height_mm",     1.0)
        self.shape           = cfg.get("shape",            "square")
        # Extended border ring (hexagon only): terrain shrinks by border_mm
        # and a base slab with raised text labels surrounds it.
        self.border_mm       = cfg.get("border_mm",           0.0)
        self.border_labels   = cfg.get("border_labels",       [])
        self.text_height_mm  = cfg.get("text_height_mm",      0.8)
        self.last_water_mask = None

    # ── Public API ────────────────────────────────────────────────────────

    def generate_zone_tris(self, elevation_grid, lat_bounds, lon_bounds,
                           buildings=None, water=None, forests=None,
                           tree_line_m=1250.0, detect_ocean_m=None):
        """
        Generate terrain geometry split into colour zones.
        Returns dict of triangle lists:
          {"land", "rock", "water", "buildings", "base"}
        Only the top SURFACE gets terrain colours — side walls and the bottom
        go into "base" so the model sides always match the base colour.

        water/forests: features from fetch_map_features().
        Land/rock split: a cell is vegetated (land) when it lies below the
        tree_line_m altitude OR inside a mapped OSM forest polygon — mapped
        forests give real tree edges, the altitude rule fills unmapped areas.
        detect_ocean_m: if not None, cells with original elevation <= this value
        (metres) are treated as sea/ocean.

        Leaves the generator set up, so route_tris() can be called afterwards.
        Masks stay available as self.last_water_mask / self.last_rock_mask
        (vertex lattice) for the thumbnail renderer.
        """
        self._setup(elevation_grid, lat_bounds, lon_bounds)
        water_mask, grid = self._build_water_features(
            elevation_grid, water, detect_ocean_m=detect_ocean_m)
        self.last_water_mask = water_mask

        # Water bodies may sit below the previous global minimum after
        # carving; refresh the vertical mapping so z stays in range.
        self.ele_min = float(np.nanmin(grid))
        self.ele_max = float(np.nanmax(grid))
        self.sz = self.max_ele_mm / max(self.ele_max - self.ele_min, 1.0)
        self._refresh_interp(grid)

        rows, cols = grid.shape
        rock_mask = self._build_rock_mask(grid, forests, tree_line_m)
        self.last_rock_mask = rock_mask

        surface_tris, wall_tris = self._terrain(grid)

        tris = np.asarray(surface_tris, dtype=np.float64)
        if tris.size == 0:
            zones = {"land": [], "rock": [], "water": []}
        else:
            # Classify by centroid cell. Floor (not round) so both triangles
            # of a quad land in the same cell — otherwise thin features like
            # rivers render dashed.
            cx = tris[:, :, 0].mean(axis=1)
            cy = tris[:, :, 1].mean(axis=1)
            j = np.clip((cx / self.model_w * (cols - 1)).astype(int), 0, cols - 1)
            i = np.clip(((1.0 - cy / self.model_h) * (rows - 1)).astype(int), 0, rows - 1)

            in_water = water_mask[i, j] if water_mask is not None \
                else np.zeros(len(tris), dtype=bool)
            is_rock = rock_mask[i, j] & ~in_water
            is_land = ~in_water & ~is_rock
            zones = {
                "land":  tris[is_land].tolist(),
                "rock":  tris[is_rock].tolist(),
                "water": tris[in_water].tolist(),
            }

        zones["buildings"] = self._buildings(buildings) if buildings else []
        zones["base"] = wall_tris
        return zones

    def _build_rock_mask(self, grid, forests, tree_line_m):
        """
        Vertex-lattice mask of non-vegetated ("rock") terrain: above the tree
        line AND not inside a mapped forest. OSM forest polygons carve real
        tree edges; the altitude rule covers areas OSM hasn't mapped.
        """
        rows, cols = grid.shape
        rock = grid >= tree_line_m
        if forests:
            forest_mask = np.zeros((rows, cols), dtype=bool)
            for f in forests:
                forest_mask |= self._mask_from_feature(f, rows, cols)
            rock &= ~forest_mask
        return rock

    def route_tris(self, gpx_points, flat=False, use_gpx_ele=False):
        """
        Trail ribbon triangles. Requires a prior generate_zone_tris()/_setup().
        flat=True: ribbon at one constant height (median terrain z along the
        path) — used for swims, which must not go up or down.
        use_gpx_ele=True: ribbon height from the GPX file's own elevation
        data instead of the map terrain.
        """
        if not gpx_points or len(gpx_points) < 2:
            return []
        flat_z = None
        if flat:
            zs = [self.z_at(p[0], p[1]) for p in gpx_points[::max(1, len(gpx_points) // 200)]
                  if self._in_bounds(p[0], p[1])]
            if zs:
                flat_z = float(np.median(zs))
        return self._route(gpx_points, flat_z=flat_z,
                           use_gpx_ele=use_gpx_ele and flat_z is None)

    def to_stl_bytes(self, triangles):
        """Serialise a triangle list to binary STL."""
        return self._to_stl(triangles)

    @staticmethod
    def to_obj_bytes(groups):
        """
        Serialise named triangle groups to Wavefront OBJ (one `o` object per
        group, so viewers can colour zones independently).
        groups: dict name -> list of triangles.
        """
        lines = []
        idx = 1
        for name, tris in groups.items():
            if not len(tris):
                continue
            lines.append(f"o {name}")
            for tri in tris:
                for v in tri:
                    lines.append(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}")
                lines.append(f"f {idx} {idx + 1} {idx + 2}")
                idx += 3
        return ("\n".join(lines) + "\n").encode("ascii")

    # ── Coordinate setup ─────────────────────────────────────────────────

    def _frame(self, lat_bounds, lon_bounds):
        """Set the lat/lon → model-mm mapping and the model shape polygons."""
        lat_min, lat_max = lat_bounds
        lon_min, lon_max = lon_bounds
        lat_mid = (lat_min + lat_max) / 2.0

        width_m  = (lon_max - lon_min) * 111_000 * math.cos(math.radians(lat_mid))
        height_m = (lat_max - lat_min) * 111_000
        max_dim  = max(width_m, height_m)

        self.sxy     = self.target_size_mm / max_dim
        self.model_w = width_m  * self.sxy
        self.model_h = height_m * self.sxy

        self.lat_min, self.lat_max = lat_min, lat_max
        self.lon_min, self.lon_max = lon_min, lon_max

        cx, cy = self.model_w / 2, self.model_h / 2
        r = min(cx, cy) * 0.98
        if self.border_mm > 0:
            self._outer_poly = self._make_shape(cx, cy, r)
            self._shape_poly = self._make_shape(cx, cy, max(r - self.border_mm, r * 0.3))
        else:
            self._outer_poly = None
            self._shape_poly = self._make_shape(cx, cy, r)

    def _setup(self, grid, lat_bounds, lon_bounds):
        self._frame(lat_bounds, lon_bounds)

        self.ele_min = float(np.nanmin(grid))
        self.ele_max = float(np.nanmax(grid))
        ele_range    = max(self.ele_max - self.ele_min, 1.0)
        self.sz      = self.max_ele_mm / ele_range

        self.rows, self.cols = grid.shape
        self._refresh_interp(grid)

    # ── Area fitting ─────────────────────────────────────────────────────

    def fit_size_km(self, points, lat_c, lon_c, span_km, margin_frac):
        """
        Smallest map area (km) centred on (lat_c, lon_c) whose model SHAPE
        (hexagon/octagon/circle/square, minus any border ring) contains every
        route point with margin_frac × model-size clearance from the edge.
        A square bounding box isn't enough — shape corners cut into it.
        """
        pts = [(p[0], p[1]) for p in points[::max(1, len(points) // 500)]]
        size = max(span_km, 0.2)
        for _ in range(60):
            if self._points_fit(pts, lat_c, lon_c, size, margin_frac):
                break
            size *= 1.05
        return size

    def _points_fit(self, pts, lat_c, lon_c, size_km, margin_frac):
        lat_d = (size_km / 2.0) / 111.0
        lon_d = (size_km / 2.0) / (111.0 * math.cos(math.radians(lat_c)))
        self._frame((lat_c - lat_d, lat_c + lat_d), (lon_c - lon_d, lon_c + lon_d))

        shrunk = self._shape_poly.buffer(-margin_frac * self.target_size_mm)
        if shrunk.is_empty:
            return False
        for lat, lon in pts:
            if not self._in_bounds(lat, lon):
                return False
            if not shrunk.contains(Point(*self.ll_to_xy(lat, lon))):
                return False
        return True

    def _refresh_interp(self, grid):
        lat_arr = np.linspace(self.lat_max, self.lat_min, grid.shape[0])
        lon_arr = np.linspace(self.lon_min, self.lon_max, grid.shape[1])
        self._interp = RegularGridInterpolator(
            (lat_arr, lon_arr), grid,
            method="linear", bounds_error=False, fill_value=self.ele_min,
        )

    def _make_shape(self, cx, cy, r):
        s = self.shape
        if s == "hexagon":
            # With a labelled border the hexagon is flat-bottom (vertices at
            # 0°,60°,…) so there are true "bottom"/"top" sides to write on.
            off = 0.0 if self.border_mm > 0 else math.pi / 6
            angles = [i * math.pi / 3 + off for i in range(6)]
            pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]
            return Polygon(pts)
        elif s == "octagon":
            angles = [i * math.pi / 4 + math.pi / 8 for i in range(8)]
            pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]
            return Polygon(pts)
        elif s in ("circle", "circle-flat"):
            return Point(cx, cy).buffer(r, resolution=64)
        else:   # square / default
            return Polygon([
                (0, 0), (self.model_w, 0),
                (self.model_w, self.model_h), (0, self.model_h),
            ])

    # ── Coordinate helpers ────────────────────────────────────────────────

    def ll_to_xy(self, lat, lon):
        x = (lon - self.lon_min) / (self.lon_max - self.lon_min) * self.model_w
        y = (lat - self.lat_min) / (self.lat_max - self.lat_min) * self.model_h
        return x, y

    def _xy_to_ll(self, x, y):
        lon = self.lon_min + x / self.model_w * (self.lon_max - self.lon_min)
        lat = self.lat_min + y / self.model_h * (self.lat_max - self.lat_min)
        return lat, lon

    def ele_to_z(self, ele):
        return (ele - self.ele_min) * self.sz + self.base_mm

    def z_at(self, lat, lon):
        return self.ele_to_z(float(self._interp([[lat, lon]])[0]))

    def _in_bounds(self, lat, lon):
        return (self.lat_min <= lat <= self.lat_max and
                self.lon_min <= lon <= self.lon_max)

    # ── Rasterisation helpers (PIL — fast replacement for per-cell contains) ──

    def _rasterize_ll_polys(self, polys, rows, cols):
        """Rasterise lat/lon rings onto the grid → bool mask (row 0 = lat_max)."""
        img = Image.new("1", (cols, rows), 0)
        d = ImageDraw.Draw(img)
        lat_rng = max(self.lat_max - self.lat_min, 1e-12)
        lon_rng = max(self.lon_max - self.lon_min, 1e-12)
        for coords in polys:
            pts = [
                (
                    (lon - self.lon_min) / lon_rng * (cols - 1),
                    (self.lat_max - lat) / lat_rng * (rows - 1),
                )
                for lat, lon in coords
            ]
            if len(pts) >= 3:
                d.polygon(pts, fill=1)
        return np.array(img, dtype=bool)

    def _rasterize_xy_poly(self, poly, rows, cols):
        """Rasterise a shapely polygon in model-xy space → bool mask."""
        img = Image.new("1", (cols, rows), 0)
        d = ImageDraw.Draw(img)
        geoms = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
        for g in geoms:
            if g.is_empty or g.geom_type != "Polygon":
                continue
            pts = [
                (x / self.model_w * (cols - 1), (1.0 - y / self.model_h) * (rows - 1))
                for x, y in g.exterior.coords
            ]
            if len(pts) >= 3:
                d.polygon(pts, fill=1)
        return np.array(img, dtype=bool)

    def _mask_from_feature(self, feature, rows, cols):
        """Mask of a polygon feature: outer ring minus its hole rings
        (islands in lakes, clearings in forests)."""
        mask = self._rasterize_ll_polys([feature["coords"]], rows, cols)
        holes = feature.get("holes") or []
        if mask.any() and holes:
            mask &= ~self._rasterize_ll_polys(holes, rows, cols)
        return mask

    def _rasterize_xy_line(self, pts_xy, rows, cols, width_px=1):
        """Rasterise a polyline in model-xy space → bool mask (every cell hit)."""
        img = Image.new("1", (cols, rows), 0)
        d = ImageDraw.Draw(img)
        pts = [
            (x / self.model_w * (cols - 1), (1.0 - y / self.model_h) * (rows - 1))
            for x, y in pts_xy
        ]
        if len(pts) >= 2:
            d.line(pts, fill=1, width=max(1, int(width_px)))
        return np.array(img, dtype=bool)

    # ── Shared solid-building helpers ────────────────────────────────────

    def _top_tris_from_polys(self, geom, z_fn):
        """Triangulate a (multi)polygon → CCW top-face tris, z from z_fn(x, y)."""
        polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        tris = []
        for poly in polys:
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
            for t in shp_triangulate(poly):
                if not poly.contains(t.centroid):
                    continue
                c = list(t.exterior.coords)[:3]
                tv = [[x, y, z_fn(x, y)] for x, y in c]
                cz = ((tv[1][0]-tv[0][0])*(tv[2][1]-tv[0][1]) -
                      (tv[1][1]-tv[0][1])*(tv[2][0]-tv[0][0]))
                if cz < 0:
                    tv[1], tv[2] = tv[2], tv[1]
                tris.append(tuple(tv))
        return tris

    @staticmethod
    def _boundary_edges(top_tris):
        """Directed edges that appear in exactly one triangle of a surface."""
        PREC = 3

        def vk(v):
            return (round(v[0], PREC), round(v[1], PREC), round(v[2], PREC))

        edge_cnt, edge_dir = {}, {}
        for tri in top_tris:
            for k in range(3):
                a, b = vk(tri[k]), vk(tri[(k + 1) % 3])
                key = (min(a, b), max(a, b))
                edge_cnt[key] = edge_cnt.get(key, 0) + 1
                edge_dir[key] = (list(tri[k]), list(tri[(k + 1) % 3]))
        return [edge_dir[k] for k, c in edge_cnt.items() if c == 1]

    def _prism_tris(self, geom, z_top_fn, z_bottom_fn):
        """
        Closed solid from a (multi)polygon footprint: top face at z_top_fn,
        bottom face (reversed winding) at z_bottom_fn, side walls between.
        Used for trail ribbons and border text.
        """
        top = self._top_tris_from_polys(geom, z_top_fn)
        tris = list(top)
        for tri in top:
            bv = [[v[0], v[1], z_bottom_fn(v[0], v[1])] for v in tri]
            tris.append((bv[0], bv[2], bv[1]))
        for t1, t2 in self._boundary_edges(top):
            b1 = [t1[0], t1[1], z_bottom_fn(t1[0], t1[1])]
            b2 = [t2[0], t2[1], z_bottom_fn(t2[0], t2[1])]
            tris.append((t2, t1, b1))   # outward-facing (right of t1→t2)
            tris.append((t2, b1, b2))
        return tris

    # ── Water ────────────────────────────────────────────────────────────

    def _grid_ele_at_xy(self, grid, xs, ys):
        """Sample grid elevation at model-xy points (nearest cell)."""
        rows, cols = grid.shape
        j = np.clip(np.round(np.asarray(xs) / self.model_w * (cols - 1)).astype(int), 0, cols - 1)
        i = np.clip(np.round((1.0 - np.asarray(ys) / self.model_h) * (rows - 1)).astype(int), 0, rows - 1)
        return grid[i, j]

    def _build_water_features(self, grid, water_features, detect_ocean_m=None):
        """
        Apply water bodies to the terrain. Returns (merged_mask, adjusted_grid).

        - Lakes/reservoirs: flattened to the median elevation under each body
          (each lake sits at its own level).
        - Seas (polygons or detect_ocean_m cells): flattened to the map minimum.
        - Rivers: line buffered to ~30 m ribbon; elevation forced to the
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
                if len(line) >= 2:
                    rivers.append(line)
            elif len(w.get("coords", [])) >= 3:
                (seas if w["type"] == "sea" else lakes).append(w)

        # Lakes: each body flat at its own level; islands (hole rings)
        # stay terrain.
        for lake in lakes:
            mask = self._mask_from_feature(lake, rows, cols)
            if mask.any():
                out[mask] = float(np.median(grid[mask]))
                merged |= mask

        # Seas: flatten to the map minimum
        sea_mask = np.zeros((rows, cols), dtype=bool)
        for sea in seas:
            sea_mask |= self._mask_from_feature(sea, rows, cols)
        if detect_ocean_m is not None:
            sea_mask |= grid <= detect_ocean_m
        if sea_mask.any():
            out[sea_mask] = ele_min
            merged |= sea_mask

        # Rivers: monotonically decreasing along flow direction.
        # Pre-compute each river's ribbon cells + nearest-vertex mapping once,
        # then carve in two passes against the CURRENT grid so junctions with
        # other rivers/lakes stay consistent (a min-only op, so it converges).
        cell_mm = self.model_w / max(cols - 1, 1)
        prepared = []
        for line in rivers:
            raw_xy = np.array([self.ll_to_xy(lat, lon) for lat, lon in line])
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
            ele0 = self._grid_ele_at_xy(grid, pts_xy[:, 0], pts_xy[:, 1])
            k = max(1, len(ele0) // 10)
            if np.mean(ele0[-k:]) > np.mean(ele0[:k]) + 5.0:
                pts_xy = pts_xy[::-1]

            # Ribbon polygon can be thinner than a grid cell — draw the
            # polyline too so every cell under the river is covered.
            # Enforce a minimum printed width so rivers stay visible.
            radius_mm = max(RIVER_WIDTH_M * self.sxy, RIVER_MIN_MM / 2.0)
            ribbon = ls.buffer(radius_mm)
            width_px = max(1, round(2 * radius_mm / cell_mm))
            mask = (self._rasterize_xy_poly(ribbon, rows, cols)
                    | self._rasterize_xy_line(pts_xy, rows, cols, width_px))
            if not mask.any():
                continue
            cells = np.argwhere(mask)
            xs = cells[:, 1] / (cols - 1) * self.model_w
            ys = (1.0 - cells[:, 0] / (rows - 1)) * self.model_h
            _, nearest = cKDTree(pts_xy).query(np.column_stack([xs, ys]))
            prepared.append((pts_xy, cells, nearest))
            merged |= mask

        for _ in range(2 if prepared else 0):
            for pts_xy, cells, nearest in prepared:
                line_ele = self._grid_ele_at_xy(out, pts_xy[:, 0], pts_xy[:, 1])
                running_min = np.minimum.accumulate(line_ele)
                # Broad ribbon cells: value of their nearest sample
                out[cells[:, 0], cells[:, 1]] = np.minimum(
                    out[cells[:, 0], cells[:, 1]], running_min[nearest]
                )
                # Cells the line actually crosses (possibly several times, e.g.
                # zigzags at cell scale): take the MIN over all their samples
                sj = np.clip(np.round(pts_xy[:, 0] / self.model_w * (cols - 1)).astype(int), 0, cols - 1)
                si = np.clip(np.round((1.0 - pts_xy[:, 1] / self.model_h) * (rows - 1)).astype(int), 0, rows - 1)
                np.minimum.at(out, (si, sj), running_min)

        if not merged.any():
            return None, out
        return merged, out

    # ── Terrain solid ─────────────────────────────────────────────────────

    def _terrain(self, elevation_grid):
        rows, cols = elevation_grid.shape

        # Vertex lattice (vectorised)
        xs = np.linspace(0.0, self.model_w, cols)
        ys = np.linspace(self.model_h, 0.0, rows)   # row 0 = north = max y
        X, Y = np.meshgrid(xs, ys)
        Z = self.ele_to_z(elevation_grid)
        V = np.stack([X, Y, Z], axis=-1)

        # Inside-shape mask via rasterisation
        inside = self._rasterize_xy_poly(self._shape_poly, rows, cols)

        # Quad classification
        n_in = (inside[:-1, :-1].astype(np.int8) + inside[:-1, 1:] +
                inside[1:, 1:] + inside[1:, :-1])
        full    = n_in == 4
        partial = (n_in > 0) & (n_in < 4)

        # Only quads near the boundary need slow edge bookkeeping; strictly
        # interior full quads can never contribute boundary edges. The grid
        # edge itself is a boundary too (square shapes fill the whole grid).
        padded = np.ones((full.shape[0] + 2, full.shape[1] + 2), dtype=bool)
        padded[1:-1, 1:-1] = ~full
        near_boundary = binary_dilation(padded, iterations=2)[1:-1, 1:-1]
        fast_full = full & ~near_boundary
        slow_full = full & near_boundary

        # Fast path: bulk-emit interior full quads
        top_tris = []
        fi, fj = np.nonzero(fast_full)
        if len(fi):
            v00 = V[fi, fj]; v01 = V[fi, fj + 1]
            v10 = V[fi + 1, fj]; v11 = V[fi + 1, fj + 1]
            bulk = np.concatenate([
                np.stack([v00, v11, v01], axis=1),
                np.stack([v00, v10, v11], axis=1),
            ])
            fast_tris = bulk.tolist()
        else:
            fast_tris = []

        # Slow path: boundary-zone quads (full near boundary + clipped partial)
        si, sj = np.nonzero(slow_full | partial)
        for i, j in zip(si.tolist(), sj.tolist()):
            if slow_full[i, j]:
                v00 = V[i,   j  ].tolist(); v01 = V[i,   j+1].tolist()
                v10 = V[i+1, j  ].tolist(); v11 = V[i+1, j+1].tolist()
                top_tris.append((v00, v11, v01))
                top_tris.append((v00, v10, v11))
            else:
                quad_poly = Polygon([
                    (V[i,j][0],     V[i,j][1]),
                    (V[i,j+1][0],   V[i,j+1][1]),
                    (V[i+1,j+1][0], V[i+1,j+1][1]),
                    (V[i+1,j][0],   V[i+1,j][1]),
                ])
                clipped = self._shape_poly.intersection(quad_poly)
                if clipped.is_empty:
                    continue
                geoms = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
                for geom in geoms:
                    if geom.geom_type != "Polygon" or geom.is_empty:
                        continue
                    pts2d = list(geom.exterior.coords[:-1])
                    if len(pts2d) < 3:
                        continue
                    verts = []
                    for px, py in pts2d:
                        la, lo = self._xy_to_ll(px, py)
                        verts.append([px, py, self.z_at(la, lo)])
                    for k in range(1, len(verts) - 1):
                        v0, v1, v2 = verts[0], verts[k], verts[k+1]
                        cz = (v1[0]-v0[0])*(v2[1]-v0[1]) - (v1[1]-v0[1])*(v2[0]-v0[0])
                        top_tris.append((v0, v1, v2) if cz >= 0 else (v0, v2, v1))

        surface = fast_tris + list(top_tris)
        walls = []
        bot_pts = []

        # With a border, the terrain sits ON the base slab: walls stop at the
        # slab top and the slab provides the bottom face.
        floor_z = self.base_mm if self.border_mm > 0 else 0.0

        # ── Smooth walls: drop each boundary edge straight to the floor ──
        for t1, t2 in self._boundary_edges(top_tris):
            b1 = [t1[0], t1[1], floor_z]
            b2 = [t2[0], t2[1], floor_z]
            walls.append((t2, t1, b1))   # outward-facing (right of t1→t2)
            walls.append((t2, b1, b2))
            bot_pts.append((t1[0], t1[1]))
            bot_pts.append((t2[0], t2[1]))

        # ── Bottom face: sort boundary projection by angle, fan-tri ──────
        if bot_pts and self.border_mm <= 0:
            cx, cy = self.model_w / 2, self.model_h / 2
            seen, uniq = set(), []
            for p in bot_pts:
                pk = (round(p[0], 3), round(p[1], 3))
                if pk not in seen:
                    seen.add(pk); uniq.append(p)
            uniq.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
            n = len(uniq)
            if n >= 3:
                cpt = [cx, cy, 0.0]
                for k in range(n):
                    p0 = [uniq[k][0],         uniq[k][1],         0.0]
                    p1 = [uniq[(k+1) % n][0], uniq[(k+1) % n][1], 0.0]
                    walls.append((cpt, p1, p0))  # -Z normal

        return surface, walls

    # ── Border ring + text labels (hexagon) ──────────────────────────────

    # Label slots in UI order → hexagon side mid-angle (flat-bottom hexagon)
    SIDE_ANGLES_DEG = [90, 30, 330, 270, 210, 150]
    # order: top, upper-right, lower-right, bottom, lower-left, upper-left

    def border_tris(self):
        """
        Base slab (full outer hexagon, z 0 → base thickness) plus raised text
        on the border ring, one label per hexagon side.
        Returns {"base": [...], "text": [...]}.
        Requires a prior generate_zone_tris()/_setup().
        """
        if self.border_mm <= 0 or self._outer_poly is None:
            return {"base": [], "text": []}

        # Slab prism: hexagon top at base_mm (under the terrain — internal
        # where covered, visible on the ring), bottom at 0, outer walls.
        base = []
        coords = list(self._outer_poly.exterior.coords)[:-1]
        c = self._outer_poly.centroid
        pcx, pcy = c.x, c.y
        n = len(coords)
        for k in range(n):
            ax, ay = coords[k]
            bx, by = coords[(k + 1) % n]
            base.append(([pcx, pcy, self.base_mm], [ax, ay, self.base_mm], [bx, by, self.base_mm]))
            base.append(([pcx, pcy, 0.0], [bx, by, 0.0], [ax, ay, 0.0]))
            base.append(([ax, ay, 0.0], [bx, by, self.base_mm], [bx, by, 0.0]))
            base.append(([ax, ay, 0.0], [ax, ay, self.base_mm], [bx, by, self.base_mm]))

        if self.shape != "hexagon":
            return {"base": base, "text": []}

        text = []
        cx, cy = self.model_w / 2, self.model_h / 2
        R = min(cx, cy) * 0.98            # outer hexagon circumradius = side length
        apothem = R * math.cos(math.pi / 6)
        band_center = apothem - 0.433 * self.border_mm  # radial middle of the ring

        for label, ang in zip(self.border_labels, self.SIDE_ANGLES_DEG):
            label = (label or "").strip()
            if not label:
                continue
            th = math.radians(ang)
            u = (math.cos(th), math.sin(th))        # outward normal of this side
            inward = (-u[0], -u[1])
            # Keep text upright: its "up" must point skyward on the model
            up = inward if inward[1] > 0 else u
            dirv = (up[1], -up[0])                  # reading direction (left→right)

            glyphs = self._text_polygons(label, self.border_mm * 0.5)
            if glyphs is None or glyphs.is_empty:
                continue
            minx, miny, maxx, maxy = glyphs.bounds
            gx, gy = (minx + maxx) / 2, (miny + maxy) / 2
            sc = min(1.0, (R * 0.9) / max(maxx - minx, 1e-6))

            px = cx + band_center * u[0]
            py = cy + band_center * u[1]
            # local (x,y) → world: P + (x-gx)·sc·dirv + (y-gy)·sc·up
            a, b_ = dirv[0] * sc, up[0] * sc
            d, e  = dirv[1] * sc, up[1] * sc
            from shapely.affinity import affine_transform
            placed = affine_transform(
                glyphs, [a, b_, d, e, px - a * gx - b_ * gy, py - d * gx - e * gy])
            top_z = self.base_mm + self.text_height_mm
            text.extend(self._prism_tris(
                placed, lambda x, y: top_z, lambda x, y: self.base_mm))

        return {"base": base, "text": text}

    @staticmethod
    def _text_polygons(text, size_mm):
        """Text → shapely geometry (even-odd fill handles letter holes)."""
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextPath

        tp = TextPath((0, 0), text, size=size_mm,
                      prop=FontProperties(family="DejaVu Sans", weight="bold"))
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

    # ── Buildings ────────────────────────────────────────────────────────

    def _buildings(self, buildings):
        tris = []
        for bld in buildings:
            coords = bld.get("coords", [])
            if len(coords) < 3:
                continue
            lats = [c[0] for c in coords]
            lons = [c[1] for c in coords]
            if (max(lats) < self.lat_min or min(lats) > self.lat_max or
                    max(lons) < self.lon_min or min(lons) > self.lon_max):
                continue

            xy = [self.ll_to_xy(lat, lon) for lat, lon in coords]
            base_z = float(np.mean([self.z_at(lat, lon) for lat, lon in coords[:4]]))
            top_z  = base_z + self.building_h_mm
            n      = len(xy)

            for k in range(n):
                x0, y0 = xy[k]
                x1, y1 = xy[(k + 1) % n]
                tris.append(([x0,y0,base_z], [x1,y1,top_z], [x1,y1,base_z]))
                tris.append(([x0,y0,base_z], [x0,y0,top_z], [x1,y1,top_z]))

            cx_ = sum(p[0] for p in xy) / n
            cy_ = sum(p[1] for p in xy) / n
            for k in range(n):
                x0, y0 = xy[k]
                x1, y1 = xy[(k + 1) % n]
                tris.append(([cx_,cy_,top_z], [x0,y0,top_z], [x1,y1,top_z]))

        return tris

    # ── GPX route ribbon ─────────────────────────────────────────────────

    def _route(self, points, flat_z=None, use_gpx_ele=False):
        hw     = self.route_width_mm / 2.0
        raise_ = self.route_height_mm
        MIN_D  = hw * 0.5   # minimum spacing to remove GPS jitter

        # Convert to model 2D, skip out-of-bounds, decimate
        pts2d = []
        for p in points:
            lat, lon = p[0], p[1]
            if not self._in_bounds(lat, lon):
                continue
            x, y = self.ll_to_xy(lat, lon)
            if pts2d and math.hypot(x - pts2d[-1][0], y - pts2d[-1][1]) < MIN_D:
                continue
            pts2d.append((x, y))

        if len(pts2d) < 2:
            return []

        def z_terrain(x, y):
            lat, lon = self._xy_to_ll(x, y)
            return self.z_at(lat, lon)

        if flat_z is not None:
            def z_top(x, y):
                return flat_z + raise_

            z_bottom = (lambda x, y: flat_z)
        elif use_gpx_ele:
            # Height from the GPX file's own elevation (nearest track point)
            in_pts = [(self.ll_to_xy(p[0], p[1]), p[2]) for p in points
                      if self._in_bounds(p[0], p[1])]
            kd = cKDTree([xy for xy, _ in in_pts])
            eles = np.array([e for _, e in in_pts])

            def z_gpx(x, y):
                return self.ele_to_z(float(eles[kd.query((x, y))[1]]))

            def z_top(x, y):
                return z_gpx(x, y) + raise_

            def z_bottom(x, y):
                # Anchor into the terrain where the GPX dips below it
                return min(z_gpx(x, y), z_terrain(x, y))
        else:
            def z_top(x, y):
                return z_terrain(x, y) + raise_

            z_bottom = z_terrain

        # Simplify before buffering to remove GPS-noise zigzags
        path = LineString(pts2d).simplify(hw * 0.8, preserve_topology=True)

        # Buffer the 2D path → smooth ribbon (round joins, flat end caps),
        # clipped to the model shape so the trail never spills over the edge
        ribbon = path.buffer(hw, cap_style=2, join_style=1, resolution=8)
        ribbon = ribbon.intersection(self._shape_poly)
        if ribbon.is_empty:
            return []

        return self._prism_tris(ribbon, z_top, z_bottom)

    # ── Binary STL serialisation (vectorised) ────────────────────────────

    def _to_stl(self, triangles):
        tris = np.asarray(triangles, dtype=np.float64)
        if tris.size == 0:
            return b"\x00" * 80 + struct.pack("<I", 0)
        tris = tris.reshape(-1, 3, 3)

        a = tris[:, 1] - tris[:, 0]
        b = tris[:, 2] - tris[:, 0]
        n = np.cross(a, b)
        L = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.where(L > 1e-10, n / np.maximum(L, 1e-30), 0.0)

        rec = np.zeros(len(tris), dtype=[
            ("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2"),
        ])
        rec["n"] = n
        rec["v"] = tris
        return b"\x00" * 80 + struct.pack("<I", len(tris)) + rec.tobytes()
