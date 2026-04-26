"""
3D mesh generator for topographic maps.

Model-space coordinate system (millimetres):
  X → East   (longitude direction)
  Y → North  (latitude direction)
  Z → Up     (elevation)
"""

import base64
import io
import math
import struct

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import triangulate as shp_triangulate, unary_union


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
        self.tree_line_m     = cfg.get("tree_line_m",        1250.0)

    # ── Public API ────────────────────────────────────────────────────────

    def generate_components_b64(self, elevation_grid, lat_bounds, lon_bounds,
                                 buildings=None, water=None, gpx_points=None,
                                 tree_line_m=1250.0):
        """
        Generate the real STL geometry split into colour zones.
        Returns dict of base64-encoded binary STL strings:
          {"land": str, "rock": str, "trail": str}

        Triangles whose average Z is below the tree-line Z threshold go to
        "land"; those above go to "rock".  Route ribbon goes to "trail".
        """
        self._setup(elevation_grid, lat_bounds, lon_bounds)
        grid = self._apply_water(elevation_grid, self._build_water_mask(elevation_grid, water))

        all_terrain = self._terrain(grid)

        # Z in model-space that corresponds to tree_line_m elevation
        tree_z = self.ele_to_z(tree_line_m)

        land_tris, rock_tris = [], []
        for tri in all_terrain:
            avg_z = (tri[0][2] + tri[1][2] + tri[2][2]) / 3.0
            (rock_tris if avg_z >= tree_z else land_tris).append(tri)

        if buildings:
            land_tris.extend(self._buildings(buildings))

        route_tris = self._route(gpx_points) if gpx_points and len(gpx_points) >= 2 else []

        def enc(tris):
            return base64.b64encode(self._to_stl(tris)).decode("ascii")

        return {"land": enc(land_tris), "rock": enc(rock_tris), "trail": enc(route_tris)}

    def generate_bytes(self, elevation_grid, lat_bounds, lon_bounds,
                       buildings=None, water=None, gpx_points=None):
        """Full map: terrain + optional buildings + optional route."""
        self._setup(elevation_grid, lat_bounds, lon_bounds)
        grid = self._apply_water(elevation_grid, self._build_water_mask(elevation_grid, water))
        tris = []
        tris.extend(self._terrain(grid))
        if buildings:
            tris.extend(self._buildings(buildings))
        if gpx_points and len(gpx_points) >= 2:
            tris.extend(self._route(gpx_points))
        return self._to_stl(tris)

    def generate_trail_only_bytes(self, elevation_grid, lat_bounds, lon_bounds, gpx_points):
        """Trail ribbon only (for separate-filament printing)."""
        self._setup(elevation_grid, lat_bounds, lon_bounds)
        if not gpx_points or len(gpx_points) < 2:
            return self._to_stl([])
        return self._to_stl(self._route(gpx_points))

    # ── Coordinate setup ─────────────────────────────────────────────────

    def _setup(self, grid, lat_bounds, lon_bounds):
        rows, cols = grid.shape
        lat_min, lat_max = lat_bounds
        lon_min, lon_max = lon_bounds
        lat_mid = (lat_min + lat_max) / 2.0

        width_m  = (lon_max - lon_min) * 111_000 * math.cos(math.radians(lat_mid))
        height_m = (lat_max - lat_min) * 111_000
        max_dim  = max(width_m, height_m)

        self.sxy     = self.target_size_mm / max_dim
        self.model_w = width_m  * self.sxy
        self.model_h = height_m * self.sxy

        self.ele_min = float(np.nanmin(grid))
        self.ele_max = float(np.nanmax(grid))
        ele_range    = max(self.ele_max - self.ele_min, 1.0)
        self.sz      = self.max_ele_mm / ele_range

        self.lat_min, self.lat_max = lat_min, lat_max
        self.lon_min, self.lon_max = lon_min, lon_max
        self.rows, self.cols = rows, cols

        lat_arr = np.linspace(lat_max, lat_min, rows)
        lon_arr = np.linspace(lon_min, lon_max, cols)
        self._interp = RegularGridInterpolator(
            (lat_arr, lon_arr), grid,
            method="linear", bounds_error=False, fill_value=self.ele_min,
        )

        cx, cy = self.model_w / 2, self.model_h / 2
        r = min(cx, cy) * 0.98
        self._shape_poly = self._make_shape(cx, cy, r)

    def _make_shape(self, cx, cy, r):
        s = self.shape
        if s == "hexagon":
            angles = [i * math.pi / 3 + math.pi / 6 for i in range(6)]
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

    # ── Water ────────────────────────────────────────────────────────────

    def _build_water_mask(self, grid, water_features):
        if not water_features:
            return None
        polys = []
        for w in water_features:
            coords = w.get("coords", [])
            if len(coords) >= 3:
                try:
                    # Shapely polygon expects (lon, lat) = (x, y)
                    polys.append(Polygon([(c[1], c[0]) for c in coords]))
                except Exception:
                    pass
        if not polys:
            return None
        water_union = unary_union(polys)

        rows, cols = grid.shape
        mask = np.zeros((rows, cols), dtype=bool)
        lat_arr = np.linspace(self.lat_max, self.lat_min, rows)
        lon_arr = np.linspace(self.lon_min, self.lon_max, cols)
        for i in range(rows):
            for j in range(cols):
                mask[i, j] = water_union.contains(Point(lon_arr[j], lat_arr[i]))
        return mask

    def _apply_water(self, grid, mask):
        if mask is None:
            return grid
        out = grid.copy()
        out[mask] = self.ele_min   # flatten water areas to minimum elevation
        return out

    # ── Terrain solid ─────────────────────────────────────────────────────

    def _terrain(self, elevation_grid):
        rows, cols = elevation_grid.shape
        V = np.zeros((rows, cols, 3))
        inside = np.zeros((rows, cols), dtype=bool)

        for i in range(rows):
            for j in range(cols):
                lat = self.lat_max - i * (self.lat_max - self.lat_min) / (rows - 1)
                lon = self.lon_min + j * (self.lon_max - self.lon_min) / (cols - 1)
                x, y = self.ll_to_xy(lat, lon)
                V[i, j] = [x, y, self.ele_to_z(elevation_grid[i, j])]
                inside[i, j] = self._shape_poly.contains(Point(x, y))

        # ── Top surface: full quads + Shapely-clipped boundary quads ─────
        top_tris = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                n_in = int(inside[i,j]) + int(inside[i,j+1]) + int(inside[i+1,j+1]) + int(inside[i+1,j])
                if n_in == 0:
                    continue
                if n_in == 4:
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

        # ── Detect boundary edges (appear in exactly 1 triangle) ─────────
        PREC = 3
        def vk(v): return (round(v[0], PREC), round(v[1], PREC), round(v[2], PREC))

        edge_cnt = {}
        edge_dir = {}
        for tri in top_tris:
            for k in range(3):
                a, b = vk(tri[k]), vk(tri[(k+1) % 3])
                key = (min(a, b), max(a, b))
                edge_cnt[key] = edge_cnt.get(key, 0) + 1
                edge_dir[key] = (tri[k], tri[(k+1) % 3])

        tris = list(top_tris)
        bot_pts = []

        # ── Smooth walls: drop each boundary edge straight to z = 0 ──────
        for key, cnt in edge_cnt.items():
            if cnt != 1:
                continue
            t1, t2 = edge_dir[key]
            b1 = [t1[0], t1[1], 0.0]
            b2 = [t2[0], t2[1], 0.0]
            tris.append((t2, t1, b1))   # outward-facing (right of t1→t2)
            tris.append((t2, b1, b2))
            bot_pts.append((t1[0], t1[1]))
            bot_pts.append((t2[0], t2[1]))

        # ── Bottom face: sort boundary projection by angle, fan-tri ──────
        if bot_pts:
            cx, cy = self.model_w / 2, self.model_h / 2
            seen, uniq = set(), []
            for p in bot_pts:
                pk = (round(p[0], PREC), round(p[1], PREC))
                if pk not in seen:
                    seen.add(pk); uniq.append(p)
            uniq.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
            n = len(uniq)
            if n >= 3:
                cpt = [cx, cy, 0.0]
                for k in range(n):
                    p0 = [uniq[k][0],         uniq[k][1],         0.0]
                    p1 = [uniq[(k+1) % n][0], uniq[(k+1) % n][1], 0.0]
                    tris.append((cpt, p1, p0))  # -Z normal

        return tris

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

    def _route(self, points):
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

        def z_xy(x, y):
            lat, lon = self._xy_to_ll(x, y)
            return self.z_at(lat, lon)

        # Simplify before buffering to remove GPS-noise zigzags
        path = LineString(pts2d).simplify(hw * 0.8, preserve_topology=True)

        # Buffer the 2D path → smooth ribbon (round joins, flat end caps)
        ribbon = path.buffer(hw, cap_style=2, join_style=1, resolution=8)
        if ribbon.is_empty:
            return []

        polys = list(ribbon.geoms) if isinstance(ribbon, MultiPolygon) else [ribbon]

        PREC = 3

        def vk(v):
            return (round(v[0], PREC), round(v[1], PREC), round(v[2], PREC))

        top_tris = []
        for poly in polys:
            for t in shp_triangulate(poly):
                if not poly.contains(t.centroid):
                    continue
                c = list(t.exterior.coords)[:3]
                tv = [[x, y, z_xy(x, y) + raise_] for x, y in c]
                cz = ((tv[1][0]-tv[0][0])*(tv[2][1]-tv[0][1]) -
                      (tv[1][1]-tv[0][1])*(tv[2][0]-tv[0][0]))
                if cz < 0:
                    tv[1], tv[2] = tv[2], tv[1]
                top_tris.append(tuple(tv))

        tris = list(top_tris)

        # Bottom face (reversed winding, at terrain surface)
        for tri in top_tris:
            bv = [[v[0], v[1], z_xy(v[0], v[1])] for v in tri]
            tris.append((bv[0], bv[2], bv[1]))

        # Side walls from boundary edges of the top surface
        edge_cnt = {}
        edge_dir = {}
        for tri in top_tris:
            for k in range(3):
                a, b = vk(tri[k]), vk(tri[(k+1) % 3])
                key = (min(a, b), max(a, b))
                edge_cnt[key] = edge_cnt.get(key, 0) + 1
                edge_dir[key] = (list(tri[k]), list(tri[(k+1) % 3]))

        for key, cnt in edge_cnt.items():
            if cnt != 1:
                continue
            t1, t2 = edge_dir[key]
            b1 = [t1[0], t1[1], z_xy(t1[0], t1[1])]
            b2 = [t2[0], t2[1], z_xy(t2[0], t2[1])]
            tris.append((t2, t1, b1))
            tris.append((t2, b1, b2))

        return tris

    # ── Binary STL serialisation ─────────────────────────────────────────

    @staticmethod
    def _normal(tri):
        a = np.subtract(tri[1], tri[0])
        b = np.subtract(tri[2], tri[0])
        n = np.cross(a, b)
        L = np.linalg.norm(n)
        return (n / L).astype(np.float32) if L > 1e-10 else np.zeros(3, np.float32)

    def _to_stl(self, triangles):
        buf = io.BytesIO()
        buf.write(b"\x00" * 80)
        buf.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            buf.write(struct.pack("<3f", *self._normal(tri)))
            for v in tri:
                buf.write(struct.pack("<3f", float(v[0]), float(v[1]), float(v[2])))
            buf.write(struct.pack("<H", 0))
        return buf.getvalue()
