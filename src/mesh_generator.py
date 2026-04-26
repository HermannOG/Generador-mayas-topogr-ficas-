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
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


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

        def fq(i, j):
            return (0 <= i < rows-1 and 0 <= j < cols-1 and
                    bool(inside[i,j]) and bool(inside[i,j+1]) and
                    bool(inside[i+1,j]) and bool(inside[i+1,j+1]))

        tris = []

        for i in range(rows - 1):
            for j in range(cols - 1):
                if not fq(i, j):
                    continue

                v00 = V[i,   j  ].tolist()  # NW
                v01 = V[i,   j+1].tolist()  # NE
                v10 = V[i+1, j  ].tolist()  # SW
                v11 = V[i+1, j+1].tolist()  # SE
                b00 = [v00[0], v00[1], 0.0]
                b01 = [v01[0], v01[1], 0.0]
                b10 = [v10[0], v10[1], 0.0]
                b11 = [v11[0], v11[1], 0.0]

                # Top face (normal +Z)
                tris.append((v00, v11, v01))
                tris.append((v00, v10, v11))

                # Bottom face (normal -Z)
                tris.append((b00, b01, b11))
                tris.append((b00, b11, b10))

                # North wall (+Y) – exposed if no full quad above
                if not fq(i-1, j):
                    tris.append((v00, v01, b01))
                    tris.append((v00, b01, b00))

                # South wall (-Y) – exposed if no full quad below
                if not fq(i+1, j):
                    tris.append((v11, v10, b10))
                    tris.append((v11, b10, b11))

                # West wall (-X) – exposed if no full quad to the left
                if not fq(i, j-1):
                    tris.append((v10, v00, b10))
                    tris.append((v00, b00, b10))

                # East wall (+X) – exposed if no full quad to the right
                if not fq(i, j+1):
                    tris.append((v01, v11, b11))
                    tris.append((v01, b11, b01))

        return tris

    def _add_square_base(self, tris, V, rows, cols):
        w, h = self.model_w, self.model_h

        # Bottom face (normal → −Z, CW from above)
        tris.append(([0,0,0], [w,h,0], [w,0,0]))
        tris.append(([0,0,0], [0,h,0], [w,h,0]))

        def wall(edge, flip):
            for k in range(len(edge) - 1):
                t, tn = edge[k], edge[k+1]
                b  = [t[0],  t[1],  0.0]
                bn = [tn[0], tn[1], 0.0]
                if not flip:
                    tris.append((list(t),  b,  bn)); tris.append((list(t), bn, list(tn)))
                else:
                    tris.append((list(t), bn,   b)); tris.append((list(t), list(tn), bn))

        wall([V[rows-1, j] for j in range(cols)], flip=False)   # south
        wall([V[0,      j] for j in range(cols)], flip=True)    # north
        wall([V[i,      0] for i in range(rows)], flip=True)    # west
        wall([V[i, cols-1] for i in range(rows)], flip=False)   # east

    def _add_shaped_base(self, tris):
        cx, cy = self.model_w / 2, self.model_h / 2
        outline = list(self._shape_poly.exterior.coords)

        # Bottom face (fan from centre, normal → −Z)
        for k in range(len(outline) - 1):
            p0 = [outline[k][0],   outline[k][1],   0]
            p1 = [outline[k+1][0], outline[k+1][1], 0]
            pc = [cx, cy, 0]
            tris.append((pc, p1, p0))

        # Side walls along shape boundary
        for k in range(len(outline) - 1):
            x0, y0 = outline[k]
            x1, y1 = outline[k+1]
            lat0, lon0 = self._xy_to_ll(x0, y0)
            lat1, lon1 = self._xy_to_ll(x1, y1)
            z0t = self.z_at(lat0, lon0)
            z1t = self.z_at(lat1, lon1)
            tris.append(([x0,y0,0],    [x0,y0,z0t], [x1,y1,z1t]))
            tris.append(([x0,y0,0],    [x1,y1,z1t], [x1,y1,0]))

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
        tris  = []
        hw    = self.route_width_mm / 2.0
        raise_ = self.route_height_mm

        for i in range(len(points) - 1):
            lat0, lon0 = points[i][0],   points[i][1]
            lat1, lon1 = points[i+1][0], points[i+1][1]

            if not (self._in_bounds(lat0, lon0) or self._in_bounds(lat1, lon1)):
                continue

            x0, y0 = self.ll_to_xy(lat0, lon0)
            x1, y1 = self.ll_to_xy(lat1, lon1)
            z0t = self.z_at(lat0, lon0) + raise_
            z1t = self.z_at(lat1, lon1) + raise_
            z0b = z0t - raise_
            z1b = z1t - raise_

            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue

            nx, ny = -dy / length * hw, dx / length * hw

            p0l = [x0+nx, y0+ny, z0t];  p0r = [x0-nx, y0-ny, z0t]
            p1l = [x1+nx, y1+ny, z1t];  p1r = [x1-nx, y1-ny, z1t]
            p0lb = [x0+nx, y0+ny, z0b]; p0rb = [x0-nx, y0-ny, z0b]
            p1lb = [x1+nx, y1+ny, z1b]; p1rb = [x1-nx, y1-ny, z1b]

            # Top face
            tris.append((p0l, p1r, p0r));  tris.append((p0l, p1l, p1r))
            # Left wall
            tris.append((p0l, p0lb, p1lb)); tris.append((p0l, p1lb, p1l))
            # Right wall
            tris.append((p0rb, p0r, p1r));  tris.append((p0rb, p1r, p1rb))

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
