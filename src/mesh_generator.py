"""
3D mesh generator for topographic maps.

Coordinate convention (model space, millimetres):
  X  →  East  (longitude direction)
  Y  →  North (latitude direction)
  Z  →  Up    (elevation)
"""

import io
import math
import struct

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class MeshGenerator:
    def __init__(self, config=None):
        cfg = config or {}
        self.target_size_mm   = cfg.get("target_size_mm",    150.0)
        self.base_mm          = cfg.get("base_thickness_mm",   3.0)
        self.max_ele_mm       = cfg.get("max_ele_height_mm",  20.0)
        self.building_h_mm    = cfg.get("building_height_mm",  2.0)
        self.route_width_mm   = cfg.get("route_width_mm",      1.0)
        self.route_height_mm  = cfg.get("route_height_mm",     1.0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_bytes(self, elevation_grid, lat_bounds, lon_bounds,
                       buildings=None, gpx_points=None):
        """Return binary STL bytes ready for download."""
        self._setup(elevation_grid, lat_bounds, lon_bounds)
        triangles = []
        triangles.extend(self._terrain(elevation_grid))
        if buildings:
            triangles.extend(self._buildings(buildings))
        if gpx_points and len(gpx_points) >= 2:
            triangles.extend(self._route(gpx_points))
        return self._to_stl_bytes(triangles)

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def _setup(self, grid, lat_bounds, lon_bounds):
        rows, cols = grid.shape
        lat_min, lat_max = lat_bounds
        lon_min, lon_max = lon_bounds
        lat_mid = (lat_min + lat_max) / 2.0

        width_m  = (lon_max - lon_min) * 111_000 * math.cos(math.radians(lat_mid))
        height_m = (lat_max - lat_min) * 111_000
        max_dim  = max(width_m, height_m)

        self.sxy      = self.target_size_mm / max_dim
        self.model_w  = width_m  * self.sxy
        self.model_h  = height_m * self.sxy

        self.ele_min  = float(np.nanmin(grid))
        self.ele_max  = float(np.nanmax(grid))
        ele_range     = max(self.ele_max - self.ele_min, 1.0)
        self.sz       = self.max_ele_mm / ele_range

        self.lat_min, self.lat_max = lat_min, lat_max
        self.lon_min, self.lon_max = lon_min, lon_max

        lat_arr = np.linspace(lat_max, lat_min, rows)
        lon_arr = np.linspace(lon_min, lon_max, cols)
        self._interp = RegularGridInterpolator(
            (lat_arr, lon_arr), grid,
            method="linear", bounds_error=False, fill_value=self.ele_min,
        )

    def ll_to_xy(self, lat, lon):
        x = (lon - self.lon_min) / (self.lon_max - self.lon_min) * self.model_w
        y = (lat - self.lat_min) / (self.lat_max - self.lat_min) * self.model_h
        return x, y

    def ele_to_z(self, ele):
        return (ele - self.ele_min) * self.sz + self.base_mm

    def z_at(self, lat, lon):
        e = float(self._interp([[lat, lon]])[0])
        return self.ele_to_z(e)

    def _in_bounds(self, lat, lon):
        return (self.lat_min <= lat <= self.lat_max and
                self.lon_min <= lon <= self.lon_max)

    # ------------------------------------------------------------------
    # Terrain solid (watertight)
    # ------------------------------------------------------------------

    def _terrain(self, grid):
        rows, cols = grid.shape
        V = np.empty((rows, cols, 3))
        for i in range(rows):
            for j in range(cols):
                lat = self.lat_max - i * (self.lat_max - self.lat_min) / (rows - 1)
                lon = self.lon_min + j * (self.lon_max - self.lon_min) / (cols - 1)
                x, y = self.ll_to_xy(lat, lon)
                V[i, j] = (x, y, self.ele_to_z(grid[i, j]))

        tris = []

        # Top surface
        for i in range(rows - 1):
            for j in range(cols - 1):
                a, b = V[i, j],   V[i,   j+1]
                c, d = V[i+1, j], V[i+1, j+1]
                tris.append((a, b, d))
                tris.append((a, d, c))

        # Bottom face (normal → −Z)
        w, h = self.model_w, self.model_h
        bl = [[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]]
        tris.append((bl[0], bl[2], bl[1]))
        tris.append((bl[0], bl[3], bl[2]))

        def wall(edge, flip):
            for k in range(len(edge) - 1):
                t, tn = edge[k], edge[k + 1]
                b  = [t[0],  t[1],  0.0]
                bn = [tn[0], tn[1], 0.0]
                if not flip:
                    tris.append((list(t), b,  bn))
                    tris.append((list(t), bn, list(tn)))
                else:
                    tris.append((list(t), bn, b))
                    tris.append((list(t), list(tn), bn))

        wall([V[rows-1, j] for j in range(cols)], flip=False)   # south
        wall([V[0,      j] for j in range(cols)], flip=True)    # north
        wall([V[i,      0] for i in range(rows)], flip=True)    # west
        wall([V[i, cols-1] for i in range(rows)], flip=False)   # east

        return tris

    # ------------------------------------------------------------------
    # Buildings
    # ------------------------------------------------------------------

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

            xy    = [self.ll_to_xy(lat, lon) for lat, lon in coords]
            sample = coords[:min(4, len(coords))]
            base_z = np.mean([self.z_at(lat, lon) for lat, lon in sample])
            top_z  = base_z + self.building_h_mm
            n      = len(xy)

            for k in range(n):
                x0, y0 = xy[k]
                x1, y1 = xy[(k + 1) % n]
                tris.append(([x0, y0, base_z], [x1, y1, top_z], [x1, y1, base_z]))
                tris.append(([x0, y0, base_z], [x0, y0, top_z], [x1, y1, top_z]))

            cx = sum(p[0] for p in xy) / n
            cy = sum(p[1] for p in xy) / n
            for k in range(n):
                x0, y0 = xy[k]
                x1, y1 = xy[(k + 1) % n]
                tris.append(([cx, cy, top_z], [x0, y0, top_z], [x1, y1, top_z]))

        return tris

    # ------------------------------------------------------------------
    # GPX route ribbon
    # ------------------------------------------------------------------

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
            z0 = self.z_at(lat0, lon0) + raise_
            z1 = self.z_at(lat1, lon1) + raise_

            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue

            nx, ny = -dy / length * hw, dx / length * hw

            p0l = [x0 + nx, y0 + ny, z0]
            p0r = [x0 - nx, y0 - ny, z0]
            p1l = [x1 + nx, y1 + ny, z1]
            p1r = [x1 - nx, y1 - ny, z1]

            # Top ribbon
            tris.append((p0l, p1r, p0r))
            tris.append((p0l, p1l, p1r))

            # Side walls (so the ribbon has thickness)
            base_z0 = self.z_at(lat0, lon0)
            base_z1 = self.z_at(lat1, lon1)
            p0lb = [x0 + nx, y0 + ny, base_z0]
            p0rb = [x0 - nx, y0 - ny, base_z0]
            p1lb = [x1 + nx, y1 + ny, base_z1]
            p1rb = [x1 - nx, y1 - ny, base_z1]

            tris.append((p0l,  p0lb, p1lb))
            tris.append((p0l,  p1lb, p1l))
            tris.append((p0rb, p0r,  p1r))
            tris.append((p0rb, p1r,  p1rb))

        return tris

    # ------------------------------------------------------------------
    # Binary STL serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normal(t):
        a = np.subtract(t[1], t[0])
        b = np.subtract(t[2], t[0])
        n = np.cross(a, b)
        length = np.linalg.norm(n)
        return (n / length).astype(np.float32) if length > 1e-10 else np.zeros(3, np.float32)

    def _to_stl_bytes(self, triangles):
        buf = io.BytesIO()
        buf.write(b"\x00" * 80)                            # header
        buf.write(struct.pack("<I", len(triangles)))       # triangle count
        for tri in triangles:
            n = self._normal(tri)
            buf.write(struct.pack("<3f", *n))
            for v in tri:
                buf.write(struct.pack("<3f", float(v[0]), float(v[1]), float(v[2])))
            buf.write(struct.pack("<H", 0))                # attribute byte count
        return buf.getvalue()
