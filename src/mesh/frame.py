"""
ModelFrame: the geometric reference frame of one generated model.

Owns the lat/lon -> model-mm mapping, the model shape polygons
(square/hexagon/octagon/circle plus an optional border ring), the vertical
scale/datum, and the elevation interpolator. Constructed once per job and
passed explicitly to the mesh-building stages.

Model-space coordinate system (millimetres):
  X -> East   (longitude direction)
  Y -> North  (latitude direction)
  Z -> Up     (elevation)
"""

import math

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Point, Polygon

MAX_RELIEF_MM = 40.0     # safety cap on terrain relief above the base


class ModelFrame:
    def __init__(self, target_size_mm=108.0, base_mm=15.0, height_scale=1.0,
                 standardize_45=False, shape="square", border_mm=0.0):
        self.target_size_mm = float(target_size_mm)
        self.base_mm = float(base_mm)
        # Vertical scale: 1.0 = true scale (same mm-per-metre as horizontal)
        self.height_scale = float(height_scale)
        # Lock total model height (base bottom -> highest peak) to 45 mm
        self.standardize_45 = bool(standardize_45)
        self.shape = shape
        # Extended border ring (hexagon only): terrain shrinks by border_mm
        # and a base slab with raised text labels surrounds it.
        self.border_mm = float(border_mm)

        self._interp = None
        self.rows = self.cols = 0
        self.ele_min = 0.0
        self.ele_max = 0.0
        self.sz = 1.0

    # -- Horizontal mapping --------------------------------------------------

    def set_bounds(self, lat_bounds, lon_bounds):
        """Set the lat/lon -> model-mm mapping and the model shape polygons."""
        lat_min, lat_max = lat_bounds
        lon_min, lon_max = lon_bounds
        lat_mid = (lat_min + lat_max) / 2.0

        width_m = (lon_max - lon_min) * 111_000 * math.cos(math.radians(lat_mid))
        height_m = (lat_max - lat_min) * 111_000
        max_dim = max(width_m, height_m)

        self.sxy = self.target_size_mm / max_dim
        self.model_w = width_m * self.sxy
        self.model_h = height_m * self.sxy

        self.lat_min, self.lat_max = lat_min, lat_max
        self.lon_min, self.lon_max = lon_min, lon_max

        # Point-to-point width of the shape = the full model size
        cx, cy = self.model_w / 2, self.model_h / 2
        r = min(cx, cy)
        if self.border_mm > 0:
            self.outer_poly = self._make_shape(cx, cy, r)
            self.shape_poly = self._make_shape(cx, cy, max(r - self.border_mm, r * 0.3))
        else:
            self.outer_poly = None
            self.shape_poly = self._make_shape(cx, cy, r)

    def _make_shape(self, cx, cy, r):
        s = self.shape
        if s == "hexagon":
            # With a labelled border the hexagon is flat-bottom (vertices at
            # 0deg,60deg,...) so there are true "bottom"/"top" sides to write on.
            off = 0.0 if self.border_mm > 0 else math.pi / 6
            angles = [i * math.pi / 3 + off for i in range(6)]
            pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]
            return Polygon(pts)
        elif s == "octagon":
            angles = [i * math.pi / 4 + math.pi / 8 for i in range(8)]
            pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]
            return Polygon(pts)
        elif s in ("circle", "circle-flat"):
            return Point(cx, cy).buffer(r, quad_segs=64)
        else:   # square / default
            return Polygon([
                (0, 0), (self.model_w, 0),
                (self.model_w, self.model_h), (0, self.model_h),
            ])

    # -- Vertical datum --------------------------------------------------------

    def vertical_scale(self, ele_range_m):
        """
        mm of model height per metre of elevation. height_scale 1.0 means
        TRUE scale (identical to the horizontal scale); the relief is capped
        at MAX_RELIEF_MM so extreme scales stay printable.
        """
        rng = max(float(ele_range_m), 1.0)
        if self.standardize_45:
            return max(45.0 - self.base_mm, 1.0) / rng
        sz = self.sxy * max(0.0, float(self.height_scale))
        return min(sz, MAX_RELIEF_MM / rng)

    def set_datum(self, ele_min, ele_max):
        self.ele_min = float(ele_min)
        self.ele_max = float(ele_max)
        self.sz = self.vertical_scale(self.ele_max - self.ele_min)

    def set_grid(self, grid):
        """Attach the (possibly water-carved) elevation grid + interpolator."""
        self.rows, self.cols = grid.shape
        lat_arr = np.linspace(self.lat_max, self.lat_min, grid.shape[0])
        lon_arr = np.linspace(self.lon_min, self.lon_max, grid.shape[1])
        self._interp = RegularGridInterpolator(
            (lat_arr, lon_arr), grid,
            method="linear", bounds_error=False, fill_value=self.ele_min,
        )

    # -- Coordinate helpers (scalar or ndarray inputs) -------------------------

    def ll_to_xy(self, lat, lon):
        x = (lon - self.lon_min) / (self.lon_max - self.lon_min) * self.model_w
        y = (lat - self.lat_min) / (self.lat_max - self.lat_min) * self.model_h
        return x, y

    def xy_to_ll(self, x, y):
        lon = self.lon_min + x / self.model_w * (self.lon_max - self.lon_min)
        lat = self.lat_min + y / self.model_h * (self.lat_max - self.lat_min)
        return lat, lon

    def ele_to_z(self, ele):
        # Never below the base top: interior terrain under the edge datum
        # (e.g. a valley lower than the model's cut edge) sits on the base
        return np.maximum((ele - self.ele_min) * self.sz + self.base_mm,
                          self.base_mm)

    def z_at(self, lat, lon):
        return float(self.ele_to_z(float(self._interp([[lat, lon]])[0])))

    def z_at_many(self, lats, lons):
        """Vectorised z lookup: arrays of lat/lon -> array of model z (mm)."""
        pts = np.column_stack([np.asarray(lats, float).ravel(),
                               np.asarray(lons, float).ravel()])
        return self.ele_to_z(self._interp(pts))

    def z_at_xy(self, xs, ys):
        """Vectorised z lookup in model-xy space."""
        lat, lon = self.xy_to_ll(np.asarray(xs, float), np.asarray(ys, float))
        return self.z_at_many(lat, lon)

    def in_bounds(self, lat, lon):
        return (self.lat_min <= lat <= self.lat_max and
                self.lon_min <= lon <= self.lon_max)


def fit_size_km(points, lat_c, lon_c, span_km, margin_frac, *,
                target_size_mm=108.0, shape="square", border_mm=0.0):
    """
    Smallest map area (km) centred on (lat_c, lon_c) whose model SHAPE
    (hexagon/octagon/circle/square, minus any border ring) contains every
    route point with margin_frac x model-size clearance from the edge.
    A square bounding box isn't enough — shape corners cut into it.
    """
    frame = ModelFrame(target_size_mm=target_size_mm, shape=shape, border_mm=border_mm)
    pts = [(p[0], p[1]) for p in points[::max(1, len(points) // 500)]]
    size = max(span_km, 0.2)
    for _ in range(60):
        if _points_fit(frame, pts, lat_c, lon_c, size, margin_frac):
            break
        size *= 1.05
    return size


def _points_fit(frame, pts, lat_c, lon_c, size_km, margin_frac):
    lat_d = (size_km / 2.0) / 111.0
    lon_d = (size_km / 2.0) / (111.0 * math.cos(math.radians(lat_c)))
    frame.set_bounds((lat_c - lat_d, lat_c + lat_d), (lon_c - lon_d, lon_c + lon_d))

    shrunk = frame.shape_poly.buffer(-margin_frac * frame.target_size_mm)
    if shrunk.is_empty:
        return False
    for lat, lon in pts:
        if not frame.in_bounds(lat, lon):
            return False
        if not shrunk.contains(Point(*frame.ll_to_xy(lat, lon))):
            return False
    return True
