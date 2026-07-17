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
from scipy.ndimage import (binary_dilation, binary_erosion,
                           distance_transform_edt, gaussian_filter)
from scipy.spatial import cKDTree
import mapbox_earcut as earcut
from contourpy import contour_generator
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree

MAX_RELIEF_MM = 40.0     # safety cap on terrain relief above the base


class MeshGenerator:

    def __init__(self, config=None):
        cfg = config or {}
        self.target_size_mm  = cfg.get("target_size_mm",    108.0)
        self.base_mm         = cfg.get("base_thickness_mm",  15.0)
        # Vertical scale: 1.0 = true scale (same mm-per-metre as horizontal)
        self.height_scale    = cfg.get("height_scale",        1.0)
        # Lock total model height (base bottom → highest peak) to 45 mm
        self.standardize_45  = cfg.get("standardize_height", False)
        # Smallest printable feature (rivers narrower than this are dropped)
        self.min_feature_mm  = cfg.get("min_feature_mm",      0.4)
        # Cut zone boundaries along smooth vector curves instead of cells
        self.smooth_zones    = cfg.get("smooth_zones",       True)
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

    # Zone ids used in the per-cell zone map (thumbnail + classification)
    ZONES = ("rock", "forest", "water", "snow")
    Z_ROCK, Z_FOREST, Z_WATER, Z_SNOW = range(4)

    # Slope used by forest growth as "too steep for trees" (rise/run)
    ROCK_SLOPE = 0.45

    # Cut-wall crust: how deep the surface colour extends down the sides
    CRUST_MM = 1.2

    def generate_zone_tris(self, elevation_grid, lat_bounds, lon_bounds,
                           buildings=None, water=None, forests=None,
                           forest_level=0.0, snow_level=0.0, detect_ocean_m=None):
        """
        Generate terrain geometry split into colour zones.
        Returns dict of triangle lists:
          {"rock", "forest", "water", "snow", "buildings", "base"}
        The surface AND a thin crust band down the cut sides take the zone
        colours (geological cross-section); the wall body below the crust is
        rock, and the bottom face/border slab stay base-coloured.

        Ground classification:
          forest : mapped OSM forest polygons (real tree edges), procedurally
                   grown outward as forest_level rises 0 → 1
          rock   : all remaining bare ground
          snow   : procedural snow cover, snow_level 0 (none) → 1 (everything)
        detect_ocean_m: if not None, cells with original elevation <= this
        value (metres) are treated as sea/ocean.

        Leaves the generator set up, so route_tris() can be called afterwards.
        The per-cell zone map stays available as self.last_zone_map for the
        thumbnail renderer (values = Z_* ids).
        """
        self._setup(elevation_grid, lat_bounds, lon_bounds)
        water_mask, grid = self._build_water_features(
            elevation_grid, water, detect_ocean_m=detect_ocean_m)
        self.last_water_mask = water_mask

        rows, cols = grid.shape

        # Vertical datum: the LOWEST terrain on the model's cut edge sits
        # exactly at the base top, so e.g. a lake reaching the edge is level
        # with the border ring. Anything lower in the interior clamps to the
        # base (ele_to_z never goes below it).
        self.ele_max = float(np.nanmax(grid))
        self.ele_min = float(np.nanmin(grid))
        shape_mask = self._rasterize_xy_poly(self._shape_poly, rows, cols)
        ring = shape_mask & ~binary_erosion(shape_mask)
        if ring.any():
            self.ele_min = float(np.nanmin(grid[ring]))
            # Water reaching the edge defines the base level exactly — the
            # lake surface sits flush with the border ring (exposed banks
            # below it clamp up to the base and read as shore)
            if water_mask is not None and (ring & water_mask).any():
                self.ele_min = float(np.nanmin(grid[ring & water_mask]))
        self.sz = self._vertical_scale(self.ele_max - self.ele_min)
        self._refresh_interp(grid)
        zone_map = self._build_zone_map(grid, water_mask, forests,
                                        forest_level, snow_level)
        self.last_zone_map = zone_map

        surfaces, crust_tris, body_tris, bottom_tris = self._terrain(
            grid, zone_map if self.smooth_zones else None)

        zones = {name: [] for name in self.ZONES}

        def classify_by_cell(tri_list, into):
            """Fallback per-cell classification (smooth mode off / crust)."""
            tris = np.asarray(tri_list, dtype=np.float64)
            if not tris.size:
                return
            cx = tris[:, :, 0].mean(axis=1)
            cy = tris[:, :, 1].mean(axis=1)
            j = np.clip((cx / self.model_w * (cols - 1)).astype(int), 0, cols - 1)
            i = np.clip(((1.0 - cy / self.model_h) * (rows - 1)).astype(int), 0, rows - 1)
            tri_zone = zone_map[i, j]
            for zid, name in enumerate(self.ZONES):
                into[name].extend(tris[tri_zone == zid].tolist())

        # Surfaces: smooth mode delivers them already cut per zone along
        # smooth vector boundaries; otherwise classify per cell.
        for zid, tri_list in surfaces.items():
            if zid is None:
                classify_by_cell(tri_list, zones)
            else:
                zones[self.ZONES[zid]].extend(tri_list)

        # Crust walls take the surface colour of their cell; the wall body
        # below the crust is always rock.
        classify_by_cell(crust_tris, zones)
        zones["rock"].extend(body_tris)
        zones["buildings"] = self._buildings(buildings) if buildings else []
        zones["base"] = bottom_tris
        return zones

    def _build_zone_map(self, grid, water_mask, forests, forest_level, snow_level):
        """Per-cell zone ids: rock/forest ground, then snow, then water."""
        rows, cols = grid.shape

        cell_m = max((self.lat_max - self.lat_min) * 111_000 / max(rows - 1, 1), 1.0)
        gy, gx = np.gradient(grid, cell_m)
        slope = np.hypot(gx, gy)
        zone = np.full((rows, cols), self.Z_ROCK, dtype=np.uint8)

        # Forests: real OSM outlines as the baseline, grown procedurally
        forest_mask = np.zeros((rows, cols), dtype=bool)
        for f in forests or []:
            forest_mask |= self._mask_from_feature(f, rows, cols)
        forest_mask = self._grow_forest_mask(
            grid, forest_mask, water_mask, forest_level, slope)
        zone[forest_mask] = self.Z_FOREST

        snow_mask = self._build_snow_mask(grid, water_mask, snow_level, gx, gy, slope)
        if snow_mask is not None:
            zone[snow_mask] = self.Z_SNOW
        if water_mask is not None:
            zone[water_mask] = self.Z_WATER
        # Frozen lakes: a lake whose entire shore is snowed-in reads as snow
        # too (it stays flat — which is exactly what a frozen lake looks like)
        if snow_mask is not None:
            for mask in getattr(self, "_lake_masks", []):
                ring = binary_dilation(mask) & ~mask
                if ring.any() and snow_mask[ring].mean() > 0.9:
                    zone[mask] = self.Z_SNOW
        return zone

    def _grow_forest_mask(self, grid, base_mask, water_mask, forest_level, slope):
        """
        Grow forests procedurally from the OSM baseline. forest_level 0..1:
        0 keeps exactly the mapped forests; 1 forests everything growable.
        New trees appear in the most likely places first:
          + next to existing forest (patches spread outward)
          + lower, gentler, moister (hollows) ground
          + a deterministic noise field seeds detached patches and keeps
            edges ragged rather than evenly cut off
        Trees never grow near the peaks: in mountainous terrain a local tree
        line caps growth a little above the highest mapped forest.
        """
        level = min(1.0, max(0.0, float(forest_level)))
        if level <= 0.0:
            return base_mask

        ele_min = float(grid.min())
        ele_range = max(float(grid.max()) - ele_min, 1.0)

        # Local tree line: only meaningful in mountainous terrain
        cap = np.inf
        if ele_range > 700.0:
            cap = ele_min + 0.8 * ele_range
            if base_mask.any():
                cap = max(cap, float(np.quantile(grid[base_mask], 0.99)) + 0.05 * ele_range)

        growable = (grid <= cap) & (slope < 2 * self.ROCK_SLOPE) & ~base_mask
        if water_mask is not None:
            growable &= ~water_mask
        if not growable.any():
            return base_mask

        def z(x):
            sd = float(np.std(x))
            return (x - float(np.mean(x))) / (sd if sd > 1e-9 else 1.0)

        # Distance from existing forest: spreading beats sprouting
        if base_mask.any():
            proximity = -distance_transform_edt(~base_mask)
        else:
            proximity = np.zeros(grid.shape)

        hollows = gaussian_filter(grid, sigma=3.0) - grid
        rng = np.random.default_rng(
            abs(hash(("forest", round(self.lat_min, 4), round(self.lon_min, 4)))) % 2**32)
        noise = gaussian_filter(rng.standard_normal(grid.shape), sigma=2.5)

        score = (1.5 * np.tanh(z(proximity)) - 1.0 * z(grid) + 0.4 * np.tanh(z(hollows))
                 - 0.5 * np.clip(z(slope), 0.0, None) + 0.9 * noise)

        # level = fraction of the growable area that gets trees
        thr = float(np.quantile(score[growable], 1.0 - level))
        return base_mask | (growable & (score >= thr))

    def _build_snow_mask(self, grid, water_mask, snow_level, gx, gy, slope):
        """
        Procedural snow cover. snow_level 0..1 sets roughly the fraction of
        (non-water) terrain under snow, distributed realistically:
          + higher terrain first (snow line drags down as the slider rises)
          + shaded slopes keep snow longer (north-facing in the northern
            hemisphere, south-facing below the equator)
          + hollows and gullies hold drifts
          + patchy noise at the melt edge (deterministic per location)
          - the steepest cliff faces shed their snow
        """
        s = min(1.0, max(0.0, float(snow_level)))
        if s <= 0.0:
            return None
        not_water = ~water_mask if water_mask is not None else np.ones(grid.shape, bool)
        if s >= 1.0:
            return not_water

        def z(x):
            sd = float(np.std(x))
            return (x - float(np.mean(x))) / (sd if sd > 1e-9 else 1.0)

        ele_n = (grid - grid.min()) / max(grid.max() - grid.min(), 1.0)

        # gy is the north→south row gradient: positive on north-facing slopes
        lat_mid = (self.lat_min + self.lat_max) / 2.0
        shade = gy if lat_mid >= 0 else -gy

        # Hollows: locally below the smoothed surface
        hollows = gaussian_filter(grid, sigma=3.0) - grid

        rng = np.random.default_rng(
            abs(hash((round(self.lat_min, 4), round(self.lon_min, 4)))) % 2**32)
        noise = gaussian_filter(rng.standard_normal(grid.shape), sigma=2.0)

        score = (3.0 * z(ele_n) + 0.8 * np.tanh(z(shade)) + 0.5 * np.tanh(z(hollows))
                 + 0.5 * noise - 0.6 * np.clip(z(slope), 0.0, None))

        # Threshold at the requested coverage over non-water terrain
        thr = float(np.quantile(score[not_water], 1.0 - s))
        return (score >= thr) & not_water

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
            # A swim sits ON the water: use the water surface height under the
            # path (not the median terrain, which mixes in shore points)
            zs, zs_water = [], []
            rows, cols = self.rows, self.cols
            for p in gpx_points[::max(1, len(gpx_points) // 300)]:
                if not self._in_bounds(p[0], p[1]):
                    continue
                z = self.z_at(p[0], p[1])
                zs.append(z)
                if self.last_water_mask is not None:
                    x, y = self.ll_to_xy(p[0], p[1])
                    j = min(cols - 1, max(0, int(x / self.model_w * (cols - 1))))
                    i = min(rows - 1, max(0, int((1.0 - y / self.model_h) * (rows - 1))))
                    if self.last_water_mask[i, j]:
                        zs_water.append(z)
            if zs_water:
                flat_z = float(np.median(zs_water)) + 0.15   # float on the surface
            elif zs:
                flat_z = float(np.median(zs))
        return self._route(gpx_points, flat_z=flat_z,
                           use_gpx_ele=use_gpx_ele and flat_z is None)

    def to_stl_bytes(self, triangles):
        """Serialise a triangle list to binary STL."""
        return self._to_stl(triangles)

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

        # Point-to-point width of the shape = the full model size
        cx, cy = self.model_w / 2, self.model_h / 2
        r = min(cx, cy)
        if self.border_mm > 0:
            self._outer_poly = self._make_shape(cx, cy, r)
            self._shape_poly = self._make_shape(cx, cy, max(r - self.border_mm, r * 0.3))
        else:
            self._outer_poly = None
            self._shape_poly = self._make_shape(cx, cy, r)

    def _vertical_scale(self, ele_range_m):
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

    def _setup(self, grid, lat_bounds, lon_bounds):
        self._frame(lat_bounds, lon_bounds)

        self.ele_min = float(np.nanmin(grid))
        self.ele_max = float(np.nanmax(grid))
        self.sz = self._vertical_scale(self.ele_max - self.ele_min)

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
        # Never below the base top: interior terrain under the edge datum
        # (e.g. a valley lower than the model's cut edge) sits on the base
        return np.maximum((ele - self.ele_min) * self.sz + self.base_mm,
                          self.base_mm)

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
        """
        Rasterise a shapely (multi)polygon in model-xy space → bool mask,
        RESPECTING interior holes (e.g. a snow patch inside a forest region).
        Parts are drawn largest-first so islands inside another part's hole
        are re-filled after the hole is punched.
        """
        img = Image.new("1", (cols, rows), 0)
        d = ImageDraw.Draw(img)

        def to_px(ring):
            return [
                (x / self.model_w * (cols - 1), (1.0 - y / self.model_h) * (rows - 1))
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
        """
        Triangulate a (multi)polygon → CCW top-face tris, z from z_fn(x, y).
        Uses earcut, which handles holes (letter glyphs, islands) and never
        drops slivers — the old Delaunay+filter approach left holes in trail
        ribbons and mangled border text.
        """
        polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        tris = []
        for poly in polys:
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
            rings = [list(poly.exterior.coords[:-1])]
            rings += [list(r.coords[:-1]) for r in poly.interiors]
            verts = np.array([p for ring in rings for p in ring], dtype=np.float64)
            if len(verts) < 3:
                continue
            ring_ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
            idx = earcut.triangulate_float64(verts, ring_ends)
            for k in range(0, len(idx), 3):
                c = verts[[idx[k], idx[k + 1], idx[k + 2]]]
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
                # Rivers render at their REAL width — ones too narrow to
                # print at the chosen resolution are simply not shown.
                width_mm = w.get("width_m", 10.0) * self.sxy
                if len(line) >= 2 and width_mm >= self.min_feature_mm:
                    rivers.append((line, width_mm))
            elif len(w.get("coords", [])) >= 3:
                (seas if w["type"] == "sea" else lakes).append(w)

        # Lakes: each body flat at the level of the LAND at its polygon edge,
        # so the water always meets the shore without a wall. The DEM inside
        # a lake polygon cannot be trusted: reservoirs may show the dry basin
        # floor or a lower water stage than the mapped (full-pool) outline —
        # only the terrain just outside the outline is reliable.
        self._lake_masks = []
        for lake in lakes:
            mask = self._mask_from_feature(lake, rows, cols)
            if mask.any():
                rim = binary_dilation(mask) & ~mask
                if not rim.any():
                    rim = mask & ~binary_erosion(mask)
                level_cells = rim if rim.any() else mask
                out[mask] = float(np.median(grid[level_cells]))
                merged |= mask
                self._lake_masks.append(mask)

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
        for line, width_mm in rivers:
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
            radius_mm = width_mm / 2.0
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

    def _lattice(self, grid):
        """Vertex lattice: model xy + z for every grid node."""
        rows, cols = grid.shape
        xs = np.linspace(0.0, self.model_w, cols)
        ys = np.linspace(self.model_h, 0.0, rows)   # row 0 = north = max y
        X, Y = np.meshgrid(xs, ys)
        return np.stack([X, Y, self.ele_to_z(grid)], axis=-1)

    def _region_surface(self, V, poly, claimed=None):
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
        cell_mm = self.model_w / max(cols - 1, 1)
        strict = self._rasterize_xy_poly(poly.buffer(-1.6 * cell_mm), rows, cols)
        loose = self._rasterize_xy_poly(poly.buffer(+1.6 * cell_mm), rows, cols)

        n_strict = (strict[:-1, :-1].astype(np.int8) + strict[:-1, 1:] +
                    strict[1:, 1:] + strict[1:, :-1])
        n_loose = (loose[:-1, :-1].astype(np.int8) + loose[:-1, 1:] +
                   loose[1:, 1:] + loose[1:, :-1])
        full    = n_strict == 4
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
            v00 = V[fi, fj]; v01 = V[fi, fj + 1]
            v10 = V[fi + 1, fj]; v11 = V[fi + 1, fj + 1]
            bulk = np.concatenate([
                np.stack([v00, v11, v01], axis=1),
                np.stack([v00, v10, v11], axis=1),
            ])
            fast_tris = bulk.tolist()
        else:
            fast_tris = []

        # Spatial index over the region's parts keeps per-quad clipping cheap
        parts = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
        tree = STRtree(parts)

        slow_tris = []
        si, sj = np.nonzero(slow_full | partial)
        for i, j in zip(si.tolist(), sj.tolist()):
            if slow_full[i, j]:
                v00 = V[i,   j  ].tolist(); v01 = V[i,   j+1].tolist()
                v10 = V[i+1, j  ].tolist(); v11 = V[i+1, j+1].tolist()
                slow_tris.append((v00, v11, v01))
                slow_tris.append((v00, v10, v11))
                continue
            quad_poly = Polygon([
                (V[i,j][0],     V[i,j][1]),
                (V[i,j+1][0],   V[i,j+1][1]),
                (V[i+1,j+1][0], V[i+1,j+1][1]),
                (V[i+1,j][0],   V[i+1,j][1]),
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
                    if len(pts2d) < 3:
                        continue
                    verts = []
                    for px, py in pts2d:
                        la, lo = self._xy_to_ll(px, py)
                        verts.append([px, py, self.z_at(la, lo)])
                    for k in range(1, len(verts) - 1):
                        v0, v1, v2 = verts[0], verts[k], verts[k+1]
                        cz = (v1[0]-v0[0])*(v2[1]-v0[1]) - (v1[1]-v0[1])*(v2[0]-v0[0])
                        slow_tris.append((v0, v1, v2) if cz >= 0 else (v0, v2, v1))

        return fast_tris, slow_tris

    def _smooth_field_polys(self, mask, rows, cols):
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
        cell = self.model_w / max(cols - 1, 1)

        geom = None
        for arr in contour_generator(z=padded).lines(0.45):
            if len(arr) < 4:
                continue
            xs = (arr[:, 0] - 1) / (cols - 1) * self.model_w
            ys = (1.0 - (arr[:, 1] - 1) / (rows - 1)) * self.model_h
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

    def _zone_regions(self, zone_map):
        """
        Partition the model shape into smooth vector regions per zone,
        precedence water > snow > forest, remainder = rock.
        """
        rows, cols = zone_map.shape
        regions = []
        occupied = None
        for zid in (self.Z_WATER, self.Z_SNOW, self.Z_FOREST):
            g = self._smooth_field_polys(zone_map == zid, rows, cols)
            if g is None:
                continue
            g = g.intersection(self._shape_poly)
            if occupied is not None:
                g = g.difference(occupied)
            if g.is_empty:
                continue
            regions.append((zid, g))
            occupied = g if occupied is None else occupied.union(g)
        rock = (self._shape_poly.difference(occupied)
                if occupied is not None else self._shape_poly)
        if not rock.is_empty:
            regions.append((self.Z_ROCK, rock))
        return regions

    def _terrain(self, elevation_grid, zone_map=None):
        """
        Build the terrain surface(s) plus cut walls and bottom.
        zone_map given (smooth mode): the surface is cut along smooth vector
        zone boundaries and returned per zone id. zone_map None: one surface
        under key None, classified per cell by the caller.
        Returns (surfaces: dict, crust, body, bottom).
        """
        V = self._lattice(elevation_grid)

        if zone_map is not None:
            region_list = self._zone_regions(zone_map)
        else:
            region_list = [(None, self._shape_poly)]

        surfaces = {}
        all_slow = []
        claimed = np.zeros((V.shape[0] - 1, V.shape[1] - 1), dtype=bool)
        for zid, poly in region_list:
            fast, slow = self._region_surface(V, poly, claimed)
            surfaces[zid] = surfaces.get(zid, []) + fast + slow
            all_slow += slow

        crust, body = [], []
        bot_pts = []

        # With a border, the terrain sits ON the base slab: walls stop at the
        # slab top and the slab provides the bottom face.
        floor_z = self.base_mm if self.border_mm > 0 else 0.0

        # ── Walls: a thin CRUST band under the surface takes the surface
        # zone colour (cutting through snow shows a white line, through a
        # lake a blue line), and the BODY below reads as rock — like a
        # geological cross-section.
        for t1, t2 in self._boundary_edges(all_slow):
            c1 = [t1[0], t1[1], max(t1[2] - self.CRUST_MM, floor_z)]
            c2 = [t2[0], t2[1], max(t2[2] - self.CRUST_MM, floor_z)]
            crust.append((t2, t1, c1))   # outward-facing (right of t1→t2)
            crust.append((t2, c1, c2))
            if c1[2] > floor_z + 1e-9 or c2[2] > floor_z + 1e-9:
                b1 = [t1[0], t1[1], floor_z]
                b2 = [t2[0], t2[1], floor_z]
                body.append((c2, c1, b1))
                body.append((c2, b1, b2))
            bot_pts.append((t1[0], t1[1]))
            bot_pts.append((t2[0], t2[1]))

        # ── Bottom face: sort boundary projection by angle, fan-tri ──────
        bottom = []
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
                    bottom.append((cpt, p1, p0))  # -Z normal

        return surfaces, crust, body, bottom

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

        # Slab: top face is only the visible RING (the terrain provides the
        # surface inside — a full top face would be coplanar with edge-level
        # water and z-fight), plus bottom at 0 and outer walls. The terrain's
        # walls end exactly on the ring's inner edge, closing the solid.
        ring_poly = self._outer_poly.difference(self._shape_poly)
        base = self._top_tris_from_polys(ring_poly, lambda x, y: self.base_mm)

        coords = list(self._outer_poly.exterior.coords)[:-1]
        c = self._outer_poly.centroid
        pcx, pcy = c.x, c.y
        n = len(coords)
        for k in range(n):
            ax, ay = coords[k]
            bx, by = coords[(k + 1) % n]
            base.append(([pcx, pcy, 0.0], [bx, by, 0.0], [ax, ay, 0.0]))
            base.append(([ax, ay, 0.0], [bx, by, self.base_mm], [bx, by, 0.0]))
            base.append(([ax, ay, 0.0], [ax, ay, self.base_mm], [bx, by, self.base_mm]))

        if self.shape != "hexagon":
            return {"base": base, "text": []}

        text = []
        cx, cy = self.model_w / 2, self.model_h / 2
        R = min(cx, cy)                   # outer hexagon circumradius = side length
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
            # Height from the GPX file's own elevation (nearest track point).
            # GPS altitude is noisy — smooth it with a moving average first,
            # otherwise the ribbon jumps step to step.
            in_pts = [(self.ll_to_xy(p[0], p[1]), p[2]) for p in points
                      if self._in_bounds(p[0], p[1])]
            kd = cKDTree([xy for xy, _ in in_pts])
            eles = np.array([e for _, e in in_pts], dtype=float)
            win = max(3, min(31, len(eles) // 50) | 1)
            kernel = np.ones(win) / win
            eles = np.convolve(np.pad(eles, win // 2, mode="edge"), kernel, "valid")

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
        path = LineString(pts2d).simplify(hw * 0.5, preserve_topology=True)

        # Buffer the 2D path → smooth ribbon (round joins, flat end caps),
        # clipped to the model shape so the trail never spills over the edge
        ribbon = path.buffer(hw, cap_style=2, join_style=1, resolution=8)

        # Out-and-back passes that don't retrace exactly leave a lumpy double
        # line; morphological closing merges any touching/nearby passes into
        # one clean combined ribbon, then the outline is relaxed.
        m = hw * 1.5
        ribbon = ribbon.buffer(m).buffer(-m).simplify(hw * 0.2)

        ribbon = ribbon.intersection(self._shape_poly)
        if ribbon.is_empty:
            return []

        # Densify the ribbon outline so the top surface actually FOLLOWS the
        # terrain: with sparse vertices the long triangles bridge over hills
        # and dip underground in valleys.
        cell_mm = self.model_w / max(self.cols - 1, 1)
        ribbon = ribbon.segmentize(max(cell_mm * 0.5, hw * 0.4))

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
