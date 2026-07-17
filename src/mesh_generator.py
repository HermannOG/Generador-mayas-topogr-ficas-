"""
3D mesh generator for topographic maps — thin facade over src/mesh/.

The heavy lifting lives in the src/mesh package as pure functions of an
explicit ModelFrame; this class wires the stages together and keeps the
public surface used by the pipeline:

    gen = MeshGenerator(cfg)
    gen.fit_size_km(...)                # area fitting (no state needed)
    gen.prepare(grid, ...)              # setup + water carving + zone map
                                        # (terrain-only stops here)
    gen.generate_zone_tris(grid, ...)   # prepare + full triangulation
    gen.route_tris(points, ...)         # trail ribbons (after prepare)
    gen.border_tris()                   # border slab + text labels
    gen.to_stl_bytes(tris)              # binary STL

Triangles are (N, 3, 3) float64 ndarrays throughout.

Model-space coordinate system (millimetres):
  X -> East   (longitude direction)
  Y -> North  (latitude direction)
  Z -> Up     (elevation)
"""

import numpy as np
from scipy.ndimage import binary_erosion

from src.mesh import border, route, stl, surface, zones
from src.mesh import buildings as mesh_buildings
from src.mesh.frame import MAX_RELIEF_MM, ModelFrame, fit_size_km
from src.mesh.raster import rasterize_xy_poly
from src.mesh.solids import EMPTY_TRIS, as_tris, concat_tris
from src.mesh.water import carve_water

__all__ = ["MeshGenerator", "MAX_RELIEF_MM"]


class MeshGenerator:

    # Zone ids used in the per-cell zone map (thumbnail + classification)
    ZONES = zones.ZONES
    Z_ROCK, Z_FOREST, Z_WATER, Z_SNOW = (
        zones.Z_ROCK, zones.Z_FOREST, zones.Z_WATER, zones.Z_SNOW)

    # Slope used by forest growth as "too steep for trees" (rise/run)
    ROCK_SLOPE = zones.ROCK_SLOPE

    # Cut-wall crust: how deep the surface colour extends down the sides
    CRUST_MM = surface.CRUST_MM

    def __init__(self, config=None):
        cfg = config or {}
        self.target_size_mm = cfg.get("target_size_mm", 108.0)
        self.base_mm = cfg.get("base_thickness_mm", 15.0)
        self.height_scale = cfg.get("height_scale", 1.0)
        self.standardize_45 = cfg.get("standardize_height", False)
        # Smallest printable feature (rivers narrower than this are dropped)
        self.min_feature_mm = cfg.get("min_feature_mm", 0.4)
        # Cut zone boundaries along smooth vector curves instead of cells
        self.smooth_zones = cfg.get("smooth_zones", True)
        self.building_h_mm = cfg.get("building_height_mm", 2.0)
        self.building_scale = cfg.get("building_scale", 1.0)
        self.route_width_mm = cfg.get("route_width_mm", 1.0)
        self.route_height_mm = cfg.get("route_height_mm", 1.0)
        self.shape = cfg.get("shape", "square")
        self.border_mm = cfg.get("border_mm", 0.0)
        self.border_labels = cfg.get("border_labels", [])
        self.text_height_mm = cfg.get("text_height_mm", 0.8)

        self.frame = None
        self.last_zone_map = None
        self.last_water_mask = None
        self._lake_masks = []
        self._carved_grid = None

    # -- Split stages -------------------------------------------------------

    def _new_frame(self):
        return ModelFrame(
            target_size_mm=self.target_size_mm, base_mm=self.base_mm,
            height_scale=self.height_scale, standardize_45=self.standardize_45,
            shape=self.shape, border_mm=self.border_mm,
        )

    def prepare(self, elevation_grid, lat_bounds, lon_bounds,
                water=None, forests=None, forest_level=0.0, snow_level=0.0,
                detect_ocean_m=None):
        """
        Everything up to (but not including) triangulation: frame setup,
        water carving, vertical datum, zone classification. Returns the
        per-cell zone map (also stored as self.last_zone_map). The fast
        terrain-only path stops here — no triangles are built.
        """
        frame = self._new_frame()
        frame.set_bounds(lat_bounds, lon_bounds)
        self.frame = frame

        water_mask, grid, lake_masks = carve_water(
            frame, elevation_grid, water, self.min_feature_mm,
            detect_ocean_m=detect_ocean_m)
        self.last_water_mask = water_mask
        self._lake_masks = lake_masks
        self._carved_grid = grid

        rows, cols = grid.shape

        # Vertical datum: the LOWEST terrain on the model's cut edge sits
        # exactly at the base top, so e.g. a lake reaching the edge is level
        # with the border ring. Anything lower in the interior clamps to the
        # base (ele_to_z never goes below it).
        ele_max = float(np.nanmax(grid))
        ele_min = float(np.nanmin(grid))
        shape_mask = rasterize_xy_poly(frame, frame.shape_poly, rows, cols)
        ring = shape_mask & ~binary_erosion(shape_mask)
        if ring.any():
            ele_min = float(np.nanmin(grid[ring]))
            # Water reaching the edge defines the base level exactly — the
            # lake surface sits flush with the border ring (exposed banks
            # below it clamp up to the base and read as shore)
            if water_mask is not None and (ring & water_mask).any():
                ele_min = float(np.nanmin(grid[ring & water_mask]))
        frame.set_datum(ele_min, ele_max)
        frame.set_grid(grid)

        zone_map = zones.build_zone_map(
            frame, grid, water_mask, forests, forest_level, snow_level,
            lake_masks=lake_masks)
        self.last_zone_map = zone_map
        return zone_map

    # -- Public API -----------------------------------------------------------

    def generate_zone_tris(self, elevation_grid, lat_bounds, lon_bounds,
                           buildings=None, water=None, forests=None,
                           forest_level=0.0, snow_level=0.0, detect_ocean_m=None):
        """
        Generate terrain geometry split into colour zones.
        Returns dict of (N,3,3) triangle arrays:
          {"rock", "forest", "water", "snow", "buildings", "base"}
        The surface AND a thin crust band down the cut sides take the zone
        colours (geological cross-section); the wall body below the crust is
        rock, and the bottom face/border slab stay base-coloured.

        Leaves the generator set up, so route_tris() can be called afterwards.
        The per-cell zone map stays available as self.last_zone_map for the
        thumbnail renderer (values = Z_* ids).
        """
        zone_map = self.prepare(
            elevation_grid, lat_bounds, lon_bounds, water=water,
            forests=forests, forest_level=forest_level, snow_level=snow_level,
            detect_ocean_m=detect_ocean_m)
        frame = self.frame
        grid = self._carved_grid
        rows, cols = grid.shape

        surfaces, crust_tris, body_tris, bottom_tris = surface.build_terrain(
            frame, grid, zone_map if self.smooth_zones else None)

        result = {name: EMPTY_TRIS for name in self.ZONES}

        def classify_by_cell(tris, into):
            """Fallback per-cell classification (smooth mode off / crust)."""
            tris = as_tris(tris)
            if not len(tris):
                return
            cx = tris[:, :, 0].mean(axis=1)
            cy = tris[:, :, 1].mean(axis=1)
            j = np.clip((cx / frame.model_w * (cols - 1)).astype(int), 0, cols - 1)
            i = np.clip(((1.0 - cy / frame.model_h) * (rows - 1)).astype(int),
                        0, rows - 1)
            tri_zone = zone_map[i, j]
            for zid, name in enumerate(self.ZONES):
                into[name] = concat_tris([into[name], tris[tri_zone == zid]])

        # Surfaces: smooth mode delivers them already cut per zone along
        # smooth vector boundaries; otherwise classify per cell.
        for zid, tris in surfaces.items():
            if zid is None:
                classify_by_cell(tris, result)
            else:
                result[self.ZONES[zid]] = concat_tris([result[self.ZONES[zid]], tris])

        # Crust walls take the surface colour of their cell; the wall body
        # below the crust is always rock.
        classify_by_cell(crust_tris, result)
        result["rock"] = concat_tris([result["rock"], body_tris])
        result["buildings"] = (
            mesh_buildings.build_buildings(
                frame, buildings, self.building_h_mm, self.building_scale)
            if buildings else EMPTY_TRIS)
        result["base"] = as_tris(bottom_tris)
        return result

    def route_tris(self, gpx_points, flat=False, use_gpx_ele=False):
        """
        Trail ribbon triangles. Requires a prior generate_zone_tris()/prepare().
        flat=True: ribbon at one constant height (the water surface under the
        path, or the median terrain z) — used for swims.
        use_gpx_ele=True: ribbon height from the GPX file's own elevation
        data instead of the map terrain.
        """
        if not gpx_points or len(gpx_points) < 2:
            return EMPTY_TRIS
        flat_z = None
        if flat:
            flat_z = route.route_flat_z(self.frame, gpx_points, self.last_water_mask)
        return route.build_route(
            self.frame, gpx_points, self.route_width_mm, self.route_height_mm,
            flat_z=flat_z, use_gpx_ele=use_gpx_ele and flat_z is None)

    def border_tris(self):
        """
        Base slab plus raised text labels on the border ring.
        Returns {"base": (N,3,3), "text": (N,3,3)}.
        Requires a prior generate_zone_tris()/prepare().
        """
        return border.build_border(self.frame, self.border_labels,
                                   self.text_height_mm)

    @staticmethod
    def to_stl_bytes(triangles):
        """Serialise triangles ((N,3,3) ndarray or lists) to binary STL."""
        return stl.to_stl_bytes(triangles)

    def fit_size_km(self, points, lat_c, lon_c, span_km, margin_frac):
        """
        Smallest map area (km) centred on (lat_c, lon_c) whose model SHAPE
        (minus any border ring) contains every route point with
        margin_frac x model-size clearance from the edge.
        """
        return fit_size_km(
            points, lat_c, lon_c, span_km, margin_frac,
            target_size_mm=self.target_size_mm, shape=self.shape,
            border_mm=self.border_mm)
