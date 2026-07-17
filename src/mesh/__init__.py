"""Mesh-building package: pure stages operating on an explicit ModelFrame.

Modules:
  frame    — ModelFrame (lat/lon<->mm mapping, shapes, datum) + fit_size_km
  raster   — PIL rasterisation helpers
  water    — water carving (lakes/seas/rivers -> mask + adjusted grid)
  zones    — zone classification (forest growth, snow, frozen lakes)
  surface  — terrain lattice, exact region clipping, walls, bottom
  solids   — earcut top faces, boundary edges, prism solids
  route    — trail ribbon
  border   — border ring slab + text labels
  buildings— extruded building footprints
  stl      — binary STL serialisation

The thin MeshGenerator facade in src.mesh_generator wires these together.
"""

from src.mesh.frame import ModelFrame, fit_size_km
from src.mesh.solids import EMPTY_TRIS, as_tris, concat_tris
from src.mesh.stl import to_stl_bytes

__all__ = ["ModelFrame", "fit_size_km", "EMPTY_TRIS", "as_tris", "concat_tris",
           "to_stl_bytes"]
