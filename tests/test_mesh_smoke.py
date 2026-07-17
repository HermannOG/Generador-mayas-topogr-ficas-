"""End-to-end mesh smoke test on synthetic terrain — no network."""

import numpy as np
import pytest

from src.mesh.stl import parse_stl_bytes
from src.mesh_generator import MeshGenerator
from tests.conftest import LAT_B, LON_B, forest_feature, lake_feature, route_points, synthetic_grid

CONFIGS = [
    ("square", {}),
    ("circle", {}),
    ("hexagon", {"border_mm": 8.0,
                 "border_labels": ["MONT BLANC", "", "2026", "", "", ""]}),
]


@pytest.mark.parametrize("shape,extra", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_full_generation(shape, extra):
    gen = MeshGenerator({
        "target_size_mm": 108.0, "base_thickness_mm": 15.0, "height_scale": 1.5,
        "min_feature_mm": 0.4, "smooth_zones": True, "shape": shape,
        "route_width_mm": 1.0, "route_height_mm": 1.0, **extra,
    })
    grid = synthetic_grid(50)
    zones = gen.generate_zone_tris(
        grid, LAT_B, LON_B, water=[lake_feature()], forests=[forest_feature()],
        forest_level=0.3, snow_level=0.2)
    route = gen.route_tris(route_points())
    border = gen.border_tris()

    total = 0
    for name, tris in {**zones, "route": route, **border}.items():
        tris = np.asarray(tris, dtype=np.float64)
        total += tris.size
        if tris.size:
            assert np.isfinite(tris).all(), f"{name} has non-finite vertices"
            zmax = 15.0 + 40.0 + 2.0   # base + MAX_RELIEF + route/text headroom
            assert tris[..., 2].min() >= -1e-6, f"{name} dips below z=0"
            assert tris[..., 2].max() <= zmax, f"{name} exceeds max height"
    assert total > 0
    assert len(np.asarray(zones["rock"])), "rock zone empty"
    assert len(np.asarray(route)), "route empty"

    if extra.get("border_mm"):
        assert len(np.asarray(border["base"])), "border slab empty"
        assert len(np.asarray(border["text"])), "border text empty"

    # STL round-trip of the biggest zone
    data = gen.to_stl_bytes(zones["rock"])
    _, verts = parse_stl_bytes(data)
    assert len(verts) == len(np.asarray(zones["rock"]))


def test_terrain_only_prepare_is_cheap_and_sufficient():
    """The fast path: prepare() must yield the zone map without triangulating."""
    gen = MeshGenerator({"shape": "hexagon"})
    zm = gen.prepare(synthetic_grid(50), LAT_B, LON_B,
                     water=[lake_feature()], forests=[forest_feature()])
    assert zm is not None and zm.shape == (50, 50)
    assert gen.last_zone_map is zm
    # and route_tris still works after prepare (no generate_zone_tris needed)
    assert len(np.asarray(gen.route_tris(route_points())))
