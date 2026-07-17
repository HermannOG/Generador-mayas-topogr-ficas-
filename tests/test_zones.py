
from src.mesh_generator import MeshGenerator
from tests.conftest import LAT_B, LON_B, forest_feature, lake_feature, synthetic_grid


def _zone_map(gen, **kw):
    return gen.prepare(synthetic_grid(), LAT_B, LON_B, **kw)


def test_water_cells_are_water_zone(gen):
    zm = _zone_map(gen, water=[lake_feature()])
    assert (zm[gen.last_water_mask] == MeshGenerator.Z_WATER).all()


def test_snow_full_covers_everything_but_water(gen):
    zm = _zone_map(gen, water=[lake_feature()], snow_level=1.0)
    not_water = zm != MeshGenerator.Z_WATER
    assert (zm[not_water] == MeshGenerator.Z_SNOW).all()


def test_forest_zero_keeps_mapped_outline_only(gen):
    zm = _zone_map(gen, forests=[forest_feature()], forest_level=0.0)
    frac = (zm == MeshGenerator.Z_FOREST).mean()
    assert 0.0 < frac < 0.2   # roughly the mapped rectangle, nothing more


def test_coverage_tracks_levels(gen):
    fracs = []
    for level in (0.2, 0.5, 0.9):
        g = MeshGenerator({"shape": "square"})
        zm = g.prepare(synthetic_grid(), LAT_B, LON_B,
                       forests=[forest_feature()], forest_level=level)
        fracs.append((zm == MeshGenerator.Z_FOREST).mean())
    assert fracs[0] < fracs[1] < fracs[2]


def test_deterministic(gen):
    a = _zone_map(gen, forests=[forest_feature()], forest_level=0.4, snow_level=0.3)
    g2 = MeshGenerator({"shape": "square"})
    b = g2.prepare(synthetic_grid(), LAT_B, LON_B,
                   forests=[forest_feature()], forest_level=0.4, snow_level=0.3)
    assert (a == b).all()
