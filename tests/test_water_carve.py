import numpy as np

from tests.conftest import LAT_B, LON_B, lake_feature, river_feature, synthetic_grid


def test_lake_flattened_to_rim_level(gen):
    grid = synthetic_grid()
    gen.prepare(grid, LAT_B, LON_B, water=[lake_feature()])
    mask = gen.last_water_mask
    carved = gen._carved_grid
    assert mask is not None and mask.any()
    lake_vals = carved[mask]
    assert np.allclose(lake_vals, lake_vals[0])          # perfectly flat
    assert grid[mask].std() > 1.0                        # was NOT flat before


def test_river_monotonic_downstream(gen):
    grid = synthetic_grid()
    river = river_feature()
    gen.prepare(grid, LAT_B, LON_B, water=[river])
    carved = gen._carved_grid
    rows, cols = carved.shape
    # sample carved elevation along the river line, in way order
    eles = []
    for lat, lon in river["line"]:
        i = round((LAT_B[1] - lat) / (LAT_B[1] - LAT_B[0]) * (rows - 1))
        j = round((lon - LON_B[0]) / (LON_B[1] - LON_B[0]) * (cols - 1))
        eles.append(carved[i, j])
    eles = np.array(eles)
    diffs = np.diff(eles)
    # allow way-order flip: monotone non-increasing in one direction
    assert (diffs <= 1e-9).all() or (diffs >= -1e-9).all()


def test_ocean_detection_flattens_to_min(gen):
    grid = synthetic_grid()
    grid[-10:, -10:] = -2.0   # a below-threshold corner
    gen.prepare(grid, LAT_B, LON_B, water=[], detect_ocean_m=0.5)
    carved = gen._carved_grid
    mask = gen.last_water_mask
    assert mask is not None
    assert mask[-5, -5]
    assert np.allclose(carved[mask], float(np.nanmin(grid)))


def test_no_water_returns_none_mask(gen, grid):
    gen.prepare(grid, LAT_B, LON_B, water=[])
    assert gen.last_water_mask is None
