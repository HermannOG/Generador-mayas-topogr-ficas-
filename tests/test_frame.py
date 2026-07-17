import numpy as np
from shapely.geometry import Point

from src.mesh.frame import ModelFrame, fit_size_km


def test_fit_size_contains_all_points_with_margin():
    pts = [(46.0 + 0.01 * np.sin(t), 7.0 + 0.015 * np.cos(t), 0.0)
           for t in np.linspace(0, 6.28, 40)]
    size = fit_size_km(pts, 46.0, 7.0, 2.0, 0.05,
                       target_size_mm=108.0, shape="hexagon")
    frame = ModelFrame(target_size_mm=108.0, shape="hexagon")
    lat_d = (size / 2) / 111.0
    lon_d = (size / 2) / (111.0 * np.cos(np.radians(46.0)))
    frame.set_bounds((46.0 - lat_d, 46.0 + lat_d), (7.0 - lon_d, 7.0 + lon_d))
    shrunk = frame.shape_poly.buffer(-0.05 * 108.0)
    for lat, lon, _ in pts:
        assert shrunk.contains(Point(*frame.ll_to_xy(lat, lon)))


def test_ele_to_z_clamps_at_base():
    frame = ModelFrame(base_mm=15.0)
    frame.set_bounds((46.0, 46.1), (7.0, 7.1))
    frame.set_datum(500.0, 1500.0)
    assert frame.ele_to_z(500.0) == 15.0
    assert frame.ele_to_z(100.0) == 15.0       # below datum clamps to base top
    assert frame.ele_to_z(1500.0) > 15.0


def test_standardize_45_locks_total_height():
    frame = ModelFrame(base_mm=15.0, standardize_45=True)
    frame.set_bounds((46.0, 46.1), (7.0, 7.1))
    frame.set_datum(500.0, 1500.0)
    assert np.isclose(frame.ele_to_z(1500.0), 45.0)
