import numpy as np
from shapely.geometry import Polygon

from src.mesh.buildings import build_buildings, parse_building_height_m
from src.mesh.frame import ModelFrame
from tests.conftest import LAT_B, LON_B


def test_height_parsing():
    assert parse_building_height_m({"height_tag": "12"}) == 12.0
    assert parse_building_height_m({"height_tag": "12 m"}) == 12.0
    assert parse_building_height_m({"height_m": 8.5}) == 8.5
    assert parse_building_height_m({"levels": "4"}) == 12.0
    assert parse_building_height_m({"height_tag": "tall"}) is None
    assert parse_building_height_m({"height_tag": None}) is None
    assert parse_building_height_m({"levels": "9999"}) is None   # junk guard
    assert parse_building_height_m({}) is None


def _flat_frame():
    frame = ModelFrame(shape="square")
    frame.set_bounds(LAT_B, LON_B)
    frame.set_datum(500.0, 600.0)
    frame.set_grid(np.full((40, 40), 550.0))
    return frame


def test_concave_roof_area_is_exact():
    # L-shaped (concave) footprint: a centroid fan would miscover it
    ring = [(46.02, 7.02), (46.02, 7.03), (46.025, 7.03), (46.025, 7.026),
            (46.022, 7.026), (46.022, 7.02)]
    frame = _flat_frame()
    tris = build_buildings(frame, [{"coords": ring}], 2.0)
    assert len(tris)
    top_z = tris[:, :, 2].max()
    roof = tris[np.all(np.isclose(tris[:, :, 2], top_z), axis=1)]
    roof_area = sum(Polygon(t[:, :2]).area for t in roof)
    xs, ys = frame.ll_to_xy(np.array([p[0] for p in ring]),
                            np.array([p[1] for p in ring]))
    want = Polygon(zip(xs, ys, strict=True)).area
    assert np.isclose(roof_area, want, rtol=1e-6)


def test_out_of_bounds_building_skipped():
    frame = _flat_frame()
    tris = build_buildings(frame, [{"coords": [(10.0, 10.0), (10.0, 10.001),
                                               (10.001, 10.001)]}], 2.0)
    assert len(tris) == 0
