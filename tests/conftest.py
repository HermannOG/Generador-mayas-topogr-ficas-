"""Shared fixtures: synthetic terrain scenes (no network anywhere in tests)."""

import numpy as np
import pytest

from src.mesh_generator import MeshGenerator

LAT_B = (46.0, 46.05)
LON_B = (7.0, 7.07)


def synthetic_grid(n=60, seed=7):
    """Smooth hilly terrain: one big peak + rolling ridges, ~500-1050 m."""
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    return (400 * np.exp(-((x - 0.4) ** 2 + (y - 0.55) ** 2) * 8)
            + 150 * np.sin(x * 9) * np.cos(y * 7) + 500)


def lake_feature():
    la0, la1, lo0, lo1 = 46.005, 46.015, 7.05, 7.06
    return {"type": "lake",
            "coords": [(la0, lo0), (la0, lo1), (la1, lo1), (la1, lo0)],
            "holes": []}


def river_feature():
    return {"type": "river", "width_m": 40.0,
            "line": [(46.045, 7.005 + 0.06 * t) for t in np.linspace(0, 1, 50)]}


def forest_feature():
    return {"coords": [(46.02, 7.01), (46.02, 7.03), (46.035, 7.03), (46.035, 7.01)],
            "holes": []}


def route_points(n=120):
    t = np.linspace(0, 1, n)
    return [(46.01 + 0.03 * ti, 7.015 + 0.04 * ti, 600.0) for ti in t]


@pytest.fixture
def grid():
    return synthetic_grid()


@pytest.fixture
def gen():
    return MeshGenerator({
        "target_size_mm": 108.0, "base_thickness_mm": 15.0, "height_scale": 1.5,
        "min_feature_mm": 0.4, "smooth_zones": True, "shape": "square",
        "route_width_mm": 1.0, "route_height_mm": 1.0,
    })
