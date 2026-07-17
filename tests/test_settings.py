import pytest

from src.settings import DEFAULT_SETTINGS, SettingsError, normalize_settings


def test_empty_gives_defaults():
    cfg = normalize_settings({})
    assert cfg == dict(DEFAULT_SETTINGS) | cfg   # same keys
    assert set(cfg) == set(DEFAULT_SETTINGS)
    assert cfg["shape"] == "hexagon"
    assert cfg["baseThickness"] == 15.0


def test_legacy_keys_accepted():
    cfg = normalize_settings({"water_color": "#123456", "include_lakes": False,
                              "trail_width": 2.5})
    assert cfg["waterColor"] == "#123456"
    assert cfg["includeLakes"] is False
    assert cfg["trailWidth"] == 2.5


def test_camelcase_wins_over_legacy():
    cfg = normalize_settings({"waterColor": "#AAAAAA", "water_color": "#BBBBBB"})
    assert cfg["waterColor"] == "#AAAAAA"


def test_numeric_clamps():
    cfg = normalize_settings({"snowLevel": 5, "trailWidth": 0,
                              "printResolution": 99, "baseThickness": -3})
    assert cfg["snowLevel"] == 1.0
    assert cfg["trailWidth"] == 0.1
    assert cfg["printResolution"] == 0.8
    assert cfg["baseThickness"] == 1.0


def test_border_labels_coerced_and_capped():
    cfg = normalize_settings({"borderLabels": [1, "two", None, "", "", "", "extra", "more"]})
    assert cfg["borderLabels"] == ["1", "two", "None", "", "", ""]


def test_bad_inputs_raise():
    with pytest.raises(SettingsError):
        normalize_settings("not a dict")
    with pytest.raises(SettingsError):
        normalize_settings({"waterColor": 42})
    with pytest.raises(SettingsError):
        normalize_settings({"borderLabels": "nope"})
    with pytest.raises(SettingsError):
        normalize_settings({"snowLevel": "high"})
