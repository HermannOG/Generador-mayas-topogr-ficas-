"""
Typed generation settings.

The public schema follows the topotrail.com front-end (camelCase) with
TopoTrail extensions; legacy snake_case names from the old front-end are
still accepted as fallbacks. Values are validated/clamped up front so bad
input fails the request with a 400 instead of a mid-generation 500.
"""

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

# Settings schema (camelCase, topotrail.com-compatible). Colours are derived
# from the Strava-orange trail (#FC5200, HSL h20/s1.0/l0.49): hues rotated,
# lightness/saturation kept in the same family so the palette reads as one
# system.
DEFAULT_SETTINGS = {
    "waterColor":            "#306BA6",   # h210, complementary blue
    "landColor":             "#327B4B",   # forest, h140
    "trackColor":            "#FC5200",
    "rockColor":             "#9A877E",   # same h20 hue, desaturated
    "snowColor":             "#F3EFED",   # near-white, warm cast
    "snowLevel":             0,           # 0 none -> 1 everything under snow
    "forestLevel":           0,           # 0 mapped forests only -> 1 fully grown
    "heightScale":           1,
    "standardizeHeight":     False,  # lock total model height to 45 mm
    "trailWidth":            1,
    "trailHeight":           1,
    "useHeightFromGpx":      False,
    "shape":                 "hexagon",
    "distanceTrackToBorder": 0,
    # Base 15 mm (0.591") and relief 15 mm -> 30 mm (1.182") total model height
    "baseThickness":         15,
    "includeSeas":           True,
    "includeLakes":          True,
    "includeRivers":         False,
    "base_size":             108,   # hexagon width, point to point (mm)
    "center":                "normal",
    "buildings":             False,
    "building_scale":        1,
    "buildingsColor":        "#777777",
    "printResolution":       0.2,   # mm per mesh cell: 0.1 / 0.2 / 0.4 / 0.8
    "smooth":                True,  # cut zone boundaries along smooth curves
    "singleColor":           False,
    "singleColor_gap":       0.5,
    # TopoTrail extensions (not on topotrail.com): hexagon border with text
    "baseColor":             "#FFFFFF",
    "textColor":             "#000000",
    "borderLabels":          ["", "", "", "", "", ""],
    # order: top, upper-right, lower-right, bottom, lower-left, upper-left
}

# Legacy snake_case names (old front-end) still accepted as fallbacks
LEGACY_KEYS = {
    "waterColor":            "water_color",
    "landColor":             "land_color",
    "trackColor":            "trail_color",
    "rockColor":             "rock_color",
    "heightScale":           "height_scale",
    "trailWidth":            "trail_width",
    "trailHeight":           "trail_height",
    "useHeightFromGpx":      "use_gpx_elevation",
    "distanceTrackToBorder": "trail_border",
    "baseThickness":         "base_thickness",
    "includeSeas":           "include_seas",
    "includeLakes":          "include_lakes",
    "includeRivers":         "include_rivers",
    "center":                "center_on",
    "buildings":             "include_buildings",
    "singleColor":           "print_separately",
}

_COLOR_KEYS = ("waterColor", "landColor", "trackColor", "rockColor",
               "snowColor", "buildingsColor", "baseColor", "textColor")

# Numeric clamp ranges (min, max) — out-of-range values are clamped, not
# rejected, matching how the sliders behave in the UI.
_CLAMPS = {
    "snowLevel":             (0.0, 1.0),
    "forestLevel":           (0.0, 1.0),
    "heightScale":           (0.0, 10.0),
    "trailWidth":            (0.1, 10.0),
    "trailHeight":           (0.0, 10.0),
    "distanceTrackToBorder": (0.0, 1.0),
    "baseThickness":         (1.0, 60.0),
    "base_size":             (20.0, 400.0),
    "building_scale":        (0.1, 10.0),
    "printResolution":       (0.1, 0.8),
    "singleColor_gap":       (0.0, 10.0),
}


class Settings(BaseModel):
    """Validated generation settings. Field names ARE the wire schema keys."""

    model_config = ConfigDict(extra="ignore")

    waterColor: str = DEFAULT_SETTINGS["waterColor"]
    landColor: str = DEFAULT_SETTINGS["landColor"]
    trackColor: str = DEFAULT_SETTINGS["trackColor"]
    rockColor: str = DEFAULT_SETTINGS["rockColor"]
    snowColor: str = DEFAULT_SETTINGS["snowColor"]
    snowLevel: float = 0.0
    forestLevel: float = 0.0
    heightScale: float = 1.0
    standardizeHeight: bool = False
    trailWidth: float = 1.0
    trailHeight: float = 1.0
    useHeightFromGpx: bool = False
    shape: str = "hexagon"
    distanceTrackToBorder: float = 0.0
    baseThickness: float = 15.0
    includeSeas: bool = True
    includeLakes: bool = True
    includeRivers: bool = False
    base_size: float = 108.0
    center: str = "normal"
    buildings: bool = False
    building_scale: float = 1.0
    buildingsColor: str = DEFAULT_SETTINGS["buildingsColor"]
    printResolution: float = 0.2
    smooth: bool = True
    singleColor: bool = False
    singleColor_gap: float = 0.5
    baseColor: str = DEFAULT_SETTINGS["baseColor"]
    textColor: str = DEFAULT_SETTINGS["textColor"]
    borderLabels: list[str] = DEFAULT_SETTINGS["borderLabels"]

    @field_validator(*_CLAMPS.keys(), mode="after")
    @classmethod
    def _clamp(cls, v, info):
        lo, hi = _CLAMPS[info.field_name]
        return min(hi, max(lo, v))

    @field_validator(*_COLOR_KEYS, mode="before")
    @classmethod
    def _color(cls, v, info):
        # Tolerate junk colours (renderers fall back gracefully); reject
        # only non-string types.
        if not isinstance(v, str):
            raise ValueError("colour must be a string like '#RRGGBB'")
        return v

    @field_validator("borderLabels", mode="before")
    @classmethod
    def _labels(cls, v):
        if not isinstance(v, (list, tuple)):
            raise ValueError("borderLabels must be a list of strings")
        return [str(x) for x in v][:6]


class SettingsError(ValueError):
    """Raised when the settings payload cannot be validated."""


def normalize_settings(raw: dict) -> dict:
    """
    camelCase settings dict (with legacy snake_case fallbacks) -> validated,
    clamped dict with exactly the DEFAULT_SETTINGS keys.
    Raises SettingsError on unusable input.
    """
    if not isinstance(raw, dict):
        raise SettingsError("settings must be a JSON object")
    merged = {}
    for key in DEFAULT_SETTINGS:
        if key in raw:
            merged[key] = raw[key]
        elif LEGACY_KEYS.get(key) in raw:
            merged[key] = raw[LEGACY_KEYS[key]]
    try:
        model = Settings(**merged)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise SettingsError(f"Invalid settings: {problems}") from exc
    return model.model_dump()
