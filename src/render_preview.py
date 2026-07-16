"""
Top-down preview thumbnail (image.png) for a generated map job.

Colours match the model zones: land / rock (above tree line) / water,
with the GPX trails drawn on top in the track colour.
"""

import numpy as np
from PIL import Image, ImageDraw


def _hex_to_rgb(h, fallback=(128, 128, 128)):
    try:
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return fallback


def render_preview_png(grid, zone_map, zone_names, lat_bounds, lon_bounds,
                       trails, colors, out_path, size=512):
    """
    grid        : elevation ndarray (rows, cols), row 0 = lat_max
    zone_map    : uint8 ndarray same shape, values index into zone_names
    zone_names  : tuple of zone names (e.g. MeshGenerator.ZONES)
    trails      : list of point lists [(lat, lon, ele), ...]
    colors      : hex strings keyed by zone name, plus "track"
    """
    rows, cols = grid.shape
    ele_min = float(np.nanmin(grid))
    ele_max = float(np.nanmax(grid))
    norm = (grid - ele_min) / max(ele_max - ele_min, 1.0)
    shade = 0.55 + 0.45 * norm  # brighter with elevation

    palette = np.array(
        [_hex_to_rgb(colors.get(name, "#888888")) for name in zone_names],
        dtype=float)
    rgb = palette[zone_map]
    flat = np.isin(zone_map, [i for i, n in enumerate(zone_names) if n == "water"])
    rgb *= np.where(flat, 1.0, shade)[..., None]   # water stays unshaded

    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    img = img.resize((size, size), Image.BILINEAR)

    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds
    draw = ImageDraw.Draw(img)
    track = _hex_to_rgb(colors.get("track", "#FC5200"))
    for points in trails or []:
        px = [
            (
                (p[1] - lon_min) / max(lon_max - lon_min, 1e-9) * size,
                (lat_max - p[0]) / max(lat_max - lat_min, 1e-9) * size,
            )
            for p in points
            if lat_min <= p[0] <= lat_max and lon_min <= p[1] <= lon_max
        ]
        if len(px) >= 2:
            draw.line(px, fill=track, width=3, joint="curve")

    img.save(out_path, "PNG")
