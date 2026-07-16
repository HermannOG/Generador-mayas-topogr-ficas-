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


def render_preview_png(grid, water_mask, lat_bounds, lon_bounds, trails,
                       rock_mask, colors, out_path, size=320):
    """
    grid        : elevation ndarray (rows, cols), row 0 = lat_max
    water_mask  : boolean ndarray same shape, or None
    trails      : list of point lists [(lat, lon, ele), ...]
    rock_mask   : boolean ndarray same shape (non-vegetated terrain), or None
    colors      : {"land","rock","water","track"} hex strings
    """
    rows, cols = grid.shape
    ele_min = float(np.nanmin(grid))
    ele_max = float(np.nanmax(grid))
    norm = (grid - ele_min) / max(ele_max - ele_min, 1.0)
    shade = 0.55 + 0.45 * norm  # brighter with elevation

    land = np.array(_hex_to_rgb(colors.get("land", "#00FF00")), dtype=float)
    rock = np.array(_hex_to_rgb(colors.get("rock", "#BDBDBD")), dtype=float)
    water = np.array(_hex_to_rgb(colors.get("water", "#0084ff")), dtype=float)

    rgb = np.empty((rows, cols, 3), dtype=float)
    rgb[:] = land
    if rock_mask is not None:
        rgb[rock_mask] = rock
    rgb *= shade[..., None]
    if water_mask is not None:
        rgb[water_mask] = water

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
