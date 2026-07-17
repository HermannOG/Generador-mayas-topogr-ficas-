"""
Top-down preview map (image.png) for a generated job.

Colours match the model zones (rock / forest / water / snow), with the GPX
trails drawn on top in the track colour. Zone boundaries are supersampled so
they render as smooth outlines instead of blocky grid cells.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _hex_to_rgb(h, fallback=(128, 128, 128)):
    try:
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return fallback


def render_preview_png(grid, zone_map, zone_names, lat_bounds, lon_bounds,
                       trails, colors, out_path, size=512, ss=2):
    """
    grid        : elevation ndarray (rows, cols), row 0 = lat_max
    zone_map    : uint8 ndarray same shape, values index into zone_names
    zone_names  : tuple of zone names (e.g. MeshGenerator.ZONES)
    trails      : list of point lists [(lat, lon, ele), ...]
    colors      : hex strings keyed by zone name, plus "track"
    ss          : supersampling factor (render large, downscale with Lanczos)
    """
    big = size * ss
    ele_min = float(np.nanmin(grid))
    ele_max = float(np.nanmax(grid))
    norm = (grid - ele_min) / max(ele_max - ele_min, 1.0)

    # Smooth zone boundaries: upscale each zone's mask bilinearly, then take
    # the strongest zone per pixel — curved outlines instead of cell blocks
    ups = []
    for zid in range(len(zone_names)):
        m = Image.fromarray(((zone_map == zid) * 255).astype(np.uint8))
        m = m.resize((big, big), Image.BILINEAR).filter(ImageFilter.GaussianBlur(ss))
        ups.append(np.asarray(m, dtype=np.float32))
    zid_up = np.argmax(np.stack(ups), axis=0)

    palette = np.array(
        [_hex_to_rgb(colors.get(name, "#888888")) for name in zone_names],
        dtype=np.float32)
    rgb = palette[zid_up]

    # Elevation shading (water stays unshaded so lakes read flat)
    shade_img = Image.fromarray((norm * 255).astype(np.uint8)).resize(
        (big, big), Image.BILINEAR)
    shade = 0.55 + 0.45 * (np.asarray(shade_img, dtype=np.float32) / 255.0)
    water_ids = [i for i, n in enumerate(zone_names) if n == "water"]
    flat = np.isin(zid_up, water_ids)
    rgb *= np.where(flat, 1.0, shade)[..., None]

    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")

    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds
    draw = ImageDraw.Draw(img)
    track = _hex_to_rgb(colors.get("track", "#FC5200"))
    for points in trails or []:
        px = [
            (
                (p[1] - lon_min) / max(lon_max - lon_min, 1e-9) * big,
                (lat_max - p[0]) / max(lat_max - lat_min, 1e-9) * big,
            )
            for p in points
            if lat_min <= p[0] <= lat_max and lon_min <= p[1] <= lon_max
        ]
        if len(px) >= 2:
            draw.line(px, fill=track, width=3 * ss, joint="curve")

    # Downscale with Lanczos = antialiased edges and trail
    img = img.resize((size, size), Image.LANCZOS)
    img.save(out_path, "PNG")
