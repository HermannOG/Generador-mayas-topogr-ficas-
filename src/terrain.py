import math
import time

import numpy as np
import requests
from scipy.ndimage import gaussian_filter


def fetch_elevation_grid(lat_center, lon_center, size_km, resolution=50):
    """
    Fetch a resolution×resolution elevation grid via OpenTopoData (SRTM 90m).

    Returns:
        elevation_grid : ndarray (resolution, resolution)  [meters]
        lat_bounds     : (lat_min, lat_max)
        lon_bounds     : (lon_min, lon_max)
    """
    resolution = max(10, min(resolution, 100))

    lat_delta = (size_km / 2.0) / 111.0
    lon_delta = (size_km / 2.0) / (111.0 * math.cos(math.radians(lat_center)))

    lat_min = lat_center - lat_delta
    lat_max = lat_center + lat_delta
    lon_min = lon_center - lon_delta
    lon_max = lon_center + lon_delta

    lats = np.linspace(lat_max, lat_min, resolution)   # north → south
    lons = np.linspace(lon_min, lon_max, resolution)   # west  → east
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    locations = list(zip(lat_grid.flatten().tolist(), lon_grid.flatten().tolist()))
    elevations = []
    batch_size = 100

    for i in range(0, len(locations), batch_size):
        batch = locations[i : i + batch_size]
        loc_str = "|".join(f"{lat},{lon}" for lat, lon in batch)
        fetched = False

        for attempt in range(3):
            try:
                resp = requests.get(
                    "https://api.opentopodata.org/v1/srtm90m",
                    params={"locations": loc_str},
                    timeout=30,
                )
                data = resp.json()
                if data.get("status") == "OK":
                    elevations.extend(
                        r["elevation"] if r["elevation"] is not None else 0.0
                        for r in data["results"]
                    )
                    fetched = True
                    break
            except Exception:
                if attempt < 2:
                    time.sleep(1 + attempt)

        if not fetched:
            elevations.extend([0.0] * len(batch))

        if i + batch_size < len(locations):
            time.sleep(1.1)   # respect public API rate limit (1 req/s)

    grid = np.array(elevations, dtype=float).reshape(resolution, resolution)
    grid = gaussian_filter(grid, sigma=0.5)   # slight smoothing
    return grid, (lat_min, lat_max), (lon_min, lon_max)
