import gpxpy
import gpxpy.gpx
import math


def parse_gpx(file_content):
    """
    Parse GPX file content and return track points.

    Args:
        file_content: bytes or str

    Returns:
        list of (lat, lon, elevation) tuples
    """
    if isinstance(file_content, bytes):
        file_content = file_content.decode("utf-8")

    gpx = gpxpy.parse(file_content)
    points = []

    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                ele = pt.elevation if pt.elevation is not None else 0.0
                points.append((pt.latitude, pt.longitude, ele))

    for route in gpx.routes:
        for pt in route.points:
            ele = pt.elevation if pt.elevation is not None else 0.0
            points.append((pt.latitude, pt.longitude, ele))

    return points


def get_gpx_bounds(points):
    if not points:
        return None
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    lat_c = (min(lats) + max(lats)) / 2
    lon_c = (min(lons) + max(lons)) / 2
    lat_span_km = (max(lats) - min(lats)) * 111.0
    lon_span_km = (max(lons) - min(lons)) * 111.0 * math.cos(math.radians(lat_c))
    return {
        "lat_min": min(lats),
        "lat_max": max(lats),
        "lon_min": min(lons),
        "lon_max": max(lons),
        "lat_center": lat_c,
        "lon_center": lon_c,
        "size_km": max(lat_span_km, lon_span_km) * 1.3,
    }
