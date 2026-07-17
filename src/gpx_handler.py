import math

import gpxpy
import gpxpy.gpx


def parse_gpx_with_type(file_content):
    """
    Parse GPX file content.

    Returns:
        (points, activity_type)
        points        : list of (lat, lon, elevation) tuples
        activity_type : lowercase activity type from the first track's <type>
                        tag (e.g. "swimming", "cycling"), or "" if absent
    """
    if isinstance(file_content, bytes):
        file_content = file_content.decode("utf-8")

    gpx = gpxpy.parse(file_content)
    points = []
    activity = ""

    for track in gpx.tracks:
        if not activity and track.type:
            activity = track.type.strip().lower()
        for segment in track.segments:
            for pt in segment.points:
                ele = pt.elevation if pt.elevation is not None else 0.0
                points.append((pt.latitude, pt.longitude, ele))

    for route in gpx.routes:
        for pt in route.points:
            ele = pt.elevation if pt.elevation is not None else 0.0
            points.append((pt.latitude, pt.longitude, ele))

    return points, activity


def is_swim(activity_type):
    """True when the GPX activity is a swim (flat-water elevation rules apply)."""
    return "swim" in (activity_type or "")


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
        # Raw extent of the points — margin/padding is the caller's business
        "span_km": max(lat_span_km, lon_span_km),
    }
