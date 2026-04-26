import time

import overpy


def fetch_buildings(lat_min, lat_max, lon_min, lon_max, max_buildings=2000):
    """
    Fetch building footprints from OpenStreetMap via Overpass API.

    Returns:
        list of dicts: {coords: [(lat, lon)...], height_tag: str|None}
    """
    api = overpy.Overpass()
    query = f"""
    [out:json][timeout:60];
    (
      way["building"]({lat_min},{lon_min},{lat_max},{lon_max});
    );
    out body;
    >;
    out skel qt;
    """

    for attempt in range(3):
        try:
            result = api.query(query)
            buildings = []
            for way in result.ways[:max_buildings]:
                try:
                    coords = [(float(n.lat), float(n.lon)) for n in way.nodes]
                    if len(coords) < 3:
                        continue
                    height = way.tags.get(
                        "height", way.tags.get("building:levels", None)
                    )
                    buildings.append({"coords": coords, "height_tag": height})
                except Exception:
                    continue
            return buildings
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)

    return []
