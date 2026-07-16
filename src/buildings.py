from src.overpass import index_elements, query_overpass, way_coords


def fetch_buildings(lat_min, lat_max, lon_min, lon_max, max_buildings=2000):
    """
    Fetch building footprints from OpenStreetMap via Overpass API.

    Returns:
        list of dicts: {coords: [(lat, lon)...], height_tag: str|None}
    """
    query = f"""
    [out:json][timeout:60];
    (
      way["building"]({lat_min},{lon_min},{lat_max},{lon_max});
    );
    out body;
    >;
    out skel qt;
    """

    try:
        data = query_overpass(query)
    except Exception:
        return []

    nodes, ways, _ = index_elements(data)

    buildings = []
    for way in list(ways.values())[:max_buildings]:
        coords = way_coords(way, nodes)
        if len(coords) < 3:
            continue
        tags = way.get("tags", {})
        height = tags.get("height", tags.get("building:levels", None))
        buildings.append({"coords": coords, "height_tag": height})
    return buildings
