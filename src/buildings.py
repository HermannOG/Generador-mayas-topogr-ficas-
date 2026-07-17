from src.overpass import index_elements, query_overpass, way_coords


def fetch_buildings(lat_min, lat_max, lon_min, lon_max, max_buildings=2000):
    """
    Fetch building footprints from OpenStreetMap via Overpass API.

    Returns:
        list of dicts: {coords: [(lat, lon)...], height_tag: str|None,
                        height_m: float|None}
        height_m is the parsed real-world height: the "height" tag in metres
        (units stripped) or building:levels x 3 m; None when untagged/junk.
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
        buildings.append({
            "coords": coords,
            "height_tag": height,
            "height_m": _parse_height_m(tags),
        })
    return buildings


def _parse_height_m(tags):
    """Real-world building height in metres from OSM tags, or None.

    "height" is metres by convention, sometimes with a unit suffix
    ("12", "12 m", "12m"); "building:levels" counts storeys (~3 m each).
    Junk tags simply yield None.
    """
    def _num(val):
        try:
            v = float(str(val).strip().split()[0].rstrip("mM"))
            return v if v > 0 else None
        except (ValueError, IndexError):
            return None

    h = _num(tags.get("height"))
    if h is not None:
        return h
    levels = _num(tags.get("building:levels"))
    if levels is not None and levels < 200:   # junk guard
        return levels * 3.0
    return None
