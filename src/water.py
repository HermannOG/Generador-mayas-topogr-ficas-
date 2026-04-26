import time

import overpy


def fetch_water_bodies(lat_min, lat_max, lon_min, lon_max, max_features=500):
    """
    Fetch water bodies from OpenStreetMap via Overpass API.

    Returns list of dicts: {type: 'lake'|'sea'|'river', coords: [(lat, lon)...]}
    """
    api = overpy.Overpass()
    query = f"""
    [out:json][timeout:60];
    (
      way["natural"~"^(water|bay)$"]({lat_min},{lon_min},{lat_max},{lon_max});
      way["waterway"~"^(river|stream|canal)$"]({lat_min},{lon_min},{lat_max},{lon_max});
      relation["natural"~"^(water|bay)$"]({lat_min},{lon_min},{lat_max},{lon_max});
    );
    out body;
    >;
    out skel qt;
    """

    for attempt in range(3):
        try:
            result = api.query(query)
            features = []
            way_lookup = {w.id: w for w in result.ways}

            def _classify(tags):
                natural  = tags.get("natural",  "")
                waterway = tags.get("waterway", "")
                water    = tags.get("water",    "")
                if natural in ("water", "bay"):
                    if natural == "bay" or water in ("sea", "ocean", "bay", "tidal", "lagoon"):
                        return "sea"
                    return "lake"
                if waterway in ("river", "stream", "canal"):
                    return "river"
                return None

            def _coords(way):
                try:
                    return [(float(n.lat), float(n.lon)) for n in way.nodes]
                except Exception:
                    return []

            # Ways
            for way in result.ways[:max_features]:
                w_type = _classify(way.tags)
                if not w_type:
                    continue
                coords = _coords(way)
                if len(coords) >= 3:
                    features.append({"type": w_type, "coords": coords})

            # Relations – extract outer member ways
            for rel in result.relations[:max_features]:
                w_type = _classify(rel.tags)
                if not w_type:
                    continue
                for member in rel.members:
                    if member.role not in ("outer", ""):
                        continue
                    if not isinstance(member, overpy.RelationWay):
                        continue
                    way = way_lookup.get(member.ref)
                    if way:
                        coords = _coords(way)
                        if len(coords) >= 3:
                            features.append({"type": w_type, "coords": coords})

            return features

        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)

    return []
