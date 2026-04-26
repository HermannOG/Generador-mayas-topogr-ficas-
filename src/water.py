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
      way["natural"="water"]({lat_min},{lon_min},{lat_max},{lon_max});
      way["waterway"~"^(river|stream|canal)$"]({lat_min},{lon_min},{lat_max},{lon_max});
      relation["natural"="water"]({lat_min},{lon_min},{lat_max},{lon_max});
    );
    out body;
    >;
    out skel qt;
    """

    for attempt in range(3):
        try:
            result = api.query(query)
            features = []

            for way in result.ways[:max_features]:
                natural = way.tags.get("natural", "")
                waterway = way.tags.get("waterway", "")

                if natural == "water":
                    water_sub = way.tags.get("water", "lake")
                    w_type = "sea" if water_sub in ("sea", "ocean", "tidal") else "lake"
                elif waterway in ("river", "stream", "canal"):
                    w_type = "river"
                else:
                    continue

                try:
                    coords = [(float(n.lat), float(n.lon)) for n in way.nodes]
                    if len(coords) >= 3:
                        features.append({"type": w_type, "coords": coords})
                except Exception:
                    continue

            return features

        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)

    return []
