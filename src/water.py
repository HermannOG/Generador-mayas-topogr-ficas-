from src.overpass import index_elements, query_overpass, way_coords


def fetch_water_bodies(lat_min, lat_max, lon_min, lon_max, max_features=2000):
    """
    Fetch water bodies from OpenStreetMap via Overpass API.

    Returns list of dicts:
      lakes/seas : {type: 'lake'|'sea', coords: [(lat, lon)...]}   (polygon ring)
      rivers     : {type: 'river', line: [(lat, lon)...]}          (ordered way;
                   OSM convention: points run downstream)
    """
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

    try:
        data = query_overpass(query)
    except Exception:
        return []

    nodes, ways, rels = index_elements(data)

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

    features = []

    # Ways — the response also contains untagged member/skeleton ways from
    # the `>` recursion, so only classify the tagged ones.
    for way in ways.values():
        w_type = _classify(way.get("tags") or {})
        if not w_type:
            continue
        coords = way_coords(way, nodes)
        if w_type == "river":
            # Rivers are linear ways, not rings — keep OSM point order
            if len(coords) >= 2:
                features.append({"type": "river", "line": coords})
        elif len(coords) >= 3:
            features.append({"type": w_type, "coords": coords})

    # Relations (multipolygons) – outer member ways are ring FRAGMENTS that
    # must be stitched end-to-end into closed rings, otherwise each fragment
    # would be treated as its own (wrongly shaped) water body.
    for rel in rels.values():
        w_type = _classify(rel.get("tags") or {})
        if not w_type or w_type == "river":
            continue
        segments = []
        for member in rel.get("members", []):
            if member.get("type") != "way" or member.get("role") not in ("outer", ""):
                continue
            way = ways.get(member.get("ref"))
            if way and len(way.get("nodes", [])) >= 2:
                segments.append(list(way["nodes"]))
        for ring_refs in _assemble_rings(segments):
            coords = [nodes[ref] for ref in ring_refs if ref in nodes]
            if len(coords) >= 3:
                features.append({"type": w_type, "coords": coords})

    # If over budget, keep the largest features (ring/line length as proxy)
    if len(features) > max_features:
        features.sort(
            key=lambda f: len(f.get("coords") or f.get("line") or ()), reverse=True)
        features = features[:max_features]

    return features


def _assemble_rings(segments):
    """Stitch way node-ref segments into rings by matching endpoints."""
    segments = [s for s in segments if len(s) >= 2]
    rings = []
    while segments:
        ring = segments.pop(0)
        progressing = True
        while progressing and ring[0] != ring[-1]:
            progressing = False
            for k, seg in enumerate(segments):
                if seg[0] == ring[-1]:
                    ring = ring + seg[1:]
                elif seg[-1] == ring[-1]:
                    ring = ring + seg[-2::-1]
                elif seg[-1] == ring[0]:
                    ring = seg[:-1] + ring
                elif seg[0] == ring[0]:
                    ring = seg[::-1][:-1] + ring
                else:
                    continue
                segments.pop(k)
                progressing = True
                break
        if len(ring) >= 3:
            rings.append(ring)
    return rings
