"""
Map features from OpenStreetMap via Overpass: water bodies and forests.

Polygon features carry optional hole rings — islands inside lakes and
clearings inside forests — assembled from multipolygon inner ways.
"""

from shapely.geometry import Point, Polygon

from src.overpass import index_elements, query_overpass, way_coords


def fetch_map_features(lat_min, lat_max, lon_min, lon_max, max_features=2000):
    """
    Fetch water bodies and forest polygons in one Overpass query.

    Returns (water, forests):
      water   : lakes/seas as {"type": 'lake'|'sea', "coords": ring, "holes": [rings]}
                rivers as {"type": 'river', "line": [(lat, lon)...]}
                (ordered way; OSM convention: points run downstream)
      forests : {"coords": ring, "holes": [rings]}
    """
    bbox = f"({lat_min},{lon_min},{lat_max},{lon_max})"
    query = f"""
    [out:json][timeout:90];
    (
      way["natural"~"^(water|bay)$"]{bbox};
      way["waterway"~"^(river|stream|canal)$"]{bbox};
      relation["natural"~"^(water|bay)$"]{bbox};
      way["natural"="wood"]{bbox};
      way["landuse"="forest"]{bbox};
      relation["natural"="wood"]{bbox};
      relation["landuse"="forest"]{bbox};
    );
    out body;
    >;
    out skel qt;
    """

    try:
        data = query_overpass(query)
    except Exception:
        return [], []

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
        if natural == "wood" or tags.get("landuse") == "forest":
            return "forest"
        return None

    water, forests = [], []

    def _add(kind, coords, holes):
        if len(coords) < 3:
            return
        feature = {"coords": coords, "holes": holes}
        if kind == "forest":
            forests.append(feature)
        else:
            feature["type"] = kind
            water.append(feature)

    # Ways — the response also contains untagged member/skeleton ways from
    # the `>` recursion, so only classify the tagged ones.
    for way in ways.values():
        kind = _classify(way.get("tags") or {})
        if not kind:
            continue
        coords = way_coords(way, nodes)
        if kind == "river":
            # Rivers are linear ways, not rings — keep OSM point order
            if len(coords) >= 2:
                water.append({"type": "river", "line": coords})
        else:
            _add(kind, coords, [])

    # Relations (multipolygons) – member ways are ring FRAGMENTS that must be
    # stitched end-to-end. "outer" rings are the body, "inner" rings are the
    # holes (islands in lakes, clearings in forests).
    for rel in rels.values():
        kind = _classify(rel.get("tags") or {})
        if not kind or kind == "river":
            continue
        outer_segs, inner_segs = [], []
        for member in rel.get("members", []):
            if member.get("type") != "way":
                continue
            way = ways.get(member.get("ref"))
            if not way or len(way.get("nodes", [])) < 2:
                continue
            role = member.get("role")
            if role in ("outer", ""):
                outer_segs.append(list(way["nodes"]))
            elif role == "inner":
                inner_segs.append(list(way["nodes"]))

        inner_rings = [
            [nodes[r] for r in ring if r in nodes]
            for ring in _assemble_rings(inner_segs)
        ]
        for ring_refs in _assemble_rings(outer_segs):
            coords = [nodes[r] for r in ring_refs if r in nodes]
            if len(coords) < 3:
                continue
            try:
                poly = Polygon(coords)
                holes = [h for h in inner_rings
                         if len(h) >= 3 and poly.contains(Point(h[0]))]
            except Exception:
                holes = []
            _add(kind, coords, holes)

    # If over budget, keep the largest features (ring/line length as proxy)
    def _cap(features):
        if len(features) <= max_features:
            return features
        features.sort(
            key=lambda f: len(f.get("coords") or f.get("line") or ()), reverse=True)
        return features[:max_features]

    return _cap(water), _cap(forests)


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
