"""
Minimal Overpass API client.

overpass-api.de rejects requests without a real User-Agent (HTTP 406),
which overpy does not set — so we talk to the interpreter directly.
"""

import time

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "TopoTrail/1.0 (github.com/topotrail; 3D-print map generator)"


def query_overpass(query, retries=3):
    """POST an Overpass QL query, return decoded JSON. Raises on failure."""
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=90,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise last_exc


def index_elements(data):
    """Split an Overpass JSON response into (nodes, ways, relations) dicts by id."""
    nodes, ways, rels = {}, {}, {}
    for el in data.get("elements", []):
        if el["type"] == "node":
            nodes[el["id"]] = (el["lat"], el["lon"])
        elif el["type"] == "way":
            ways[el["id"]] = el
        elif el["type"] == "relation":
            rels[el["id"]] = el
    return nodes, ways, rels


def way_coords(way, nodes):
    """Resolve a way's node refs to [(lat, lon), ...], skipping missing nodes."""
    return [nodes[ref] for ref in way.get("nodes", []) if ref in nodes]
