"""
Minimal Overpass API client.

overpass-api.de rejects requests without a real User-Agent (HTTP 406),
which overpy does not set — so we talk to the interpreter directly.
Responses are cached on disk (keyed by query) so re-generating the same
area doesn't re-download map features.
"""

import hashlib
import json
import time
from pathlib import Path

import requests

# Mirrors are rotated through on failure (rate limits, timeouts)
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
USER_AGENT = "TopoTrail/1.0 (github.com/topotrail; 3D-print map generator)"

CACHE_DIR = Path("generated") / "overpass"
CACHE_TTL_S = 24 * 3600


def query_overpass(query, retries=4):
    """POST an Overpass QL query, return decoded JSON (disk-cached, 24 h)."""
    cache_path = CACHE_DIR / f"{hashlib.sha256(query.encode()).hexdigest()}.json"
    if cache_path.exists() and time.time() - cache_path.stat().st_mtime < CACHE_TTL_S:
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass  # corrupt cache entry — refetch

    last_exc = None
    for attempt in range(retries):
        url = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]
        try:
            resp = requests.post(
                url,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data))
            return data
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
