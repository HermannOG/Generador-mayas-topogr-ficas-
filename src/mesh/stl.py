"""Binary STL serialisation (vectorised, ndarray-native)."""

import struct

import numpy as np


def to_stl_bytes(triangles):
    """Serialise triangles ((N,3,3) ndarray or nested lists) to binary STL."""
    tris = np.asarray(triangles, dtype=np.float64)
    if tris.size == 0:
        return b"\x00" * 80 + struct.pack("<I", 0)
    tris = tris.reshape(-1, 3, 3)

    a = tris[:, 1] - tris[:, 0]
    b = tris[:, 2] - tris[:, 0]
    n = np.cross(a, b)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(length > 1e-10, n / np.maximum(length, 1e-30), 0.0)

    rec = np.zeros(len(tris), dtype=[
        ("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2"),
    ])
    rec["n"] = n
    rec["v"] = tris
    return b"\x00" * 80 + struct.pack("<I", len(tris)) + rec.tobytes()


def parse_stl_bytes(data):
    """Parse binary STL back to (normals (N,3), vertices (N,3,3)). For tests."""
    count = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:], dtype=[
        ("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2"),
    ], count=count)
    return rec["n"].astype(np.float64), rec["v"].astype(np.float64)
