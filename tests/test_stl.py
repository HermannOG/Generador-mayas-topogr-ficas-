import numpy as np

from src.mesh.stl import parse_stl_bytes, to_stl_bytes


def test_roundtrip():
    tris = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                     [[0, 0, 1], [1, 0, 1], [1, 1, 1]]], dtype=np.float64)
    normals, verts = parse_stl_bytes(to_stl_bytes(tris))
    assert verts.shape == (2, 3, 3)
    assert np.allclose(verts, tris)
    assert np.allclose(normals[0], [0, 0, 1])


def test_accepts_nested_lists():
    data = to_stl_bytes([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]])
    _, verts = parse_stl_bytes(data)
    assert verts.shape == (1, 3, 3)


def test_empty():
    data = to_stl_bytes(np.zeros((0, 3, 3)))
    assert len(data) == 84
    normals, verts = parse_stl_bytes(data)
    assert len(verts) == 0
