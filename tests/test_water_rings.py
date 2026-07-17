from src.water import _assemble_rings


def test_closed_ring_passthrough():
    assert _assemble_rings([[1, 2, 3, 1]]) == [[1, 2, 3, 1]]


def test_two_fragments_forward():
    rings = _assemble_rings([[1, 2, 3], [3, 4, 1]])
    assert len(rings) == 1
    assert rings[0][0] == rings[0][-1]
    assert set(rings[0]) == {1, 2, 3, 4}


def test_reversed_fragment():
    # second segment shares endpoints but runs the other way
    rings = _assemble_rings([[1, 2, 3], [1, 4, 3]])
    assert len(rings) == 1
    assert rings[0][0] == rings[0][-1]
    assert set(rings[0]) == {1, 2, 3, 4}


def test_prepend_fragment():
    rings = _assemble_rings([[2, 3, 4], [1, 2], [4, 1]])
    assert len(rings) == 1
    assert rings[0][0] == rings[0][-1]


def test_multiple_independent_rings():
    rings = _assemble_rings([[1, 2, 3, 1], [10, 11], [11, 12, 10]])
    assert len(rings) == 2


def test_short_segments_dropped():
    assert _assemble_rings([[1], []]) == []
