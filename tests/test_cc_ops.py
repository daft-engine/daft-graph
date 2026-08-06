"""Tests for the large star and small star operations."""

from __future__ import annotations

import daft

from daft_graph.algorithms.connected_components import large_star, small_star
from daft_graph.schema import DST, SRC


def _edge_set(df: daft.DataFrame) -> set:
    d = df.select(SRC, DST).collect().to_pydict()
    return set(zip(d[SRC], d[DST]))


def test_large_star_points_to_min() -> None:
    edges = daft.from_pydict({SRC: [1, 2], DST: [2, 3]})
    assert _edge_set(large_star(edges)) == {(2, 1), (3, 1)}


def test_small_star_merges_local_minima() -> None:
    edges = daft.from_pydict({SRC: [1, 1, 2], DST: [2, 3, 3]})
    assert _edge_set(small_star(edges)) == {(2, 1), (3, 1), (3, 2)}
