"""Tests for power_iteration_clustering."""

from __future__ import annotations

import daft
import pytest

from daft_graph.algorithms.power_iteration_clustering import (
    CLUSTER,
    power_iteration_clustering,
)
from daft_graph.graph import UndirectedGraph
from daft_graph.schema import DST, ID, SRC


def _graph(node_ids: list[int], edges: list[tuple[int, int]]) -> UndirectedGraph:
    return UndirectedGraph(
        vertices=daft.from_pydict({ID: node_ids}),
        edges=daft.from_pydict({SRC: [u for u, _ in edges], DST: [v for _, v in edges]}),
    )


def _clusters(g: UndirectedGraph, k: int) -> dict:
    d = power_iteration_clustering(g, k).collect().to_pydict()
    return dict(zip(d[ID], d[CLUSTER]))


def _barbell() -> UndirectedGraph:
    # asymmetric barbell: a triangle {0,1,2} and a 5-clique {3,4,5,6,7}, bridged 2-3
    triangle = [(0, 1), (0, 2), (1, 2)]
    clique = [
        (3, 4),
        (3, 5),
        (3, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (4, 7),
        (5, 6),
        (5, 7),
        (6, 7),
    ]
    bridge = [(2, 3)]
    return _graph(list(range(8)), triangle + clique + bridge)


def test_separates_two_communities() -> None:
    cl = _clusters(_barbell(), 2)
    # each community core lands in a single cluster, and the two are different
    assert cl[0] == cl[1] == cl[2]
    assert cl[4] == cl[5] == cl[6] == cl[7]
    assert cl[0] != cl[4]


def test_at_most_k_clusters() -> None:
    assert len(set(_clusters(_barbell(), 2).values())) <= 2


def test_deterministic() -> None:
    g = _barbell()
    assert _clusters(g, 2) == _clusters(g, 2)


def test_invalid_k_raises() -> None:
    with pytest.raises(ValueError, match="k must be"):
        power_iteration_clustering(_barbell(), 0)
