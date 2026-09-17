"""Tests for daft_graph.algorithms.label_propagation."""

from __future__ import annotations

from collections import defaultdict

import daft

from daft_graph.algorithms.label_propagation import label_propagation
from daft_graph.graph import UndirectedGraph
from daft_graph.schema import DST, ID, LABEL, SRC


def _label_map(df: daft.DataFrame) -> dict:
    d = df.collect().to_pydict()
    return dict(zip(d[ID], d[LABEL]))


def _partition(df: daft.DataFrame) -> set:
    d = df.collect().to_pydict()
    groups: dict[int, set] = defaultdict(set)
    for node, lab in zip(d[ID], d[LABEL]):
        groups[lab].add(node)
    return {frozenset(members) for members in groups.values()}


def test_two_triangles_recovers_communities() -> None:
    # triangle {1,2,3} and triangle {4,5,6}, disconnected
    edges = daft.from_pydict({SRC: [1, 2, 1, 4, 5, 4], DST: [2, 3, 3, 5, 6, 6]})
    g = UndirectedGraph(edges)
    assert _partition(label_propagation(g)) == {
        frozenset({1, 2, 3}),
        frozenset({4, 5, 6}),
    }


def test_clique_is_single_community() -> None:
    edges = daft.from_pydict({SRC: [1, 1, 1, 2, 2, 3], DST: [2, 3, 4, 3, 4, 4]})
    g = UndirectedGraph(edges)
    assert _partition(label_propagation(g)) == {frozenset({1, 2, 3, 4})}


def test_deterministic_across_runs() -> None:
    edges = daft.from_pydict({SRC: [1, 2, 1, 4, 5, 4], DST: [2, 3, 3, 5, 6, 6]})
    g = UndirectedGraph(edges)
    assert _label_map(label_propagation(g)) == _label_map(label_propagation(g))


def test_isolated_vertex_is_its_own_community() -> None:
    vertices = daft.from_pydict({ID: [1, 2, 3]})
    edges = daft.from_pydict({SRC: [1], DST: [2]})
    g = UndirectedGraph(vertices=vertices, edges=edges)
    assert frozenset({3}) in _partition(label_propagation(g))
