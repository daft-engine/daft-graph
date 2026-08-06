"""Tests for empty and edgeless graphs across all algorithms."""

from __future__ import annotations

import daft

from daft_graph import DirectedGraph, connected_components, label_propagation, pagerank
from daft_graph.schema import COMPONENT, DST, ID, LABEL, RANK, SRC


def _edgeless_graph(ids: list[int]) -> DirectedGraph:
    vertices = daft.from_pydict({ID: ids})
    edges = daft.from_pydict({SRC: [], DST: []})
    return DirectedGraph(vertices=vertices, edges=edges)


def _comp_map(df: daft.DataFrame) -> dict:
    d = df.collect().to_pydict()
    return dict(zip(d[ID], d[COMPONENT]))


def test_cc_distributed_no_edges() -> None:
    g = _edgeless_graph([1, 2, 3])
    assert _comp_map(connected_components(g, strategy="distributed")) == {
        1: 1,
        2: 2,
        3: 3,
    }


def test_cc_local_no_edges() -> None:
    g = _edgeless_graph([1, 2, 3])
    assert _comp_map(connected_components(g, strategy="local")) == {1: 1, 2: 2, 3: 3}


def test_cc_auto_no_edges() -> None:
    g = _edgeless_graph([1, 2, 3])
    assert _comp_map(connected_components(g)) == {1: 1, 2: 2, 3: 3}


def test_label_propagation_no_edges() -> None:
    g = _edgeless_graph([1, 2, 3])
    d = label_propagation(g).collect().to_pydict()
    assert dict(zip(d[ID], d[LABEL])) == {1: 1, 2: 2, 3: 3}


def test_pagerank_no_edges_is_uniform() -> None:
    # With no edges every node is dangling; the stationary distribution is the
    # uniform personalization vector 1/n.
    ids = [0, 1, 2, 3]
    g = _edgeless_graph(ids)
    d = pagerank(g).collect().to_pydict()
    ranks = dict(zip(d[ID], d[RANK]))
    for value in ranks.values():
        assert abs(value - 0.25) < 1e-9
    assert abs(sum(ranks.values()) - 1.0) < 1e-9
