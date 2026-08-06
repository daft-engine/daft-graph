"""Tests for the connected_components driver."""

from __future__ import annotations

import daft

from daft_graph.algorithms.connected_components import connected_components
from daft_graph.graph import UndirectedGraph
from daft_graph.schema import COMPONENT, DST, ID, SRC


def _comp_map(df: daft.DataFrame) -> dict:
    d = df.collect().to_pydict()
    return dict(zip(d[ID], d[COMPONENT]))


def test_two_components() -> None:
    edges = daft.from_pydict({SRC: [1, 2, 4], DST: [2, 3, 5]})
    g = UndirectedGraph(edges)
    assert _comp_map(connected_components(g)) == {1: 1, 2: 1, 3: 1, 4: 4, 5: 4}


def test_single_edge() -> None:
    edges = daft.from_pydict({SRC: [10], DST: [20]})
    g = UndirectedGraph(edges)
    assert _comp_map(connected_components(g)) == {10: 10, 20: 10}


def test_star_graph() -> None:
    edges = daft.from_pydict({SRC: [1, 1, 1, 1], DST: [2, 3, 4, 5]})
    g = UndirectedGraph(edges)
    assert _comp_map(connected_components(g)) == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}


def test_isolated_vertex_is_its_own_component() -> None:
    vertices = daft.from_pydict({ID: [1, 2, 3]})
    edges = daft.from_pydict({SRC: [1], DST: [2]})
    g = UndirectedGraph(vertices=vertices, edges=edges)
    assert _comp_map(connected_components(g)) == {1: 1, 2: 1, 3: 3}


def test_chain_collapses_to_global_min() -> None:
    # 5-4-3-2-1 chain should all land in component 1
    edges = daft.from_pydict({SRC: [5, 4, 3, 2], DST: [4, 3, 2, 1]})
    g = UndirectedGraph(edges)
    assert _comp_map(connected_components(g)) == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
