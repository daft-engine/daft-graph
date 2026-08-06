"""The graph transforms must return the caller's concrete class.

``filter_vertices``, ``filter_edges``, and ``drop_isolated_vertices`` rebuild a
graph internally. Before the class split they hardcoded the single ``Graph``, so
after the split a naive port would silently hand back the wrong flavor and an
undirected graph would start traversing directionally. These tests pin that down.
"""

from __future__ import annotations

import daft
import pytest
from daft import col

from daft_graph.graph import DirectedGraph, Graph, UndirectedGraph
from daft_graph.schema import DST, ID, SRC

_FLAVORS = [DirectedGraph, UndirectedGraph]


def _build(flavor: type[Graph]) -> Graph:
    edges = daft.from_pydict({SRC: [1, 1, 2], DST: [2, 3, 3], "weight": [1.0, 5.0, 9.0]})
    vertices = daft.from_pydict({ID: [1, 2, 3, 99], "score": [10, 20, 30, 40]})
    return flavor(edges, vertices)


@pytest.mark.parametrize("flavor", _FLAVORS)
def test_filter_vertices_preserves_flavor(flavor: type[Graph]) -> None:
    g = _build(flavor)
    assert type(g.filter_vertices(col(ID) != 3)) is flavor


@pytest.mark.parametrize("flavor", _FLAVORS)
def test_filter_edges_preserves_flavor(flavor: type[Graph]) -> None:
    g = _build(flavor)
    assert type(g.filter_edges(col("weight") >= 5.0)) is flavor


@pytest.mark.parametrize("flavor", _FLAVORS)
def test_drop_isolated_vertices_preserves_flavor(flavor: type[Graph]) -> None:
    g = _build(flavor)
    assert type(g.drop_isolated_vertices()) is flavor


@pytest.mark.parametrize("flavor", _FLAVORS)
def test_chained_transforms_preserve_flavor(flavor: type[Graph]) -> None:
    g = _build(flavor)
    chained = g.filter_edges(col("weight") >= 5.0).drop_isolated_vertices()
    assert type(chained) is flavor


@pytest.mark.parametrize("flavor", _FLAVORS)
def test_transforms_keep_traversal_semantics(flavor: type[Graph]) -> None:
    """A rebuilt graph must walk edges the same way the original did."""
    g = _build(flavor)
    rebuilt = g.filter_edges(col("weight") >= 1.0)
    assert rebuilt._traversal_edges().count_rows() == g._traversal_edges().count_rows()


@pytest.mark.parametrize("flavor", _FLAVORS)
def test_transforms_preserve_attribute_columns(flavor: type[Graph]) -> None:
    g = _build(flavor)
    kept = g.filter_edges(col("weight") >= 5.0)
    assert "weight" in kept.edges.column_names
    assert "score" in kept.vertices.column_names


def test_directed_transform_result_still_has_directed_methods() -> None:
    g = _build(DirectedGraph)
    assert isinstance(g, DirectedGraph)
    kept = g.filter_edges(col("weight") >= 5.0)
    # would raise AttributeError if the rebuild downcast to the base class
    assert kept.out_degrees().count_rows() >= 1
    assert isinstance(kept.reverse(), DirectedGraph)


def test_undirected_transform_result_still_has_undirected_methods() -> None:
    g = _build(UndirectedGraph)
    assert isinstance(g, UndirectedGraph)
    kept = g.filter_edges(col("weight") >= 5.0)
    assert isinstance(kept.as_directed(), DirectedGraph)


@pytest.mark.parametrize("flavor", _FLAVORS)
def test_filter_vertices_drops_dangling_edges(flavor: type[Graph]) -> None:
    """Behavior check alongside the type check, so the port did not break semantics."""
    g = _build(flavor)
    kept = g.filter_vertices(col(ID) != 3)
    pairs = kept.edges.collect().to_pydict()
    assert set(zip(pairs[SRC], pairs[DST])) == {(1, 2)}


@pytest.mark.parametrize("flavor", _FLAVORS)
def test_drop_isolated_vertices_removes_unreferenced(flavor: type[Graph]) -> None:
    g = _build(flavor)
    kept = g.drop_isolated_vertices()
    assert set(kept.vertices.collect().to_pydict()[ID]) == {1, 2, 3}
