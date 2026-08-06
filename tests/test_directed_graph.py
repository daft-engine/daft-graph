"""Tests for daft_graph.graph.DirectedGraph."""

from __future__ import annotations

import daft
import pytest

from daft_graph.graph import DirectedGraph, UndirectedGraph
from daft_graph.schema import DST, ID, SRC


def _edges() -> daft.DataFrame:
    # 1->2, 1->3, 2->3, 4->1
    return daft.from_pydict({SRC: [1, 1, 2, 4], DST: [2, 3, 3, 1]})


def _deg_map(df: daft.DataFrame) -> dict:
    d = df.collect().to_pydict()
    return dict(zip(d[ID], d["degree"]))


def test_derives_vertices_when_none_given() -> None:
    g = DirectedGraph(_edges())
    assert g.num_vertices() == 4
    assert g.num_edges() == 4
    assert set(g.vertices.collect().to_pydict()[ID]) == {1, 2, 3, 4}


def test_accepts_explicit_vertices() -> None:
    vertices = daft.from_pydict({ID: [1, 2, 3, 4, 99]})
    g = DirectedGraph(_edges(), vertices)
    assert g.num_vertices() == 5


def test_custom_columns_normalize_to_canonical() -> None:
    df = daft.from_pydict({"a": [1, 2], "b": [3, 4]})
    g = DirectedGraph(df, src_col="a", dst_col="b")
    assert set(g.edges.column_names) == {SRC, DST}
    assert g.num_edges() == 2


def test_custom_id_column_normalizes() -> None:
    edges = daft.from_pydict({SRC: [1], DST: [2]})
    vertices = daft.from_pydict({"node": [1, 2]})
    g = DirectedGraph(edges, vertices, id_col="node")
    assert ID in g.vertices.column_names
    assert g.num_vertices() == 2


def test_edge_attributes_are_preserved_through_rename() -> None:
    df = daft.from_pydict({"a": [1, 2], "b": [3, 4], "weight": [0.5, 1.5]})
    g = DirectedGraph(df, src_col="a", dst_col="b")
    assert set(g.edges.column_names) == {SRC, DST, "weight"}


def test_out_degrees() -> None:
    assert _deg_map(DirectedGraph(_edges()).out_degrees()) == {1: 2, 2: 1, 4: 1}


def test_in_degrees() -> None:
    assert _deg_map(DirectedGraph(_edges()).in_degrees()) == {2: 1, 3: 2, 1: 1}


def test_degrees_is_in_plus_out() -> None:
    assert _deg_map(DirectedGraph(_edges()).degrees()) == {1: 3, 2: 2, 3: 2, 4: 1}


def test_traversal_edges_are_unchanged() -> None:
    g = DirectedGraph(_edges())
    assert g._traversal_edges().count_rows() == g.num_edges()


def test_reverse_flips_every_edge() -> None:
    g = DirectedGraph(_edges())
    r = g.reverse()
    assert isinstance(r, DirectedGraph)
    original = set(zip(*_edges().collect().to_pydict().values()))
    flipped = r.edges.collect().to_pydict()
    assert set(zip(flipped[DST], flipped[SRC])) == original


def test_reverse_is_an_involution() -> None:
    g = DirectedGraph(_edges())
    back = g.reverse().reverse().edges.collect().to_pydict()
    start = g.edges.collect().to_pydict()
    assert sorted(zip(back[SRC], back[DST])) == sorted(zip(start[SRC], start[DST]))


def test_reverse_preserves_edge_attributes() -> None:
    edges = daft.from_pydict({SRC: [1], DST: [2], "weight": [7.0]})
    r = DirectedGraph(edges).reverse()
    assert set(r.edges.column_names) == {SRC, DST, "weight"}
    assert r.edges.collect().to_pydict()["weight"] == [7.0]


def test_as_undirected_returns_undirected_graph() -> None:
    g = DirectedGraph(_edges())
    u = g.as_undirected()
    assert isinstance(u, UndirectedGraph)
    assert u.num_vertices() == g.num_vertices()


def test_requires_id_column_on_vertices() -> None:
    vertices = daft.from_pydict({"node": [1, 2]})
    edges = daft.from_pydict({SRC: [1], DST: [2]})
    with pytest.raises(ValueError):
        DirectedGraph(edges, vertices)


def test_requires_edge_columns() -> None:
    bad_edges = daft.from_pydict({"a": [1], "b": [2]})
    with pytest.raises(ValueError):
        DirectedGraph(bad_edges)


def test_validate_rejects_duplicate_vertex_ids() -> None:
    vertices = daft.from_pydict({ID: [1, 1, 2]})
    edges = daft.from_pydict({SRC: [1], DST: [2]})
    with pytest.raises(ValueError, match="duplicate"):
        DirectedGraph(edges, vertices, validate=True)


def test_validate_rejects_dangling_endpoints() -> None:
    vertices = daft.from_pydict({ID: [1]})
    edges = daft.from_pydict({SRC: [1], DST: [2]})
    with pytest.raises(ValueError, match="not in the vertex set"):
        DirectedGraph(edges, vertices, validate=True)


def test_validate_accepts_a_consistent_graph() -> None:
    vertices = daft.from_pydict({ID: [1, 2]})
    edges = daft.from_pydict({SRC: [1], DST: [2]})
    assert DirectedGraph(edges, vertices, validate=True).num_edges() == 1


def test_validation_is_off_by_default() -> None:
    vertices = daft.from_pydict({ID: [1]})
    edges = daft.from_pydict({SRC: [1], DST: [2]})
    assert DirectedGraph(edges, vertices).num_edges() == 1
