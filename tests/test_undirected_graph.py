"""Tests for daft_graph.graph.UndirectedGraph."""

from __future__ import annotations

import daft

from daft_graph.graph import DirectedGraph, UndirectedGraph
from daft_graph.schema import DST, ID, SRC


def _edges() -> daft.DataFrame:
    # undirected: 1-2, 1-3, 2-3, 4-1
    return daft.from_pydict({SRC: [1, 1, 2, 4], DST: [2, 3, 3, 1]})


def _deg_map(df: daft.DataFrame) -> dict:
    d = df.collect().to_pydict()
    return dict(zip(d[ID], d["degree"]))


def test_derives_vertices_when_none_given() -> None:
    g = UndirectedGraph(_edges())
    assert g.num_vertices() == 4
    assert set(g.vertices.collect().to_pydict()[ID]) == {1, 2, 3, 4}


def test_edges_are_stored_one_row_per_edge() -> None:
    assert UndirectedGraph(_edges()).num_edges() == 4


def test_single_edge_gives_each_endpoint_degree_one() -> None:
    g = UndirectedGraph(daft.from_pydict({SRC: [1], DST: [2]}))
    assert _deg_map(g.degrees()) == {1: 1, 2: 1}


def test_degrees_count_each_edge_once() -> None:
    assert _deg_map(UndirectedGraph(_edges()).degrees()) == {1: 3, 2: 2, 3: 2, 4: 1}


def test_self_loop_contributes_two() -> None:
    g = UndirectedGraph(daft.from_pydict({SRC: [1], DST: [1]}))
    assert _deg_map(g.degrees()) == {1: 2}


def test_traversal_edges_are_symmetrized() -> None:
    g = UndirectedGraph(daft.from_pydict({SRC: [1], DST: [2]}))
    t = g._traversal_edges().collect().to_pydict()
    assert sorted(zip(t[SRC], t[DST])) == [(1, 2), (2, 1)]


def test_traversal_reaches_both_endpoints() -> None:
    g = UndirectedGraph(daft.from_pydict({SRC: [1, 2], DST: [2, 3]}))
    t = g._traversal_edges().collect().to_pydict()
    pairs = set(zip(t[SRC], t[DST]))
    # every stored edge is walkable in both directions
    assert {(1, 2), (2, 1), (2, 3), (3, 2)} == pairs


def test_stored_edges_are_not_duplicated_by_traversal() -> None:
    g = UndirectedGraph(_edges())
    assert g.edges.count_rows() == 4
    assert g._traversal_edges().count_rows() == 8


def test_as_directed_returns_directed_graph() -> None:
    g = UndirectedGraph(_edges())
    d = g.as_directed()
    assert isinstance(d, DirectedGraph)
    assert d.num_edges() == g.num_edges()


def test_as_directed_can_swap_orientation() -> None:
    g = UndirectedGraph(daft.from_pydict({SRC: [1], DST: [2]}))
    d = g.as_directed(src_col=DST, dst_col=SRC)
    e = d.edges.collect().to_pydict()
    assert list(zip(e[SRC], e[DST])) == [(2, 1)]


def test_round_trip_through_directed_preserves_edges() -> None:
    g = UndirectedGraph(_edges())
    back = g.as_directed().as_undirected()
    assert isinstance(back, UndirectedGraph)
    assert back.num_edges() == g.num_edges()


def test_custom_columns_normalize_to_canonical() -> None:
    df = daft.from_pydict({"a": [1, 2], "b": [3, 4]})
    g = UndirectedGraph(df, src_col="a", dst_col="b")
    assert set(g.edges.column_names) == {SRC, DST}
