"""Tests for DirectedGraph triplets and subgraph filters."""

from __future__ import annotations

import daft
from daft import col

from daft_graph.graph import DirectedGraph
from daft_graph.schema import DST, ID, SRC


def _ids(df: daft.DataFrame, column: str = ID) -> set:
    return set(df.collect().to_pydict()[column])


def _edge_set(df: daft.DataFrame) -> set:
    d = df.select(SRC, DST).collect().to_pydict()
    return set(zip(d[SRC], d[DST]))


def test_triplets_carry_endpoint_attributes() -> None:
    vertices = daft.from_pydict({ID: [1, 2, 3], "name": ["a", "b", "c"]})
    edges = daft.from_pydict({SRC: [1, 2], DST: [2, 3]})
    tr = DirectedGraph(vertices=vertices, edges=edges).triplets().collect().to_pydict()
    assert "src_name" in tr and "dst_name" in tr
    rows = set(zip(tr[SRC], tr[DST], tr["src_name"], tr["dst_name"]))
    assert rows == {(1, 2, "a", "b"), (2, 3, "b", "c")}


def test_filter_vertices_drops_incident_edges() -> None:
    vertices = daft.from_pydict({ID: [1, 2, 3, 4]})
    edges = daft.from_pydict({SRC: [1, 3], DST: [2, 4]})
    g = DirectedGraph(vertices=vertices, edges=edges).filter_vertices(col(ID) != 2)
    assert _ids(g.vertices) == {1, 3, 4}
    assert _edge_set(g.edges) == {(3, 4)}


def test_filter_edges_keeps_all_vertices() -> None:
    vertices = daft.from_pydict({ID: [1, 2, 3, 4]})
    edges = daft.from_pydict({SRC: [1, 2, 3], DST: [2, 3, 4], "weight": [5, 1, 5]})
    g = DirectedGraph(vertices=vertices, edges=edges).filter_edges(col("weight") >= 5)
    assert _ids(g.vertices) == {1, 2, 3, 4}
    assert _edge_set(g.edges) == {(1, 2), (3, 4)}


def test_drop_isolated_vertices() -> None:
    vertices = daft.from_pydict({ID: [1, 2, 3, 4]})
    edges = daft.from_pydict({SRC: [1], DST: [2]})
    g = DirectedGraph(vertices=vertices, edges=edges).drop_isolated_vertices()
    assert _ids(g.vertices) == {1, 2}
    assert _edge_set(g.edges) == {(1, 2)}
