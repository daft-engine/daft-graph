"""Tests for motif find()."""

from __future__ import annotations

import daft
import pytest
from daft import col

from daft_graph.graph import DirectedGraph
from daft_graph.schema import DST, ID, SRC


def _graph() -> DirectedGraph:
    vertices = daft.from_pydict({ID: [1, 2, 3], "name": ["x", "y", "z"]})
    # edges: 1->2, 2->3, 3->1, 2->1
    edges = daft.from_pydict({SRC: [1, 2, 3, 2], DST: [2, 3, 1, 1]})
    return DirectedGraph(vertices=vertices, edges=edges)


def _pairs(df: daft.DataFrame, a: str, b: str) -> set:
    d = df.select(col(a)["id"].alias("a"), col(b)["id"].alias("b")).collect().to_pydict()
    return set(zip(d["a"], d["b"]))


def test_single_edge_matches_all_edges() -> None:
    g = _graph()
    result = g.find("(a)-[e]->(b)")
    assert _pairs(result, "a", "b") == {(1, 2), (2, 3), (3, 1), (2, 1)}
    # the edge struct exposes src and dst
    de = result.select(col("e")["src"].alias("s"), col("e")["dst"].alias("t")).collect().to_pydict()
    assert set(zip(de["s"], de["t"])) == {(1, 2), (2, 3), (3, 1), (2, 1)}


def test_directed_triangle() -> None:
    g = _graph()
    result = g.find("(a)-[]->(b); (b)-[]->(c); (c)-[]->(a)")
    d = (
        result.select(
            col("a")["id"].alias("a"),
            col("b")["id"].alias("b"),
            col("c")["id"].alias("c"),
        )
        .collect()
        .to_pydict()
    )
    assert set(zip(d["a"], d["b"], d["c"])) == {(1, 2, 3), (2, 3, 1), (3, 1, 2)}


def test_negation_excludes_reciprocal() -> None:
    g = _graph()
    result = g.find("(a)-[]->(b); !(b)-[]->(a)")
    assert _pairs(result, "a", "b") == {(2, 3), (3, 1)}


def test_repeated_name_identity_mutual_edges() -> None:
    g = _graph()
    result = g.find("(a)-[]->(b); (b)-[]->(a)")
    assert _pairs(result, "a", "b") == {(1, 2), (2, 1)}


def test_vertex_struct_carries_attributes() -> None:
    g = _graph()
    result = g.find("(a)-[e]->(b)")
    d = result.select(col("a")["id"].alias("aid"), col("a")["name"].alias("an")).collect().to_pydict()
    assert dict(zip(d["aid"], d["an"])) == {1: "x", 2: "y", 3: "z"}


def test_no_named_elements_raises() -> None:
    g = _graph()
    with pytest.raises(ValueError):
        g.find("()-[]->()")


def test_self_loop_pattern_raises() -> None:
    g = _graph()
    with pytest.raises(ValueError):
        g.find("(a)-[e]->(a)")


def test_reused_edge_name_raises() -> None:
    g = _graph()
    with pytest.raises(ValueError):
        g.find("(a)-[e]->(b); (b)-[e]->(c)")
