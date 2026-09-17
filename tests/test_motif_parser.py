"""Tests for the motif pattern parser."""

from __future__ import annotations

import pytest

from daft_graph.motif import EdgePattern, VertexPattern, is_named, parse_motif


def test_single_edge() -> None:
    assert parse_motif("(a)-[e]->(b)") == [EdgePattern(src="a", dst="b", edge="e", negated=False)]


def test_chain() -> None:
    assert parse_motif("(a)-[e]->(b); (b)-[e2]->(c)") == [
        EdgePattern(src="a", dst="b", edge="e"),
        EdgePattern(src="b", dst="c", edge="e2"),
    ]


def test_anonymous_edge_and_vertex() -> None:
    clauses = parse_motif("(a)-[]->()")
    assert len(clauses) == 1
    clause = clauses[0]
    assert isinstance(clause, EdgePattern)
    assert clause.src == "a"
    assert clause.edge is None
    assert not is_named(clause.dst)


def test_negation() -> None:
    clauses = parse_motif("(a)-[e]->(b); !(b)-[]->(a)")
    assert clauses[1] == EdgePattern(src="b", dst="a", edge=None, negated=True)


def test_negated_named_edge_raises() -> None:
    with pytest.raises(ValueError):
        parse_motif("!(a)-[e]->(b)")


def test_lone_vertex() -> None:
    assert parse_motif("(a)") == [VertexPattern(name="a")]


def test_unparseable_raises() -> None:
    with pytest.raises(ValueError):
        parse_motif("not a motif")


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        parse_motif("   ")


def test_negated_vertex_raises() -> None:
    with pytest.raises(ValueError):
        parse_motif("!(a)")
