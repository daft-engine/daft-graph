"""Tests for daft_graph.edges."""

from __future__ import annotations

import daft
import pytest

from daft_graph.edges import (
    canonicalize,
    dedupe_edges,
    drop_self_loops,
    symmetrize,
    to_edges,
    validate_edges,
)
from daft_graph.schema import DST, SRC


def _edge_set(df: daft.DataFrame) -> set:
    d = df.select(SRC, DST).collect().to_pydict()
    return set(zip(d[SRC], d[DST]))


def test_to_edges_renames_columns() -> None:
    df = daft.from_pydict({"a": [1, 2], "b": [3, 4]})
    edges = to_edges(df, src="a", dst="b")
    assert set(edges.column_names) == {SRC, DST}
    assert _edge_set(edges) == {(1, 3), (2, 4)}


def test_validate_edges_passes() -> None:
    edges = daft.from_pydict({SRC: [1], DST: [2]})
    assert validate_edges(edges) is edges


def test_validate_edges_raises_on_missing_columns() -> None:
    bad = daft.from_pydict({"a": [1], "b": [2]})
    with pytest.raises(ValueError):
        validate_edges(bad)


def test_drop_self_loops() -> None:
    edges = daft.from_pydict({SRC: [1, 1], DST: [1, 2]})
    assert _edge_set(drop_self_loops(edges)) == {(1, 2)}


def test_dedupe_edges() -> None:
    edges = daft.from_pydict({SRC: [1, 1, 2], DST: [2, 2, 3]})
    assert _edge_set(dedupe_edges(edges)) == {(1, 2), (2, 3)}


def test_canonicalize_collapses_reverse_and_self_loops() -> None:
    edges = daft.from_pydict({SRC: [2, 1, 3, 5], DST: [1, 2, 3, 4]})
    assert _edge_set(canonicalize(edges)) == {(1, 2), (4, 5)}


def test_symmetrize_adds_reverse_edges() -> None:
    edges = daft.from_pydict({SRC: [1, 3], DST: [2, 4]})
    assert _edge_set(symmetrize(edges)) == {(1, 2), (2, 1), (3, 4), (4, 3)}
