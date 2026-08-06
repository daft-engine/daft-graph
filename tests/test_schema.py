"""Tests for daft_graph.schema."""

from __future__ import annotations

from daft_graph import schema


def test_column_constants() -> None:
    assert schema.ID == "id"
    assert schema.SRC == "src"
    assert schema.DST == "dst"
    assert schema.COMPONENT == "component"
    assert schema.LABEL == "label"
    assert schema.RANK == "rank"


def test_internal_columns_are_distinct() -> None:
    cols = {schema.U, schema.V, schema.REP}
    assert len(cols) == 3
