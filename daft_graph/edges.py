"""Edge list utilities for daft-graph.

Edges are Daft DataFrames with two integer columns, ``src`` and ``dst``. These
helpers normalize, deduplicate, and reshape edge lists. They return lazy
DataFrames; materialization is the caller's responsibility (see
:mod:`daft_graph.iterate`).
"""

from __future__ import annotations

from daft import DataFrame, col
from daft.functions import when

from daft_graph.schema import DST, SRC


def to_edges(df: DataFrame, src: str, dst: str) -> DataFrame:
    """Project two columns of ``df`` into a canonical ``src``/``dst`` edge list.

    Args:
        df: Any DataFrame holding a source and destination column.
        src: Name of the column to use as ``src``.
        dst: Name of the column to use as ``dst``.

    Returns:
        A DataFrame with exactly the ``src`` and ``dst`` columns.
    """
    return df.select(col(src).alias(SRC), col(dst).alias(DST))


def validate_edges(edges: DataFrame) -> DataFrame:
    """Return ``edges`` unchanged, or raise if it lacks ``src``/``dst`` columns.

    Args:
        edges: The edge DataFrame to validate.

    Returns:
        The same DataFrame, for convenient chaining.

    Raises:
        ValueError: If either the ``src`` or ``dst`` column is missing.
    """
    columns = set(edges.column_names)
    missing = {SRC, DST} - columns
    if missing:
        raise ValueError(f"edges must have columns {SRC!r} and {DST!r}; missing {sorted(missing)}")
    return edges


def drop_self_loops(edges: DataFrame) -> DataFrame:
    """Drop edges whose endpoints are equal."""
    return edges.where(col(SRC) != col(DST))


def dedupe_edges(edges: DataFrame) -> DataFrame:
    """Remove duplicate ``(src, dst)`` rows."""
    return edges.select(SRC, DST).distinct()


def canonicalize(edges: DataFrame) -> DataFrame:
    """Orient every edge so ``src <= dst``, drop self loops, and deduplicate.

    Produces a stable undirected representation: ``(a, b)`` and ``(b, a)``
    collapse to a single row.
    """
    oriented = edges.select(
        when(col(SRC) <= col(DST), col(SRC)).otherwise(col(DST)).alias(SRC),
        when(col(SRC) <= col(DST), col(DST)).otherwise(col(SRC)).alias(DST),
    )
    return oriented.where(col(SRC) != col(DST)).distinct()


def symmetrize(edges: DataFrame) -> DataFrame:
    """Add the reverse of every edge, yielding an undirected adjacency list.

    Non ``src``/``dst`` columns (edge attributes) are preserved and carried onto
    the reversed rows unchanged, so the result keeps the input's schema.
    """
    attrs = [c for c in edges.column_names if c not in (SRC, DST)]
    forward = edges.select(SRC, DST, *attrs)
    backward = edges.select(col(DST).alias(SRC), col(SRC).alias(DST), *[col(c) for c in attrs])
    return forward.union_all(backward)
