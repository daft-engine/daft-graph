"""Triangle counting on Daft DataFrames.

Counts the number of triangles each vertex participates in, treating edges as
undirected. Triangles are found by joining canonical edges (src < dst) into
paths ``a-b-c`` with ``a < b < c`` and keeping those whose closing edge
``(a, c)`` exists. Each triangle has a unique median vertex ``b``, so it is
counted exactly once.
"""

from __future__ import annotations

from daft import DataFrame, col, lit
from daft.functions import when

from daft_graph.edges import canonicalize
from daft_graph.graph import Graph
from daft_graph.schema import DST, ID, SRC

TRIANGLE_COUNT = "triangle_count"
_A = "a"
_B = "b"
_C = "c"
_ONE = "_one"


def triangle_count(graph: Graph) -> DataFrame:
    """Count triangles per vertex (undirected).

    Accepts either graph flavor; edges are treated as undirected either way.

    Returns a DataFrame with one row per vertex: ``id`` and ``triangle_count``.
    """
    all_v = graph.vertices.select(ID).distinct()
    if graph.edges.count_rows() == 0:
        return all_v.with_column(TRIANGLE_COUNT, lit(0)).select(ID, TRIANGLE_COUNT)
    e = canonicalize(graph.edges)
    paths = e.select(col(SRC).alias(_A), col(DST).alias(_B)).join(
        e.select(col(SRC).alias(_B), col(DST).alias(_C)), on=_B, how="inner"
    )
    triangles = paths.join(e.select(col(SRC).alias(_A), col(DST).alias(_C)), on=[_A, _C], how="semi")
    members = (
        triangles.select(col(_A).alias(ID))
        .union_all(triangles.select(col(_B).alias(ID)))
        .union_all(triangles.select(col(_C).alias(ID)))
        .with_column(_ONE, lit(1))
    )
    counts = members.groupby(ID).agg(col(_ONE).sum().alias(TRIANGLE_COUNT))
    return (
        all_v.join(counts, on=ID, how="left")
        .with_column(
            TRIANGLE_COUNT,
            when(col(TRIANGLE_COUNT).is_null(), lit(0)).otherwise(col(TRIANGLE_COUNT)),
        )
        .select(ID, TRIANGLE_COUNT)
    )
