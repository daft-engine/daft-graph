"""PageRank on Daft DataFrames.

Power iteration that matches networkx semantics: a personalization vector ``p``
drives both teleport and dangling redistribution, and ranks sum to one. With no
``source_ids`` the vector is uniform (standard PageRank); with ``source_ids`` it
concentrates on the seeds (personalized PageRank). The per round incoming
contribution is computed with the message passing primitive; the global dangling
mass and teleport are applied around it.
"""

from __future__ import annotations

import os

import daft
from daft import DataFrame, col, lit
from daft.functions import when

from daft_graph.graph import DirectedGraph
from daft_graph.iterate import iterate_to_fixed_point
from daft_graph.message_passing import MSG, aggregate_messages
from daft_graph.schema import DST, ID, RANK, SRC

_OD = "outdeg"
_P = "_p"
_DIFF = "_diff"
SOURCE = "source"


def _scalar_sum(df: DataFrame, column: str) -> float:
    """Return the global sum of ``column``, treating empty input as 0.0."""
    rows = df.agg(col(column).sum().alias("_s")).collect().to_pydict()["_s"]
    return float(rows[0]) if rows and rows[0] is not None else 0.0


def _personalization(vertices: DataFrame, n: int, source_ids: list[int] | None) -> DataFrame:
    """Build the personalization vector as columns ``id`` and ``_p`` summing to one.

    ``vertices`` is expected to be already collected; only its ``id`` column is read.
    """
    if source_ids is None:
        return vertices.with_column(_P, lit(1.0 / n)).select(ID, _P).collect()
    if len(source_ids) == 0:
        raise ValueError("source_ids must not be empty")
    vertex_id_set = set(vertices.to_pydict()[ID])
    valid = sorted({int(s) for s in source_ids} & vertex_id_set)
    if not valid:
        raise ValueError("source_ids must contain at least one vertex in the graph")
    seeds = daft.from_pydict({ID: valid}).with_column(_P, lit(1.0 / len(valid)))
    return (
        vertices.join(seeds, on=ID, how="left")
        .with_column(_P, when(col(_P).is_null(), lit(0.0)).otherwise(col(_P)))
        .select(ID, _P)
        .collect()
    )


def pagerank(
    graph: DirectedGraph,
    *,
    damping: float = 0.85,
    max_iters: int = 100,
    tol: float = 1e-6,
    source_ids: list[int] | None = None,
    materialize_every: int = 1,
    checkpoint_dir: str | None = None,
) -> DataFrame:
    """Compute PageRank, returning columns ``id`` and ``rank`` that sum to one.

    Args:
        graph: The directed graph to rank.
        damping: Damping factor (networkx ``alpha``), typically 0.85.
        max_iters: Maximum power iterations.
        tol: Convergence threshold; iteration stops when the total L1 change
            across all vertices falls below ``n * tol`` (same tol semantics as
            networkx).
        source_ids: If given, run personalized PageRank seeded uniformly on these
            vertices instead of standard uniform PageRank.
        materialize_every: How often to truncate the Daft plan between rounds.
        checkpoint_dir: Optional parquet checkpoint directory for long runs.

    Returns:
        A DataFrame with one row per vertex: ``id`` and its ``rank``.
    """
    vertices = graph.vertices.select(ID).distinct().collect()
    n = vertices.count_rows()
    if n == 0:
        return vertices.with_column(RANK, lit(0.0))

    pvec = _personalization(vertices, n, source_ids)
    edges = graph.edges.select(SRC, DST).distinct().collect()
    if edges.count_rows() == 0:
        return pvec.select(col(ID), col(_P).alias(RANK))

    outdeg = (edges.groupby(SRC).agg(col(DST).count().alias(_OD)).select(col(SRC).alias(ID), col(_OD))).collect()
    dangling_ids = vertices.join(outdeg.select(col(ID)), on=ID, how="anti").collect()
    init = vertices.with_column(RANK, lit(1.0 / n))

    def step(ranks: DataFrame) -> DataFrame:
        state = ranks.join(outdeg, on=ID, how="left")
        incoming = aggregate_messages(edges, state, to_dst=col(f"src_{RANK}") / col(f"src_{_OD}"))
        dangling_sum = _scalar_sum(ranks.join(dangling_ids, on=ID, how="inner"), RANK)
        pcoef = damping * dangling_sum + (1.0 - damping)
        return (
            vertices.join(incoming, on=ID, how="left")
            .join(pvec, on=ID, how="inner")
            .with_column(
                RANK,
                lit(damping) * when(col(MSG).is_null(), lit(0.0)).otherwise(col(MSG)) + lit(pcoef) * col(_P),
            )
            .select(ID, RANK)
        )

    def converged(prev: DataFrame, nxt: DataFrame) -> bool:
        diff = (
            prev.select(col(ID), col(RANK).alias("_prev"))
            .join(nxt.select(col(ID), col(RANK).alias("_next")), on=ID, how="inner")
            .with_column(_DIFF, (col("_prev") - col("_next")).abs())
        )
        return _scalar_sum(diff, _DIFF) < n * tol

    final, _ = iterate_to_fixed_point(
        init,
        step,
        converged,
        max_iters=max_iters,
        materialize_every=materialize_every,
        checkpoint_dir=checkpoint_dir,
    )
    return final


def parallel_personalized_pagerank(
    graph: DirectedGraph,
    source_ids: list[int],
    *,
    damping: float = 0.85,
    max_iters: int = 100,
    tol: float = 1e-6,
    materialize_every: int = 1,
    checkpoint_dir: str | None = None,
) -> DataFrame:
    """Personalized PageRank computed separately for each source vertex.

    This is the GraphFrames ``parallelPersonalizedPageRank``: rather than one run
    seeded on the whole set, it produces an independent personalized vector per
    source.

    Args:
        graph: The directed graph to rank.
        source_ids: The source vertices; one personalized vector is produced each.
        damping: Damping factor.
        max_iters: Maximum power iterations per source.
        tol: Convergence threshold per source.
        materialize_every: How often to truncate the Daft plan between rounds.
        checkpoint_dir: Optional parquet checkpoint directory.

    Returns:
        A DataFrame ``[id, source, rank]``: ``rank`` is the PageRank of vertex
        ``id`` in the run personalized to ``source``.

    Raises:
        ValueError: If ``source_ids`` is empty.
    """
    if not source_ids:
        raise ValueError("source_ids must not be empty")
    results: list[DataFrame] = []
    for source in source_ids:
        source_ckpt = os.path.join(checkpoint_dir, str(source)) if checkpoint_dir else None
        ranks = pagerank(
            graph,
            damping=damping,
            max_iters=max_iters,
            tol=tol,
            source_ids=[source],
            materialize_every=materialize_every,
            checkpoint_dir=source_ckpt,
        )
        results.append(ranks.with_column(SOURCE, lit(source)).select(ID, SOURCE, RANK))
    out = results[0]
    for part in results[1:]:
        out = out.union_all(part)
    return out
