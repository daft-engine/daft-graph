"""Maximal independent set on Daft DataFrames.

Computes a maximal independent set deterministically: in each round every
undecided vertex that is a local minimum (a smaller id than all of its undecided
neighbors) joins the set, and its neighbors are excluded. This repeats until
every vertex is decided. Treats edges as undirected. The result is independent
(no two selected vertices are adjacent) and maximal (every unselected vertex has
a selected neighbor).
"""

from __future__ import annotations

import warnings

from daft import DataFrame, col, lit
from daft.functions import when

from daft_graph.edges import canonicalize, symmetrize
from daft_graph.graph import Graph
from daft_graph.iterate import collect_bounded
from daft_graph.schema import DST, ID, SRC

SELECTED = "selected"
_STATUS = "_status"
_NMIN = "_nmin"
_IN = "_in"
_EX = "_ex"
_UNDECIDED, _IN_SET, _EXCLUDED = 0, 1, 2


def maximal_independent_set(graph: Graph, *, max_iters: int = 1000) -> DataFrame:
    """Compute a maximal independent set, as columns ``id`` and ``selected``.

    Accepts either graph flavor; edges are treated as undirected either way.

    Args:
        graph: The graph to analyze.
        max_iters: Maximum peeling rounds. One round decides every current local
            minimum, so a graph whose ids increase along a long induced path needs
            one round per vertex on it. A warning is emitted if the cap is reached,
            because the result is then independent but not guaranteed maximal.

    Returns:
        A DataFrame with one row per vertex: ``id`` and whether it was
        ``selected`` into the set.
    """
    all_v = graph.vertices.select(col(ID)).distinct()
    if graph.edges.count_rows() == 0:
        return all_v.with_column(SELECTED, lit(True))

    undirected = collect_bounded(symmetrize(canonicalize(graph.edges)))
    status = all_v.with_column(_STATUS, lit(_UNDECIDED)).collect()

    converged = False
    for _ in range(max_iters):
        undecided = status.where(col(_STATUS) == lit(_UNDECIDED)).select(ID).collect()
        if undecided.count_rows() == 0:
            converged = True
            break
        adj = undirected.join(undecided.select(col(ID).alias(SRC)), on=SRC, how="semi").join(
            undecided.select(col(ID).alias(DST)), on=DST, how="semi"
        )
        nbr_min = adj.groupby(SRC).agg(col(DST).min().alias(_NMIN)).select(col(SRC).alias(ID), col(_NMIN))
        joiners = (
            undecided.join(nbr_min, on=ID, how="left")
            .where(col(_NMIN).is_null() | (col(ID) < col(_NMIN)))
            .select(ID)
            .collect()
        )
        excluded = (
            undirected.join(joiners.select(col(ID).alias(SRC)), on=SRC, how="semi")
            .select(col(DST).alias(ID))
            .distinct()
        )
        status = collect_bounded(
            status.join(joiners.select(col(ID), lit(1).alias(_IN)), on=ID, how="left")
            .join(excluded.select(col(ID), lit(1).alias(_EX)), on=ID, how="left")
            .with_column(
                _STATUS,
                when(col(_STATUS) != lit(_UNDECIDED), col(_STATUS)).otherwise(
                    when(~col(_IN).is_null(), lit(_IN_SET)).otherwise(
                        when(~col(_EX).is_null(), lit(_EXCLUDED)).otherwise(lit(_UNDECIDED))
                    )
                ),
            )
            .select(ID, _STATUS)
        )

    if not converged:
        warnings.warn(
            f"maximal_independent_set did not converge within {max_iters} rounds; "
            "undecided vertices are reported as not selected, so the result is "
            "independent but may not be maximal. Raise max_iters.",
            UserWarning,
            stacklevel=2,
        )
    return status.select(col(ID), (col(_STATUS) == lit(_IN_SET)).alias(SELECTED))
