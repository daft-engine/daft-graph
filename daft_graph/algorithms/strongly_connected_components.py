"""Strongly connected components on Daft DataFrames.

Two vertices share a strongly connected component (SCC) when each can reach the
other following edge directions. Two strategies mirror connected_components:
``local`` collects the edges and uses scipy.sparse.csgraph; ``distributed`` runs
the coloring algorithm on the message passing primitive (forward max id color
propagation, then backward confirmation within each color). ``auto`` picks local
at or below an edge threshold. Each vertex is labelled by the smallest id in its
SCC, regardless of strategy.
"""

from __future__ import annotations

import os
import warnings

from daft import DataFrame, Expression, col, lit
from daft.functions import when

from daft_graph._optional import has_local_extra
from daft_graph.algorithms.connected_components import (
    _local_scipy_components,
    _resolve_strategy,
)
from daft_graph.graph import DirectedGraph
from daft_graph.iterate import bound_partitions, collect_bounded
from daft_graph.message_passing import MSG as _MSG
from daft_graph.message_passing import VALUE, pregel
from daft_graph.schema import COMPONENT, DST, ID, SRC, Strategy

_DEFAULT_LOCAL_THRESHOLD = 2_000_000
_COLOR = "color"
_CSRC = "_csrc"
_CDST = "_cdst"
_MIN = "_min"


def _max_update() -> Expression:
    """Expression setting value to max(value, msg), treating a null msg as value."""
    return when(col(_MSG).is_null(), col(VALUE)).otherwise(
        when(col(VALUE) >= col(_MSG), col(VALUE)).otherwise(col(_MSG))
    )


def _local_scc(graph: DirectedGraph) -> DataFrame:
    """SCCs on a single node via scipy.sparse.csgraph, labelled by min id."""
    return _local_scipy_components(graph, directed=True, connection="strong")


def _distributed_scc(
    graph: DirectedGraph,
    *,
    max_iters: int,
    materialize_every: int,
    checkpoint_dir: str | None,
) -> DataFrame:
    """SCCs by the coloring algorithm on the message passing primitive.

    Each round assigns every vertex the max id that can reach it (its color),
    restricts to same color edges, then confirms the vertices that can reach
    their color's root. Confirmed vertices form an SCC and are removed; the loop
    repeats on the rest. The final labels are remapped to the minimum id per SCC.
    """
    active_v = graph.vertices.select(ID).distinct().collect()
    active_e = collect_bounded(graph.edges.select(SRC, DST).distinct())
    parts: list[DataFrame] = []
    for outer in range(active_v.count_rows() + 1):
        if active_v.count_rows() == 0:
            break
        fwd_ckpt = os.path.join(checkpoint_dir, f"scc_{outer}_forward") if checkpoint_dir else None
        bwd_ckpt = os.path.join(checkpoint_dir, f"scc_{outer}_backward") if checkpoint_dir else None
        colors = (
            pregel(
                active_e,
                active_v.with_column(VALUE, col(ID)),
                to_dst=col("src_value"),
                agg=lambda m: m.max(),
                update=_max_update(),
                max_iters=max_iters,
                materialize_every=materialize_every,
                checkpoint_dir=fwd_ckpt,
            )
            .select(col(ID), col(VALUE).alias(_COLOR))
            .collect()
        )
        same_color_edges = (
            active_e.join(
                colors.select(col(ID).alias(SRC), col(_COLOR).alias(_CSRC)),
                on=SRC,
                how="inner",
            )
            .join(
                colors.select(col(ID).alias(DST), col(_COLOR).alias(_CDST)),
                on=DST,
                how="inner",
            )
            .where(col(_CSRC) == col(_CDST))
            .select(SRC, DST)
            .collect()
        )
        bwd_init = colors.select(
            col(ID),
            when(col(ID) == col(_COLOR), lit(1)).otherwise(lit(0)).alias(VALUE),
            col(_COLOR),
        )
        flags = pregel(
            same_color_edges,
            bwd_init,
            to_src=col("dst_value"),
            agg=lambda m: m.max(),
            update=_max_update(),
            max_iters=max_iters,
            materialize_every=materialize_every,
            checkpoint_dir=bwd_ckpt,
        )
        confirmed = flags.where(col(VALUE) == lit(1)).select(col(ID), col(_COLOR).alias(COMPONENT)).collect()
        parts.append(confirmed)
        # Bound the carried state each outer round: active_v/active_e are joined
        # and fed to the inner pregels every round, and a plain collect keeps the
        # growing partition count from the anti/semi joins, which then compounds
        # through the pregel shuffles. Cap is a plan rewrite, no extra execution.
        active_v = collect_bounded(active_v.join(confirmed.select(ID), on=ID, how="anti"))
        active_e = collect_bounded(
            active_e.join(active_v.select(col(ID).alias(SRC)), on=SRC, how="semi").join(
                active_v.select(col(ID).alias(DST)), on=DST, how="semi"
            )
        )
    if active_v.count_rows() > 0:
        warnings.warn(
            "strongly_connected_components did not fully partition the graph within the iteration bound",
            stacklevel=2,
        )
    if not parts:
        return graph.vertices.select(col(ID), col(ID).alias(COMPONENT)).distinct()
    out = parts[0]
    for part in parts[1:]:
        out = out.union_all(part)
    # The fold sums each part's partition count; cap before the groupby shuffle.
    out = bound_partitions(out)
    # The coloring uses max id roots; remap to the min id in each component.
    min_label = out.groupby(COMPONENT).agg(col(ID).min().alias(_MIN))
    return out.join(min_label, on=COMPONENT, how="inner").select(col(ID), col(_MIN).alias(COMPONENT))


def strongly_connected_components(
    graph: DirectedGraph,
    *,
    strategy: Strategy = "auto",
    max_iters: int = 100,
    materialize_every: int = 1,
    checkpoint_dir: str | None = None,
    local_threshold: int = _DEFAULT_LOCAL_THRESHOLD,
) -> DataFrame:
    """Compute strongly connected components as columns ``id`` and ``component``.

    Every vertex carries the smallest id in its SCC; isolated vertices and those
    in no cycle form singleton components.

    Args:
        graph: The directed graph to analyze.
        strategy: ``auto`` (default), ``distributed``, or ``local``.
        max_iters: Maximum rounds per propagation phase (distributed only).
        materialize_every: How often to truncate the Daft plan between rounds.
        checkpoint_dir: Optional parquet checkpoint directory for long runs.
        local_threshold: Edge count at or below which ``auto`` uses the local solve.

    Returns:
        A DataFrame with one row per vertex: ``id`` and its ``component`` label.
    """
    if graph.edges.count_rows() == 0:
        return graph.vertices.select(col(ID), col(ID).alias(COMPONENT)).distinct()
    # `auto` can only pick local when the optional extra is installed, so skip
    # the extra edge count on a core only install, matching connected_components.
    if strategy == "auto" and not has_local_extra():
        resolved = "distributed"
    else:
        num_edges = graph.num_edges() if strategy == "auto" else 0
        resolved = _resolve_strategy(strategy, num_edges, local_threshold)
    if resolved == "local":
        return _local_scc(graph)
    return _distributed_scc(
        graph,
        max_iters=max_iters,
        materialize_every=materialize_every,
        checkpoint_dir=checkpoint_dir,
    )
