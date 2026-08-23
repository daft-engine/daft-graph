"""Shortest path hop distances to landmark vertices.

For each vertex, computes the unweighted distance to each landmark by running
min distance relaxation on the message passing primitive: every vertex
repeatedly takes one plus the minimum distance of its out neighbors, seeded with
zero at the landmark. This is the analog of GraphFrames shortestPaths and
demonstrates the pregel API.
"""

from __future__ import annotations

import daft
from daft import DataFrame, col, lit
from daft.functions import when

from daft_graph.graph import Graph
from daft_graph.iterate import bound_partitions, collect_bounded
from daft_graph.message_passing import MSG, VALUE, pregel
from daft_graph.schema import DST, ID, SRC

LANDMARK = "landmark"
DISTANCE = "distance"
_UNREACHABLE = 1 << 30


def shortest_paths(
    graph: Graph,
    landmarks: list[int],
    *,
    max_iters: int = 100,
    materialize_every: int = 1,
    checkpoint_dir: str | None = None,
) -> DataFrame:
    """Hop distance from each vertex to each landmark.

    Args:
        graph: The graph to analyze.
        landmarks: Vertex ids to measure distance to.
        max_iters: Maximum relaxation rounds; bounds the largest distance found.
        materialize_every: How often to truncate the Daft plan between rounds.
        checkpoint_dir: Optional parquet checkpoint directory for long runs.

    Returns:
        A DataFrame ``[id, landmark, distance]`` with one row per vertex that can
        reach a landmark. Unreachable pairs are omitted.

    Raises:
        ValueError: If ``landmarks`` is empty.
    """
    if not landmarks:
        raise ValueError("landmarks must not be empty")
    if graph.edges.count_rows() == 0:
        vertex_ids = set(graph.vertices.select(ID).distinct().collect().to_pydict()[ID])
        valid = [lm for lm in landmarks if lm in vertex_ids]
        return daft.from_pydict({ID: valid, LANDMARK: valid, DISTANCE: [0] * len(valid)})
    # Materialized once: reused by every landmark's pregel run and every round
    # within it, so leaving it lazy replans the orient landmarks x max_iters times.
    edges = collect_bounded(graph.orient(graph.edges.select(SRC, DST)))
    vertices = graph.vertices.select(ID).distinct().collect()
    update = when(col(MSG).is_null(), col(VALUE)).otherwise(
        when(col(VALUE) <= col(MSG), col(VALUE)).otherwise(col(MSG))
    )
    results: list[DataFrame] = []
    for landmark in landmarks:
        init = vertices.with_column(VALUE, when(col(ID) == lit(landmark), lit(0)).otherwise(lit(_UNREACHABLE)))
        final = pregel(
            edges,
            init,
            to_src=col("dst_value") + lit(1),
            agg=lambda m: m.min(),
            update=update,
            max_iters=max_iters,
            materialize_every=materialize_every,
            checkpoint_dir=checkpoint_dir,
        )
        results.append(
            final.where(col(VALUE) < lit(_UNREACHABLE))
            .with_column(LANDMARK, lit(landmark))
            .select(ID, LANDMARK, col(VALUE).alias(DISTANCE))
        )
    out = results[0]
    for r in results[1:]:
        out = out.union_all(r)
    # The fold sums each landmark's partition count; cap the result. Plan rewrite.
    return bound_partitions(out)
