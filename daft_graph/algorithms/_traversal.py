"""Shared frontier traversal helpers for BFS based algorithms."""

from __future__ import annotations

from collections import defaultdict

import daft
from daft import DataFrame, Expression, col, lit

from daft_graph.graph import Graph
from daft_graph.iterate import bound_partitions
from daft_graph.schema import DST, ID, SRC

#: Working columns of the level frame returned by :func:`bfs_levels`.
DIST = "__dist"
PRED = "__pred"


def prepare_edges(graph: Graph, *, edge_filter: Expression | None = None) -> DataFrame:
    """Project graph edges to ``(src, dst)``, applying ``edge_filter`` first.

    The filter runs before the projection so it can reference any edge attribute
    column, and the orientation is applied afterwards so an undirected graph
    symmetrizes only the edges that survived the filter.

    Args:
        graph: The graph to read edges from. Its flavor decides the orientation.
        edge_filter: Optional predicate applied to the edges before projection.

    Returns:
        A two column ``(src, dst)`` DataFrame oriented for traversal.
    """
    edges = graph.edges
    if edge_filter is not None:
        edges = edges.where(edge_filter)
    edges = edges.select(SRC, DST)
    return graph.orient(edges)


def bfs_levels(
    edges: DataFrame,
    sources: DataFrame,
    *,
    max_hops: int,
    targets: DataFrame | None = None,
) -> tuple[DataFrame | None, int | None]:
    """Multi source BFS that keeps the frontier inside Daft.

    The frontier, the visited set, and the predecessor records are all
    DataFrames, so no hop pulls the frontier's adjacency into the driver. Only
    scalar row counts cross back per round. This is what lets the traversal
    algorithms scale past what fits in driver memory during the search itself;
    whatever they return to Python afterwards is their own bound to document.

    Args:
        edges: Materialized ``(src, dst)`` edge frame, already oriented.
        sources: One column ``id`` frame of starting vertices.
        max_hops: Maximum number of hops to expand.
        targets: Optional one column ``id`` frame. When given the search stops at
            the first hop that reaches any of them.

    Returns:
        ``(levels, depth)``. ``levels`` holds one row per discovered
        ``(id, __dist, __pred)`` triple, meaning ``id`` sits at distance
        ``__dist`` and ``__pred`` is one of its shortest path predecessors; it is
        None when nothing was discovered. ``depth`` is the hop count at which a
        target was first reached, or None if none was (or no ``targets`` given).
    """
    visited = bound_partitions(sources.select(col(ID)).distinct()).collect()
    frontier = visited
    levels: DataFrame | None = None
    for hop in range(1, max_hops + 1):
        if frontier.count_rows() == 0:
            break
        # One hop, entirely as a Daft join: frontier -> its out neighbors.
        step = (
            frontier.join(edges, left_on=ID, right_on=SRC, how="inner")
            .select(col(DST).alias(ID), col(SRC).alias(PRED))
            .distinct()
        )
        # Keep only vertices not already at a shorter distance, so every
        # surviving (id, pred) pair is a genuine shortest path predecessor.
        fresh = step.join(visited, on=ID, how="anti").collect()
        if fresh.count_rows() == 0:
            break
        layer = fresh.with_column(DIST, lit(hop))
        # visited and levels are carried across hops via union_all (which sums
        # partition counts) and re-collected, then joined against the next hop, so
        # bound them or the frontier join grid compounds every hop.
        levels = layer if levels is None else bound_partitions(levels.union_all(layer)).collect()
        new_ids = fresh.select(col(ID)).distinct().collect()
        visited = bound_partitions(visited.union_all(new_ids)).collect()
        if targets is not None and new_ids.join(targets, on=ID, how="semi").count_rows() > 0:
            return (levels.collect(), hop)
        frontier = new_ids
    return (levels.collect() if levels is not None else None, None)


def min_predecessor(levels: DataFrame, vertex: int, dist: int) -> int:
    """Smallest shortest path predecessor of ``vertex`` at distance ``dist``.

    A single row point lookup, so path reconstruction costs one small query per
    hop rather than collecting the whole predecessor map.
    """
    row = (
        levels.where((col(ID) == lit(vertex)) & (col(DIST) == lit(dist)))
        .agg(col(PRED).min().alias(PRED))
        .collect()
        .to_pydict()
    )
    return int(row[PRED][0])


def prune_to_dag(levels: DataFrame, reached: DataFrame, depth: int) -> tuple[dict[int, list[int]], set[int]]:
    """Collect only the shortest path sub DAG that leads to ``reached``.

    Walks the predecessor records backwards one level at a time, in Daft, keeping
    just the vertices that lie on a shortest path to a reached target. The result
    is what path enumeration actually needs, and is far smaller than the full
    visited set that a naive collect would pull in.

    Args:
        levels: The level frame from :func:`bfs_levels`.
        reached: One column ``id`` frame of targets hit at ``depth``.
        depth: The shortest distance at which a target was reached.

    Returns:
        ``(preds, origins)``: ``preds`` maps each vertex on the sub DAG to its
        sorted shortest path predecessors, and ``origins`` is the set of distance
        zero vertices the paths start from.
    """
    preds: dict[int, list[int]] = defaultdict(list)
    layer = reached.select(col(ID)).distinct().collect()
    for dist in range(depth, 0, -1):
        step = levels.where(col(DIST) == lit(dist)).join(layer, on=ID, how="semi").collect()
        rows = step.select(col(ID), col(PRED)).distinct().collect().to_pydict()
        for vertex, pred in zip(rows[ID], rows[PRED]):
            preds[int(vertex)].append(int(pred))
        layer = step.select(col(PRED).alias(ID)).distinct().collect()
    origins = {int(x) for x in layer.select(col(ID)).collect().to_pydict()[ID]}
    return ({v: sorted(set(ps)) for v, ps in preds.items()}, origins)


def neighbors(edges: DataFrame, sources: list[int]) -> dict[int, list[int]]:
    """Out neighbors of each source vertex, fetched with a single Daft join.

    Pulls the adjacency of ``sources`` into the driver, so callers must bound
    ``sources`` themselves. Prefer :func:`bfs_levels` for anything whose frontier
    grows with the graph.
    """
    rows = (
        edges.join(daft.from_pydict({SRC: sources}), on=SRC, how="inner")
        .select(SRC, DST)
        .distinct()
        .collect()
        .to_pydict()
    )
    adjacency: dict[int, list[int]] = defaultdict(list)
    for s, d in zip(rows[SRC], rows[DST]):
        adjacency[int(s)].append(int(d))
    return adjacency
