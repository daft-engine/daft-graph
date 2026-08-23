"""Breadth first search and shortest path enumeration on Daft DataFrames.

``bfs`` returns one shortest path between two vertices, ``all_shortest_paths``
returns every shortest path, and ``bfs_paths`` is the GraphFrames style search
between two vertex sets. All expand the frontier inside Daft (see
:func:`daft_graph.algorithms._traversal.bfs_levels`), so the search itself does
not pull the graph into the driver; only the resulting paths do, bounded by
``max_paths``. All accept an optional ``edge_filter`` predicate over the edge
columns. ``bfs`` breaks ties toward the smaller predecessor id for determinism.
"""

from __future__ import annotations

from typing import Any

import daft
from daft import DataFrame, Expression, col, lit
from daft.functions import to_struct

from daft_graph.algorithms._traversal import (
    DIST,
    bfs_levels,
    min_predecessor,
    prepare_edges,
    prune_to_dag,
)
from daft_graph.graph import Graph
from daft_graph.iterate import collect_bounded
from daft_graph.schema import DST, ID, SRC


def bfs(
    graph: Graph,
    source: int,
    target: int,
    *,
    max_path_length: int = 10,
    edge_filter: Expression | None = None,
) -> list[int] | None:
    """Return a shortest path of vertex ids from source to target, or None.

    Args:
        graph: The graph to search.
        source: Starting vertex id.
        target: Goal vertex id.
        max_path_length: Maximum number of hops to explore.
        edge_filter: Optional predicate over the edge columns; only matching edges
            are traversed.

    Returns:
        The list of vertex ids on a shortest path (both ends inclusive), or None
        if the target is not reachable within ``max_path_length`` hops.

    Note:
        The frontier is expanded inside Daft, so the search does not pull the
        graph into the driver. Only the returned path itself crosses back, one
        small lookup per hop, bounded by ``max_path_length``.
    """
    if source == target:
        return [source]
    edges = collect_bounded(prepare_edges(graph, edge_filter=edge_filter))
    levels, depth = bfs_levels(
        edges,
        daft.from_pydict({ID: [source]}),
        max_hops=max_path_length,
        targets=daft.from_pydict({ID: [target]}),
    )
    if depth is None or levels is None:
        return None
    # Walk the predecessor records back from the target, taking the smallest
    # predecessor at each hop so the chosen path is deterministic.
    path = [target]
    current = target
    for dist in range(depth, 0, -1):
        current = min_predecessor(levels, current, dist)
        path.append(current)
    return list(reversed(path))


def all_shortest_paths(
    graph: Graph,
    source: int,
    target: int,
    *,
    max_path_length: int = 10,
    edge_filter: Expression | None = None,
    max_paths: int = 1_000_000,
) -> list[list[int]]:
    """Return every shortest path from source to target.

    Expands the frontier layer by layer and returns all paths that first reach
    the target (which therefore all have the shortest length). Returns an empty
    list if the target is unreachable within ``max_path_length`` hops, or
    ``[[source]]`` when source == target.

    Note:
        The frontier is expanded inside Daft, and only the shortest path sub DAG
        is brought back to enumerate paths from. The returned paths are Python
        lists, so ``max_paths`` bounds what crosses into the driver.

    Raises:
        ValueError: if the number of shortest paths exceeds ``max_paths``
            (a guard against exponential blow up on dense graphs).
    """
    if source == target:
        return [[source]]
    edges = collect_bounded(prepare_edges(graph, edge_filter=edge_filter))
    depth, reached, preds, origins = _shortest_path_dag(
        edges,
        daft.from_pydict({ID: [source]}),
        daft.from_pydict({ID: [target]}),
        max_path_length,
    )
    if depth is None or reached is None:
        return []
    reached_ids = sorted(int(x) for x in reached.select(col(ID)).collect().to_pydict()[ID])
    return [list(path) for path in _enumerate_paths(reached_ids, origins, preds, max_paths)]


_VID = "__vid"
_VSTRUCT = "__vstruct"
_TAIL = "__tail"
_HEAD = "__head"
_ESTRUCT = "__estruct"


def _vertex_struct_lookup(vertices: DataFrame) -> DataFrame:
    """Map each vertex id to a struct of its full row, keyed by ``__vid``."""
    attrs = [c for c in vertices.column_names if c != ID]
    fields = [col(ID), *[col(c) for c in attrs]]
    return vertices.select(col(ID).alias(_VID), to_struct(*fields).alias(_VSTRUCT))


def _edge_struct_lookup(edges: DataFrame, *, directed: bool) -> DataFrame:
    """Map each traversable ordered pair to a struct of the edge's full row.

    Keyed by ``__tail`` (the vertex stepped from) and ``__head`` (stepped to). For
    undirected search both orientations point to the original edge struct.
    """
    attrs = [c for c in edges.column_names if c not in (SRC, DST)]
    fields = [col(SRC), col(DST), *[col(c) for c in attrs]]
    base = edges.select(col(SRC), col(DST), to_struct(*fields).alias(_ESTRUCT))
    fwd = base.select(col(SRC).alias(_TAIL), col(DST).alias(_HEAD), col(_ESTRUCT))
    if directed:
        return fwd
    rev = base.select(col(DST).alias(_TAIL), col(SRC).alias(_HEAD), col(_ESTRUCT))
    return fwd.union_all(rev)


def _shortest_path_dag(
    edges: DataFrame, sources: DataFrame, targets: DataFrame, max_path_length: int
) -> tuple[int | None, DataFrame | None, dict[int, list[int]], set[int]]:
    """Multi source BFS returning the shortest path sub DAG to any target.

    The search runs inside Daft via :func:`bfs_levels`, then only the sub DAG that
    leads to a reached target is collected, so driver memory scales with the
    answer rather than with the graph.

    Args:
        edges: Materialized ``(src, dst)`` edge frame, already oriented.
        sources: One column ``id`` frame of starting vertices.
        targets: One column ``id`` frame of goal vertices.
        max_path_length: Maximum hops to expand.

    Returns:
        ``(depth, reached, preds, origins)``: ``depth`` is the shortest distance
        from any source to any target (None if none was reached), ``reached`` is a
        one column ``id`` frame of the targets at that distance, ``preds`` maps
        each sub DAG vertex to its sorted predecessors, and ``origins`` is the set
        of sources those paths start from.
    """
    levels, depth = bfs_levels(edges, sources, max_hops=max_path_length, targets=targets)
    if depth is None or levels is None:
        return None, None, {}, set()
    reached = (
        levels.where(col(DIST) == lit(depth)).select(col(ID)).distinct().join(targets, on=ID, how="semi").collect()
    )
    preds, origins = prune_to_dag(levels, reached, depth)
    return depth, reached, preds, origins


def _enumerate_paths(
    reached: list[int], sources: set[int], preds: dict[int, list[int]], max_paths: int
) -> list[tuple[int, ...]]:
    """Enumerate every shortest path id tuple from a source to a reached target."""
    memo: dict[int, list[tuple[int, ...]]] = {}

    def build(v: int) -> list[tuple[int, ...]]:
        if v in memo:
            return memo[v]
        result: list[tuple[int, ...]]
        if v in sources:
            result = [(v,)]
        else:
            result = []
            for p in preds[v]:
                for sub in build(p):
                    result.append((*sub, v))
                    if len(result) > max_paths:
                        raise ValueError(f"shortest path count exceeded max_paths={max_paths}")
        memo[v] = result
        return result

    out: list[tuple[int, ...]] = []
    for target in reached:
        out.extend(build(target))
        if len(out) > max_paths:
            raise ValueError(f"shortest path count exceeded max_paths={max_paths}")
    return sorted(out)


def _vertex_col(index: int, depth: int) -> str:
    if index == 0:
        return "from"
    if index == depth:
        return "to"
    return f"v{index}"


def bfs_paths(
    graph: Graph,
    from_filter: Expression,
    to_filter: Expression,
    *,
    max_path_length: int = 10,
    edge_filter: Expression | None = None,
    max_paths: int = 1_000_000,
) -> DataFrame:
    """Shortest paths from a source vertex set to a target vertex set.

    This is the GraphFrames ``bfs`` analog. ``from_filter`` and ``to_filter`` are
    predicates over the vertex columns that select the source and target vertex
    sets; the search runs from all sources at once and returns every shortest
    path to the nearest targets (all returned paths share the shortest length).

    Args:
        graph: The graph to search.
        from_filter: Predicate over vertex columns selecting the source vertices.
        to_filter: Predicate over vertex columns selecting the target vertices.
        max_path_length: Maximum number of hops to explore.
        edge_filter: Optional predicate over the edge columns; only matching edges
            are traversed and appear in the path.
        max_paths: Guard against exponential path blow up on dense graphs.

    Returns:
        A DataFrame with one row per shortest path. For paths of length ``d`` the
        columns are ``from, e0, v1, e1, ..., to``: each vertex column holds a
        struct of the vertex row, each edge column a struct of the edge row.
        Parallel edges between the same pair multiply the matching rows. When any
        source is also a target the global shortest length is 0, so only the
        zero-hop overlap is returned (columns ``from`` and ``to``, the same
        vertex) and other sources are not searched. An empty DataFrame with
        columns ``from`` and ``to`` is returned when no target is reachable within
        ``max_path_length``.

    Note:
        The search is driver mediated: the matched source and target vertex sets
        are collected into the driver, and each hop pulls the current frontier's
        adjacency in as well. Frontier width is not bounded by ``max_path_length``, so on a
        large well connected graph the frontier can reach a sizeable fraction of
        the graph within a few hops. Keep the hop count tight, narrow the search
        with ``edge_filter``, or use :func:`daft_graph.shortest_paths` (which
        stays in Daft) for whole graph distances.

    Raises:
        ValueError: if the number of shortest paths exceeds ``max_paths``.
    """
    sources = graph.vertices.where(from_filter).select(col(ID)).distinct()
    targets = graph.vertices.where(to_filter).select(col(ID)).distinct()
    if sources.count_rows() == 0 or targets.count_rows() == 0:
        return daft.from_pydict({"from": [], "to": []})

    full_edges = graph.edges if edge_filter is None else graph.edges.where(edge_filter)
    vlk = _vertex_struct_lookup(graph.vertices)

    overlap = sources.join(targets, on=ID, how="semi")
    if overlap.count_rows() > 0:
        result = overlap.select(col(ID).alias("__p0"))
        result = result.join(
            vlk.select(col(_VID).alias("__p0"), col(_VSTRUCT).alias("from")),
            on="__p0",
            how="left",
        ).join(
            vlk.select(col(_VID).alias("__p0"), col(_VSTRUCT).alias("to")),
            on="__p0",
            how="left",
        )
        return result.select("from", "to")

    traversal = collect_bounded(graph.orient(full_edges.select(SRC, DST)))
    depth, reached, preds, origins = _shortest_path_dag(traversal, sources, targets, max_path_length)
    if depth is None or reached is None:
        return daft.from_pydict({"from": [], "to": []})

    reached_ids = sorted(int(x) for x in reached.select(col(ID)).collect().to_pydict()[ID])
    paths = _enumerate_paths(reached_ids, origins, preds, max_paths)
    pcols: dict[str, Any] = {f"__p{i}": [path[i] for path in paths] for i in range(depth + 1)}
    result = daft.from_pydict(pcols)
    for i in range(depth + 1):
        result = result.join(
            vlk.select(col(_VID).alias(f"__p{i}"), col(_VSTRUCT).alias(_vertex_col(i, depth))),
            on=f"__p{i}",
            how="left",
        )
    elk = _edge_struct_lookup(full_edges, directed=graph.is_directed)
    for i in range(depth):
        keys: list[str | Expression] = [f"__p{i}", f"__p{i + 1}"]
        result = result.join(
            elk.select(
                col(_TAIL).alias(f"__p{i}"),
                col(_HEAD).alias(f"__p{i + 1}"),
                col(_ESTRUCT).alias(f"e{i}"),
            ),
            on=keys,
            how="inner",
        )

    order = ["from"]
    for i in range(depth):
        order.append(f"e{i}")
        order.append("to" if i == depth - 1 else f"v{i + 1}")
    return result.select(*order)
