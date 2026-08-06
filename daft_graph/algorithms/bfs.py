"""Breadth first search and shortest path enumeration on Daft DataFrames.

``bfs`` returns one shortest path between two vertices; ``all_shortest_paths``
returns every shortest path. Both expand the BFS frontier one hop at a time,
using Daft to look up the frontier's out neighbors, and accept an optional
``edge_filter`` predicate over the edge columns. ``bfs`` breaks ties toward the
smaller predecessor id for determinism.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import daft
from daft import DataFrame, Expression, col
from daft.functions import to_struct

from daft_graph.algorithms._traversal import neighbors, prepare_edges
from daft_graph.graph import Graph
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
    """
    if source == target:
        return [source]
    edges = prepare_edges(graph, edge_filter=edge_filter)
    pred: dict[int, int] = {}
    visited: set[int] = {source}
    frontier: list[int] = [source]
    for _ in range(max_path_length):
        if not frontier:
            break
        adjacency = neighbors(edges, frontier)
        layer_pred: dict[int, int] = {}
        for s in frontier:
            for d in adjacency.get(s, []):
                if d not in visited and (d not in layer_pred or s < layer_pred[d]):
                    layer_pred[d] = s
        for d, s in layer_pred.items():
            visited.add(d)
            pred[d] = s
        if target in visited:
            path = [target]
            while path[-1] != source:
                path.append(pred[path[-1]])
            return list(reversed(path))
        frontier = sorted(layer_pred)
    return None


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

    Raises:
        ValueError: if the working set of partial paths exceeds ``max_paths``
            (a guard against exponential blow up on dense graphs).
    """
    if source == target:
        return [[source]]
    edges = prepare_edges(graph, edge_filter=edge_filter)
    frontier: list[list[int]] = [[source]]
    for _ in range(max_path_length):
        if not frontier:
            break
        if len(frontier) > max_paths:
            raise ValueError(
                f"partial path frontier exceeded max_paths={max_paths}; reduce max_path_length or raise max_paths"
            )
        endpoints = sorted({path[-1] for path in frontier})
        adjacency = neighbors(edges, endpoints)
        found: list[tuple[int, ...]] = []
        nxt: list[list[int]] = []
        for path in frontier:
            for neighbor in adjacency.get(path[-1], []):
                if neighbor in path:
                    continue
                extended = path + [neighbor]
                if neighbor == target:
                    found.append(tuple(extended))
                else:
                    nxt.append(extended)
        if found:
            return [list(path) for path in sorted(set(found))]
        frontier = nxt
    return []


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
    edges: DataFrame, sources: set[int], targets: set[int], max_path_length: int
) -> tuple[int | None, list[int], dict[int, int], dict[int, list[int]]]:
    """Multi source BFS returning the shortest distance to any target.

    Returns ``(depth, reached, dist, preds)``: ``depth`` is the shortest distance
    from any source to any target (None if none is reached within
    ``max_path_length``), ``reached`` is the sorted targets at that distance,
    ``dist`` maps each visited vertex to its distance, and ``preds`` maps each
    visited non source vertex to all of its shortest path predecessors.
    """
    dist: dict[int, int] = {s: 0 for s in sources}
    preds: dict[int, list[int]] = {}
    frontier = sorted(sources)
    for level in range(1, max_path_length + 1):
        if not frontier:
            break
        adjacency = neighbors(edges, frontier)
        newly: dict[int, list[int]] = defaultdict(list)
        for u in frontier:
            for v in adjacency.get(u, []):
                if v not in dist:
                    newly[v].append(u)
        if not newly:
            break
        for v, ps in newly.items():
            dist[v] = level
            preds[v] = sorted(set(ps))
        reached = sorted(set(newly) & targets)
        if reached:
            return level, reached, dist, preds
        frontier = sorted(newly)
    return None, [], dist, preds


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

    Raises:
        ValueError: if the number of shortest paths exceeds ``max_paths``.
    """
    sources = {int(x) for x in graph.vertices.where(from_filter).select(ID).distinct().collect().to_pydict()[ID]}
    targets = {int(x) for x in graph.vertices.where(to_filter).select(ID).distinct().collect().to_pydict()[ID]}
    if not sources or not targets:
        return daft.from_pydict({"from": [], "to": []})

    full_edges = graph.edges if edge_filter is None else graph.edges.where(edge_filter)
    vlk = _vertex_struct_lookup(graph.vertices)

    overlap = sources & targets
    if overlap:
        result = daft.from_pydict({"__p0": sorted(overlap)})
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

    traversal = graph._orient(full_edges.select(SRC, DST))
    depth, reached, _dist, preds = _shortest_path_dag(traversal, sources, targets, max_path_length)
    if depth is None:
        return daft.from_pydict({"from": [], "to": []})

    paths = _enumerate_paths(reached, sources, preds, max_paths)
    pcols: dict[str, Any] = {f"__p{i}": [path[i] for path in paths] for i in range(depth + 1)}
    result = daft.from_pydict(pcols)
    for i in range(depth + 1):
        result = result.join(
            vlk.select(col(_VID).alias(f"__p{i}"), col(_VSTRUCT).alias(_vertex_col(i, depth))),
            on=f"__p{i}",
            how="left",
        )
    elk = _edge_struct_lookup(full_edges, directed=graph._directed)
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
