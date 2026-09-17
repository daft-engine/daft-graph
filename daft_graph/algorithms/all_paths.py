"""All simple paths between two vertices, bounded by length.

Enumerates every simple path (no repeated vertex) from source to target with at
most ``max_path_length`` edges, extending partial paths one hop at a time and
using Daft to look up the frontier's out neighbors. The number of simple paths
can grow exponentially, so ``max_path_length`` and ``max_paths`` bound the search.
"""

from __future__ import annotations

from daft import Expression

from daft_graph.algorithms._traversal import neighbors, prepare_edges
from daft_graph.graph import Graph
from daft_graph.iterate import collect_bounded


def all_paths(
    graph: Graph,
    source: int,
    target: int,
    *,
    max_path_length: int = 5,
    edge_filter: Expression | None = None,
    max_paths: int = 1_000_000,
) -> list[list[int]]:
    """Return all simple paths from source to target with at most N edges.

    Args:
        graph: The graph to search.
        source: Starting vertex id.
        target: Goal vertex id.
        max_path_length: Maximum number of edges in a returned path.
        edge_filter: Optional predicate over the edge columns; only matching edges
            are traversed.
        max_paths: Guard against exponential blow up; raises if the working set of
            partial paths exceeds this.

    Returns:
        A sorted list of paths, each a list of vertex ids from source to target.
        If source == target the single trivial path ``[source]`` is returned.

    Note:
        The search is driver mediated: each hop pulls the current frontier's
        adjacency into the driver process, and the returned paths are Python
        objects. Frontier width is not bounded by ``max_path_length``, so on a
        large well connected graph the frontier can reach a sizeable fraction of
        the graph within a few hops. Keep the hop count tight, narrow the search
        with ``edge_filter``, or use :func:`daft_graph.shortest_paths` (which
        stays in Daft) for whole graph distances.

    Raises:
        ValueError: if the partial path frontier exceeds ``max_paths``.
    """
    if source == target:
        return [[source]]
    edges = collect_bounded(prepare_edges(graph, edge_filter=edge_filter))

    frontier: list[list[int]] = [[source]]
    results: list[tuple[int, ...]] = []
    for _ in range(max_path_length):
        if not frontier:
            break
        if len(frontier) > max_paths:
            raise ValueError(
                f"partial path frontier exceeded max_paths={max_paths}; reduce max_path_length or raise max_paths"
            )
        endpoints = sorted({path[-1] for path in frontier})
        adjacency = neighbors(edges, endpoints)
        nxt: list[list[int]] = []
        for path in frontier:
            for neighbor in adjacency.get(path[-1], []):
                if neighbor in path:
                    continue
                extended = path + [neighbor]
                if neighbor == target:
                    results.append(tuple(extended))
                else:
                    nxt.append(extended)
        frontier = nxt
    return [list(path) for path in sorted(set(results))]
