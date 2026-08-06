"""Undirected traversal and filter-before-orient coverage.

The refactor routes direction through the graph flavor, and `prepare_edges`
applies `edge_filter` before symmetrizing so an undirected graph only walks the
edges that survived the filter. These tests pin that behavior, which the
per-algorithm suites did not exercise for the undirected + filtered combination.
"""

from __future__ import annotations

import daft
from daft import col

from daft_graph import (
    UndirectedGraph,
    all_paths,
    all_shortest_paths,
    bfs,
    hyper_anf,
    shortest_paths,
)
from daft_graph.schema import DST, ID, SRC


def _typed_graph() -> UndirectedGraph:
    # 0-1 (a), 1-2 (b), 0-2 (a). Stored directed as given; walked undirected.
    edges = daft.from_pydict({SRC: [0, 1, 0], DST: [1, 2, 2], "type": ["a", "b", "a"]})
    return UndirectedGraph(edges)


def test_bfs_undirected_walks_both_ways() -> None:
    g = _typed_graph()
    # 2 -> 0 is only reachable undirected (stored edges point 0->2, 0->1, 1->2)
    assert bfs(g, 2, 0) == [2, 0]


def test_bfs_undirected_edge_filter_removes_edge_both_directions() -> None:
    g = _typed_graph()
    # keep only type "a" edges (0-1 and 0-2). 1->2 (type b) is gone both ways,
    # so 1 to 2 must detour through 0.
    path = bfs(g, 1, 2, edge_filter=col("type") == "a")
    assert path == [1, 0, 2]


def test_all_shortest_paths_undirected_with_filter() -> None:
    g = _typed_graph()
    paths = all_shortest_paths(g, 1, 2, edge_filter=col("type") == "a")
    assert paths == [[1, 0, 2]]


def test_all_shortest_paths_undirected_no_filter_is_direct() -> None:
    g = _typed_graph()
    # without the filter the 1-2 edge exists, so the shortest path is direct
    assert all_shortest_paths(g, 1, 2) == [[1, 2]]


def test_all_paths_undirected() -> None:
    g = _typed_graph()
    paths = all_paths(g, 1, 2, max_path_length=5)
    # both the direct 1-2 and the detour 1-0-2 are simple undirected paths
    assert sorted(paths) == [[1, 0, 2], [1, 2]]


def test_shortest_paths_undirected_reaches_all() -> None:
    g = _typed_graph()
    d = shortest_paths(g, landmarks=[0]).collect().to_pydict()
    dist = dict(zip(d[ID], d["distance"]))
    # undirected: every vertex is within one hop of 0
    assert dist == {0: 0, 1: 1, 2: 1}


def test_hyper_anf_undirected_runs() -> None:
    g = _typed_graph()
    out = hyper_anf(g, max_hops=2).collect().to_pydict()
    assert set(out["id"]) == {0, 1, 2}
