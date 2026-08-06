"""Tests for all_shortest_paths and bfs edge_filter against networkx."""

from __future__ import annotations

import random

import daft
import networkx as nx
import pytest
from daft import col

from daft_graph.algorithms.bfs import all_shortest_paths, bfs
from daft_graph.graph import DirectedGraph
from daft_graph.schema import DST, ID, SRC


def _graph(
    node_ids: list[int],
    edges: list[tuple[int, int]],
    types: list[str] | None = None,
) -> DirectedGraph:
    columns = {SRC: [u for u, _ in edges], DST: [v for _, v in edges]}
    if types is not None:
        columns["type"] = types
    return DirectedGraph(
        vertices=daft.from_pydict({ID: node_ids}),
        edges=daft.from_pydict(columns),
    )


def test_diamond_returns_both_paths() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (0, 2), (1, 3), (2, 3)])
    assert all_shortest_paths(g, 0, 3) == [[0, 1, 3], [0, 2, 3]]


def test_only_shortest_returned() -> None:
    g = _graph([0, 1, 2, 3, 4], [(0, 1), (1, 3), (0, 2), (2, 4), (4, 3)])
    assert all_shortest_paths(g, 0, 3) == [[0, 1, 3]]


def test_source_equals_target() -> None:
    g = _graph([0, 1], [(0, 1)])
    assert all_shortest_paths(g, 0, 0) == [[0]]


def test_unreachable_returns_empty() -> None:
    g = _graph([0, 1, 2], [(0, 1)])
    assert all_shortest_paths(g, 0, 2) == []


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_matches_networkx(seed: int) -> None:
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    while len(edges) < 16:
        u, v = rng.randint(0, 8), rng.randint(0, 8)
        if u != v:
            edges.add((u, v))
    edge_list = sorted(edges)
    nodes = list(range(9))
    g = _graph(nodes, edge_list)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edge_list)
    for target in nodes:
        ours = {tuple(p) for p in all_shortest_paths(g, 0, target, max_path_length=9)}
        if nx.has_path(graph, 0, target):
            theirs = {tuple(p) for p in nx.all_shortest_paths(graph, 0, target)}
            assert ours == theirs
        else:
            assert ours == set()


def test_edge_filter_on_bfs() -> None:
    g = _graph([0, 1, 2], [(0, 1), (1, 2), (0, 2)], types=["a", "a", "b"])
    assert bfs(g, 0, 2) == [0, 2]
    assert bfs(g, 0, 2, edge_filter=col("type") == "a") == [0, 1, 2]


def test_edge_filter_on_all_shortest_paths() -> None:
    g = _graph([0, 1, 2], [(0, 1), (1, 2), (0, 2)], types=["a", "a", "b"])
    assert all_shortest_paths(g, 0, 2, edge_filter=col("type") == "a") == [[0, 1, 2]]


def test_max_paths_guard() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (0, 2), (1, 3), (2, 3)])
    with pytest.raises(ValueError):
        all_shortest_paths(g, 0, 3, max_paths=1)
