"""Tests for bfs."""

from __future__ import annotations

import itertools
import random

import daft
import networkx as nx
import pytest

from daft_graph.algorithms.bfs import bfs
from daft_graph.graph import DirectedGraph
from daft_graph.schema import DST, ID, SRC


def _graph(node_ids: list[int], edges: list[tuple[int, int]]) -> DirectedGraph:
    vertices = daft.from_pydict({ID: node_ids})
    edges_df = daft.from_pydict({SRC: [u for u, _ in edges], DST: [v for _, v in edges]})
    return DirectedGraph(vertices=vertices, edges=edges_df)


def test_directed_path() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    assert bfs(g, 0, 3) == [0, 1, 2, 3]


def test_source_equals_target() -> None:
    g = _graph([0, 1, 2], [(0, 1), (1, 2)])
    assert bfs(g, 2, 2) == [2]


def test_unreachable_directed_returns_none() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    assert bfs(g, 3, 0) is None


def test_undirected_reaches_backwards() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    assert bfs(g.as_undirected(), 3, 0) == [3, 2, 1, 0]


def test_max_path_length_cutoff() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    assert bfs(g, 0, 3, max_path_length=2) is None


def test_picks_shorter_branch() -> None:
    g = _graph([0, 1, 2, 3, 4], [(0, 1), (1, 4), (0, 2), (2, 3), (3, 4)])
    assert bfs(g, 0, 4) == [0, 1, 4]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_length_matches_networkx(seed: int) -> None:
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    while len(edges) < 20:
        u, v = rng.randint(0, 9), rng.randint(0, 9)
        if u != v:
            edges.add((u, v))
    edge_list = sorted(edges)
    g = _graph(list(range(10)), edge_list)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(10))
    graph.add_edges_from(edge_list)
    edge_set = set(edge_list)
    for target in range(10):
        ours = bfs(g, 0, target, max_path_length=20)
        if ours is None:
            assert not nx.has_path(graph, 0, target)
        else:
            assert len(ours) - 1 == nx.shortest_path_length(graph, 0, target)
            assert ours[0] == 0 and ours[-1] == target
            for a, b in itertools.pairwise(ours):
                assert (a, b) in edge_set
