"""Tests for random_walks."""

from __future__ import annotations

import itertools

import daft

from daft_graph.algorithms.random_walks import random_walks
from daft_graph.graph import DirectedGraph
from daft_graph.schema import DST, ID, SRC


def _graph(node_ids: list[int], edges: list[tuple[int, int]]) -> DirectedGraph:
    return DirectedGraph(
        vertices=daft.from_pydict({ID: node_ids}),
        edges=daft.from_pydict({SRC: [u for u, _ in edges], DST: [v for _, v in edges]}),
    )


def test_walks_traverse_real_edges() -> None:
    edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
    g = _graph([0, 1, 2, 3], edges)
    edge_set = set(edges)
    walks = random_walks(g, walk_length=5, num_walks=2, seed=1)
    for walk in walks:
        assert len(walk) <= 6
        for a, b in itertools.pairwise(walk):
            assert (a, b) in edge_set


def test_walk_count_and_start() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 0), (2, 3)])
    walks = random_walks(g, walk_length=4, num_walks=3, seed=1)
    assert len(walks) == 4 * 3  # num vertices * num_walks
    starts = sorted(walk[0] for walk in walks)
    assert starts == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_deterministic_with_seed() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 0), (2, 3)])
    assert random_walks(g, walk_length=6, num_walks=2, seed=7) == random_walks(g, walk_length=6, num_walks=2, seed=7)


def test_dead_end_stops_walk() -> None:
    g = _graph([0, 1], [(0, 1)])  # 1 is a sink
    walks = random_walks(g, walk_length=5, num_walks=1, seed=1)
    by_start = {walk[0]: walk for walk in walks}
    assert by_start[1] == [1]  # no out edge, walk is just the start
    assert by_start[0] == [0, 1]  # 0 -> 1 then stuck


def test_undirected_can_walk_back() -> None:
    g = _graph([0, 1], [(0, 1)])
    walks = random_walks(g.as_undirected(), walk_length=3, num_walks=1, seed=1)
    by_start = {walk[0]: walk for walk in walks}
    # undirected: 1 can step back to 0
    assert len(by_start[1]) > 1
