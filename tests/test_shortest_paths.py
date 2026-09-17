"""Tests for shortest_paths against networkx."""

from __future__ import annotations

import random

import daft
import networkx as nx
import pytest

from daft_graph.algorithms.shortest_paths import DISTANCE, LANDMARK, shortest_paths
from daft_graph.graph import DirectedGraph
from daft_graph.schema import DST, ID, SRC


def _graph(node_ids: list[int], edges: list[tuple[int, int]]) -> DirectedGraph:
    vertices = daft.from_pydict({ID: node_ids})
    edges_df = daft.from_pydict({SRC: [u for u, _ in edges], DST: [v for _, v in edges]})
    return DirectedGraph(vertices=vertices, edges=edges_df)


def _dist_map(df: daft.DataFrame) -> dict:
    d = df.collect().to_pydict()
    return {(i, lm): dist for i, lm, dist in zip(d[ID], d[LANDMARK], d[DISTANCE])}


def test_path_distances_to_landmark() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    got = _dist_map(shortest_paths(g, [3]))
    assert got == {(0, 3): 3, (1, 3): 2, (2, 3): 1, (3, 3): 0}


def test_empty_landmarks_raises() -> None:
    g = _graph([0, 1], [(0, 1)])
    with pytest.raises(ValueError):
        shortest_paths(g, [])


def test_empty_edges_only_valid_landmark_self() -> None:
    g = _graph([0, 1, 2], [])
    got = _dist_map(shortest_paths(g, [1, 99]))
    assert got == {(1, 1): 0}


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_matches_networkx(seed: int) -> None:
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    while len(edges) < 22:
        u, v = rng.randint(0, 9), rng.randint(0, 9)
        if u != v:
            edges.add((u, v))
    edge_list = sorted(edges)
    nodes = list(range(10))
    g = _graph(nodes, edge_list)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edge_list)
    landmarks = [0, 5, 9]
    got = _dist_map(shortest_paths(g, landmarks, max_iters=20))
    for landmark in landmarks:
        nx_dist = dict(nx.single_target_shortest_path_length(graph, landmark))
        for v in nodes:
            if (v, landmark) in got:
                assert got[(v, landmark)] == nx_dist[v]
            else:
                assert v not in nx_dist
