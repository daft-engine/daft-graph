"""Tests for daft_graph.algorithms.pagerank against networkx."""

from __future__ import annotations

import random

import daft
import networkx as nx
import pytest

from daft_graph.algorithms.pagerank import pagerank
from daft_graph.graph import DirectedGraph
from daft_graph.schema import DST, ID, RANK, SRC


def _graph(node_ids: list[int], edges: list[tuple[int, int]]) -> DirectedGraph:
    vertices = daft.from_pydict({ID: node_ids})
    edges_df = daft.from_pydict({SRC: [u for u, _ in edges], DST: [v for _, v in edges]})
    return DirectedGraph(vertices=vertices, edges=edges_df)


def _our_ranks(g: DirectedGraph) -> dict:
    d = pagerank(g, damping=0.85, tol=1e-10, max_iters=300).collect().to_pydict()
    return dict(zip(d[ID], d[RANK]))


def _nx_ranks(node_ids: list[int], edges: list[tuple[int, int]]) -> dict:
    graph = nx.DiGraph()
    graph.add_nodes_from(node_ids)
    graph.add_edges_from(edges)
    return nx.pagerank(graph, alpha=0.85, tol=1e-12, max_iter=1000)


def _random_directed_edges(n_nodes: int, n_edges: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    while len(edges) < n_edges:
        u = rng.randint(0, n_nodes - 1)
        v = rng.randint(0, n_nodes - 1)
        if u != v:
            edges.add((u, v))
    return sorted(edges)


def test_directed_cycle_is_uniform() -> None:
    edges = [(0, 1), (1, 2), (2, 0)]
    ranks = _our_ranks(_graph([0, 1, 2], edges))
    for value in ranks.values():
        assert abs(value - 1.0 / 3.0) < 1e-6
    assert abs(sum(ranks.values()) - 1.0) < 1e-9


def test_dangling_node_matches_networkx() -> None:
    # node 2 has no out edge (dangling)
    edges = [(0, 1), (1, 2), (0, 2)]
    node_ids = [0, 1, 2]
    ours = _our_ranks(_graph(node_ids, edges))
    theirs = _nx_ranks(node_ids, edges)
    for nid in node_ids:
        assert abs(ours[nid] - theirs[nid]) < 1e-4


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_graph_matches_networkx(seed: int) -> None:
    node_ids = list(range(8))
    edges = _random_directed_edges(8, 14, seed)
    ours = _our_ranks(_graph(node_ids, edges))
    theirs = _nx_ranks(node_ids, edges)
    for nid in node_ids:
        assert abs(ours[nid] - theirs[nid]) < 1e-4


def test_ranks_sum_to_one() -> None:
    edges = _random_directed_edges(10, 20, 99)
    ranks = _our_ranks(_graph(list(range(10)), edges))
    assert abs(sum(ranks.values()) - 1.0) < 1e-6
