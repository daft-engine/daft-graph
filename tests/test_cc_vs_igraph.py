"""Validate connected_components against igraph on random graphs."""

from __future__ import annotations

import random
from collections import defaultdict

import daft
import igraph as ig
import pytest

from daft_graph.algorithms.connected_components import connected_components
from daft_graph.graph import UndirectedGraph
from daft_graph.schema import COMPONENT, DST, ID, SRC


def _random_edges(n_nodes: int, n_edges: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    edges = []
    for _ in range(n_edges):
        u = rng.randint(0, n_nodes - 1)
        v = rng.randint(0, n_nodes - 1)
        if u != v:
            edges.append((u, v))
    if not edges:
        edges.append((0, 1))
    return edges


def _our_partition(node_ids: list[int], edges: list[tuple[int, int]]) -> set:
    vertices = daft.from_pydict({ID: node_ids})
    edges_df = daft.from_pydict({SRC: [u for u, _ in edges], DST: [v for _, v in edges]})
    g = UndirectedGraph(vertices=vertices, edges=edges_df)
    d = connected_components(g).collect().to_pydict()
    groups: dict[int, set] = defaultdict(set)
    for node, comp in zip(d[ID], d[COMPONENT]):
        groups[comp].add(node)
    return {frozenset(members) for members in groups.values()}


def _igraph_partition(node_ids: list[int], edges: list[tuple[int, int]]) -> set:
    idx = {n: i for i, n in enumerate(node_ids)}
    ig_edges = [(idx[u], idx[v]) for u, v in edges]
    graph = ig.Graph(n=len(node_ids), edges=ig_edges, directed=False)
    comps = graph.connected_components(mode="weak")
    return {frozenset(node_ids[i] for i in comp) for comp in comps}


@pytest.mark.parametrize(
    ("n_nodes", "n_edges", "seed"),
    [(5, 8, 1), (10, 15, 2), (20, 25, 3), (50, 70, 4), (30, 10, 5), (15, 40, 6)],
)
def test_matches_igraph(n_nodes: int, n_edges: int, seed: int) -> None:
    node_ids = list(range(n_nodes))
    edges = _random_edges(n_nodes, n_edges, seed)
    assert _our_partition(node_ids, edges) == _igraph_partition(node_ids, edges)
