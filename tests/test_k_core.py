"""Tests for k_core against networkx."""

from __future__ import annotations

import random

import daft
import networkx as nx
import pytest

from daft_graph.algorithms.k_core import CORE, k_core
from daft_graph.graph import UndirectedGraph
from daft_graph.schema import DST, ID, SRC


def _graph(node_ids: list[int], edges: list[tuple[int, int]]) -> UndirectedGraph:
    vertices = daft.from_pydict({ID: node_ids})
    edges_df = daft.from_pydict({SRC: [u for u, _ in edges], DST: [v for _, v in edges]})
    return UndirectedGraph(vertices=vertices, edges=edges_df)


def _core_map(df: daft.DataFrame) -> dict:
    d = df.collect().to_pydict()
    return dict(zip(d[ID], d[CORE]))


def _undirected_edges(n_nodes: int, n_edges: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    while len(edges) < n_edges:
        u = rng.randint(0, n_nodes - 1)
        v = rng.randint(0, n_nodes - 1)
        if u != v:
            edges.add((min(u, v), max(u, v)))
    return sorted(edges)


def test_triangle_core_is_two() -> None:
    g = _graph([1, 2, 3], [(1, 2), (2, 3), (1, 3)])
    assert _core_map(k_core(g)) == {1: 2, 2: 2, 3: 2}


def test_path_core_is_one() -> None:
    g = _graph([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)])
    assert _core_map(k_core(g)) == {1: 1, 2: 1, 3: 1, 4: 1}


def test_isolated_vertex_core_zero() -> None:
    g = _graph([1, 2, 3], [(1, 2)])
    assert _core_map(k_core(g)) == {1: 1, 2: 1, 3: 0}


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_matches_networkx(seed: int) -> None:
    nodes = list(range(12))
    edges = _undirected_edges(12, 26, seed)
    ours = _core_map(k_core(_graph(nodes, edges)))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    assert ours == nx.core_number(graph)
