"""Tests for triangle_count against networkx."""

from __future__ import annotations

import random

import daft
import networkx as nx
import pytest

from daft_graph.algorithms.triangle_count import TRIANGLE_COUNT, triangle_count
from daft_graph.graph import UndirectedGraph
from daft_graph.schema import DST, ID, SRC


def _graph(node_ids: list[int], edges: list[tuple[int, int]]) -> UndirectedGraph:
    vertices = daft.from_pydict({ID: node_ids})
    edges_df = daft.from_pydict({SRC: [u for u, _ in edges], DST: [v for _, v in edges]})
    return UndirectedGraph(vertices=vertices, edges=edges_df)


def _count_map(df: daft.DataFrame) -> dict:
    d = df.collect().to_pydict()
    return dict(zip(d[ID], d[TRIANGLE_COUNT]))


def _undirected_edges(n_nodes: int, n_edges: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    while len(edges) < n_edges:
        u = rng.randint(0, n_nodes - 1)
        v = rng.randint(0, n_nodes - 1)
        if u != v:
            edges.add((min(u, v), max(u, v)))
    return sorted(edges)


def test_single_triangle() -> None:
    g = _graph([1, 2, 3], [(1, 2), (2, 3), (1, 3)])
    assert _count_map(triangle_count(g)) == {1: 1, 2: 1, 3: 1}


def test_path_has_no_triangles() -> None:
    g = _graph([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)])
    assert _count_map(triangle_count(g)) == {1: 0, 2: 0, 3: 0, 4: 0}


def test_no_edges() -> None:
    g = _graph([1, 2, 3], [])
    assert _count_map(triangle_count(g)) == {1: 0, 2: 0, 3: 0}


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_matches_networkx(seed: int) -> None:
    node_ids = list(range(12))
    edges = _undirected_edges(12, 24, seed)
    ours = _count_map(triangle_count(_graph(node_ids, edges)))
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    graph.add_edges_from(edges)
    theirs = nx.triangles(graph)
    assert ours == theirs
