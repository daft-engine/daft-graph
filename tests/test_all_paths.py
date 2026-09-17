"""Tests for all_paths against networkx."""

from __future__ import annotations

import random

import daft
import networkx as nx
import pytest
from daft import col

from daft_graph.algorithms.all_paths import all_paths
from daft_graph.graph import DirectedGraph
from daft_graph.schema import DST, ID, SRC


def _graph(node_ids: list[int], edges: list[tuple[int, int]]) -> DirectedGraph:
    return DirectedGraph(
        vertices=daft.from_pydict({ID: node_ids}),
        edges=daft.from_pydict({SRC: [u for u, _ in edges], DST: [v for _, v in edges]}),
    )


def test_single_path() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    assert all_paths(g, 0, 3) == [[0, 1, 2, 3]]


def test_multiple_paths_diamond() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (0, 2), (1, 3), (2, 3)])
    assert all_paths(g, 0, 3) == [[0, 1, 3], [0, 2, 3]]


def test_max_length_bounds_search() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    assert all_paths(g, 0, 3, max_path_length=2) == []


def test_source_equals_target() -> None:
    g = _graph([0, 1], [(0, 1)])
    assert all_paths(g, 0, 0) == [[0]]


def test_no_repeated_vertices() -> None:
    g = _graph([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
    assert all_paths(g, 0, 2) == [[0, 1, 2]]


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
    ours = {tuple(p) for p in all_paths(g, 0, 8, max_path_length=4)}
    theirs = {tuple(p) for p in nx.all_simple_paths(graph, 0, 8, cutoff=4)}
    assert ours == theirs


def test_edge_filter() -> None:
    g = DirectedGraph(
        vertices=daft.from_pydict({ID: [0, 1, 2]}),
        edges=daft.from_pydict({SRC: [0, 1, 0], DST: [1, 2, 2], "type": ["a", "a", "b"]}),
    )
    assert all_paths(g, 0, 2, edge_filter=col("type") == "a") == [[0, 1, 2]]
    assert all_paths(g, 0, 2) == [[0, 1, 2], [0, 2]]


def test_max_paths_guard() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (0, 2), (1, 3), (2, 3)])
    with pytest.raises(ValueError):
        all_paths(g, 0, 3, max_paths=1)
