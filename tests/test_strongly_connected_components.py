"""Tests for strongly_connected_components against networkx."""

from __future__ import annotations

import random
from collections import defaultdict

import daft
import networkx as nx
import pytest

from daft_graph.algorithms.strongly_connected_components import (
    strongly_connected_components,
)
from daft_graph.graph import DirectedGraph
from daft_graph.schema import COMPONENT, DST, ID, SRC


def _graph(node_ids: list[int], edges: list[tuple[int, int]]) -> DirectedGraph:
    vertices = daft.from_pydict({ID: node_ids})
    edges_df = daft.from_pydict({SRC: [u for u, _ in edges], DST: [v for _, v in edges]})
    return DirectedGraph(vertices=vertices, edges=edges_df)


def _partition(df: daft.DataFrame) -> set:
    d = df.collect().to_pydict()
    groups: dict[int, set] = defaultdict(set)
    for node, comp in zip(d[ID], d[COMPONENT]):
        groups[comp].add(node)
    return {frozenset(members) for members in groups.values()}


def _nx_partition(node_ids: list[int], edges: list[tuple[int, int]]) -> set:
    graph = nx.DiGraph()
    graph.add_nodes_from(node_ids)
    graph.add_edges_from(edges)
    return {frozenset(c) for c in nx.strongly_connected_components(graph)}


def _random_digraph(n_nodes: int, n_edges: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    while len(edges) < n_edges:
        u = rng.randint(0, n_nodes - 1)
        v = rng.randint(0, n_nodes - 1)
        if u != v:
            edges.add((u, v))
    return sorted(edges)


def test_cycle_is_one_scc() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 0), (0, 3)])
    expected = {frozenset({0, 1, 2}), frozenset({3})}
    assert _partition(strongly_connected_components(g, strategy="local")) == expected
    assert _partition(strongly_connected_components(g, strategy="distributed")) == expected


def test_dag_all_singletons() -> None:
    g = _graph([0, 1, 2], [(0, 1), (1, 2)])
    expected = {frozenset({0}), frozenset({1}), frozenset({2})}
    assert _partition(strongly_connected_components(g, strategy="distributed")) == expected


def test_distributed_labels_by_min_id() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 0), (0, 3)])
    d = strongly_connected_components(g, strategy="distributed").collect().to_pydict()
    labels = dict(zip(d[ID], d[COMPONENT]))
    assert labels[0] == labels[1] == labels[2] == 0
    assert labels[3] == 3


def test_local_and_distributed_labels_agree() -> None:
    nodes = list(range(10))
    edges = _random_digraph(10, 18, 7)
    g = _graph(nodes, edges)
    local = strongly_connected_components(g, strategy="local").collect().to_pydict()
    dist = strongly_connected_components(g, strategy="distributed").collect().to_pydict()
    assert dict(zip(local[ID], local[COMPONENT])) == dict(zip(dist[ID], dist[COMPONENT]))


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_local_matches_networkx(seed: int) -> None:
    nodes = list(range(12))
    edges = _random_digraph(12, 22, seed)
    assert _partition(strongly_connected_components(_graph(nodes, edges), strategy="local")) == _nx_partition(
        nodes, edges
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_distributed_matches_networkx(seed: int) -> None:
    nodes = list(range(12))
    edges = _random_digraph(12, 22, seed)
    assert _partition(strongly_connected_components(_graph(nodes, edges), strategy="distributed")) == _nx_partition(
        nodes, edges
    )
