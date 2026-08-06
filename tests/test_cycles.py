"""Tests for cycle detection against networkx."""

from __future__ import annotations

import random

import daft
import networkx as nx
import pytest

from daft_graph.algorithms.cycles import has_cycle, vertices_on_cycles
from daft_graph.graph import DirectedGraph
from daft_graph.schema import DST, ID, SRC


def _graph(node_ids: list[int], edges: list[tuple[int, int]]) -> DirectedGraph:
    return DirectedGraph(
        vertices=daft.from_pydict({ID: node_ids}),
        edges=daft.from_pydict({SRC: [u for u, _ in edges], DST: [v for _, v in edges]}),
    )


def _ids(df: daft.DataFrame) -> set:
    return set(df.collect().to_pydict()[ID])


def _nx_cyclic(graph: nx.DiGraph) -> set:
    cyclic: set = set()
    for component in nx.strongly_connected_components(graph):
        if len(component) > 1:
            cyclic |= component
    for node in graph:
        if graph.has_edge(node, node):
            cyclic.add(node)
    return cyclic


def test_dag_has_no_cycle() -> None:
    g = _graph([0, 1, 2], [(0, 1), (1, 2)])
    assert has_cycle(g) is False
    assert _ids(vertices_on_cycles(g)) == set()


def test_directed_cycle() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 0), (2, 3)])
    assert has_cycle(g) is True
    assert _ids(vertices_on_cycles(g)) == {0, 1, 2}


def test_self_loop_is_a_cycle() -> None:
    g = _graph([0, 1, 2], [(0, 0), (1, 2)])
    assert has_cycle(g) is True
    assert _ids(vertices_on_cycles(g)) == {0}


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_matches_networkx(seed: int) -> None:
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    while len(edges) < 22:
        u, v = rng.randint(0, 11), rng.randint(0, 11)
        if u != v:
            edges.add((u, v))
    edge_list = sorted(edges)
    nodes = list(range(12))
    g = _graph(nodes, edge_list)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edge_list)
    assert _ids(vertices_on_cycles(g)) == _nx_cyclic(graph)
    assert has_cycle(g) == (not nx.is_directed_acyclic_graph(graph))
