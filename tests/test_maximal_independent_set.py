"""Tests for maximal_independent_set: validate the result is a valid MIS."""

from __future__ import annotations

import random
from collections import defaultdict

import daft
import pytest

from daft_graph.algorithms.maximal_independent_set import (
    SELECTED,
    maximal_independent_set,
)
from daft_graph.graph import UndirectedGraph
from daft_graph.schema import DST, ID, SRC


def _graph(node_ids: list[int], edges: list[tuple[int, int]]) -> UndirectedGraph:
    return UndirectedGraph(
        vertices=daft.from_pydict({ID: node_ids}),
        edges=daft.from_pydict({SRC: [u for u, _ in edges], DST: [v for _, v in edges]}),
    )


def _selected(g: UndirectedGraph) -> set:
    d = maximal_independent_set(g).collect().to_pydict()
    return {i for i, s in zip(d[ID], d[SELECTED]) if s}


def _assert_valid_mis(nodes: list[int], edges: list[tuple[int, int]], selected: set) -> None:
    adj: dict[int, set] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    # independent: no edge between two selected vertices
    for u, v in edges:
        assert not (u in selected and v in selected)
    # maximal: every unselected vertex has a selected neighbor
    for node in nodes:
        if node not in selected:
            assert any(neighbor in selected for neighbor in adj[node])


def test_path_selects_min_greedy() -> None:
    g = _graph([0, 1, 2], [(0, 1), (1, 2)])
    assert _selected(g) == {0, 2}


def test_triangle_selects_one() -> None:
    g = _graph([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
    assert _selected(g) == {0}


def test_empty_edges_selects_all() -> None:
    g = _graph([0, 1, 2], [])
    assert _selected(g) == {0, 1, 2}


def test_deterministic() -> None:
    g = _graph([0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    assert _selected(g) == _selected(g)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_valid_maximal_independent_set(seed: int) -> None:
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    while len(edges) < 22:
        u = rng.randint(0, 11)
        v = rng.randint(0, 11)
        if u != v:
            edges.add((min(u, v), max(u, v)))
    edge_list = sorted(edges)
    nodes = list(range(12))
    selected = _selected(_graph(nodes, edge_list))
    _assert_valid_mis(nodes, edge_list, selected)
