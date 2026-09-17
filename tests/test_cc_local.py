"""Tests for the local solve and strategy routing in connected_components."""

from __future__ import annotations

import random
from collections import defaultdict

import daft
import igraph as ig
import pytest

from daft_graph.algorithms.connected_components import (
    _resolve_strategy,
    connected_components,
)
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


def _graph(node_ids: list[int], edges: list[tuple[int, int]]) -> UndirectedGraph:
    vertices = daft.from_pydict({ID: node_ids})
    edges_df = daft.from_pydict({SRC: [u for u, _ in edges], DST: [v for _, v in edges]})
    return UndirectedGraph(vertices=vertices, edges=edges_df)


def _partition(df: daft.DataFrame) -> set:
    d = df.collect().to_pydict()
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
    [(8, 12, 1), (20, 25, 2), (40, 55, 3), (30, 8, 4)],
)
def test_local_matches_igraph(n_nodes: int, n_edges: int, seed: int) -> None:
    node_ids = list(range(n_nodes))
    edges = _random_edges(n_nodes, n_edges, seed)
    g = _graph(node_ids, edges)
    assert _partition(connected_components(g, strategy="local")) == _igraph_partition(node_ids, edges)


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_local_matches_distributed(seed: int) -> None:
    node_ids = list(range(25))
    edges = _random_edges(25, 30, seed)
    g = _graph(node_ids, edges)
    local = _partition(connected_components(g, strategy="local"))
    distributed = _partition(connected_components(g, strategy="distributed"))
    assert local == distributed


def test_resolve_strategy_routing() -> None:
    assert _resolve_strategy("local", 10**9, 5) == "local"
    assert _resolve_strategy("distributed", 0, 5) == "distributed"
    assert _resolve_strategy("auto", 3, 5) == "local"
    assert _resolve_strategy("auto", 10, 5) == "distributed"
    with pytest.raises(ValueError):
        _resolve_strategy("bogus", 0, 5)


def test_auto_routes_both_ways_to_same_result() -> None:
    node_ids = list(range(15))
    edges = _random_edges(15, 20, 7)
    g = _graph(node_ids, edges)
    via_distributed = _partition(connected_components(g, strategy="auto", local_threshold=0))
    via_local = _partition(connected_components(g, strategy="auto", local_threshold=10**9))
    assert via_distributed == via_local == _igraph_partition(node_ids, edges)


def test_auto_falls_back_to_distributed_without_the_local_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A core only install must still answer the default strategy=auto call."""
    import daft_graph.algorithms.connected_components as cc

    monkeypatch.setattr(cc, "has_local_extra", lambda: False)
    # small graph, so the edge count alone would have selected the local solve
    assert cc._resolve_strategy("auto", num_edges=1, local_threshold=1_000) == "distributed"


def test_auto_uses_local_when_the_extra_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import daft_graph.algorithms.connected_components as cc

    monkeypatch.setattr(cc, "has_local_extra", lambda: True)
    assert cc._resolve_strategy("auto", num_edges=1, local_threshold=1_000) == "local"


def test_explicit_local_is_still_honored_so_it_can_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking for local by name must not be silently downgraded."""
    import daft_graph.algorithms.connected_components as cc

    monkeypatch.setattr(cc, "has_local_extra", lambda: False)
    assert cc._resolve_strategy("local", num_edges=1, local_threshold=1_000) == "local"
