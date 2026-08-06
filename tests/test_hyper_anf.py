"""Tests for hyper_anf (approximate neighborhood function)."""

from __future__ import annotations

import itertools
import random

import daft
import networkx as nx
import pytest

from daft_graph.algorithms.hyper_anf import APPROX_COUNT, HOP, _hash64, hyper_anf
from daft_graph.graph import DirectedGraph
from daft_graph.schema import DST, ID, SRC


def _graph(node_ids: list[int], edges: list[tuple[int, int]]) -> DirectedGraph:
    return DirectedGraph(
        vertices=daft.from_pydict({ID: node_ids}),
        edges=daft.from_pydict({SRC: [u for u, _ in edges], DST: [v for _, v in edges]}),
    )


def _counts(g: DirectedGraph, max_hops: int) -> dict:
    d = hyper_anf(g, max_hops=max_hops).collect().to_pydict()
    return {(i, h): c for i, h, c in zip(d[ID], d[HOP], d[APPROX_COUNT])}


def test_hop0_is_one() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    counts = _counts(g, 0)
    for node in [0, 1, 2, 3]:
        assert abs(counts[(node, 0)] - 1.0) < 0.5


def test_monotonic_over_hops() -> None:
    g = _graph([0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 3), (3, 4)])
    counts = _counts(g, 4)
    for node in range(5):
        series = [counts[(node, h)] for h in range(5)]
        assert all(b >= a - 1e-9 for a, b in itertools.pairwise(series))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_approximates_reachable(seed: int) -> None:
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    while len(edges) < 18:
        u, v = rng.randint(0, 9), rng.randint(0, 9)
        if u != v:
            edges.add((u, v))
    edge_list = sorted(edges)
    nodes = list(range(10))
    g = _graph(nodes, edge_list)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edge_list)

    max_hops = 6
    counts = _counts(g, max_hops)
    for node in nodes:
        for hop in range(max_hops + 1):
            exact = len(nx.single_source_shortest_path_length(graph, node, cutoff=hop))
            est = counts[(node, hop)]
            assert abs(est - exact) <= 0.3 * exact + 1.5


@pytest.mark.parametrize("precision", [0, 3, 19, 64])
def test_precision_out_of_range_raises(precision: int) -> None:
    g = _graph([0, 1], [(0, 1)])
    with pytest.raises(ValueError, match="precision"):
        hyper_anf(g, precision=precision)


def test_hash64_handles_full_int64_and_uint64_range() -> None:
    # must not raise for negatives or values past the signed-64 boundary
    for value in [-(2**63), -1, 0, 2**63 - 1, 2**63, 2**64 - 1]:
        digest = _hash64(value)
        assert 0 <= digest < 2**64
