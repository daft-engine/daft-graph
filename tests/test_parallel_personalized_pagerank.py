"""Tests for parallel_personalized_pagerank against networkx."""

from __future__ import annotations

import random

import daft
import networkx as nx
import pytest

from daft_graph.algorithms.pagerank import (
    SOURCE,
    parallel_personalized_pagerank,
)
from daft_graph.graph import DirectedGraph
from daft_graph.schema import DST, ID, RANK, SRC


def _graph(node_ids: list[int], edges: list[tuple[int, int]]) -> DirectedGraph:
    return DirectedGraph(
        vertices=daft.from_pydict({ID: node_ids}),
        edges=daft.from_pydict({SRC: [u for u, _ in edges], DST: [v for _, v in edges]}),
    )


def _random_digraph(n_nodes: int, n_edges: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    while len(edges) < n_edges:
        u, v = rng.randint(0, n_nodes - 1), rng.randint(0, n_nodes - 1)
        if u != v:
            edges.add((u, v))
    return sorted(edges)


def test_output_columns() -> None:
    g = _graph([0, 1, 2], [(0, 1), (1, 2)])
    out = parallel_personalized_pagerank(g, [0, 2])
    assert set(out.column_names) == {ID, SOURCE, RANK}


def test_empty_sources_raises() -> None:
    g = _graph([0, 1], [(0, 1)])
    with pytest.raises(ValueError):
        parallel_personalized_pagerank(g, [])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_each_vector_matches_networkx(seed: int) -> None:
    nodes = list(range(8))
    edges = _random_digraph(8, 14, seed)
    sources = [0, 4, 7]
    g = _graph(nodes, edges)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    d = parallel_personalized_pagerank(g, sources, tol=1e-12, max_iters=300).collect().to_pydict()
    ours: dict[int, dict[int, float]] = {s: {} for s in sources}
    for node, source, rank in zip(d[ID], d[SOURCE], d[RANK]):
        ours[source][node] = rank

    for source in sources:
        theirs = nx.pagerank(graph, alpha=0.85, personalization={source: 1.0}, tol=1e-12, max_iter=1000)
        for node in nodes:
            assert abs(ours[source][node] - theirs[node]) < 1e-4
