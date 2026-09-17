"""Tests for labeled property graph support (typed degree)."""

from __future__ import annotations

import random
from collections import defaultdict

import daft

from daft_graph.graph import DirectedGraph
from daft_graph.schema import DST, ID, SRC


def _typed_graph(node_ids: list[int], edges: list[tuple[int, int, str]]) -> DirectedGraph:
    return DirectedGraph(
        vertices=daft.from_pydict({ID: node_ids}),
        edges=daft.from_pydict(
            {
                SRC: [u for u, _, _ in edges],
                DST: [v for _, v, _ in edges],
                "type": [t for _, _, t in edges],
            }
        ),
    )


def _degree_map(g: DirectedGraph) -> dict:
    d = g.degree_by_type("type").collect().to_pydict()
    return {(i, t): deg for i, t, deg in zip(d[ID], d["type"], d["degree"])}


def test_degree_by_type_counts() -> None:
    g = _typed_graph([0, 1, 2, 3], [(0, 1, "a"), (0, 2, "a"), (0, 3, "b"), (1, 2, "b")])
    assert _degree_map(g) == {
        (0, "a"): 2,
        (1, "a"): 1,
        (2, "a"): 1,
        (0, "b"): 1,
        (3, "b"): 1,
        (1, "b"): 1,
        (2, "b"): 1,
    }


def test_degree_by_type_matches_manual_count() -> None:
    rng = random.Random(7)
    types = ["x", "y", "z"]
    edges = []
    seen: set[tuple[int, int]] = set()
    while len(edges) < 30:
        u, v = rng.randint(0, 9), rng.randint(0, 9)
        if u != v and (u, v) not in seen:
            seen.add((u, v))
            edges.append((u, v, rng.choice(types)))
    g = _typed_graph(list(range(10)), edges)
    expected: dict[tuple[int, str], int] = defaultdict(int)
    for u, v, t in edges:
        expected[(u, t)] += 1
        expected[(v, t)] += 1
    assert _degree_map(g) == dict(expected)
