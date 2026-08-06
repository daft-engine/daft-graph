"""Validate motif find() against brute force ground truth on real datasets."""

from __future__ import annotations

from collections import defaultdict

import daft
import networkx as nx
from daft import col

from daft_graph.graph import DirectedGraph
from daft_graph.schema import DST, ID, SRC


def _bidir_karate() -> tuple[list[int], list[tuple[int, int]]]:
    g = nx.convert_node_labels_to_integers(nx.karate_club_graph())
    bidir: set[tuple[int, int]] = set()
    for u, v in g.edges():
        bidir.add((u, v))
        bidir.add((v, u))
    return sorted(g.nodes()), sorted(bidir)


def _daft_graph(nodes: list[int], edges: list[tuple[int, int]]) -> DirectedGraph:
    return DirectedGraph(
        vertices=daft.from_pydict({ID: nodes}),
        edges=daft.from_pydict({SRC: [u for u, _ in edges], DST: [v for _, v in edges]}),
    )


def _adjacency(edges: list[tuple[int, int]]) -> dict[int, set]:
    adj: dict[int, set] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
    return adj


def test_single_edge_equals_edge_set() -> None:
    nodes, edges = _bidir_karate()
    result = _daft_graph(nodes, edges).find("(a)-[e]->(b)")
    d = result.select(col("a")["id"].alias("a"), col("b")["id"].alias("b")).collect().to_pydict()
    assert set(zip(d["a"], d["b"])) == set(edges)


def test_two_paths_match_bruteforce() -> None:
    nodes, edges = _bidir_karate()
    adj = _adjacency(edges)
    result = _daft_graph(nodes, edges).find("(a)-[]->(b); (b)-[]->(c)")
    d = (
        result.select(
            col("a")["id"].alias("a"),
            col("b")["id"].alias("b"),
            col("c")["id"].alias("c"),
        )
        .collect()
        .to_pydict()
    )
    ours = set(zip(d["a"], d["b"], d["c"]))
    truth = {(a, b, c) for (a, b) in edges for c in adj[b]}
    assert ours == truth


def test_directed_triangles_match_bruteforce() -> None:
    nodes, edges = _bidir_karate()
    adj = _adjacency(edges)
    edge_set = set(edges)
    result = _daft_graph(nodes, edges).find("(a)-[]->(b); (b)-[]->(c); (c)-[]->(a)")
    d = (
        result.select(
            col("a")["id"].alias("a"),
            col("b")["id"].alias("b"),
            col("c")["id"].alias("c"),
        )
        .collect()
        .to_pydict()
    )
    ours = set(zip(d["a"], d["b"], d["c"]))
    truth = {(a, b, c) for (a, b) in edges for c in adj[b] if (c, a) in edge_set}
    assert ours == truth


def test_single_edge_scale_wiki_vote() -> None:
    import gzip
    from pathlib import Path

    data = Path(__file__).parent / "data" / "wiki-Vote.txt.gz"
    g = nx.DiGraph()
    with gzip.open(data, "rt") as handle:
        for line in handle:
            if not line.startswith("#"):
                a, b = line.split()
                g.add_edge(int(a), int(b))
    g = nx.convert_node_labels_to_integers(g)
    nodes = sorted(g.nodes())
    edges = [(u, v) for u, v in g.edges()]
    result = _daft_graph(nodes, edges).find("(a)-[e]->(b)")
    d = result.select(col("a")["id"].alias("a"), col("b")["id"].alias("b")).collect().to_pydict()
    assert set(zip(d["a"], d["b"])) == set(edges)
