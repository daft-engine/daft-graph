"""End to end regression tests on real graph datasets.

Validates every algorithm against networkx as ground truth on the canonical
Zachary Karate Club and Les Miserables graphs (both shipped by networkx, so no
download is needed and the test is fully reproducible).
"""

from __future__ import annotations

from collections import defaultdict

import daft
import networkx as nx
import pytest
from daft import lit

from daft_graph import (
    DirectedGraph,
    aggregate_messages,
    bfs,
    connected_components,
    k_core,
    label_propagation,
    pagerank,
    shortest_paths,
    strongly_connected_components,
    triangle_count,
)
from daft_graph.algorithms.k_core import CORE
from daft_graph.algorithms.shortest_paths import DISTANCE, LANDMARK
from daft_graph.algorithms.triangle_count import TRIANGLE_COUNT
from daft_graph.message_passing import MSG
from daft_graph.schema import COMPONENT, DST, ID, RANK, SRC

DATASETS = [
    ("karate", nx.karate_club_graph),
    ("lesmis", nx.les_miserables_graph),
]


def _undirected(factory) -> tuple[list[int], list[tuple[int, int]]]:
    g = nx.convert_node_labels_to_integers(factory())
    nodes = sorted(g.nodes())
    edges = sorted({(min(u, v), max(u, v)) for u, v in g.edges() if u != v})
    return nodes, edges


def _daft_graph(nodes: list[int], edges: list[tuple[int, int]]) -> DirectedGraph:
    vertices = daft.from_pydict({ID: nodes})
    edges_df = daft.from_pydict({SRC: [u for u, _ in edges], DST: [v for _, v in edges]})
    return DirectedGraph(vertices=vertices, edges=edges_df)


def _bidir(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return edges + [(v, u) for u, v in edges]


def _nx_undirected(nodes: list[int], edges: list[tuple[int, int]]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def _nx_digraph(nodes: list[int], edges: list[tuple[int, int]]) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def _partition(df: daft.DataFrame) -> set:
    d = df.collect().to_pydict()
    groups: dict[int, set] = defaultdict(set)
    for node, comp in zip(d[ID], d[COMPONENT]):
        groups[comp].add(node)
    return {frozenset(members) for members in groups.values()}


@pytest.mark.parametrize("name,factory", DATASETS)
@pytest.mark.parametrize("strategy", ["local", "distributed"])
def test_connected_components(name: str, factory, strategy: str) -> None:
    nodes, edges = _undirected(factory)
    ours = _partition(connected_components(_daft_graph(nodes, edges), strategy=strategy))
    theirs = {frozenset(c) for c in nx.connected_components(_nx_undirected(nodes, edges))}
    assert ours == theirs


@pytest.mark.parametrize("name,factory", DATASETS)
def test_triangle_count(name: str, factory) -> None:
    nodes, edges = _undirected(factory)
    d = triangle_count(_daft_graph(nodes, edges)).collect().to_pydict()
    ours = dict(zip(d[ID], d[TRIANGLE_COUNT]))
    assert ours == nx.triangles(_nx_undirected(nodes, edges))


@pytest.mark.parametrize("name,factory", DATASETS)
def test_k_core(name: str, factory) -> None:
    nodes, edges = _undirected(factory)
    d = k_core(_daft_graph(nodes, edges)).collect().to_pydict()
    ours = dict(zip(d[ID], d[CORE]))
    assert ours == nx.core_number(_nx_undirected(nodes, edges))


@pytest.mark.parametrize("name,factory", DATASETS)
def test_pagerank(name: str, factory) -> None:
    nodes, edges = _undirected(factory)
    bidir = _bidir(edges)
    d = pagerank(_daft_graph(nodes, bidir), tol=1e-12, max_iters=500).collect().to_pydict()
    ours = dict(zip(d[ID], d[RANK]))
    theirs = nx.pagerank(_nx_digraph(nodes, bidir), alpha=0.85, tol=1e-12, max_iter=1000)
    for node in nodes:
        assert abs(ours[node] - theirs[node]) < 1e-4


@pytest.mark.parametrize("name,factory", DATASETS)
@pytest.mark.parametrize("strategy", ["local", "distributed"])
def test_scc_symmetric_is_one_component(name: str, factory, strategy: str) -> None:
    nodes, edges = _undirected(factory)
    bidir = _bidir(edges)
    ours = _partition(strongly_connected_components(_daft_graph(nodes, bidir), strategy=strategy))
    theirs = {frozenset(c) for c in nx.strongly_connected_components(_nx_digraph(nodes, bidir))}
    assert ours == theirs


@pytest.mark.parametrize("strategy", ["local", "distributed"])
def test_scc_dag_orientation_all_singletons(strategy: str) -> None:
    # Orienting each edge low -> high yields a DAG: every vertex its own SCC.
    nodes, edges = _undirected(nx.karate_club_graph)
    ours = _partition(strongly_connected_components(_daft_graph(nodes, edges), strategy=strategy))
    theirs = {frozenset(c) for c in nx.strongly_connected_components(_nx_digraph(nodes, edges))}
    assert ours == theirs
    assert len(ours) == len(nodes)


@pytest.mark.parametrize("name,factory", DATASETS)
def test_bfs_and_shortest_paths(name: str, factory) -> None:
    nodes, edges = _undirected(factory)
    bidir = _bidir(edges)
    g = _daft_graph(nodes, bidir)
    graph = _nx_digraph(nodes, bidir)
    target = nodes[-1]

    path = bfs(g, nodes[0], target, max_path_length=len(nodes))
    assert path is not None
    assert len(path) - 1 == nx.shortest_path_length(graph, nodes[0], target)

    d = shortest_paths(g, [nodes[0]], max_iters=len(nodes)).collect().to_pydict()
    ours = {i: dist for i, lm, dist in zip(d[ID], d[LANDMARK], d[DISTANCE])}
    theirs = dict(nx.single_target_shortest_path_length(graph, nodes[0]))
    assert ours == theirs


def test_label_propagation_is_valid_and_deterministic() -> None:
    nodes, edges = _undirected(nx.karate_club_graph)
    g = _daft_graph(nodes, edges)
    first = label_propagation(g).collect().to_pydict()
    second = label_propagation(g).collect().to_pydict()
    labels = dict(zip(first[ID], first["label"]))
    assert set(labels) == set(nodes)  # every vertex labelled
    assert dict(zip(second[ID], second["label"])) == labels  # deterministic


def test_aggregate_messages_recovers_degree() -> None:
    nodes, edges = _undirected(nx.karate_club_graph)
    bidir = _bidir(edges)
    edges_df = daft.from_pydict({SRC: [u for u, _ in bidir], DST: [v for _, v in bidir]})
    state = daft.from_pydict({ID: nodes})
    d = aggregate_messages(edges_df, state, to_dst=lit(1), agg=lambda m: m.sum()).collect().to_pydict()
    ours = dict(zip(d[ID], d[MSG]))
    assert ours == dict(_nx_undirected(nodes, edges).degree())
