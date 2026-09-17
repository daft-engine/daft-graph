"""Regression tests on the SNAP Wikipedia vote network (a larger directed graph).

Wiki-Vote (https://snap.stanford.edu/data/wiki-Vote.html): 7115 nodes, 103689
directed edges, ~5800 strongly connected components. The dataset is vendored
under tests/data/ so the gate runs offline and is fully reproducible. Every
algorithm is validated against networkx as ground truth.

triangle_count and distributed strongly_connected_components are intentionally
exercised at smaller scale (test_datasets.py) rather than here: at 100k edges the
triangle self join and the many component coloring loop are the heavy paths, and
``auto`` correctly selects the local solver for SCC at this size.
"""

from __future__ import annotations

import gzip
from collections import defaultdict
from pathlib import Path

import daft
import networkx as nx

from daft_graph import (
    DirectedGraph,
    bfs,
    connected_components,
    k_core,
    pagerank,
    shortest_paths,
    strongly_connected_components,
)
from daft_graph.algorithms.k_core import CORE
from daft_graph.algorithms.shortest_paths import DISTANCE, LANDMARK
from daft_graph.schema import COMPONENT, DST, ID, RANK, SRC

_DATA = Path(__file__).parent / "data" / "wiki-Vote.txt.gz"


def _load() -> nx.DiGraph:
    g: nx.DiGraph = nx.DiGraph()
    with gzip.open(_DATA, "rt") as handle:
        for line in handle:
            if not line.startswith("#"):
                a, b = line.split()
                g.add_edge(int(a), int(b))
    return nx.convert_node_labels_to_integers(g)


NXG = _load()
NODES = sorted(NXG.nodes())
EDGES = [(u, v) for u, v in NXG.edges()]


def _daft_graph() -> DirectedGraph:
    return DirectedGraph(
        vertices=daft.from_pydict({ID: NODES}),
        edges=daft.from_pydict({SRC: [u for u, _ in EDGES], DST: [v for _, v in EDGES]}),
    )


def _partition(df: daft.DataFrame) -> set:
    d = df.collect().to_pydict()
    groups: dict[int, set] = defaultdict(set)
    for node, comp in zip(d[ID], d[COMPONENT]):
        groups[comp].add(node)
    return {frozenset(members) for members in groups.values()}


def test_dataset_shape() -> None:
    assert len(NODES) == 7115
    assert len(EDGES) == 103689


def test_weakly_connected_components_local() -> None:
    ours = _partition(connected_components(_daft_graph(), strategy="local"))
    theirs = {frozenset(c) for c in nx.weakly_connected_components(NXG)}
    assert ours == theirs


def test_weakly_connected_components_distributed() -> None:
    ours = _partition(connected_components(_daft_graph(), strategy="distributed"))
    theirs = {frozenset(c) for c in nx.weakly_connected_components(NXG)}
    assert ours == theirs


def test_strongly_connected_components() -> None:
    ours = _partition(strongly_connected_components(_daft_graph()))
    theirs = {frozenset(c) for c in nx.strongly_connected_components(NXG)}
    assert ours == theirs


def test_pagerank_matches_networkx() -> None:
    d = pagerank(_daft_graph(), tol=1e-10, max_iters=200).collect().to_pydict()
    ours = dict(zip(d[ID], d[RANK]))
    theirs = nx.pagerank(NXG, alpha=0.85, tol=1e-12, max_iter=1000)
    for node in NODES:
        assert abs(ours[node] - theirs[node]) < 1e-6


def test_k_core_matches_networkx() -> None:
    d = k_core(_daft_graph(), max_iters=200).collect().to_pydict()
    ours = dict(zip(d[ID], d[CORE]))
    undirected = nx.Graph()
    undirected.add_nodes_from(NODES)
    undirected.add_edges_from((u, v) for u, v in EDGES if u != v)
    assert ours == nx.core_number(undirected)


def test_shortest_paths_matches_networkx() -> None:
    landmark = next(n for n in NODES if NXG.in_degree(n) > 0)
    d = shortest_paths(_daft_graph(), [landmark], max_iters=50).collect().to_pydict()
    ours = {i: dist for i, lm, dist in zip(d[ID], d[LANDMARK], d[DISTANCE])}
    theirs = dict(nx.single_target_shortest_path_length(NXG, landmark))
    assert ours == theirs


def test_bfs_matches_networkx() -> None:
    source = next(n for n in NODES if NXG.out_degree(n) > 0)
    lengths = dict(nx.single_source_shortest_path_length(NXG, source))
    target = max(lengths, key=lengths.get)
    path = bfs(_daft_graph(), source, target, max_path_length=50)
    assert path is not None
    assert len(path) - 1 == lengths[target]
