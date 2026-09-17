"""Tests for bfs_paths (GraphFrames style breadth first search)."""

from __future__ import annotations

import random

import daft
import networkx as nx
import pytest
from daft import col

from daft_graph.algorithms.bfs import bfs_paths
from daft_graph.graph import DirectedGraph
from daft_graph.schema import DST, ID, SRC


def _graph(
    node_ids: list[int],
    edges: list[tuple[int, int]],
    vattrs: dict | None = None,
    eattrs: dict | None = None,
) -> DirectedGraph:
    vcols: dict = {ID: node_ids}
    if vattrs:
        vcols.update(vattrs)
    ecols: dict = {SRC: [u for u, _ in edges], DST: [v for _, v in edges]}
    if eattrs:
        ecols.update(eattrs)
    return DirectedGraph(vertices=daft.from_pydict(vcols), edges=daft.from_pydict(ecols))


def _id_paths(result: daft.DataFrame) -> list[tuple[int, ...]]:
    """Extract the vertex id sequence of every path row, sorted."""
    cols = set(result.column_names)
    ordered = ["from"]
    i = 1
    while f"v{i}" in cols:
        ordered.append(f"v{i}")
        i += 1
    if "to" in cols:
        ordered.append("to")
    sel = result.select(*[col(c)["id"].alias(c) for c in ordered]).collect().to_pydict()
    n = len(sel[ordered[0]])
    return sorted(tuple(int(sel[c][r]) for c in ordered) for r in range(n))


def test_chain_single_path() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    result = bfs_paths(g, col(ID) == 0, col(ID) == 3)
    assert result.column_names == ["from", "e0", "v1", "e1", "v2", "e2", "to"]
    assert _id_paths(result) == [(0, 1, 2, 3)]


def test_diamond_returns_all_shortest() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (0, 2), (1, 3), (2, 3)])
    result = bfs_paths(g, col(ID) == 0, col(ID) == 3)
    assert _id_paths(result) == [(0, 1, 3), (0, 2, 3)]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_matches_networkx(seed: int) -> None:
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    while len(edges) < 25:
        u, v = rng.randint(0, 11), rng.randint(0, 11)
        if u != v:
            edges.add((u, v))
    edge_list = sorted(edges)
    nodes = list(range(12))
    g = _graph(nodes, edge_list)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edge_list)

    for s, t in [(0, 5), (1, 9), (3, 11), (2, 7)]:
        result = bfs_paths(g, col(ID) == s, col(ID) == t)
        if nx.has_path(graph, s, t):
            expected = sorted(tuple(p) for p in nx.all_shortest_paths(graph, s, t))
            assert _id_paths(result) == expected
        else:
            assert result.count_rows() == 0


def test_multi_source_multi_target() -> None:
    g = _graph(list(range(6)), [(0, 1), (1, 2), (3, 4), (4, 5)])
    result = bfs_paths(g, (col(ID) == 0) | (col(ID) == 3), (col(ID) == 2) | (col(ID) == 5))
    assert _id_paths(result) == [(0, 1, 2), (3, 4, 5)]


def test_returns_only_nearest_target() -> None:
    g = _graph(list(range(5)), [(0, 1), (1, 2), (2, 3), (3, 4)])
    # targets {2, 4}: 2 is nearer (dist 2) than 4 (dist 4), so only the path to 2
    result = bfs_paths(g, col(ID) == 0, (col(ID) == 2) | (col(ID) == 4))
    assert _id_paths(result) == [(0, 1, 2)]


def test_no_path_returns_empty() -> None:
    g = _graph([0, 1, 2, 3], [(0, 1), (2, 3)])
    assert bfs_paths(g, col(ID) == 0, col(ID) == 3).count_rows() == 0


def test_source_equals_target() -> None:
    g = _graph([0, 1, 2], [(0, 1), (1, 2)])
    result = bfs_paths(g, col(ID) == 0, col(ID) == 0)
    assert set(result.column_names) == {"from", "to"}
    rows = result.select(col("from")["id"].alias("f"), col("to")["id"].alias("t")).collect().to_pydict()
    assert rows["f"] == [0]
    assert rows["t"] == [0]


def test_mixed_overlap_returns_only_zero_hop() -> None:
    # sources {0, 1}, targets {0, 2}: 0 is in both, so the global shortest length
    # is 0 and only the zero-hop overlap is returned; the 1 -> 2 path is dropped
    g = _graph([0, 1, 2], [(1, 2)])
    result = bfs_paths(g, (col(ID) == 0) | (col(ID) == 1), (col(ID) == 0) | (col(ID) == 2))
    assert set(result.column_names) == {"from", "to"}
    rows = result.select(col("from")["id"].alias("f"), col("to")["id"].alias("t")).collect().to_pydict()
    assert rows["f"] == [0]
    assert rows["t"] == [0]


def test_edge_filter_changes_path() -> None:
    g = _graph([0, 1, 2], [(0, 1), (1, 2), (0, 2)], eattrs={"type": ["a", "a", "b"]})
    assert _id_paths(bfs_paths(g, col(ID) == 0, col(ID) == 2)) == [(0, 2)]
    filtered = bfs_paths(g, col(ID) == 0, col(ID) == 2, edge_filter=col("type") == "a")
    assert _id_paths(filtered) == [(0, 1, 2)]


def test_undirected_search() -> None:
    g = _graph([0, 1, 2], [(1, 0), (2, 1)])
    assert bfs_paths(g, col(ID) == 0, col(ID) == 2).count_rows() == 0
    undirected = bfs_paths(g.as_undirected(), col(ID) == 0, col(ID) == 2)
    assert _id_paths(undirected) == [(0, 1, 2)]


def test_structs_carry_attributes() -> None:
    g = _graph([0, 1], [(0, 1)], vattrs={"name": ["a", "b"]}, eattrs={"w": [1.5]})
    result = bfs_paths(g, col(ID) == 0, col(ID) == 1)
    rows = (
        result.select(
            col("from")["name"].alias("fn"),
            col("to")["name"].alias("tn"),
            col("e0")["w"].alias("w"),
        )
        .collect()
        .to_pydict()
    )
    assert rows["fn"] == ["a"]
    assert rows["tn"] == ["b"]
    assert rows["w"] == [1.5]


def test_graph_method_matches_function() -> None:
    g = _graph([0, 1, 2], [(0, 1), (1, 2)])
    from_function = _id_paths(bfs_paths(g, col(ID) == 0, col(ID) == 2))
    from_method = _id_paths(g.bfs_paths(col(ID) == 0, col(ID) == 2))
    assert from_function == from_method == [(0, 1, 2)]


def test_max_paths_guard() -> None:
    g = _graph(list(range(5)), [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)])
    with pytest.raises(ValueError, match="max_paths"):
        bfs_paths(g, col(ID) == 0, col(ID) == 4, max_paths=1)
