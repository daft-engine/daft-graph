"""Tests for vertex id indexing (reindex / restore_ids)."""

from __future__ import annotations

import daft
import pytest

from daft_graph.algorithms.connected_components import connected_components
from daft_graph.graph import DirectedGraph, Graph, UndirectedGraph
from daft_graph.indexing import ORIGINAL, IndexedGraph, reindex, restore_ids
from daft_graph.schema import COMPONENT, DST, ID, SRC


def _string_graph() -> DirectedGraph:
    # two components: {alice, bob, carol} and {dave, eve}
    vertices = daft.from_pydict({ID: ["alice", "bob", "carol", "dave", "eve"]})
    edges = daft.from_pydict({SRC: ["alice", "bob", "dave"], DST: ["bob", "carol", "eve"]})
    return DirectedGraph(vertices=vertices, edges=edges)


def test_reindex_makes_contiguous_int_ids() -> None:
    indexed = reindex(_string_graph())
    assert isinstance(indexed, IndexedGraph)
    new_ids = sorted(indexed.graph.vertices.select(ID).collect().to_pydict()[ID])
    assert new_ids == [0, 1, 2, 3, 4]
    # mapping is sorted original order -> contiguous ids
    rows = indexed.mapping.sort(ID).collect().to_pydict()
    assert rows[ORIGINAL] == ["alice", "bob", "carol", "dave", "eve"]
    assert rows[ID] == [0, 1, 2, 3, 4]


def test_reindex_remaps_edges() -> None:
    indexed = reindex(_string_graph())
    edges = indexed.graph.edges.sort(SRC).collect().to_pydict()
    # alice=0 bob=1 carol=2 dave=3 eve=4
    assert list(zip(edges[SRC], edges[DST])) == [(0, 1), (1, 2), (3, 4)]


def test_reindex_preserves_attributes() -> None:
    vertices = daft.from_pydict({ID: ["x", "y"], "color": ["red", "blue"]})
    edges = daft.from_pydict({SRC: ["x"], DST: ["y"], "weight": [2.5]})
    indexed = reindex(DirectedGraph(vertices=vertices, edges=edges))
    assert "color" in indexed.graph.vertices.column_names
    assert "weight" in indexed.graph.edges.column_names
    erow = indexed.graph.edges.collect().to_pydict()
    assert erow["weight"] == [2.5]


def test_connected_components_on_string_ids_round_trip() -> None:
    indexed = reindex(_string_graph())
    components = connected_components(indexed.graph)
    restored = restore_ids(components, indexed.mapping, [ID, COMPONENT])
    rows = restored.collect().to_pydict()
    by_id = dict(zip(rows[ID], rows[COMPONENT]))
    # ids are back to strings, and the two components are intact
    assert set(by_id) == {"alice", "bob", "carol", "dave", "eve"}
    assert by_id["alice"] == by_id["bob"] == by_id["carol"]
    assert by_id["dave"] == by_id["eve"]
    assert by_id["alice"] != by_id["dave"]


def test_restore_ids_preserves_column_order() -> None:
    indexed = reindex(_string_graph())
    df = daft.from_pydict({ID: [0, 1], "score": [9.0, 8.0]})
    restored = restore_ids(df, indexed.mapping, [ID])
    assert restored.column_names == [ID, "score"]
    rows = restored.sort("score", desc=True).collect().to_pydict()
    assert rows[ID] == ["alice", "bob"]


def test_restore_ids_unknown_column_raises() -> None:
    indexed = reindex(_string_graph())
    df = daft.from_pydict({ID: [0]})
    with pytest.raises(ValueError, match="missing"):
        restore_ids(df, indexed.mapping, ["missing"])


def test_reindex_internal_name_collision_raises() -> None:
    vertices = daft.from_pydict({ID: ["a", "b"], "__new_id": [1, 2]})
    edges = daft.from_pydict({SRC: ["a"], DST: ["b"]})
    with pytest.raises(ValueError, match="internal names"):
        reindex(DirectedGraph(vertices=vertices, edges=edges))


def test_reindex_edge_internal_name_collision_raises() -> None:
    vertices = daft.from_pydict({ID: ["a", "b"]})
    edges = daft.from_pydict({SRC: ["a"], DST: ["b"], "__new_src": [99]})
    with pytest.raises(ValueError, match="internal names"):
        reindex(DirectedGraph(vertices=vertices, edges=edges))


def test_original_constant_is_exported() -> None:
    import daft_graph

    assert daft_graph.ORIGINAL == ORIGINAL


def test_reindex_is_deterministic() -> None:
    a = reindex(_string_graph()).mapping.sort(ID).collect().to_pydict()
    b = reindex(_string_graph()).mapping.sort(ID).collect().to_pydict()
    assert a == b


@pytest.mark.parametrize("flavor", [DirectedGraph, UndirectedGraph])
def test_reindex_preserves_the_graph_flavor(flavor: type[Graph]) -> None:
    """An algorithm typed to DirectedGraph must still accept a reindexed one."""
    edges = daft.from_pydict({SRC: ["a", "b"], DST: ["b", "c"]})
    indexed = reindex(flavor(edges))
    assert type(indexed.graph) is flavor


def test_reindexed_directed_graph_keeps_directed_methods() -> None:
    edges = daft.from_pydict({SRC: ["a", "b"], DST: ["b", "c"]})
    indexed = reindex(DirectedGraph(edges))
    assert isinstance(indexed.graph, DirectedGraph)
    # would raise AttributeError if reindex downcast to the base class
    assert indexed.graph.out_degrees().count_rows() == 2
    assert isinstance(indexed.graph.reverse(), DirectedGraph)


def test_reindexed_undirected_graph_keeps_undirected_traversal() -> None:
    edges = daft.from_pydict({SRC: ["a"], DST: ["b"]})
    indexed = reindex(UndirectedGraph(edges))
    assert isinstance(indexed.graph, UndirectedGraph)
    # symmetrized at traversal, so one stored edge walks both ways
    assert indexed.graph.edges.count_rows() == 1
    assert indexed.graph._traversal_edges().count_rows() == 2
