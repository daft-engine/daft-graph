"""Tests for global minimum label propagation in connected_components."""

from __future__ import annotations

import daft

from daft_graph.algorithms.connected_components import connected_components
from daft_graph.graph import UndirectedGraph
from daft_graph.schema import COMPONENT, DST, ID, SRC


def _comp_map(df: daft.DataFrame) -> dict:
    d = df.collect().to_pydict()
    return dict(zip(d[ID], d[COMPONENT]))


def test_cycle_is_one_component() -> None:
    edges = daft.from_pydict({SRC: [1, 2, 3, 4], DST: [2, 3, 4, 1]})
    g = UndirectedGraph(edges)
    assert _comp_map(connected_components(g)) == {1: 1, 2: 1, 3: 1, 4: 1}


def test_joined_stars_collapse_to_global_min() -> None:
    # star at 1 (1-2, 1-3), star at 4 (4-5, 4-6), bridged by 3-6
    edges = daft.from_pydict({SRC: [1, 1, 4, 4, 3], DST: [2, 3, 5, 6, 6]})
    g = UndirectedGraph(edges)
    assert _comp_map(connected_components(g)) == {
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
    }


def test_complete_graph_min_label() -> None:
    # K4 over {2,3,4,5}: every node takes the smallest id, 2
    src = [2, 2, 2, 3, 3, 4]
    dst = [3, 4, 5, 4, 5, 5]
    g = UndirectedGraph(daft.from_pydict({SRC: src, DST: dst}))
    assert _comp_map(connected_components(g)) == {2: 2, 3: 2, 4: 2, 5: 2}


def test_distinct_components_keep_distinct_mins() -> None:
    edges = daft.from_pydict({SRC: [2, 5], DST: [3, 6]})
    g = UndirectedGraph(edges)
    assert _comp_map(connected_components(g)) == {2: 2, 3: 2, 5: 5, 6: 5}
