"""The BFS family must keep its frontier inside Daft.

`bfs`, `all_shortest_paths`, and `bfs_paths` used to pull the current frontier's
whole adjacency into the driver on every hop, which made them fail on ordinary
large connected graphs. The search now runs as Daft joins, so what crosses into
the driver scales with the answer rather than with the graph. These tests pin that
property by counting the rows that actually cross.
"""

from __future__ import annotations

import daft
import pytest
from daft import DataFrame, col

from daft_graph import DirectedGraph, all_shortest_paths, bfs, bfs_paths
from daft_graph.schema import DST, ID, SRC


class _PullCounter:
    """Count rows crossing into the driver via `DataFrame.to_pydict`."""

    def __init__(self) -> None:
        self.rows = 0

    def __enter__(self) -> _PullCounter:
        self._orig = DataFrame.to_pydict
        counter = self

        def spy(df: DataFrame) -> dict:
            result = counter._orig(df)
            counter.rows += max((len(v) for v in result.values()), default=0)
            return result

        DataFrame.to_pydict = spy  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: object) -> None:
        DataFrame.to_pydict = self._orig  # type: ignore[method-assign]


def _wide_graph(width: int) -> DirectedGraph:
    """A hub whose frontier is `width` wide after one hop, then one more layer."""
    src: list[int] = []
    dst: list[int] = []
    for i in range(1, width + 1):
        src.append(0)
        dst.append(i)
    for i in range(1, width + 1):
        src.append(i)
        dst.append(1000 + i)
    return DirectedGraph(daft.from_pydict({SRC: src, DST: dst}))


@pytest.mark.parametrize("width", [50, 200])
def test_bfs_driver_pull_does_not_grow_with_the_frontier(width: int) -> None:
    """The rows pulled must not scale with frontier width."""
    graph = _wide_graph(width)
    with _PullCounter() as counter:
        path = bfs(graph, 0, 1000 + width // 2)
    assert path == [0, width // 2, 1000 + width // 2]
    # A path of 3 vertices needs a handful of single row lookups. The pre-fix
    # implementation pulled 2 * width rows here, so anything near the frontier
    # width means the search went back to collecting adjacency per hop.
    assert counter.rows <= 20, f"pulled {counter.rows} rows for a 3 vertex path"


def test_all_shortest_paths_driver_pull_is_bounded_by_the_answer() -> None:
    graph = _wide_graph(200)
    with _PullCounter() as counter:
        paths = all_shortest_paths(graph, 0, 1100)
    assert paths == [[0, 100, 1100]]
    assert counter.rows <= 20, f"pulled {counter.rows} rows for one 3 vertex path"


def test_bfs_paths_does_not_collect_the_whole_source_set() -> None:
    """A broad from_filter must not be materialized into the driver."""
    width = 200
    graph = _wide_graph(width)
    # `col(ID) >= 0` matches every vertex, so a source-set collect would pull
    # them all; the search only needs them as a Daft frame.
    with _PullCounter() as counter:
        result = bfs_paths(graph, col(ID) >= 0, col(ID) == 1100, max_path_length=3)
    assert result.count_rows() >= 1
    assert counter.rows <= 60, f"pulled {counter.rows} rows for a broad source filter"


def test_rewrite_preserves_bfs_tie_break_determinism() -> None:
    """Two equal length paths: the smaller predecessor must win, every time."""
    graph = DirectedGraph(daft.from_pydict({SRC: [0, 0, 5, 2], DST: [5, 2, 9, 9]}))
    # 0->2->9 and 0->5->9 are both length 2; predecessor 2 < 5 wins.
    assert bfs(graph, 0, 9) == [0, 2, 9]
    assert [bfs(graph, 0, 9) for _ in range(3)] == [[0, 2, 9]] * 3


def test_all_shortest_paths_still_returns_every_tie() -> None:
    graph = DirectedGraph(daft.from_pydict({SRC: [0, 0, 5, 2], DST: [5, 2, 9, 9]}))
    assert all_shortest_paths(graph, 0, 9) == [[0, 2, 9], [0, 5, 9]]
