"""Shared frontier traversal helpers for BFS based algorithms."""

from __future__ import annotations

from collections import defaultdict

import daft
from daft import DataFrame, Expression

from daft_graph.graph import Graph
from daft_graph.schema import DST, SRC


def prepare_edges(graph: Graph, *, edge_filter: Expression | None = None) -> DataFrame:
    """Project graph edges to ``(src, dst)``, applying ``edge_filter`` first.

    The filter runs before the projection so it can reference any edge attribute
    column, and the orientation is applied afterwards so an undirected graph
    symmetrizes only the edges that survived the filter.

    Args:
        graph: The graph to read edges from. Its flavor decides the orientation.
        edge_filter: Optional predicate applied to the edges before projection.

    Returns:
        A two column ``(src, dst)`` DataFrame oriented for traversal.
    """
    edges = graph.edges
    if edge_filter is not None:
        edges = edges.where(edge_filter)
    edges = edges.select(SRC, DST)
    return graph._orient(edges)


def neighbors(edges: DataFrame, sources: list[int]) -> dict[int, list[int]]:
    """Out neighbors of each source vertex, fetched with a single Daft join."""
    rows = (
        edges.join(daft.from_pydict({SRC: sources}), on=SRC, how="inner")
        .select(SRC, DST)
        .distinct()
        .collect()
        .to_pydict()
    )
    adjacency: dict[int, list[int]] = defaultdict(list)
    for s, d in zip(rows[SRC], rows[DST]):
        adjacency[int(s)].append(int(d))
    return adjacency
