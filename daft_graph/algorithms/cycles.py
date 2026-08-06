"""Cycle detection on Daft DataFrames.

Reports the vertices that lie on a directed cycle, and hence whether the graph
is acyclic. A vertex is on a cycle iff it belongs to a strongly connected
component with more than one vertex, or it has a self loop. Full cycle
enumeration (listing every cycle as a path) is intentionally not provided: the
number of cycles can be exponential. Use ``strongly_connected_components`` for
the component structure.
"""

from __future__ import annotations

from daft import DataFrame, col, lit

from daft_graph.algorithms.strongly_connected_components import (
    strongly_connected_components,
)
from daft_graph.graph import DirectedGraph
from daft_graph.schema import COMPONENT, DST, ID, SRC

_SIZE = "_size"


def vertices_on_cycles(graph: DirectedGraph) -> DataFrame:
    """Return the ids of vertices that lie on at least one directed cycle.

    Takes a :class:`DirectedGraph`; an undirected graph has no directed cycles.

    Returns a DataFrame with a single ``id`` column.
    """
    scc = strongly_connected_components(graph)
    big = scc.groupby(COMPONENT).agg(col(ID).count().alias(_SIZE)).where(col(_SIZE) >= lit(2)).select(COMPONENT)
    in_cycle = scc.join(big, on=COMPONENT, how="semi").select(ID)
    self_loops = graph.edges.where(col(SRC) == col(DST)).select(col(SRC).alias(ID)).distinct()
    return in_cycle.union_all(self_loops).distinct()


def has_cycle(graph: DirectedGraph) -> bool:
    """True if the graph contains at least one directed cycle."""
    return vertices_on_cycles(graph).count_rows() > 0
