"""k-core decomposition on Daft DataFrames.

Computes each vertex's core number via the distributed local algorithm of
Montresor, Pellegrini, and Miorandi (2013): seed every vertex with its degree,
then repeatedly set its estimate to the h-index of its neighbors' estimates
until stable. Treats edges as undirected. Built on the message passing primitive.
"""

from __future__ import annotations

import daft
from daft import DataFrame, col, lit
from daft.functions import list_agg, when

from daft_graph.edges import canonicalize, symmetrize
from daft_graph.graph import Graph
from daft_graph.message_passing import MSG, VALUE, pregel
from daft_graph.schema import DST, ID, SRC

CORE = "core"


@daft.func.batch(return_dtype=daft.DataType.int64())
def _h_index(neighbor_cores: daft.Series) -> list[int]:
    """Largest k such that at least k neighbor estimates are >= k, per vertex."""
    out: list[int] = []
    for values in neighbor_cores.to_pylist():
        ordered = sorted(values or [], reverse=True)
        h = 0
        for i, value in enumerate(ordered, start=1):
            if value >= i:
                h = i
            else:
                break
        out.append(h)
    return out


def k_core(graph: Graph, *, max_iters: int = 100) -> DataFrame:
    """Compute the core number of each vertex (undirected).

    Accepts either graph flavor; edges are treated as undirected either way.

    Returns a DataFrame with one row per vertex: ``id`` and ``core``.
    """
    all_v = graph.vertices.select(ID).distinct()
    if graph.edges.count_rows() == 0:
        return all_v.with_column(CORE, lit(0)).select(ID, CORE)
    undirected = symmetrize(canonicalize(graph.edges))
    degree = undirected.groupby(SRC).agg(col(DST).count().alias(VALUE)).select(col(SRC).alias(ID), col(VALUE))
    init = all_v.join(degree, on=ID, how="left").with_column(
        VALUE, when(col(VALUE).is_null(), lit(0)).otherwise(col(VALUE))
    )
    final = pregel(
        undirected,
        init,
        to_src=col("dst_value"),
        agg=lambda m: list_agg(m),
        update=_h_index(col(MSG)),
        max_iters=max_iters,
    )
    return final.select(col(ID), col(VALUE).alias(CORE))
