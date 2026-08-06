"""Community detection via synchronous label propagation (LPA).

Built on the message passing primitive: each round every vertex collects its
neighbors' labels and adopts the most frequent one, breaking ties toward the
smallest label. Updates are synchronous and bounded by ``max_iters`` since LPA
can oscillate on bipartite structures, the same as the GraphFrames LPA.
"""

from __future__ import annotations

from collections import Counter

import daft
from daft import DataFrame, col
from daft.functions import list_agg

from daft_graph.edges import symmetrize
from daft_graph.graph import Graph
from daft_graph.message_passing import MSG, VALUE, pregel
from daft_graph.schema import ID, LABEL


@daft.func.batch(return_dtype=daft.DataType.int64())
def _plurality(neighbor_labels: daft.Series, own: daft.Series) -> list[int]:
    """Most frequent neighbor label per vertex (min on ties); own if no neighbors."""
    out: list[int] = []
    owns = own.to_pylist()
    for labels, current in zip(neighbor_labels.to_pylist(), owns):
        if not labels:
            out.append(current)
        else:
            counts = Counter(labels)
            best = max(counts.values())
            out.append(min(label for label, count in counts.items() if count == best))
    return out


def label_propagation(
    graph: Graph,
    *,
    max_iters: int = 10,
    materialize_every: int = 1,
    checkpoint_dir: str | None = None,
) -> DataFrame:
    """Assign a community label to each vertex via synchronous LPA.

    Args:
        graph: The graph to analyze. Accepts either graph flavor; edges are treated as undirected either way.
        max_iters: Maximum propagation rounds. LPA may oscillate, so this caps it.
        materialize_every: How often to truncate the Daft plan between rounds.
        checkpoint_dir: Optional parquet checkpoint directory for long runs.

    Returns:
        A DataFrame with one row per vertex: ``id`` and its ``label``.
    """
    base = graph.vertices.select(col(ID), col(ID).alias(VALUE)).distinct()
    if graph.edges.count_rows() == 0:
        return base.select(col(ID), col(VALUE).alias(LABEL))
    undirected = symmetrize(graph.edges)
    final = pregel(
        undirected,
        base,
        to_src=col("dst_value"),
        agg=lambda m: list_agg(m),
        update=_plurality(col(MSG), col(VALUE)),
        max_iters=max_iters,
        materialize_every=materialize_every,
        checkpoint_dir=checkpoint_dir,
    )
    return final.select(col(ID), col(VALUE).alias(LABEL))
