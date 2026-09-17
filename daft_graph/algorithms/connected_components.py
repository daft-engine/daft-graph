"""Connected components via large star and small star contraction.

Ports the star contraction recipe from Kiveris et al., "Connected Components in
MapReduce and Beyond" (SoCC 2014), the same algorithm Spark GraphFrames uses by
default. Large star and small star passes alternate until the edge set stops
changing, then a global minimum label propagation pass guarantees every node in
a component adopts the single smallest id (exact parity with igraph).

Two strategies are exposed. ``distributed`` runs the star contraction on Daft
and scales to the full edge set. ``local`` collects the (typically already
reduced) edge set and finishes with ``scipy.sparse.csgraph`` on one node.
``auto`` picks local below an edge count threshold and distributed above it, and
falls back to distributed when the optional ``local`` extra is absent.
"""

from __future__ import annotations

import os
from typing import Literal

import daft
from daft import DataFrame, col
from daft.functions import list_agg, list_min, when

from daft_graph._compare import rows_equal
from daft_graph._optional import has_local_extra, require_numpy, require_scipy
from daft_graph.edges import canonicalize, symmetrize
from daft_graph.graph import Graph
from daft_graph.iterate import bound_partitions, collect_bounded, iterate_to_fixed_point
from daft_graph.schema import COMPONENT, DST, ID, SRC, Strategy

_NBRS = "nbrs"
_M = "m"
_NBR = "_nbr"
_NBR_MIN = "_nbr_min"
_DEFAULT_LOCAL_THRESHOLD = 2_000_000


def _point_to_min(neighborhood: DataFrame) -> DataFrame:
    """Set ``m = min(src, min(neighbors))`` for each grouped neighborhood."""
    with_min = neighborhood.with_column(_M, list_min(col(_NBRS)))
    # The outer null guard covers groups whose neighbor list is empty after
    # list_min; otherwise m is the smaller of src and the neighbor minimum.
    return with_min.with_column(
        _M,
        when(col(_M).is_null(), col(SRC)).otherwise(when(col(SRC) < col(_M), col(SRC)).otherwise(col(_M))),
    )


def large_star(edges: DataFrame) -> DataFrame:
    """Large star pass: point every node to the min over its full neighborhood.

    Emits edges ``(neighbor, m)`` only where ``neighbor > src``, pulling higher
    id nodes toward low id hubs.
    """
    undirected = symmetrize(edges)
    neighborhood = undirected.groupby(SRC).agg(list_agg(col(DST)).alias(_NBRS))
    pointed = _point_to_min(neighborhood)
    return (
        pointed.explode(_NBRS)
        .where(col(_NBRS) > col(SRC))
        .select(col(_NBRS).alias(SRC), col(_M).alias(DST))
        .where(col(SRC) != col(DST))
        .distinct()
    )


def small_star(edges: DataFrame) -> DataFrame:
    """Small star pass: orient edges so ``src < dst``, then point neighbors to the min.

    Merges the local minima discovered by :func:`large_star`.
    """
    oriented = canonicalize(edges)
    neighborhood = oriented.groupby(SRC).agg(list_agg(col(DST)).alias(_NBRS))
    pointed = _point_to_min(neighborhood)
    return (
        pointed.explode(_NBRS).select(col(_NBRS).alias(SRC), col(_M).alias(DST)).where(col(SRC) != col(DST)).distinct()
    )


def _star_step(edges: DataFrame) -> DataFrame:
    """One alternating round: large star followed by small star.

    The large star result is partition capped before the small star pass reads
    it, so the two passes' shuffles do not stack into an ever growing grid (each
    pass symmetrizes or unions, and Daft's shuffles inherit their input's
    partition count). The cap is a plan rewrite, not a materialization, so it
    adds no distributed round trip per round.
    """
    return small_star(bound_partitions(large_star(edges)))


def _canonical_equal(prev: DataFrame, nxt: DataFrame) -> bool:
    """True when two edge sets are identical after canonicalization."""
    return rows_equal(canonicalize(prev), canonicalize(nxt), [SRC, DST])


def _assign_components(edges: DataFrame) -> DataFrame:
    """Map each node to the minimum id it points to in the contracted edge set."""
    nodes = edges.select(col(SRC).alias(ID)).union_all(edges.select(col(DST).alias(ID))).distinct()
    rep_map = edges.groupby(SRC).agg(col(DST).min().alias(COMPONENT)).select(col(SRC).alias(ID), col(COMPONENT))
    return (
        nodes.join(rep_map, on=ID, how="left")
        .with_column(
            COMPONENT,
            when(col(COMPONENT).is_null(), col(ID)).otherwise(col(COMPONENT)),
        )
        .select(ID, COMPONENT)
    )


def _propagate_min_labels(
    edges: DataFrame,
    assignments: DataFrame,
    *,
    max_iters: int,
    materialize_every: int,
    checkpoint_dir: str | None,
) -> DataFrame:
    """Diffuse the minimum label across components for exact igraph parity.

    Star contraction can leave a component split across several local minima.
    This pass repeatedly lowers each node's label to the minimum among its
    neighbors until stable, so every node in a component adopts the single
    global minimum id.

    Note: this phase collects the symmetrized contracted edge set on the driver.
    After star contraction that set is typically small, but a very large
    contracted graph will materialize here.
    """
    adjacency = collect_bounded(symmetrize(edges))

    def step(labels: DataFrame) -> DataFrame:
        neighbor_labels = labels.select(col(ID).alias(DST), col(COMPONENT).alias(_NBR))
        # Cap the per-neighbor minimum before the second join so the two joins
        # and the aggregation between them cannot stack their shuffle partitions.
        # A plan rewrite, so it costs no extra execution per round.
        nbr_min = bound_partitions(
            adjacency.join(neighbor_labels, on=DST, how="left")
            .groupby(SRC)
            .agg(col(_NBR).min().alias(_NBR_MIN))
            .select(col(SRC).alias(ID), col(_NBR_MIN))
        )
        return (
            labels.join(nbr_min, on=ID, how="left")
            .with_column(
                COMPONENT,
                when(col(_NBR_MIN).is_null(), col(COMPONENT)).otherwise(
                    when(col(COMPONENT) <= col(_NBR_MIN), col(COMPONENT)).otherwise(col(_NBR_MIN))
                ),
            )
            .select(ID, COMPONENT)
        )

    final, _ = iterate_to_fixed_point(
        assignments,
        step,
        lambda prev, nxt: rows_equal(prev, nxt, [ID, COMPONENT]),
        max_iters=max_iters,
        materialize_every=materialize_every,
        checkpoint_dir=checkpoint_dir,
    )
    return final


def _attach_isolated(vertices: DataFrame, assignments: DataFrame) -> DataFrame:
    """Give every vertex a component, mapping edge free vertices to themselves."""
    return (
        vertices.select(ID)
        .distinct()
        .join(assignments, on=ID, how="left")
        .with_column(
            COMPONENT,
            when(col(COMPONENT).is_null(), col(ID)).otherwise(col(COMPONENT)),
        )
        .select(ID, COMPONENT)
    )


def _distributed_connected_components(
    graph: Graph,
    *,
    max_iters: int,
    materialize_every: int,
    checkpoint_dir: str | None,
) -> DataFrame:
    """Connected components by distributed star contraction on Daft."""
    if graph.edges.count_rows() == 0:
        return graph.vertices.select(col(ID), col(ID).alias(COMPONENT)).distinct()
    star_ckpt = os.path.join(checkpoint_dir, "star") if checkpoint_dir else None
    label_ckpt = os.path.join(checkpoint_dir, "labels") if checkpoint_dir else None
    edges = canonicalize(graph.edges)
    final_edges, _ = iterate_to_fixed_point(
        edges,
        _star_step,
        _canonical_equal,
        max_iters=max_iters,
        materialize_every=materialize_every,
        checkpoint_dir=star_ckpt,
    )
    assignments = _assign_components(final_edges)
    assignments = _propagate_min_labels(
        final_edges,
        assignments,
        max_iters=max_iters,
        materialize_every=materialize_every,
        checkpoint_dir=label_ckpt,
    )
    return _attach_isolated(graph.vertices, assignments)


def _local_scipy_components(graph: Graph, *, directed: bool, connection: str) -> DataFrame:
    """Connected components on a single node via scipy.sparse.csgraph.

    Builds a sparse adjacency matrix from the collected edges, runs scipy's
    component finder, then relabels each component by its minimum node id.
    ``directed`` and ``connection`` select weak (undirected) or strong components.

    Raises:
        ImportError: If the optional ``local`` extra is not installed.
    """
    require_numpy("the local connected components solve")
    require_scipy("the local connected components solve")
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components as _scipy_cc

    vd = graph.vertices.select(ID).distinct().collect().to_pydict()
    ed = graph.edges.select(SRC, DST).collect().to_pydict()
    srcs = [int(x) for x in ed[SRC]]
    dsts = [int(x) for x in ed[DST]]
    node_ids = sorted({int(x) for x in vd[ID]} | set(srcs) | set(dsts))
    index = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    rows = [index[s] for s in srcs]
    cols = [index[d] for d in dsts]
    data = np.ones(len(rows), dtype=np.int8)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))
    _, labels = _scipy_cc(adjacency, directed=directed, connection=connection, return_labels=True)
    rep_of_label: dict[int, int] = {}
    for nid in node_ids:
        lab = int(labels[index[nid]])
        if lab not in rep_of_label or nid < rep_of_label[lab]:
            rep_of_label[lab] = nid
    components = [rep_of_label[int(labels[index[nid]])] for nid in node_ids]
    return daft.from_pydict({ID: node_ids, COMPONENT: components})


def _local_connected_components(graph: Graph) -> DataFrame:
    """Weakly connected components on a single node via scipy."""
    return _local_scipy_components(graph, directed=False, connection="weak")


def _resolve_strategy(strategy: Strategy, num_edges: int, local_threshold: int) -> Literal["local", "distributed"]:
    """Resolve ``auto`` to ``local`` or ``distributed`` based on edge count.

    ``auto`` means the best available strategy, so it falls back to
    ``distributed`` when the optional ``local`` extra is not installed rather
    than failing. An explicit ``local`` is honored and raises later if the extra
    is missing, because the caller asked for it by name.
    """
    if strategy == "local":
        return "local"
    if strategy == "distributed":
        return "distributed"
    if strategy == "auto":
        if num_edges <= local_threshold and has_local_extra():
            return "local"
        return "distributed"
    raise ValueError(f"unknown strategy {strategy!r}; expected 'auto', 'distributed', or 'local'")


def connected_components(
    graph: Graph,
    *,
    strategy: Strategy = "auto",
    max_iters: int = 30,
    materialize_every: int = 1,
    checkpoint_dir: str | None = None,
    local_threshold: int = _DEFAULT_LOCAL_THRESHOLD,
) -> DataFrame:
    """Compute weakly connected components as columns ``id`` and ``component``.

    Every vertex carries the smallest id in its component; edge free vertices
    form singleton components.

    Args:
        graph: The graph to analyze. Accepts either graph flavor; edges are treated as undirected either way.
        strategy: ``auto`` (default), ``distributed``, or ``local``. ``auto``
            picks the local solve only when the edge count is under
            ``local_threshold`` and the optional ``local`` extra is installed,
            otherwise it runs distributed.
        max_iters: Maximum rounds for each iterative phase (distributed only).
        materialize_every: How often to truncate the Daft plan between rounds.
        checkpoint_dir: Optional parquet checkpoint directory for long runs.
        local_threshold: Edge count at or below which ``auto`` uses the local solve.

    Returns:
        A DataFrame with one row per vertex: ``id`` and its ``component`` label.
    """
    # Only count edges when it could change the decision. On a core only install
    # `auto` is always distributed, so skip the count job entirely there.
    if strategy == "auto" and not has_local_extra():
        resolved: Literal["local", "distributed"] = "distributed"
    else:
        num_edges = graph.num_edges() if strategy == "auto" else 0
        resolved = _resolve_strategy(strategy, num_edges, local_threshold)
    if resolved == "local":
        return _local_connected_components(graph)
    return _distributed_connected_components(
        graph,
        max_iters=max_iters,
        materialize_every=materialize_every,
        checkpoint_dir=checkpoint_dir,
    )
