"""Power iteration clustering (Lin and Cohen 2010) on Daft DataFrames.

Runs power iteration on the row normalized affinity matrix to produce a one
dimensional embedding, then 1D k-means to assign clusters. Crucially the
iteration stops on the acceleration criterion (when the per step increment
stabilizes) rather than running to full convergence, since the stationary vector
carries no cluster structure. The matrix vector product is distributed via the
message passing primitive; the small 1D vector and the k-means run on the driver.
"""

from __future__ import annotations

import daft
from daft import col

from daft_graph.edges import canonicalize, symmetrize
from daft_graph.graph import Graph
from daft_graph.iterate import collect_bounded
from daft_graph.message_passing import MSG, VALUE, aggregate_messages
from daft_graph.schema import DST, ID, SRC

CLUSTER = "cluster"
_DEG = "deg"


def _kmeans_1d(values: list[float], k: int, iters: int) -> list[int]:
    """Deterministic 1D k-means; returns a cluster index per input value."""
    n = len(values)
    if k <= 1 or n == 0:
        return [0] * n
    unique = sorted(set(values))
    if len(unique) <= k:
        rank = {v: i for i, v in enumerate(unique)}
        return [rank[v] for v in values]
    centroids = [unique[round(i * (len(unique) - 1) / (k - 1))] for i in range(k)]
    assignment = [0] * n
    for _ in range(iters):
        nxt = [min(range(k), key=lambda c: abs(values[i] - centroids[c])) for i in range(n)]
        if nxt == assignment:
            break
        assignment = nxt
        for c in range(k):
            members = [values[i] for i in range(n) if assignment[i] == c]
            if members:
                centroids[c] = sum(members) / len(members)
    return assignment


def power_iteration_clustering(
    graph: Graph,
    k: int,
    *,
    max_iters: int = 100,
    tol: float = 1e-5,
    kmeans_iters: int = 50,
) -> daft.DataFrame:
    """Cluster vertices via power iteration clustering.

    Args:
        graph: The graph to cluster. Accepts either graph flavor; edges are treated as undirected either way.
        k: Number of clusters.
        max_iters: Maximum power iterations.
        tol: Acceleration threshold for early stopping.
        kmeans_iters: Maximum 1D k-means iterations.

    Returns:
        A DataFrame ``[id, cluster]`` over the vertices that have edges. Vertices
        with no incident edge are absent from the output rather than forming
        singleton clusters.

    Raises:
        ValueError: If ``k`` is less than 1.

    Note:
        With the degree based initialization, perfectly symmetric communities
        may not separate (the separating component is absent from a symmetric
        start). This matches the known behavior of PIC with a symmetric init.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    undirected = collect_bounded(symmetrize(canonicalize(graph.edges)))
    degree_rows = (
        undirected.groupby(SRC)
        .agg(col(DST).count().alias(_DEG))
        .select(col(SRC).alias(ID), col(_DEG))
        .collect()
        .to_pydict()
    )
    degree = {int(i): int(d) for i, d in zip(degree_rows[ID], degree_rows[_DEG])}
    ids = sorted(degree)
    if not ids:
        return daft.from_pydict({ID: ids, CLUSTER: []})

    total = float(sum(degree.values()))
    vector = {i: degree[i] / total for i in ids}
    delta_prev: dict[int, float] | None = None
    for _ in range(max_iters):
        state = daft.from_pydict({ID: ids, VALUE: [vector[i] for i in ids]})
        msg = (
            aggregate_messages(undirected, state, to_src=col("dst_value"), agg=lambda m: m.sum()).collect().to_pydict()
        )
        neighbor_sum = {int(i): float(s) for i, s in zip(msg[ID], msg[MSG])}
        updated = {i: neighbor_sum.get(i, 0.0) / degree[i] for i in ids}
        norm = sum(abs(x) for x in updated.values()) or 1.0
        updated = {i: x / norm for i, x in updated.items()}
        delta = {i: updated[i] - vector[i] for i in ids}
        if delta_prev is not None:
            acceleration = sum(abs(delta[i] - delta_prev[i]) for i in ids)
            if acceleration < tol:
                vector = updated
                break
        delta_prev = delta
        vector = updated

    values = [vector[i] for i in ids]
    clusters = _kmeans_1d(values, k, kmeans_iters)
    return daft.from_pydict({ID: ids, CLUSTER: [int(c) for c in clusters]})
