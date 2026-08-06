"""Approximate neighborhood function via HyperLogLog (HyperANF).

For each vertex and hop ``t``, estimates the number of vertices reachable within
``t`` hops. Each vertex keeps a HyperLogLog sketch of the vertices it has reached;
each hop a vertex merges (register wise max) its neighbors' sketches, and the
sketch cardinality estimates the reachable count. This is the scalable
approximation that the exact neighborhood function (count of reachable vertices)
would otherwise require materializing reachable sets for.

Estimates carry HyperLogLog's relative error (about ``1.04 / sqrt(2**precision)``)
with linear counting for small cardinalities. Built on the message passing
primitive; the sketch merge is a UDF.
"""

from __future__ import annotations

import hashlib
import math

import daft
from daft import DataFrame, Series, col, lit
from daft.functions import list_agg

from daft_graph.graph import Graph
from daft_graph.message_passing import MSG, aggregate_messages
from daft_graph.schema import DST, ID, SRC

HOP = "hop"
APPROX_COUNT = "approx_count"
_HLL = "hll"
_LIST = daft.DataType.list(daft.DataType.int64())


def _hash64(value: int) -> int:
    masked = int(value) & ((1 << 64) - 1)
    digest = hashlib.blake2b(masked.to_bytes(8, "little"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _alpha(m: int) -> float:
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / m)


def hyper_anf(
    graph: Graph,
    *,
    max_hops: int = 10,
    precision: int = 10,
) -> DataFrame:
    """Estimate the neighborhood function per vertex via HyperLogLog.

    Args:
        graph: The graph to analyze.
        max_hops: Number of hops to expand (rows are produced for hops 0..max_hops).
        precision: HyperLogLog precision ``p``; uses ``2**p`` registers.

    Returns:
        A DataFrame ``[id, hop, approx_count]``: the estimated number of vertices
        reachable from ``id`` within ``hop`` hops (``approx_count`` is a float).
    """
    if not 4 <= precision <= 18:
        raise ValueError(f"precision must be in [4, 18], got {precision}")
    m = 1 << precision
    bits = 64 - precision
    alpha = _alpha(m)

    @daft.func.batch(return_dtype=_LIST)
    def init_hll(ids: Series) -> list[list[int]]:
        out: list[list[int]] = []
        for vid in ids.to_pylist():
            registers = [0] * m
            h = _hash64(int(vid))
            idx = h >> bits
            w = h & ((1 << bits) - 1)
            registers[idx] = bits - w.bit_length() + 1
            out.append(registers)
        return out

    @daft.func.batch(return_dtype=_LIST)
    def merge_hll(own: Series, msgs: Series) -> list[list[int]]:
        out: list[list[int]] = []
        for current, neighbor_sketches in zip(own.to_pylist(), msgs.to_pylist()):
            acc = list(current)
            for sketch in neighbor_sketches or []:
                for j, value in enumerate(sketch):
                    acc[j] = max(acc[j], value)
            out.append(acc)
        return out

    @daft.func.batch(return_dtype=daft.DataType.float64())
    def estimate(sketches: Series) -> list[float]:
        out: list[float] = []
        for registers in sketches.to_pylist():
            raw = alpha * m * m / sum(2.0 ** (-r) for r in registers)
            zeros = registers.count(0)
            if raw <= 2.5 * m and zeros > 0:
                raw = m * math.log(m / zeros)
            out.append(raw)
        return out

    edges = graph._orient(graph.edges.select(SRC, DST))
    edges = edges.collect()

    current = graph.vertices.select(col(ID)).distinct().with_column(_HLL, init_hll(col(ID))).collect()

    def snapshot(state: DataFrame, hop: int) -> DataFrame:
        return (
            state.with_column(HOP, lit(hop))
            .with_column(APPROX_COUNT, estimate(col(_HLL)))
            .select(ID, HOP, APPROX_COUNT)
        )

    parts = [snapshot(current, 0)]
    for hop in range(1, max_hops + 1):
        msg = aggregate_messages(edges, current, to_src=col(f"dst_{_HLL}"), agg=lambda values: list_agg(values))
        current = (
            current.join(msg, on=ID, how="left")
            .with_column(_HLL, merge_hll(col(_HLL), col(MSG)))
            .select(ID, _HLL)
            .collect()
        )
        parts.append(snapshot(current, hop))

    out = parts[0]
    for part in parts[1:]:
        out = out.union_all(part)
    return out
