"""Benchmark connected components on synthetic random graphs.

Reports the star contraction round count and wall time on the active Daft
runner. Uses a few internal helpers (prefixed ``_``) so it can report the round
count, which the public ``connected_components`` does not expose.

Requires the optional ``local`` extra for numpy (``uv sync --extra local`` or
``pip install 'daft-graph[local]'``).

Examples:
    uv run python benchmarks/bench_cc.py --edges 1000000
    DAFT_RUNNER=ray uv run python benchmarks/bench_cc.py --edges 10000000
"""

from __future__ import annotations

import argparse
import time

import daft
import numpy as np

from daft_graph.algorithms.connected_components import (
    _assign_components,
    _attach_isolated,
    _canonical_equal,
    _propagate_min_labels,
    _star_step,
)
from daft_graph.edges import canonicalize
from daft_graph.graph import UndirectedGraph
from daft_graph.iterate import iterate_to_fixed_point
from daft_graph.schema import COMPONENT, DST, SRC


def _generate_edges(n_nodes: int, n_edges: int, seed: int) -> daft.DataFrame:
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n_nodes, size=n_edges)
    dst = rng.integers(0, n_nodes, size=n_edges)
    return daft.from_pydict({SRC: src.tolist(), DST: dst.tolist()})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="daft-graph connected components benchmark")
    parser.add_argument("--edges", type=int, default=1_000_000)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    n_nodes = args.nodes if args.nodes is not None else max(2, args.edges // 5)

    graph = UndirectedGraph(_generate_edges(n_nodes, args.edges, args.seed))

    start = time.perf_counter()
    edges = canonicalize(graph.edges)
    final_edges, star_rounds = iterate_to_fixed_point(edges, _star_step, _canonical_equal, max_iters=args.max_iters)
    after_star = time.perf_counter()

    assignments = _assign_components(final_edges)
    assignments = _propagate_min_labels(
        final_edges, assignments, max_iters=args.max_iters, materialize_every=1, checkpoint_dir=None
    )
    result = _attach_isolated(graph.vertices, assignments).collect()
    end = time.perf_counter()

    n_vertices = result.count_rows()
    n_components = result.select(COMPONENT).distinct().count_rows()

    rows = [
        ("requested edges", f"{args.edges:,}"),
        ("requested nodes", f"{n_nodes:,}"),
        ("vertices", f"{n_vertices:,}"),
        ("components", f"{n_components:,}"),
        ("star rounds", f"{star_rounds:,}"),
        ("star loop seconds", f"{after_star - start:.2f}"),
        ("assign + label seconds", f"{end - after_star:.2f}"),
        ("total seconds", f"{end - start:.2f}"),
    ]
    print("daft-graph connected components benchmark")
    print(f"{'metric':<26}{'value':>16}")
    print("-" * 42)
    for name, value in rows:
        print(f"{name:<26}{value:>16}")


if __name__ == "__main__":
    main()
