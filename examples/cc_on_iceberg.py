"""Example: connected components over an edge table with daft-graph.

Reads an edge table (here a local parquet fixture standing in for an Iceberg
table), computes connected components, and writes the labels back out.

Run locally:
    uv run python examples/cc_on_iceberg.py

Run distributed on Ray by setting the runner before launching:
    DAFT_RUNNER=ray uv run python examples/cc_on_iceberg.py

To read a real Iceberg table instead of the parquet fixture, install the
``iceberg`` extra and use ``daft.read_iceberg(table)`` in place of
``daft.read_parquet`` below. Pass ``io_config`` by keyword, since Daft 0.7
inserted ``branch`` and ``tag`` before it in the signature.

Connected components applies undirected semantics, so this builds an
``UndirectedGraph``. Use ``DirectedGraph`` when direction matters, for example
for ``pagerank`` or ``strongly_connected_components``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import daft

from daft_graph import UndirectedGraph, connected_components
from daft_graph.schema import COMPONENT, DST, ID, SRC


def _write_edge_fixture(path: Path) -> None:
    """Write a small two component edge table to parquet."""
    edges = daft.from_pydict({SRC: [1, 2, 3, 10, 11], DST: [2, 3, 1, 11, 12]})
    edges.write_parquet(str(path))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        edge_table = root / "edges"
        _write_edge_fixture(edge_table)

        # In production: daft.read_iceberg(catalog.load_table("db.edges"))
        # Pass io_config by keyword; Daft 0.7 inserted branch and tag ahead of it.
        edges = daft.read_parquet(str(edge_table))
        graph = UndirectedGraph(edges)

        components = connected_components(graph).collect()
        n_components = components.select(COMPONENT).distinct().count_rows()
        components.write_parquet(str(root / "components"))

        result = components.to_pydict()
        labels = dict(zip(result[ID], result[COMPONENT]))
        print(f"vertices: {graph.num_vertices()}  edges: {graph.num_edges()}")
        print(f"components: {n_components}")
        print(f"labels: {labels}")


if __name__ == "__main__":
    main()
