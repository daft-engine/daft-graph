# daft-graph

Graph processing algorithms using Daft under the hood.

`daft-graph` brings GraphFrames style graph operations to the [Daft](https://docs.daft.ai)
DataFrame engine. A graph is two DataFrames, a vertex table keyed by `id` and an
edge table keyed by `src`/`dst`, and every algorithm returns an ordinary DataFrame
that composes with the rest of a Daft pipeline. It runs on the native runner locally
and on Ray when distributed.

## Install

```bash
uv sync
```

Requires Python 3.10 to 3.13. The core install depends on `daft` alone. The single
node solves (connected components `strategy="local"` and `svd_plus_plus`) use numpy
and scipy from the optional `local` extra:

```bash
uv sync --extra local          # or: pip install 'daft-graph[local]'
```

## Quick start

```python
import daft
from daft_graph import UndirectedGraph, connected_components

edges = daft.from_pydict({"src": [0, 1, 3], "dst": [1, 2, 4]})
graph = UndirectedGraph(edges)          # vertices derived from the endpoints

connected_components(graph).show()
```

Graphs come in two flavors and the type carries the direction semantics, so an
algorithm declares which one it needs. `pagerank` takes a `DirectedGraph`;
`connected_components` accepts either.

```python
from daft_graph import DirectedGraph, pagerank

g = DirectedGraph(edges)
pagerank(g).show()                      # directed
connected_components(g).show()          # undirected semantics either way
pagerank(g.reverse()).show()            # every edge flipped
```

Convert between flavors with `g.as_undirected()` and `g.as_directed()`. Traversal
follows the graph's own semantics, so there is no `directed` keyword; pass an
`UndirectedGraph` (or call `as_undirected()`) to walk edges both ways.

## API

### Graph model

- `DirectedGraph(edges, vertices=None, *, src_col="src", dst_col="dst", id_col="id", validate=False)`
- `UndirectedGraph(edges, vertices=None, *, src_col="src", dst_col="dst", id_col="id", validate=False)`
- `Graph` - the abstract base, for type annotations and isinstance checks

Base methods: `degrees`, `triplets`, `filter_vertices`, `filter_edges`,
`drop_isolated_vertices`, `degree_by_type`, `num_vertices`, `num_edges`, `bfs_paths`.
`DirectedGraph` adds `in_degrees`, `out_degrees`, `reverse`, `as_undirected`, `find`.
`UndirectedGraph` adds `as_directed`. Transforms return the caller's flavor.

### Algorithms

| Group | Functions |
|---|---|
| Connected components | `connected_components`, `strongly_connected_components` |
| Centrality | `pagerank` (+ personalized), `parallel_personalized_pagerank` |
| Traversal | `bfs`, `bfs_paths`, `shortest_paths`, `all_shortest_paths`, `all_paths` |
| Community | `label_propagation`, `power_iteration_clustering` |
| Motif | `find` (GraphFrames style DSL) |
| Message passing | `aggregate_messages`, `pregel` |
| Other | `triangle_count`, `k_core`, `has_cycle`, `vertices_on_cycles`, `maximal_independent_set`, `random_walks`, `svd_plus_plus`, `hyper_anf` |
| Id indexing | `reindex`, `restore_ids` for arbitrary (non int) ids |
| Edge utils | `canonicalize`, `symmetrize`, `dedupe_edges`, `drop_self_loops`, `to_edges`, `validate_edges` |

Which flavor each algorithm takes, and full examples, are in [`docs/usage.md`](docs/usage.md).

## Design notes

- Iterative algorithms run on `iterate_to_fixed_point`, which materializes the state
  between rounds to truncate the Daft logical plan. This is the analog of GraphFrames
  checkpointing and is what keeps the iterative joins from growing an unbounded plan.
- Connected components ports the large star and small star contraction algorithm
  (Kiveris et al. 2014), the same method Spark GraphFrames uses by default, with an
  optional `scipy.sparse.csgraph` single node solve for small edge sets.
- Correctness is validated against igraph (connected components) and networkx
  (PageRank and others), which are test only dependencies.

## Development

```bash
uv sync                      # install with dev group
uv run pytest tests/ -v      # run the suite
uv run pre-commit run --all-files   # ruff + mypy style checks
```
