# daft-graph usage

daft-graph provides graph operations over Daft DataFrames. Edges are a DataFrame
with `src` and `dst` integer columns; vertices are a DataFrame with an `id`
column. Everything runs on the Daft native runner locally or on Ray when
distributed.

## Building a graph

There are two graph classes. `DirectedGraph` treats `src` to `dst` as meaningful,
`UndirectedGraph` walks every edge both ways. `Graph` is the abstract base, used
for type annotations and isinstance checks, and cannot be constructed.

Pick the class that matches your data. The type is what tells an algorithm which
semantics you meant, so `pagerank` accepts only a `DirectedGraph` while
`connected_components` accepts either.

```python
import daft
from daft_graph import DirectedGraph, UndirectedGraph

edges = daft.from_pydict({"src": [1, 2, 4], "dst": [2, 3, 5]})

# Vertices are derived from the edge endpoints when not supplied
graph = DirectedGraph(edges)

# Or supply vertices explicitly (keeps isolated vertices)
vertices = daft.from_pydict({"id": [1, 2, 3, 4, 5, 99]})
graph = DirectedGraph(edges, vertices)

graph.num_vertices()   # 6
graph.num_edges()      # 3
graph.degrees()        # DataFrame[id, degree]
graph.out_degrees()    # directed only
```

Custom column names are read on ingest and normalized to `src`, `dst`, and `id`.

```python
df = daft.from_pydict({"from": [1, 2], "to": [2, 3]})
graph = DirectedGraph(df, src_col="from", dst_col="to")
graph.edges.column_names        # ["src", "dst"]

people = daft.from_pydict({"node": [1, 2, 3]})
graph = DirectedGraph(df, people, src_col="from", dst_col="to", id_col="node")
```

Validation is opt in. Structural column checks always run because they only read
the schema, while the checks that scan data are behind `validate=True`.

```python
DirectedGraph(edges, vertices, validate=True)   # unique ids, no dangling endpoints
```

## Converting between flavors

```python
directed = DirectedGraph(edges)

directed.reverse()          # DirectedGraph with every edge flipped
directed.as_undirected()    # UndirectedGraph over the same rows

undirected = UndirectedGraph(edges)
undirected.as_directed()                            # read src -> dst as given
undirected.as_directed(src_col="dst", dst_col="src")  # or flip on the way out
```

An `UndirectedGraph` stores one row per edge, so `num_edges` and `degrees` count
each edge once. The symmetrization happens at traversal time, which is why a
single undirected edge gives each endpoint degree 1.

## Connected components

Weakly connected components labelled by the smallest id in each component. Edge
free vertices form singleton components. Output is a DataFrame `[id, component]`.

```python
from daft_graph import connected_components

graph = UndirectedGraph(edges)                       # or pass a DirectedGraph

components = connected_components(graph)            # strategy="auto"
components = connected_components(graph, strategy="distributed")
components = connected_components(graph, strategy="local")
```

Undirected semantics apply whichever flavor you pass, since the implementation
symmetrizes internally.

- `distributed` runs large star and small star contraction on Daft and scales to
  the full edge set.
- `local` collects the edge set and finishes with `scipy.sparse.csgraph` on one
  node. Useful once deduplication has collapsed the edge count.
- `auto` (default) picks `local` at or below `local_threshold` edges *and* only
  when the optional `local` extra is installed, else `distributed`.

The `local` strategy needs `numpy` and `scipy`, which ship in the optional extra.
Install it with `uv sync --extra local` or `pip install 'daft-graph[local]'`. A
core install has `daft` alone, where `auto` quietly stays distributed and an
explicit `strategy="local"` raises an `ImportError` naming the extra.

## Label propagation

Community detection by synchronous label propagation. Output is `[id, label]`.

```python
from daft_graph import label_propagation

labels = label_propagation(graph, max_iters=10)
```

Updates are synchronous and bounded by `max_iters` because label propagation can
oscillate on bipartite structures.

## PageRank

PageRank matching networkx semantics, with ranks summing to one. Output is
`[id, rank]`.

```python
from daft_graph import pagerank

ranks = pagerank(graph, damping=0.85, tol=1e-6)

# Personalized PageRank seeded on specific vertices
ranks = pagerank(graph, source_ids=[1, 2])
```

## Strongly connected components

SCCs of a directed graph, labelled by the smallest id in each component. Same
`strategy` options as connected components.

```python
from daft_graph import strongly_connected_components

sccs = strongly_connected_components(graph)               # strategy="auto"
sccs = strongly_connected_components(graph, strategy="distributed")
```

## Triangle count

Triangles per vertex (undirected). Output is `[id, triangle_count]`.

```python
from daft_graph import triangle_count

counts = triangle_count(graph)
```

## BFS and shortest paths

`bfs` returns a shortest path between two vertices as a list of ids, or None if
unreachable within `max_path_length`. `shortest_paths` returns hop distances from
every vertex to each landmark as `[id, landmark, distance]`.

```python
from daft_graph import bfs, shortest_paths

path = bfs(graph, source=0, target=9)                 # [0, 3, 9] or None

# to walk edges both ways, hand it an undirected graph
path = bfs(graph.as_undirected(), source=0, target=9)

distances = shortest_paths(graph, landmarks=[0, 5])
```

There is no `directed` keyword. Traversal follows the graph's own semantics, so
`bfs`, `bfs_paths`, `all_shortest_paths`, `all_paths`, `shortest_paths`,
`random_walks`, and `hyper_anf` all take direction from the class you built.

## k-core

Core number per vertex (undirected), as `[id, core]`.

```python
from daft_graph import k_core

cores = k_core(graph)
```

## Message passing

Build custom vertex centric algorithms on the same primitive the library uses.
`aggregate_messages` runs one round; `pregel` runs to a fixed point. Vertex state
is `[id, value, ...]`; message expressions read `src_<col>` and `dst_<col>`.

```python
from daft import col
from daft.functions import when
from daft_graph import pregel
from daft_graph.message_passing import MSG, VALUE

# Propagate the minimum reachable id (undirected)
labels = pregel(
    undirected_edges,
    init_state,                       # [id, value]
    to_src=col("dst_value"),
    agg=lambda m: m.min(),
    update=when(col(MSG).is_null(), col(VALUE)).otherwise(
        when(col(VALUE) <= col(MSG), col(VALUE)).otherwise(col(MSG))
    ),
)
```

## Motif finding

Match subgraph patterns with a GraphFrames style DSL. `find` returns one row per
match, with a struct column per named vertex (the vertex row) and per named edge
(the edge row). Filter with struct field access.

`find` is a `DirectedGraph` method, since motif edge patterns are directed.

```python
from daft import col

# directed triangles
triangles = graph.find("(a)-[]->(b); (b)-[]->(c); (c)-[]->(a)")

# edges without a reciprocal edge
one_way = graph.find("(a)-[e]->(b); !(b)-[]->(a)")

# read fields off the struct columns
one_way.where(col("a")["id"] < col("b")["id"]).show()
```

Vertices are `(name)`, edges `-[name]->`, empty names are anonymous, a leading
`!` negates an edge, and repeated names bind to the same element. Variable
length and undirected motif edges are not supported in this version.

## Cycle detection

Both take a `DirectedGraph`; an undirected graph has no directed cycles.

```python
from daft_graph import has_cycle, vertices_on_cycles

has_cycle(graph)              # bool
vertices_on_cycles(graph)     # DataFrame[id] of vertices on a directed cycle
```

## All simple paths

```python
from daft_graph import all_paths

paths = all_paths(graph, source=0, target=9, max_path_length=5)  # list[list[int]]
```

## Maximal independent set

```python
from daft_graph import maximal_independent_set

mis = maximal_independent_set(graph)   # DataFrame[id, selected]
```

## Per source personalized PageRank

```python
from daft_graph import parallel_personalized_pagerank

# one personalized vector per source, as [id, source, rank]
vectors = parallel_personalized_pagerank(graph, source_ids=[0, 5])
```

## All shortest paths and edge filters

```python
from daft import col
from daft_graph import all_shortest_paths, bfs

paths = all_shortest_paths(graph, source=0, target=9)   # every shortest path

# restrict traversal to edges matching a predicate
paths = all_shortest_paths(graph, 0, 9, edge_filter=col("type") == "follows")
one = bfs(graph, 0, 9, edge_filter=col("type") == "follows")
```

## BFS over vertex sets (GraphFrames style)

`bfs_paths` is the GraphFrames `bfs` analog: it selects source and target vertex
sets with predicate expressions and returns one row per shortest path. Vertex
columns (`from`, `v1`, ..., `to`) hold a struct of the vertex row; edge columns
(`e0`, `e1`, ...) hold a struct of the edge row.

```python
from daft import col
from daft_graph import bfs_paths

# shortest paths from any active user to any admin user
paths = bfs_paths(graph, col("status") == "active", col("role") == "admin")

# equivalently as a method; use an undirected graph to walk both ways
paths = graph.as_undirected().bfs_paths(col("id") == 0, col("id") == 9)

# read attributes off the path structs
paths.select(col("from")["id"], col("to")["id"]).show()
```

All returned paths share the shortest length. When a source is also a target the
length is 0 and the columns are just `from` and `to`. An empty DataFrame means no
target was reachable within `max_path_length`.

## Property graphs

Model labeled property graphs with attribute columns on vertices and edges, then
query with `filter_vertices`, `filter_edges`, `triplets`, and `find`. For degree
broken down by edge type:

```python
g.degree_by_type("type")   # [id, type, degree]
```

## Vertex id indexing

The algorithms operate on integer ids. To run them on a graph keyed by strings
(or any sortable id type), `reindex` relabels the graph to contiguous `int64`
ids and returns the mapping; `restore_ids` maps result id columns back. The
relabelled graph keeps the flavor it was given, so a `DirectedGraph` stays
directed.

```python
from daft_graph import connected_components, reindex, restore_ids

indexed = reindex(graph)                       # IndexedGraph(graph, mapping)
components = connected_components(indexed.graph)
result = restore_ids(components, indexed.mapping, ["id", "component"])
```

`indexed.mapping` is a DataFrame `[original_id, id]`. Edges whose endpoints are
not in the vertex set are dropped (they have no index).

## Random walks

```python
from daft_graph import random_walks

walks = random_walks(graph, walk_length=10, num_walks=5, seed=0)  # list[list[int]]
```

Seeded and deterministic; the input to walk based node embeddings (embedding
training itself is left to a dedicated ML library).

## Power iteration clustering

```python
from daft_graph import power_iteration_clustering

clusters = power_iteration_clustering(graph, k=3)   # [id, cluster]
```

## Approximate neighborhood function

```python
from daft_graph import hyper_anf

nf = hyper_anf(graph, max_hops=6)   # [id, hop, approx_count] via HyperLogLog
```

## SVD++ on rating graphs

Edges are users (`src`) to items (`dst`) with a rating column, so this takes a
`DirectedGraph`. Needs the optional `local` extra for `numpy`.

```python
from daft_graph import svd_plus_plus

result = svd_plus_plus(graph, rating_column="rating", rank=8)
result.factors   # [id, kind, bias, factor]
result.rmse      # training RMSE
```

## Edge utilities

```python
from daft_graph import (
    canonicalize,     # orient src <= dst, drop self loops, dedupe
    symmetrize,       # add reverse of every edge
    dedupe_edges,     # remove duplicate (src, dst)
    drop_self_loops,  # remove src == dst
    to_edges,         # project two columns into src/dst
    validate_edges,   # raise if src/dst columns are missing
)
```

## Running on Ray

Set the runner before launching, then use the same API:

```bash
DAFT_RUNNER=ray python my_job.py
```

```python
import daft
daft.set_runner_ray()
```

## Reading from Iceberg

Install Daft's `iceberg` extra (`pip install 'daft[iceberg]'`) and read an edge table
directly. daft-graph itself needs no extra for this:

```python
import daft
from daft_graph import UndirectedGraph, connected_components

# pass io_config by keyword; daft 0.7 inserted branch and tag ahead of it
edges = daft.read_iceberg(catalog.load_table("db.edges"))
components = connected_components(UndirectedGraph(edges))
```

See `examples/cc_on_iceberg.py` for a runnable version backed by a local parquet
fixture.

## Which flavor does an algorithm take

| Accepts | Algorithms |
|---|---|
| `DirectedGraph` only | `pagerank`, `parallel_personalized_pagerank`, `strongly_connected_components`, `has_cycle`, `vertices_on_cycles`, `svd_plus_plus`, `find` |
| Either flavor, undirected semantics | `connected_components`, `label_propagation`, `triangle_count`, `k_core`, `maximal_independent_set`, `power_iteration_clustering` |
| Either flavor, traversal follows the type | `bfs`, `bfs_paths`, `all_shortest_paths`, `shortest_paths`, `all_paths`, `random_walks`, `hyper_anf` |
| No graph, DataFrames only | `aggregate_messages`, `pregel`, and the edge utilities |

Passing an `UndirectedGraph` where a `DirectedGraph` is required is a type error
that mypy catches, which is the point of the split.

## Scaling notes

- Iterative algorithms run on `iterate_to_fixed_point`, which materializes the
  state between rounds to truncate the Daft logical plan. This is the analog of
  Spark GraphFrames checkpointing and is what keeps the iterative joins from
  growing an unbounded plan.
- For very long runs, pass `checkpoint_dir=...` to round trip the state through
  parquet between rounds.
- For connected components at large edge counts that collapse after
  deduplication, prefer `strategy="local"` or a low `local_threshold`. That path
  needs the `local` extra installed.
- Benchmark with `benchmarks/bench_cc.py --edges 1000000`.
- Edgeless graphs are handled: every vertex forms its own component or community,
  and PageRank returns the uniform distribution. The exception is
  `power_iteration_clustering`, which only returns vertices that have edges.
  Iterative algorithms emit a warning if they reach `max_iters` without
  converging.
