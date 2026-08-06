"""daft-graph: distributed graph operations for the Daft DataFrame engine on Ray.

Construct a :class:`DirectedGraph` or an :class:`UndirectedGraph`. ``Graph`` is
the abstract base, exported for type annotations and isinstance checks.

Example:
    >>> import daft
    >>> from daft_graph import UndirectedGraph, connected_components
    >>> edges = daft.from_pydict({"src": [0, 1, 3], "dst": [1, 2, 4]})
    >>> components = connected_components(UndirectedGraph(edges))
"""

from importlib.metadata import PackageNotFoundError, version

from daft_graph.algorithms.all_paths import all_paths
from daft_graph.algorithms.bfs import all_shortest_paths, bfs, bfs_paths
from daft_graph.algorithms.connected_components import connected_components
from daft_graph.algorithms.cycles import has_cycle, vertices_on_cycles
from daft_graph.algorithms.hyper_anf import hyper_anf
from daft_graph.algorithms.k_core import k_core
from daft_graph.algorithms.label_propagation import label_propagation
from daft_graph.algorithms.maximal_independent_set import maximal_independent_set
from daft_graph.algorithms.pagerank import pagerank, parallel_personalized_pagerank
from daft_graph.algorithms.power_iteration_clustering import power_iteration_clustering
from daft_graph.algorithms.random_walks import random_walks
from daft_graph.algorithms.shortest_paths import shortest_paths
from daft_graph.algorithms.strongly_connected_components import (
    strongly_connected_components,
)
from daft_graph.algorithms.svd_plus_plus import SvdPlusPlusResult, svd_plus_plus
from daft_graph.algorithms.triangle_count import triangle_count
from daft_graph.edges import (
    canonicalize,
    dedupe_edges,
    drop_self_loops,
    symmetrize,
    to_edges,
    validate_edges,
)
from daft_graph.graph import DirectedGraph, Graph, UndirectedGraph
from daft_graph.indexing import ORIGINAL, IndexedGraph, reindex, restore_ids
from daft_graph.message_passing import aggregate_messages, pregel
from daft_graph.motif import find

try:
    __version__ = version("daft-graph")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "ORIGINAL",
    "DirectedGraph",
    "Graph",
    "IndexedGraph",
    "SvdPlusPlusResult",
    "UndirectedGraph",
    "__version__",
    "aggregate_messages",
    "all_paths",
    "all_shortest_paths",
    "bfs",
    "bfs_paths",
    "canonicalize",
    "connected_components",
    "dedupe_edges",
    "drop_self_loops",
    "find",
    "has_cycle",
    "hyper_anf",
    "k_core",
    "label_propagation",
    "maximal_independent_set",
    "pagerank",
    "parallel_personalized_pagerank",
    "power_iteration_clustering",
    "pregel",
    "random_walks",
    "reindex",
    "restore_ids",
    "shortest_paths",
    "strongly_connected_components",
    "svd_plus_plus",
    "symmetrize",
    "to_edges",
    "triangle_count",
    "validate_edges",
    "vertices_on_cycles",
]
