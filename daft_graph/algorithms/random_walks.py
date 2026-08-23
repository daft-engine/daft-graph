"""Random walks on Daft DataFrames.

Generates fixed length random walks, the input to walk based node embeddings such
as DeepWalk and node2vec. Walks are seeded for determinism. The adjacency is
collected to the driver, so this fits moderate graphs; a fully distributed walker
would be needed for billion edge graphs. Embedding training (skip-gram) is left
to a dedicated ML library.
"""

from __future__ import annotations

import random
from collections import defaultdict

from daft_graph.graph import Graph
from daft_graph.schema import DST, ID, SRC


def random_walks(
    graph: Graph,
    *,
    walk_length: int = 10,
    num_walks: int = 1,
    seed: int = 0,
) -> list[list[int]]:
    """Generate random walks from every vertex.

    Args:
        graph: The graph to walk.
        walk_length: Maximum number of steps (edges) per walk.
        num_walks: Number of walks started from each vertex.
        seed: Seed for the walk randomness; the result is deterministic given it.

    Returns:
        A list of walks, each a list of vertex ids beginning at the start vertex.
        A walk stops early if it reaches a vertex with no out neighbor.

    Neighbor lists are sorted before walking, so the result depends only on
    ``seed`` regardless of the edge row order Daft returns.
    """
    edges = graph.orient(graph.edges.select(SRC, DST))
    rows = edges.distinct().collect().to_pydict()
    adjacency: dict[int, list[int]] = defaultdict(list)
    for s, d in zip(rows[SRC], rows[DST]):
        adjacency[int(s)].append(int(d))
    for neighbors in adjacency.values():
        neighbors.sort()

    vertices = sorted(int(x) for x in graph.vertices.select(ID).distinct().collect().to_pydict()[ID])
    rng = random.Random(seed)
    walks: list[list[int]] = []
    for source in vertices:
        for _ in range(num_walks):
            walk = [source]
            current = source
            for _ in range(walk_length):
                options = adjacency.get(current)
                if not options:
                    break
                current = rng.choice(options)
                walk.append(current)
            walks.append(walk)
    return walks
