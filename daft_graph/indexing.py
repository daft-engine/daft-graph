"""Vertex id indexing: relabel arbitrary vertex ids to contiguous ``int64``.

The graph algorithms operate on integer vertex ids (``id``, ``src``, ``dst`` are
``int64``). GraphFrames hides this by remapping any id type (string, UUID, long)
to a packed long internally and mapping results back. ``reindex`` provides the
same convenience for daft-graph: it relabels a graph whose ids are strings (or
any sortable type) to contiguous ``int64`` ids ``0..n-1`` and returns a mapping
DataFrame, so an integer only algorithm can run on a string keyed graph. Use
``restore_ids`` to translate id columns of a result back to the original ids.

The vertex id set is collected to the driver to assign contiguous, deterministic
ids (sorted original order). Vertex counts are typically far smaller than edge
counts, so this fits the same scale as the rest of the library; a fully
distributed relabel is future work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import daft
from daft import DataFrame, col

from daft_graph.graph import Graph
from daft_graph.schema import DST, ID, SRC

GraphT = TypeVar("GraphT", bound=Graph)

ORIGINAL = "original_id"
_NEW = "__new_id"
_NEW_SRC = "__new_src"
_NEW_DST = "__new_dst"


@dataclass(frozen=True)
class IndexedGraph(Generic[GraphT]):
    """A graph relabelled to contiguous ``int64`` ids, plus the id mapping.

    Generic in the graph flavor, so reindexing a :class:`DirectedGraph` yields an
    ``IndexedGraph[DirectedGraph]`` and the directed methods stay available.

    Attributes:
        graph: The relabelled graph; ``id``, ``src``, ``dst`` are ``int64`` in
            ``0..n-1``. Same concrete class as the graph handed to
            :func:`reindex`.
        mapping: DataFrame ``[original_id, id]`` from each original id to its new
            integer id.
    """

    graph: GraphT
    mapping: DataFrame


def reindex(graph: GraphT) -> IndexedGraph[GraphT]:
    """Relabel a graph's vertex ids to contiguous ``int64`` ids ``0..n-1``.

    Ids are assigned in sorted order of the original ids, so the result is
    deterministic. Vertex and edge attribute columns are preserved. Edges whose
    endpoints are not in the vertex set are dropped (they have no index), which
    matches how GraphFrames joins edges to its indexed vertices.

    Args:
        graph: The graph to relabel; its ``id`` column may hold any sortable type.

    Returns:
        An :class:`IndexedGraph` with the relabelled graph and the id mapping.
        The relabelled graph is the same concrete class as ``graph``.

    Raises:
        ValueError: if the graph has a vertex or edge column whose name collides
            with an internal working name (``__new_id``/``__new_src``/``__new_dst``).
    """
    conflicts = sorted((set(graph.vertices.column_names) | set(graph.edges.column_names)) & {_NEW, _NEW_SRC, _NEW_DST})
    if conflicts:
        raise ValueError(f"graph columns conflict with reindex internal names: {conflicts}")
    originals = sorted(graph.vertices.select(ID).distinct().collect().to_pydict()[ID])
    new_ids = list(range(len(originals)))
    mapping = daft.from_pydict({ORIGINAL: originals, ID: new_ids})

    v_attrs = [c for c in graph.vertices.column_names if c != ID]
    v_remap = daft.from_pydict({ORIGINAL: originals, _NEW: new_ids})
    new_vertices = graph.vertices.join(v_remap, left_on=ID, right_on=ORIGINAL, how="inner").select(
        col(_NEW).alias(ID), *v_attrs
    )

    e_attrs = [c for c in graph.edges.column_names if c not in (SRC, DST)]
    src_remap = daft.from_pydict({SRC: originals, _NEW_SRC: new_ids})
    dst_remap = daft.from_pydict({DST: originals, _NEW_DST: new_ids})
    new_edges = (
        graph.edges.join(src_remap, on=SRC, how="inner")
        .join(dst_remap, on=DST, how="inner")
        .select(col(_NEW_SRC).alias(SRC), col(_NEW_DST).alias(DST), *e_attrs)
    )
    relabelled = type(graph)(new_edges, new_vertices)
    return IndexedGraph(graph=relabelled, mapping=mapping)


def restore_ids(df: DataFrame, mapping: DataFrame, columns: list[str]) -> DataFrame:
    """Translate integer id columns of ``df`` back to the original ids.

    Args:
        df: A result DataFrame whose ``columns`` hold reindexed ``int64`` ids.
        mapping: The ``[original_id, id]`` mapping from :func:`reindex`.
        columns: Names of the columns in ``df`` to map back to original ids.

    Returns:
        ``df`` with each named column's values replaced by the original ids,
        preserving column order. Id values absent from ``mapping`` become null;
        no error is raised for them.

    Raises:
        ValueError: if a requested column is missing from ``df``, or if ``df`` has
            a column that collides with the internal working name.
    """
    missing = [c for c in columns if c not in df.column_names]
    if missing:
        raise ValueError(f"columns not found in df: {missing}")
    if _NEW in df.column_names:
        raise ValueError(f"df column {_NEW!r} conflicts with restore_ids internal name")
    result = df
    for column in columns:
        order = result.column_names
        lookup = mapping.select(col(ID).alias(column), col(ORIGINAL).alias(_NEW))
        result = result.join(lookup, on=column, how="left").select(
            *[col(_NEW).alias(column) if name == column else col(name) for name in order]
        )
    return result
