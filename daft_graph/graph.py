"""The Graph abstraction over a vertices DataFrame and an edges DataFrame.

``Graph`` is abstract. Construct a :class:`DirectedGraph` or an
:class:`UndirectedGraph` instead, so the type carries the direction semantics and
an algorithm can say which flavor it needs. Algorithms that traverse edges read
:meth:`Graph._traversal_edges`, which is the edge set as given for a directed
graph and the symmetrized edge set for an undirected one.

Incoming column names are normalized on construction. The ``src_col``,
``dst_col``, and ``id_col`` arguments describe the frames handed in, not the
frames stored, so the rest of the library reads the canonical ``src``, ``dst``,
and ``id`` names from :mod:`daft_graph.schema`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from daft import DataFrame, Expression, col

from daft_graph.edges import symmetrize, validate_edges
from daft_graph.schema import DST, ID, SRC

if TYPE_CHECKING:
    # `typing.Self` is 3.11+, so import from typing_extensions for the 3.10
    # target. Guarded under TYPE_CHECKING, so there is no runtime import; every
    # use is a return annotation kept as a string by `from __future__`.
    from typing_extensions import Self

_DEGREE = "degree"


def _rename(df: DataFrame, mapping: dict[str, str]) -> DataFrame:
    """Rename columns per ``mapping``, leaving every other column untouched.

    Args:
        df: The frame to rename columns on.
        mapping: Incoming column name to canonical column name. Identity entries
            are ignored.

    Returns:
        ``df`` unchanged when no rename is needed, else a projection with the
        renamed columns and all other columns preserved in their original order.

    Raises:
        ValueError: If a source column is missing, or a rename would collide with
            an existing column that is not itself being renamed away.
    """
    pending = {src: dst for src, dst in mapping.items() if src != dst}
    if not pending:
        return df
    columns = df.column_names
    missing = [src for src in pending if src not in columns]
    if missing:
        raise ValueError(f"column(s) not found: {sorted(missing)}; have {sorted(columns)}")
    collisions = [dst for dst in pending.values() if dst in columns and dst not in pending]
    if collisions:
        raise ValueError(f"renaming to {sorted(collisions)} would collide with existing column(s)")
    return df.select(*[col(c).alias(pending.get(c, c)) for c in columns])


def _derive_vertices(edges: DataFrame) -> DataFrame:
    """Return the distinct endpoint ids of ``edges`` as a one column vertex frame."""
    endpoints = edges.select(col(SRC).alias(ID)).union_all(edges.select(col(DST).alias(ID)))
    return endpoints.distinct()


class Graph(ABC):
    """A graph backed by a vertices DataFrame and an edges DataFrame.

    Abstract. Use :class:`DirectedGraph` or :class:`UndirectedGraph`.

    Instances are immutable after construction, like the frozen dataclass this
    replaced. The stored frames are reachable read only through the ``edges`` and
    ``vertices`` properties; rebind nothing.

    Attributes:
        vertices: DataFrame with a unique ``id`` column plus optional properties.
        edges: DataFrame with ``src`` and ``dst`` columns plus optional properties.
    """

    __slots__ = ("_edges", "_vertices")

    _vertices: DataFrame
    _edges: DataFrame

    def __setattr__(self, name: str, value: object) -> None:
        """Block attribute assignment, so instances stay immutable."""
        raise AttributeError(f"{type(self).__name__} is immutable; cannot set {name!r}")

    def __init__(
        self,
        edges: DataFrame,
        vertices: DataFrame | None = None,
        *,
        src_col: str = SRC,
        dst_col: str = DST,
        id_col: str = ID,
        validate: bool = False,
    ) -> None:
        """Build a graph from an edge list and, optionally, a vertex list.

        Args:
            edges: DataFrame holding the edges.
            vertices: DataFrame holding the vertices. When None the distinct edge
                endpoints are used.
            src_col: Source column name in ``edges``.
            dst_col: Destination column name in ``edges``.
            id_col: Id column name in ``vertices``. Ignored when ``vertices`` is None.
            validate: Run the semantic checks, namely that vertex ids are unique
                and that every edge endpoint is a known vertex. Both scan the
                data, so they are off by default. Structural column checks always
                run because they only read the schema.

        Raises:
            ValueError: If a required column is missing, or if ``validate`` is set
                and a semantic check fails.
        """
        if src_col == dst_col:
            raise ValueError(f"src_col and dst_col must differ, got {src_col!r} for both")
        normalized_edges = _rename(edges, {src_col: SRC, dst_col: DST})
        validate_edges(normalized_edges)
        if vertices is None:
            normalized_vertices = _derive_vertices(normalized_edges)
        else:
            normalized_vertices = _rename(vertices, {id_col: ID})
            if ID not in normalized_vertices.column_names:
                raise ValueError(f"vertices must have an {ID!r} column")
        object.__setattr__(self, "_edges", normalized_edges)
        object.__setattr__(self, "_vertices", normalized_vertices)
        if validate:
            self._validate()

    def _validate(self) -> None:
        """Check that vertex ids are unique and every edge endpoint is known.

        Raises:
            ValueError: If a duplicate vertex id exists, or an edge references a
                vertex that is not in the vertex set.
        """
        ids = self._vertices.select(ID)
        total = ids.count_rows()
        if total != ids.distinct().count_rows():
            raise ValueError(f"vertices contain duplicate {ID!r} values")
        endpoints = _derive_vertices(self._edges)
        dangling = endpoints.join(ids, on=ID, how="anti").count_rows()
        if dangling:
            raise ValueError(f"{dangling} edge endpoint(s) are not in the vertex set")

    @property
    def edges(self) -> DataFrame:
        """The edge DataFrame, with canonical ``src`` and ``dst`` columns."""
        return self._edges

    @property
    def vertices(self) -> DataFrame:
        """The vertex DataFrame, with a canonical ``id`` column."""
        return self._vertices

    def _rebuild(self, *, vertices: DataFrame, edges: DataFrame) -> Self:
        """Construct the same concrete class from already normalized frames."""
        return type(self)(edges, vertices)

    @property
    @abstractmethod
    def _directed(self) -> bool:
        """Whether edge direction is meaningful for this graph."""

    def _orient(self, edges: DataFrame) -> DataFrame:
        """Orient an edge frame per this graph's direction semantics.

        Takes the frame rather than reading ``self.edges`` so callers can filter
        or project first and still get the right orientation applied afterwards.
        ``symmetrize`` preserves edge attribute columns, so an undirected
        traversal keeps the same schema it was given.
        """
        return edges if self._directed else symmetrize(edges)

    def _traversal_edges(self) -> DataFrame:
        """The whole edge set, oriented per this graph's direction semantics."""
        return self._orient(self._edges)

    def degrees(self) -> DataFrame:
        """Total degree per vertex, as columns ``id`` and ``degree``.

        Counts incident endpoint slots, so an edge contributes one to each of its
        endpoints and a self loop contributes two. For a directed graph that is
        the same number as in degree plus out degree. Only vertices touched by at
        least one edge appear, matching GraphFrames.
        """
        endpoints = self._edges.select(col(SRC).alias(ID)).union_all(self._edges.select(col(DST).alias(ID)))
        return endpoints.groupby(ID).agg(col(ID).count().alias(_DEGREE))

    def num_vertices(self) -> int:
        """Return the number of distinct vertices."""
        return self._vertices.select(ID).distinct().count_rows()

    def num_edges(self) -> int:
        """Return the number of edge rows."""
        return self._edges.count_rows()

    def triplets(self) -> DataFrame:
        """Edges joined with their endpoint vertex attributes.

        Returns the edge columns plus ``src_<attr>`` and ``dst_<attr>`` for every
        non id vertex attribute, so each row carries both endpoints' properties.
        """
        vattrs = [c for c in self._vertices.column_names if c != ID]
        src_v = self._vertices.select(col(ID).alias(SRC), *[col(c).alias(f"src_{c}") for c in vattrs])
        dst_v = self._vertices.select(col(ID).alias(DST), *[col(c).alias(f"dst_{c}") for c in vattrs])
        return self._edges.join(src_v, on=SRC, how="inner").join(dst_v, on=DST, how="inner")

    def filter_vertices(self, condition: Expression) -> Self:
        """Keep vertices matching ``condition`` and drop edges touching removed ones."""
        kept = self._vertices.where(condition)
        kept_ids = kept.select(ID)
        edges = self._edges.join(kept_ids.select(col(ID).alias(SRC)), on=SRC, how="semi").join(
            kept_ids.select(col(ID).alias(DST)), on=DST, how="semi"
        )
        return self._rebuild(vertices=kept, edges=edges)

    def filter_edges(self, condition: Expression) -> Self:
        """Keep edges matching ``condition``; all vertices are retained."""
        return self._rebuild(vertices=self._vertices, edges=self._edges.where(condition))

    def drop_isolated_vertices(self) -> Self:
        """Drop vertices that do not appear in any edge."""
        endpoints = _derive_vertices(self._edges)
        kept = self._vertices.join(endpoints, on=ID, how="semi")
        return self._rebuild(vertices=kept, edges=self._edges)

    def degree_by_type(self, type_column: str) -> DataFrame:
        """Degree of each vertex broken down by edge type.

        Treats edges as undirected and counts incident edges of each type.
        Returns columns ``id``, ``<type_column>``, and ``degree``. Useful for
        labeled property graphs where edges carry a type or relationship column.
        """
        endpoints = self._edges.select(col(SRC).alias(ID), col(type_column)).union_all(
            self._edges.select(col(DST).alias(ID), col(type_column))
        )
        return endpoints.groupby([ID, type_column]).agg(col(ID).count().alias(_DEGREE))

    def bfs_paths(
        self,
        from_filter: Expression,
        to_filter: Expression,
        *,
        max_path_length: int = 10,
        edge_filter: Expression | None = None,
        max_paths: int = 1_000_000,
    ) -> DataFrame:
        """Shortest paths from a source vertex set to a target vertex set.

        The GraphFrames ``bfs`` analog. See
        :func:`daft_graph.algorithms.bfs.bfs_paths` for the semantics and the
        returned column shape.
        """
        from daft_graph.algorithms.bfs import bfs_paths

        return bfs_paths(
            self,
            from_filter,
            to_filter,
            max_path_length=max_path_length,
            edge_filter=edge_filter,
            max_paths=max_paths,
        )


class DirectedGraph(Graph):
    """A directed graph. Edges run from ``src`` to ``dst``.

    Example:
        >>> import daft
        >>> from daft_graph import DirectedGraph
        >>> edges = daft.from_pydict({"src": [0, 1], "dst": [1, 2]})
        >>> g = DirectedGraph(edges)
    """

    __slots__ = ()

    @property
    def _directed(self) -> bool:
        """Directed traversal walks the edges as given."""
        return True

    def out_degrees(self) -> DataFrame:
        """Out degree per source vertex, as columns ``id`` and ``degree``.

        Only vertices with at least one outgoing edge appear, matching GraphFrames.
        """
        return self._edges.groupby(SRC).agg(col(DST).count().alias(_DEGREE)).select(col(SRC).alias(ID), col(_DEGREE))

    def in_degrees(self) -> DataFrame:
        """In degree per destination vertex, as columns ``id`` and ``degree``."""
        return self._edges.groupby(DST).agg(col(SRC).count().alias(_DEGREE)).select(col(DST).alias(ID), col(_DEGREE))

    def degrees(self) -> DataFrame:
        """Total degree (in plus out) per vertex, as columns ``id`` and ``degree``.

        Equivalent to the base implementation, kept explicit because in degree
        plus out degree is how a directed graph's total degree is defined.
        """
        combined = self.out_degrees().union_all(self.in_degrees())
        return combined.groupby(ID).agg(col(_DEGREE).sum().alias(_DEGREE))

    def reverse(self) -> DirectedGraph:
        """Return the graph with every edge's direction flipped.

        Edge attributes are preserved; only ``src`` and ``dst`` swap.
        """
        attrs = [c for c in self._edges.column_names if c not in (SRC, DST)]
        flipped = self._edges.select(col(DST).alias(SRC), col(SRC).alias(DST), *[col(c) for c in attrs])
        return DirectedGraph(flipped, self._vertices)

    def as_undirected(self) -> UndirectedGraph:
        """Return an undirected view over the same vertices and edges.

        The edge rows are carried over as they are. Direction is dropped by the
        undirected traversal and degree semantics, not by rewriting the edges.
        """
        return UndirectedGraph(self._edges, self._vertices)

    def find(self, pattern: str) -> DataFrame:
        """Find subgraphs matching a motif pattern.

        Motif patterns are directed, so this lives on the directed graph. See
        :func:`daft_graph.motif.find` for the pattern syntax and return shape.
        """
        from daft_graph.motif import find

        return find(self, pattern)


class UndirectedGraph(Graph):
    """An undirected graph. Each edge row is one undirected edge.

    Edges are stored exactly as handed in, one row per edge, so ``num_edges`` and
    ``degrees`` count each edge once. Direction is dropped at traversal time by
    :meth:`_traversal_edges`, which symmetrizes, rather than by duplicating the
    stored rows.

    Example:
        >>> import daft
        >>> from daft_graph import UndirectedGraph
        >>> edges = daft.from_pydict({"src": [0, 1], "dst": [1, 2]})
        >>> g = UndirectedGraph(edges)
    """

    __slots__ = ()

    @property
    def _directed(self) -> bool:
        """Undirected traversal walks both orientations of every edge."""
        return False

    def as_directed(self, src_col: str = SRC, dst_col: str = DST) -> DirectedGraph:
        """Reinterpret this graph as directed, reading direction from the columns.

        Args:
            src_col: Column of the stored edges to treat as the source.
            dst_col: Column of the stored edges to treat as the destination.

        Returns:
            A :class:`DirectedGraph` over the same vertices and edges.
        """
        return DirectedGraph(self._edges, self._vertices, src_col=src_col, dst_col=dst_col)
