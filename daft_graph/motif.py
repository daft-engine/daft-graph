"""Motif finding for daft-graph: a GraphFrames style find() DSL.

Patterns are chains of directed edge patterns and lone vertices separated by
semicolons, for example::

    "(a)-[e]->(b); (b)-[e2]->(c); !(c)-[]->(a)"

Vertices are written ``(name)`` and edges ``-[name]->``; empty names denote
anonymous (non output) elements. A leading ``!`` negates an edge pattern: the
match must not contain that edge. Repeated names bind to the same element.

``find`` compiles the pattern to a chain of Daft joins and returns one row per
match, with a struct column per named vertex (the vertex row) and per named edge
(the edge row). Filter matches with struct field access, e.g.
``result.where(col("a")["id"] != col("c")["id"])``.

Not supported in this version: variable length edges (``-[e*1..3]->``),
undirected or bidirectional motif edges, and self referential edge patterns
(``(a)-[e]->(a)``). Parallel edges (duplicate src/dst) multiply matches; dedupe
the edge set first if that is not wanted. Names starting with ``__`` are reserved.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from daft import DataFrame, Expression, col
from daft.functions import to_struct

from daft_graph.schema import DST, ID, SRC

if TYPE_CHECKING:
    from daft_graph.graph import DirectedGraph

_EDGE_RE = re.compile(r"^\(\s*(\w*)\s*\)\s*-\s*\[\s*(\w*)\s*\]\s*->\s*\(\s*(\w*)\s*\)$")
_VERTEX_RE = re.compile(r"^\(\s*(\w+)\s*\)$")
_ANON_PREFIX = "__v"


@dataclass(frozen=True)
class EdgePattern:
    """A directed edge pattern ``(src)-[edge]->(dst)``."""

    src: str
    dst: str
    edge: str | None  # None when the edge is anonymous
    negated: bool = False


@dataclass(frozen=True)
class VertexPattern:
    """A lone vertex pattern ``(name)``."""

    name: str


Clause = EdgePattern | VertexPattern


def is_named(name: str) -> bool:
    """True if ``name`` is a user provided (output) name, not anonymous."""
    return not name.startswith(_ANON_PREFIX)


def parse_motif(pattern: str) -> list[Clause]:
    """Parse a motif pattern string into a list of clauses.

    Raises:
        ValueError: on an unparseable clause, a named negated edge, or an empty
            pattern.
    """
    anon = itertools.count()
    clauses: list[Clause] = []
    for raw in pattern.split(";"):
        text = raw.strip()
        if not text:
            continue
        negated = text.startswith("!")
        if negated:
            text = text[1:].strip()
        edge_match = _EDGE_RE.match(text)
        if edge_match:
            src, edge, dst = (
                edge_match.group(1),
                edge_match.group(2),
                edge_match.group(3),
            )
            if negated and edge:
                raise ValueError(f"negated edges cannot be named: {raw!r}")
            clauses.append(
                EdgePattern(
                    src=src or f"{_ANON_PREFIX}{next(anon)}",
                    dst=dst or f"{_ANON_PREFIX}{next(anon)}",
                    edge=edge or None,
                    negated=negated,
                )
            )
            continue
        vertex_match = _VERTEX_RE.match(text)
        if vertex_match:
            if negated:
                raise ValueError(f"negated lone vertex patterns are not supported: {raw!r}")
            clauses.append(VertexPattern(name=vertex_match.group(1)))
            continue
        raise ValueError(f"could not parse motif clause: {raw!r}")
    if not clauses:
        raise ValueError("empty motif pattern")
    return clauses


def find(graph: DirectedGraph, pattern: str) -> DataFrame:
    """Find subgraphs matching a motif pattern.

    Args:
        graph: The directed graph to search. Motif edge patterns are directed.
        pattern: A motif pattern string (see module docstring).

    Returns:
        A DataFrame with one row per match and one struct column per named vertex
        (the vertex row) and per named edge (the edge row).

    Raises:
        ValueError: if the pattern has no named elements, or a negated edge
            references an unbound vertex.
    """
    clauses = parse_motif(pattern)
    edges, vertices = graph.edges, graph.vertices
    v_attrs = [c for c in vertices.column_names if c != ID]
    e_attrs = [c for c in edges.column_names if c not in (SRC, DST)]

    bindings: DataFrame | None = None
    bound: set[str] = set()
    output: list[tuple[str, str]] = []
    edge_specs: dict[str, tuple[str, str]] = {}
    negations: list[tuple[str, str]] = []
    seen: set[str] = set()

    def emit(kind: str, name: str) -> None:
        if is_named(name) and name not in seen:
            seen.add(name)
            output.append((kind, name))

    for clause in clauses:
        if isinstance(clause, VertexPattern):
            if clause.name not in bound:
                rel = vertices.select(col(ID).alias(clause.name))
                bindings = rel if bindings is None else bindings.join(rel, how="cross")
                bound.add(clause.name)
            emit("vertex", clause.name)
            continue
        if clause.negated:
            negations.append((clause.src, clause.dst))
            continue
        a, b, e = clause.src, clause.dst, clause.edge
        if a == b:
            raise ValueError(f"self referential edge patterns are not supported: ({a})-[]->({b})")
        rel = edges.select(col(SRC).alias(a), col(DST).alias(b))
        if bindings is None:
            bindings = rel
        else:
            shared: list[str | Expression] = [name for name in (a, b) if name in bound]
            bindings = bindings.join(rel, on=shared, how="inner") if shared else bindings.join(rel, how="cross")
        bound.update((a, b))
        emit("vertex", a)
        if e is not None:
            if e in edge_specs:
                raise ValueError(f"edge name {e!r} used more than once in pattern")
            edge_specs[e] = (a, b)
            emit("edge", e)
        emit("vertex", b)

    if bindings is None:
        raise ValueError("motif pattern produced no bindings")
    for a, b in negations:
        if a not in bound or b not in bound:
            raise ValueError(f"negated edge endpoints must be bound: ({a})-[]->({b})")
        neg = edges.select(col(SRC).alias(a), col(DST).alias(b))
        bindings = bindings.join(neg, on=[a, b], how="anti")

    if not output:
        raise ValueError("motif pattern has no named elements to return")

    result = bindings
    for kind, name in output:
        if kind == "edge":
            a, b = edge_specs[name]
            rel = edges.select(
                col(SRC).alias(f"__e_{name}_src"),
                col(DST).alias(f"__e_{name}_dst"),
                *[col(c).alias(f"__e_{name}_{c}") for c in e_attrs],
            )
            result = result.join(
                rel,
                left_on=[a, b],
                right_on=[f"__e_{name}_src", f"__e_{name}_dst"],
                how="inner",
            )
    for kind, name in output:
        if kind == "vertex":
            rel = vertices.select(
                col(ID).alias(f"__v_{name}_id"),
                *[col(c).alias(f"__v_{name}_{c}") for c in v_attrs],
            )
            result = result.join(rel, left_on=name, right_on=f"__v_{name}_id", how="inner")

    for kind, name in output:
        if kind == "vertex":
            fields = [col(f"__v_{name}_id").alias(ID)]
            fields += [col(f"__v_{name}_{c}").alias(c) for c in v_attrs]
        else:
            fields = [
                col(f"__e_{name}_src").alias(SRC),
                col(f"__e_{name}_dst").alias(DST),
            ]
            fields += [col(f"__e_{name}_{c}").alias(c) for c in e_attrs]
        result = result.with_column(name, to_struct(*fields))

    return result.select(*[name for _, name in output])
