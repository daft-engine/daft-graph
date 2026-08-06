"""Message passing primitives: aggregate_messages (one round) and pregel (loop).

These generalize the pattern the built in algorithms use: join the current
vertex state onto the edges to form triplets, send messages along edges, and
aggregate them at the receiving vertices. ``pregel`` runs this to a fixed point
on the shared iteration engine. This is the analog of the GraphFrames
``aggregateMessages`` and Pregel APIs.

State is a vertex DataFrame keyed by ``id``. Its non id columns are exposed on
the triplets as ``src_<col>`` and ``dst_<col>``, so message expressions can read
both endpoints, e.g. ``col("src_value") / col("src_degree")``.
"""

from __future__ import annotations

from collections.abc import Callable

from daft import DataFrame, Expression, col

from daft_graph._compare import rows_equal
from daft_graph.iterate import ConvergedFn, iterate_to_fixed_point
from daft_graph.schema import DST, ID, SRC

VALUE = "value"
MSG = "msg"

AggFn = Callable[[Expression], Expression]


def _default_agg(msg: Expression) -> Expression:
    return msg.sum()


def _triplets(edges: DataFrame, state: DataFrame) -> DataFrame:
    """Join vertex state onto both endpoints, prefixing columns src_ and dst_."""
    state_cols = [c for c in state.column_names if c != ID]
    reserved = [c for c in state_cols if c.startswith(("src_", "dst_"))]
    if reserved:
        raise ValueError(f"state columns must not start with 'src_' or 'dst_': {reserved}")
    src_state = state.select(col(ID).alias(SRC), *[col(c).alias(f"src_{c}") for c in state_cols])
    dst_state = state.select(col(ID).alias(DST), *[col(c).alias(f"dst_{c}") for c in state_cols])
    return edges.join(src_state, on=SRC, how="inner").join(dst_state, on=DST, how="inner")


def aggregate_messages(
    edges: DataFrame,
    state: DataFrame,
    *,
    to_src: Expression | None = None,
    to_dst: Expression | None = None,
    agg: AggFn | None = None,
) -> DataFrame:
    """One round of message passing along edges, aggregated at recipients.

    Args:
        edges: Edge DataFrame with ``src`` and ``dst`` columns.
        state: Vertex DataFrame keyed by ``id``; non id columns appear on the
            triplets as ``src_<col>`` and ``dst_<col>``.
        to_src: Message expression sent to each edge's source, or None.
        to_dst: Message expression sent to each edge's destination, or None.
        agg: Aggregator applied to the ``msg`` column per recipient (default sum).

    Returns:
        A DataFrame ``[id, msg]`` with one row per vertex that received a message.

    Raises:
        ValueError: If neither ``to_src`` nor ``to_dst`` is given.
    """
    if to_src is None and to_dst is None:
        raise ValueError("at least one of to_src or to_dst must be provided")
    aggregator = agg or _default_agg
    triplets = _triplets(edges, state)
    parts: list[DataFrame] = []
    if to_dst is not None:
        parts.append(triplets.select(col(DST).alias(ID), to_dst.alias(MSG)))
    if to_src is not None:
        parts.append(triplets.select(col(SRC).alias(ID), to_src.alias(MSG)))
    combined = parts[0]
    for part in parts[1:]:
        combined = combined.union_all(part)
    return combined.groupby(ID).agg(aggregator(col(MSG)).alias(MSG))


def pregel(
    edges: DataFrame,
    init_state: DataFrame,
    *,
    update: Expression,
    to_src: Expression | None = None,
    to_dst: Expression | None = None,
    agg: AggFn | None = None,
    max_iters: int = 20,
    converged: ConvergedFn | None = None,
    materialize_every: int = 1,
    checkpoint_dir: str | None = None,
) -> DataFrame:
    """Run message passing to a fixed point, updating a ``value`` column.

    Each round sends messages (built from ``src_<col>``/``dst_<col>`` triplet
    columns), aggregates them into ``msg`` per vertex, then sets ``value`` to
    ``update`` (an expression over ``value`` and ``msg``; ``msg`` is null for
    vertices that received nothing). Other columns are carried unchanged.

    Args:
        edges: Edge DataFrame with ``src`` and ``dst`` columns.
        init_state: Initial vertex state ``[id, value, ...]``.
        update: Expression over ``value`` and ``msg`` giving the new value.
        to_src: Message expression sent to each edge's source, or None.
        to_dst: Message expression sent to each edge's destination, or None.
        agg: Aggregator for the message column (default sum).
        max_iters: Maximum rounds.
        converged: Optional ``(prev, next) -> bool``; default exact equality on
            ``[id, value]``.
        materialize_every: How often to truncate the Daft plan between rounds.
        checkpoint_dir: Optional parquet checkpoint directory.

    Returns:
        The final vertex state DataFrame, same schema as ``init_state``.
    """
    extra_cols = [c for c in init_state.column_names if c not in (ID, VALUE)]

    def step(state: DataFrame) -> DataFrame:
        msgs = aggregate_messages(edges, state, to_src=to_src, to_dst=to_dst, agg=agg)
        return state.join(msgs, on=ID, how="left").select(col(ID), update.alias(VALUE), *[col(c) for c in extra_cols])

    converged_fn = converged or (lambda prev, nxt: rows_equal(prev, nxt, [ID, VALUE]))
    final, _ = iterate_to_fixed_point(
        init_state,
        step,
        converged_fn,
        max_iters=max_iters,
        materialize_every=materialize_every,
        checkpoint_dir=checkpoint_dir,
    )
    return final
