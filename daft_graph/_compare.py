"""Internal helper for comparing DataFrame row sets during iteration."""

from __future__ import annotations

from daft import DataFrame, Expression

from daft_graph.iterate import bound_partitions


def rows_equal(a: DataFrame, b: DataFrame, on: list[str | Expression]) -> bool:
    """True when ``a`` and ``b`` hold the same rows over the columns ``on``.

    Uses anti join counts so the comparison stays inside the Daft engine instead
    of materializing rows into Python. Both sides are capped with
    :func:`daft_graph.iterate.bound_partitions` first, a plan rewrite that costs
    no execution: this runs every iteration, so collecting here would add a
    distributed round trip per round, while leaving the inputs uncapped would let
    the anti join inherit a large partition count from the caller's shuffles.
    """
    a2 = bound_partitions(a)
    b2 = bound_partitions(b)
    left = a2.join(b2, on=on, how="anti").count_rows()
    right = b2.join(a2, on=on, how="anti").count_rows()
    return left == 0 and right == 0
