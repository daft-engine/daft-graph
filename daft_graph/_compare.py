"""Internal helper for comparing DataFrame row sets during iteration."""

from __future__ import annotations

from daft import DataFrame, Expression


def rows_equal(a: DataFrame, b: DataFrame, on: list[str | Expression]) -> bool:
    """True when ``a`` and ``b`` hold the same rows over the columns ``on``.

    Uses anti join counts so the comparison stays inside the Daft engine instead
    of materializing rows into Python.
    """
    left = a.join(b, on=on, how="anti").count_rows()
    right = b.join(a, on=on, how="anti").count_rows()
    return left == 0 and right == 0
