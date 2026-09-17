"""Tests for daft_graph.iterate."""

from __future__ import annotations

import daft
import pytest
from daft import col
from daft.functions import when

from daft_graph.iterate import iterate_to_fixed_point


def _decrement_to_zero(df: daft.DataFrame) -> daft.DataFrame:
    return df.select(when(col("n") > 0, col("n") - 1).otherwise(0).alias("n"))


def _increment(df: daft.DataFrame) -> daft.DataFrame:
    return df.select((col("n") + 1).alias("n"))


def _value(df: daft.DataFrame) -> int:
    return int(df.collect().to_pydict()["n"][0])


def _converged(prev: daft.DataFrame, nxt: daft.DataFrame) -> bool:
    return _value(prev) == _value(nxt)


def _never_converged(prev: daft.DataFrame, nxt: daft.DataFrame) -> bool:
    return False


def test_converges_and_counts_rounds() -> None:
    state = daft.from_pydict({"n": [5]})
    final, rounds = iterate_to_fixed_point(state, _decrement_to_zero, _converged, max_iters=30)
    assert _value(final) == 0
    assert rounds == 6
    assert rounds < 30


def test_respects_max_iters_cap() -> None:
    state = daft.from_pydict({"n": [0]})
    with pytest.warns(UserWarning):
        final, rounds = iterate_to_fixed_point(state, _increment, _never_converged, max_iters=3)
    assert rounds == 3
    assert _value(final) == 3


def test_checkpoint_dir_round_trip(tmp_path) -> None:
    state = daft.from_pydict({"n": [3]})
    final, _rounds = iterate_to_fixed_point(
        state,
        _decrement_to_zero,
        _converged,
        max_iters=30,
        checkpoint_dir=str(tmp_path),
    )
    assert _value(final) == 0
    assert any(tmp_path.iterdir())


def test_invalid_args() -> None:
    state = daft.from_pydict({"n": [1]})
    with pytest.raises(ValueError):
        iterate_to_fixed_point(state, _decrement_to_zero, _converged, max_iters=0)
    with pytest.raises(ValueError):
        iterate_to_fixed_point(state, _decrement_to_zero, _converged, materialize_every=0)


def test_checkpoint_dir_reuse_is_not_corrupted(tmp_path) -> None:
    ckpt = str(tmp_path / "ck")
    first, _ = iterate_to_fixed_point(
        daft.from_pydict({"n": [3]}),
        _decrement_to_zero,
        _converged,
        max_iters=30,
        checkpoint_dir=ckpt,
    )
    second, _ = iterate_to_fixed_point(
        daft.from_pydict({"n": [4]}),
        _decrement_to_zero,
        _converged,
        max_iters=30,
        checkpoint_dir=ckpt,
    )
    assert _value(first) == 0
    assert _value(second) == 0
    # The second run must not inherit checkpoint files from the first.
    assert second.count_rows() == 1
