"""Iteration engine for fixed point graph algorithms.

The core job of this module is to truncate Daft's logical plan between rounds.
Iterative joins otherwise accumulate an ever growing plan and memory blows up,
the same failure Spark GraphFrames solves with checkpointing. Materializing the
state with ``.collect()`` starts a fresh plan; optionally the state is round
tripped through parquet for a stronger break on very long iterations.

Static inputs (adjacency, out degrees) should be materialized by the caller
before building the step closure, so they are not replanned every round.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable

import daft
from daft import DataFrame

StepFn = Callable[[DataFrame], DataFrame]
ConvergedFn = Callable[[DataFrame, DataFrame], bool]


def iterate_to_fixed_point(
    state: DataFrame,
    step_fn: StepFn,
    converged_fn: ConvergedFn,
    *,
    max_iters: int = 30,
    materialize_every: int = 1,
    checkpoint_dir: str | None = None,
) -> tuple[DataFrame, int]:
    """Run ``step_fn`` until ``converged_fn`` is true or ``max_iters`` is reached.

    Between rounds the state is materialized to truncate the Daft logical plan,
    which is what keeps iterative joins from growing an unbounded plan. This is
    the analog of Spark GraphFrames checkpointing.

    With ``materialize_every > 1`` the state is collected only every N rounds, so
    ``converged_fn`` may receive un-materialized frames on intermediate rounds
    (its own aggregation still forces evaluation). If the loop does not converge
    within ``max_iters`` a warning is emitted.

    Args:
        state: The initial DataFrame state.
        step_fn: Maps the current state to the next state.
        converged_fn: Given ``(previous_state, next_state)``, returns True to stop.
        max_iters: Maximum number of rounds before giving up.
        materialize_every: Materialize the state every N rounds. Must be >= 1.
        checkpoint_dir: If set, round trip the materialized state through parquet
            here for a stronger plan break on very long iterations.

    Returns:
        A tuple of ``(final_state, num_rounds_run)``.

    Raises:
        ValueError: If ``max_iters`` or ``materialize_every`` is less than 1.
    """
    if max_iters < 1:
        raise ValueError("max_iters must be >= 1")
    if materialize_every < 1:
        raise ValueError("materialize_every must be >= 1")

    current = state.collect()
    rounds = 0
    converged = False
    for i in range(max_iters):
        nxt = step_fn(current)
        if (i + 1) % materialize_every == 0:
            nxt = _materialize(nxt, checkpoint_dir, i)
        rounds += 1
        if converged_fn(current, nxt):
            current = nxt
            converged = True
            break
        current = nxt
    if not converged:
        warnings.warn(
            f"iterate_to_fixed_point did not converge within {max_iters} rounds",
            stacklevel=2,
        )
    return current, rounds


def _materialize(df: DataFrame, checkpoint_dir: str | None, round_index: int) -> DataFrame:
    """Truncate the plan by collecting, optionally via a parquet round trip."""
    if checkpoint_dir is None:
        return df.collect()
    path = os.path.join(checkpoint_dir, f"round_{round_index}")
    df.write_parquet(path, write_mode="overwrite")
    return daft.read_parquet(path)
