"""Iteration engine for fixed point graph algorithms.

The core job of this module is to truncate Daft's logical plan between rounds.
Iterative joins otherwise accumulate an ever growing plan and memory blows up,
the same failure Spark GraphFrames solves with checkpointing. Materializing the
state with ``.collect()`` starts a fresh plan; optionally the state is round
tripped through parquet for a stronger break on very long iterations.

Materializing also has to bound the state's *partition count*. Each round's
joins and unions grow the partition count (``union_all`` sums it, so a self
referential step doubles it every round), and ``.collect()`` preserves that
count in the materialized result. On the native runner partitions are invisible,
but on the distributed (Ray/Flotilla) runner the count compounds - 200 -> 400 ->
800 -> ... -> thousands - and each shuffle then needs a partition-count-squared
number of pieces, which exhausts head node memory and stalls the job even on a
tiny graph. So every materialize coalesces the state back to a count scaled to
its row count and hard capped (see :data:`_MAX_PARTITIONS`). This mirrors what
GraphFrames does when it checkpoints and coalesces; the intra-round shuffles
still parallelize freely, only the carried-over state is bounded.

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

#: Rows per partition used to scale the coalesced state to its size.
_TARGET_ROWS_PER_PARTITION = 50_000
#: Hard ceiling on the carried-over state's partition count, so an iterative
#: step can never compound partitions round over round no matter the runner.
_MAX_PARTITIONS = 256
#: Ceiling applied to intermediate frames inside a single round. Daft's shuffles
#: inherit their input's partition count (a repartition with no explicit count
#: resolves to ``input_num_partitions``), so capping an intermediate caps every
#: shuffle downstream of it in that round.
_MAX_INTERMEDIATE_PARTITIONS = 16


def _supports_partitioning(df: DataFrame) -> bool:
    """True when the active runner exposes a partition count for ``df``.

    The native runner has no partitions: ``num_partitions`` returns None there and
    ``into_partitions`` is a documented no-op that warns. Gating on this keeps the
    single node path free of both the warning and the pointless plan node, while
    the distributed runner (where the partition count is the whole problem) gets
    the bounding.
    """
    return df.num_partitions() is not None


def bound_partitions(df: DataFrame, cap: int = _MAX_INTERMEDIATE_PARTITIONS) -> DataFrame:
    """Cap ``df``'s planned partition count without executing anything.

    ``DataFrame.num_partitions`` inspects the physical plan rather than running
    it, and ``into_partitions`` only merges partitions, so this is a pure plan
    rewrite: no job, no round trip. Use it on intermediate frames inside an
    iterative step, where an extra ``collect`` would cost a distributed round
    trip every round.

    Why it is needed: Daft resolves a shuffle's output partition count to the
    repartition spec's count *or else the input's* count, and never lowers it. A
    ``union_all`` sums its inputs' counts, so a step that symmetrizes or unions
    doubles the count every round and each shuffle then needs a
    count-squared number of pieces. Capping the intermediate breaks that chain.

    Args:
        df: The frame to cap. May be lazy; nothing is materialized.
        cap: Maximum partition count to allow through.

    Returns:
        ``df`` unchanged when its planned count is unknown or already within
        ``cap``, otherwise ``df`` coalesced to ``cap`` partitions.
    """
    planned = df.num_partitions()
    if planned is None or planned <= cap:
        return df
    return df.into_partitions(cap)


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
        raise ValueError(f"max_iters must be >= 1, got {max_iters}")
    if materialize_every < 1:
        raise ValueError(f"materialize_every must be >= 1, got {materialize_every}")

    current = collect_bounded(state)
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


def _bounded_partition_count(df: DataFrame) -> int:
    """Partitions to coalesce a materialized state into: scaled to rows, capped.

    Called on an already-materialized frame, so ``count_rows`` is a cheap
    metadata read rather than a job. The result is at least one partition, grows
    one partition per :data:`_TARGET_ROWS_PER_PARTITION` rows, and never exceeds
    :data:`_MAX_PARTITIONS` - which is what stops the round-over-round compounding.
    """
    rows = df.count_rows()
    if rows <= 0:
        return 1
    scaled = (rows + _TARGET_ROWS_PER_PARTITION - 1) // _TARGET_ROWS_PER_PARTITION
    return max(1, min(_MAX_PARTITIONS, scaled))


def collect_bounded(df: DataFrame) -> DataFrame:
    """Materialize ``df`` once, then present it at a size-scaled, capped count.

    Collects to truncate the plan, then returns ``into_partitions`` *lazily* on the
    materialized result rather than re-collecting. This is deliberate: a second
    ``.collect()`` returns a frame whose ``num_partitions`` reports 0 (a
    materialized frame has no clustering spec), which would make every downstream
    :func:`bound_partitions` check see 0, conclude "already small", and skip the
    cap - silently disabling the whole mechanism in a loop. Returning the lazy
    ``into_partitions`` keeps the reported count at the bound, so counts flow
    correctly through the union_all/join chain that follows. The base is
    materialized, so the coalesce reads cached partitions and does not replan.

    Use for loop state and once-collected static inputs (adjacency, degrees) that
    are joined every round. For purely intermediate frames use
    :func:`bound_partitions`, which needs no execution at all.
    """
    collected = df.collect()
    if not _supports_partitioning(collected):
        return collected
    return collected.into_partitions(_bounded_partition_count(collected))


def _materialize(df: DataFrame, checkpoint_dir: str | None, round_index: int) -> DataFrame:
    """Truncate the plan and present the state at a bounded partition count.

    Collects once to break the logical plan, then coalesces *lazily* (see
    :func:`collect_bounded` for why re-collecting would zero the reported count
    and disable downstream caps). The coalesce is fused into the next round's
    execution over cached partitions, so bounding the state adds no distributed
    round trip - which matters because this runs every round.
    """
    collected = df.collect()
    bounded = (
        collected.into_partitions(_bounded_partition_count(collected))
        if _supports_partitioning(collected)
        else collected
    )
    if checkpoint_dir is None:
        return bounded
    path = os.path.join(checkpoint_dir, f"round_{round_index}")
    bounded.write_parquet(path, write_mode="overwrite")
    return daft.read_parquet(path)
