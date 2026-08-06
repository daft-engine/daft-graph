"""Tests for daft_graph.message_passing."""

from __future__ import annotations

import daft
import pytest
from daft import col
from daft.functions import when

from daft_graph.message_passing import MSG, VALUE, aggregate_messages, pregel
from daft_graph.schema import DST, ID, SRC


def _min_update() -> object:
    return when(col(MSG).is_null(), col(VALUE)).otherwise(when(col(VALUE) <= col(MSG), col(VALUE)).otherwise(col(MSG)))


def test_aggregate_messages_to_dst() -> None:
    edges = daft.from_pydict({SRC: [0, 1], DST: [1, 2]})
    state = daft.from_pydict({ID: [0, 1, 2], VALUE: [10, 20, 30]})
    d = aggregate_messages(edges, state, to_dst=col("src_value")).collect().to_pydict()
    assert dict(zip(d[ID], d[MSG])) == {1: 10, 2: 20}


def test_aggregate_messages_to_src() -> None:
    edges = daft.from_pydict({SRC: [0, 1], DST: [1, 2]})
    state = daft.from_pydict({ID: [0, 1, 2], VALUE: [10, 20, 30]})
    d = aggregate_messages(edges, state, to_src=col("dst_value")).collect().to_pydict()
    assert dict(zip(d[ID], d[MSG])) == {0: 20, 1: 30}


def test_aggregate_messages_both_directions() -> None:
    edges = daft.from_pydict({SRC: [0, 1], DST: [1, 2]})
    state = daft.from_pydict({ID: [0, 1, 2], VALUE: [10, 20, 30]})
    d = aggregate_messages(edges, state, to_src=col("dst_value"), to_dst=col("src_value")).collect().to_pydict()
    # 0 <- dst_value(0,1)=20; 1 <- src_value(0,1)=10 + dst_value(1,2)=30; 2 <- src_value(1,2)=20
    assert dict(zip(d[ID], d[MSG])) == {0: 20, 1: 40, 2: 20}


def test_aggregate_messages_custom_agg_min() -> None:
    edges = daft.from_pydict({SRC: [0, 1], DST: [2, 2]})
    state = daft.from_pydict({ID: [0, 1, 2], VALUE: [5, 3, 99]})
    d = aggregate_messages(edges, state, to_dst=col("src_value"), agg=lambda m: m.min()).collect().to_pydict()
    assert dict(zip(d[ID], d[MSG])) == {2: 3}


def test_aggregate_messages_requires_direction() -> None:
    edges = daft.from_pydict({SRC: [0], DST: [1]})
    state = daft.from_pydict({ID: [0, 1], VALUE: [1, 2]})
    with pytest.raises(ValueError):
        aggregate_messages(edges, state)


def test_reserved_state_column_raises() -> None:
    edges = daft.from_pydict({SRC: [0], DST: [1]})
    state = daft.from_pydict({ID: [0, 1], VALUE: [1, 2], "src_bad": [9, 9]})
    with pytest.raises(ValueError):
        aggregate_messages(edges, state, to_dst=col("src_value"))


def test_pregel_min_label_propagation() -> None:
    edges = daft.from_pydict({SRC: [0, 1], DST: [1, 2]})
    init = daft.from_pydict({ID: [0, 1, 2], VALUE: [0, 1, 2]})
    result = pregel(
        edges,
        init,
        to_src=col("dst_value"),
        to_dst=col("src_value"),
        agg=lambda m: m.min(),
        update=_min_update(),
        max_iters=10,
    )
    d = result.collect().to_pydict()
    assert dict(zip(d[ID], d[VALUE])) == {0: 0, 1: 0, 2: 0}


def test_pregel_carries_extra_columns() -> None:
    edges = daft.from_pydict({SRC: [0], DST: [1]})
    init = daft.from_pydict({ID: [0, 1], VALUE: [0, 1], "tag": [100, 200]})
    result = pregel(
        edges,
        init,
        to_dst=col("src_value"),
        agg=lambda m: m.min(),
        update=_min_update(),
        max_iters=5,
    )
    d = result.collect().to_pydict()
    assert "tag" in d
    assert dict(zip(d[ID], d["tag"])) == {0: 100, 1: 200}
