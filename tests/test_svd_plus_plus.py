"""Tests for svd_plus_plus on a synthetic low-rank rating graph."""

from __future__ import annotations

import daft
import numpy as np

from daft_graph.algorithms.svd_plus_plus import BIAS, FACTOR, KIND, svd_plus_plus
from daft_graph.graph import DirectedGraph
from daft_graph.schema import DST, ID, SRC

_ITEM_OFFSET = 1000


def _synthetic_rating_graph() -> DirectedGraph:
    rng = np.random.default_rng(0)
    n_users, n_items, dim = 15, 12, 3
    p = rng.normal(0.0, 1.0, (n_users, dim))
    q = rng.normal(0.0, 1.0, (n_items, dim))
    mu = 3.0
    users, items, ratings = [], [], []
    for u in range(n_users):
        for it in range(n_items):
            users.append(u)
            items.append(_ITEM_OFFSET + it)
            ratings.append(float(mu + p[u].dot(q[it])))
    vertices = daft.from_pydict({ID: list(range(n_users)) + [_ITEM_OFFSET + i for i in range(n_items)]})
    edges = daft.from_pydict({SRC: users, DST: items, "rating": ratings})
    return DirectedGraph(vertices=vertices, edges=edges)


def test_recovers_low_rank_ratings() -> None:
    g = _synthetic_rating_graph()
    result = svd_plus_plus(g, rank=5, epochs=60, learning_rate=0.02, regularization=0.02, seed=0)
    # the ratings are exactly low rank, so the fit should be tight
    assert result.rmse < 0.5
    assert abs(result.global_mean - 3.0) < 1.0


def test_more_epochs_reduce_error() -> None:
    g = _synthetic_rating_graph()
    few = svd_plus_plus(g, rank=5, epochs=2, learning_rate=0.02, seed=0)
    many = svd_plus_plus(g, rank=5, epochs=60, learning_rate=0.02, seed=0)
    assert many.rmse < few.rmse


def test_factor_schema() -> None:
    g = _synthetic_rating_graph()
    result = svd_plus_plus(g, rank=4, epochs=5, seed=0)
    assert set(result.factors.column_names) == {ID, KIND, BIAS, FACTOR}
    kinds = set(result.factors.select(KIND).distinct().collect().to_pydict()[KIND])
    assert kinds == {"user", "item"}


def test_deterministic() -> None:
    g = _synthetic_rating_graph()
    a = svd_plus_plus(g, rank=4, epochs=10, seed=7)
    b = svd_plus_plus(g, rank=4, epochs=10, seed=7)
    assert a.rmse == b.rmse


def test_empty_rating_graph() -> None:
    vertices = daft.from_pydict({ID: [0, 1]})
    edges = daft.from_pydict({SRC: [], DST: [], "rating": []})
    result = svd_plus_plus(DirectedGraph(vertices=vertices, edges=edges))
    assert result.rmse == 0.0
    assert result.global_mean == 0.0
    assert result.factors.count_rows() == 0
