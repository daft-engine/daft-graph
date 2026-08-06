"""SVD++ matrix factorization on a bipartite rating graph (Koren 2008).

This is the GraphX ``svdPlusPlus`` algorithm: a recommender style factorization
of a user/item rating graph with global mean, per node biases, latent factors,
and an implicit feedback term. Edges are ``src`` (user) to ``dst`` (item) with a
rating column. It is recsys flavored rather than a structural graph algorithm;
the ratings are collected and trained with numpy SGD, which fits moderate rating
graphs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import daft
from daft import DataFrame, col

from daft_graph._optional import require_numpy
from daft_graph.graph import DirectedGraph
from daft_graph.schema import DST, ID, SRC

KIND = "kind"
BIAS = "bias"
FACTOR = "factor"
_RATING = "_r"


@dataclass(frozen=True)
class SvdPlusPlusResult:
    """Learned SVD++ parameters and the final training RMSE.

    Attributes:
        factors: DataFrame ``[id, kind, bias, factor]`` for every user and item.
        global_mean: The global mean rating ``mu``.
        rmse: Root mean squared error on the training ratings after fitting.
    """

    factors: DataFrame
    global_mean: float
    rmse: float


def svd_plus_plus(
    graph: DirectedGraph,
    *,
    rating_column: str = "rating",
    rank: int = 8,
    epochs: int = 20,
    learning_rate: float = 0.01,
    regularization: float = 0.05,
    seed: int = 0,
) -> SvdPlusPlusResult:
    """Fit SVD++ on the user/item rating graph (``src`` users, ``dst`` items).

    Takes a :class:`DirectedGraph` because the rating edges are inherently
    directional (``src`` is the user, ``dst`` is the item). Unlike pagerank or
    SCC the body does not orient or symmetrize; the type documents the column
    convention rather than an ``_orient`` based behavior difference.

    Raises:
        ImportError: If the optional ``local`` extra is not installed.
    """
    require_numpy("svd_plus_plus")
    import numpy as np

    rows = graph.edges.select(SRC, DST, col(rating_column).alias(_RATING)).collect().to_pydict()
    user_ids = sorted({int(u) for u in rows[SRC]})
    item_ids = sorted({int(i) for i in rows[DST]})
    uidx = {u: i for i, u in enumerate(user_ids)}
    iidx = {it: j for j, it in enumerate(item_ids)}
    edges = [(uidx[int(u)], iidx[int(it)], float(r)) for u, it, r in zip(rows[SRC], rows[DST], rows[_RATING])]
    if not edges:
        return SvdPlusPlusResult(
            factors=daft.from_pydict({ID: [], KIND: [], BIAS: [], FACTOR: []}),
            global_mean=0.0,
            rmse=0.0,
        )
    n_users, n_items = len(user_ids), len(item_ids)
    mu = sum(r for _, _, r in edges) / len(edges)

    rng = np.random.default_rng(seed)
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    p_user = rng.normal(0.0, 0.1, (n_users, rank))
    q_item = rng.normal(0.0, 0.1, (n_items, rank))
    y_item = rng.normal(0.0, 0.1, (n_items, rank))

    rated: dict[int, list[int]] = defaultdict(list)
    for u, it, _ in edges:
        rated[u].append(it)

    lr, reg = learning_rate, regularization
    for _ in range(epochs):
        for u, it, r in edges:
            items_u = rated[u]
            scale = 1.0 / np.sqrt(len(items_u))
            implicit = scale * y_item[items_u].sum(axis=0)
            pred = mu + bu[u] + bi[it] + q_item[it].dot(p_user[u] + implicit)
            err = r - pred
            bu[u] += lr * (err - reg * bu[u])
            bi[it] += lr * (err - reg * bi[it])
            q_old = q_item[it].copy()
            p_user[u] += lr * (err * q_item[it] - reg * p_user[u])
            q_item[it] += lr * (err * (p_user[u] + implicit) - reg * q_item[it])
            y_item[items_u] += lr * (err * scale * q_old - reg * y_item[items_u])

    squared_error = 0.0
    for u, it, r in edges:
        items_u = rated[u]
        implicit = (1.0 / np.sqrt(len(items_u))) * y_item[items_u].sum(axis=0)
        pred = mu + bu[u] + bi[it] + q_item[it].dot(p_user[u] + implicit)
        squared_error += (r - pred) ** 2
    rmse = float(np.sqrt(squared_error / len(edges)))

    factors_df = daft.from_pydict(
        {
            ID: user_ids + item_ids,
            KIND: ["user"] * n_users + ["item"] * n_items,
            BIAS: [float(b) for b in bu] + [float(b) for b in bi],
            FACTOR: (
                [[float(x) for x in p_user[i]] for i in range(n_users)]
                + [[float(x) for x in q_item[j]] for j in range(n_items)]
            ),
        }
    )
    return SvdPlusPlusResult(factors=factors_df, global_mean=float(mu), rmse=rmse)
