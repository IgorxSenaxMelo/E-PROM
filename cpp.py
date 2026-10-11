"""Experimental: probability of being best under normally distributed scores.

NOT used by the E-PROM application and NOT the full CPP (Composition of
Probabilistic Preferences) method: it only estimates, by simulation, how often
each alternative has the highest weighted min-max-normalized score. It is kept
as a stand-alone utility.
"""

from __future__ import annotations
import numpy as np


def cpp_normal(
    X,
    S,
    directions,
    nsim=10000,
    seed=0,
    weights=None,
):
    """
    X: [alternatives, criteria] — means / central values.
    S: [alternatives, criteria] — standard deviations.
    directions: +1 maximize, -1 minimize.
    """
    X = np.asarray(X, dtype=float)
    S = np.asarray(S, dtype=float)
    directions = np.asarray(directions, dtype=int)

    if X.shape != S.shape:
        raise ValueError("X and S must have the same shape.")

    if X.ndim != 2:
        raise ValueError("X and S must be 2-D matrices.")

    if len(directions) != X.shape[1]:
        raise ValueError("Number of directions is incompatible.")

    if np.any(S < 0):
        raise ValueError("Standard deviations cannot be negative.")

    rng = np.random.default_rng(seed)

    n_alt, n_crit = X.shape

    if weights is None:
        weights = np.ones(n_crit) / n_crit
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != n_crit:
            raise ValueError("Weights do not match the criteria.")
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Invalid weights.")
        weights = weights / weights.sum()

    samples = np.empty((int(nsim), n_alt, n_crit), dtype=float)

    for k in range(n_crit):
        samples[:, :, k] = rng.normal(
            loc=X[:, k],
            scale=S[:, k],
            size=(int(nsim), n_alt),
        )

    oriented = samples * directions[None, None, :]

    mins = oriented.min(axis=1, keepdims=True)
    maxs = oriented.max(axis=1, keepdims=True)
    span = maxs - mins

    norm = np.divide(
        oriented - mins,
        span,
        out=np.zeros_like(oriented),
        where=span > 1e-15,
    )

    scores = np.sum(norm * weights[None, None, :], axis=2)
    winners = np.argmax(scores, axis=1)

    prob_best = np.bincount(winners, minlength=n_alt) / float(nsim)

    score = prob_best.copy()
    rank_order = np.argsort(-score, kind="mergesort")

    rank = np.empty(n_alt, dtype=int)
    for r, idx in enumerate(rank_order, start=1):
        rank[idx] = r

    return {"score": score, "rank": rank, "prob_best": prob_best}
