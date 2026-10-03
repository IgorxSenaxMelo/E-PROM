"""CPP — Composition of Probabilistic Preferences.

Módulo independente do E-PROM principal para critérios contínuos.
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
    X: [alternativas, critérios] — médias/valores centrais.
    S: [alternativas, critérios] — desvios-padrão.
    directions: +1 maximiza, -1 minimiza.
    """
    X = np.asarray(X, dtype=float)
    S = np.asarray(S, dtype=float)
    directions = np.asarray(directions, dtype=int)

    if X.shape != S.shape:
        raise ValueError("X e S devem possuir a mesma dimensão.")

    if X.ndim != 2:
        raise ValueError("X e S devem ser matrizes 2D.")

    if len(directions) != X.shape[1]:
        raise ValueError("Número de direções incompatível.")

    if np.any(S < 0):
        raise ValueError("Desvios-padrão não podem ser negativos.")

    rng = np.random.default_rng(seed)

    n_alt, n_crit = X.shape

    if weights is None:
        weights = np.ones(n_crit) / n_crit
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != n_crit:
            raise ValueError("Pesos incompatíveis com os critérios.")
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Pesos inválidos.")
        weights = weights / weights.sum()

    samples = np.empty(
        (int(nsim), n_alt, n_crit),
        dtype=float,
    )

    for k in range(n_crit):
        samples[:, :, k] = rng.normal(
            loc=X[:, k],
            scale=S[:, k],
            size=(int(nsim), n_alt),
        )

    oriented = samples * directions[None, None, :]

    # For each simulation, rank alternatives by the weighted
    # normalized oriented performance.
    mins = oriented.min(axis=1, keepdims=True)
    maxs = oriented.max(axis=1, keepdims=True)
    span = maxs - mins

    norm = np.divide(
        oriented - mins,
        span,
        out=np.zeros_like(oriented),
        where=span > 1e-15,
    )

    scores = np.sum(
        norm * weights[None, None, :],
        axis=2,
    )

    winners = np.argmax(
        scores,
        axis=1,
    )

    prob_best = np.bincount(
        winners,
        minlength=n_alt,
    ) / float(nsim)

    score = prob_best.copy()

    rank_order = np.argsort(
        -score,
        kind="mergesort",
    )

    rank = np.empty(
        n_alt,
        dtype=int,
    )

    for r, idx in enumerate(rank_order, start=1):
        rank[idx] = r

    return {
        "score": score,
        "rank": rank,
        "prob_best": prob_best,
    }
