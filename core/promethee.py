"""Fuzzy PROMETHEE for E-PROM."""

from __future__ import annotations
import numpy as np

from .fuzzy_numbers import defuzz_coa

# Preference-function codes:
# 1 = Usual, 2 = U-shape, 3 = V-shape, 4 = Level, 5 = Linear

TIE_TOL = 1e-12


def _check_thresholds(pref_type, q, p):
    if pref_type == 3 and p <= 0:
        raise ValueError("V-shape requires p > 0.")
    if pref_type in (4, 5) and p <= q:
        raise ValueError(
            "Level requires p > q." if pref_type == 4 else "Linear requires p > q."
        )
    if pref_type not in (1, 2, 3, 4, 5):
        raise ValueError(f"Preference function not implemented: {pref_type}")


def pref_array(d, pref_type, q=0.0, p=0.0):
    """PROMETHEE preference function applied element-wise to differences."""
    d = np.maximum(np.asarray(d, dtype=float), 0.0)
    q = float(q)
    p = float(p)
    pref_type = int(pref_type)
    _check_thresholds(pref_type, q, p)

    if pref_type == 1:                       # Usual
        return (d > 0).astype(float)
    if pref_type == 2:                       # U-shape
        return (d > q).astype(float)
    if pref_type == 3:                       # V-shape
        return np.where(d <= 0, 0.0, np.where(d < p, d / p, 1.0))
    if pref_type == 4:                       # Level
        return np.where(d <= q, 0.0, np.where(d <= p, 0.5, 1.0))
    # Linear
    return np.where(
        d <= q, 0.0, np.where(d < p, (d - q) / (p - q), 1.0)
    )


def pref_scalar(d, pref_type, q=0.0, p=0.0):
    """Scalar version of :func:`pref_array`."""
    return float(pref_array(d, pref_type, q, p))


def pref_trap(D, pref_type, q=0.0, p=0.0):
    """Apply a (non-decreasing) preference function to the four corners of D."""
    D = np.asarray(D, dtype=float).reshape(4)
    return np.sort(pref_array(D, pref_type, q, p))


def _mul_geldermann_vec(W, P):
    """Vectorized Geldermann product: W is [4], P is [..., 4]."""
    ml1, mu1 = W[1], W[2]
    alpha1 = W[1] - W[0]
    beta1 = W[3] - W[2]

    ml2, mu2 = P[..., 1], P[..., 2]
    alpha2 = P[..., 1] - P[..., 0]
    beta2 = P[..., 3] - P[..., 2]

    ml = ml1 * ml2
    mu = mu1 * mu2
    alpha = np.maximum(ml1 * alpha2 + ml2 * alpha1 - alpha1 * alpha2, 0.0)
    beta = np.maximum(mu1 * beta2 + mu2 * beta1 + beta1 * beta2, 0.0)

    C = np.stack([ml - alpha, ml, mu, mu + beta], axis=-1)
    C = np.where(C < 0, 0.0, C)
    return np.sort(C, axis=-1)


def _sub_vec(X, Y):
    """Interval subtraction X - Y for arrays shaped [..., 4]."""
    return np.stack([
        X[..., 0] - Y[..., 3],
        X[..., 1] - Y[..., 2],
        X[..., 2] - Y[..., 1],
        X[..., 3] - Y[..., 0],
    ], axis=-1)


def _weighted_preference(Fk, wk, direction, pref_type, q, p):
    """w_k (x) P_k(i, j) for one criterion: array [n, n, 4] (diagonal not zeroed)."""
    Ai = Fk[:, None, :]
    Bj = Fk[None, :, :]

    D = _sub_vec(Ai, Bj) if direction == 1 else _sub_vec(Bj, Ai)

    Pk = np.sort(
        pref_array(D, int(pref_type), float(q), float(p)),
        axis=-1,
    )
    return _mul_geldermann_vec(wk, Pk)


def preference_tensor(F, w_fuzzy, crit_dir, pref_types, q_vals, p_vals):
    """Weighted fuzzy preferences per criterion: [criterion, i, j, 4].

    The diagonal (i == j) is zero. Summing over criteria gives the fuzzy
    multicriteria preference index used by :func:`run_fuzzy_promethee_trap`.
    """
    F = np.asarray(F, dtype=float)
    w_fuzzy = np.asarray(w_fuzzy, dtype=float)
    n_req, n_crit, _ = F.shape

    out = np.zeros((n_crit, n_req, n_req, 4), dtype=float)
    idx = np.arange(n_req)

    for k in range(n_crit):
        out[k] = _weighted_preference(
            F[:, k, :], w_fuzzy[k], crit_dir[k], pref_types[k], q_vals[k], p_vals[k]
        )
        out[k][idx, idx, :] = 0.0

    return out


def sub_vec(X, Y):
    """Interval subtraction X - Y for arrays shaped [..., 4]."""
    return _sub_vec(np.asarray(X, dtype=float), np.asarray(Y, dtype=float))


def run_fuzzy_promethee_trap(
    F,
    w_fuzzy,
    crit_dir,
    pref_types,
    q_vals,
    p_vals,
    defuzzify=True,
):
    """
    F: [alternative, criterion, 4] fuzzy evaluations.
    w_fuzzy: [criterion, 4] fuzzy criterion weights.

    Fuzzy weights are propagated directly into Fuzzy PROMETHEE:
        fuzzy weight (x) fuzzy preference
    and defuzzification occurs only at the flow level.
    """
    F = np.asarray(F, dtype=float)
    w_fuzzy = np.asarray(w_fuzzy, dtype=float)

    if F.ndim != 3 or F.shape[2] != 4:
        raise ValueError(
            "F must contain trapezoidal fuzzy numbers [a, b, c, d]."
        )

    n_req, n_crit, _ = F.shape

    if n_req < 2:
        raise ValueError("At least two alternatives are required.")

    if w_fuzzy.shape != (n_crit, 4):
        raise ValueError(
            f"Fuzzy weights must have shape [{n_crit}, 4]."
        )

    PI = np.zeros((n_req, n_req, 4), dtype=float)

    for k in range(n_crit):
        PI += _weighted_preference(
            F[:, k, :], w_fuzzy[k], crit_dir[k], pref_types[k], q_vals[k], p_vals[k]
        )

    idx = np.arange(n_req)
    PI[idx, idx, :] = 0.0

    avg_p = PI.sum(axis=1) / (n_req - 1)
    avg_m = PI.sum(axis=0) / (n_req - 1)
    net_f = _sub_vec(avg_p, avg_m)

    phi_plus = np.zeros(n_req)
    phi_minus = np.zeros(n_req)
    phi_net = np.zeros(n_req)

    if defuzzify:
        for i in range(n_req):
            phi_plus[i] = defuzz_coa(avg_p[i])
            phi_minus[i] = defuzz_coa(avg_m[i])
            phi_net[i] = defuzz_coa(net_f[i])

    return (
        phi_plus,
        phi_minus,
        phi_net,
        avg_p,
        avg_m,
        net_f,
    )


def rankdata_average(values, tol=TIE_TOL):
    """Average ranks (1 = smallest). Values within ``tol`` are tied."""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)

    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] - x[order[j]] <= tol:
            j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1

    return ranks


def rank_from_net_flow(phi_net):
    """Rank 1 = highest net flow; ties share the average rank."""
    return len(phi_net) + 1 - rankdata_average(phi_net)


def promethee_i_relation(plus_i, minus_i, plus_j, minus_j, tol=1e-10):
    """PROMETHEE I relation of i with respect to j: 'P', 'P-', 'I' or 'R'.

    i P j  iff  phi+(i) >= phi+(j) and phi-(i) <= phi-(j), with at least one
    strict inequality.
    """
    not_worse_plus = plus_i >= plus_j - tol
    not_worse_minus = minus_i <= minus_j + tol
    strictly_better = (plus_i > plus_j + tol) or (minus_i < minus_j - tol)

    if not_worse_plus and not_worse_minus and strictly_better:
        return "P"

    not_better_plus = plus_i <= plus_j + tol
    not_better_minus = minus_i >= minus_j - tol
    strictly_worse = (plus_i < plus_j - tol) or (minus_i > minus_j + tol)

    if not_better_plus and not_better_minus and strictly_worse:
        return "P-"

    if (
        abs(plus_i - plus_j) <= tol
        and abs(minus_i - minus_j) <= tol
    ):
        return "I"

    return "R"
