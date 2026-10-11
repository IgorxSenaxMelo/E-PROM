"""Trapezoidal FAHP + Buckley weights for E-PROM (FAHP-Express)."""

from __future__ import annotations
import numpy as np

from .labels import parse_importance

# Fuzzy representation of the 1-9 linguistic importance scale.
FAHP_SCALE = np.array([
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.5, 2.5, 3.0],
    [2.0, 2.5, 3.5, 4.0],
    [3.0, 3.5, 4.5, 5.0],
    [4.0, 4.5, 5.5, 6.0],
    [5.0, 5.5, 6.5, 7.0],
    [6.0, 6.5, 7.5, 8.0],
    [7.0, 7.5, 8.5, 9.0],
    [8.0, 8.5, 9.0, 9.0],
], dtype=float)


def saaty_linguistic2num(x):
    """Linguistic term (English or Portuguese), number or numeric string -> float."""
    return parse_importance(x)


def crisp_to_fuzzy_trap_interp(x):
    x = float(x)

    if not np.isfinite(x) or x <= 0:
        raise ValueError("A Saaty comparison must be positive and finite.")
    if x < 1 / 9 - 1e-12 or x > 9 + 1e-12:
        raise ValueError(
            f"Reconstructed comparison {x:.6f} is outside the 1/9-9 Saaty scale. "
            "Check the FAHP-Express values (the ratio between two criteria "
            "cannot exceed 9)."
        )

    if abs(x - 1) < 1e-12:
        return np.array([1., 1., 1., 1.])

    if x < 1:
        direct = crisp_to_fuzzy_trap_interp(1 / x)
        return np.array([
            1 / direct[3],
            1 / direct[2],
            1 / direct[1],
            1 / direct[0],
        ])

    if abs(x - round(x)) < 1e-12:
        return FAHP_SCALE[int(round(x)) - 1].copy()

    lower = int(np.floor(x))
    upper = int(np.ceil(x))
    alpha = (x - lower) / (upper - lower)

    return (1 - alpha) * FAHP_SCALE[lower - 1] + alpha * FAHP_SCALE[upper - 1]


def calc_fahp_buckley(m_fuzzy):
    m_fuzzy = np.asarray(m_fuzzy, dtype=float)

    n = m_fuzzy.shape[0]
    if m_fuzzy.shape != (n, n, 4):
        raise ValueError("The fuzzy matrix must have shape [n, n, 4].")

    r = np.ones((n, 4), dtype=float)

    for i in range(n):
        for j in range(n):
            r[i] *= m_fuzzy[i, j]
        r[i] = r[i] ** (1 / n)

    sum_r = np.sum(r, axis=0)

    inv_sum = np.array([
        1 / sum_r[3],
        1 / sum_r[2],
        1 / sum_r[1],
        1 / sum_r[0],
    ])

    w_fuzzy = r * inv_sum

    w_crisp = np.mean(w_fuzzy, axis=1)
    w_crisp = w_crisp / np.sum(w_crisp)

    return w_crisp, w_fuzzy


def weights_from_reference_row(reference_values):
    """Rebuild the AHP matrix from one FAHP-Express reference row.

    The reference criterion has value 1. Every other value says how many times
    the reference is more important than that criterion. The matrix is
    A_ij = v_j / v_i (so A_ref,j = v_j and A_j,ref = 1 / v_j). The result does
    not depend on which column holds the reference, so the row can be given in
    any criterion order.
    """
    vals = np.array(
        [saaty_linguistic2num(x) for x in reference_values],
        dtype=float,
    )

    n = len(vals)

    if n < 2:
        raise ValueError("FAHP-Express requires at least 2 criteria.")

    A = vals[None, :] / vals[:, None]

    m_fuzzy = np.zeros((n, n, 4), dtype=float)
    for i in range(n):
        for j in range(n):
            m_fuzzy[i, j] = crisp_to_fuzzy_trap_interp(A[i, j])

    wc, wf = calc_fahp_buckley(m_fuzzy)

    return wc, wf, A, m_fuzzy


def group_weights_from_reference_rows(rows):
    """Individual and group fuzzy weights from one reference row per DM."""
    individual_fuzzy = []
    individual_crisp = []
    matrices = []
    fuzzy_matrices = []

    for row in rows:
        wc, wf, A, mf = weights_from_reference_row(row)

        individual_crisp.append(wc)
        individual_fuzzy.append(wf)
        matrices.append(A)
        fuzzy_matrices.append(mf)

    individual_fuzzy = np.stack(individual_fuzzy, axis=2)    # [criteria, 4, DMs]
    individual_crisp = np.stack(individual_crisp, axis=1)    # [criteria, DMs]

    group_fuzzy = np.mean(individual_fuzzy, axis=2)
    group_crisp = np.mean(group_fuzzy, axis=1)
    group_crisp = group_crisp / np.sum(group_crisp)

    return {
        "w_fuzzy_group": group_fuzzy,
        "w_crisp_group": group_crisp,
        "w_fuzzy_individual": individual_fuzzy,
        "w_crisp_individual": individual_crisp,
        "A_individual": matrices,
        "m_fuzzy_individual": fuzzy_matrices,
    }
