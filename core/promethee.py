"""Fuzzy PROMETHEE para o E-PROM."""

from __future__ import annotations
import math
import numpy as np

from .fuzzy_numbers import (
    trap_add,
    trap_sub,
    trap_scalar_div,
    trap_mul_geldermann,
    defuzz_coa,
)


# Codes:
# 1 = Usual
# 2 = U-shape
# 3 = V-shape
# 4 = Level
# 5 = Linear


def pref_scalar(d, pref_type, q=0.0, p=0.0):
    """PROMETHEE preference function applied to a non-negative difference."""
    d = max(0.0, float(d))
    q = float(q)
    p = float(p)

    if pref_type == 1:  # Usual
        return float(d > 0)

    if pref_type == 2:  # U-shape
        if d <= q:
            return 0.0
        return 1.0

    if pref_type == 3:  # V-shape
        if d <= 0:
            return 0.0
        if p <= 0:
            raise ValueError("V-shape requer p > 0.")
        if d < p:
            return d / p
        return 1.0

    if pref_type == 4:  # Level
        if d <= q:
            return 0.0
        if d <= p:
            return 0.5
        return 1.0

    if pref_type == 5:  # Linear
        if d <= q:
            return 0.0
        if p <= q:
            raise ValueError("Linear requer p > q.")
        if d < p:
            return (d - q) / (p - q)
        return 1.0

    raise ValueError(
        f"Tipo de preferência não implementado: {pref_type}"
    )


def pref_trap(D, pref_type, q=0.0, p=0.0):
    """
    Apply a PROMETHEE scalar preference function to the four
    endpoints of a fuzzy difference.
    """
    D = np.asarray(D, dtype=float).reshape(4)
    out = np.array([
        pref_scalar(x, pref_type, q, p)
        for x in D
    ], dtype=float)

    return np.sort(out)


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

    Fuzzy weights are directly propagated into Fuzzy PROMETHEE:
        fuzzy weight ⊗ fuzzy preference
    and defuzzification occurs only at the flow level.
    """
    F = np.asarray(F, dtype=float)
    w_fuzzy = np.asarray(w_fuzzy, dtype=float)

    n_req, n_crit, n_comp = F.shape

    if n_comp != 4:
        raise ValueError(
            "F deve conter números fuzzy trapezoidais [a,b,c,d]."
        )

    if w_fuzzy.shape != (n_crit, 4):
        raise ValueError(
            f"Pesos fuzzy devem possuir dimensão [{n_crit},4]."
        )

    PI = np.zeros(
        (n_req, n_req, 4),
        dtype=float,
    )

    for i in range(n_req):

        for j in range(n_req):

            if i == j:
                continue

            pi_ij = np.zeros(4, dtype=float)

            for k in range(n_crit):

                Ai = F[i, k]
                Bj = F[j, k]

                if crit_dir[k] == 1:
                    D = trap_sub(Ai, Bj)
                else:
                    D = trap_sub(Bj, Ai)

                Pk = pref_trap(
                    D,
                    int(pref_types[k]),
                    float(q_vals[k]),
                    float(p_vals[k]),
                )

                term = trap_mul_geldermann(
                    w_fuzzy[k],
                    Pk,
                )

                pi_ij = trap_add(
                    pi_ij,
                    term,
                )

            PI[i, j] = pi_ij

    phi_plus_fuzzy = np.zeros(
        (n_req, 4)
    )
    phi_minus_fuzzy = np.zeros(
        (n_req, 4)
    )
    phi_net_fuzzy = np.zeros(
        (n_req, 4)
    )

    phi_plus = np.zeros(n_req)
    phi_minus = np.zeros(n_req)
    phi_net = np.zeros(n_req)

    for i in range(n_req):

        sum_p = np.zeros(4)
        sum_m = np.zeros(4)

        for j in range(n_req):

            if i == j:
                continue

            sum_p = trap_add(
                sum_p,
                PI[i, j],
            )

            sum_m = trap_add(
                sum_m,
                PI[j, i],
            )

        avg_p = trap_scalar_div(
            sum_p,
            n_req - 1,
        )

        avg_m = trap_scalar_div(
            sum_m,
            n_req - 1,
        )

        net_f = trap_sub(
            avg_p,
            avg_m,
        )

        phi_plus_fuzzy[i] = avg_p
        phi_minus_fuzzy[i] = avg_m
        phi_net_fuzzy[i] = net_f

        if defuzzify:
            phi_plus[i] = defuzz_coa(avg_p)
            phi_minus[i] = defuzz_coa(avg_m)
            phi_net[i] = defuzz_coa(net_f)

    return (
        phi_plus,
        phi_minus,
        phi_net,
        phi_plus_fuzzy,
        phi_minus_fuzzy,
        phi_net_fuzzy,
    )


def rankdata_average(values):
    x = np.asarray(values, dtype=float)

    order = np.argsort(
        x,
        kind="mergesort",
    )

    ranks = np.empty(
        len(x),
        dtype=float,
    )

    i = 0

    while i < len(x):

        j = i

        while (
            j + 1 < len(x)
            and x[order[j + 1]] == x[order[i]]
        ):
            j += 1

        ranks[order[i:j + 1]] = (
            (i + 1 + j + 1) / 2
        )

        i = j + 1

    return ranks


def rank_from_net_flow(phi_net):
    """Rank 1 = highest net flow."""
    return (
        len(phi_net)
        + 1
        - rankdata_average(phi_net)
    )
