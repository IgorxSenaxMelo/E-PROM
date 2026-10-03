"""Fuzzy PROMETHEE do SiPREM."""

from __future__ import annotations
import math
import numpy as np

from .fuzzy_numbers import trap_add, trap_sub, trap_scalar_div, trap_mul_geldermann, defuzz_coa


def pref_scalar(d, pref_type, q, p):
    d = max(0.0, float(d))

    if pref_type == 1:  # Usual
        return float(d > 0)
    if pref_type == 2:  # U-shape
        return 0.0 if d <= q else 1.0
    if pref_type == 3:  # V-shape
        if d <= 0:
            return 0.0
        if d <= p:
            return d/p
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
        if d <= p:
            return (d-q)/(p-q)
        return 1.0
    if pref_type == 6:  # Gaussian
        if d <= 0:
            return 0.0
        sigma = p/3
        return 1 - math.exp(-(d*d)/(2*sigma*sigma))

    raise ValueError(f"Tipo de preferência não implementado: {pref_type}")


def pref_trap(D, pref_type, q, p):
    out = np.array([pref_scalar(x, pref_type, q, p) for x in D], dtype=float)
    return np.sort(out)


def run_fuzzy_promethee_trap(F, w_fuzzy, crit_dir, pref_types, q_vals, p_vals):
    F = np.asarray(F, dtype=float)
    w_fuzzy = np.asarray(w_fuzzy, dtype=float)
    n_req, n_crit, n_comp = F.shape

    if n_comp != 4:
        raise ValueError("F deve conter trapézios [a,b,c,d].")

    PI = np.zeros((n_req, n_req, 4), dtype=float)

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

                Pk = pref_trap(D, pref_types[k], q_vals[k], p_vals[k])
                term = trap_mul_geldermann(w_fuzzy[k], Pk)
                pi_ij = trap_add(pi_ij, term)

            PI[i, j] = pi_ij

    phi_plus_fuzzy = np.zeros((n_req, 4))
    phi_minus_fuzzy = np.zeros((n_req, 4))
    phi_net_fuzzy = np.zeros((n_req, 4))
    phi_plus = np.zeros(n_req)
    phi_minus = np.zeros(n_req)
    phi_net = np.zeros(n_req)

    for i in range(n_req):
        sum_p = np.zeros(4)
        sum_m = np.zeros(4)

        for j in range(n_req):
            if i == j:
                continue
            sum_p = trap_add(sum_p, PI[i, j])
            sum_m = trap_add(sum_m, PI[j, i])

        avg_p = trap_scalar_div(sum_p, n_req-1)
        avg_m = trap_scalar_div(sum_m, n_req-1)
        net_f = trap_sub(avg_p, avg_m)

        phi_plus_fuzzy[i] = avg_p
        phi_minus_fuzzy[i] = avg_m
        phi_net_fuzzy[i] = net_f

        phi_plus[i] = defuzz_coa(avg_p)
        phi_minus[i] = defuzz_coa(avg_m)
        phi_net[i] = defuzz_coa(net_f)

    return phi_plus, phi_minus, phi_net, phi_plus_fuzzy, phi_minus_fuzzy, phi_net_fuzzy


def rankdata_average(values):
    """Equivalente ao tiedrank do MATLAB para empates: rank médio, crescente."""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j+1]] == x[order[i]]:
            j += 1
        ranks[order[i:j+1]] = (i+1 + j+1) / 2
        i = j + 1
    return ranks


def rank_from_net_flow(phi_net):
    n = len(phi_net)
    return n + 1 - rankdata_average(phi_net)
