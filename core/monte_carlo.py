"""Monte Carlo stability analysis for E-PROM.

Ordinal criteria:
    empirical sampling of the decision makers' answers (as in MC-SIPREM).

Continuous criteria:
    uniform perturbation of each observed value:
    x_MC ~ U[x(1-v), x(1+v)], where v is given in percent.

Criterion weights:
    each iteration bootstraps the decision makers' FAHP-Express answers
    (whole reference rows, so the DMs' different reference criteria stay
    consistent) and recomputes the fuzzy weights.
"""

from __future__ import annotations
import numpy as np

from .e_prom_core import (
    _validate_preferences,
    build_fuzzy_matrix,
    normalize_expert_weights,
    normalize_precision_matrix,
)
from .fahp import (
    calc_fahp_buckley,
    crisp_to_fuzzy_trap_interp,
    weights_from_reference_row,
)
from .fuzzy_numbers import defuzz_coa
from .labels import CONTINUOUS, ORDINAL, normalize_type
from .flowsort import _fuzzy_inputs, flowsort_from_fuzzy
from .promethee import TIE_TOL, rank_from_net_flow, run_fuzzy_promethee_trap

WEIGHT_SAMPLING_MODES = ("dm_bootstrap", "independent")


def _independent_weights(rows, rng):
    """Legacy sampling: every criterion draws its value from a random DM.

    Only consistent when all DMs use the same reference criterion.
    """
    n_exp, n_crit = rows.shape
    picks = rng.integers(0, n_exp, size=n_crit)
    sampled = rows[picks, np.arange(n_crit)]

    A = sampled[None, :] / sampled[:, None]
    M = np.zeros((n_crit, n_crit, 4))
    for i in range(n_crit):
        for j in range(n_crit):
            M[i, j] = crisp_to_fuzzy_trap_interp(A[i, j])
    return calc_fahp_buckley(M)


def _add_rank_shares(phi, acc, tol=TIE_TOL):
    """Add one iteration to the rank-position counts.

    Tied alternatives split their share evenly over the tied positions, so
    every row of ``acc`` keeps summing to the number of iterations.
    """
    order = np.argsort(-phi, kind="mergesort")
    n = len(phi)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(phi[order[j + 1]] - phi[order[j]]) <= tol:
            j += 1
        share = 1.0 / (j - i + 1)
        for idx in order[i:j + 1]:
            acc[idx, i:j + 1] += share
        i = j + 1


def run_e_prom_monte_carlo(
    evaluations,
    reference_rows,
    types,
    directions,
    preference_types,
    q_vals,
    p_vals,
    scales,
    continuous_variation_pct,
    n_mc=1000,
    seed=42,
    precision_levels=None,
    precision_support_pct=None,
    precision_core_pct=None,
    weight_sampling="dm_bootstrap",
    expert_weights=None,
    flowsort_profiles=None,
):
    """Stability analysis of E-PROM.

    For every iteration:
      1. resample the criterion weights (see ``weight_sampling``);
      2. for every decision maker:
         - ordinal: draw one of the answers observed for that
           alternative/criterion across decision makers;
         - continuous: perturb the observed value by +-v %;
      3. run individual Fuzzy PROMETHEE;
      4. aggregate the fuzzy flows of the decision makers;
      5. defuzzify, rank and accumulate the statistics.

    ``precision_levels`` must have shape (decision makers, criteria).

    If ``flowsort_profiles`` ([profiles, criteria], worst boundary first) is
    given, every iteration also classifies the alternatives with FlowSort and
    the result includes ``class_acceptability`` (% of simulations in each class,
    class 0 = best).
    """
    if weight_sampling not in WEIGHT_SAMPLING_MODES:
        raise ValueError(
            f"weight_sampling must be one of {WEIGHT_SAMPLING_MODES}."
        )

    rng = np.random.default_rng(seed)

    experts = list(evaluations.keys())
    n_exp = len(experts)
    stack = np.stack(
        [np.asarray(evaluations[name], dtype=float) for name in experts]
    )                                                    # [DM, alt, crit]
    _, n_req, n_crit = stack.shape

    types = [normalize_type(t) for t in types]
    _validate_preferences(types, preference_types, q_vals, p_vals)

    variation = np.asarray(continuous_variation_pct, dtype=float)
    if len(variation) != n_crit:
        raise ValueError("Continuous variations do not match the criteria.")
    if np.any(variation < 0):
        raise ValueError("The percentage variation cannot be negative.")

    reference_rows = np.asarray(reference_rows, dtype=float)
    if reference_rows.shape != (n_exp, n_crit):
        raise ValueError(
            f"reference_rows must have shape ({n_exp}, {n_crit})."
        )

    precision_matrix = normalize_precision_matrix(precision_levels, n_exp, n_crit)
    ew = normalize_expert_weights(expert_weights, n_exp)

    ordinal_cols = [k for k, t in enumerate(types) if t == ORDINAL]
    continuous_cols = [k for k, t in enumerate(types) if t == CONTINUOUS]

    # Per-DM fuzzy weights (criterion order = canonical order).
    dm_fuzzy = np.stack([
        weights_from_reference_row(row)[1] for row in reference_rows
    ])                                                   # [DM, crit, 4]

    phi_mc = np.zeros((n_req, n_mc))
    phi_plus_mc = np.zeros((n_req, n_mc))
    phi_minus_mc = np.zeros((n_req, n_mc))
    rank_mc = np.zeros((n_req, n_mc))
    rank_counts = np.zeros((n_req, n_req))
    outranking = np.zeros((n_req, n_req))
    tol = 1e-10

    W_samples = np.zeros((n_mc, n_crit))

    if flowsort_profiles is not None:
        flowsort_profiles = np.asarray(flowsort_profiles, dtype=float)
        _, F_prof_list = _fuzzy_inputs(
            evaluations, flowsort_profiles, types, scales, precision_matrix
        )
        n_classes = flowsort_profiles.shape[0] + 1
        class_counts = np.zeros((n_req, n_classes))

    v_frac = variation / 100.0
    alt_idx = np.arange(n_req)[:, None]

    for mc in range(n_mc):

        # 1) weights
        if weight_sampling == "dm_bootstrap":
            draw = rng.integers(0, n_exp, size=n_exp)
            w_fuzzy_group = dm_fuzzy[draw].mean(axis=0)
            w_crisp = w_fuzzy_group.mean(axis=1)
            w_crisp = w_crisp / w_crisp.sum()
        else:
            w_crisp, w_fuzzy_group = _independent_weights(reference_rows, rng)
        W_samples[mc] = w_crisp

        phi_plus_fuzzy_exp = np.zeros((n_req, 4, n_exp))
        phi_minus_fuzzy_exp = np.zeros((n_req, 4, n_exp))
        phi_net_fuzzy_exp = np.zeros((n_req, 4, n_exp))
        F_alt_list = []

        # 2) attributes per decision maker
        for e in range(n_exp):
            data_mc = stack[e].copy()

            if ordinal_cols:
                picks = rng.integers(0, n_exp, size=(n_req, len(ordinal_cols)))
                data_mc[:, ordinal_cols] = stack[picks, alt_idx, ordinal_cols]

            for k in continuous_cols:
                x = stack[e][:, k]
                lo = x * (1.0 - v_frac[k])
                hi = x * (1.0 + v_frac[k])
                lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
                data_mc[:, k] = rng.uniform(lo, hi)

            F_e = build_fuzzy_matrix(
                data_mc,
                types,
                scales,
                precision_levels=precision_matrix[e],
                precision_support_pct=precision_support_pct,
                precision_core_pct=precision_core_pct,
            )

            F_alt_list.append(F_e)

            _, _, _, pp_f, pm_f, pn_f = run_fuzzy_promethee_trap(
                F_e,
                w_fuzzy_group,
                directions,
                preference_types,
                q_vals,
                p_vals,
                defuzzify=False,
            )

            phi_plus_fuzzy_exp[:, :, e] = pp_f
            phi_minus_fuzzy_exp[:, :, e] = pm_f
            phi_net_fuzzy_exp[:, :, e] = pn_f

        if flowsort_profiles is not None:
            fs = flowsort_from_fuzzy(
                F_alt_list, F_prof_list, [w_fuzzy_group] * n_exp,
                directions, preference_types, q_vals, p_vals,
                expert_weights=ew,
            )
            class_counts[np.arange(n_req), fs["class_index"]] += 1

        # 3) aggregate in the fuzzy domain, defuzzify afterwards
        plus_f = np.average(phi_plus_fuzzy_exp, axis=2, weights=ew)
        minus_f = np.average(phi_minus_fuzzy_exp, axis=2, weights=ew)
        net_f = np.average(phi_net_fuzzy_exp, axis=2, weights=ew)

        pp = np.array([defuzz_coa(x) for x in plus_f])
        pm = np.array([defuzz_coa(x) for x in minus_f])
        phi = np.array([defuzz_coa(x) for x in net_f])

        phi_plus_mc[:, mc] = pp
        phi_minus_mc[:, mc] = pm
        phi_mc[:, mc] = phi
        rank_mc[:, mc] = rank_from_net_flow(phi)
        _add_rank_shares(phi, rank_counts)

        # PROMETHEE I: i outranks j if it is not worse on both flows and
        # strictly better on at least one.
        not_worse_plus = pp[:, None] >= pp[None, :] - tol
        not_worse_minus = pm[:, None] <= pm[None, :] + tol
        strictly_better = (
            (pp[:, None] > pp[None, :] + tol)
            | (pm[:, None] < pm[None, :] - tol)
        )
        outranking += (not_worse_plus & not_worse_minus & strictly_better)

    rank_acceptability = 100.0 * rank_counts / n_mc
    outranking = 100.0 * outranking / n_mc

    ddof = 1 if n_mc > 1 else 0

    extra = {}
    if flowsort_profiles is not None:
        extra["class_acceptability"] = 100.0 * class_counts / n_mc
        extra["flowsort_profiles"] = flowsort_profiles

    return {
        **extra,
        "phi_mc": phi_mc,
        "phi_plus_mc": phi_plus_mc,
        "phi_minus_mc": phi_minus_mc,
        "rank_mc": rank_mc,
        "mean_rank": np.mean(rank_mc, axis=1),
        "std_rank": np.std(rank_mc, axis=1, ddof=ddof),
        "mean_phi": np.mean(phi_mc, axis=1),
        "std_phi": np.std(phi_mc, axis=1, ddof=ddof),
        # Ties for 1st place split their credit evenly.
        "freq_top1": rank_counts[:, 0] / n_mc,
        "rank_acceptability": rank_acceptability,
        "outranking_acceptability": outranking,
        "W_samples": W_samples,
        "n_mc": n_mc,
        "seed": seed,
        "weight_sampling": weight_sampling,
        "continuous_variation_pct": variation,
        "precision_levels": precision_matrix,
    }
