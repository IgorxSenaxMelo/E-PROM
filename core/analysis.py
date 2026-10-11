"""Explainability, weight sensitivity and group-agreement analyses for E-PROM."""

from __future__ import annotations
import numpy as np

from .e_prom_core import (
    build_fuzzy_matrix,
    normalize_expert_weights,
    normalize_precision_matrix,
    run_e_prom,
)
from .fuzzy_numbers import defuzz_coa_array
from .labels import normalize_type
from .promethee import (
    TIE_TOL,
    preference_tensor,
    rank_from_net_flow,
    rankdata_average,
    sub_vec,
)


# --------------------------------------------------------------------------
# Shared preparation
# --------------------------------------------------------------------------

def _prepare(evaluations, w_ind, types, scales, precision_levels, expert_weights):
    names = list(evaluations.keys())
    n_exp = len(names)
    first = np.asarray(evaluations[names[0]], dtype=float)
    n_req, n_crit = first.shape
    types = [normalize_type(t) for t in types]
    precision = normalize_precision_matrix(precision_levels, n_exp, n_crit)
    ew = normalize_expert_weights(expert_weights, n_exp)
    w_ind = np.asarray(w_ind, dtype=float)

    F_list = [
        build_fuzzy_matrix(
            np.asarray(evaluations[name], dtype=float),
            types,
            scales,
            precision_levels=precision[e],
        )
        for e, name in enumerate(names)
    ]
    return names, F_list, w_ind, ew


# --------------------------------------------------------------------------
# Contribution of each criterion to the flows
# --------------------------------------------------------------------------

def flow_terms(
    evaluations, w_ind, types, directions, preference_types, q_vals, p_vals,
    scales, precision_levels=None, expert_weights=None,
):
    """Per-criterion fuzzy flows for every decision maker.

    Returns ``plus_k`` and ``minus_k`` with shape [DM, criterion, alternative, 4]
    (the criterion's weighted contribution to the positive / negative flow).
    """
    names, F_list, w_ind, ew = _prepare(
        evaluations, w_ind, types, scales, precision_levels, expert_weights
    )
    n_req = F_list[0].shape[0]
    n_crit = F_list[0].shape[1]

    plus_k = np.zeros((len(names), n_crit, n_req, 4))
    minus_k = np.zeros((len(names), n_crit, n_req, 4))

    for e, F_e in enumerate(F_list):
        PIk = preference_tensor(
            F_e, w_ind[:, :, e], directions, preference_types, q_vals, p_vals
        )
        plus_k[e] = PIk.sum(axis=2) / (n_req - 1)       # sum over j
        minus_k[e] = PIk.sum(axis=1) / (n_req - 1)      # sum over i

    return {
        "plus_k": plus_k,
        "minus_k": minus_k,
        "expert_weights": ew,
        "expert_names": names,
    }


def criterion_contributions(terms):
    """Contribution of each criterion to the net flow: array [alternative, criterion].

    contribution_k(i) = COA(phi+_k(i)) - COA(phi-_k(i)) after aggregating the
    decision makers in the fuzzy domain. Because defuzzification is not exactly
    additive, the sum of the contributions can differ slightly from the net
    flow; the difference is returned by the caller as a residual.
    """
    ew = terms["expert_weights"]
    plus = np.average(terms["plus_k"], axis=0, weights=ew)     # [K, n, 4]
    minus = np.average(terms["minus_k"], axis=0, weights=ew)
    contrib = defuzz_coa_array(plus) - defuzz_coa_array(minus)  # [K, n]
    return contrib.T


# --------------------------------------------------------------------------
# Pairwise comparison
# --------------------------------------------------------------------------

def pairwise_comparison(
    evaluations, w_ind, types, directions, preference_types, q_vals, p_vals,
    scales, i, j, precision_levels=None, expert_weights=None,
):
    """Criterion-by-criterion preference of alternative i over j (and j over i)."""
    names, F_list, w_ind, ew = _prepare(
        evaluations, w_ind, types, scales, precision_levels, expert_weights
    )
    n_crit = F_list[0].shape[1]
    ij = np.zeros((len(names), n_crit, 4))
    ji = np.zeros((len(names), n_crit, 4))

    for e, F_e in enumerate(F_list):
        PIk = preference_tensor(
            F_e, w_ind[:, :, e], directions, preference_types, q_vals, p_vals
        )
        ij[e] = PIk[:, i, j, :]
        ji[e] = PIk[:, j, i, :]

    ij = np.average(ij, axis=0, weights=ew)       # [K, 4]
    ji = np.average(ji, axis=0, weights=ew)

    pref_ij = defuzz_coa_array(ij)
    pref_ji = defuzz_coa_array(ji)

    return {
        "pref_ij": pref_ij,
        "pref_ji": pref_ji,
        "net": pref_ij - pref_ji,
        "total_ij": float(defuzz_coa_array(ij.sum(axis=0))),
        "total_ji": float(defuzz_coa_array(ji.sum(axis=0))),
    }


# --------------------------------------------------------------------------
# Weight sensitivity
# --------------------------------------------------------------------------

def _rank_key(ranks):
    return tuple(np.round(ranks, 6))


def _top_key(ranks):
    return tuple(np.where(ranks <= 1.0 + 1e-9)[0])


def weight_sensitivity(terms, w_ind, n_grid=121, a_min=1e-3, a_max=1e3):
    """Sensitivity of the ranking to the importance of each criterion.

    For criterion k the weight of every decision maker is multiplied by a
    factor ``a`` and the weights are renormalized (the other criteria keep
    their relative proportions). ``a = 1`` is the current analysis. Because the
    weighted preference is linear in the weight, the flows are recomputed
    instantly from the stored per-criterion terms.

    Returns, for every criterion, the group weight share that corresponds to
    each factor, the net flows and ranks along the sweep, and the range of the
    weight share around the current value inside which the full ranking (and the
    first place) do not change.
    """
    w_ind = np.asarray(w_ind, dtype=float)               # [K, 4, E]
    ew = terms["expert_weights"]
    plus_k, minus_k = terms["plus_k"], terms["minus_k"]  # [E, K, n, 4]
    n_exp, n_crit, n_req, _ = plus_k.shape

    means = w_ind.mean(axis=1)                           # [K, E]
    shares = (means / means.sum(axis=0, keepdims=True)).T   # [E, K]

    def evaluate(k, a):
        f = np.ones((n_exp, n_crit))
        f[:, k] = a
        norm = (f * shares).sum(axis=1)                  # [E]
        coef = ew[:, None] * f / norm[:, None]           # [E, K]
        plus = np.einsum("ek,eknc->nc", coef, plus_k)
        minus = np.einsum("ek,eknc->nc", coef, minus_k)
        phi = defuzz_coa_array(sub_vec(plus, minus))
        group_share = float(np.sum(ew * a * shares[:, k] / norm))
        return group_share, phi

    base_share = [float(np.sum(ew * shares[:, k])) for k in range(n_crit)]

    results = []
    for k in range(n_crit):
        grid = np.exp(np.linspace(np.log(a_min), np.log(a_max), n_grid))
        grid = np.unique(np.concatenate([grid, [1.0]]))
        base_idx = int(np.where(grid == 1.0)[0][0])

        sweep_share = np.zeros(len(grid))
        sweep_phi = np.zeros((n_req, len(grid)))
        sweep_rank = np.zeros((n_req, len(grid)))
        for g, a in enumerate(grid):
            sweep_share[g], sweep_phi[:, g] = evaluate(k, a)
            sweep_rank[:, g] = rank_from_net_flow(sweep_phi[:, g])

        base_rank = sweep_rank[:, base_idx]

        def boundary(key_fn, step):
            ref = key_fn(base_rank)
            g = base_idx
            while 0 <= g + step < len(grid):
                nxt = g + step
                if key_fn(sweep_rank[:, nxt]) != ref:
                    same_pt, diff_pt = np.log(grid[g]), np.log(grid[nxt])
                    for _ in range(40):
                        mid = 0.5 * (same_pt + diff_pt)
                        _, phi_mid = evaluate(k, np.exp(mid))
                        if key_fn(rank_from_net_flow(phi_mid)) == ref:
                            same_pt = mid
                        else:
                            diff_pt = mid
                    edge = np.exp(0.5 * (same_pt + diff_pt))
                    return evaluate(k, edge)[0]
                g = nxt
            return None                                  # no change in the range

        order_lo, order_hi = boundary(_rank_key, -1), boundary(_rank_key, +1)
        top_lo, top_hi = boundary(_top_key, -1), boundary(_top_key, +1)

        results.append({
            "criterion": k,
            "share_now": base_share[k],
            "order_lower": 0.0 if order_lo is None else order_lo,
            "order_upper": 1.0 if order_hi is None else order_hi,
            "top_lower": 0.0 if top_lo is None else top_lo,
            "top_upper": 1.0 if top_hi is None else top_hi,
            "order_changes_down": order_lo is not None,
            "order_changes_up": order_hi is not None,
            "top_changes_down": top_lo is not None,
            "top_changes_up": top_hi is not None,
            "sweep_share": sweep_share,
            "sweep_phi": sweep_phi,
            "sweep_rank": sweep_rank,
        })

    return results


# --------------------------------------------------------------------------
# Decision-maker agreement
# --------------------------------------------------------------------------

def kendall_w(ranks):
    """Kendall's coefficient of concordance W (with tie correction).

    ranks: [alternative, decision maker]. Returns NaN with fewer than 2 DMs.
    """
    ranks = np.asarray(ranks, dtype=float)
    n, m = ranks.shape
    if m < 2 or n < 2:
        return float("nan")

    totals = ranks.sum(axis=1)
    s = np.sum((totals - totals.mean()) ** 2)

    tie_term = 0.0
    for j in range(m):
        _, counts = np.unique(np.round(ranks[:, j], 9), return_counts=True)
        tie_term += np.sum(counts ** 3 - counts)

    den = m ** 2 * (n ** 3 - n) - m * tie_term
    if den <= 0:
        return float("nan")
    return float(12.0 * s / den)


def _spearman(a, b):
    ra = rankdata_average(a)
    rb = rankdata_average(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def expert_agreement(phi_net_experts, expert_names=None, low_threshold=0.5):
    """Agreement among decision makers' individual rankings.

    phi_net_experts: [alternative, decision maker] individual net flows.
    """
    phi = np.asarray(phi_net_experts, dtype=float)
    n, m = phi.shape
    names = list(expert_names) if expert_names is not None else [f"DM{e + 1}" for e in range(m)]

    ranks = np.column_stack([rank_from_net_flow(phi[:, e]) for e in range(m)])

    sp = np.full((m, m), np.nan)
    for a in range(m):
        for b in range(m):
            sp[a, b] = 1.0 if a == b else _spearman(phi[:, a], phi[:, b])

    mean_agreement = np.full(m, np.nan)
    for a in range(m):
        others = [sp[a, b] for b in range(m) if b != a and not np.isnan(sp[a, b])]
        if others:
            mean_agreement[a] = float(np.mean(others))

    w = kendall_w(ranks)
    if np.isnan(w):
        label = "n/a"
    elif w >= 0.6:
        label = "strong"
    elif w >= 0.3:
        label = "moderate"
    else:
        label = "weak"

    flagged = [
        names[a] for a in range(m)
        if m >= 3 and not np.isnan(mean_agreement[a]) and mean_agreement[a] < low_threshold
    ]

    return {
        "kendall_w": w,
        "kendall_w_label": label,
        "spearman": sp,
        "mean_agreement": mean_agreement,
        "ranks": ranks,
        "flagged": flagged,
        "expert_names": names,
    }


def leave_one_expert_out(
    evaluations, w_ind, types, directions, preference_types, q_vals, p_vals,
    scales, precision_levels=None, expert_weights=None, base_rank=None,
):
    """Re-run E-PROM removing one decision maker at a time."""
    names = list(evaluations.keys())
    n_exp = len(names)
    if n_exp < 2:
        return None

    w_ind = np.asarray(w_ind, dtype=float)
    n_crit = w_ind.shape[0]
    precision = normalize_precision_matrix(precision_levels, n_exp, n_crit)
    ew = normalize_expert_weights(expert_weights, n_exp)

    if base_rank is None:
        base = run_e_prom(
            evaluations, w_ind, types, directions, preference_types, q_vals,
            p_vals, scales, precision_levels=precision, expert_weights=ew,
        )
        base_rank = base["rank_e_prom"]
    base_rank = np.asarray(base_rank, dtype=float)

    rows = []
    for drop in range(n_exp):
        keep = [e for e in range(n_exp) if e != drop]
        sub_eval = {names[e]: evaluations[names[e]] for e in keep}
        res = run_e_prom(
            sub_eval,
            w_ind[:, :, keep],
            types, directions, preference_types, q_vals, p_vals, scales,
            precision_levels=[precision[e] for e in keep],
            expert_weights=ew[keep],
        )
        rank = np.asarray(res["rank_e_prom"], dtype=float)
        rows.append({
            "removed": names[drop],
            "rank": rank,
            "spearman_vs_full": _spearman(-rank, -base_rank),
            "same_order": bool(np.allclose(rank, base_rank)),
            "same_first": _top_key(rank) == _top_key(base_rank),
        })
    return rows


def rank_reversal_test(
    evaluations, w_ind, types, directions, preference_types, q_vals, p_vals,
    scales, precision_levels=None, expert_weights=None, base_rank=None,
):
    """Remove each alternative in turn and check whether the relative order of
    the remaining alternatives changes (rank reversal)."""
    names = list(evaluations.keys())
    first = np.asarray(evaluations[names[0]], dtype=float)
    n_req = first.shape[0]
    if n_req < 3:
        return None

    if base_rank is None:
        base = run_e_prom(
            evaluations, w_ind, types, directions, preference_types, q_vals,
            p_vals, scales, precision_levels=precision_levels,
            expert_weights=expert_weights,
        )
        base_rank = base["rank_e_prom"]
    base_rank = np.asarray(base_rank, dtype=float)

    rows = []
    for drop in range(n_req):
        keep = [i for i in range(n_req) if i != drop]
        sub_eval = {
            name: np.asarray(evaluations[name], dtype=float)[keep, :]
            for name in names
        }
        res = run_e_prom(
            sub_eval, w_ind, types, directions, preference_types, q_vals,
            p_vals, scales, precision_levels=precision_levels,
            expert_weights=expert_weights,
        )
        new_rank = np.asarray(res["rank_e_prom"], dtype=float)
        old_rank = base_rank[keep]

        pairs = []
        for a in range(len(keep)):
            for b in range(a + 1, len(keep)):
                before = np.sign(old_rank[b] - old_rank[a])
                after = np.sign(new_rank[b] - new_rank[a])
                if before != 0 and after != 0 and before != after:
                    pairs.append((keep[a], keep[b]))
        rows.append({"removed": drop, "reversals": pairs, "new_rank": new_rank, "kept": keep})
    return rows
