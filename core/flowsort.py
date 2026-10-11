"""FlowSort with limiting profiles (net-flow version) on top of Fuzzy PROMETHEE.

Classes are ordered from best (index 0) to worst. They are separated by
``m = n_classes - 1`` limiting profiles, listed from the WORST boundary to the
BEST boundary. Each alternative a is compared with the profiles inside the
reference set R + {a}; it is assigned to the class whose bounding profiles
enclose its net flow:  phi(r_h) <= phi(a) < phi(r_{h+1}).

Fuzzy flows of the decision makers are aggregated in the fuzzy domain before
defuzzification, exactly as in the ranking.
"""

from __future__ import annotations
import numpy as np

from .e_prom_core import (
    build_fuzzy_matrix,
    normalize_expert_weights,
    normalize_precision_matrix,
)
from .fuzzy_numbers import defuzz_coa_array
from .labels import CONTINUOUS, ORDINAL, normalize_type
from .promethee import preference_tensor, sub_vec


def suggest_profiles(evaluations, types, directions, scales, n_classes):
    """Default limiting profiles from quantiles of the (DM-averaged) data.

    Returns an array [n_classes - 1, criteria] ordered from the worst boundary
    to the best boundary.
    """
    if n_classes < 2:
        raise ValueError("At least two classes are required.")

    data = np.mean(
        [np.asarray(v, dtype=float) for v in evaluations.values()], axis=0
    )
    types = [normalize_type(t) for t in types]
    n_crit = data.shape[1]
    m = n_classes - 1
    out = np.zeros((m, n_crit))

    for k in range(n_crit):
        for h in range(m):
            frac = (h + 1) / n_classes
            q = frac if directions[k] == 1 else 1.0 - frac
            value = float(np.quantile(data[:, k], q))
            if types[k] == ORDINAL:
                value = float(np.clip(np.rint(value), 1, int(scales[k])))
            out[h, k] = value
    return out


def _fuzzy_inputs(evaluations, profiles, types, scales, precision_matrix):
    types = [normalize_type(t) for t in types]
    names = list(evaluations.keys())
    profiles = np.asarray(profiles, dtype=float)
    F_alt = [
        build_fuzzy_matrix(
            np.asarray(evaluations[name], dtype=float), types, scales,
            precision_levels=precision_matrix[e],
        )
        for e, name in enumerate(names)
    ]
    F_prof = [
        build_fuzzy_matrix(
            profiles, types, scales, precision_levels=precision_matrix[e],
        )
        for e in range(len(names))
    ]
    return F_alt, F_prof


def flowsort_from_fuzzy(
    F_alt_by_dm, F_prof_by_dm, w_by_dm, directions, preference_types,
    q_vals, p_vals, expert_weights=None, per_dm=False, tol=1e-9,
):
    """Core FlowSort computation from fuzzy evaluation matrices.

    F_alt_by_dm: list over DMs of [n, K, 4]; F_prof_by_dm: list of [m, K, 4];
    w_by_dm: list of [K, 4] fuzzy weights (one per DM).
    """
    n_exp = len(F_alt_by_dm)
    n = F_alt_by_dm[0].shape[0]
    m = F_prof_by_dm[0].shape[0]
    ew = normalize_expert_weights(expert_weights, n_exp)

    net_a = np.zeros((n_exp, n, 4))
    net_h = np.zeros((n_exp, n, m, 4))
    net_rr = np.zeros((n_exp, m, 4))

    for e in range(n_exp):
        F_all = np.concatenate([F_alt_by_dm[e], F_prof_by_dm[e]], axis=0)
        PI = preference_tensor(
            F_all, w_by_dm[e], directions, preference_types, q_vals, p_vals
        ).sum(axis=0)                                   # [n+m, n+m, 4]

        PI_ar = PI[:n, n:]                              # a -> r   [n, m, 4]
        PI_ra = PI[n:, :n]                              # r -> a   [m, n, 4]
        PI_rr = PI[n:, n:]                              # r -> r'  [m, m, 4]

        net_a[e] = sub_vec(PI_ar.sum(axis=1), PI_ra.sum(axis=0)) / m

        out = PI_rr.sum(axis=1)[None, :, :] + PI_ra.transpose(1, 0, 2)
        inn = PI_rr.sum(axis=0)[None, :, :] + PI_ar
        net_h[e] = sub_vec(out, inn) / m

        if m >= 2:
            net_rr[e] = sub_vec(PI_rr.sum(axis=1), PI_rr.sum(axis=0)) / (m - 1)

    def classify(a_fuzzy, h_fuzzy):
        phi_a = defuzz_coa_array(a_fuzzy)
        phi_h = defuzz_coa_array(h_fuzzy)
        exceeded = (phi_h <= phi_a[:, None] + tol).sum(axis=1)
        return m - exceeded, phi_a, phi_h

    cls, phi_a, phi_h = classify(
        np.average(net_a, axis=0, weights=ew),
        np.average(net_h, axis=0, weights=ew),
    )

    out = {
        "class_index": cls,                             # 0 = best class
        "phi": phi_a,
        "phi_profiles": phi_h,
    }

    if m >= 2:
        prof_phi = defuzz_coa_array(np.average(net_rr, axis=0, weights=ew))
        out["profile_phi"] = prof_phi
        out["profile_order_ok"] = bool(np.all(np.diff(prof_phi) > tol))
    else:
        out["profile_phi"] = np.zeros(m)
        out["profile_order_ok"] = True

    if per_dm:
        out["class_per_dm"] = np.array([
            classify(net_a[e], net_h[e])[0] for e in range(n_exp)
        ])                                              # [DM, n]
    return out


def run_flowsort(
    evaluations, w_fuzzy_individual, profiles, types, directions,
    preference_types, q_vals, p_vals, scales, precision_levels=None,
    expert_weights=None,
):
    """Deterministic FlowSort for the current analysis."""
    names = list(evaluations.keys())
    n_exp = len(names)
    w_ind = np.asarray(w_fuzzy_individual, dtype=float)
    n_crit = w_ind.shape[0]

    profiles = np.asarray(profiles, dtype=float)
    if profiles.ndim != 2 or profiles.shape[1] != n_crit or profiles.shape[0] < 1:
        raise ValueError(
            f"Profiles must be a matrix [profiles, {n_crit} criteria]."
        )

    precision = normalize_precision_matrix(precision_levels, n_exp, n_crit)
    F_alt, F_prof = _fuzzy_inputs(evaluations, profiles, types, scales, precision)

    res = flowsort_from_fuzzy(
        F_alt, F_prof, [w_ind[:, :, e] for e in range(n_exp)],
        directions, preference_types, q_vals, p_vals,
        expert_weights=expert_weights, per_dm=True,
    )
    res["expert_names"] = names
    return res
