"""Mathematical core of E-PROM."""

from __future__ import annotations
import numpy as np

from .fuzzy_numbers import defuzz_coa
from .labels import (
    CONTINUOUS,
    HIGH,
    LOW,
    MEDIUM,
    ORDINAL,
    normalize_precision,
    normalize_type,
)
from .promethee import (
    run_fuzzy_promethee_trap,
    rank_from_net_flow,
)


# Ordinal fuzzy evaluation scales (Medium-precision representation).
TFN_EVAL_5 = np.array([
    [1.0, 1.0, 1.5, 2.0],
    [1.5, 2.0, 2.0, 2.5],
    [2.0, 2.5, 3.0, 3.5],
    [3.0, 3.5, 4.0, 4.5],
    [4.0, 4.5, 5.0, 5.0],
])

TFN_EVAL_7 = np.array([
    [1.0, 1.0, 1.5, 2.0],
    [1.0, 1.5, 2.5, 3.0],
    [2.0, 2.5, 3.5, 4.0],
    [3.0, 3.5, 4.5, 5.0],
    [4.0, 4.5, 5.5, 6.0],
    [5.0, 5.5, 6.5, 7.0],
    [6.0, 6.5, 7.0, 7.0],
])

TFN_EVAL_9 = np.array([
    [1.0, 1.0, 1.5, 2.0],
    [1.0, 1.5, 2.5, 3.0],
    [2.0, 2.5, 3.5, 4.0],
    [3.0, 3.5, 4.5, 5.0],
    [4.0, 4.5, 5.5, 6.0],      # level 5 (was [4, 4.5, 5, 5.5], an asymmetric typo)
    [5.0, 5.5, 6.5, 7.0],
    [6.0, 6.5, 7.5, 8.0],
    [7.0, 7.5, 8.5, 9.0],
    [8.0, 8.5, 9.0, 9.0],
])

ORDINAL_PRECISION_FACTORS = {HIGH: 0.5, MEDIUM: 1.0, LOW: 1.5}


def fuzzify_ordinal(value, scale_max, precision=MEDIUM):
    """Convert an ordinal evaluation into a trapezoidal fuzzy number.

    The standard linguistic table is the Medium-precision representation.
    The precision level contracts or dilates it around the observed value
    (keeping the limits of the scale):

        High   -> smaller fuzzy uncertainty
        Medium -> standard linguistic representation
        Low    -> larger fuzzy uncertainty
    """
    value = float(value)
    scale_max = int(scale_max)
    if abs(value - round(value)) > 1e-9:
        raise ValueError(f"Ordinal value {value} is not an integer.")
    v = int(round(value))
    if v < 1 or v > scale_max:
        raise ValueError(f"Ordinal value {v} is outside the scale 1-{scale_max}.")
    if scale_max == 5:
        base = TFN_EVAL_5[v - 1].copy()
    elif scale_max == 7:
        base = TFN_EVAL_7[v - 1].copy()
    elif scale_max == 9:
        base = TFN_EVAL_9[v - 1].copy()
    else:
        raise ValueError("Supported ordinal scales: 5, 7 or 9.")

    factor = ORDINAL_PRECISION_FACTORS[normalize_precision(precision)]
    fuzzy = v + factor * (base - v)
    fuzzy = np.clip(fuzzy, 1.0, float(scale_max))
    return np.sort(fuzzy.astype(float))


PRECISION_DEFAULTS = {
    # (support half-width %, core half-width %), relative to the observed value.
    # Monotonic by design: higher precision -> smaller fuzzy dispersion.
    HIGH: (5.0, 2.0),
    MEDIUM: (10.0, 4.0),
    LOW: (100.0 / 6.0, 20.0 / 3.0),
}


def continuous_to_fuzzy(value, precision=HIGH, support_pct=None, core_pct=None):
    """Convert a continuous value into a relative trapezoidal fuzzy number.

        x -> [x(1-r_s), x(1-r_c), x(1+r_c), x(1+r_s)]

    r_s = outer half-width (support), r_c = inner half-width (core).
    The attribute stays in its original unit.
    """
    x = float(value)
    if not np.isfinite(x):
        raise ValueError("Invalid continuous value.")

    if support_pct is None or core_pct is None:
        support_pct, core_pct = PRECISION_DEFAULTS[normalize_precision(precision)]

    rs = abs(float(support_pct)) / 100.0
    rc = abs(float(core_pct)) / 100.0

    if rc > rs:
        raise ValueError("The core half-width cannot exceed the support half-width.")

    scale = abs(x)
    if scale == 0:
        return np.zeros(4, dtype=float)

    return np.sort(np.array([
        x - scale * rs,
        x - scale * rc,
        x + scale * rc,
        x + scale * rs,
    ], dtype=float))


def _ordinal_table(scale_max):
    scale_max = int(scale_max)
    if scale_max == 5:
        return TFN_EVAL_5
    if scale_max == 7:
        return TFN_EVAL_7
    if scale_max == 9:
        return TFN_EVAL_9
    raise ValueError("Supported ordinal scales: 5, 7 or 9.")


def _fuzzify_ordinal_column(values, scale_max, precision):
    """Vectorized :func:`fuzzify_ordinal` for one criterion column."""
    values = np.asarray(values, dtype=float)
    table = _ordinal_table(scale_max)
    scale_max = int(scale_max)

    if np.any(np.abs(values - np.rint(values)) > 1e-9):
        bad = values[np.abs(values - np.rint(values)) > 1e-9][0]
        raise ValueError(f"Ordinal value {bad} is not an integer.")
    v = np.rint(values).astype(int)
    if np.any(v < 1) or np.any(v > scale_max):
        bad = v[(v < 1) | (v > scale_max)][0]
        raise ValueError(f"Ordinal value {bad} is outside the scale 1-{scale_max}.")

    factor = ORDINAL_PRECISION_FACTORS[normalize_precision(precision)]
    base = table[v - 1]
    fuzzy = v[:, None] + factor * (base - v[:, None])
    fuzzy = np.clip(fuzzy, 1.0, float(scale_max))
    return np.sort(fuzzy, axis=1)


def _continuous_column(values, precision, support_pct, core_pct):
    """Vectorized :func:`continuous_to_fuzzy` for one criterion column."""
    x = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Invalid continuous value.")

    if support_pct is None or core_pct is None:
        support_pct, core_pct = PRECISION_DEFAULTS[normalize_precision(precision)]

    rs = abs(float(support_pct)) / 100.0
    rc = abs(float(core_pct)) / 100.0
    if rc > rs:
        raise ValueError("The core half-width cannot exceed the support half-width.")

    scale = np.abs(x)
    return np.sort(np.stack([
        x - scale * rs,
        x - scale * rc,
        x + scale * rc,
        x + scale * rs,
    ], axis=1), axis=1)


def build_fuzzy_matrix(
    data_numeric,
    types,
    scales,
    precision_levels=None,
    precision_support_pct=None,
    precision_core_pct=None,
):
    """Ordinal -> linguistic trapezoid; Continuous -> precision-based trapezoid."""
    data_numeric = np.asarray(data_numeric, dtype=float)
    n_req, n_crit = data_numeric.shape

    if len(types) != n_crit:
        raise ValueError("Number of data types differs from the number of criteria.")

    types = [normalize_type(t) for t in types]

    F = np.zeros((n_req, n_crit, 4), dtype=float)

    if precision_levels is None:
        precision_levels = [HIGH] * n_crit
    if len(precision_levels) != n_crit:
        raise ValueError("Number of precision levels is incompatible.")

    if precision_support_pct is None:
        precision_support_pct = [None] * n_crit
    if precision_core_pct is None:
        precision_core_pct = [None] * n_crit

    for k in range(n_crit):
        col = data_numeric[:, k]
        try:
            if types[k] == ORDINAL:
                F[:, k, :] = _fuzzify_ordinal_column(col, scales[k], precision_levels[k])
            else:
                F[:, k, :] = _continuous_column(
                    col,
                    precision_levels[k],
                    precision_support_pct[k],
                    precision_core_pct[k],
                )
        except ValueError as exc:
            raise ValueError(f"Criterion {k + 1}: {exc}") from None

    return F


def _validate_preferences(types, pref_types, q_vals, p_vals):
    for k, typ in enumerate(types):
        code = int(pref_types[k])
        q = float(q_vals[k])
        p = float(p_vals[k])

        if typ == ORDINAL:
            if code not in (1, 4):
                raise ValueError("Ordinal criteria accept only Usual or Level.")
            if code == 4 and p <= q:
                raise ValueError(f"Criterion {k + 1}: Level requires p > q.")

        elif typ == CONTINUOUS:
            if code not in (2, 3, 5):
                raise ValueError(
                    "Continuous criteria accept only U-shape, V-shape or Linear."
                )
            if code == 2 and q < 0:
                raise ValueError(f"Criterion {k + 1}: U-shape requires q >= 0.")
            if code == 3 and p <= 0:
                raise ValueError(f"Criterion {k + 1}: V-shape requires p > 0.")
            if code == 5 and (q < 0 or p <= q):
                raise ValueError(
                    f"Criterion {k + 1}: Linear requires 0 <= q < p."
                )
        else:
            raise ValueError(f"Unrecognized data type: {typ!r}")


def normalize_precision_matrix(precision_levels, n_exp, n_crit):
    """Return a [DM][criterion] list of canonical precision labels."""
    if precision_levels is None:
        return [[HIGH] * n_crit for _ in range(n_exp)]

    arr = np.asarray(precision_levels, dtype=object)
    if arr.ndim == 1:
        if len(arr) != n_crit:
            raise ValueError("Number of precision levels is incompatible.")
        rows = [list(arr) for _ in range(n_exp)]
    elif arr.ndim == 2:
        if arr.shape != (n_exp, n_crit):
            raise ValueError(
                "The precision matrix must have shape "
                f"(decision makers, criteria) = ({n_exp}, {n_crit}); "
                f"got {arr.shape}."
            )
        rows = arr.tolist()
    else:
        raise ValueError("Invalid precision format.")

    return [[normalize_precision(x) for x in row] for row in rows]


def run_e_prom(
    evaluations,
    weights_fuzzy_individual,
    types,
    directions,
    preference_types,
    q_vals,
    p_vals,
    scales,
    precision_levels=None,
    precision_support_pct=None,
    precision_core_pct=None,
):
    """Run E-PROM for each decision maker and aggregate the fuzzy flows.

    The fuzzy FAHP-Express weights are kept fuzzy and sent directly to Fuzzy
    PROMETHEE. Defuzzification happens only on the individual flows and on the
    aggregated flows.
    """
    expert_names = list(evaluations.keys())

    if not expert_names:
        raise ValueError("No decision maker found.")

    n_exp = len(expert_names)
    n_req, n_crit = np.asarray(evaluations[expert_names[0]]).shape

    if len(types) != n_crit:
        raise ValueError(
            "The configuration has a different number of criteria than the "
            "decision matrix."
        )
    if len(directions) != n_crit:
        raise ValueError("Number of directions is incompatible.")
    if len(preference_types) != n_crit:
        raise ValueError("Number of preference functions is incompatible.")
    if len(q_vals) != n_crit or len(p_vals) != n_crit:
        raise ValueError("Number of q/p thresholds is incompatible.")
    if len(scales) != n_crit:
        raise ValueError("Number of scales is incompatible.")

    types = [normalize_type(t) for t in types]
    precision_matrix = normalize_precision_matrix(precision_levels, n_exp, n_crit)

    weights_fuzzy_individual = np.asarray(weights_fuzzy_individual, dtype=float)
    if weights_fuzzy_individual.shape != (n_crit, 4, n_exp):
        raise ValueError(
            "Fuzzy weight dimensions are incompatible with the number of "
            f"criteria/decision makers: {weights_fuzzy_individual.shape}; "
            f"expected ({n_crit}, 4, {n_exp})."
        )

    _validate_preferences(types, preference_types, q_vals, p_vals)

    phi_plus_exp = np.zeros((n_req, n_exp))
    phi_minus_exp = np.zeros((n_req, n_exp))
    phi_net_exp = np.zeros((n_req, n_exp))

    phi_plus_fuzzy_exp = np.zeros((n_req, 4, n_exp))
    phi_minus_fuzzy_exp = np.zeros((n_req, 4, n_exp))
    phi_net_fuzzy_exp = np.zeros((n_req, 4, n_exp))

    for e, expert in enumerate(expert_names):
        data_e = np.asarray(evaluations[expert], dtype=float)

        F_e = build_fuzzy_matrix(
            data_e,
            types,
            scales,
            precision_levels=precision_matrix[e],
            precision_support_pct=precision_support_pct,
            precision_core_pct=precision_core_pct,
        )

        pp, pm, pn, ppf, pmf, pnf = run_fuzzy_promethee_trap(
            F_e,
            weights_fuzzy_individual[:, :, e],
            directions,
            preference_types,
            q_vals,
            p_vals,
        )

        phi_plus_exp[:, e] = pp
        phi_minus_exp[:, e] = pm
        phi_net_exp[:, e] = pn

        phi_plus_fuzzy_exp[:, :, e] = ppf
        phi_minus_fuzzy_exp[:, :, e] = pmf
        phi_net_fuzzy_exp[:, :, e] = pnf

    # A-posteriori aggregation of the fuzzy flows.
    plus_fuzzy = np.mean(phi_plus_fuzzy_exp, axis=2)
    minus_fuzzy = np.mean(phi_minus_fuzzy_exp, axis=2)
    net_fuzzy = np.mean(phi_net_fuzzy_exp, axis=2)

    plus = np.array([defuzz_coa(x) for x in plus_fuzzy])
    minus = np.array([defuzz_coa(x) for x in minus_fuzzy])
    net = np.array([defuzz_coa(x) for x in net_fuzzy])

    rank_experts = np.column_stack([
        rank_from_net_flow(phi_net_exp[:, e]) for e in range(n_exp)
    ])

    rank_e_prom = rank_from_net_flow(net)

    return {
        "expert_names": expert_names,
        "phi_plus_experts": phi_plus_exp,
        "phi_minus_experts": phi_minus_exp,
        "phi_net_experts": phi_net_exp,
        "phi_plus_fuzzy_experts": phi_plus_fuzzy_exp,
        "phi_minus_fuzzy_experts": phi_minus_fuzzy_exp,
        "phi_net_fuzzy_experts": phi_net_fuzzy_exp,
        "phi_plus_fuzzy": plus_fuzzy,
        "phi_minus_fuzzy": minus_fuzzy,
        "phi_net_fuzzy": net_fuzzy,
        "phi_plus": plus,
        "phi_minus": minus,
        "phi_net": net,
        "rank_experts": rank_experts,
        "rank_e_prom": rank_e_prom,
        "q_vals": np.asarray(q_vals, dtype=float),
        "p_vals": np.asarray(p_vals, dtype=float),
        "types": list(types),
        "directions": np.asarray(directions, dtype=int),
        "preference_types": np.asarray(preference_types, dtype=int),
        "scales": np.asarray(scales, dtype=int),
        "precision_levels": precision_matrix,
        "precision_support_pct": precision_support_pct,
        "precision_core_pct": precision_core_pct,
    }
