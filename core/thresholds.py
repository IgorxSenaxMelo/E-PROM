"""q / p assistant: data- and precision-driven starting points for the thresholds."""

from __future__ import annotations
import numpy as np

from .e_prom_core import PRECISION_DEFAULTS
from .labels import CONTINUOUS, ORDINAL, normalize_precision, normalize_type
from .promethee import pref_array

PREF_CODES = {"Usual": 1, "U-shape": 2, "V-shape": 3, "Level": 4, "Linear": 5}


def nice_number(x, mode="nearest"):
    """Round x to a 'nice' value (1, 2, 2.5, 5, 10 times a power of ten)."""
    if x <= 0 or not np.isfinite(x):
        return 0.0
    exp = np.floor(np.log10(x))
    base = x / 10 ** exp
    steps = [1, 2, 2.5, 5, 10]
    if mode == "up":
        choice = next(s for s in steps if s >= base - 1e-12)
    elif mode == "down":
        choice = max(s for s in steps if s <= base + 1e-12)
    else:
        choice = min(steps, key=lambda s: abs(s - base))
    return float(choice * 10 ** exp)


def _pairwise_differences(values):
    v = np.asarray(values, dtype=float)
    diffs = np.abs(v[:, None] - v[None, :])[np.triu_indices(len(v), k=1)]
    return diffs


def suggest_thresholds(evaluations, types, scales, precision_levels):
    """Suggest a preference function with q and p for every criterion.

    evaluations: {DM: [alternatives, criteria]}; precision_levels: [DM][criterion].

    Continuous criteria: q covers the fuzzy width of two equal-precision values
    (about 2 * support half-width * typical value) so near-ties are treated as
    indifference; p is the 75th percentile of the pairwise differences.
    Ordinal criteria: Level function with q / p at the 25th / 75th percentiles of
    the pairwise differences (in scale units).
    """
    types = [normalize_type(t) for t in types]
    stack = np.stack([np.asarray(v, dtype=float) for v in evaluations.values()])
    data = stack.mean(axis=0)                                  # [alternatives, criteria]
    n_crit = data.shape[1]
    precision = np.asarray(precision_levels, dtype=object)
    out = []

    for k in range(n_crit):
        diffs = _pairwise_differences(data[:, k])
        if types[k] == CONTINUOUS:
            levels = [normalize_precision(x) for x in precision[:, k]]
            rs = max(PRECISION_DEFAULTS[lv][0] for lv in levels) / 100.0
            typical = float(np.median(np.abs(data[:, k])))
            band = 2.0 * rs * typical
            q = nice_number(band, "up") if band > 0 else 0.0
            p_raw = float(np.percentile(diffs, 75)) if len(diffs) else 0.0
            p = nice_number(max(p_raw, 2.0 * q), "up") if max(p_raw, 2.0 * q) > 0 else 1.0
            if p <= q:
                p = nice_number(q * 2 if q > 0 else 1.0, "up")
            out.append({
                "preference": "Linear",
                "q": q,
                "p": p,
                "rationale": (
                    f"q = fuzzy width of two equal values (about 2 x {rs:.1%} x {typical:g} = {band:g}); "
                    f"p = 75th percentile of the pairwise differences ({p_raw:g}), at least 2q."
                ),
            })
        else:
            top = int(scales[k])
            q = float(np.floor(np.percentile(diffs, 25))) if len(diffs) else 0.0
            p = float(np.ceil(np.percentile(diffs, 75))) if len(diffs) else 1.0
            q = float(np.clip(q, 0, top - 2))
            p = float(np.clip(max(p, q + 1.0), q + 1.0, top - 1))
            out.append({
                "preference": "Level",
                "q": q,
                "p": p,
                "rationale": (
                    f"q / p at the 25th / 75th percentiles of the pairwise differences "
                    f"(scale 1-{top}); differences up to {q:g} are indifference, above {p:g} a strict preference."
                ),
            })
    return out


def preference_curve(preference, q, p, d_max, n_points=120):
    """Points of the preference function P(d) for plotting, d in [0, d_max]."""
    d = np.linspace(0.0, float(d_max), n_points)
    return d, pref_array(d, PREF_CODES[preference], float(q), float(p))
