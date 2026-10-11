"""PROMETHEE V: portfolio selection under constraints.

Maximize the sum of the net flows of the selected alternatives subject to
resource budgets, a minimum / maximum number of alternatives and mandatory /
excluded alternatives. Solved exactly as a 0-1 integer program (SciPy's MILP
solver when available, otherwise exhaustive search for small problems).
"""

from __future__ import annotations
from itertools import combinations

import numpy as np


def _constraints_matrix(resources, budgets, n):
    if resources is None or len(budgets) == 0:
        return np.zeros((0, n)), np.zeros(0)
    A = np.asarray(resources, dtype=float)
    if A.ndim == 1:
        A = A[None, :]
    if A.shape[1] != n:
        raise ValueError("Each resource must have one value per alternative.")
    b = np.asarray(budgets, dtype=float).reshape(-1)
    if len(b) != A.shape[0]:
        raise ValueError("One budget per resource is required.")
    return A, b


def _solve_milp(phi, A, b, min_count, max_count, must_in, must_out):
    from scipy.optimize import Bounds, LinearConstraint, milp

    n = len(phi)
    lb = np.zeros(n)
    ub = np.ones(n)
    lb[list(must_in)] = 1.0
    ub[list(must_out)] = 0.0

    rows, lo, hi = [], [], []
    for r in range(A.shape[0]):
        rows.append(A[r]); lo.append(-np.inf); hi.append(b[r])
    rows.append(np.ones(n)); lo.append(min_count); hi.append(max_count)

    res = milp(
        c=-phi,
        constraints=LinearConstraint(np.array(rows), np.array(lo), np.array(hi)),
        integrality=np.ones(n),
        bounds=Bounds(lb, ub),
    )
    if not res.success or res.x is None:
        return None
    return np.rint(res.x).astype(int)


def _solve_bruteforce(phi, A, b, min_count, max_count, must_in, must_out):
    n = len(phi)
    if n > 22:
        raise RuntimeError(
            "SciPy is required to solve portfolios with more than 22 alternatives."
        )
    free = [i for i in range(n) if i not in must_in and i not in must_out]
    best, best_val = None, -np.inf
    for size in range(0, len(free) + 1):
        for combo in combinations(free, size):
            chosen = set(must_in) | set(combo)
            if not (min_count <= len(chosen) <= max_count):
                continue
            x = np.zeros(n)
            x[list(chosen)] = 1
            if A.shape[0] and np.any(A @ x > b + 1e-9):
                continue
            val = float(phi @ x)
            if val > best_val:
                best, best_val = x.astype(int), val
    return best


def solve_portfolio(
    phi, resources=None, budgets=None, min_count=0, max_count=None,
    must_include=(), must_exclude=(),
):
    """Select the portfolio with the largest total net flow.

    phi: net flow of each alternative.
    resources: [n_resources, n] consumption of each alternative.
    budgets: [n_resources] upper limits.
    Returns a dict with the 0/1 ``selected`` vector, the objective value and
    the resource usage, or ``feasible = False``.
    """
    phi = np.asarray(phi, dtype=float)
    n = len(phi)
    budgets = [] if budgets is None else list(budgets)
    A, b = _constraints_matrix(resources, budgets, n)
    max_count = n if max_count is None else int(max_count)
    min_count = int(min_count)
    must_in, must_out = set(must_include), set(must_exclude)

    if must_in & must_out:
        raise ValueError("An alternative cannot be both mandatory and excluded.")
    if min_count > max_count:
        raise ValueError("The minimum number of alternatives exceeds the maximum.")

    try:
        x = _solve_milp(phi, A, b, min_count, max_count, must_in, must_out)
    except ImportError:
        x = _solve_bruteforce(phi, A, b, min_count, max_count, must_in, must_out)

    if x is None:
        return {"feasible": False}

    return {
        "feasible": True,
        "selected": x,
        "objective": float(phi @ x),
        "usage": (A @ x) if A.shape[0] else np.zeros(0),
        "budgets": b,
        "count": int(x.sum()),
    }


def greedy_by_rank(
    phi, resources=None, budgets=None, min_count=0, max_count=None,
    must_include=(), must_exclude=(),
):
    """Baseline: walk down the ranking and keep every alternative that still fits."""
    phi = np.asarray(phi, dtype=float)
    n = len(phi)
    budgets = [] if budgets is None else list(budgets)
    A, b = _constraints_matrix(resources, budgets, n)
    max_count = n if max_count is None else int(max_count)

    x = np.zeros(n, dtype=int)
    for i in must_include:
        x[i] = 1
    for i in np.argsort(-phi, kind="mergesort"):
        if x[i] or i in must_exclude or x.sum() >= max_count:
            continue
        if phi[i] <= 0 and x.sum() >= min_count:
            continue                      # no gain in adding it
        trial = x.copy()
        trial[i] = 1
        if A.shape[0] and np.any(A @ trial > b + 1e-9):
            continue
        x = trial

    feasible = (
        x.sum() >= min_count
        and (not A.shape[0] or not np.any(A @ x > b + 1e-9))
    )
    return {
        "feasible": bool(feasible),
        "selected": x,
        "objective": float(phi @ x),
        "usage": (A @ x) if A.shape[0] else np.zeros(0),
        "count": int(x.sum()),
    }
