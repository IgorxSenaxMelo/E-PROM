from __future__ import annotations
import numpy as np


def _t(x):
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size != 4:
        raise ValueError(
            "Um número fuzzy trapezoidal deve ter 4 elementos [a,b,c,d]."
        )
    return a


def trap_add(A, B):
    return _t(A) + _t(B)


def trap_sub(A, B):
    A, B = _t(A), _t(B)
    return np.array([
        A[0] - B[3],
        A[1] - B[2],
        A[2] - B[1],
        A[3] - B[0],
    ])


def trap_scalar_div(A, s):
    if s == 0:
        raise ZeroDivisionError("Divisão por zero.")
    return _t(A) / float(s)


def trap_mul_geldermann(A, B):
    A, B = _t(A), _t(B)

    ml1, mu1 = A[1], A[2]
    alpha1 = A[1] - A[0]
    beta1 = A[3] - A[2]

    ml2, mu2 = B[1], B[2]
    alpha2 = B[1] - B[0]
    beta2 = B[3] - B[2]

    ml = ml1 * ml2
    mu = mu1 * mu2

    alpha = (
        ml1 * alpha2
        + ml2 * alpha1
        - alpha1 * alpha2
    )

    beta = (
        mu1 * beta2
        + mu2 * beta1
        + beta1 * beta2
    )

    alpha = max(alpha, 0.0)
    beta = max(beta, 0.0)

    C = np.array([
        ml - alpha,
        ml,
        mu,
        mu + beta,
    ])

    C[C < 0] = 0.0

    return np.sort(C)


def defuzz_coa(T):
    T = _t(T)
    a, b, c, d = T

    if abs(d - a) < 1e-12:
        return float(b)

    den = 3 * (c + d - a - b)

    if abs(den) < 1e-12:
        return float(np.mean(T))

    return float(
        (
            (c + d) ** 2
            - c * d
            - (a + b) ** 2
            + a * b
        )
        / den
    )
