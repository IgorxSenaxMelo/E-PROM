"""Operações com números fuzzy trapezoidais.

Implementação baseada diretamente nas funções MATLAB do SiPREM:
trap_add, trap_sub, trap_scalar_div, trap_inv,
trap_mul_positive, trap_mul_geldermann e defuzz_coa.
"""

from __future__ import annotations
import numpy as np


def _t(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size != 4:
        raise ValueError("Um número fuzzy trapezoidal deve ter 4 elementos [a,b,c,d].")
    return a


def trap_add(A, B):
    A, B = _t(A), _t(B)
    return A + B


def trap_sub(A, B):
    A, B = _t(A), _t(B)
    return np.array([A[0]-B[3], A[1]-B[2], A[2]-B[1], A[3]-B[0]], dtype=float)


def trap_scalar_div(A, s):
    if s == 0:
        raise ZeroDivisionError("Divisão de trapézio por zero.")
    return _t(A) / s


def trap_inv(A):
    A = _t(A)
    if np.any(A <= 0):
        raise ValueError("trap_inv requer trapézio estritamente positivo.")
    return np.array([1/A[3], 1/A[2], 1/A[1], 1/A[0]], dtype=float)


def trap_mul_positive(A, B):
    A, B = _t(A), _t(B)
    return A * B


def trap_mul_geldermann(A, B):
    """Multiplicação fuzzy trapezoidal usada pelo PROMETHEE no MATLAB."""
    A, B = _t(A), _t(B)

    ml1, mu1 = A[1], A[2]
    alpha1, beta1 = A[1]-A[0], A[3]-A[2]
    ml2, mu2 = B[1], B[2]
    alpha2, beta2 = B[1]-B[0], B[3]-B[2]

    ml = ml1 * ml2
    mu = mu1 * mu2
    alpha = ml1 * alpha2 + ml2 * alpha1 - alpha1 * alpha2
    beta = mu1 * beta2 + mu2 * beta1 + beta1 * beta2

    alpha = max(alpha, 0.0)
    beta = max(beta, 0.0)

    C = np.array([ml-alpha, ml, mu, mu+beta], dtype=float)
    C[C < 0] = 0.0
    return np.sort(C)


def defuzz_coa(T):
    """Center of Area (COA) exatamente conforme defuzz_coa do MATLAB."""
    T = _t(T)
    a, b, c, d = T

    if abs(d-a) < 1e-12:
        return float(b)

    den = 3 * (c+d-a-b)
    if abs(den) < 1e-12:
        return float(np.mean(T))

    return float(((c+d)**2 - c*d - (a+b)**2 + a*b) / den)
