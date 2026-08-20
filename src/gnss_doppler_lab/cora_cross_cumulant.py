"""Unbiased alpha=0 fourth-order cross-cumulants for CORA-GNSS."""
from __future__ import annotations

import numpy as np


def fourth_cross_cumulant_kstat(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased k-statistic for cum(x,x*,y,y*).

    The multivariate fourth k-statistic uses sample-centered variables and
    explicitly subtracts all three Gaussian pair partitions.  Its population
    target is E|x|²|y|²-E|x|²E|y|²-|E[xy*]|²-|E[xy]|².
    """
    a = np.asarray(x, dtype=np.complex128).reshape(-1)
    b = np.asarray(y, dtype=np.complex128).reshape(-1)
    if a.shape != b.shape or len(a) < 8 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("equal finite complex sequences with n>=8 are required")
    a = a - np.mean(a); b = b - np.mean(b); n = len(a)
    m4 = np.mean(a * np.conj(a) * b * np.conj(b))
    pair = (
        np.mean(a * np.conj(a)) * np.mean(b * np.conj(b))
        + np.mean(a * b) * np.mean(np.conj(a) * np.conj(b))
        + np.mean(a * np.conj(b)) * np.mean(np.conj(a) * b)
    )
    value = n * n * ((n + 1) * m4 - (n - 1) * pair) / ((n - 1) * (n - 2) * (n - 3))
    return float(np.real(value))


def brute_force_cross_cumulant(x: np.ndarray, y: np.ndarray) -> float:
    """Direct centered plug-in moment relation used as an independent reference."""
    a = np.asarray(x, dtype=np.complex128).reshape(-1); b = np.asarray(y, dtype=np.complex128).reshape(-1)
    a = a - np.mean(a); b = b - np.mean(b)
    return float(np.real(
        np.mean(np.abs(a) ** 2 * np.abs(b) ** 2)
        - np.mean(np.abs(a) ** 2) * np.mean(np.abs(b) ** 2)
        - abs(np.mean(a * np.conj(b))) ** 2
        - abs(np.mean(a * b)) ** 2
    ))


def normalized_cross_cumulant(x: np.ndarray, y: np.ndarray, *, variance_floor: float = 1e-8) -> float:
    a = np.asarray(x); b = np.asarray(y)
    denom = float(np.mean(np.abs(a - np.mean(a)) ** 2) * np.mean(np.abs(b - np.mean(b)) ** 2))
    return fourth_cross_cumulant_kstat(a, b) / max(denom, variance_floor)


def cross_cumulant_matrix(tokens: np.ndarray, *, variance_floor: float = 1e-8) -> np.ndarray:
    """Average normalized cross-cumulants over fixed token projections.

    tokens has shape (epochs, prns, projections).  PRN identity and ordering do
    not enter the statistic; the result is symmetric with a zero diagonal.
    """
    z = np.asarray(tokens, dtype=np.complex128)
    if z.ndim != 3 or z.shape[0] < 8 or z.shape[1] < 4:
        raise ValueError("tokens must have shape (epochs, >=4 PRNs, projections)")
    n_prn = z.shape[1]; matrix = np.zeros((n_prn, n_prn), dtype=np.float64)
    for i in range(n_prn):
        for j in range(i + 1, n_prn):
            values = [normalized_cross_cumulant(z[:, i, k], z[:, j, k], variance_floor=variance_floor)
                      for k in range(z.shape[2])]
            matrix[i, j] = matrix[j, i] = float(np.mean(values))
    return matrix


def permute_matrix(matrix: np.ndarray, order: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix); idx = np.asarray(order, dtype=int)
    return value[np.ix_(idx, idx)]
