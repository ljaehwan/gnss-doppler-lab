"""Frozen A2 no-geometry rank-one GCSPO ablation."""
from __future__ import annotations

import math
import numpy as np

from .gcspo_clean import EPOCH_S, residual_table, window_endpoints
from .gcspo_core import common_epoch_covariance
from .gcspo_full import _score_terms


def fit_a2_loading(train_z_by_epoch):
    medians = []
    for epoch in train_z_by_epoch:
        z = np.asarray(epoch, dtype=float)
        if z.ndim != 2 or not np.all(np.isfinite(z)): raise ValueError("invalid A2 train z")
        if len(z) >= 4: medians.append(np.median(z, axis=0))
    if len(medians) < 2: raise ValueError("A2 requires at least two train epochs")
    matrix = np.vstack(medians); centered = matrix - matrix.mean(axis=0)
    covariance = centered.T @ centered / (len(centered) - 1)
    eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    largest, second = float(eigenvalues[-1]), float(eigenvalues[-2]) if len(eigenvalues) > 1 else 0.
    gap = largest - second
    if gap <= 1e-10 * max(largest, 1): raise ValueError("A2 PCA eigengap unavailable")
    loading = vectors[:, -1]
    pivot = int(np.flatnonzero(np.abs(loading) == np.max(np.abs(loading)))[0])
    if loading[pivot] < 0: loading = -loading
    return loading, {"status": "PASS", "largest_eigenvalue": largest, "second_eigenvalue": second,
                     "eigengap": gap, "sign_pivot_channel": pivot}


def scalar_random_walk_precision(epoch_count, smoothness):
    if epoch_count < 1 or smoothness <= 0: raise ValueError("invalid A2 prior")
    operator = np.zeros((epoch_count, epoch_count)); operator[0, 0] = 1
    for index in range(1, epoch_count): operator[index, index] = 1; operator[index, index - 1] = -1
    return float(smoothness) * operator.T @ operator


def role_a2_terms(data, model, whitener, gamma, loading, start_s, end_s):
    epochs, prns, _, z = residual_table(data, model, whitener, start_s, end_s)
    lookup = {(int(e), int(p)): value for e, p, value in zip(epochs, prns, z)}
    by_epoch = {int(e): sorted(map(int, prns[epochs == e])) for e in np.unique(epochs)}
    rows = []
    for endpoint in window_endpoints(start_s, end_s):
        ids = np.arange(round((endpoint - 1) / EPOCH_S), round(endpoint / EPOCH_S), dtype=np.int64)
        if not all(int(e) in by_epoch and len(by_epoch[int(e)]) >= 4 for e in ids): continue
        normal = np.zeros((50, 50)); vector = np.zeros(50); yty = 0.; nobs = 0
        for index, epoch in enumerate(ids):
            ps = by_epoch[int(epoch)]; y = np.concatenate([lookup[(int(epoch), p)] for p in ps])
            design = np.zeros((len(y), 50))
            for prn_index in range(len(ps)): design[prn_index * len(loading):(prn_index + 1) * len(loading), index] = loading
            factor = np.linalg.cholesky(common_epoch_covariance(gamma, prn_count=len(ps)))
            wy, wg = np.linalg.solve(factor, y), np.linalg.solve(factor, design)
            normal += wg.T @ wg; vector += wg.T @ wy; yty += float(wy @ wy); nobs += len(wy)
        actual_epochs = tuple(map(int, ids))
        support = tuple((epoch, tuple(by_epoch[epoch])) for epoch in actual_epochs)
        rows.append({"window_start_s": endpoint - 1, "availability_s": endpoint,
                     "epoch_ids": actual_epochs, "epoch_prn_support": support,
                     "prns": sorted(set().union(*(set(ps) for _, ps in support))),
                     "terms": (normal, vector, yty, nobs)})
    return rows


def select_a2_lambda(rows, lambda_grid):
    if len(rows) < 100: raise ValueError("A2 has fewer than 100 common validation windows")
    objectives = []
    for value in map(float, lambda_grid):
        prior = scalar_random_walk_precision(50, value)
        scores = [_score_terms(row["terms"], prior) for row in rows]
        objectives.append({"lambda": value, "mean_gcv": float(np.mean([x["gcv"] for x in scores]))})
    best = objectives[0]
    for candidate in objectives[1:]:
        scale = max(abs(candidate["mean_gcv"]), abs(best["mean_gcv"]), 1)
        if candidate["mean_gcv"] < best["mean_gcv"] - 1e-12 * scale or (abs(candidate["mean_gcv"] - best["mean_gcv"]) <= 1e-12 * scale and candidate["lambda"] > best["lambda"]): best = candidate
    return best["lambda"], objectives


def score_a2_terms(rows, smoothness):
    prior = scalar_random_walk_precision(50, smoothness)
    return [{**{key: row[key] for key in ("window_start_s", "availability_s", "prns", "epoch_ids", "epoch_prn_support")},
             **_score_terms(row["terms"], prior)} for row in rows]
