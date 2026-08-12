"""Frozen A5 independent per-PRN unknown-input ablation."""
from __future__ import annotations

import numpy as np

from .gcspo_clean import EPOCH_S, residual_table, window_endpoints
from .gcspo_core import CODE_CHIP_M, L1_WAVELENGTH_M, common_epoch_covariance
from .gcspo_full import _score_terms


def a5_direct_loading(validated_rows):
    """Direct q shift for eta=[rho/10m,rhodot/1mps]."""
    loading = np.zeros((10, 2))
    if "code_error_chips" in validated_rows: loading[6, 0] = -10 / CODE_CHIP_M
    if "pll_phase_error_cycles" in validated_rows: loading[7, 0] = -10 / L1_WAVELENGTH_M
    if "carrier_doppler_hz" in validated_rows: loading[8, 1] = -1 / L1_WAVELENGTH_M
    if "code_frequency_offset_chips_s" in validated_rows: loading[9, 1] = -1 / CODE_CHIP_M
    return loading


def a5_prior_precision(*, prn_count, epoch_count=50, smoothness=1.0):
    if prn_count < 1 or epoch_count < 1 or smoothness <= 0: raise ValueError("invalid A5 prior")
    width = epoch_count * prn_count * 2; operator = np.zeros((width, width))
    f2 = np.asarray([[1., .002], [0., 1.]])
    for epoch in range(epoch_count):
        for prn_index in range(prn_count):
            row = (epoch * prn_count + prn_index) * 2
            operator[row:row + 2, row:row + 2] = np.eye(2)
            if epoch:
                previous = ((epoch - 1) * prn_count + prn_index) * 2
                operator[row:row + 2, previous:previous + 2] = -f2
    return float(smoothness) * operator.T @ operator


def role_a5_terms(data, model, whitener, gamma, validated_rows, start_s, end_s):
    epochs, prns, _, z = residual_table(data, model, whitener, start_s, end_s)
    lookup = {(int(e), int(p)): value for e, p, value in zip(epochs, prns, z)}
    available = {int(e): set(map(int, prns[epochs == e])) for e in np.unique(epochs)}
    direct = a5_direct_loading(set(validated_rows)); rows = []
    for endpoint in window_endpoints(start_s, end_s):
        epoch_ids = np.arange(round((endpoint - 1) / EPOCH_S), round(endpoint / EPOCH_S), dtype=np.int64)
        if not all(int(epoch) in available for epoch in epoch_ids): continue
        common = sorted(set.intersection(*(available[int(epoch)] for epoch in epoch_ids)))
        if len(common) < 4: continue
        prn_index = {prn: index for index, prn in enumerate(common)}
        state_width = 50 * len(common) * 2
        normal = np.zeros((state_width, state_width)); vector = np.zeros(state_width); yty = 0.; nobs = 0
        for time_index, epoch in enumerate(epoch_ids):
            observations = np.concatenate([lookup[(int(epoch), prn)] for prn in common])
            design = np.zeros((len(observations), state_width))
            for prn in common:
                row = slice(prn_index[prn] * 10, (prn_index[prn] + 1) * 10)
                current = (time_index * len(common) + prn_index[prn]) * 2
                design[row, current:current + 2] = direct
                for lag, coefficient in enumerate(model.coefficients, start=1):
                    if time_index >= lag:
                        previous = ((time_index - lag) * len(common) + prn_index[prn]) * 2
                        design[row, previous:previous + 2] -= coefficient @ direct
            design = np.einsum("ij,njk->nik", whitener.inverse_sqrt, design.reshape(len(common), 10, state_width)).reshape(len(common) * 10, state_width)
            factor = np.linalg.cholesky(common_epoch_covariance(gamma, prn_count=len(common)))
            wy, wg = np.linalg.solve(factor, observations), np.linalg.solve(factor, design)
            normal += wg.T @ wg; vector += wg.T @ wy; yty += float(wy @ wy); nobs += len(wy)
        actual_epochs = tuple(map(int, epoch_ids)); actual_prns = tuple(map(int, common))
        rows.append({"window_start_s": endpoint - 1, "availability_s": endpoint, "prns": common,
                     "epoch_ids": actual_epochs,
                     "epoch_prn_support": tuple((epoch, actual_prns) for epoch in actual_epochs),
                     "terms": (normal, vector, yty, nobs)})
    return rows


def select_a5_lambda(rows, lambda_grid):
    if len(rows) < 100: raise ValueError("A5 has fewer than 100 common lambda-validation windows")
    objectives = []
    for value in map(float, lambda_grid):
        scores = [_score_terms(row["terms"], a5_prior_precision(prn_count=len(row["prns"]), smoothness=value)) for row in rows]
        objectives.append({"lambda": value, "mean_gcv": float(np.mean([score["gcv"] for score in scores]))})
    best = objectives[0]
    for candidate in objectives[1:]:
        scale = max(abs(candidate["mean_gcv"]), abs(best["mean_gcv"]), 1)
        if candidate["mean_gcv"] < best["mean_gcv"] - 1e-12 * scale or (abs(candidate["mean_gcv"] - best["mean_gcv"]) <= 1e-12 * scale and candidate["lambda"] > best["lambda"]): best = candidate
    return best["lambda"], objectives


def score_a5_terms(rows, smoothness):
    result = []
    for row in rows:
        score = _score_terms(row["terms"], a5_prior_precision(prn_count=len(row["prns"]), smoothness=smoothness))
        result.append({key: row[key] for key in ("window_start_s", "availability_s", "prns", "epoch_ids", "epoch_prn_support")}
                      | score)
    return result
