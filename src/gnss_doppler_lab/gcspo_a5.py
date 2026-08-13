"""Frozen A5 independent per-PRN unknown-input ablation."""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

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


def a5_segment_prior_precision(state_segments, *, smoothness=1.0):
    """Block-independent A5 precision with a fresh prior after every PRN gap."""
    if smoothness <= 0 or not state_segments:
        raise ValueError("invalid A5 segment prior")
    width = max(int(segment["state_stop"]) for segment in state_segments)
    operator = np.zeros((width, width)); f2 = np.asarray([[1., .002], [0., 1.]])
    occupied = np.zeros(width, dtype=bool)
    for segment in state_segments:
        epochs = tuple(map(int, segment["epoch_ids"]))
        start, stop = int(segment["state_start"]), int(segment["state_stop"])
        if not epochs or stop - start != 2 * len(epochs) or start < 0 or stop > width:
            raise ValueError("invalid A5 state segment")
        if np.any(occupied[start:stop]):
            raise ValueError("overlapping A5 state segments")
        occupied[start:stop] = True
        for offset in range(len(epochs)):
            row = start + 2 * offset
            operator[row:row + 2, row:row + 2] = np.eye(2)
            if offset:
                operator[row:row + 2, row - 2:row] = -f2
    if not occupied.all():
        raise ValueError("A5 segment prior has unbound state coordinates")
    return float(smoothness) * operator.T @ operator


def a5_spectral_scores(terms, state_segments, smoothnesses):
    """Evaluate an A5 lambda grid from one exact generalized eigendecomposition."""
    h, vector, yty, nobs = terms
    prior = a5_segment_prior_precision(state_segments, smoothness=1.0)
    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except ImportError:
        use_cuda = False
    if use_cuda:
        device = "cuda"
        th = torch.as_tensor(h, dtype=torch.float64, device=device)
        tp = torch.as_tensor(prior, dtype=torch.float64, device=device)
        tv = torch.as_tensor(vector, dtype=torch.float64, device=device)
        lower = torch.linalg.cholesky(tp)
        transformed = torch.linalg.solve_triangular(lower, th, upper=False)
        transformed = torch.linalg.solve_triangular(lower, transformed.T, upper=False).T
        eigenvalues_t, eigenvectors_t = torch.linalg.eigh((transformed + transformed.T) / 2)
        projected_t = eigenvectors_t.T @ torch.linalg.solve_triangular(
            lower, tv[:, None], upper=False)[:, 0]
        eigenvalues = eigenvalues_t.cpu().numpy()
    else:
        eigenvalues, vectors = eigh(h, prior, check_finite=False, driver="gvd")
        projected = vectors.T @ vector
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -1e-9 * scale:
        raise ValueError("A5 normal matrix is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    rank = int(min(nobs, np.count_nonzero(eigenvalues > 0.0)))
    result = []
    for smoothness in map(float, smoothnesses):
        if smoothness <= 0: raise ValueError("invalid A5 spectral smoothness")
        if use_cuda:
            weights_t = projected_t / (eigenvalues_t + smoothness)
            state_t = torch.linalg.solve_triangular(
                lower.T, (eigenvectors_t @ weights_t)[:, None], upper=True)[:, 0]
            state = state_t.cpu().numpy()
        else:
            weights = projected / (eigenvalues + smoothness)
            state = vectors @ weights
        rss = float(yty - 2 * state @ vector + state @ h @ state)
        improvement = float(yty - rss)
        edf = float(np.sum(eigenvalues / (eigenvalues + smoothness)))
        if edf < -1e-7 or edf > rank + 1e-7 or rank > nobs:
            raise ValueError("A5 influence trace bounds failed")
        edf = min(max(edf, 0.0), float(rank)); penalty = edf * np.log(nobs)
        result.append({"score": improvement - penalty, "state": state, "rss": rss,
                       "likelihood_improvement_twice": improvement, "effective_dof": edf,
                       "penalty": penalty, "n_obs": nobs, "rank": rank,
                       "gcv": nobs * rss / max(nobs - edf, 1e-12) ** 2})
    return result


def _state_segments(epoch_ids, available):
    segments = []
    cursor = 0
    for prn in sorted(set().union(*(available[int(epoch)] for epoch in epoch_ids))):
        present = [int(epoch) for epoch in epoch_ids if prn in available[int(epoch)]]
        left = 0
        for index in range(1, len(present) + 1):
            if index == len(present) or present[index] != present[index - 1] + 1:
                epochs = tuple(present[left:index]); width = 2 * len(epochs)
                segments.append({"prn": int(prn), "epoch_ids": epochs,
                                 "state_start": cursor, "state_stop": cursor + width})
                cursor += width; left = index
    return segments


def role_a5_terms(data, model, whitener, gamma, validated_rows, start_s, end_s):
    epochs, prns, _, z = residual_table(data, model, whitener, start_s, end_s)
    lookup = {(int(e), int(p)): value for e, p, value in zip(epochs, prns, z)}
    available = {int(e): set(map(int, prns[epochs == e])) for e in np.unique(epochs)}
    direct = a5_direct_loading(set(validated_rows)); rows = []
    for endpoint in window_endpoints(start_s, end_s):
        epoch_ids = np.arange(round((endpoint - 1) / EPOCH_S), round(endpoint / EPOCH_S), dtype=np.int64)
        if not all(int(epoch) in available for epoch in epoch_ids): continue
        if any(len(available[int(epoch)]) < 4 for epoch in epoch_ids): continue
        segments = _state_segments(epoch_ids, available)
        state_index = {(epoch, segment["prn"]): segment["state_start"] + 2 * offset
                       for segment in segments for offset, epoch in enumerate(segment["epoch_ids"])}
        state_width = max(segment["state_stop"] for segment in segments)
        normal = np.zeros((state_width, state_width)); vector = np.zeros(state_width); yty = 0.; nobs = 0
        for time_index, epoch in enumerate(epoch_ids):
            current_prns = sorted(available[int(epoch)])
            observations = np.concatenate([lookup[(int(epoch), prn)] for prn in current_prns])
            design = np.zeros((len(observations), state_width))
            for prn_index, prn in enumerate(current_prns):
                row = slice(prn_index * 10, (prn_index + 1) * 10)
                current = state_index[(int(epoch), prn)]
                design[row, current:current + 2] = direct
                for lag, coefficient in enumerate(model.coefficients, start=1):
                    previous = state_index.get((int(epoch) - lag, prn))
                    if previous is not None:
                        design[row, previous:previous + 2] -= coefficient @ direct
            design = np.einsum("ij,njk->nik", whitener.inverse_sqrt, design.reshape(len(current_prns), 10, state_width)).reshape(len(current_prns) * 10, state_width)
            factor = np.linalg.cholesky(common_epoch_covariance(gamma, prn_count=len(current_prns)))
            wy, wg = np.linalg.solve(factor, observations), np.linalg.solve(factor, design)
            normal += wg.T @ wg; vector += wg.T @ wy; yty += float(wy @ wy); nobs += len(wy)
        actual_epochs = tuple(map(int, epoch_ids)); actual_prns = sorted(set().union(*(available[epoch] for epoch in actual_epochs)))
        support = tuple((epoch, tuple(sorted(available[epoch]))) for epoch in actual_epochs)
        rows.append({"window_start_s": endpoint - 1, "availability_s": endpoint, "prns": actual_prns,
                     "epoch_ids": actual_epochs,
                     "epoch_prn_support": support, "state_segments": segments,
                     "terms": (normal, vector, yty, nobs)})
    return rows


def select_a5_lambda(rows, lambda_grid):
    if len(rows) < 100: raise ValueError("A5 has fewer than 100 common lambda-validation windows")
    objectives = []
    for value in map(float, lambda_grid):
        scores = [a5_spectral_scores(row["terms"], row["state_segments"], (value,))[0] for row in rows]
        objectives.append({"lambda": value, "mean_gcv": float(np.mean([score["gcv"] for score in scores]))})
    best = objectives[0]
    for candidate in objectives[1:]:
        scale = max(abs(candidate["mean_gcv"]), abs(best["mean_gcv"]), 1)
        if candidate["mean_gcv"] < best["mean_gcv"] - 1e-12 * scale or (abs(candidate["mean_gcv"] - best["mean_gcv"]) <= 1e-12 * scale and candidate["lambda"] > best["lambda"]): best = candidate
    return best["lambda"], objectives


def score_a5_terms(rows, smoothness):
    result = []
    for row in rows:
        score = a5_spectral_scores(row["terms"], row["state_segments"], (smoothness,))[0]
        result.append({key: row[key] for key in ("window_start_s", "availability_s", "prns", "epoch_ids", "epoch_prn_support")}
                      | score)
    return result
