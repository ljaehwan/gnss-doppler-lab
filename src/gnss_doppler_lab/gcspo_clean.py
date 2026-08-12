"""CleanStatic-only loading, VAR fitting, whitening, and dry-run scoring."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import h5py
import numpy as np

from .gcspo_core import SharedVAR, empirical_threshold, fit_common_gamma, fit_whitener, pooled_signed_innovation_score

SAMPLE_RATE_HZ = 25_000_000
EPOCH_S = .02
EPOCH_SAMPLES = 500_000
Q_FIELDS = ("I_E", "Q_E", "I_P", "Q_P", "I_L", "Q_L", "code_error_chips", "carr_error_hz", "carrier_doppler_hz", "code_freq_chips")


@dataclass(frozen=True)
class AggregatedClean:
    epoch: np.ndarray
    prn: np.ndarray
    channel: np.ndarray
    segment: np.ndarray
    q: np.ndarray
    sample_min: np.ndarray
    sample_max: np.ndarray
    epsilons: dict[int, float]
    source_files: tuple[str, ...]


def signed_q(columns, *, epsilon):
    values = {name: np.asarray(columns[name], dtype=np.float64).reshape(-1) for name in Q_FIELDS}
    lengths = {len(value) for value in values.values()}
    if len(lengths) != 1: raise ValueError("signed-q columns have different lengths")
    eps = np.broadcast_to(np.asarray(epsilon, dtype=float), (next(iter(lengths)),))
    denominator = np.sqrt(sum(values[name] ** 2 for name in Q_FIELDS[:6])) + eps
    if np.any(denominator <= 0) or not np.all(np.isfinite(denominator)): raise ValueError("invalid complex normalization denominator")
    return np.column_stack([*(values[name] / denominator for name in Q_FIELDS[:6]), values["code_error_chips"],
                            values["carr_error_hz"], values["carrier_doppler_hz"], values["code_freq_chips"] - 1_023_000])


def _read_vector(handle, name):
    if name not in handle: raise ValueError(f"tracking MAT is missing {name}")
    result = np.asarray(handle[name]).reshape(-1)
    if not np.all(np.isfinite(result)): raise ValueError(f"tracking MAT contains nonfinite {name}")
    return result


def load_cleanstatic_mat(receiver_root, *, start_s=30., end_s=470.):
    root = Path(receiver_root)
    paths = tuple(sorted((root / "raw").glob("epl_tracking_ch_*.mat")))
    if len(paths) != 11: raise ValueError(f"expected 11 cleanStatic tracking MATs, found {len(paths)}")
    chunks = {name: [] for name in ("sample", "prn", "channel", "segment", *Q_FIELDS)}
    for channel_id, path in enumerate(paths):
        with h5py.File(path, "r") as handle:
            sample = _read_vector(handle, "PRN_start_sample_count").astype(np.int64)
            prn_values = _read_vector(handle, "PRN").astype(np.int64)
            if len(prn_values) != len(sample): raise ValueError("tracking MAT PRN/sample length mismatch")
            breaks = np.r_[True, (np.diff(prn_values) != 0) | (np.diff(sample) <= 0)]
            segments = np.cumsum(breaks, dtype=np.int64) - 1
            mask = (sample >= math.ceil(start_s * SAMPLE_RATE_HZ)) & (sample < math.ceil(end_s * SAMPLE_RATE_HZ))
            chunks["sample"].append(sample[mask]); chunks["prn"].append(prn_values[mask])
            chunks["channel"].append(np.full(np.count_nonzero(mask), channel_id, np.int64))
            chunks["segment"].append(segments[mask])
            for name in Q_FIELDS: chunks[name].append(_read_vector(handle, name)[mask])
    columns = {name: np.concatenate(parts) for name, parts in chunks.items()}
    denominator = np.sqrt(sum(columns[name].astype(float) ** 2 for name in Q_FIELDS[:6]))
    train_mask = (columns["sample"] >= 30 * SAMPLE_RATE_HZ) & (columns["sample"] < 210 * SAMPLE_RATE_HZ) & (denominator > 0)
    epsilons = {}
    for prn in sorted(set(columns["prn"][train_mask])):
        values = denominator[train_mask & (columns["prn"] == prn)]
        if not len(values): raise ValueError(f"PRN {prn} has no train-fit normalization data")
        epsilons[int(prn)] = float(np.quantile(values, .001))
    known = np.isin(columns["prn"], np.asarray(sorted(epsilons)))
    columns = {name: value[known] for name, value in columns.items()}
    epsilon = np.asarray([epsilons[int(prn)] for prn in columns["prn"]])
    q = signed_q(columns, epsilon=epsilon)
    epoch = columns["sample"] // EPOCH_SAMPLES
    channel, segment = columns["channel"], columns["segment"]
    order = np.lexsort((columns["sample"], segment, channel, columns["prn"], epoch))
    epoch, prn, channel, segment, sample, q = (epoch[order], columns["prn"][order], channel[order], segment[order],
                                               columns["sample"][order], q[order])
    boundaries = np.r_[0, np.flatnonzero((np.diff(epoch) != 0) | (np.diff(prn) != 0) |
                                         (np.diff(channel) != 0) | (np.diff(segment) != 0)) + 1, len(epoch)]
    out_epoch, out_prn, out_channel, out_segment, out_q, sample_min, sample_max = [], [], [], [], [], [], []
    for left, right in zip(boundaries, boundaries[1:]):
        block = q[left:right]
        # Identical source sample/PRN/q rows are fatal.
        identities = {(int(sample[i]), block[i - left].tobytes()) for i in range(left, right)}
        if len(identities) != right - left: raise ValueError("exact duplicate scientific rows")
        out_epoch.append(epoch[left]); out_prn.append(prn[left]); out_channel.append(channel[left]); out_segment.append(segment[left]); out_q.append(np.median(block, axis=0))
        sample_min.append(sample[left:right].min()); sample_max.append(sample[left:right].max())
    return AggregatedClean(np.asarray(out_epoch, np.int64), np.asarray(out_prn, np.int64),
                           np.asarray(out_channel, np.int64), np.asarray(out_segment, np.int64), np.vstack(out_q),
                           np.asarray(sample_min, np.int64), np.asarray(sample_max, np.int64), epsilons,
                           tuple(str(path) for path in paths))


def causal_histories(epochs, values, *, lags=10, identities=None):
    epoch, q = np.asarray(epochs, dtype=np.int64), np.asarray(values, dtype=float)
    if epoch.ndim != 1 or q.ndim != 2 or len(epoch) != len(q) or lags < 1: raise ValueError("invalid causal history inputs")
    identity = np.zeros((len(epoch), 1), dtype=object) if identities is None else np.asarray(identities, dtype=object)
    if identity.ndim == 1: identity = identity[:, None]
    if len(identity) != len(epoch): raise ValueError("identity length mismatch")
    histories, targets, target_epochs = [], [], []
    segment_start = 0
    for index in range(1, len(epoch) + 1):
        if index == len(epoch) or epoch[index] != epoch[index - 1] + 1 or not np.array_equal(identity[index], identity[index - 1]):
            for target in range(segment_start + lags, index):
                histories.append(q[target - lags:target]); targets.append(q[target]); target_epochs.append(epoch[target])
            segment_start = index
    width = q.shape[1]
    return (np.asarray(histories, dtype=float).reshape(-1, lags, width),
            np.asarray(targets, dtype=float).reshape(-1, width), np.asarray(target_epochs, np.int64))


def role_histories(data: AggregatedClean, start_s, end_s, *, lags=10):
    first, final = math.ceil(start_s / EPOCH_S), math.floor(end_s / EPOCH_S)
    mask = (data.epoch >= first) & (data.epoch < final)
    aliases = np.column_stack((data.epoch[mask], data.prn[mask]))
    if len(aliases) and len({tuple(row) for row in aliases.tolist()}) != len(aliases):
        raise ValueError("simultaneous channel identity alias is ambiguous")
    histories, targets, epochs, prns = [], [], [], []
    identities = sorted(set(zip(map(int, data.channel[mask]), map(int, data.prn[mask]), map(int, data.segment[mask]))))
    for channel, prn, segment in identities:
        selected = mask & (data.channel == channel) & (data.prn == prn) & (data.segment == segment)
        identity_rows = np.tile(np.asarray([channel, prn, segment], dtype=object), (np.count_nonzero(selected), 1))
        h, t, e = causal_histories(data.epoch[selected], data.q[selected], lags=lags, identities=identity_rows)
        histories.append(h); targets.append(t); epochs.append(e); prns.append(np.full(len(e), prn, np.int64))
    width = data.q.shape[1]
    return (np.concatenate(histories) if histories else np.empty((0, lags, width)),
            np.concatenate(targets) if targets else np.empty((0, width)),
            np.concatenate(epochs) if epochs else np.empty(0, np.int64),
            np.concatenate(prns) if prns else np.empty(0, np.int64))


def select_shared_var_ridge(train_histories, train_targets, validation_histories, validation_targets, ridge_grid):
    results, models = [], {}
    for ridge in ridge_grid:
        model = SharedVAR.fit(train_histories, train_targets, ridge=float(ridge)); models[float(ridge)] = model
        residual = model.residuals(validation_histories, validation_targets)
        results.append({"ridge": float(ridge), "mean_squared_error": float(np.mean(residual ** 2))})
    best = results[0]
    for candidate in results[1:]:
        scale = max(abs(candidate["mean_squared_error"]), abs(best["mean_squared_error"]), 1)
        if candidate["mean_squared_error"] < best["mean_squared_error"] - 1e-12 * scale or (
            abs(candidate["mean_squared_error"] - best["mean_squared_error"]) <= 1e-12 * scale and candidate["ridge"] > best["ridge"]
        ): best = candidate
    return models[best["ridge"]], {"selected_ridge": best["ridge"], "validation": results}


def window_endpoints(start_s, end_s):
    first = math.ceil((float(start_s) - 1 + 1e-12) / .5) * .5 + 1
    endpoint = np.arange(first, float(end_s) + 1e-12, .5)
    return endpoint[(endpoint - 1 >= start_s - 1e-12) & (endpoint <= end_s + 1e-12)]


def residual_table(data, model, whitener, start_s, end_s):
    histories, targets, epochs, prns = role_histories(data, start_s, end_s, lags=model.lags)
    residuals = model.residuals(histories, targets)
    z = whitener.transform(residuals)
    order = np.lexsort((prns, epochs))
    return epochs[order], prns[order], residuals[order], z[order]


def a1_role_scores(data, model, whitener, start_s, end_s):
    epochs, prns, _, z = residual_table(data, model, whitener, start_s, end_s)
    lookup = {(int(epoch), int(prn)): value for epoch, prn, value in zip(epochs, prns, z)}
    rows = []
    for endpoint in window_endpoints(start_s, end_s):
        epoch_ids = np.arange(round((endpoint - 1) / EPOCH_S), round(endpoint / EPOCH_S), dtype=np.int64)
        common = [prn for prn in sorted(set(prns)) if all((int(epoch), int(prn)) in lookup for epoch in epoch_ids)]
        if len(common) < 4: continue
        cube = np.stack([[lookup[(int(epoch), int(prn))] for epoch in epoch_ids] for prn in common])
        actual_epochs = tuple(map(int, epoch_ids)); actual_prns = tuple(map(int, common))
        rows.append({"window_start_s": endpoint - 1, "availability_s": endpoint,
                     "score": pooled_signed_innovation_score(cube), "prns": list(actual_prns),
                     "epoch_ids": actual_epochs,
                     "epoch_prn_support": tuple((epoch, actual_prns) for epoch in actual_epochs)})
    return rows


def run_clean_a1(receiver_root, *, ridge_grid):
    data = load_cleanstatic_mat(receiver_root)
    train_h, train_t, _, _ = role_histories(data, 30, 140)
    validation_h, validation_t, _, _ = role_histories(data, 150, 210)
    model, ridge_report = select_shared_var_ridge(train_h, train_t, validation_h, validation_t, ridge_grid)
    train_residuals = model.residuals(train_h, train_t)
    whitener = fit_whitener(train_residuals)
    train_epochs, _, _, train_z = residual_table(data, model, whitener, 30, 140)
    gamma = fit_common_gamma([train_z[train_epochs == epoch] for epoch in np.unique(train_epochs)])
    calibration = a1_role_scores(data, model, whitener, 220, 340)
    holdout = a1_role_scores(data, model, whitener, 350, 470)
    thresholds = {"q99": empirical_threshold(np.asarray([row["score"] for row in calibration]), .99),
                  "q995": empirical_threshold(np.asarray([row["score"] for row in calibration]), .995)}
    return {"data": data, "model": model, "whitener": whitener, "gamma": gamma, "ridge": ridge_report,
            "calibration": calibration, "holdout": holdout, "thresholds": thresholds}
