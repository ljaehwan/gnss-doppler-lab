"""Normal-only analytic + shared-linear TRACE Stage-0 utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from sklearn.covariance import LedoitWolf

from .trace_action_warp import prompt_normalize, receiver_action, warp_complex_taps

TAPS = ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")
REQUIRED_FIELDS = (
    "PRN", "PRN_start_sample_count", "CN0_SNV_dB_Hz", "carrier_lock_test",
    "carr_error_hz", "carr_error_filt_hz", "carrier_doppler_hz",
    "code_error_chips", "code_error_filt_chips", "code_freq_chips", "aux1",
    *(f"{part}_{tap}" for tap in TAPS for part in ("I", "Q")),
)


@dataclass(frozen=True)
class TracePairs:
    current: np.ndarray
    target: np.ndarray
    warped: np.ndarray
    valid_support: np.ndarray
    code_action: np.ndarray
    carrier_action: np.ndarray
    dt_s: np.ndarray
    time_s: np.ndarray
    sample_count: np.ndarray
    prn: np.ndarray
    channel: np.ndarray
    cn0_db_hz: np.ndarray
    lock: np.ndarray
    source_row: np.ndarray

    def take(self, mask: np.ndarray) -> "TracePairs":
        return TracePairs(**{name: getattr(self, name)[mask] for name in self.__dataclass_fields__})


def _vector(handle: h5py.File, name: str) -> np.ndarray:
    return np.asarray(handle[name]).reshape(-1)


def load_trace_pairs(
    mat_dir: Path | str,
    sample_rate_hz: float,
    *,
    cn0_min_db_hz: float = 28.0,
    lock_min: float = 0.85,
    prompt_epsilon: float = 1e-12,
) -> TracePairs:
    """Load causal row-t action to row-(t+1) pairs from receiver MAT files."""
    batches: dict[str, list[np.ndarray]] = {name: [] for name in TracePairs.__dataclass_fields__}
    for channel, path in enumerate(sorted(Path(mat_dir).glob("epl_tracking_ch_*.mat"))):
        with h5py.File(path, "r") as handle:
            missing = [field for field in REQUIRED_FIELDS if field not in handle]
            if missing:
                raise ValueError(f"{path} missing TRACE fields: {missing}")
            n = len(handle["PRN"])
            taps = np.empty((n, 9), dtype=np.complex128)
            for k, tap in enumerate(TAPS):
                taps[:, k] = _vector(handle, f"I_{tap}") + 1j * _vector(handle, f"Q_{tap}")
            normalized, prompt_valid = prompt_normalize(taps, prompt_epsilon)
            samples = _vector(handle, "PRN_start_sample_count").astype(np.uint64)
            prn = _vector(handle, "PRN").astype(np.int16)
            cn0 = _vector(handle, "CN0_SNV_dB_Hz").astype(np.float64)
            lock = _vector(handle, "carrier_lock_test").astype(np.float64)
            code_freq = _vector(handle, "code_freq_chips").astype(np.float64)
            doppler = _vector(handle, "carrier_doppler_hz").astype(np.float64)
        ds = np.diff(samples.astype(np.int64))
        same_track = (prn[:-1] == prn[1:]) & (ds >= round(sample_rate_hz * 0.0009)) & (ds <= round(sample_rate_hz * 0.0011))
        quality = (
            prompt_valid[:-1] & prompt_valid[1:] & np.isfinite(cn0[:-1]) & np.isfinite(lock[:-1])
            & (cn0[:-1] >= cn0_min_db_hz) & (lock[:-1] >= lock_min)
        )
        indices = np.flatnonzero(same_track & quality)
        if not len(indices):
            continue
        current = normalized[indices]
        target = normalized[indices + 1]
        dt = ds[indices].astype(np.float64) / float(sample_rate_hz)
        code_action = np.empty(len(indices), dtype=np.float64)
        carrier_action = np.empty(len(indices), dtype=np.float64)
        warped = np.empty_like(current)
        support = np.empty(current.shape, dtype=bool)
        for j, (row, duration) in enumerate(zip(indices, dt, strict=True)):
            code_action[j], carrier_action[j] = receiver_action(code_freq[row], doppler[row], duration)
            warped[j], support[j] = warp_complex_taps(current[j], code_action[j], carrier_action[j])
        batches["current"].append(current)
        batches["target"].append(target)
        batches["warped"].append(warped)
        batches["valid_support"].append(support)
        batches["code_action"].append(code_action)
        batches["carrier_action"].append(carrier_action)
        batches["dt_s"].append(dt)
        batches["time_s"].append(samples[indices + 1].astype(np.float64) / float(sample_rate_hz))
        batches["sample_count"].append(samples[indices + 1])
        batches["prn"].append(prn[indices + 1])
        batches["channel"].append(np.full(len(indices), channel, dtype=np.int16))
        batches["cn0_db_hz"].append(cn0[indices])
        batches["lock"].append(lock[indices])
        batches["source_row"].append(indices.astype(np.int64))
    if not batches["current"]:
        raise ValueError(f"no quality-gated causal TRACE pairs in {mat_dir}")
    return TracePairs(**{name: np.concatenate(parts, axis=0) for name, parts in batches.items()})


def complex_to_real(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    return np.concatenate((arr.real, arr.imag), axis=-1)


def linear_features(pairs: TracePairs, *, include_action: bool) -> np.ndarray:
    base = complex_to_real(pairs.current)
    features = [np.ones((len(base), 1)), base]
    if include_action:
        features.extend((pairs.code_action[:, None], np.sin(pairs.carrier_action)[:, None], np.cos(pairs.carrier_action)[:, None]))
    return np.concatenate(features, axis=1)


@dataclass
class RidgePredictor:
    coefficients: np.ndarray
    include_action: bool
    target_mode: str
    output_indices: np.ndarray

    def predict(self, pairs: TracePairs) -> np.ndarray:
        correction = linear_features(pairs, include_action=self.include_action) @ self.coefficients
        width = len(self.output_indices)
        complex_correction = correction[:, :width] + 1j * correction[:, width:]
        out = np.full_like(pairs.current, np.nan + 1j * np.nan)
        if self.target_mode == "warp_residual":
            out[:, self.output_indices] = pairs.warped[:, self.output_indices] + complex_correction
        else:
            out[:, self.output_indices] = complex_correction
        return out


def fit_ridge(
    pairs: TracePairs, *, include_action: bool, target_mode: str, alpha: float,
    output_indices: Iterable[int] = range(1, 8),
) -> RidgePredictor:
    if target_mode not in {"warp_residual", "direct"}:
        raise ValueError("invalid target mode")
    x = linear_features(pairs, include_action=include_action)
    indices = np.asarray(tuple(output_indices), dtype=int)
    y_complex = pairs.target[:, indices] - pairs.warped[:, indices] if target_mode == "warp_residual" else pairs.target[:, indices]
    y = complex_to_real(y_complex)
    gram = x.T @ x
    penalty = np.eye(gram.shape[0]) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(gram + penalty, x.T @ y)
    return RidgePredictor(coefficients, include_action, target_mode, indices)


@dataclass
class ResidualModel:
    mean: np.ndarray
    precision: np.ndarray

    @classmethod
    def fit(cls, residuals: np.ndarray) -> "ResidualModel":
        real = complex_to_real(residuals)
        covariance = LedoitWolf(assume_centered=False).fit(real)
        return cls(covariance.location_, covariance.precision_)

    def score(self, residuals: np.ndarray) -> np.ndarray:
        centered = complex_to_real(residuals) - self.mean
        return np.einsum("ni,ij,nj->n", centered, self.precision, centered)


def robust_epoch_blocks(
    pairs: TracePairs, scores: np.ndarray, *, block_s: float = 0.5, minimum_prns: int = 4
) -> np.ndarray:
    """Fixed median aggregation in non-overlapping blocks."""
    block = np.floor(pairs.time_s / float(block_s)).astype(np.int64)
    rows = []
    for value in np.unique(block):
        mask = block == value
        count = len(np.unique(pairs.prn[mask]))
        if count < minimum_prns:
            continue
        rows.append((value * block_s, (value + 0.5) * block_s, float(np.median(scores[mask])), count, int(mask.sum())))
    return np.asarray(rows, dtype=[("block_start_s", "f8"), ("block_mid_s", "f8"), ("score", "f8"), ("tracked_prn_count", "i4"), ("pair_count", "i4")])


def consecutive_alarm(times: np.ndarray, scores: np.ndarray, threshold: float, run_length: int = 3) -> np.ndarray:
    alarm = np.zeros(len(scores), dtype=bool)
    count = 0
    previous = None
    for i, (time_s, score) in enumerate(zip(times, scores, strict=True)):
        if previous is None or not np.isclose(time_s - previous, 0.5, atol=1e-9):
            count = 0
        count = count + 1 if score > threshold else 0
        alarm[i] = count >= run_length
        previous = time_s
    return alarm


def action_shuffle_indices(prn: np.ndarray, cn0: np.ndarray, seed: int = 23017) -> np.ndarray:
    """Shuffle within PRN and fixed 3 dB-Hz C/N0 bins, preserving marginals."""
    rng = np.random.default_rng(seed)
    result = np.arange(len(prn))
    bins = np.floor(np.asarray(cn0) / 3.0).astype(int)
    for p in np.unique(prn):
        for b in np.unique(bins[prn == p]):
            idx = np.flatnonzero((prn == p) & (bins == b))
            result[idx] = rng.permutation(idx)
    return result


def persistent_alarm_ratio(alarm: np.ndarray, attack_mask: np.ndarray) -> float:
    selected = np.asarray(alarm, dtype=bool)[np.asarray(attack_mask, dtype=bool)]
    return float(selected.mean()) if len(selected) else 0.0


def chronological_masks(time_s: np.ndarray, duration_s: float, guard_s: float = 5.0) -> dict[str, np.ndarray]:
    """Predeclared 50/25/25 chronological split with guard intervals."""
    t = np.asarray(time_s)
    first = 0.50 * duration_s
    second = 0.75 * duration_s
    return {
        "train": t < first - guard_s,
        "calibration": (t >= first + guard_s) & (t < second - guard_s),
        "holdout": t >= second + guard_s,
    }
