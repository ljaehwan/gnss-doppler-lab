"""MCTD Stage-0 utilities.

The module is deliberately non-neural.  It aligns two native TRACE receiver
dumps on PRN and nominal raw-IQ millisecond, constructs signed slow/fast
divergence vectors, fits clean-only robust Gaussian models, and applies the
frozen block/alarm contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from sklearn.covariance import LedoitWolf

from .trace_native_1ms import TAPS, complex_taps, read_records


MIN_COMMON_PRNS = 4
BLOCK_MS = 100
PROMPT_EPSILON = 1e-9
PROMPT_MIN_MAGNITUDE = 1e-6


@dataclass(frozen=True)
class AlignedDivergence:
    dataset: np.ndarray
    prn: np.ndarray
    epoch_ms: np.ndarray
    raw_start_slow: np.ndarray
    raw_start_fast: np.ndarray
    time_s: np.ndarray
    state: np.ndarray
    action: np.ndarray
    taps: np.ndarray
    slow_taps: np.ndarray
    fast_taps: np.ndarray
    cn0_min: np.ndarray

    @property
    def full(self) -> np.ndarray:
        return np.column_stack((self.state, self.action, self.taps))


@dataclass(frozen=True)
class RobustGaussian:
    center: np.ndarray
    precision: np.ndarray
    covariance: np.ndarray
    regularization: float


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_normalize(taps: np.ndarray, *, epsilon: float = PROMPT_EPSILON,
                     min_magnitude: float = PROMPT_MIN_MAGNITUDE) -> tuple[np.ndarray, np.ndarray]:
    """Divide complex taps by Prompt with an explicit magnitude gate."""
    values = np.asarray(taps, dtype=np.complex128)
    if values.ndim != 2 or values.shape[1] != len(TAPS):
        raise ValueError("expected [N,9] complex taps")
    prompt = values[:, 4]
    magnitude = np.abs(prompt)
    valid = np.isfinite(values.real).all(axis=1) & np.isfinite(values.imag).all(axis=1)
    valid &= magnitude >= min_magnitude
    denominator = np.where(magnitude >= min_magnitude, prompt, epsilon + 0j)
    return values / denominator[:, None], valid


def unwrap_by_prn(prn: np.ndarray, epoch_ms: np.ndarray, phase: np.ndarray) -> np.ndarray:
    output = np.asarray(phase, dtype=np.float64).copy()
    for value in np.unique(prn):
        idx = np.flatnonzero(prn == value)
        idx = idx[np.argsort(epoch_ms[idx], kind="stable")]
        output[idx] = np.unwrap(output[idx])
    return output


def nominal_epoch_ms(raw_start: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    samples_per_ms = sample_rate_hz / 1000.0
    return np.rint(np.asarray(raw_start, dtype=np.float64) / samples_per_ms).astype(np.int64)


def _load_dump_directory(path: Path | str) -> tuple[float, np.ndarray]:
    arrays = []
    sample_rates = set()
    for dump in sorted(Path(path).glob("trace_native_1ms_ch_*.bin")):
        header, records = read_records(dump)
        sample_rates.add(header.sample_rate_hz)
        if len(records):
            arrays.append(np.asarray(records))
    if not arrays:
        raise ValueError(f"no physical native rows in {path}")
    if len(sample_rates) != 1:
        raise ValueError("dump directory contains multiple sample rates")
    return sample_rates.pop(), np.concatenate(arrays)


def _unique_rows(records: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epochs = nominal_epoch_ms(records["raw_interval_start_sample"], fs)
    keys = np.rec.fromarrays((records["prn"].astype(np.int64), epochs), names="prn,epoch")
    order = np.lexsort((records["raw_interval_start_sample"], epochs, records["prn"]))
    sorted_keys = keys[order]
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
    chosen = order[keep]
    return records[chosen], records["prn"][chosen].astype(np.int64), epochs[chosen]


def align_dump_directories(
    slow_dir: Path | str,
    fast_dir: Path | str,
    *,
    dataset: str,
    min_prompt_magnitude: float = PROMPT_MIN_MAGNITUDE,
    minimum_cn0_db_hz: float = 20.0,
) -> AlignedDivergence:
    """Align counterfactual receiver rows without interpolating state."""
    fs_slow, slow = _load_dump_directory(slow_dir)
    fs_fast, fast = _load_dump_directory(fast_dir)
    if fs_slow != fs_fast:
        raise ValueError("slow/fast sample-rate mismatch")
    slow, sprn, sepoch = _unique_rows(slow, fs_slow)
    fast, fprn, fepoch = _unique_rows(fast, fs_fast)
    skey = np.rec.fromarrays((sprn, sepoch), names="prn,epoch")
    fkey = np.rec.fromarrays((fprn, fepoch), names="prn,epoch")
    _, si, fi = np.intersect1d(skey, fkey, assume_unique=True, return_indices=True)
    slow, fast = slow[si], fast[fi]
    prn, epoch = sprn[si], sepoch[si]
    slow_norm, slow_prompt_ok = prompt_normalize(complex_taps(slow), min_magnitude=min_prompt_magnitude)
    fast_norm, fast_prompt_ok = prompt_normalize(complex_taps(fast), min_magnitude=min_prompt_magnitude)
    valid = slow_prompt_ok & fast_prompt_ok
    valid &= slow["valid_tracking"].astype(bool) & fast["valid_tracking"].astype(bool)
    valid &= slow["valid_lock"].astype(bool) & fast["valid_lock"].astype(bool)
    valid &= ~slow["pull_in_transitory"].astype(bool) & ~fast["pull_in_transitory"].astype(bool)
    cn0 = np.minimum(slow["cn0_db_hz"], fast["cn0_db_hz"])
    valid &= np.isfinite(cn0) & (cn0 >= minimum_cn0_db_hz)
    slow, fast, prn, epoch = slow[valid], fast[valid], prn[valid], epoch[valid]
    slow_norm, fast_norm, cn0 = slow_norm[valid], fast_norm[valid], cn0[valid]

    slow_carrier_phase = unwrap_by_prn(prn, epoch, slow["action_next_carrier_phase_accumulator_rad"])
    fast_carrier_phase = unwrap_by_prn(prn, epoch, fast["action_next_carrier_phase_accumulator_rad"])
    state = np.column_stack(
        (
            slow["action_next_residual_code_phase_chips"] - fast["action_next_residual_code_phase_chips"],
            slow_carrier_phase - fast_carrier_phase,
            slow["action_next_carrier_doppler_hz"] - fast["action_next_carrier_doppler_hz"],
            slow["action_next_code_nco_rate_chips_s"] - fast["action_next_code_nco_rate_chips_s"],
        )
    )
    action = np.column_stack(
        (
            slow["dll_discriminator_chips"] - fast["dll_discriminator_chips"],
            slow["pll_phase_error_cycles"] - fast["pll_phase_error_cycles"],
            slow["fll_frequency_error_hz"] - fast["fll_frequency_error_hz"],
            slow["action_next_dll_filter_output_chips_s"] - fast["action_next_dll_filter_output_chips_s"],
            slow["action_next_pll_fll_filter_output_hz"] - fast["action_next_pll_fll_filter_output_hz"],
        )
    )
    tap_delta = slow_norm - fast_norm
    taps = np.column_stack((tap_delta.real, tap_delta.imag))
    finite = np.isfinite(state).all(axis=1) & np.isfinite(action).all(axis=1) & np.isfinite(taps).all(axis=1)
    return AlignedDivergence(
        dataset=np.full(int(finite.sum()), dataset, dtype=object),
        prn=prn[finite], epoch_ms=epoch[finite],
        raw_start_slow=slow["raw_interval_start_sample"][finite].astype(np.int64),
        raw_start_fast=fast["raw_interval_start_sample"][finite].astype(np.int64),
        time_s=epoch[finite].astype(np.float64) / 1000.0,
        state=state[finite], action=action[finite], taps=taps[finite],
        slow_taps=np.column_stack((slow_norm[finite].real, slow_norm[finite].imag)),
        fast_taps=np.column_stack((fast_norm[finite].real, fast_norm[finite].imag)),
        cn0_min=cn0[finite],
    )


def robust_fit(values: np.ndarray, *, regularization: float = 1e-8) -> RobustGaussian:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) < max(20, x.shape[1] + 2):
        raise ValueError("insufficient clean rows for covariance fit")
    center = np.median(x, axis=0)
    residual = x - center
    scale = np.maximum(1.4826 * np.median(np.abs(residual), axis=0), 1e-9)
    radius = np.sqrt(np.sum((residual / scale) ** 2, axis=1))
    cutoff = np.quantile(radius, 0.995)
    trimmed = residual[radius <= cutoff]
    covariance = LedoitWolf(assume_centered=True).fit(trimmed).covariance_
    ridge = regularization * max(float(np.trace(covariance) / covariance.shape[0]), 1.0)
    covariance = covariance + ridge * np.eye(covariance.shape[0])
    return RobustGaussian(center, np.linalg.pinv(covariance, hermitian=True), covariance, ridge)


def mahalanobis_score(values: np.ndarray, model: RobustGaussian) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float64) - model.center
    return np.einsum("ni,ij,nj->n", residual, model.precision, residual)


def epoch_scores(epoch_ms: np.ndarray, prn: np.ndarray, scores: np.ndarray,
                 *, minimum_prns: int = MIN_COMMON_PRNS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epochs = np.unique(epoch_ms)
    out_epoch, out_score, out_n = [], [], []
    for epoch in epochs:
        mask = epoch_ms == epoch
        unique_prns = np.unique(prn[mask])
        if len(unique_prns) < minimum_prns:
            continue
        out_epoch.append(epoch)
        out_score.append(float(np.median(scores[mask])))
        out_n.append(len(unique_prns))
    return np.asarray(out_epoch, dtype=np.int64), np.asarray(out_score), np.asarray(out_n, dtype=np.int64)


def nonoverlap_blocks(epoch_ms: np.ndarray, scores: np.ndarray, *, block_ms: int = BLOCK_MS,
                      minimum_epochs: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block = (np.asarray(epoch_ms, dtype=np.int64) // block_ms) * block_ms
    out_block, out_score, out_count = [], [], []
    for value in np.unique(block):
        mask = block == value
        if int(mask.sum()) < minimum_epochs:
            continue
        out_block.append(value)
        out_score.append(float(np.median(np.asarray(scores)[mask])))
        out_count.append(int(mask.sum()))
    return np.asarray(out_block, dtype=np.int64), np.asarray(out_score), np.asarray(out_count, dtype=np.int64)


def consecutive_alarms(block_ms: np.ndarray, scores: np.ndarray, threshold: float,
                       *, required: int = 3, cadence_ms: int = BLOCK_MS) -> np.ndarray:
    block_ms = np.asarray(block_ms, dtype=np.int64)
    exceeded = np.asarray(scores) > threshold
    alarm = np.zeros(len(exceeded), dtype=bool)
    run = 0
    previous = None
    for index, (block, flag) in enumerate(zip(block_ms, exceeded, strict=True)):
        if previous is None or block - previous != cadence_ms:
            run = 0
        run = run + 1 if flag else 0
        alarm[index] = run >= required
        previous = block
    return alarm


def chronological_masks(time_s: np.ndarray, *, guard_s: float = 5.0) -> Mapping[str, np.ndarray]:
    time = np.asarray(time_s, dtype=np.float64)
    lo, hi = float(time.min()), float(time.max())
    span = hi - lo
    cuts = [lo + span * value for value in (0.40, 0.60, 0.80)]
    return {
        "train": time < cuts[0],
        "validation": (time >= cuts[0] + guard_s) & (time < cuts[1]),
        "calibration": (time >= cuts[1] + guard_s) & (time < cuts[2]),
        "holdout": time >= cuts[2] + guard_s,
    }


def permutation_invariant_score(epoch_ms: np.ndarray, prn: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return epoch_scores(epoch_ms, prn, scores)


def paired_bootstrap_blocks(times_s: np.ndarray, *, width_s: float = 10.0) -> np.ndarray:
    return np.floor(np.asarray(times_s, dtype=np.float64) / width_s).astype(np.int64)

