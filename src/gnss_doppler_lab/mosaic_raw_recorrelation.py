"""Receiver-faithful cleanStatic raw-IQ recorrelation for MOSAIC Stage-0A R1."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr

from .acquisition_surface import gps_l1ca_code
from .trace_native_1ms import TAPS, complex_taps, read_records


TAP_OFFSETS_CHIPS = np.arange(-4, 5, dtype=np.float64) * 0.125
FROZEN_GATE = {
    "delay_center_error_abs_max_chips": 0.125,
    "doppler_center_error_abs_max_hz": 50.0,
    "complex_tap_cosine_min": 0.90,
    "magnitude_spearman_min": 0.90,
}


@dataclass(frozen=True)
class RecorrelationResult:
    reconstructed_taps: np.ndarray
    fitted_taps: np.ndarray
    complex_amplitude: complex
    complex_cosine: float
    magnitude_spearman: float
    delay_center_error_chips: float
    doppler_center_error_hz: float
    prompt_magnitude_ratio: float
    prompt_phase_error_rad: float
    gate_pass: bool


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receiver_l1ca_code(prn: int) -> np.ndarray:
    """Return the sign convention emitted by GNSS-SDR gps_l1_ca_code_gen_float.

    The repository's canonical generator uses the equally valid opposite global
    sign.  GNSS-SDR's source maps XOR=true to +1, hence the negation here.
    """
    return -gps_l1ca_code(prn)


def read_ishort_complex_window(path: str | Path, start_sample: int, sample_count: int) -> np.ndarray:
    """Read interleaved little-endian int16 I/Q without scanning the recording."""
    if start_sample < 0 or sample_count <= 0:
        raise ValueError("invalid raw sample window")
    byte_offset = start_sample * 4
    byte_count = sample_count * 4
    with Path(path).open("rb") as stream:
        stream.seek(byte_offset)
        payload = stream.read(byte_count)
    if len(payload) != byte_count:
        raise ValueError("raw sample window crosses source boundary")
    raw = np.frombuffer(payload, dtype="<i2")
    return raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)


def receiver_code_replicas(
    prn: int,
    sample_count: int,
    code_phase_step_chips_per_sample: float,
    residual_code_phase_chips: float,
    tap_offsets_chips: Iterable[float] = TAP_OFFSETS_CHIPS,
) -> np.ndarray:
    """Reproduce the non-high-dynamics VOLK resampler floor/mod convention."""
    n = np.arange(sample_count, dtype=np.float32)
    step = np.float32(code_phase_step_chips_per_sample)
    residual = np.float32(residual_code_phase_chips)
    shifts = np.asarray(tuple(tap_offsets_chips), dtype=np.float32)
    indices = np.floor(step * n[None, :] + shifts[:, None] - residual).astype(np.int64) % 1023
    return receiver_l1ca_code(prn)[indices]


def receiver_carrier_wipeoff(
    sample_count: int,
    residual_carrier_phase_rad: float,
    carrier_phase_step_rad_per_sample: float,
) -> np.ndarray:
    """Reproduce GNSS-SDR's exp(-j phase) baseband carrier wipeoff sign."""
    n = np.arange(sample_count, dtype=np.float32)
    phase = np.float32(residual_carrier_phase_rad) + np.float32(carrier_phase_step_rad_per_sample) * n
    return np.exp(-1j * phase).astype(np.complex64)


def correlate_nine_taps(iq: np.ndarray, *, prn: int, action: np.void, tap_offsets_chips: Iterable[float]) -> np.ndarray:
    samples = np.asarray(iq, dtype=np.complex64)
    replicas = receiver_code_replicas(
        prn,
        samples.size,
        float(action["action_used_code_phase_step_chips_per_sample"]),
        float(action["action_used_residual_code_phase_chips"]),
        tap_offsets_chips,
    )
    wipeoff = receiver_carrier_wipeoff(
        samples.size,
        float(action["action_used_residual_carrier_phase_rad"]),
        float(action["action_used_carrier_phase_step_rad_per_sample"]),
    )
    return np.sum(replicas * (samples * wipeoff)[None, :], axis=1, dtype=np.complex64).astype(np.complex128)


def fit_complex_amplitude(reconstructed: np.ndarray, native: np.ndarray) -> tuple[complex, np.ndarray]:
    reconstructed = np.asarray(reconstructed, dtype=np.complex128)
    native = np.asarray(native, dtype=np.complex128)
    denominator = np.vdot(reconstructed, reconstructed)
    if not np.isfinite(denominator) or abs(denominator) <= 1e-18:
        raise ValueError("zero or non-finite reconstructed tap energy")
    alpha = complex(np.vdot(reconstructed, native) / denominator)
    return alpha, alpha * reconstructed


def prompt_normalize_safe(taps: np.ndarray, *, floor: float = 1e-12) -> np.ndarray:
    taps = np.asarray(taps, dtype=np.complex128)
    if taps.shape != (9,) or not np.isfinite(taps).all() or abs(taps[4]) <= floor:
        raise ValueError("unsafe Prompt normalization")
    return taps / taps[4]


def normalized_complex_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.complex128)
    right = np.asarray(right, dtype=np.complex128)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 1e-18 or not np.isfinite(denominator):
        raise ValueError("zero or non-finite tap-vector norm")
    return float(abs(np.vdot(left, right)) / denominator)


def evaluate_recorrelation(reconstructed: np.ndarray, native: np.ndarray, action: np.void, sample_rate_hz: float) -> RecorrelationResult:
    alpha, fitted = fit_complex_amplitude(reconstructed, native)
    cosine = normalized_complex_cosine(fitted, native)
    rho = float(spearmanr(np.abs(fitted), np.abs(native)).statistic)
    residual_from_samples = (
        float(action["action_used_code_nco_rate_chips_s"])
        * float(action["action_used_residual_code_phase_samples"])
        / sample_rate_hz
    )
    delay_error = float(action["action_used_residual_code_phase_chips"]) - residual_from_samples
    doppler_from_step = float(action["action_used_carrier_phase_step_rad_per_sample"]) * sample_rate_hz / (2.0 * np.pi)
    doppler_error = float(action["action_used_carrier_doppler_hz"]) - doppler_from_step
    native_prompt = complex(native[4])
    fitted_prompt = complex(fitted[4])
    if abs(native_prompt) <= 1e-12 or abs(fitted_prompt) <= 1e-12:
        raise ValueError("unsafe Prompt consistency metric")
    prompt_ratio = abs(fitted_prompt) / abs(native_prompt)
    prompt_phase = float(np.angle(fitted_prompt * np.conj(native_prompt)))
    passed = bool(
        abs(delay_error) <= FROZEN_GATE["delay_center_error_abs_max_chips"]
        and abs(doppler_error) <= FROZEN_GATE["doppler_center_error_abs_max_hz"]
        and cosine >= FROZEN_GATE["complex_tap_cosine_min"]
        and rho >= FROZEN_GATE["magnitude_spearman_min"]
    )
    return RecorrelationResult(
        reconstructed_taps=np.asarray(reconstructed, dtype=np.complex128),
        fitted_taps=fitted,
        complex_amplitude=alpha,
        complex_cosine=cosine,
        magnitude_spearman=rho,
        delay_center_error_chips=delay_error,
        doppler_center_error_hz=doppler_error,
        prompt_magnitude_ratio=float(prompt_ratio),
        prompt_phase_error_rad=prompt_phase,
        gate_pass=passed,
    )


def select_epoch_records(
    dump_dir: str | Path,
    target_prns: Iterable[int],
    *,
    legacy_total: int = 120,
    legacy_per_dump: int = 25,
    temporal_per_prn: int = 24,
    minimum_cn0_db_hz: float = 28.0,
    minimum_lock: float = 0.85,
) -> list[dict[str, object]]:
    """Reuse the frozen legacy support and add deterministic temporal coverage."""
    wanted = set(int(p) for p in target_prns)
    candidates: dict[int, tuple[Path, np.ndarray, np.ndarray]] = {}
    legacy: list[tuple[Path, int, np.void]] = []
    for path in sorted(Path(dump_dir).glob("trace_native_1ms_ch_*.bin")):
        _, records = read_records(path, mmap=True)
        ok = (
            (records["valid_tracking"] == 1)
            & (records["valid_lock"] == 1)
            & (records["cn0_db_hz"] >= minimum_cn0_db_hz)
            & (records["carrier_lock_test"] >= minimum_lock)
        )
        valid = np.flatnonzero(ok)
        if not valid.size:
            continue
        prn = int(records[valid[0]]["prn"])
        if prn not in wanted:
            continue
        candidates[prn] = (path, records, valid)
        for index in valid[:legacy_per_dump]:
            if len(legacy) < legacy_total:
                legacy.append((path, int(index), records[index]))
    if set(candidates) != wanted:
        raise ValueError(f"missing target PRNs: {sorted(wanted - set(candidates))}")

    selected: dict[tuple[str, int], dict[str, object]] = {}
    for path, index, record in legacy:
        selected[(str(path), index)] = {"source_dump": str(path), "record_index": index, "selection_origin": "legacy_stage0a"}
    for prn in sorted(wanted):
        path, records, valid = candidates[prn]
        if valid.size < temporal_per_prn:
            raise ValueError(f"PRN {prn} has insufficient stable temporal support")
        positions = np.linspace(0, valid.size - 1, temporal_per_prn, dtype=np.int64)
        for index in valid[positions]:
            key = (str(path), int(index))
            if key in selected:
                selected[key]["selection_origin"] = "legacy_stage0a+temporal"
            else:
                selected[key] = {"source_dump": str(path), "record_index": int(index), "selection_origin": "temporal_extension"}

    out: list[dict[str, object]] = []
    cache: dict[str, np.ndarray] = {}
    for item in selected.values():
        path = item["source_dump"]
        if path not in cache:
            _, cache[path] = read_records(path, mmap=True)
        rec = cache[path][int(item["record_index"])]
        out.append(item | {
            "prn": int(rec["prn"]),
            "channel": int(rec["channel"]),
            "loop_sequence": int(rec["loop_sequence"]),
            "raw_sample_start": int(rec["raw_interval_start_sample"]),
            "raw_sample_end": int(rec["raw_interval_end_sample"]),
            "receiver_timestamp_s": float(rec["receiver_timestamp_s"]),
        })
    return sorted(out, key=lambda row: (int(row["prn"]), int(row["raw_sample_start"])))


def native_taps_for_item(item: dict[str, object]) -> tuple[object, np.ndarray, object]:
    header, records = read_records(str(item["source_dump"]), mmap=True)
    index = int(item["record_index"])
    return header, complex_taps(records[index:index + 1])[0], records[index]
