"""Stateful R1 replica and receiver-in-the-loop preregistered contracts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .acquisition_surface import gps_l1ca_code


@dataclass
class ReplicaState:
    absolute_sample_index: int
    code_phase_chips: float
    carrier_phase_rad: float


class StatefulReplica:
    """Replica whose state advances across chunks, epochs, and NAV boundaries."""
    def __init__(self, prn: int, sample_rate_hz: float, state: ReplicaState):
        if sample_rate_hz <= 0:
            raise ValueError("sample rate must be positive")
        self.prn = int(prn)
        self.fs = float(sample_rate_hz)
        self.state = state
        self.code = gps_l1ca_code(prn).astype(np.float64)

    def render(self, sample_count: int, *, code_rate_chips_s: float, carrier_doppler_hz: float,
               nav_signs: np.ndarray, delay_chips: float = 0.0, delta_f_hz: float = 0.0,
               phase_offset_rad: float = 0.0) -> np.ndarray:
        if sample_count < 0:
            raise ValueError("negative sample count")
        signs = np.asarray(nav_signs, dtype=np.float64)
        if signs.shape != (sample_count,) or not np.all(np.isin(signs, [-1.0, 1.0])):
            raise ValueError("NAV signs must provide one ±1 value per sample")
        n = np.arange(sample_count, dtype=np.float64)
        code_phase = self.state.code_phase_chips + n * float(code_rate_chips_s) / self.fs
        indices = np.floor(code_phase + float(delay_chips)).astype(np.int64) % 1023
        omega = 2 * np.pi * (float(carrier_doppler_hz) + float(delta_f_hz)) / self.fs
        carrier = np.exp(1j * (self.state.carrier_phase_rad + phase_offset_rad + omega * n))
        result = self.code[indices] * signs * carrier
        self.state.code_phase_chips = float((self.state.code_phase_chips + sample_count * float(code_rate_chips_s) / self.fs) % 1023)
        self.state.carrier_phase_rad = float((self.state.carrier_phase_rad + sample_count * omega) % (2 * np.pi))
        self.state.absolute_sample_index += sample_count
        return result.astype(np.complex128)


class NavBitTimeline:
    def __init__(self, mapping_rows: list[dict[str, str]], *, prn: int):
        rows = sorted((r for r in mapping_rows if int(r["prn"]) == prn), key=lambda r: int(r["corrected_raw_start_sample"]))
        if not rows:
            raise ValueError("missing R0c NAV mapping")
        self.starts = np.asarray([int(r["corrected_raw_start_sample"]) for r in rows], dtype=np.int64)
        self.signs_pm1 = np.asarray([int(r["bit_value_pm1"]) for r in rows], dtype=np.int8)
        self.final_end = int(rows[-1]["corrected_raw_end_sample_exclusive"])
        if np.any(np.diff(self.starts) <= 0):
            raise ValueError("non-monotonic NAV mapping")

    def signs(self, absolute_start: int, sample_count: int) -> np.ndarray:
        positions = absolute_start + np.arange(sample_count, dtype=np.int64)
        indices = np.searchsorted(self.starts, positions, side="right") - 1
        if sample_count and (indices.min() < 0 or positions[-1] >= self.final_end):
            raise ValueError("sample range outside frozen R0c NAV mapping")
        return self.signs_pm1[indices].astype(np.float64)


@dataclass(frozen=True)
class ReceiverNcoEpoch:
    raw_start_sample: int
    raw_end_sample_exclusive: int
    code_rate_chips_s: float
    carrier_doppler_hz: float


def iter_stateful_replica_epochs(replica: StatefulReplica, epochs: list[ReceiverNcoEpoch], nav: NavBitTimeline,
                                 *, delay_chips: float, delta_f_hz: float, phase_offset_rad: float):
    """Render each authenticated receiver-NCO epoch without resetting state."""
    for epoch in epochs:
        if epoch.raw_start_sample != replica.state.absolute_sample_index:
            raise ValueError("receiver NCO schedule is not continuous at the absolute sample index")
        count = epoch.raw_end_sample_exclusive - epoch.raw_start_sample
        if count <= 0:
            raise ValueError("invalid receiver NCO epoch")
        signs = nav.signs(epoch.raw_start_sample, count)
        yield epoch.raw_start_sample, replica.render(
            count, code_rate_chips_s=epoch.code_rate_chips_s,
            carrier_doppler_hz=epoch.carrier_doppler_hz, nav_signs=signs,
            delay_chips=delay_chips, delta_f_hz=delta_f_hz, phase_offset_rad=phase_offset_rad)


def estimate_authentic_amplitude(clean_iq: np.ndarray, target_prn_replica: np.ndarray) -> complex:
    clean = np.asarray(clean_iq, dtype=np.complex128)
    replica = np.asarray(target_prn_replica, dtype=np.complex128)
    if clean.shape != replica.shape or not clean.size:
        raise ValueError("clean and replica must be non-empty and equal length")
    denominator = np.vdot(replica, replica)
    if abs(denominator) == 0:
        raise ValueError("zero-energy target PRN replica")
    return complex(np.vdot(replica, clean) / denominator)


def spoof_amplitude(alpha_auth: complex, rho_db: float) -> float:
    return float(abs(alpha_auth) * 10.0 ** (float(rho_db) / 20.0))


def realized_scer_db(alpha_spoof: complex | float, alpha_auth: complex) -> float:
    if abs(alpha_auth) == 0:
        raise ValueError("authentic PRN amplitude is zero")
    return float(20 * np.log10(abs(alpha_spoof) / abs(alpha_auth)))


def raised_cosine_envelope(relative_seconds: np.ndarray, duration_seconds: float) -> np.ndarray:
    t = np.asarray(relative_seconds, dtype=np.float64)
    if duration_seconds <= 10:
        raise ValueError("validated interval must extend beyond ten seconds")
    out = np.zeros(t.shape, dtype=np.float64)
    ramp = (t >= 2) & (t < 4)
    out[ramp] = 0.5 * (1 - np.cos(np.pi * (t[ramp] - 2) / 2))
    out[(t >= 4) & (t < 10)] = 1.0
    release = (t >= 10) & (t <= duration_seconds)
    out[release] = 0.5 * (1 + np.cos(np.pi * (t[release] - 10) / (duration_seconds - 10)))
    return out


def assign_case_targets(design: list[dict[str, object]], prns_by_dataset: dict[str, list[int]]) -> list[dict[str, object]]:
    counters: dict[tuple[str, str], int] = {}
    assigned = []
    for case in design:
        dataset, mode = str(case["dataset"]), str(case["mode"])
        key = (dataset, mode)
        index = counters.get(key, 0); counters[key] = index + 1
        prns = sorted(prns_by_dataset[dataset])
        if len(prns) != 5:
            raise ValueError("R1 assignment requires exactly five sorted PRNs")
        if mode == "single_prn":
            targets, excluded = [prns[index % 5]], None
        elif mode == "four_prn_diagnostic_after_single_prn_pass_only":
            excluded = prns[index % 5]; targets = [p for p in prns if p != excluded]
        else:
            raise ValueError(f"unknown case mode {mode}")
        assigned.append({"case_id": case["case_id"], "dataset": dataset, "mode": mode,
                         "mode_case_index": index, "sorted_prns": prns, "target_prns": targets,
                         "excluded_prn": excluded})
    return assigned


def verify_sha256(path: str | Path, expected: str) -> dict[str, object]:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"source hash mismatch: {digest} != {expected}")
    return {"path": str(path), "sha256": digest, "status": "PASS"}


def receiver_command(executable: str | Path, config: str | Path) -> list[str]:
    return [str(executable), f"--config_file={Path(config)}", "--keyboard=false"]
