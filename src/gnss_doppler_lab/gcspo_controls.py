"""Deterministic clean-only falsification controls for GCSPO Stage 0."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np

from .gcspo_core import apply_var_transfer, build_physical_loading


_STAGES = {
    "COMMON_GAIN": "raw_q",
    "PROMPT_AMPLITUDE": "raw_q",
    "CN0_METADATA_EXCLUSION_INVARIANCE": "metadata_only",
    "PRN_DROP_ONLY": "mask",
    "EMPIRICAL_NOISE": "pre_whitening_residual",
    "ONE_PRN_DISTURBANCE": "pre_whitening_residual",
    "INDEPENDENT_MULTIPATH_LIKE": "pre_whitening_residual",
    "CLOCK_DRIFT": "physical_state",
}


@dataclass(frozen=True)
class ControlContext:
    prns: np.ndarray
    times_s: np.ndarray
    raw_complex: np.ndarray
    other_q: np.ndarray
    epsilon_by_prn: np.ndarray
    residual: np.ndarray
    source_residual: np.ndarray
    cn0: np.ndarray
    robust_scale: np.ndarray | None = None
    channels: np.ndarray | None = None
    segments: np.ndarray | None = None
    source_block_index: int | None = None
    source_prns: np.ndarray | None = None


@dataclass(frozen=True)
class ControlResult:
    stage: str
    seed_material: str
    seed: int
    history_reset: bool
    first_eligible_epoch: int
    numpy_version: str
    prns: np.ndarray
    times_s: np.ndarray
    channels: np.ndarray
    segments: np.ndarray
    q: np.ndarray
    baseline_q: np.ndarray
    raw_complex: np.ndarray
    residual: np.ndarray
    cn0: np.ndarray
    source_mapping: dict[int, int] = field(default_factory=dict)
    prn_phases_rad: dict[int, float] = field(default_factory=dict)
    var_transfer_application_count: int = 0
    state_energy: dict[str, float] = field(default_factory=dict)
    source_block_index: int | None = None


def _level_text(level):
    return format(float(level), ".17g") if isinstance(level, (float, np.floating)) else str(level)


def _seed(control_id, scenario, phase, block_id, level, object_id="BLOCK"):
    material = f"23|{control_id}|{scenario}|{phase}|{block_id}|{_level_text(level)}|{object_id}"
    return material, int.from_bytes(hashlib.sha256(material.encode()).digest()[:16], "big")


def _validated(context):
    prns = np.asarray(context.prns, dtype=np.int64)
    times = np.asarray(context.times_s, dtype=np.float64)
    raw = np.asarray(context.raw_complex, dtype=np.float64)
    other = np.asarray(context.other_q, dtype=np.float64)
    epsilon = np.asarray(context.epsilon_by_prn, dtype=np.float64)
    residual = np.asarray(context.residual, dtype=np.float64)
    source = np.asarray(context.source_residual, dtype=np.float64)
    cn0 = np.asarray(context.cn0, dtype=np.float64)
    p, t = len(prns), len(times)
    if (raw.shape != (p, t, 6) or other.shape != (p, t, 4) or epsilon.shape != (p,)
            or residual.shape != (p, t, 10) or source.ndim != 3 or source.shape[1:] != (t, 10) or cn0.shape != (p, t)):
        raise ValueError("control context shape mismatch")
    if len(set(map(int, prns))) != p or p < 4 or t < 11:
        raise ValueError("control context requires distinct PRNs and at least eleven epochs")
    arrays = (times, raw, other, epsilon, residual, source, cn0)
    if context.robust_scale is not None and np.asarray(context.robust_scale).shape != (p, 10):
        raise ValueError("control robust-scale shape mismatch")
    if not all(np.all(np.isfinite(value)) for value in arrays) or np.any(epsilon <= 0):
        raise ValueError("control context contains invalid numeric values")
    if np.any(np.diff(times) <= 0):
        raise ValueError("control epochs must be strictly increasing")
    return prns, times, raw, other, epsilon, residual, source, cn0


def _q(raw, other, epsilon):
    denominator = np.sqrt(np.sum(raw * raw, axis=2)) + epsilon[:, None]
    return np.concatenate((raw / denominator[:, :, None], other), axis=2)


def holdout_blocks(start_s, end_s):
    duration = float(end_s) - float(start_s)
    count = int(round(duration / 10.0))
    if count <= 0 or not np.isclose(duration, count * 10.0, rtol=0, atol=1e-12):
        raise ValueError("control interval must contain exact non-overlapping 10 s blocks")
    return [(index, float(start_s) + 10.0 * index, float(start_s) + 10.0 * (index + 1))
            for index in range(count)]


def apply_control(context, *, control_id, level, scenario, phase, block_id, var_coefficients):
    if control_id not in _STAGES:
        raise ValueError(f"unknown control {control_id}")
    prns, times, raw0, other, epsilon, residual0, source, cn00 = _validated(context)
    source_prns = prns if context.source_prns is None else np.asarray(context.source_prns, np.int64)
    if source_prns.shape != (len(source),) or len(set(map(int, source_prns))) != len(source_prns):
        raise ValueError("control source PRN identity mismatch")
    channels = np.arange(len(prns), dtype=np.int64) if context.channels is None else np.asarray(context.channels, np.int64).copy()
    segments = np.zeros(len(prns), dtype=np.int64) if context.segments is None else np.asarray(context.segments, np.int64).copy()
    if channels.shape != prns.shape or segments.shape != prns.shape: raise ValueError("control identity shape mismatch")
    coefficients = np.asarray(var_coefficients, dtype=np.float64)
    if coefficients.ndim != 3 or coefficients.shape[1:] != (10, 10):
        raise ValueError("VAR coefficient shape mismatch")
    material, seed = _seed(control_id, scenario, phase, block_id, level)
    rng = np.random.Generator(np.random.PCG64(seed))
    raw, residual, cn0 = raw0.copy(), residual0.copy(), cn00.copy()
    baseline_q = _q(raw0, other, epsilon)
    mapping, phases, transfer_count = {}, {}, 0
    state_energy = {"Epos": 0.0, "Eclock": 0.0, "specificity_ratio": 0.0}

    if control_id == "COMMON_GAIN":
        raw *= float(level)
    elif control_id == "PROMPT_AMPLITUDE":
        raw[:, :, 2:4] *= float(level)
    elif control_id == "CN0_METADATA_EXCLUSION_INVARIANCE":
        cn0 += float(level)
    elif control_id == "PRN_DROP_ONLY":
        drop = int(level)
        if drop < 1 or drop >= len(prns):
            raise ValueError("PRN drop must retain at least one PRN")
        keep = np.ones(len(prns), dtype=bool)
        keep[rng.permutation(len(prns))[:drop]] = False
        prns, channels, segments, raw, other, epsilon = prns[keep], channels[keep], segments[keep], raw[keep], other[keep], epsilon[keep]
        residual, cn0, baseline_q = residual[keep], cn0[keep], baseline_q[keep]
    elif control_id == "EMPIRICAL_NOISE":
        centered = source - source.mean(axis=1, keepdims=True)
        offset = seed % len(source_prns)
        for target in range(len(prns)):
            source_index = (target + offset) % len(source_prns)
            mapping[int(prns[target])] = int(source_prns[source_index])
            residual[target] += float(level) * centered[source_index]
    elif control_id == "ONE_PRN_DISTURBANCE":
        target = int(rng.permutation(len(prns))[0])
        source_index = (target + seed % len(source_prns)) % len(source_prns)
        disturbance = source[source_index] - source[source_index].mean(axis=0, keepdims=True)
        residual[target] += float(level) * disturbance
        mapping[int(prns[target])] = int(source_prns[source_index])
    elif control_id == "INDEPENDENT_MULTIPATH_LIKE":
        relative_time = times - times[0]
        for index, prn in enumerate(prns):
            phase_material, phase_seed = _seed(control_id, scenario, phase, block_id, level, f"PRN:{int(prn)}")
            del phase_material
            phase_rng = np.random.Generator(np.random.PCG64(phase_seed))
            phase_rad = float(phase_rng.uniform(0., 2 * np.pi))
            phases[int(prn)] = phase_rad
            if context.robust_scale is None:
                centered = residual0[index, :, :7] - np.median(residual0[index, :, :7], axis=0)
                scale = 1.4826 * np.median(np.abs(centered), axis=0)
            else:
                scale = np.asarray(context.robust_scale, float)[index, :7]
            residual[index, :, :7] += float(level) * np.sin(2 * np.pi * .2 * relative_time + phase_rad)[:, None] * scale
    elif control_id == "CLOCK_DRIFT":
        state = np.zeros((len(times), 8), dtype=np.float64)
        state[:, 3] = float(level) * (times - times[0])
        state[:, 7] = float(level)
        direct = np.empty((len(times), 10, len(prns)), dtype=np.float64)
        for index in range(len(prns)):
            angle = 2 * np.pi * index / len(prns)
            los = np.asarray([np.cos(angle), np.sin(angle), 0.0])
            loading = build_physical_loading(los, validated_rows={"code_error_chips", "pll_phase_error_cycles",
                                                                   "carrier_doppler_hz", "code_frequency_offset_chips_s"})
            direct[:, :, index] = state @ loading.T
        transferred = apply_var_transfer(direct, coefficients)
        residual += np.transpose(transferred, (2, 0, 1))
        transfer_count = 1
        epos = float(np.median(np.linalg.norm(state[:, :3] / 10.0, axis=1)))
        eclock = float(np.median(np.sqrt((state[:, 3] / 10.0) ** 2 + state[:, 7] ** 2)))
        state_energy = {"Epos": epos, "Eclock": eclock,
                        "specificity_ratio": 0.0 if eclock == 0 else epos / eclock}

    q = _q(raw, other, epsilon)
    return ControlResult(stage=_STAGES[control_id], seed_material=material, seed=seed,
                         history_reset=True, first_eligible_epoch=10, numpy_version=np.__version__,
                         prns=prns, times_s=times, channels=channels, segments=segments, q=q, baseline_q=baseline_q, raw_complex=raw,
                         residual=residual, cn0=cn0, source_mapping=mapping,
                         prn_phases_rad=phases, var_transfer_application_count=transfer_count,
                         state_energy=state_energy, source_block_index=context.source_block_index)


from .gcspo_control_grid import generate_control_grid
