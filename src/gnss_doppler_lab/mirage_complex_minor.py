"""MIRAGE complex delay-Doppler minor primitives.

The primary representation is the normalized field of adjacent complex 2x2
determinants.  Absolute CAF power and SVD ratios are diagnostics only.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np

DELAY_GRID_CHIPS = np.arange(-4, 5, dtype=np.float64) * 0.125
NORMALIZED_DOPPLER_GRID = np.arange(-2, 3, dtype=np.float64)
INTEGRATION_TIMES_S = (0.020, 0.100, 0.500)


def doppler_grid_hz(integration_s: float) -> np.ndarray:
    """Return offsets with frozen normalized frequency xi=delta_f*T."""
    value = float(integration_s)
    if value not in INTEGRATION_TIMES_S:
        raise ValueError("unsupported MIRAGE integration time")
    return NORMALIZED_DOPPLER_GRID / value


def normalized_complex_minors(caf: np.ndarray, epsilon: float = 1e-24) -> np.ndarray:
    """Compute 8x4 adjacent normalized complex-minor magnitudes."""
    matrix = np.asarray(caf, dtype=np.complex128)
    if matrix.shape != (9, 5):
        raise ValueError("MIRAGE CAF must be 9x5")
    if epsilon <= 0 or not np.isfinite(matrix).all():
        raise ValueError("finite CAF and positive epsilon required")
    x = matrix[:-1, :-1] * matrix[1:, 1:]
    y = matrix[:-1, 1:] * matrix[1:, :-1]
    denominator = np.sqrt(np.abs(x) ** 2 + np.abs(y) ** 2 + epsilon)
    result = np.abs(x - y) / denominator
    if result.shape != (8, 4) or not np.isfinite(result).all():
        raise AssertionError("invalid complex-minor field")
    return result


def magnitude_minors(caf: np.ndarray, epsilon: float = 1e-24) -> np.ndarray:
    return normalized_complex_minors(np.abs(np.asarray(caf, np.complex128)).astype(np.complex128), epsilon)


def svd_second_energy_ratio(caf: np.ndarray, epsilon: float = 1e-24) -> float:
    singular = np.linalg.svd(np.asarray(caf, np.complex128), compute_uv=False)
    return float(singular[1] ** 2 / (np.sum(singular ** 2) + epsilon))


def caf_total_energy(caf: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(caf, np.complex128)) ** 2))


def robust_minor_reference(fields: np.ndarray, scale_floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Median/MAD reference estimated only from clean train minor squares."""
    values = np.asarray(fields, np.float64)
    if values.ndim != 3 or values.shape[1:] != (8, 4) or len(values) < 3:
        raise ValueError("at least three train minor fields required")
    squared = values ** 2
    location = np.median(squared, axis=0)
    mad = np.median(np.abs(squared - location), axis=0) * 1.4826
    scale = np.maximum(mad, float(scale_floor))
    return location, scale


def empirical_surprise(field: np.ndarray, train_fields: np.ndarray, *, epsilon: float = 1e-12) -> float:
    """Right-tail empirical-CDF surprise of robust standardized minor squares."""
    train = np.asarray(train_fields, np.float64)
    location, scale = robust_minor_reference(train)
    train_stat = np.mean(np.maximum((train ** 2 - location) / scale, 0.0), axis=(1, 2))
    value = float(np.mean(np.maximum((np.asarray(field) ** 2 - location) / scale, 0.0)))
    cdf = (1.0 + np.count_nonzero(train_stat <= value)) / (len(train_stat) + 1.0)
    return float(-np.log(max(1.0 - cdf, epsilon)))


def node_score(scale_surprises: Iterable[float]) -> float:
    values = np.asarray(tuple(scale_surprises), dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("three finite scale surprises required")
    return float(np.max(values))


def full_score(node_scores: Iterable[float], minimum_prns: int = 4) -> float | None:
    values = np.asarray(tuple(node_scores), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) >= minimum_prns else None


def deterministic_design(seed: int, datasets: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    """Balanced 30 single-PRN plus six four-PRN cases per dataset."""
    rho = (-10, -6, 0)
    delay = (-.50, -.25, -.10, .10, .25, .50)
    doppler = (0, -2, 2, -5, 5)
    phase = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
    rows: list[dict[str, object]] = []
    for dataset_index, (dataset, spec) in enumerate(sorted(datasets.items())):
        prns = tuple(int(x) for x in spec["prns"])
        anchors = tuple(int(x) for x in spec["anchor_start_samples"])
        if len(prns) < 5 or len(anchors) != 6:
            raise ValueError("design requires at least five PRNs and six anchors")
        for p_index, prn in enumerate(prns):
            for a_index, anchor in enumerate(anchors):
                k = p_index * len(anchors) + a_index
                rows.append({"case_id": f"{dataset}.single.p{prn}.a{a_index}", "dataset": dataset,
                    "mode": "single_prn", "anchor_index": a_index, "anchor_start_sample": anchor,
                    "target_prns": [prn], "rho_db": rho[k % len(rho)],
                    "delta_tau_chips": delay[k % len(delay)],
                    "delta_f_hz": doppler[k % len(doppler)],
                    "relative_phase_rad": float(phase[k % len(phase)])})
        for a_index, anchor in enumerate(anchors):
            k = len(prns) * len(anchors) + a_index
            excluded = prns[k % len(prns)]
            rows.append({"case_id": f"{dataset}.four.a{a_index}", "dataset": dataset,
                "mode": "simultaneous_four_prn", "anchor_index": a_index,
                "anchor_start_sample": anchor, "target_prns": [p for p in prns if p != excluded][:4],
                "rho_db": rho[k % len(rho)], "delta_tau_chips": delay[k % len(delay)],
                "delta_f_hz": doppler[k % len(doppler)],
                "relative_phase_rad": float(phase[k % len(phase)])})
    return rows


def design_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def requested_split_span_seconds(role_seconds: float = 3.0, guard_seconds: float = 10.0) -> float:
    if role_seconds <= 0 or guard_seconds < 10:
        raise ValueError("invalid chronological split contract")
    return 3 * role_seconds + 2 * guard_seconds


def split_support_audit(authorized_duration_s: float, *, role_seconds: float = 3.0,
                        guard_seconds: float = 10.0) -> dict[str, object]:
    required = requested_split_span_seconds(role_seconds, guard_seconds)
    return {"authorized_duration_s": float(authorized_duration_s), "role_seconds_each": role_seconds,
            "guard_seconds_each": guard_seconds, "required_duration_s": required,
            "deficit_seconds": max(0.0, required - float(authorized_duration_s)),
            "status": "PASS" if authorized_duration_s >= required else "FAIL"}


def temporal_desynchronize(sequences: np.ndarray, shifts: Iterable[int]) -> np.ndarray:
    values = np.asarray(sequences, dtype=np.float64)
    offsets = tuple(int(x) for x in shifts)
    if values.ndim != 2 or len(offsets) != values.shape[1]:
        raise ValueError("one deterministic temporal shift per PRN required")
    return np.column_stack([np.roll(values[:, i], offsets[i]) for i in range(values.shape[1])])


def clean_calibration_threshold(clean_scores: Iterable[float], quantile: float = 0.99) -> float:
    values = np.asarray(tuple(clean_scores), dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or quantile not in (0.99, 0.995):
        raise ValueError("finite clean calibration scores and frozen quantile required")
    return float(np.quantile(values, quantile, method="higher"))


def raw_ranges_nonoverlap(ranges: Iterable[tuple[int, int]]) -> bool:
    ordered = sorted((int(a), int(b)) for a, b in ranges)
    if any(a < 0 or b <= a for a, b in ordered):
        raise ValueError("invalid raw range")
    return all(ordered[i][1] <= ordered[i + 1][0] for i in range(len(ordered) - 1))


def validate_clean_source_path(path: str) -> None:
    lower = str(path).lower()
    forbidden = tuple(f"/{prefix}{index}" for prefix, stop in (("ds", 8), ("os", 4)) for index in range(1, stop + 1))
    if "cleanstatic" not in lower or any(token in lower for token in forbidden):
        raise ValueError("MIRAGE Stage-0A accepts only explicit cleanStatic sources")


def nav_wipeoff(samples: np.ndarray, nav_signs: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.complex128)
    signs = np.asarray(nav_signs, dtype=np.float64)
    if values.shape != signs.shape or not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("one authenticated +/-1 NAV sign per sample required")
    return values * signs
