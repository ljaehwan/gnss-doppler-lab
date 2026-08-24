"""Frozen SPLITCLOCK Stage-0A physics and provenance contracts.

The real-data runner must stop before scoring when the receiver does not
provide every required observable.  The numerical helpers in this module are
therefore independently testable with observation-domain synthetic panels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BASE_SHA = "a0e687de330a8ae1844e57f936aaf906144f693f"
BRANCH = "research/splitclock-stage0a-clean-identifiability"
REFERENCE_SHA = "798667d0902efc04b09b40f5f4c6064b41c418da"
REFERENCE_CONFIG_BLOB = "46a6bc220926f0bb3aca3eba8d04b53792a07fa9"
REFERENCE_RUNNER_BLOB = "01714abd4754767f581842e4c72a342da048a179"
REFERENCE_ADAPTER_BLOB = "0d52d9165e4c0797850c9065842d1275b7bd8b01"
REFERENCE_RECEIVER_SHA256 = "3ed059f699201807cb86eb54a24ba00fde7e248f37a8081306cbc76d6be1b06f"
REFERENCE_RECEIVER_SIZE = 36_607_208

SPEED_OF_LIGHT_MPS = 299_792_458.0
GALILEO_E1_HZ = 1_575_420_000.0
GALILEO_E1_WAVELENGTH_M = SPEED_OF_LIGHT_MPS / GALILEO_E1_HZ
EPOCH_RATE_HZ = 1.0
WINDOW_SECONDS = 10
FIT_FRACTION = 0.70
STUDENT_T_DF = 4.0
RIDGE = 1e-9
SEED = 20250824

VERDICTS = {
    "STOP_SPLITCLOCK_OBSERVABLES_UNAVAILABLE",
    "STOP_SPLITCLOCK_CLEAN_PANEL_UNSUPPORTED",
    "NO_GO_SPLITCLOCK_CLEAN_FALSE_ALARMS",
    "NO_GO_SPLITCLOCK_SYNTHETIC_IDENTIFIABILITY",
    "NO_GO_SPLITCLOCK_NEGATIVE_CONTROLS",
    "INCONCLUSIVE_SPLITCLOCK_EXECUTION_OR_PROVENANCE",
    "READY_FOR_SPLITCLOCK_ATTACK_FREEZE",
}

REQUIRED_OBSERVABLES = (
    "receiver_relative_epoch_or_sample_index",
    "decoded_satellite_id",
    "pseudorange_or_absolute_code_range_m",
    "cycle_consistent_carrier_phase_or_increment",
    "carrier_doppler_hz_or_range_rate_mps",
    "code_rate",
    "decoded_receive_time_or_tow",
    "ephemeris_or_satellite_position_velocity",
    "receiver_position_for_geometry",
    "lock_reacquisition_cycle_slip_flags",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def doppler_hz_to_range_rate_mps(doppler_hz: np.ndarray | float) -> np.ndarray:
    """Frozen convention: positive Doppler means decreasing geometric range."""

    return -GALILEO_E1_WAVELENGTH_M * np.asarray(doppler_hz, dtype=float)


def carrier_phase_radians_to_increment_m(phase_radians: np.ndarray) -> np.ndarray:
    """Convert cycle-consistent accumulated phase to range increments."""

    phase = np.asarray(phase_radians, dtype=float)
    return -GALILEO_E1_WAVELENGTH_M * np.diff(phase, axis=0) / (2.0 * np.pi)


def student_t_loglik(residual: np.ndarray, scale: np.ndarray, df: float = STUDENT_T_DF) -> float:
    values = np.asarray(residual, dtype=float)
    sigma = np.maximum(np.asarray(scale, dtype=float), 1e-9)
    z2 = (values / sigma) ** 2
    constant = (
        math.lgamma((df + 1.0) / 2.0)
        - math.lgamma(df / 2.0)
        - 0.5 * math.log(df * math.pi)
    )
    return float(np.sum(constant - np.log(sigma) - 0.5 * (df + 1.0) * np.log1p(z2 / df)))


def canonical_labels(labels: np.ndarray, signatures: np.ndarray) -> np.ndarray:
    """Canonicalize K=2 without PRN identity or truth labels."""

    result = np.asarray(labels, dtype=int).copy()
    means = [float(np.mean(signatures[result == group, 0])) for group in (0, 1)]
    if means[0] > means[1]:
        result = 1 - result
    return result


def _line_fit(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack((np.ones(len(times)), times))
    gram = design.T @ design + RIDGE * np.eye(2)
    coefficients = np.linalg.solve(gram, design.T @ values)
    return coefficients, design @ coefficients


def _prn_signatures(panel: np.ndarray, fit_epochs: int) -> np.ndarray:
    times = np.arange(fit_epochs, dtype=float)
    signatures = []
    for prn in range(panel.shape[1]):
        coefficients, _ = _line_fit(times, panel[:fit_epochs, prn])
        signatures.append(coefficients.reshape(-1))
    return np.asarray(signatures)


def _deterministic_two_cluster(signatures: np.ndarray, iterations: int = 30) -> np.ndarray:
    """Deterministic K=2 clustering with no PRN identity input."""

    centered = signatures - np.median(signatures, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[0]
    labels = (projection >= np.median(projection)).astype(int)
    for _ in range(iterations):
        if min(np.sum(labels == 0), np.sum(labels == 1)) < 2:
            order = np.argsort(projection, kind="mergesort")
            labels[:] = 1
            labels[order[: max(2, len(order) // 2)]] = 0
        centroids = np.asarray([signatures[labels == group].mean(axis=0) for group in (0, 1)])
        distance = np.sum((signatures[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        updated = np.argmin(distance, axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
    return canonical_labels(labels, signatures)


@dataclass(frozen=True)
class WindowScore:
    score: float
    k1_loglik: float
    k2_loglik: float
    mdl_penalty: float
    labels: np.ndarray
    valid_observation_count: int


def score_window(panel: np.ndarray, valid_mask: np.ndarray | None = None) -> WindowScore:
    """Score one 10-epoch window with 70/30 causal fit/held-out split.

    ``panel`` has shape ``(epoch, PRN, modality)``.  A fixed assignment for the
    whole window is the Stage-0A persistence prior; per-epoch relabeling is not
    permitted.
    """

    values = np.asarray(panel, dtype=float)
    if values.ndim != 3 or values.shape[0] != WINDOW_SECONDS:
        raise ValueError("panel must have shape (10, PRN, modality)")
    if values.shape[1] < 5 or values.shape[2] < 1:
        raise ValueError("at least five PRNs and one modality are required")
    mask = np.isfinite(values) if valid_mask is None else np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    if not np.all(mask):
        raise ValueError("synthetic baseline requires a complete valid window")
    fit_epochs = int(WINDOW_SECONDS * FIT_FRACTION)
    times_fit = np.arange(fit_epochs, dtype=float)
    times_hold = np.arange(fit_epochs, WINDOW_SECONDS, dtype=float)
    modalities = values.shape[2]

    # Shared scales are fitted once and used by both models.
    shared_scale = np.maximum(np.median(np.abs(values[:fit_epochs] - np.median(values[:fit_epochs], axis=1, keepdims=True)), axis=(0, 1)) / 0.67448975, 1e-6)

    k1_prediction = np.empty((len(times_hold), modalities), dtype=float)
    for modality in range(modalities):
        path = np.median(values[:fit_epochs, :, modality], axis=1)
        coefficients, _ = _line_fit(times_fit, path[:, None])
        k1_prediction[:, modality] = np.column_stack((np.ones(len(times_hold)), times_hold)) @ coefficients[:, 0]
    k1_residual = values[fit_epochs:] - k1_prediction[:, None, :]
    k1_loglik = student_t_loglik(k1_residual, shared_scale)

    signatures = _prn_signatures(values, fit_epochs)
    labels = _deterministic_two_cluster(signatures)
    if min(np.sum(labels == 0), np.sum(labels == 1)) < 2:
        return WindowScore(float("-inf"), k1_loglik, float("-inf"), float("inf"), labels, int(values.size))
    k2_prediction = np.empty((len(times_hold), 2, modalities), dtype=float)
    for group in (0, 1):
        for modality in range(modalities):
            path = np.median(values[:fit_epochs, labels == group, modality], axis=1)
            coefficients, _ = _line_fit(times_fit, path[:, None])
            k2_prediction[:, group, modality] = np.column_stack((np.ones(len(times_hold)), times_hold)) @ coefficients[:, 0]
    predicted = np.stack([k2_prediction[:, labels[prn], :] for prn in range(values.shape[1])], axis=1)
    k2_residual = values[fit_epochs:] - predicted
    k2_loglik = student_t_loglik(k2_residual, shared_scale)
    valid_count = int(np.sum(mask[fit_epochs:]))
    delta_parameters = 2 * modalities + max(0, values.shape[1] - 1)
    penalty = 0.5 * delta_parameters * math.log(valid_count)
    return WindowScore(k2_loglik - k1_loglik - penalty, k1_loglik, k2_loglik, penalty, labels, valid_count)


def inject_secondary_clock(
    panel: np.ndarray,
    subset: np.ndarray,
    *,
    d0_m: float,
    velocity_mps: float,
    acceleration_mps2: float,
    epoch_rate_hz: float = EPOCH_RATE_HZ,
) -> np.ndarray:
    """Coherently perturb range, range-rate, and carrier-increment modalities."""

    result = np.asarray(panel, dtype=float).copy()
    if result.shape[2] < 3:
        raise ValueError("three modalities are required")
    time = np.arange(result.shape[0], dtype=float) / epoch_rate_hz
    delta_range = d0_m + velocity_mps * time + 0.5 * acceleration_mps2 * time ** 2
    delta_rate = velocity_mps + acceleration_mps2 * time
    result[:, subset, 0] += delta_range[:, None]
    result[:, subset, 1] += delta_rate[:, None]
    result[:, subset, 2] += delta_rate[:, None] / epoch_rate_hz
    return result


def manifest(artifact: Path) -> dict:
    excluded = {"artifact_manifest_sha256.json", "test_output.txt", "verifier_output.txt"}
    value = {
        "algorithm": "sha256",
        "excluded_self_and_mutable_logs": sorted(excluded),
        "files": {
            path.relative_to(artifact).as_posix(): sha256_file(path)
            for path in sorted(artifact.rglob("*"))
            if path.is_file() and path.name not in excluded
        },
    }
    write_json(artifact / "artifact_manifest_sha256.json", value)
    return value
