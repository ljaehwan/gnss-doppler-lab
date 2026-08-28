#!/usr/bin/env python3
"""Freeze a signal-normalized Galileo CGC model from TUNI clean C-1 only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TAP_NAMES = ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")
PROMPT_INDEX = 4
BIN_SECONDS = 0.25
MIN_EPOCHS_PER_BIN = 20
STABILIZATION_SECONDS = 1.0
CALIBRATION_END_SECONDS = 5.0
VALIDATION_END_SECONDS = 10.0
SCALE_FLOOR = 0.02
DISTORTION_THRESHOLD = 1.5
COHERENCE_THRESHOLD = 0.8
PERSISTENCE_BINS = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_shape(tap_i: np.ndarray, tap_q: np.ndarray) -> np.ndarray:
    """Remove prompt amplitude/phase and return 16 real non-prompt features."""
    z = np.asarray(tap_i, dtype=np.float64) + 1j * np.asarray(tap_q, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != len(TAP_NAMES):
        raise ValueError("complex tap matrix must have nine columns")
    prompt = z[:, PROMPT_INDEX]
    if np.any(np.abs(prompt) <= np.finfo(np.float64).eps):
        raise ValueError("prompt contains a zero complex sample")
    profile = z / prompt[:, None]
    nonprompt = np.column_stack((profile[:, :PROMPT_INDEX], profile[:, PROMPT_INDEX + 1 :]))
    return np.column_stack((nonprompt.real, nonprompt.imag))


def _vector(handle: h5py.File, name: str) -> np.ndarray:
    if name not in handle:
        raise ValueError(f"tracking MAT is missing {name}")
    return np.asarray(handle[name]).reshape(-1)


def load_binned_shapes(raw_dir: Path, sample_rate_hz: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("epl_tracking_ch_*.mat")):
        with h5py.File(path, "r") as handle:
            prn_values = _vector(handle, "PRN")
            if prn_values.shape == (2,) and np.array_equal(prn_values, np.array([1, 0])):
                continue
            if not prn_values.size or len(set(map(int, prn_values))) != 1:
                raise ValueError(f"expected one nonempty PRN per clean channel: {path}")
            prn = int(prn_values[0])
            time_s = _vector(handle, "PRN_start_sample_count").astype(np.float64) / sample_rate_hz
            tap_i = np.column_stack([_vector(handle, f"tap_I_{name}") for name in TAP_NAMES])
            tap_q = np.column_stack([_vector(handle, f"tap_Q_{name}") for name in TAP_NAMES])
            feature = normalized_shape(tap_i, tap_q)
            bin_index = np.floor(time_s / BIN_SECONDS).astype(np.int64)
            for index in sorted(set(bin_index)):
                selected = bin_index == index
                if int(selected.sum()) < MIN_EPOCHS_PER_BIN:
                    continue
                rows.append(
                    {
                        "time_s": float(index * BIN_SECONDS),
                        "prn": prn,
                        "feature": np.median(feature[selected], axis=0),
                        "epochs": int(selected.sum()),
                        "mat": path.name,
                    }
                )
    if not rows:
        raise ValueError("clean receiver run supplied no eligible nine-tap bins")
    return rows


def pair_cosines(times: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for time_s in sorted(set(map(float, times))):
        indices = np.flatnonzero(times == time_s)
        for left_pos, left in enumerate(indices):
            for right in indices[left_pos + 1 :]:
                a, b = residuals[left], residuals[right]
                denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
                if denominator > 0:
                    values.append(float(np.dot(a, b) / denominator))
    return np.asarray(values, dtype=np.float64)


def quantiles(values: np.ndarray) -> dict[str, float]:
    levels = (0.5, 0.9, 0.95, 0.99, 1.0)
    result = np.quantile(values, levels)
    return {f"q{int(level * 100):02d}": float(value) for level, value in zip(levels, result)}


def freeze_model(clean_run: Path, sample_rate_hz: int) -> dict[str, Any]:
    summary_path = clean_run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("scope") != "clean-only receiver compatibility; no attack payload accessed":
        raise ValueError("source receiver run is not marked clean-only")
    if summary.get("scenario") != "C-1" or summary.get("tracking", {}).get("tap_count") != 9:
        raise ValueError("model source must be the nine-tap TUNI Galileo C-1 run")
    rows = load_binned_shapes(clean_run / "raw", sample_rate_hz)
    times = np.asarray([row["time_s"] for row in rows], dtype=np.float64)
    features = np.stack([row["feature"] for row in rows])
    calibration = (times >= STABILIZATION_SECONDS) & (times < CALIBRATION_END_SECONDS)
    validation = (times >= CALIBRATION_END_SECONDS) & (times < VALIDATION_END_SECONDS)
    if int(calibration.sum()) < 30 or int(validation.sum()) < 30:
        raise ValueError("clean time split has insufficient eligible bins")
    center = np.median(features[calibration], axis=0)
    mad = np.median(np.abs(features[calibration] - center), axis=0)
    scale = np.maximum(1.4826 * mad, SCALE_FLOOR)
    residuals = (features - center) / scale
    distortion = np.linalg.norm(residuals, axis=1) / np.sqrt(residuals.shape[1])
    calibration_cosine = pair_cosines(times[calibration], residuals[calibration])
    validation_cosine = pair_cosines(times[validation], residuals[validation])
    validation_pass = bool(
        float(np.max(distortion[validation])) < DISTORTION_THRESHOLD
        and float(np.max(validation_cosine)) < COHERENCE_THRESHOLD
    )
    if not validation_pass:
        raise RuntimeError("clean-only model failed the fixed validation margin")
    mat_files = sorted((clean_run / "raw").glob("epl_tracking_ch_*.mat"))
    return {
        "schema": "gnss-doppler-lab.tuni-galileo-cgc-clean-model.v1",
        "training_boundary": "TUNI C-1 only; no attack payload accessed",
        "source": {
            "receiver_run": str(clean_run),
            "receiver_summary_sha256": sha256(summary_path),
            "receiver_executable_sha256": summary["receiver"]["executable_sha256"],
            "receiver_config_sha256": summary["receiver"]["config_sha256"],
            "tracking_mat_sha256": {path.name: sha256(path) for path in mat_files},
            "prns": sorted({int(row["prn"]) for row in rows}),
        },
        "preprocessing": {
            "tap_names": list(TAP_NAMES),
            "prompt_index": PROMPT_INDEX,
            "feature_order": [
                *(f"real_{name}" for name in (*TAP_NAMES[:PROMPT_INDEX], *TAP_NAMES[PROMPT_INDEX + 1 :])),
                *(f"imag_{name}" for name in (*TAP_NAMES[:PROMPT_INDEX], *TAP_NAMES[PROMPT_INDEX + 1 :])),
            ],
            "bin_seconds": BIN_SECONDS,
            "minimum_epochs_per_bin": MIN_EPOCHS_PER_BIN,
            "internal_sample_rate_hz": sample_rate_hz,
            "stabilization_seconds": STABILIZATION_SECONDS,
            "calibration_interval_s": [STABILIZATION_SECONDS, CALIBRATION_END_SECONDS],
            "validation_interval_s": [CALIBRATION_END_SECONDS, VALIDATION_END_SECONDS],
            "scale_floor": SCALE_FLOOR,
        },
        "model": {"center": center.tolist(), "robust_scale": scale.tolist()},
        "detector": {
            "distortion_z_rms_threshold": DISTORTION_THRESHOLD,
            "residual_cosine_threshold": COHERENCE_THRESHOLD,
            "persistence_bins": PERSISTENCE_BINS,
            "rule": "two PRNs must both exceed distortion and pair cosine thresholds for four consecutive common bins",
        },
        "clean_validation": {
            "eligible_bin_count": len(rows),
            "calibration_bin_count": int(calibration.sum()),
            "validation_bin_count": int(validation.sum()),
            "distortion_calibration": quantiles(distortion[calibration]),
            "distortion_validation": quantiles(distortion[validation]),
            "cosine_calibration": quantiles(calibration_cosine),
            "cosine_validation": quantiles(validation_cosine),
            "fixed_margin_pass": validation_pass,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean-run", type=Path, required=True,
        help="Path containing clean-only summary.json and raw nine-tap MAT files",
    )
    parser.add_argument("--sample-rate-hz", type=int, default=12_500_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = freeze_model(args.clean_run.resolve(), args.sample_rate_hz)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256(output), **model["clean_validation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

