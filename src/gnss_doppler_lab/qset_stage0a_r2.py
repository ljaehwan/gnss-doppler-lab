"""Frozen deterministic Q-SET-GNSS Stage-0A R2 execution library."""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .trace_native_1ms import TAPS, complex_taps, read_records, sha256_file

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/qset_gnss_stage0a_r2_galileo_partial_prn_execution"
SSD_ROOT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/qset-gnss-stage0a-r2-galileo-partial-prn-execution")
DATA_ROOT = Path("/home/ubuntu/ssd_data/gnss-datasets/tuni2025/galileo")
R2C_SOURCE = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2c-terminal-drain-repair/receiver-source")
RECEIVER_SOURCE = SSD_ROOT / "receiver-source"
RECEIVER_BUILD = SSD_ROOT / "receiver-build"
RECEIVER = RECEIVER_BUILD / "src/main/gnss-sdr"
CONFIG_TEMPLATE = ROOT / "configs/qset_galileo_e1_trace9.conf.template"
GALILEO_PATCH = ROOT / "patches/qset_galileo_e1_native_trace.patch"
R2C_PATCH = ROOT / "artifacts/trace_stage0_r2c_terminal_drain_repair/receiver_repair.diff"
BASE_RECEIVER_COMMIT = "1ddd4562723040fd66cb334b578a5b69455625f4"
R2C_PATCH_SHA256 = "81b680bfae682fa92743fca53ce30ae09790383d008841d6eae3a25407442540"
CONFIG_SHA256 = "b37da63cdbf9e386d5a5dec3eb5ae7dfcd35d07e9e9785a56e5900f85bd3026d"
PREREGISTRATION_SHA = "0a964599205db3a8fdf50d553261b5ce6fd02c1d"
RAW_FS = 50_000_000
OUTPUT_FS = 4_000_000
BYTES_PER_COMPLEX = 4
RESAMPLER_RATIO = 25.0 / 2.0
RESAMPLER_GROUP_DELAY_RAW_SAMPLES = 205.25
MIN_EPOCHS_PER_WINDOW = 125
MIN_EVENT_PRNS = 5
AGGREGATORS = ("MEAN", "MAX", "Q50", "Q70", "Q90", "MULTI_Q")
QUANTILES = (0.5, 0.7, 0.9, 1.0)
TARGET_FPR = 0.01
TRACE_CHANNELS = 12

SCENARIOS: dict[str, dict[str, Any]] = {
    "C-1": {"filename": "clearsky_signal_C-1.bin", "size": 29_999_832_000, "md5": "4ff0e86938792bf3150c30d5f1481917", "clean": True, "spoofed": []},
    "C-3": {"filename": "clearsky_signal_C-3.bin", "size": 29_999_832_000, "md5": "1b7c99c754faec3c8fa625849ef70014", "clean": True, "spoofed": []},
    "SS-1": {"filename": "Galileo_1_Spoofer_Static_NoMP_TruePosition.bin", "size": 29_999_832_000, "md5": "f2784d9f7d3c85edffeaf30bbd7efdb2", "clean": False, "spoofed": [9]},
    "SS-3": {"filename": "Galileo_3_Spoofer_Static_NoMP_TruePosition.bin", "size": 21_925_920_000, "md5": "cdee5a9b6a6509af016bdb3776ebd231", "clean": False, "spoofed": [6, 9, 23]},
    "SS-5": {"filename": "Galileo_5_Spoofer_Static_NoMP_TruePosition.bin", "size": 29_999_832_000, "md5": "8a8e2ffc0bc5420b9df07b4faddb35bc", "clean": False, "spoofed": [4, 6, 9, 23, 31]},
    "SS-11": {"filename": "Galileo_1_Spoofer_Static_MP_TruePosition.bin", "size": 29_999_832_000, "md5": "71de773dce5684c98daed0a48496891a", "clean": False, "spoofed": [31], "multipath": True},
}


class QSetError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QSetError(message)


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def md5_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str, cwd: Path = ROOT) -> str:
    run = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True)
    require(run.returncode == 0, f"git {' '.join(args)}: {run.stderr.strip()}")
    return run.stdout.strip()


def scenario_path(name: str, *, allow_search: bool = False) -> Path:
    spec = SCENARIOS[name]
    direct = DATA_ROOT / name / spec["filename"]
    if direct.is_file():
        return direct
    require(allow_search, f"expected scenario path absent: {direct}")
    matches = list(DATA_ROOT.glob(f"*/{spec['filename']}"))
    require(len(matches) == 1, f"allowlisted scenario resolution count {name}: {len(matches)}")
    return matches[0]


def load_r1() -> Any:
    path = ROOT / "scripts/run_qset_gnss_stage0a_r1.py"
    spec = importlib.util.spec_from_file_location("qset_r1_decoder", path)
    require(spec is not None and spec.loader is not None, "cannot load R1 decoder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def feature_names() -> list[str]:
    names: list[str] = []
    for part in ("median_re", "median_im", "residual_mad_re", "residual_mad_im"):
        names.extend(f"{part}_{tap}" for tap in TAPS)
    for part in ("antisym_re", "antisym_im"):
        names.extend(f"{part}_{index}" for index in range(4))
    for part in ("curvature_re", "curvature_im"):
        names.extend(f"{part}_{index}" for index in range(1, 8))
    return names


FEATURE_NAMES = feature_names()


def normalized_complex_taps(taps: np.ndarray, epsilon: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    taps = np.asarray(taps, dtype=np.complex128)
    require(taps.ndim == 2 and taps.shape[1] == 9, "nine complex taps required")
    prompt = taps[:, 4]
    denominator = np.abs(prompt) ** 2
    valid = np.isfinite(taps.real).all(axis=1) & np.isfinite(taps.imag).all(axis=1) & (denominator > epsilon)
    normalized = np.full_like(taps, np.nan + 1j * np.nan)
    normalized[valid] = taps[valid] * np.conj(prompt[valid, None]) / denominator[valid, None]
    return normalized, valid


def morphology_feature(taps: np.ndarray) -> np.ndarray:
    normalized, valid = normalized_complex_taps(taps)
    normalized = normalized[valid]
    require(len(normalized) >= MIN_EPOCHS_PER_WINDOW, "insufficient native epochs in PRN window")
    med_re = np.median(normalized.real, axis=0)
    med_im = np.median(normalized.imag, axis=0)
    mad_re = np.median(np.abs(normalized.real - med_re), axis=0)
    mad_im = np.median(np.abs(normalized.imag - med_im), axis=0)
    anti = np.median(normalized[:, :4] - normalized[:, :4:-1], axis=0)
    curve = np.median(normalized[:, :-2] - 2.0 * normalized[:, 1:-1] + normalized[:, 2:], axis=0)
    result = np.concatenate((med_re, med_im, mad_re, mad_im, anti.real, anti.imag, curve.real, curve.imag))
    require(len(result) == len(FEATURE_NAMES) and np.isfinite(result).all(), "nonfinite morphology feature")
    return result


def fit_robust_model(features: np.ndarray) -> dict[str, Any]:
    values = np.asarray(features, dtype=float)
    require(values.ndim == 2 and values.shape[1] == len(FEATURE_NAMES) and len(values), "invalid development feature matrix")
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-6)
    return {"feature_names": FEATURE_NAMES, "median": median.tolist(), "scale": scale.tolist(), "development_rows": int(len(values))}


def local_scores(features: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    median = np.asarray(model["median"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    z = np.clip((values - median) / scale, -8.0, 8.0)
    return np.sqrt(np.mean(z * z, axis=1))


def fit_multi_q_reference(score_sets: Iterable[np.ndarray]) -> dict[str, list[float]]:
    matrix = np.asarray([[float(np.quantile(scores, q)) for q in QUANTILES] for scores in score_sets], dtype=float)
    require(len(matrix), "empty calibration quantile matrix")
    median = np.median(matrix, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(matrix - median), axis=0), 1e-9)
    return {"median": median.tolist(), "scale": scale.tolist()}


def aggregate_scores(scores: np.ndarray, multi_q_reference: dict[str, Any]) -> dict[str, float]:
    values = np.asarray(scores, dtype=float)
    require(len(values) >= MIN_EVENT_PRNS and np.isfinite(values).all(), "invalid dynamic PRN panel")
    quantile_values = np.asarray([np.quantile(values, q) for q in QUANTILES])
    ref_median = np.asarray(multi_q_reference["median"], dtype=float)
    ref_scale = np.asarray(multi_q_reference["scale"], dtype=float)
    return {
        "MEAN": float(np.mean(values)), "MAX": float(np.max(values)),
        "Q50": float(quantile_values[0]), "Q70": float(quantile_values[1]), "Q90": float(quantile_values[2]),
        "MULTI_Q": float(np.max((quantile_values - ref_median) / ref_scale)),
    }


def persistence(values: Iterable[float], ends: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    values_array = np.asarray(list(values), dtype=float)
    ends_array = np.asarray(list(ends), dtype=int)
    result = np.full(len(values_array), np.nan)
    warm = np.ones(len(values_array), dtype=bool)
    for index in range(2, len(values_array)):
        if ends_array[index - 2] + 1 == ends_array[index - 1] and ends_array[index - 1] + 1 == ends_array[index]:
            result[index] = float(np.partition(values_array[index - 2:index + 1], 1)[1])
            warm[index] = False
    return result, warm


def calibrate_threshold(values: Iterable[float], ends: Iterable[int], target_fpr: float = TARGET_FPR) -> dict[str, Any]:
    values_array = np.asarray(list(values), dtype=float)
    ends_array = np.asarray(list(ends), dtype=int)
    candidates = np.unique(values_array[np.isfinite(values_array)])
    require(len(candidates), "no calibration statistics")
    selected = float(candidates[-1])
    selected_fpr = 0.0
    for candidate in candidates:
        above = values_array > candidate
        persistent = np.zeros(len(above), dtype=bool)
        for index in range(2, len(above)):
            if ends_array[index - 2] + 1 == ends_array[index - 1] and ends_array[index - 1] + 1 == ends_array[index]:
                persistent[index] = bool(np.sum(above[index - 2:index + 1]) >= 2)
        denominator = max(1, len(above) - 2)
        fpr = float(np.sum(persistent) / denominator)
        if fpr <= target_fpr:
            selected, selected_fpr = float(candidate), fpr
            break
    return {"threshold": selected, "calibration_fpr": selected_fpr, "candidate_count": int(len(candidates)), "comparison": ">"}


def wilson_upper(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 1.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return float((center + radius) / denominator)


def binary_auc(negative: Iterable[float], positive: Iterable[float]) -> float:
    neg = np.asarray(list(negative), dtype=float); pos = np.asarray(list(positive), dtype=float)
    require(len(neg) and len(pos), "AUC needs both classes")
    comparisons = pos[:, None] - neg[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def normalized_pauc(negative: Iterable[float], positive: Iterable[float], max_fpr: float = 0.01) -> float:
    neg = np.asarray(list(negative), dtype=float); pos = np.asarray(list(positive), dtype=float)
    require(len(neg) and len(pos), "pAUC needs both classes")
    thresholds = np.r_[np.inf, np.sort(np.unique(np.r_[neg, pos]))[::-1], -np.inf]
    fpr = np.asarray([np.mean(neg > threshold) for threshold in thresholds])
    tpr = np.asarray([np.mean(pos > threshold) for threshold in thresholds])
    order = np.argsort(fpr, kind="stable"); fpr, tpr = fpr[order], tpr[order]
    keep = fpr <= max_fpr
    x, y = list(fpr[keep]), list(tpr[keep])
    above = np.flatnonzero(fpr > max_fpr)
    if above.size:
        j = int(above[0]); i = max(0, j - 1)
        if fpr[j] != fpr[i]:
            boundary = tpr[i] + (tpr[j] - tpr[i]) * (max_fpr - fpr[i]) / (fpr[j] - fpr[i])
        else:
            boundary = max(tpr[i], tpr[j])
        x.append(max_fpr); y.append(float(boundary))
    elif not x or x[-1] < max_fpr:
        x.append(max_fpr); y.append(float(tpr[-1]))
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(np.asarray(y), np.asarray(x)) / max_fpr)


def validate_galileo_trace(dump_paths: Iterable[Path], scenario: str) -> dict[str, Any]:
    paths = sorted(dump_paths)
    require(paths, f"no TRACE dumps for {scenario}")
    require(len(paths) == TRACE_CHANNELS, f"TRACE dump count {len(paths)} != {TRACE_CHANNELS} for {scenario}")
    summaries: list[dict[str, Any]] = []
    prns: set[int] = set(); finite_failures = 0; cadence_failures = 0; causal_failures = 0
    for path in paths:
        if path.stat().st_size == 0:
            summaries.append({"path": str(path), "record_count": 0, "size_bytes": 0, "sha256": sha256_file(path), "status": "EMPTY_OPTIONAL_CHANNEL"})
            continue
        header, records = read_records(path)
        require(header.scenario_id == scenario, "TRACE scenario mismatch")
        require(abs(header.sample_rate_hz - OUTPUT_FS) < 1e-6, "TRACE sample rate mismatch")
        require(tuple(round(x, 6) for x in header.tap_offsets_chips) == (-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5), "TRACE tap offsets mismatch")
        if not len(records):
            summaries.append({"path": str(path), "record_count": 0, "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "status": "EMPTY_OPTIONAL_CHANNEL"})
            continue
        taps = complex_taps(records)
        finite_failures += int((~np.isfinite(taps.real).all(axis=1) | ~np.isfinite(taps.imag).all(axis=1)).sum())
        finite_failures += int(np.all(taps == 0, axis=1).sum())
        prns.update(int(x) for x in records["prn"] if 1 <= int(x) <= 36)
        same = (records["tracking_session_id"][1:] == records["tracking_session_id"][:-1]) & (records["prn"][1:] == records["prn"][:-1])
        indices = np.flatnonzero(same) + 1
        if len(indices):
            prev = indices - 1
            starts = records["raw_interval_start_sample"]
            dt = (starts[indices] - starts[prev]).astype(float) / OUTPUT_FS
            cadence_failures += int(np.sum((dt < 0.0035) | (dt > 0.0045)))
            causal_failures += int(np.sum(records["loop_sequence"][indices] != records["loop_sequence"][prev] + 1))
            causal_failures += int(np.sum(records["action_used_source_loop_sequence"][indices] != records["loop_sequence"][prev]))
        summaries.append({"path": str(path), "record_count": int(len(records)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "prns": sorted(set(map(int, records["prn"]))), "status": "PASS"})
    status = "PASS" if not finite_failures and not cadence_failures and not causal_failures and len(prns) >= 4 else "FAIL"
    return {"schema": "gnss-doppler-lab.qset-galileo-native-trace.v1", "status": status, "scenario": scenario, "files": summaries, "tracked_prns": sorted(prns), "tracked_prn_count": len(prns), "finite_failures": finite_failures, "cadence_failures": cadence_failures, "causal_failures": causal_failures, "native_integration_s": 0.004}


def extract_window_features(dump_dir: Path, scenario: str, raw_duration_s: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    audit: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: {"cn0": [], "lock": []})
    maximum_end = int(math.floor(raw_duration_s))
    for path in sorted(dump_dir.glob("trace_native_1ms_ch_*.bin")):
        if path.stat().st_size == 0:
            continue
        _, records = read_records(path)
        if not len(records):
            continue
        taps = complex_taps(records)
        _, prompt_valid = normalized_complex_taps(taps)
        valid = (records["valid_tracking"] == 1) & (records["valid_lock"] == 1) & prompt_valid
        raw_time = (records["raw_interval_start_sample"].astype(float) * RESAMPLER_RATIO - RESAMPLER_GROUP_DELAY_RAW_SAMPLES) / RAW_FS
        ends = np.floor(raw_time).astype(int) + 1
        for index in np.flatnonzero(valid & (ends >= 1) & (ends <= maximum_end)):
            key = (int(records["prn"][index]), int(ends[index]))
            grouped[key].append(taps[index])
            audit[key]["cn0"].append(float(records["cn0_db_hz"][index]))
            audit[key]["lock"].append(float(records["carrier_lock_test"][index]))
    rows: list[dict[str, Any]] = []
    for (prn, end), values in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        if len(values) < MIN_EPOCHS_PER_WINDOW:
            continue
        feature = morphology_feature(np.asarray(values))
        rows.append({"scenario": scenario, "prn": prn, "window_start_s": end - 1, "window_end_s": end, "epoch_count": len(values), "feature": feature, "cn0_median": float(np.median(audit[(prn, end)]["cn0"])), "lock_median": float(np.median(audit[(prn, end)]["lock"]))})
    return rows


def dynamic_windows(rows: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    if not rows:
        return []
    features = np.asarray([row["feature"] for row in rows])
    scores = local_scores(features, model)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        grouped[int(row["window_end_s"])].append({**row, "local_score": float(score)})
    result = []
    for end, per_prn in sorted(grouped.items()):
        unique = {int(row["prn"]): row for row in per_prn}
        if len(unique) < MIN_EVENT_PRNS:
            continue
        ordered = [unique[key] for key in sorted(unique)]
        result.append({"window_end_s": end, "window_start_s": end - 1, "prns": [row["prn"] for row in ordered], "scores": np.asarray([row["local_score"] for row in ordered]), "cn0": np.asarray([row["cn0_median"] for row in ordered]), "lock": np.asarray([row["lock_median"] for row in ordered])})
    return result

