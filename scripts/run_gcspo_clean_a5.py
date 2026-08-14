#!/usr/bin/env python3
"""Run the frozen clean-only A5 independent-input ablation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.gcspo_a5 import a5_spectral_scores, role_a5_terms
from gnss_doppler_lab.gcspo_artifacts import canonical_write_json
from gnss_doppler_lab.gcspo_clean import run_clean_a1
from gnss_doppler_lab.gcspo_core import empirical_threshold
from gnss_doppler_lab.gcspo_full import _score_terms
from gnss_doppler_lab.gcspo_provenance import canonical_json_bytes

_ROWS = None
_BACKEND = None


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _identity(path):
    payload = Path(path).read_bytes()
    if not payload:
        raise ValueError(f"execution receipt refuses empty file: {path}")
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _write_execution_receipt(path, document):
    path = Path(path)
    payload = canonical_json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _process_identity():
    tail = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": os.getpid(), "proc_start_ticks": int(tail[19])}


def _source_commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          text=True, capture_output=True).stdout.strip()




def _grid_objective(index_and_grid):
    index, grid = index_and_grid; row = _ROWS[index]
    return [score["gcv"] for score in a5_spectral_scores(row["terms"], row["state_segments"], grid, backend=_BACKEND)]


def _score_index(index_and_smoothness):
    index, smoothness = index_and_smoothness; row = _ROWS[index]
    score = a5_spectral_scores(row["terms"], row["state_segments"], (smoothness,), backend=_BACKEND)[0]
    score["state"] = score["state"].tolist()
    return {key: row[key] for key in ("window_start_s", "availability_s", "prns", "epoch_ids", "epoch_prn_support")} | score


def _pool(processes):
    return mp.get_context("fork").Pool(processes=processes)


def _choose(objectives):
    best = objectives[0]
    for candidate in objectives[1:]:
        scale = max(abs(candidate["mean_gcv"]), abs(best["mean_gcv"]), 1)
        if candidate["mean_gcv"] < best["mean_gcv"] - 1e-12 * scale or (abs(candidate["mean_gcv"] - best["mean_gcv"]) <= 1e-12 * scale and candidate["lambda"] > best["lambda"]): best = candidate
    return best["lambda"]


def score_parallel(rows, smoothness, workers):
    global _ROWS; _ROWS = rows
    if workers == 1:
        return [_score_index((index, smoothness)) for index in range(len(rows))]
    with _pool(workers) as pool: return pool.map(_score_index, [(index, smoothness) for index in range(len(rows))], chunksize=1)


def _backend_truth(requested):
    import torch

    available = bool(torch.cuda.is_available())
    resolved = "cuda" if requested == "auto" and available else ("cpu" if requested == "auto" else requested)
    if resolved == "cuda" and not available:
        raise RuntimeError("A5 CUDA backend requested but CUDA is unavailable before computation")
    if resolved == "cuda":
        torch.cuda.init()
        device = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
    return {
        "requested": requested, "resolved": resolved,
        "cuda_available": available, "device": device,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/gcspo_stage0_static_rerun")
    parser.add_argument("--clean-root", type=Path, default=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/receiver/cleanStatic-complex9"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--numeric-trace", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--nonce")
    parser.add_argument("--execution-receipt", type=Path)
    parser.add_argument("--challenge-file", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4: raise ValueError("workers must be in [1,4]")
    witness_values = (args.run_id, args.nonce, args.execution_receipt, args.challenge_file)
    witnessed = any(value is not None for value in witness_values)
    if witnessed and not all(value is not None for value in witness_values):
        raise ValueError("witnessed A5 requires run ID, nonce, receipt, and challenge")
    if witnessed and (not args.numeric_trace or not args.run_id or
                      not isinstance(args.nonce, str) or len(args.nonce) != 64):
        raise ValueError("witnessed A5 run identity/nonce/output contract is invalid")
    child_started_utc = _utc_now()
    process_identity = _process_identity()
    actual_argv = [sys.executable, *sys.argv]
    transcript_events = []
    backend_truth = _backend_truth(args.backend)
    backend_line = f"A5_BACKEND_PASS requested={args.backend} resolved={backend_truth['resolved']} device={backend_truth['device']}"
    transcript_events.append(backend_line); print(backend_line, flush=True)
    config = json.loads((args.artifact_dir / "config.json").read_text())
    clean = run_clean_a1(args.clean_root, ridge_grid=config["h0_predictor"]["ridge_grid"])
    validated = {"code_error_chips", "pll_phase_error_cycles", "carrier_doppler_hz", "code_frequency_offset_chips_s"}
    global _ROWS, _BACKEND
    _BACKEND = args.backend
    _ROWS = role_a5_terms(clean["data"], clean["model"], clean["whitener"], clean["gamma"], validated, 150, 210)
    if len(_ROWS) < 100: raise ValueError("A5 has fewer than 100 common validation windows")
    validation_windows = len(_ROWS)
    grid = list(map(float, config["lambda_selection"]["grid"]))
    if args.workers == 1:
        per_window = [_grid_objective((index, grid)) for index in range(len(_ROWS))]
    else:
        with _pool(args.workers) as pool:
            per_window = pool.map(_grid_objective, [(index, grid) for index in range(len(_ROWS))], chunksize=1)
    objectives = [{"lambda": value, "mean_gcv": float(np.mean([row[index] for row in per_window]))}
                  for index, value in enumerate(grid)]
    selected = _choose(objectives)
    lambda_line = f"A5_LAMBDA_PASS windows={len(_ROWS)} lambda={selected}"
    transcript_events.append(lambda_line); print(lambda_line, flush=True)
    _ROWS = role_a5_terms(clean["data"], clean["model"], clean["whitener"], clean["gamma"], validated, 220, 340)
    calibration_trace = score_parallel(_ROWS, selected, args.workers); del _ROWS
    calibration = [{key: value for key, value in row.items() if key != "state"} for row in calibration_trace]
    calibration_line = f"A5_CALIBRATION_PASS windows={len(calibration)}"
    transcript_events.append(calibration_line); print(calibration_line, flush=True)
    _ROWS = role_a5_terms(clean["data"], clean["model"], clean["whitener"], clean["gamma"], validated, 350, 470)
    holdout_trace = score_parallel(_ROWS, selected, args.workers); del _ROWS
    holdout = [{key: value for key, value in row.items() if key != "state"} for row in holdout_trace]
    values = np.asarray([row["score"] for row in calibration])
    thresholds = {"q99": empirical_threshold(values, .99), "q995": empirical_threshold(values, .995)}
    report = {"schema": "gnss-doppler-lab.gcspo-stage0.clean-a5-report.v1", "run_status": "CLEAN_A5_PASS",
              "protected_attack_rows_read": False, "attack_access_count": 0, "validated_rows": sorted(validated),
              "lambda": selected, "lambda_objectives": objectives, "lambda_validation_windows": validation_windows,
              "thresholds": thresholds, "calibration": calibration, "holdout": holdout}
    canonical_write_json(args.artifact_dir / "clean_a5_report.json", report)
    threshold_path = args.artifact_dir / "thresholds.json"; payload = json.loads(threshold_path.read_text()); payload["A5"] = thresholds
    canonical_write_json(threshold_path, payload)
    clean_line = f"A5_CLEAN_PASS calibration={len(calibration)} holdout={len(holdout)}"
    transcript_events.append(clean_line); print(clean_line, flush=True)
    if args.numeric_trace:
        canonical_write_json(args.artifact_dir / "a5_numeric_trace.json", {
            "schema": "gnss-doppler-lab.gcspo-stage0.a5-numeric-trace.v1",
            "backend": backend_truth["resolved"], "lambda": selected, "lambda_objectives": objectives,
            "thresholds": thresholds, "calibration": calibration_trace, "holdout": holdout_trace,
        })
        backend_document = {
            "schema": "gnss-doppler-lab.gcspo-stage0.a5-backend-truth.v1",
            **backend_truth,
        }
        canonical_write_json(args.artifact_dir / "a5_backend_truth.json", backend_document)
    if witnessed:
        challenge_identity = _identity(args.challenge_file)
        output_names = ("clean_a5_report.json", "thresholds.json",
                        "a5_numeric_trace.json", "a5_backend_truth.json")
        scientific_outputs = {name: _identity(args.artifact_dir / name)
                              for name in output_names}
        child_finished_utc = _utc_now()
        transcript_bytes = ("\n".join(transcript_events) + "\n").encode()
        _write_execution_receipt(args.execution_receipt, {
            "schema": "gnss-doppler-lab.gcspo-stage0.a5-execution-receipt.v1",
            "run_id": args.run_id, "nonce": args.nonce,
            "process_identity": process_identity,
            "child_started_utc": child_started_utc,
            "child_finished_utc": child_finished_utc,
            "backend_truth": backend_document,
            "source_commit": _source_commit(), "challenge": challenge_identity,
            "argv": actual_argv, "scientific_outputs": scientific_outputs,
            "transcript_state": {"event_count": len(transcript_events),
                                 "sha256": hashlib.sha256(transcript_bytes).hexdigest()},
        })
    return 0


if __name__ == "__main__": raise SystemExit(main())
