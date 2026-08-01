#!/usr/bin/env python3
"""CUDA-first r3 score-reproduction diagnosis; never emits corrected diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from train_gcmr_peak_innovation import peak_indexes, records
from run_gcmr_oakbat_poc import SCENARIOS as SOURCE_SCENARIOS, load_scenario
from gnss_doppler_lab.gcmr_pi_r4_corrected import reconstruct_event_innovation, rescore_from_innovations
from gnss_doppler_lab.gcmr_pi_r4_reproduction import component_agreement

FIELDS = ("S_common", "N_eff", "S_pair", "energy", "Full")
SCENARIOS = ("os1", "os2", "os3", "os4")


def configure(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_pipe(path: Path, device: torch.device, seed: int):
    configure(seed, device)
    pipeline = torch.load(path, map_location=device, weights_only=False)
    pipeline.device = device
    pipeline.network.to(device).eval()
    return pipeline


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def runtime() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def rescore_event(pipeline, event):
    """Single event score path with the immutable residual/z order made explicit."""
    residual, z = reconstruct_event_innovation(pipeline, event)
    diagnostics, scores = rescore_from_innovations(pipeline, event, z, residual)
    return diagnostics, scores


def compare(pipeline, frozen: Path, scenario: str) -> dict[str, dict[str, object]]:
    events, _, _ = load_scenario(scenario, frozen / "cache", False)
    built, _ = records(
        events,
        SOURCE_SCENARIOS[scenario],
        pipeline.window,
        peak_indexes(events, SOURCE_SCENARIOS[scenario], pipeline.feature_dim),
    )
    reference = rows(frozen / f"{scenario}_scores.csv")
    if len(built) != len(reference):
        raise RuntimeError(f"{scenario}: record count mismatch {len(built)} != {len(reference)}")

    times: list[float] = []
    actual = {field: [] for field in FIELDS}
    expected = {field: [] for field in FIELDS}
    for event, row in zip(built, reference, strict=True):
        diagnostics, scores = rescore_event(pipeline, event)
        values = {
            "S_common": diagnostics.s_common,
            "N_eff": diagnostics.n_eff,
            "S_pair": diagnostics.s_pair,
            "energy": diagnostics.energy,
            "Full": scores["Full"],
        }
        times.append(float(event.time))
        for field in FIELDS:
            actual[field].append(float(values[field]))
            expected[field].append(float(row[field]))

    thresholds = json.loads((frozen / "thresholds.json").read_text())
    full_threshold = thresholds.get("Full", {}).get("q99")
    return {
        field: component_agreement(
            expected[field],
            actual[field],
            threshold=float(full_threshold) if field == "Full" and full_threshold is not None else None,
            times=times,
        )
        for field in FIELDS
    }


def reproduction_pass(cuda_results: dict[str, dict[str, dict[str, object]]], atol: float) -> bool:
    """Score identity needs every frozen component within tolerance and identical Full-q99 alarms."""
    for scenario in SCENARIOS:
        for field in FIELDS:
            statistics = cuda_results[scenario][field]
            if float(statistics["max_abs_error"]) > atol:
                return False
        full = cuda_results[scenario]["Full"]
        if full["threshold"] is not None and full["alarm_agreement_rate"] != 1.0:
            return False
    return True


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()
    if not np.isfinite(args.atol) or args.atol < 0:
        parser.error("--atol must be a finite non-negative number")
    args.out.mkdir(parents=True, exist_ok=True)

    seed = json.loads((args.frozen / "training_summary.json").read_text())["seed"]
    configure(seed, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    report: dict[str, object] = {
        "current_commit": source_commit(),
        "runtime": runtime(),
        "seed": seed,
        "evaluation": {
            "cuda_is_authoritative": True,
            "cpu_is_reference_only": True,
            "component_identity_atol": args.atol,
            "full_q99_alarm_contract": "agreement must equal 1.0 when frozen Full.q99 exists",
        },
        "modes": {},
    }
    output = args.out / "reproduction_diagnosis.json"
    if not torch.cuda.is_available():
        report["status"] = "blocked_no_cuda"
        write_json(output, report)
        return 2

    cuda_pipeline = load_pipe(args.frozen / "model.pt", torch.device("cuda"), seed)
    cuda_results = {scenario: compare(cuda_pipeline, args.frozen, scenario) for scenario in SCENARIOS}
    report["modes"]["cuda"] = cuda_results
    report["cuda_reproduction_pass"] = reproduction_pass(cuda_results, args.atol)
    report["status"] = "reproduction_pass" if report["cuda_reproduction_pass"] else "reproduction_fail"

    try:
        cpu_pipeline = load_pipe(args.frozen / "model.pt", torch.device("cpu"), seed)
        report["modes"]["cpu_reference_only"] = {
            scenario: compare(cpu_pipeline, args.frozen, scenario) for scenario in SCENARIOS
        }
    except Exception as error:
        report["modes"]["cpu_reference_only"] = {"error": f"{type(error).__name__}: {error}"}
    write_json(output, report)
    return 0 if report["cuda_reproduction_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
