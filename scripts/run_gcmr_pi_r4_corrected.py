#!/usr/bin/env python3
"""Fail-closed corrected r4 reconstruction gate for frozen OAKBAT GCMR-PI r3."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch

from train_gcmr_peak_innovation import peak_indexes, records
from run_gcmr_oakbat_poc import SCENARIOS as SOURCE_SCENARIOS, load_scenario
from gnss_doppler_lab.gcmr_pi_r4_corrected import FailClosedError, reconstruct_event_innovation, rescore_from_innovations

FIELDS = ("S_common", "N_eff", "S_pair", "energy", "Full")
SCENARIOS = ("os1", "os2", "os3", "os4")


def load_rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def cpu_frozen_pipeline(path: Path):
    pipeline = torch.load(path, map_location="cpu", weights_only=False)
    pipeline.device = torch.device("cpu")
    pipeline.network.to(pipeline.device).eval()
    return pipeline


def compare_scenario(pipeline, frozen: Path, scenario: str, atol: float = 1e-8):
    relation_events, _, _ = load_scenario(scenario, frozen / "cache", False)
    built, rejected = records(relation_events, SOURCE_SCENARIOS[scenario], pipeline.window, peak_indexes(relation_events, SOURCE_SCENARIOS[scenario], pipeline.feature_dim))
    reference = load_rows(frozen / f"{scenario}_scores.csv")
    if len(built) != len(reference):
        raise FailClosedError(f"{scenario}: regenerated EventRecord count {len(built)} != frozen score count {len(reference)}")
    max_abs = {key: 0.0 for key in FIELDS}
    for index, (event, expected) in enumerate(zip(built, reference)):
        residual, z = reconstruct_event_innovation(pipeline, event)
        diagnostics, score = rescore_from_innovations(pipeline, event, z, residual)
        actual = {"S_common": diagnostics.s_common, "N_eff": diagnostics.n_eff, "S_pair": diagnostics.s_pair, "energy": diagnostics.energy, "Full": score["Full"]}
        if abs(event.time - float(expected["time"])) > 1e-9:
            raise FailClosedError(f"{scenario}: event {index} time mismatch")
        for key, value in actual.items():
            max_abs[key] = max(max_abs[key], abs(value - float(expected[key])))
    bad = {key: value for key, value in max_abs.items() if value > atol}
    if bad:
        raise FailClosedError(f"{scenario}: frozen r3 reproduction mismatch; max_abs={bad}; tolerance={atol}")
    return {"scenario": scenario, "events": len(built), "rejected": len(rejected), "max_abs": max_abs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    evidence = {"status": "blocked", "contract": "No proxy calculation. Relation destruction is forbidden until actual raw residual/whitened innovation reproduction matches frozen r3 scores.", "frozen_model": str(args.frozen / "model.pt"), "scenarios_checked": []}
    try:
        pipe = cpu_frozen_pipeline(args.frozen / "model.pt")
        for scenario in SCENARIOS:
            result = compare_scenario(pipe, args.frozen, scenario)
            evidence["scenarios_checked"].append(result)
    except Exception as exc:
        evidence.update({"exception_type": type(exc).__name__, "reason": str(exc), "destruction_executed": False, "thresholds_emitted": False, "innovation_archives_emitted": False})
        (args.out / "blocker_evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        return 2
    evidence.update({"status": "reproduction_passed", "destruction_executed": False})
    (args.out / "reproduction_check.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
