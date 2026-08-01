#!/usr/bin/env python3
"""Fail-closed CLIF-IP Phase-1 alignment/provenance inventory.

This script deliberately does not train, calibrate, or score a fusion model. It only
permits cross-layer Phase-1 when both existing modalities prove a common raw-IQ
recording identity and a common causal timestamp grid.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grid(path: Path, time_column: str) -> dict[str, Any]:
    values: set[float] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if time_column not in (reader.fieldnames or []):
            return {"row_count": 0, "unique_times": [], "cadence_s": None, "time_column_present": False}
        for row in reader:
            values.add(round(float(row[time_column]), 9))
    ordered = sorted(values)
    diffs = sorted({round(b - a, 9) for a, b in zip(ordered, ordered[1:]) if b > a})
    return {
        "row_count": sum(1 for _ in path.open()) - 1,
        "unique_time_count": len(ordered),
        "time_start_s": ordered[0] if ordered else None,
        "time_end_s": ordered[-1] if ordered else None,
        "cadence_s": diffs[0] if len(diffs) == 1 else None,
        "all_cadences_s": diffs,
        "time_column_present": True,
        "unique_times": ordered,
    }


def assess_pair(*, scenario: str, b0: dict[str, Any], m1: dict[str, Any]) -> dict[str, Any]:
    relative = (
        bool(b0.get("exists"))
        and bool(m1.get("exists"))
        and bool(b0.get("sha_matches_manifest"))
        and bool(m1.get("sha_matches_manifest"))
        and b0.get("timestamp_grid_s") == m1.get("timestamp_grid_s")
    )
    same_raw = b0.get("raw_iq_sha256") is not None and b0.get("raw_iq_sha256") == m1.get("raw_iq_sha256")
    blockers: list[str] = []
    if not relative:
        blockers.append("no verified common relative timestamp grid")
    if b0.get("raw_iq_sha256") is None:
        blockers.append("B0 raw_iq_sha256 is unavailable in the paired evidence")
    if m1.get("raw_iq_sha256") is None:
        blockers.append("M1 raw_iq_sha256 is unavailable in the paired evidence")
    if b0.get("raw_iq_sha256") is not None and m1.get("raw_iq_sha256") is not None and not same_raw:
        blockers.append("B0/M1 raw_iq_sha256 mismatch")
    return {
        "scenario": scenario,
        "relative_time_alignment_available": relative,
        "same_recording_proven": same_raw,
        "phase1_cross_layer_permitted": relative and same_raw,
        "blockers": blockers,
    }


def pair_record(name: str, item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    morph = Path(item["morph_csv"])
    floor = Path(item["floor_csv"])
    b_grid = grid(morph, "window_bin_s") if morph.exists() else {}
    m_grid = grid(floor, "window_start_s") if floor.exists() else {}
    b0 = {
        "exists": morph.exists(),
        "path": str(morph),
        "sha256": sha256(morph) if morph.exists() else None,
        "sha_matches_manifest": sha256(morph) == item["morph_sha256"] if morph.exists() else False,
        "timestamp_grid_s": b_grid.get("cadence_s"),
        "time": {k: v for k, v in b_grid.items() if k != "unique_times"},
        "raw_iq_sha256": None,
        "provenance_note": "manifest has morphology file SHA and source_fingerprint, but no paired raw_iq_sha256 field",
    }
    m1 = {
        "exists": floor.exists(),
        "path": str(floor),
        "sha256": sha256(floor) if floor.exists() else None,
        "sha_matches_manifest": sha256(floor) == item["floor_sha256"] if floor.exists() else False,
        "timestamp_grid_s": m_grid.get("cadence_s"),
        "time": {k: v for k, v in m_grid.items() if k != "unique_times"},
        "raw_iq_sha256": None,
        "provenance_note": "manifest names floor_scenario but does not bind the raw-IQ file hash or recording-start sample",
    }
    return b0, m1, assess_pair(scenario=name, b0=b0, m1=m1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("configs/da_pfrt_oakbat_manifest.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    pairs = {}
    for name, item in manifest["datasets"].items():
        b0, m1, gate = pair_record(name, item)
        pairs[name] = {"b0": b0, "m1": m1, "gate": gate, "manifest_identity": item["identity"], "onset_s": item["onset_s"]}
    permitted = all(value["gate"]["phase1_cross_layer_permitted"] for value in pairs.values())
    report = {
        "schema": "gnss-doppler-lab.clif-ip-alignment-gate.v1",
        "phase": "Phase 1 feasibility prerequisite",
        "dataset": "OAKBAT paired morphology/raw-IQ-floor manifest",
        "result": "permitted" if permitted else "blocked",
        "reason": "Cross-layer time statistics require same-recording evidence, not only matching scenario labels and relative timestamps.",
        "pairs": pairs,
    }
    inventory = {
        "B0": {
            "input": "per-PRN 9 prompt-normalized Method-A correlation taps; [batch, 12, 9] history predicts next [9] target",
            "evidence_before_binomial_tail": "per-PRN standardized 9-tap prediction residual exists before RMSE; standard scorer persists RMSE/MAE rather than signed vector",
            "window": "1.0 s window, 0.5 s stride, target window_start_s; available at window end",
            "normalization": "train-split feature mean/std in frozen checkpoint",
            "threshold": "cleanStatic+cleanDynamic PRN quantiles -> binomial tail -> EWMA q99",
        },
        "M1": {
            "input": "raw pre-correlation interleaved int16 I/Q features: power, phase increments/coherence, PSD entropy/flatness/bands, autocorrelation, amplitude statistics",
            "evidence_before_scalar": "PCA AR innovation vector and its RMSE; feature-level robust drift vector",
            "window": "10 ms raw-IQ block at 0.5 s stride; recording-relative window_start_s",
            "normalization": "fit-prefix mean/std before PCA; robust median/MAD for residual and level scores",
            "threshold": "per-recording fit-prefix q99 of causal EWMA ratios",
        },
    }
    (args.out / "data_alignment_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out / "feature_inventory.json").write_text(json.dumps(inventory, indent=2, allow_nan=False) + "\n")
    return 0 if permitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
