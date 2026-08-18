#!/usr/bin/env python3
"""Fail-closed verifier for the compact MIRAGE R1 artifact."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/mirage_stage0a_r1_full_execution"
REQUIRED = [
    "README.md", "CURRENT_STATE.md", "config.json", "versioned_preregistration_amendment.json",
    "preregistration.json", "preregistration_freeze.json", "execution_code_binding.json", "source_commit.json",
    "data_inventory.json", "extended_nav_mapping.csv.gz", "extended_nav_validation.json",
    "common_support_segments.csv", "clean_split_audit.json", "caf_grid.json",
    "caf_reconstruction_validation.json", "injection_design.json", "injection_design_balance.json",
    "injection_design_sha256.json", "case_execution_status.csv", "thresholds.json", "clean_metrics.csv",
    "per_epoch_scores.csv.gz", "per_case_scores.csv.gz", "injection_metrics.csv", "control_metrics.csv",
    "scale_ablation.csv", "relation_destruction_metrics.json", "prn_dominance.json", "shortcut_audit.json",
    "bootstrap_intervals.csv", "final_verdict.json", "runner_phase_evidence.json", "artifact_manifest_sha256.json",
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [name for name in REQUIRED if not (ART / name).is_file()]
    if missing: raise SystemExit(f"missing required files: {missing}")
    manifest = json.loads((ART / "artifact_manifest_sha256.json").read_text())
    for item in manifest["files"]:
        path = ART / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha(path) != item["sha256"]:
            raise SystemExit(f"manifest mismatch: {item['path']}")
    design = json.loads((ART / "injection_design.json").read_text())
    if len(design["cases"]) != 84 or len({row["case_id"] for row in design["cases"]}) != 84:
        raise SystemExit("84-case design binding failed")
    if json.loads((ART / "injection_design_balance.json").read_text())["status"] != "PASS":
        raise SystemExit("design balance is not PASS")
    if json.loads((ART / "caf_reconstruction_validation.json").read_text())["status"] != "PASS":
        raise SystemExit("raw recorrelation gate is not PASS")
    statuses = list(csv.DictReader((ART / "case_execution_status.csv").open()))
    if len(statuses) != 84:
        raise SystemExit("case status row count mismatch")
    verdict = json.loads((ART / "final_verdict.json").read_text())
    recomputed = "GO_FOR_FROZEN_STAGE0B_REAL_STATIC_EVALUATION" if all(row["pass"] for row in verdict["gates"]) else "NO_GO_MIRAGE_PHYSICAL_HYPOTHESIS"
    if verdict["verdict"] != recomputed:
        raise SystemExit(f"verdict recomputation mismatch: {verdict['verdict']} != {recomputed}")
    if verdict["attack_data_accessed"] or verdict["neural_model_executed"]:
        raise SystemExit("forbidden execution marker")
    print(json.dumps({"status": "PASS", "files": len(manifest["files"]), "cases": len(statuses),
                      "verdict": verdict["verdict"]}, sort_keys=True))


if __name__ == "__main__": main()
