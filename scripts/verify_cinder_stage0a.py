#!/usr/bin/env python3
"""Fail-closed verifier for the CINDER Stage-0A artifact."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/cinder_stage0a_clean_emitter_identifiability"
REQUIRED = {
    "README.md", "preregistration.json", "source_commit.json", "source_inventory.json",
    "raw_alignment_verification.json", "clean_split.json", "feature_contract.json",
    "cyclic_feature_summary.json", "pair_inventory.json", "baseline_metrics.csv",
    "verification_metrics.csv", "per_prn_pair_metrics.csv", "window_sensitivity.csv",
    "seed_stability.csv", "shortcut_controls.json", "code_leakage_controls.json",
    "invariance_controls.json", "permutation_controls.json", "bootstrap_intervals.csv",
    "final_verdict.json", "artifact_manifest_sha256.json",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def main() -> int:
    missing = sorted(name for name in REQUIRED if not (ART / name).is_file())
    assert not missing, f"missing files: {missing}"
    manifest = json.loads((ART / "artifact_manifest_sha256.json").read_text())
    actual = {str(path.relative_to(ART)): digest(path) for path in sorted(ART.rglob("*"))
              if path.is_file() and path.name != "artifact_manifest_sha256.json"}
    assert manifest == actual, "artifact manifest mismatch"
    prereg = json.loads((ART / "preregistration.json").read_text())
    assert prereg["frozen_before_results"] and prereg["attack_data_used"] is False and prereg["neural_model_used"] is False
    source = json.loads((ART / "source_inventory.json").read_text())
    assert source["attack_data_used"] is False and source["attack_paths_enumerated_or_read"] is False
    assert all(v["raw"]["status"] == "PASS" and v["raw"]["full_hash_read_this_run"] for v in source["datasets"].values())
    alignment = json.loads((ART / "raw_alignment_verification.json").read_text())
    assert alignment["overall_status"] == "PASS"
    split = json.loads((ART / "clean_split.json").read_text())
    for item in split["datasets"].values():
        used = set(item["feature_train_blocks"] + item["metric_train_blocks"] + item["calibration_blocks"] + item["final_holdout_blocks"])
        assert used.isdisjoint(item["guard_blocks"]) and item["independent_holdout_parent_blocks"] >= 6
    with (ART / "seed_stability.csv").open() as stream: seeds = list(csv.DictReader(stream))
    for dataset in source["datasets"]:
        primary = {int(r["seed"]) for r in seeds if r["dataset"] == dataset and r["feature"] == "Full_C4" and int(r["window_ms"]) == 500}
        assert len(primary) == 10
    verdict = json.loads((ART / "final_verdict.json").read_text())
    assert verdict["verdict"] in {"GO_FOR_CINDER_STAGE0B", "NO_GO_CINDER_CLEAN_IDENTIFIABILITY", "INCONCLUSIVE_INPUT_OR_SAMPLE_SIZE"}
    assert verdict["attack_data_used"] is False and verdict["neural_model_used"] is False
    plots = list((ART / "plots").glob("*.png")); assert len(plots) >= 9
    print(json.dumps({"status": "PASS", "verdict": verdict["verdict"], "manifest_files": len(manifest), "plots": len(plots)}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
