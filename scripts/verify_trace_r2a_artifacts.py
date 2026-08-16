#!/usr/bin/env python3
"""Verify the committed TRACE-R2a artifact inventory and SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts/trace_stage0_r2a_reproducibility_repair"
REQUIRED = {
    "README.md",
    "config.json",
    "preregistration.json",
    "source_commit.json",
    "r2_existing_failure_preservation.json",
    "rep1_rep2_root_cause_audit.json",
    "canonical_record_comparison.csv.gz",
    "channel_prn_assignment_comparison.csv",
    "first_divergence_analysis.json",
    "receiver_nondeterminism_audit.json",
    "receiver_repair.diff",
    "receiver_build_manifest.json",
    "semantic_reproduction_contract.json",
    "rep3_rep4_reproduction_metrics.json",
    "action_mapping_validation.json",
    "raw_source_binding.json",
    "replay_inventory.json",
    "clean_split_audit.json",
    "thresholds.json",
    "scenario_metrics.csv",
    "ablation_metrics.csv",
    "per_epoch_scores.csv.gz",
    "action_shuffle_metrics.json",
    "physical_controls.json",
    "bootstrap_intervals.csv",
    "final_verdict.json",
    "runner_runs.json",
    "test_results.json",
    "artifact_manifest_sha256.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    artifact = args.artifact_root.resolve()
    manifest_path = artifact / "artifact_manifest_sha256.json"
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["files"]
    failures = []
    for relative, expected in entries.items():
        path = artifact / relative
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
            continue
        actual = sha256_file(path)
        if actual != expected["sha256"] or path.stat().st_size != expected["byte_size"]:
            failures.append(
                {
                    "path": relative,
                    "reason": "hash_or_size_mismatch",
                    "expected_sha256": expected["sha256"],
                    "actual_sha256": actual,
                    "expected_byte_size": expected["byte_size"],
                    "actual_byte_size": path.stat().st_size,
                }
            )
    missing_required = sorted(name for name in REQUIRED if not (artifact / name).is_file())
    status = "PASS" if not failures and not missing_required else "FAIL"
    result = {
        "schema": "gnss-doppler-lab.trace-r2a-artifact-verification.v1",
        "status": status,
        "artifact_root": str(artifact),
        "verified_file_count": len(entries) - len(failures),
        "manifest_entry_count": len(entries),
        "manifest_declared_exclusions": manifest["excluded_self_referential_or_run_provenance_files"],
        "failures": failures,
        "missing_required_artifacts": missing_required,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
