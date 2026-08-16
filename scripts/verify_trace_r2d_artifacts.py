#!/usr/bin/env python3
"""Verify R2d artifacts, repaired clean support, and scientific sealing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "artifacts/trace_stage0_r2d_oakbat_clean_support_repair"
REQUIRED = {
    "README.md", "preregistration.json", "source_commit.json", "diagnosis.json",
    "repair_plan_preregistered.json", "oakbat_clean_support_audit.json",
    "receiver_build_manifest.json", "handoff_manifest.json", "config.json",
    "semantic_reproduction_contract.json", "raw_source_binding.json",
    "rep3_rep4_reproduction_metrics.json", "terminal_row_set_audit.json",
    "action_mapping_validation.json", "replay_inventory.json", "phase_b_metrics.json",
    "final_verdict.json", "test_results.json", "runner_runs.json",
    "artifact_manifest_sha256.json",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    artifact = args.artifact_root.resolve()
    missing = sorted(name for name in REQUIRED if not (artifact / name).is_file())
    failures = []
    manifest = json.loads((artifact / "artifact_manifest_sha256.json").read_text())
    for relative, expected in manifest["files"].items():
        path = artifact / relative
        if not path.is_file() or path.stat().st_size != expected["byte_size"] or sha(path) != expected["sha256"]:
            failures.append(relative)
    diagnosis = json.loads((artifact / "diagnosis.json").read_text())
    support = json.loads((artifact / "oakbat_clean_support_audit.json").read_text())
    phase_a = json.loads((artifact / "rep3_rep4_reproduction_metrics.json").read_text())
    terminal = json.loads((artifact / "terminal_row_set_audit.json").read_text())
    verdict = json.loads((artifact / "final_verdict.json").read_text())
    phase_b = json.loads((artifact / "phase_b_metrics.json").read_text())
    scientific = (
        diagnosis["attack_performance_read_or_computed"] is False
        and phase_a["phase_a_status"] == "PASS"
        and phase_a["phase_b_authorized"] is True
        and terminal["status"] == "PASS"
        and terminal["whole_replay_row_set_identical"] is True
        and terminal["terminal_row_counts_per_prn_channel_identical"] is True
        and support["status"] == "PASS"
        and all(value > 0 for value in support["chronological_clean_support"]["role_pair_counts"].values())
        and verdict["phase_a_passed"] is True
        and verdict["phase_b_authorized"] is True
    )
    computed = verdict["verdict"] in {"GO_TRACE_PHYSICAL_HYPOTHESIS", "NO_GO_TRACE_PHYSICAL_HYPOTHESIS"}
    if computed:
        scientific = scientific and verdict["attack_metrics_computed"] is True and phase_b["status"] == "AVAILABLE"
    else:
        scientific = scientific and verdict["attack_metrics_computed"] is False and phase_b["status"] == "UNAVAILABLE" and not verdict.get("performance_claimed", False)
    status = "PASS" if not missing and not failures and scientific else "FAIL"
    report = {
        "schema": "gnss-doppler-lab.trace-r2d-artifact-verification.v1",
        "status": status,
        "artifact_root": str(artifact),
        "manifest_entry_count": len(manifest["files"]),
        "missing_required_artifacts": missing,
        "hash_failures": failures,
        "scientific_seal_valid": scientific,
        "phase_b_metrics_computed": computed,
        "final_verdict": verdict["verdict"],
    }
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(output, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
