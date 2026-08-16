#!/usr/bin/env python3
"""Verify R2e artifacts, repaired attack support, and scientific sealing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "artifacts/trace_stage0_r2e_attack_support_repair"
REQUIRED = {
    "README.md", "preregistration.json", "source_commit.json", "diagnosis.json",
    "ds7_attack_support_audit.json", "os4_attack_support_audit.json",
    "receiver_build_manifest.json", "handoff_manifest.json", "config.json",
    "handoff_path_mirror_manifest.json", "semantic_reproduction_contract.json",
    "raw_source_binding.json", "rep3_rep4_reproduction_metrics.json",
    "phase_a_reproduction_metrics.json", "terminal_row_set_audit.json",
    "action_mapping_validation.json", "replay_inventory.json", "phase_b_metrics.json",
    "final_verdict.json", "test_results.json", "runner_runs.json",
    "scientific_seal.json", "artifact_manifest_sha256.json",
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
    prereg = json.loads((artifact / "preregistration.json").read_text())
    diagnosis = json.loads((artifact / "diagnosis.json").read_text())
    ds7 = json.loads((artifact / "ds7_attack_support_audit.json").read_text())
    os4 = json.loads((artifact / "os4_attack_support_audit.json").read_text())
    phase_a = json.loads((artifact / "rep3_rep4_reproduction_metrics.json").read_text())
    phase_a_alias = json.loads((artifact / "phase_a_reproduction_metrics.json").read_text())
    terminal = json.loads((artifact / "terminal_row_set_audit.json").read_text())
    verdict = json.loads((artifact / "final_verdict.json").read_text())
    phase_b = json.loads((artifact / "phase_b_metrics.json").read_text())
    seal = json.loads((artifact / "scientific_seal.json").read_text())
    scientific = (
        prereg["task_base_commit"] == "e66619e31a937186e522d8566711436e24f2b99d"
        and prereg["frozen_phase_b_scorer"]["sha256"] == sha(ROOT / "scripts/evaluate_trace_r2_phase_b.py")
        and diagnosis["attack_scores_read_or_computed"] is False
        and phase_a == phase_a_alias
        and phase_a["phase_a_status"] == "PASS"
        and phase_a["phase_b_authorized"] is True
        and terminal["status"] == "PASS"
        and terminal["whole_replay_row_set_identical"] is True
        and terminal["terminal_row_counts_per_prn_channel_identical"] is True
        and seal["status"] == "SEALED"
        and seal["frozen_contract_unchanged"] is True
    )
    support_complete = (
        ds7["status"] == "PASS" and os4["status"] == "PASS"
        and ds7["frozen_support"]["pre_onset_four_prn_block_count"] > 0
        and ds7["frozen_support"]["post_onset_four_prn_block_count"] > 0
        and os4["frozen_support"]["pre_onset_four_prn_block_count"] > 0
        and os4["frozen_support"]["post_onset_four_prn_block_count"] > 0
    )
    computed = verdict["verdict"] in {"GO_TRACE_PHYSICAL_HYPOTHESIS", "NO_GO_TRACE_PHYSICAL_HYPOTHESIS"}
    if computed:
        scientific = (
            scientific and support_complete and verdict["attack_metrics_computed"] is True
            and verdict["attack_scores_computed"] is True
            and phase_b["status"] == "AVAILABLE"
            and phase_b["metrics_computed"] is True
            and seal["phase_b_metrics_computed"] is True
        )
    else:
        support_failure = artifact / "phase_b_support_failure_audit.json"
        scientific = (
            scientific and not support_complete and support_failure.is_file()
            and verdict["attack_metrics_computed"] is False
            and phase_b["status"] == "UNAVAILABLE"
            and verdict.get("performance_claimed", False) is False
            and phase_b.get("performance_claimed", False) is False
        )
    runner = json.loads((artifact / "runner_runs.json").read_text())
    required_run_markers = (
        "parent-audit-preregistration", "ds7-root-cause-audit", "os4-root-cause-audit",
        "support-acquisition", "receiver-handoff-repair-freeze", "phase-a-semantic-evaluate",
        "phase-b-support-validation", "phase-b-finalize",
    )
    run_names = [run["name"] for run in runner["runs"] if run["status"] == "succeeded"]
    durable = all(any(marker in name for name in run_names) for marker in required_run_markers)
    status = "PASS" if not missing and not failures and scientific and durable else "FAIL"
    report = {
        "schema": "gnss-doppler-lab.trace-r2e-artifact-verification.v1",
        "status": status,
        "artifact_root": str(artifact),
        "manifest_entry_count": len(manifest["files"]),
        "missing_required_artifacts": missing,
        "hash_failures": failures,
        "scientific_seal_valid": scientific,
        "durable_run_coverage_valid": durable,
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
