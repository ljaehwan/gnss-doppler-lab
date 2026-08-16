#!/usr/bin/env python3
"""Verify required R2c artifacts, scientific sealing, and SHA-256 manifest."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "artifacts/trace_stage0_r2c_terminal_drain_repair"
REQUIRED = {
    "README.md",
    "diagnosis.json",
    "repair_plan_preregistered.json",
    "terminal_row_set_audit.json",
    "source_commit.json",
    "receiver_build_manifest.json",
    "handoff_manifest.json",
    "config.json",
    "preregistration.json",
    "semantic_reproduction_contract.json",
    "raw_source_binding.json",
    "replay_inventory.json",
    "rep3_rep4_reproduction_metrics.json",
    "action_mapping_validation.json",
    "final_verdict.json",
    "test_results.json",
    "runner_runs.json",
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
    artifact = parser.parse_args().artifact_root.resolve()
    manifest = json.loads((artifact / "artifact_manifest_sha256.json").read_text())
    failures = []
    for relative, expected in manifest["files"].items():
        path = artifact / relative
        if (
            not path.is_file()
            or path.stat().st_size != expected["byte_size"]
            or sha(path) != expected["sha256"]
        ):
            failures.append(relative)
    missing = sorted(name for name in REQUIRED if not (artifact / name).is_file())
    diagnosis = json.loads((artifact / "diagnosis.json").read_text())
    phase_a = json.loads((artifact / "rep3_rep4_reproduction_metrics.json").read_text())
    audit = json.loads((artifact / "terminal_row_set_audit.json").read_text())
    verdict = json.loads((artifact / "final_verdict.json").read_text())
    scientific = (
        diagnosis["attack_performance_read_or_computed"] is False
        and phase_a["phase_b_authorized"] == verdict["phase_b_authorized"]
        and verdict["phase_a_passed"] == (phase_a["phase_a_status"] == "PASS")
    )
    if phase_a["phase_b_authorized"]:
        scientific = scientific and all(
            (
                audit["status"] == "PASS",
                audit["whole_replay_row_set_identical"],
                audit["terminal_row_counts_per_prn_channel_identical"],
                audit["rep3_only_rows"] == 0,
                audit["rep4_only_rows"] == 0,
                audit["rep3"]["row_set_sha256"] == audit["rep4"]["row_set_sha256"],
                verdict["phase_b_run"],
            )
        )
        if verdict["verdict"] in {
            "GO_TRACE_PHYSICAL_HYPOTHESIS",
            "NO_GO_TRACE_PHYSICAL_HYPOTHESIS",
        }:
            scientific = scientific and verdict["attack_metrics_computed"]
        else:
            scientific = (
                scientific
                and verdict["verdict"] == "INCONCLUSIVE_INPUT_OR_RECEIVER"
                and verdict["attack_metrics_computed"] is False
                and verdict["normal_fpr"]["status"] == "UNAVAILABLE"
            )
    else:
        scientific = (
            scientific
            and verdict["attack_metrics_computed"] is False
            and verdict["normal_fpr"]["status"] == "UNAVAILABLE"
        )
    status = "PASS" if not failures and not missing and scientific else "FAIL"
    print(
        json.dumps(
            {
                "schema": "gnss-doppler-lab.trace-r2c-artifact-verification.v1",
                "status": status,
                "artifact_root": str(artifact),
                "manifest_entry_count": len(manifest["files"]),
                "hash_failures": failures,
                "missing_required_artifacts": missing,
                "scientific_seal_valid": scientific,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
