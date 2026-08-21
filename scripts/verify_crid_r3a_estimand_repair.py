#!/usr/bin/env python3
"""Compact fail-closed verifier for committed CRID R3a artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ART = ROOT / "artifacts/crid_stage0_r3a_independent_reference_estimand_repair"
ALLOWED = {
    "INDEPENDENT_REFERENCE_ESTIMAND_REPAIR_PASS",
    "INDEPENDENT_REFERENCE_ESTIMAND_REPAIR_FAIL",
    "INCONCLUSIVE_REFERENCE_PROVENANCE",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for payload in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(payload)
    return digest.hexdigest()


def verify_artifact(artifact: Path) -> dict:
    artifact = Path(artifact)
    manifest = json.loads((artifact / "artifact_manifest_sha256.json").read_text())
    checks = []
    for entry in manifest["files"]:
        path = artifact / entry["path"]
        match = path.is_file() and path.stat().st_size == int(entry["size_bytes"]) and sha256_file(path) == entry["sha256"]
        checks.append({"path": entry["path"], "match": match})
    required = {
        "README.md", "repair_preregistration.json", "estimand_definition.md", "source_binding.json",
        "legacy_validator_reproduction.csv", "legacy_reproduction_summary.json",
        "joint_reference_validation.csv", "joint_reference_summary.json", "denominator_diagnostic.csv",
        "solver_diagnostics.csv", "validation_checkpoint.json", "attack_access_audit.json",
        "final_verdict.json", "plots/prn_denominator_comparison.png", "plots/requested_vs_recovered_power.png",
    }
    listed = {entry["path"] for entry in manifest["files"]}
    verdict = json.loads((artifact / "final_verdict.json").read_text())
    legacy = json.loads((artifact / "legacy_reproduction_summary.json").read_text())
    joint = json.loads((artifact / "joint_reference_summary.json").read_text())
    binding = json.loads((artifact / "source_binding.json").read_text())
    attack = json.loads((artifact / "attack_access_audit.json").read_text())
    with (artifact / "legacy_validator_reproduction.csv").open(newline="") as stream:
        legacy_rows = list(csv.DictReader(stream))
    with (artifact / "joint_reference_validation.csv").open(newline="") as stream:
        joint_rows = list(csv.DictReader(stream))
    inventory_ok = (
        len(legacy_rows) == len(joint_rows) == 180
        and len({(row["domain"], row["case_id"]) for row in joint_rows}) == 36
        and all(sum(x["domain"] == row["domain"] and x["case_id"] == row["case_id"] for x in joint_rows) == 5 for row in joint_rows)
    )
    pass_claim_ok = True
    if verdict["verdict"] == "INDEPENDENT_REFERENCE_ESTIMAND_REPAIR_PASS":
        pass_claim_ok = (
            legacy["passed"] == 171 and legacy["failed"] == 9
            and legacy["same_nine_oak_prn21_failures"] and legacy["numeric_match_to_committed_r3"]
            and joint["passed"] == 180 and joint["failed"] == 0
            and joint["all_rank_five"] and joint["all_condition_at_most_1e6"]
            and joint["deterministic_rerun_match"]
            and binding["status"] == "PASS" and binding["full_hash_executed"]
            and attack["attack_bytes_read"] == 0 and not attack["crid_score_computed"]
            and verdict["next_state"] == "READY_TO_REPEAT_CRID_PHASE_A"
        )
    passed = (
        all(row["match"] for row in checks)
        and required.issubset(listed)
        and verdict["verdict"] in ALLOWED
        and inventory_ok
        and pass_claim_ok
        and manifest.get("status") == "PASS"
    )
    return {
        "schema": "gnss-doppler-lab.crid-r3a-compact-verifier.v1",
        "manifest_files": len(checks), "manifest_checks_pass": all(row["match"] for row in checks),
        "required_files_present": required.issubset(listed), "inventory_complete": inventory_ok,
        "pass_claim_contract": pass_claim_ok, "verdict": verdict["verdict"],
        "status": "PASS" if passed else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", type=Path, default=DEFAULT_ART)
    args = parser.parse_args(); result = verify_artifact(args.artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
