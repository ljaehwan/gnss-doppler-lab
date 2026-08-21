#!/usr/bin/env python3
"""Compact fail-closed verifier for committed CRID R4a evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "artifacts/crid_stage0_r4a_threshold_decision_equivalence_repair"
REQUIRED = {
    "README.md", "repair_preregistration.json", "preregistration_commit.json", "source_binding.json",
    "threshold_numeric_comparison.json", "clean_split_identity.json", "holdout_alarm_equivalence.json",
    "attack_and_control_access_audit.json", "validation_checkpoint.json", "final_verdict.json",
    "tamper_test_results.json", "test_results.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for payload in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(payload)
    return digest.hexdigest()


def verify(artifact: Path) -> dict:
    artifact = Path(artifact); failures = []
    try:
        manifest = json.loads((artifact / "artifact_manifest_sha256.json").read_text())
        entries = manifest["files"]; listed = {row["path"] for row in entries}
        if manifest.get("schema") != "gnss-doppler-lab.crid-r4a-artifact-manifest.v1" or manifest.get("status") != "PASS" or manifest.get("file_count") != len(entries):
            failures.append("manifest_contract")
        if len(listed) != len(entries): failures.append("manifest_duplicates")
        for row in entries:
            relative = Path(row["path"]); path = artifact / relative
            safe = not relative.is_absolute() and ".." not in relative.parts
            if not safe or not path.is_file() or path.stat().st_size != int(row["size_bytes"]) or sha256_file(path) != row["sha256"]:
                failures.append(f"manifest:{row['path']}")
        failures.extend(f"missing:{name}" for name in sorted(REQUIRED - listed))
        prereg = json.loads((artifact / "repair_preregistration.json").read_text())
        prereg_commit = json.loads((artifact / "preregistration_commit.json").read_text())
        source = json.loads((artifact / "source_binding.json").read_text())
        numeric = json.loads((artifact / "threshold_numeric_comparison.json").read_text())
        split = json.loads((artifact / "clean_split_identity.json").read_text())
        alarms = json.loads((artifact / "holdout_alarm_equivalence.json").read_text())
        audit = json.loads((artifact / "attack_and_control_access_audit.json").read_text())
        checkpoint = json.loads((artifact / "validation_checkpoint.json").read_text())
        tests = json.loads((artifact / "test_results.json").read_text())
        tamper = json.loads((artifact / "tamper_test_results.json").read_text())
        final = json.loads((artifact / "final_verdict.json").read_text())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"schema": "gnss-doppler-lab.crid-r4a-compact-verifier.v1", "status": "FAIL", "failures": [f"exception:{type(exc).__name__}"]}
    if prereg.get("status") != "PRE_IMPLEMENTATION_METHOD_REPAIR_FROZEN": failures.append("preregistration")
    if prereg_commit.get("status") != "PASS" or prereg_commit.get("ahead") != 0 or prereg_commit.get("behind") != 0: failures.append("preregistration_commit")
    if source.get("status") != "PASS" or source["r4_artifact_unchanged"].get("status") != "PASS": failures.append("source_binding")
    if split.get("status") != "PASS" or numeric.get("status") != "PASS" or alarms.get("status") != "PASS": failures.append("equivalence_gate")
    for domain in ("OAK", "TEX"):
        n = numeric["domains"][domain]; a = alarms["domains"][domain]
        if not n.get("numeric_sanity_pass") or not n.get("fit_and_independent_q99_equal"): failures.append(f"numeric:{domain}")
        if not a.get("alarm_vectors_byte_identical") or not a.get("false_positive_count_and_fpr_equal") or not a.get("expected_committed_fpr_exact_match"): failures.append(f"decision:{domain}")
        if not a.get("causal_delays_match_r4") or not a.get("all_scored_epochs_finite_four_config_min_four_prn"): failures.append(f"support:{domain}")
    if any(audit.get(name) != 0 for name in ("control_replays_executed", "control_scores_read", "control_scores_computed", "attack_stats", "attack_hashes", "attack_opens", "attack_mmaps", "attack_bytes_read")): failures.append("forbidden_access")
    if audit.get("phase_a_executed") is not False or audit.get("phase_b_executed") is not False: failures.append("phase_execution")
    if checkpoint.get("status") != "PASS" or tests.get("status") != "PASS" or tamper.get("status") != "PASS": failures.append("validation")
    if final.get("verdict") == "THRESHOLD_DECISION_EQUIVALENCE_REPAIR_PASS":
        if final.get("next_state") != "READY_TO_REPEAT_CRID_PHASE_A" or final.get("r4_verdict_preserved") != "INCONCLUSIVE_CRID_PHASE_A_EXECUTION_OR_PROVENANCE": failures.append("pass_claim")
    elif final.get("verdict") == "INCONCLUSIVE_THRESHOLD_DECISION_PROVENANCE":
        if final.get("next_state") != "NOT_AUTHORIZED": failures.append("inconclusive_claim")
    else: failures.append("verdict")
    return {
        "schema": "gnss-doppler-lab.crid-r4a-compact-verifier.v1",
        "manifest_files": len(entries), "verdict": final.get("verdict"),
        "failures": failures, "status": "PASS" if not failures else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", type=Path, default=DEFAULT)
    args = parser.parse_args(); result = verify(args.artifact)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__": raise SystemExit(main())
