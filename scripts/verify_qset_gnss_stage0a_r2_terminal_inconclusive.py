#!/usr/bin/env python3
"""Independent compact verifier for the fail-closed Q-SET R2 result."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.qset_stage0a_r2 import AGGREGATORS, canonical_sha, sha256_file

ARTIFACT_REL = Path("artifacts/qset_gnss_stage0a_r2_galileo_partial_prn_execution")
REQUIRED = {
    "README.md",
    "access_audit.json",
    "aggregator_comparison.csv",
    "artifact_manifest_sha256.json",
    "clean_score_summary.json",
    "dataset_download_plan.json",
    "dataset_preflight.json",
    "deterministic_reproduction.json",
    "execution_freeze.json",
    "final_verdict.json",
    "freeze_commit.json",
    "normal_model.json",
    "per_prn_ground_truth_metrics.csv",
    "per_prn_support.csv",
    "per_window_scores.csv.gz",
    "preregistration.json",
    "preregistration_commit.json",
    "receiver_binary_inventory.json",
    "scenario_metrics.json",
    "shortcut_audit.json",
    "source_binding.json",
    "stage0a_gate.json",
    "synthetic_dilution_control.json",
    "terminal_execution_attestation.json",
    "threshold_binding.json",
}


class VerificationError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise VerificationError(message)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_terminal_failure(failure: dict) -> None:
    require(failure["status"] == "FAIL_INSUFFICIENT_GALILEO_RECEIVER_SUPPORT", "SS-1 failure status")
    trace = failure["trace_validation"]
    require(trace["status"] == "FAIL", "SS-1 TRACE status")
    require(trace["tracked_prn_count"] == 3 and trace["tracked_prns"] == [9, 30, 36], "SS-1 tracked support")
    require(trace["finite_failures"] == trace["cadence_failures"] == trace["causal_failures"] == 0, "SS-1 TRACE integrity")
    require(failure["receiver"]["terminal_drain"] is True and failure["receiver"]["program_ended"] is True, "SS-1 terminal evidence")
    require(failure["support_gate"]["pass"] is False, "SS-1 support gate")
    require(failure["score_computed"] is False, "attack score unexpectedly computed")
    require(failure["downstream_attack_scenarios_opened"] == [], "downstream attack access")
    output_set = failure["output_set"]
    require(output_set["file_count"] == len(output_set["files"]), "SS-1 output file count")
    require(canonical_sha(output_set["files"]) == output_set["aggregate_sha256"], "SS-1 output aggregate")


def validate_access(audit: dict) -> None:
    require(audit["status"] == "PASS", "access audit status")
    require(audit["attack_access_after_freeze_only"] is True, "attack accessed before freeze")
    payload = audit["attack_payload"]
    require(payload["allowlisted_scenarios"] == ["SS-1"], "attack access scenario set")
    require(payload["bytes_read"] == 59_999_664_000, "SS-1 raw byte audit")
    require(audit["unopened_allowlisted_scenarios"] == ["SS-3", "SS-5", "SS-11"], "unopened scenario audit")
    require(
        all(audit["unallowlisted_tuni2025_raw"][key] == 0 for key in ("stats", "hashes", "opens", "mmaps", "bytes_read")),
        "unallowlisted access",
    )
    operations = audit["attack_scientific_operations"]
    require(operations["feature_windows"] == operations["scores"] == operations["attack_evaluations"] == 0, "attack score operations")


def validate_manifest(root: Path, manifest: dict) -> None:
    require(manifest["status"] == "PASS", "manifest status")
    rows = []
    for row in manifest["files"]:
        path = root / row["path"]
        require(path.is_file(), f"manifest missing {path}")
        require(path.stat().st_size == row["size_bytes"], f"manifest size drift {path}")
        require(sha256_file(path) == row["sha256"], f"manifest hash drift {path}")
        rows.append(row)
    require(canonical_sha(rows) == manifest["aggregate_sha256"], "manifest aggregate")


def validate_artifact(root: Path) -> dict:
    missing = sorted(name for name in REQUIRED if not (root / name).is_file())
    require(not missing, f"missing {missing}")
    require((root / "plots/receiver_support_failure.png").is_file(), "support failure plot")

    freeze = load(root / "execution_freeze.json")
    require(freeze["status"] == "PASS_PRE_ATTACK", "freeze status")
    for relative, expected in freeze["code_bindings"].items():
        require(sha256_file(ROOT / relative) == expected, f"frozen scientific code drift {relative}")
    require(canonical_sha(load(root / "normal_model.json")) == freeze["model_sha256"], "literal model binding")
    threshold = load(root / "threshold_binding.json")
    threshold_sha = canonical_sha(
        {"multi_q_reference": threshold["multi_q_reference"], "thresholds": threshold["thresholds"]}
    )
    require(threshold_sha == threshold["threshold_sha256"] == freeze["threshold_sha256"], "threshold binding")

    failure = load(root / "receiver_manifests/SS-1.json")
    validate_terminal_failure(failure)
    require(len(list((root / "receiver_manifests").glob("*.json"))) == 3, "receiver manifest count")

    with (root / "aggregator_comparison.csv").open(newline="", encoding="utf-8") as stream:
        aggregator_rows = list(csv.DictReader(stream))
    require(len(aggregator_rows) == len(AGGREGATORS), "not-computed aggregator rows")
    require({row["aggregator"] for row in aggregator_rows} == set(AGGREGATORS), "aggregator names")
    require(all(row["status"] == "NOT_COMPUTED_TECHNICAL_SUPPORT_GATE" for row in aggregator_rows), "aggregator status")

    with gzip.open(root / "per_window_scores.csv.gz", "rt", newline="", encoding="utf-8") as stream:
        score_rows = list(csv.DictReader(stream))
    require(score_rows == [], "attack scores must be empty")

    gate = load(root / "stage0a_gate.json")
    require(gate["technical"]["pass"] is False and gate["overall_pass"] is False, "technical gate")
    final = load(root / "final_verdict.json")
    require(final["verdict"] == "INCONCLUSIVE_QSET_DATA_FORMAT_OR_RECEIVER_SUPPORT", "final verdict")
    require(final["stage0b_authorized"] is False and final["score_computed"] is False, "authorization")

    freeze_commit = load(root / "freeze_commit.json")
    require(freeze_commit["status"] == "PASS" and freeze_commit["commit_sha"] == final["freeze_sha"], "freeze attestation")
    terminal = load(root / "terminal_execution_attestation.json")
    require(terminal["status"] == "PASS_FAIL_CLOSED_AT_FROZEN_SUPPORT_GATE", "terminal attestation")
    require(terminal["freeze_sha"] == final["freeze_sha"], "terminal freeze SHA")
    require(terminal["scientific_code_bindings_unchanged"] is True, "scientific code changed")
    require(terminal["attack_score_computed"] is False and terminal["downstream_attack_scenarios_opened"] == [], "terminal access")
    for relative, expected in terminal["reporting_code_bindings"].items():
        require(sha256_file(ROOT / relative) == expected, f"reporting code drift {relative}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", terminal["reporting_commit"], "HEAD"],
        cwd=ROOT,
        check=False,
    )
    require(ancestor.returncode == 0, "reporting commit ancestry")

    deterministic = load(root / "deterministic_reproduction.json")
    require(
        deterministic["status"] == "PASS_TECHNICAL_FAILURE_REPRODUCED"
        and deterministic["byte_identical_compact_failure_evidence"] is True,
        "technical failure determinism",
    )
    validate_access(load(root / "access_audit.json"))
    validate_manifest(root, load(root / "artifact_manifest_sha256.json"))
    return {
        "status": "PASS",
        "verdict": final["verdict"],
        "stage0b_authorized": False,
        "attack_score_computed": False,
        "tracked_prns_ss1": failure["trace_validation"]["tracked_prns"],
        "manifest_files": load(root / "artifact_manifest_sha256.json")["file_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=ROOT / ARTIFACT_REL)
    args = parser.parse_args()
    try:
        result = validate_artifact(args.artifact)
    except (VerificationError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
