#!/usr/bin/env python3
"""Independent compact verifier for Q-SET-GNSS Stage-0A R2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.qset_stage0a_r2 import AGGREGATORS, CONFIG_SHA256, PREREGISTRATION_SHA, canonical_sha, sha256_file

ARTIFACT_REL = Path("artifacts/qset_gnss_stage0a_r2_galileo_partial_prn_execution")
REQUIRED = {
    "README.md", "dataset_preflight.json", "dataset_download_plan.json", "source_binding.json", "preregistration.json", "preregistration_commit.json",
    "execution_freeze.json", "freeze_commit.json", "receiver_binary_inventory.json", "per_prn_support.csv", "clean_score_summary.json", "normal_model.json", "threshold_binding.json",
    "synthetic_dilution_control.json", "scenario_metrics.json", "aggregator_comparison.csv", "per_prn_ground_truth_metrics.csv", "shortcut_audit.json",
    "access_audit.json", "stage0a_gate.json", "final_verdict.json", "deterministic_reproduction.json", "artifact_manifest_sha256.json",
}
VERDICTS = {"QSET_STAGE0A_PARTIAL_PRN_AGGREGATION_PASS", "QSET_STAGE0A_PARTIAL_SIGNAL_WITHOUT_AGGREGATION_GAIN", "NO_GO_QSET_PARTIAL_PRN_SIGNAL", "INCONCLUSIVE_QSET_DATA_FORMAT_OR_RECEIVER_SUPPORT", "BLOCKED_TUNI2025_DATASET_NOT_LOCAL"}


class VerificationError(RuntimeError): pass


def require(value: bool, message: str) -> None:
    if not value: raise VerificationError(message)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_access(audit: dict) -> None:
    require(audit["status"] == "PASS", "access audit")
    require(audit["attack_access_after_freeze_only"] is True, "attack pre-freeze access")
    require(audit["attack_payload"]["allowlisted_scenarios"] == ["SS-1", "SS-3", "SS-5", "SS-11"], "attack allowlist")
    require(all(audit["unallowlisted_tuni2025_raw"][key] == 0 for key in ("stats", "hashes", "opens", "mmaps", "bytes_read")), "unallowlisted access")


def validate_manifest(root: Path, manifest: dict) -> None:
    require(manifest["status"] == "PASS", "manifest status")
    rows = []
    for row in manifest["files"]:
        path = root / row["path"]; require(path.is_file(), f"manifest missing {path}")
        require(path.stat().st_size == row["size_bytes"] and sha256_file(path) == row["sha256"], f"manifest drift {path}")
        rows.append(row)
    require(canonical_sha(rows) == manifest["aggregate_sha256"], "manifest aggregate")


def validate_artifact(root: Path) -> dict:
    missing = sorted(name for name in REQUIRED if not (root / name).is_file()); require(not missing, f"missing {missing}")
    require((root / "plots/aggregator_detection_rate.png").is_file(), "missing plot")
    prereg_commit = load(root / "preregistration_commit.json"); require(prereg_commit["commit_sha"] == PREREGISTRATION_SHA, "prereg SHA")
    freeze = load(root / "execution_freeze.json"); require(freeze["status"] == "PASS_PRE_ATTACK" and freeze["configuration_sha256"] == CONFIG_SHA256, "freeze")
    for relative, expected in freeze["code_bindings"].items(): require(sha256_file(ROOT / relative) == expected, f"code drift {relative}")
    threshold = load(root / "threshold_binding.json"); require(threshold["attack_data_used"] is False and set(threshold["thresholds"]) == set(AGGREGATORS), "threshold binding")
    threshold_sha = canonical_sha({"multi_q_reference": threshold["multi_q_reference"], "thresholds": threshold["thresholds"]})
    require(threshold_sha == threshold["threshold_sha256"] == freeze["threshold_sha256"], "threshold hash")
    require(canonical_sha(load(root / "normal_model.json")) == freeze["model_sha256"], "model hash")
    require(load(root / "receiver_binary_inventory.json")["receiver_sha256"] == freeze["receiver_sha256"], "receiver binding")
    require(load(root / "synthetic_dilution_control.json")["claimed_as_detection_evidence"] is False, "synthetic claim")
    with (root / "aggregator_comparison.csv").open(newline="", encoding="utf-8") as stream: rows = list(csv.DictReader(stream))
    require(len(rows) == 24 and {row["scenario"] for row in rows} == {"SS-1", "SS-3", "SS-5", "SS-11"}, "aggregator rows")
    require(len(list((root / "receiver_manifests").glob("*.json"))) == 6, "receiver manifests")
    deterministic = load(root / "deterministic_reproduction.json"); require(deterministic["status"] == "PASS" and deterministic["byte_identical_compact_metrics"], "determinism")
    validate_access(load(root / "access_audit.json"))
    final = load(root / "final_verdict.json"); require(final["verdict"] in VERDICTS, "verdict")
    freeze_commit = load(root / "freeze_commit.json"); require(freeze_commit["status"] == "PASS" and freeze_commit["commit_sha"] == final["freeze_sha"] and freeze_commit["pushed_before_attack_access"] is True, "freeze commit attestation")
    require(final["stage0b_authorized"] == (final["verdict"] == "QSET_STAGE0A_PARTIAL_PRN_AGGREGATION_PASS"), "authorization")
    validate_manifest(root, load(root / "artifact_manifest_sha256.json"))
    return {"status": "PASS", "verdict": final["verdict"], "stage0b_authorized": final["stage0b_authorized"], "manifest_files": load(root / "artifact_manifest_sha256.json")["file_count"]}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", type=Path, default=ROOT / ARTIFACT_REL); args = parser.parse_args()
    try: result = validate_artifact(args.artifact)
    except (VerificationError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True)); return 2
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
