#!/usr/bin/env python3
"""Fail-closed compact verifier for CRID Stage-0 R4 Phase A."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "artifacts/crid_stage0_r4_phase_a_physical_identifiability"
BASE = "8cf594bde9c8e48bf80b5872e4ca13e1d0d13b0d"
BRANCH = "research/crid-stage0-r4-phase-a-physical-identifiability"
CONFIGS = ("C0", "C1", "C2", "C3")
R3_PREFIX = (
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "crid-stage0-r3-control-generator-foundation/"
)
REQUIRED = {
    "README.md",
    "preregistration.json",
    "execution_freeze.json",
    "source_binding.json",
    "control_input_inventory.csv",
    "replay_completion.csv",
    "clean_threshold_binding.json",
    "physical_control_metrics.csv",
    "positive_response_surface.csv",
    "phase_a_gate.json",
    "attack_access_audit.json",
    "final_verdict.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for payload in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(payload)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify_manifest(artifact: Path) -> tuple[bool, list[str], dict]:
    failures = []
    try:
        manifest = json.loads((artifact / "artifact_manifest_sha256.json").read_text())
        entries = manifest["files"]
        seen = set()
        for entry in entries:
            relative = Path(entry["path"])
            safe = not relative.is_absolute() and ".." not in relative.parts and entry["path"] not in seen
            seen.add(entry["path"])
            path = artifact / relative
            if not (
                safe
                and path.is_file()
                and path.stat().st_size == int(entry["size_bytes"])
                and sha256_file(path) == entry["sha256"]
            ):
                failures.append(f"manifest:{entry['path']}")
        if manifest.get("schema") != "gnss-doppler-lab.crid-r4-artifact-manifest.v1":
            failures.append("manifest_schema")
        if manifest.get("status") != "PASS" or manifest.get("file_count") != len(entries):
            failures.append("manifest_contract")
        if len(seen) != len(entries):
            failures.append("manifest_duplicates")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, [f"manifest_exception:{type(exc).__name__}"], {}
    return not failures, failures, manifest


def _hash_format(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def verify(artifact: Path, mode: str = "auto") -> dict:
    failures = []
    manifest_ok, manifest_failures, manifest = verify_manifest(artifact)
    failures.extend(manifest_failures)
    listed = {entry["path"] for entry in manifest.get("files", [])}
    missing = sorted(REQUIRED - listed)
    failures.extend(f"missing:{name}" for name in missing)
    try:
        prereg = json.loads((artifact / "preregistration.json").read_text())
        execution = json.loads((artifact / "execution_freeze.json").read_text())
        source = json.loads((artifact / "source_binding.json").read_text())
        threshold = json.loads((artifact / "clean_threshold_binding.json").read_text())
        gate = json.loads((artifact / "phase_a_gate.json").read_text())
        audit = json.loads((artifact / "attack_access_audit.json").read_text())
        final = json.loads((artifact / "final_verdict.json").read_text())
        inventory = read_csv(artifact / "control_input_inventory.csv")
        replays = read_csv(artifact / "replay_completion.csv")
        metrics = read_csv(artifact / "physical_control_metrics.csv")
        positives = read_csv(artifact / "positive_response_surface.csv")
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"artifact_exception:{type(exc).__name__}")
        return _result(failures, "invalid", {})

    if mode == "auto":
        mode = "freeze" if final.get("verdict") == "NOT_EVALUATED_PRE_RESULT_FREEZE" else "final"
    if prereg.get("base_r3b_sha") != BASE or prereg.get("phase_b_execution") is not False:
        failures.append("preregistration_scope")
    inputs = prereg.get("inputs", {})
    if inputs.get("controls") != 66 or inputs.get("total_replays") != 264:
        failures.append("preregistration_counts")
    if not (
        inputs.get("control_regeneration") is False
        and inputs.get("overwrite") is False
        and inputs.get("truth_modification") is False
    ):
        failures.append("input_prohibitions")
    window = prereg.get("window_contract", {})
    if not (
        window.get("positive_primary") == "active support only"
        and window.get("negative_primary") == "full replacement support"
        and window.get("full_45_second_positive_ratio_forbidden") is True
    ):
        failures.append("window_contract")
    if execution.get("branch") != BRANCH or execution.get("worker_count") != 1:
        failures.append("execution_freeze")
    for name, expected in execution.get("executable_sha256", {}).items():
        if not (ROOT / name).is_file() or sha256_file(ROOT / name) != expected:
            failures.append(f"executable:{name}")
    if source.get("status") != "PASS" or source.get("base_r3b_sha") != BASE:
        failures.append("source_binding")
    for name, binding in source.get("scientific_code", {}).items():
        if not (ROOT / name).is_file() or sha256_file(ROOT / name) != binding.get("sha256"):
            failures.append(f"science:{name}")
    for binding in source.get("artifact_manifests", {}).values():
        path = ROOT / binding["path"]
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            failures.append("upstream_manifest")
    if len(inventory) != 66 or len({row["case_id"] for row in inventory}) != 66:
        failures.append("control_inventory_count")
    if {domain: sum(row["domain"] == domain for row in inventory) for domain in ("OAK", "TEX")} != {"OAK": 33, "TEX": 33}:
        failures.append("control_inventory_domains")
    for row in inventory:
        if not all(row[name].startswith(R3_PREFIX) for name in ("control_path", "truth_json_path", "truth_epochs_path", "package_path", "existing_c0_manifest_path")):
            failures.append(f"allowlist:{row['case_id']}")
        for name in ("control_sha256", "truth_json_sha256", "truth_epochs_sha256", "package_sha256", "existing_c0_manifest_sha256"):
            if not _hash_format(row[name]):
                failures.append(f"inventory_hash:{row['case_id']}:{name}")
    if audit.get("attack_bytes_read") != 0 or audit.get("phase_b_executed") is not False:
        failures.append("attack_access")

    if mode == "freeze":
        if replays or metrics or positives:
            failures.append("pre_result_rows_present")
        if final.get("status") != "FROZEN_NOT_EXECUTED" or final.get("phase_a_executed") is not False:
            failures.append("freeze_verdict")
        if gate.get("status") != "FROZEN_NOT_EVALUATED":
            failures.append("freeze_gate")
        if threshold.get("status") != "FROZEN_PENDING_DETERMINISTIC_RECOMPUTATION":
            failures.append("freeze_threshold")
    elif mode == "final":
        if len(replays) != 264 or len({(row["case_id"], row["config"]) for row in replays}) != 264:
            failures.append("replay_count")
        for row in replays:
            if not (
                row["config"] in CONFIGS
                and row["status"] == "PASS"
                and row["exit_code"] == "0"
                and row["terminal_drain_status"] == "PASS"
                and row["native_trace_status"] == "PASS"
                and row["target_tracking_pass"] == "True"
            ):
                failures.append(f"replay:{row['case_id']}:{row['config']}")
        if len(metrics) != 66 or len(positives) != 36:
            failures.append("metric_counts")
        if any(row["technical_status"] != "PASS" for row in metrics):
            failures.append("technical_metrics")
        if {row["case_id"] for row in positives} != {row["case_id"] for row in metrics if row["family"] == "positive"}:
            failures.append("positive_surface")
        for row in metrics:
            primary = "active_truth_delay_support" if row["family"] == "positive" else "full_truth_replacement_support"
            if row["primary_window"] != primary:
                failures.append(f"metric_window:{row['case_id']}")
            if int(row["valid_full_epochs"]) <= 0 or (row["family"] == "positive" and int(row["valid_active_epochs"]) <= 0):
                failures.append(f"metric_support:{row['case_id']}")
        if threshold.get("status") != "PASS" or not all(row.get("exact_match") is True for row in threshold.get("domains", {}).values()):
            failures.append("threshold_binding")
        if gate.get("status") not in ("PASS", "FAIL"):
            failures.append("gate_status")
        allowed = {
            "INCONCLUSIVE_CRID_PHASE_A_EXECUTION_OR_PROVENANCE",
            "NO_GO_CRID_CLEAN_PHYSICAL_IDENTIFIABILITY",
            "CRID_PHASE_A_PHYSICAL_IDENTIFIABILITY_PASS",
        }
        if final.get("verdict") not in allowed or final.get("phase_a_executed") is not True:
            failures.append("final_verdict")
        if final.get("phase_b_executed") is not False or final.get("attack_bytes_read") != 0:
            failures.append("final_scope")
        if final.get("verdict") == "CRID_PHASE_A_PHYSICAL_IDENTIFIABILITY_PASS":
            if gate.get("status") != "PASS" or final.get("next_state") != "READY_FOR_CRID_PHASE_B":
                failures.append("pass_claim")
        elif final.get("next_state") != "NOT_AUTHORIZED":
            failures.append("nonpass_next_state")
        if "plots/positive_response_surface.png" not in listed:
            failures.append("plot_missing")
    else:
        failures.append("mode")
    return _result(failures, mode, {"manifest_ok": manifest_ok, "manifest_files": manifest.get("file_count", 0), "inventory_rows": len(inventory), "replay_rows": len(replays), "metric_rows": len(metrics), "positive_rows": len(positives), "verdict": final.get("verdict")})


def _result(failures: list[str], mode: str, checks: dict) -> dict:
    return {
        "schema": "gnss-doppler-lab.crid-r4-compact-verifier.v1",
        "mode": mode,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT)
    parser.add_argument("--mode", choices=("auto", "freeze", "final"), default="auto")
    args = parser.parse_args()
    result = verify(args.artifact, args.mode)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
