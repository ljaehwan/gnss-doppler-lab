#!/usr/bin/env python3
"""Fail-closed compact verifier for CRID Stage-0 R4b Phase A."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.crid_r4b_phase_a import (  # noqa: E402
    AUTHORITATIVE_THRESHOLDS,
    BASE_SHA,
    BRANCH,
    R4A_ART,
    R4B_ART,
    R4B_SSD,
    R4_ART,
    R4_FINAL_SHA,
)


DEFAULT = ROOT / R4B_ART
CONFIGS = ("C0", "C1", "C2", "C3")
REQUIRED = {
    "README.md",
    "preregistration.json",
    "execution_freeze.json",
    "freeze_commit.json",
    "source_binding.json",
    "control_input_inventory.csv",
    "replay_completion.csv",
    "clean_threshold_binding.json",
    "physical_control_metrics.csv",
    "positive_response_surface.csv",
    "phase_a_gate.json",
    "support_audit.json",
    "deterministic_reproduction.json",
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


def _hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def verify_manifest(artifact: Path) -> tuple[bool, list[str], dict]:
    failures: list[str] = []
    try:
        manifest = json.loads((artifact / "artifact_manifest_sha256.json").read_text())
        entries = manifest["files"]
        seen: set[str] = set()
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
        if manifest.get("schema") != "gnss-doppler-lab.crid-r4b-artifact-manifest.v1":
            failures.append("manifest_schema")
        if manifest.get("status") != "PASS" or manifest.get("file_count") != len(entries):
            failures.append("manifest_contract")
        if len(seen) != len(entries):
            failures.append("manifest_duplicates")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, [f"manifest_exception:{type(exc).__name__}"], {}
    return not failures, failures, manifest


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify(artifact: Path, mode: str = "auto") -> dict:
    failures: list[str] = []
    manifest_ok, manifest_failures, manifest = verify_manifest(artifact)
    failures.extend(manifest_failures)
    listed = {entry["path"] for entry in manifest.get("files", [])}
    failures.extend(f"missing:{name}" for name in sorted(REQUIRED - listed))
    try:
        prereg = json.loads((artifact / "preregistration.json").read_text())
        execution = json.loads((artifact / "execution_freeze.json").read_text())
        freeze = json.loads((artifact / "freeze_commit.json").read_text())
        source = json.loads((artifact / "source_binding.json").read_text())
        threshold = json.loads((artifact / "clean_threshold_binding.json").read_text())
        gate = json.loads((artifact / "phase_a_gate.json").read_text())
        support = json.loads((artifact / "support_audit.json").read_text())
        reproduction = json.loads((artifact / "deterministic_reproduction.json").read_text())
        audit = json.loads((artifact / "attack_access_audit.json").read_text())
        final = json.loads((artifact / "final_verdict.json").read_text())
        inventory = read_csv(artifact / "control_input_inventory.csv")
        replays = read_csv(artifact / "replay_completion.csv")
        metrics = read_csv(artifact / "physical_control_metrics.csv")
        positives = read_csv(artifact / "positive_response_surface.csv")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _result([*failures, f"artifact_exception:{type(exc).__name__}"], mode, {})
    if mode == "auto":
        mode = "freeze" if final.get("verdict") == "NOT_EVALUATED_PRE_RESULT_FREEZE" else "final"
    if prereg.get("base_sha") != BASE_SHA or prereg.get("phase_b_execution") is not False:
        failures.append("preregistration_scope")
    inputs = prereg.get("inputs", {})
    if inputs.get("controls") != 66 or inputs.get("total_replays") != 264 or inputs.get("r4_inventory_byte_identity_required") is not True:
        failures.append("input_contract")
    score = prereg.get("score_contract", {})
    if not (
        score.get("authoritative_thresholds") == AUTHORITATIVE_THRESHOLDS
        and score.get("comparison") == "score > authoritative_threshold"
        and score.get("exact_float_recomputation_gate_reused") is False
        and score.get("threshold_reestimated_or_replaced") is False
        and score.get("minimum_configurations_per_epoch") == 4
        and score.get("minimum_common_prns_per_epoch") == 4
    ):
        failures.append("score_contract")
    window = prereg.get("window_contract", {})
    if not (
        window.get("positive_primary") == "active truth delay support only"
        and window.get("negative_primary") == "full truth replacement support"
        and window.get("positive_full_45_second_primary_forbidden") is True
    ):
        failures.append("window_contract")
    if execution.get("branch") != BRANCH or execution.get("worker_count") != 1 or execution.get("output_root") != str(R4B_SSD):
        failures.append("execution_contract")
    for name, expected in execution.get("executable_sha256", {}).items():
        if not (ROOT / name).is_file() or sha256_file(ROOT / name) != expected:
            failures.append(f"executable:{name}")
    if source.get("status") != "PASS" or source.get("base_sha") != BASE_SHA or source.get("r4_inventory_byte_identical") is not True:
        failures.append("source_binding")
    try:
        if source.get("r4_tree_sha") != source.get("r4_tree_sha_at_base"):
            failures.append("r4_tree_preservation")
        if _git("rev-parse", f"{R4_FINAL_SHA}:{R4_ART}") != _git("rev-parse", f"HEAD:{R4_ART}"):
            failures.append("r4_tree_current")
    except subprocess.CalledProcessError:
        failures.append("r4_tree_exception")
    for binding in (*source.get("r4_bindings", {}).values(), *source.get("r4a_bindings", {}).values()):
        path = ROOT / binding["path"]
        if not path.is_file() or path.stat().st_size != int(binding["size_bytes"]) or sha256_file(path) != binding["sha256"]:
            failures.append(f"upstream_binding:{binding.get('path')}")
    if len(inventory) != 66 or len({row["case_id"] for row in inventory}) != 66:
        failures.append("inventory_count")
    counts = {(domain, family): sum(row["domain"] == domain and row["family"] == family for row in inventory) for domain in ("OAK", "TEX") for family in ("positive", "negative")}
    if counts != {("OAK", "positive"): 18, ("OAK", "negative"): 15, ("TEX", "positive"): 18, ("TEX", "negative"): 15}:
        failures.append("inventory_distribution")
    r4_inventory = ROOT / R4_ART / "control_input_inventory.csv"
    if not r4_inventory.is_file() or sha256_file(r4_inventory) != sha256_file(artifact / "control_input_inventory.csv"):
        failures.append("inventory_byte_identity")
    for row in inventory:
        for name in ("control_sha256", "truth_json_sha256", "truth_epochs_sha256", "package_sha256", "existing_c0_manifest_sha256"):
            if not _hash(row.get(name)):
                failures.append(f"inventory_hash:{row['case_id']}:{name}")
    if not (
        threshold.get("comparison") == "score > authoritative_threshold"
        and threshold.get("threshold_recomputation_executed") is False
        and all(threshold.get("domains", {}).get(domain, {}).get("authoritative_threshold") == value for domain, value in AUTHORITATIVE_THRESHOLDS.items())
    ):
        failures.append("threshold_binding")
    if not (
        audit.get("attack_stats") == 0
        and audit.get("attack_hashes") == 0
        and audit.get("attack_opens") == 0
        and audit.get("attack_mmaps") == 0
        and audit.get("attack_bytes_read") == 0
        and audit.get("phase_b_executed") is False
    ):
        failures.append("attack_access")
    if mode == "freeze":
        if replays or metrics or positives:
            failures.append("pre_result_rows_present")
        if final.get("status") != "FROZEN_NOT_EXECUTED" or final.get("phase_a_executed") is not False:
            failures.append("freeze_verdict")
        if gate.get("status") != "FROZEN_NOT_EVALUATED" or support.get("status") != "FROZEN_NOT_EVALUATED" or reproduction.get("status") != "FROZEN_NOT_EVALUATED":
            failures.append("freeze_pending_state")
        if freeze.get("status") != "PENDING_PUSH" or freeze.get("freeze_sha") is not None:
            failures.append("freeze_commit_pending")
    elif mode == "final":
        if len(replays) != 264 or len({(row["case_id"], row["config"]) for row in replays}) != 264:
            failures.append("replay_count")
        for row in replays:
            if not (
                row.get("config") in CONFIGS
                and row.get("status") == "PASS"
                and row.get("exit_code") == "0"
                and row.get("terminal_drain_status") == "PASS"
                and row.get("native_trace_status") == "PASS"
                and row.get("target_tracking_pass") == "True"
                and row.get("common_support_status") == "PASS"
                and int(row.get("common_valid_epochs", "0")) > 0
                and all(_hash(row.get(name)) for name in ("input_sha256", "config_sha256", "receiver_sha256", "output_set_sha256", "manifest_sha256"))
            ):
                failures.append(f"replay:{row.get('case_id')}:{row.get('config')}")
        if len(metrics) != 66 or len(positives) != 36:
            failures.append("metric_counts")
        positive_ids = {row["case_id"] for row in metrics if row.get("family") == "positive"}
        if {row["case_id"] for row in positives} != positive_ids:
            failures.append("positive_surface")
        for row in metrics:
            expected_window = "active_truth_delay_support" if row["family"] == "positive" else "full_truth_replacement_support"
            if not (
                row.get("technical_status") == "PASS"
                and row.get("primary_window") == expected_window
                and int(row.get("valid_full_epochs", "0")) > 0
                and (row["family"] != "positive" or int(row.get("valid_active_epochs", "0")) > 0)
                and float(row["threshold_q99"]) == AUTHORITATIVE_THRESHOLDS[row["domain"]]
            ):
                failures.append(f"metric:{row.get('case_id')}")
        if support.get("status") != "PASS" or support.get("case_count") != 66 or support.get("zero_support_cases") != 0:
            failures.append("support_audit")
        cases = reproduction.get("cases", [])
        if not (
            reproduction.get("status") == "PASS"
            and reproduction.get("case_count") == 66
            and len(cases) == 66
            and all(row.get("score_sha256_first") == row.get("score_sha256_second") and _hash(row.get("score_sha256_first")) for row in cases)
        ):
            failures.append("deterministic_reproduction")
        if threshold.get("status") != "PASS" or gate.get("status") not in ("PASS", "FAIL"):
            failures.append("final_gate_binding")
        allowed = {
            "INCONCLUSIVE_CRID_PHASE_A_EXECUTION_OR_PROVENANCE",
            "NO_GO_CRID_CLEAN_PHYSICAL_IDENTIFIABILITY",
            "CRID_PHASE_A_PHYSICAL_IDENTIFIABILITY_PASS",
        }
        if not (
            final.get("verdict") in allowed
            and final.get("phase_a_executed") is True
            and final.get("phase_a_replays") == 264
            and final.get("phase_b_executed") is False
            and final.get("attack_bytes_read") == 0
            and final.get("threshold_recomputation_executed") is False
            and final.get("post_result_code_threshold_window_score_gate_changes") is False
        ):
            failures.append("final_verdict")
        if final.get("verdict") == "CRID_PHASE_A_PHYSICAL_IDENTIFIABILITY_PASS":
            if gate.get("status") != "PASS" or final.get("next_state") != "READY_FOR_CRID_PHASE_B":
                failures.append("pass_claim")
        elif final.get("next_state") != "NOT_AUTHORIZED":
            failures.append("nonpass_claim")
        if freeze.get("status") != "PASS" or not _hash(freeze.get("freeze_sha")) or freeze.get("ahead") != 0 or freeze.get("behind") != 0:
            failures.append("freeze_commit")
        if "plots/positive_response_surface.png" not in listed:
            failures.append("plot_missing")
    else:
        failures.append("mode")
    return _result(
        failures,
        mode,
        {
            "manifest_ok": manifest_ok,
            "manifest_files": manifest.get("file_count", 0),
            "inventory_rows": len(inventory),
            "replay_rows": len(replays),
            "metric_rows": len(metrics),
            "positive_rows": len(positives),
            "verdict": final.get("verdict"),
        },
    )


def _result(failures: list[str], mode: str, checks: dict) -> dict:
    return {
        "schema": "gnss-doppler-lab.crid-r4b-compact-verifier.v1",
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
