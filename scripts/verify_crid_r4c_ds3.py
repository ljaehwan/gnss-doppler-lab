#!/usr/bin/env python3
"""Fail-closed compact verifier for CRID R4c exploratory DS3 audit."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.crid_r4c_ds3 import (  # noqa: E402
    ARTIFACT_REL,
    BASE_SHA,
    BRANCH,
    CONFIG_ORDER,
    DS3_RAW,
    END_S,
    EPOCH_FIELDS,
    EXECUTABLE_FILES,
    ONSET_S,
    PULL_OFF_S,
    REPLAY_FIELDS,
    START_S,
    THRESHOLD,
    evaluate_signal_gate,
    scenario_metrics,
    shortcut_audit,
)

REQUIRED = {
    "README.md",
    "preregistration.json",
    "execution_freeze.json",
    "freeze_commit.json",
    "source_binding.json",
    "ds3_input_inventory.json",
    "replay_completion.csv",
    "support_audit.json",
    "per_epoch_scores.csv.gz",
    "scenario_metrics.json",
    "shortcut_audit.json",
    "attack_access_audit.json",
    "final_verdict.json",
    "frozen_handoff_texbat_ds3_78p9s.csv",
}
FINAL_REQUIRED = {"alarm_timeline.png", "score_components.png"}
ALLOWED_VERDICTS = {
    "INCONCLUSIVE_TEXBAT_DS3_EXPLORATORY_EXECUTION",
    "EXPLORATORY_TEXBAT_DS3_SIGNAL_PRESENT",
    "EXPLORATORY_TEXBAT_DS3_NO_USEFUL_SIGNAL",
}
FORBIDDEN_CLAIMS = ("PHASE_B_PASS", "SPOOFING_DETECTOR_VALIDATED", "READY_FOR_DEPLOYMENT")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for payload in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(payload)
    return digest.hexdigest()


def _sha(value: object, length: int = 64) -> bool:
    return isinstance(value, str) and len(value) == length and all(char in "0123456789abcdef" for char in value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_epochs(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_manifest(artifact: Path) -> tuple[bool, list[str], dict]:
    failures = []
    try:
        manifest = json.loads((artifact / "artifact_manifest_sha256.json").read_text())
        entries = manifest["files"]
        seen = set()
        for row in entries:
            relative = Path(row["path"])
            path = artifact / relative
            safe = not relative.is_absolute() and ".." not in relative.parts and row["path"] not in seen
            seen.add(row["path"])
            if not (safe and path.is_file() and path.stat().st_size == int(row["size_bytes"]) and sha256_file(path) == row["sha256"]):
                failures.append(f"manifest:{row['path']}")
        if manifest.get("schema") != "gnss-doppler-lab.crid-r4c-artifact-manifest.v1" or manifest.get("status") != "PASS":
            failures.append("manifest_contract")
        if manifest.get("file_count") != len(entries) or len(seen) != len(entries):
            failures.append("manifest_count")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return False, [f"manifest_exception:{type(error).__name__}"], {}
    return not failures, failures, manifest


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify(artifact: Path, mode: str = "auto") -> dict[str, object]:
    failures: list[str] = []
    manifest_ok, manifest_failures, manifest = verify_manifest(artifact)
    failures.extend(manifest_failures)
    listed = {row["path"] for row in manifest.get("files", [])}
    failures.extend(f"missing:{name}" for name in sorted(REQUIRED - listed))
    try:
        prereg = json.loads((artifact / "preregistration.json").read_text())
        execution = json.loads((artifact / "execution_freeze.json").read_text())
        freeze = json.loads((artifact / "freeze_commit.json").read_text())
        source = json.loads((artifact / "source_binding.json").read_text())
        inventory = json.loads((artifact / "ds3_input_inventory.json").read_text())
        support = json.loads((artifact / "support_audit.json").read_text())
        recorded_metrics = json.loads((artifact / "scenario_metrics.json").read_text())
        recorded_shortcut = json.loads((artifact / "shortcut_audit.json").read_text())
        access = json.loads((artifact / "attack_access_audit.json").read_text())
        final = json.loads((artifact / "final_verdict.json").read_text())
        replays = read_csv(artifact / "replay_completion.csv")
        epochs = read_epochs(artifact / "per_epoch_scores.csv.gz")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _result([*failures, f"artifact_exception:{type(error).__name__}"], mode, {})
    if mode == "auto":
        mode = "freeze" if final.get("verdict") == "NOT_EVALUATED_PRE_ACCESS_FREEZE" else "final"
    if not (
        prereg.get("base_sha") == BASE_SHA
        and prereg.get("scope") == "EXPLORATORY_TEXBAT_DS3_LOCKED_MODEL_AUDIT"
        and prereg.get("r4b_phase_a_passed") is False
        and prereg.get("formal_phase_b") is False
        and prereg.get("score", {}).get("threshold") == THRESHOLD
        and prereg.get("score", {}).get("comparison") == "score > threshold"
        and prereg.get("score", {}).get("threshold_recomputation_or_recalibration") is False
    ):
        failures.append("preregistration")
    planned = prereg.get("input", {})
    if not (
        planned.get("planned_path") == str(DS3_RAW)
        and planned.get("start_s") == START_S
        and planned.get("end_s") == END_S
        and planned.get("duration_s") == 160.0
    ):
        failures.append("timeline")
    if execution.get("branch") != BRANCH or execution.get("worker_count") != 1:
        failures.append("execution_contract")
    for name in EXECUTABLE_FILES:
        if execution.get("executable_sha256", {}).get(name) != sha256_file(ROOT / name):
            failures.append(f"executable:{name}")
    if not (
        source.get("status") == "PASS"
        and source.get("base_sha") == BASE_SHA
        and source.get("r4b_phase_a_passed") is False
        and source.get("authoritative_threshold") == THRESHOLD
        and source.get("threshold_recomputation_executed") is False
        and _sha(source.get("locked_model", {}).get("model_sha256"))
    ):
        failures.append("source_binding")
    for binding in (*source.get("science_files", {}).values(), source.get("r4b_final_verdict", {}), source.get("r4b_source_binding", {}), source.get("r4b_artifact_manifest", {}), source.get("r4a_final_verdict", {}), source.get("source_handoff", {}), source.get("reanchored_handoff", {})):
        path = ROOT / binding.get("path", "")
        if not (path.is_file() and path.stat().st_size == int(binding.get("size_bytes", -1)) and sha256_file(path) == binding.get("sha256")):
            failures.append(f"source_file:{binding.get('path')}")
    text = "\n".join(path.read_text(errors="replace") for path in artifact.rglob("*") if path.is_file() and path.suffix in {".md", ".json", ".txt", ".csv"})
    if any(claim in text for claim in FORBIDDEN_CLAIMS):
        failures.append("forbidden_claim")
    forbidden = access.get("forbidden_inputs", {})
    if not (
        forbidden.get("stats") == 0
        and forbidden.get("hashes") == 0
        and forbidden.get("opens") == 0
        and forbidden.get("mmaps") == 0
        and forbidden.get("bytes_read") == 0
        and access.get("phase_b_executed") is False
    ):
        failures.append("forbidden_access")
    if mode == "freeze":
        if replays or epochs:
            failures.append("pre_access_rows")
        if inventory.get("status") != "FROZEN_NOT_ACCESSED" or any(inventory.get(name) not in (0, None) for name in ("size_bytes", "sha256", "payload_stat_count", "payload_open_count", "payload_hash_count", "payload_mmap_count", "payload_bytes_read")):
            failures.append("pre_access_inventory")
        if access.get("status") != "PASS_PRE_ACCESS_ZERO" or any(access.get("allowed_ds3", {}).get(name) != 0 for name in ("stats", "hashes", "opens", "mmaps", "bytes_read")):
            failures.append("pre_access_audit")
        if final.get("status") != "FROZEN_NOT_EXECUTED" or final.get("verdict") != "NOT_EVALUATED_PRE_ACCESS_FREEZE":
            failures.append("freeze_verdict")
        if freeze.get("status") != "PENDING_PUSH" or freeze.get("freeze_sha") is not None:
            failures.append("freeze_commit")
    elif mode == "final":
        failures.extend(f"missing:{name}" for name in sorted(FINAL_REQUIRED - listed))
        if not (
            inventory.get("status") == "PASS"
            and inventory.get("path") == str(DS3_RAW)
            and _sha(inventory.get("sha256"))
            and int(inventory.get("size_bytes", 0)) > 0
            and inventory.get("access_after_pushed_freeze_only") is True
        ):
            failures.append("input_inventory")
        if len(replays) != 4 or [row.get("config") for row in replays] != list(CONFIG_ORDER):
            failures.append("replay_count_order")
        for row in replays:
            if not (
                row.get("status") == "PASS"
                and row.get("exit_code") == "0"
                and row.get("terminal_drain_status") == "PASS"
                and row.get("native_trace_status") == "PASS"
                and row.get("target_tracking_pass") == "True"
                and int(row.get("dump_count", 0)) == 11
                and all(_sha(row.get(name)) for name in ("input_sha256", "receiver_sha256", "config_sha256", "handoff_sha256", "output_set_sha256", "manifest_sha256"))
            ):
                failures.append(f"replay:{row.get('config')}")
        if not epochs or tuple(epochs[0]) != EPOCH_FIELDS:
            failures.append("epoch_contract")
        parsed = []
        for row in epochs:
            try:
                parsed.append(
                    {
                        **row,
                        "sample": int(row["sample"]),
                        "time_s": float(row["time_s"]),
                        "score": float(row["score"]),
                        "alarm": int(row["alarm"]),
                        "label": int(row["label"]),
                        "prn_count": int(row["prn_count"]),
                        "config_count": int(row["config_count"]),
                        "cn0_median_db_hz": float(row["cn0_median_db_hz"]),
                        "lock_median": float(row["lock_median"]),
                        "tracked_prn_count_min_config": float(row["tracked_prn_count_min_config"]),
                        "tracked_prn_count_median_config": float(row["tracked_prn_count_median_config"]),
                        "h0_loglike": float(row["h0_loglike"]),
                        "h1_loglike": float(row["h1_loglike"]),
                        "h1_improvement": float(row["h1_improvement"]),
                        "penalty": float(row["penalty"]),
                        "configuration_disagreement": float(row["configuration_disagreement"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                failures.append("epoch_parse")
                break
        if parsed and not all(row["alarm"] == int(row["score"] > THRESHOLD) and row["config_count"] == 4 and row["prn_count"] >= 4 for row in parsed):
            failures.append("epoch_score_support")
        if parsed:
            try:
                metrics = scenario_metrics(parsed)
                shortcut = shortcut_audit(parsed)
                gate = evaluate_signal_gate(metrics, shortcut, True)
                if recorded_metrics.get("results") != metrics or recorded_metrics.get("exploratory_signal_gate") != gate:
                    failures.append("metric_recomputation")
                if recorded_shortcut != shortcut:
                    failures.append("shortcut_recomputation")
            except (KeyError, TypeError, ValueError):
                failures.append("metric_exception")
        if support.get("status") != "PASS" or support.get("deterministic_exact_match") is not True or support.get("minimum_configurations") != 4 or support.get("minimum_common_prns") != 4:
            failures.append("support")
        if not (
            final.get("verdict") in ALLOWED_VERDICTS
            and final.get("exploratory_only") is True
            and final.get("formal_phase_b") is False
            and final.get("phase_b_executed") is False
            and final.get("threshold") == THRESHOLD
            and final.get("threshold_recomputation_executed") is False
            and final.get("post_access_code_threshold_score_feature_model_window_gate_changes") is False
            and final.get("forbidden_attack_access_bytes") == 0
        ):
            failures.append("final_verdict")
        if _sha(freeze.get("freeze_sha"), 40) is False or freeze.get("status") != "PASS" or freeze.get("ahead") != 0 or freeze.get("behind") != 0:
            failures.append("freeze_commit")
    else:
        failures.append("mode")
    return _result(
        failures,
        mode,
        {
            "manifest_ok": manifest_ok,
            "manifest_files": manifest.get("file_count", 0),
            "replay_rows": len(replays),
            "epoch_rows": len(epochs),
            "verdict": final.get("verdict"),
        },
    )


def _result(failures: list[str], mode: str, checks: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "gnss-doppler-lab.crid-r4c-compact-verifier.v1",
        "mode": mode,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=ROOT / ARTIFACT_REL)
    parser.add_argument("--mode", choices=("auto", "freeze", "final"), default="auto")
    args = parser.parse_args()
    result = verify(args.artifact, args.mode)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
