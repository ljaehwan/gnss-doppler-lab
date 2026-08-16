#!/usr/bin/env python3
"""Fail closed when authorized Phase B lacks a usable frozen clean split."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2c_terminal_drain_repair"
SSD = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2c-terminal-drain-repair"
)
REASON = (
    "Phase A passed and Phase B was authorized, but the frozen OAKBAT.cleanStatic "
    "receiver replay supplied an empty preregistered chronological clean split. "
    "The frozen scorer stopped before attack evaluation; no attack score, normal-FPR "
    "claim, action control, AUROC/pAUC, or alarm-delay metric is available."
)
SCENARIOS = {
    "TEXBAT.cleanStatic": "texbat_cleanstatic",
    "TEXBAT.DS3": "texbat_ds3",
    "TEXBAT.DS7": "texbat_ds7",
    "OAKBAT.cleanStatic": "oakbat_cleanstatic",
    "OAKBAT.OS3": "oakbat_os3",
    "OAKBAT.OS4": "oakbat_os4",
}


def dump(name: str, value: object) -> None:
    (ARTIFACT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def main() -> int:
    phase_a = json.loads((ARTIFACT / "rep3_rep4_reproduction_metrics.json").read_text())
    if phase_a["phase_a_status"] != "PASS" or not phase_a["phase_b_authorized"]:
        raise RuntimeError("input-inconclusive finalizer requires an authorized Phase B")
    unavailable = {
        "status": "UNAVAILABLE",
        "reason": REASON,
        "metrics_computed": False,
    }
    dump(
        "clean_split_audit.json",
        {
            "schema": "gnss-doppler-lab.trace-r2c-clean-split-audit.v1",
            **unavailable,
            "failure_label": "OAKBAT_CLEAN_SPLIT_EMPTY",
            "failing_run_id": "20260816T131141Z-r2c-phase-b-fit-oakbat",
        },
    )
    dump(
        "thresholds.json",
        {"schema": "gnss-doppler-lab.trace-r2c-thresholds.v1", **unavailable, "thresholds": {}},
    )
    dump(
        "action_shuffle_metrics.json",
        {"schema": "gnss-doppler-lab.trace-r2c-action-shuffle.v1", **unavailable, "scenarios": {}},
    )
    dump(
        "physical_controls.json",
        {"schema": "gnss-doppler-lab.trace-r2c-physical-controls.v1", **unavailable, "controls": {}},
    )
    for name, columns in (
        ("scenario_metrics.csv", ["dataset", "scenario", "model", "status", "reason"]),
        ("ablation_metrics.csv", ["dataset", "scenario", "model", "status", "reason"]),
        ("bootstrap_intervals.csv", ["dataset", "scenario", "comparison", "status", "reason"]),
    ):
        with (ARTIFACT / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerow(
                {
                    columns[0]: "UNAVAILABLE",
                    columns[1]: "UNAVAILABLE",
                    columns[2]: "UNAVAILABLE",
                    "status": "UNAVAILABLE",
                    "reason": REASON,
                }
            )
    with gzip.open(ARTIFACT / "per_epoch_scores.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("status", "reason"))
        writer.writerow(("UNAVAILABLE", REASON))
    inventory = json.loads((ARTIFACT / "replay_inventory.json").read_text())
    inventory["schema"] = "gnss-doppler-lab.trace-r2c-replay-inventory.v1"
    inventory["phase_b"] = {}
    for name, slug in SCENARIOS.items():
        path = SSD / "dumps/phase_b" / slug / "rep1/manifest.json"
        manifest = json.loads(path.read_text())
        inventory["phase_b"][name] = {
            "manifest_path": str(path),
            "manifest": manifest,
            "receiver_validation_status": manifest["replay_validation"]["status"],
        }
    inventory["phase_b_decision"] = {
        "status": "INCONCLUSIVE_INPUT_OR_RECEIVER",
        "failure_label": "OAKBAT_CLEAN_SPLIT_EMPTY",
        "attack_metrics_computed": False,
    }
    dump("replay_inventory.json", inventory)
    dump("smoke_replay_results.json", phase_a)
    dump(
        "phase_b_attempt_audit.json",
        {
            "schema": "gnss-doppler-lab.trace-r2c-phase-b-attempt-audit.v1",
            "status": "INCONCLUSIVE_INPUT_OR_RECEIVER",
            "selected_receiver_run_ids": [
                "20260816T124201Z-r2c-phase-b-receiver-texbat-r1",
                "20260816T130441Z-r2c-phase-b-receiver-texbat-ds7-r2",
                "20260816T130747Z-r2c-phase-b-receiver-oakbat-r1",
            ],
            "preserved_nonselected_run_ids": [
                "20260816T122941Z-r2c-phase-b-receiver-texbat",
                "20260816T122942Z-r2c-phase-b-receiver-oakbat",
            ],
            "preserved_external_attempt_labels": [
                "rep1-validator-failed-20260816T122941Z",
                "rep1-resource-contention-20260816T122942Z",
                "rep1-support-validator-failed-20260816T124201Z",
            ],
            "scorer_failure_run_id": "20260816T131141Z-r2c-phase-b-fit-oakbat",
            "attack_performance_read_or_computed": False,
        },
    )
    dump(
        "final_verdict.json",
        {
            "schema": "gnss-doppler-lab.trace-r2c-final-verdict.v1",
            "verdict": "INCONCLUSIVE_INPUT_OR_RECEIVER",
            "phase_a_passed": True,
            "phase_b_authorized": True,
            "phase_b_run": True,
            "attack_scores_computed": False,
            "attack_metrics_computed": False,
            "failure_label": "OAKBAT_CLEAN_SPLIT_EMPTY",
            "reason": REASON,
            "normal_fpr": {"status": "UNAVAILABLE", "reason": REASON},
            "actual_action_vs_shuffled_no_action": {"status": "UNAVAILABLE", "reason": REASON},
            "next_action": (
                "Repair the frozen OAKBAT clean receiver/handoff support without changing "
                "TRACE scoring or gates, repeat Phase A, then repeat the frozen Phase B."
            ),
            "hermes_independent_verification_required": True,
        },
    )
    print(json.dumps({"status": "PASS_FAIL_CLOSED", "verdict": "INCONCLUSIVE_INPUT_OR_RECEIVER"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
