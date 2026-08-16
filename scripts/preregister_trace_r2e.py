#!/usr/bin/env python3
"""Audit the R2d terminal evidence and seal the bounded R2e repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2e_attack_support_repair"
PARENT = ROOT / "artifacts/trace_stage0_r2d_oakbat_clean_support_repair"
BASE = "e66619e31a937186e522d8566711436e24f2b99d"
SCENARIOS = {
    "TEXBAT.DS7": {
        "onset_s": 110.0,
        "source_skip_s": 90.0,
        "support_duration_s": 15.0,
        "selection_time_s": 95.0,
        "replacement_handoff": "texbat_ds7.csv",
        "parent_handoff": "texbat_ds3.csv",
    },
    "OAKBAT.OS4": {
        "onset_s": 120.0,
        "source_skip_s": 90.0,
        "support_duration_s": 15.0,
        "selection_time_s": 95.0,
        "replacement_handoff": "oakbat_os4.csv",
        "parent_handoff": "oakbat_os3.csv",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(name: str, payload: object) -> None:
    path = ARTIFACT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> int:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    if head != BASE:
        raise ValueError(f"R2e preregistration requires base {BASE}, got {head}")
    if branch != "research/trace-stage0-r2e-attack-support-repair":
        raise ValueError(f"unexpected research branch: {branch}")
    parent_verdict = json.loads((PARENT / "final_verdict.json").read_text())
    parent_failure = json.loads((PARENT / "phase_b_support_failure_audit.json").read_text())
    parent_phase_a = json.loads((PARENT / "rep3_rep4_reproduction_metrics.json").read_text())
    if parent_verdict["failure_label"] != "FROZEN_ATTACK_SUPPORT_INCOMPLETE":
        raise ValueError("parent failure label changed")
    if parent_phase_a["phase_a_status"] != "PASS":
        raise ValueError("parent Phase A was not PASS")

    diagnosis = {
        "schema": "gnss-doppler-lab.trace-r2e-attack-support-diagnosis.v1",
        "status": "PREREGISTERED_FROM_PARENT_EVIDENCE",
        "parent_commit": BASE,
        "parent_failure_label": parent_verdict["failure_label"],
        "diagnosis_label": "CROSS_SCENARIO_ATTACK_HANDOFF_STATE_INCOMPATIBLE",
        "root_cause_hypothesis": (
            "R2d mapped TEXBAT.DS7 to a DS3-derived handoff and OAKBAT.OS4 to an "
            "OS3-derived handoff. Both receiver processes exited successfully and emitted "
            "physical native rows, but the selected quality/common support ended before "
            "the frozen evaluation windows. Scenario-specific pre-onset tracking state is "
            "therefore required for these two attack recordings."
        ),
        "parent_scenario_evidence": parent_failure["scenario_evidence"],
        "not_changed": [
            "authenticated raw IQ sources or hashes",
            "R2c receiver executable and finite-source terminal drain semantics",
            "R2d OAKBAT cleanStatic handoff repair",
            "TRACE scorer, thresholds, gates, windows, tolerances, block keys, controls, and metric semantics",
        ],
        "attack_scores_read_or_computed": False,
    }
    write("diagnosis.json", diagnosis)
    write(
        "preregistration.json",
        {
            "schema": "gnss-doppler-lab.trace-r2e-preregistration.v1",
            "status": "SEALED_BEFORE_REPAIRED_SUPPORT_ACQUISITION_OR_METRIC_EVALUATION",
            "task_base_commit": BASE,
            "parent_r2d_commit": BASE,
            "scientific_objective": (
                "Repair only frozen DS7/OS4 receiver handoff support, repeat unchanged "
                "Phase A, and run the unchanged frozen Phase B if all support gates pass."
            ),
            "repair_strategy": "SCENARIO_SPECIFIC_PRE_ONSET_TARGET_ALIGNED_HANDOFF",
            "scenario_repairs": SCENARIOS,
            "support_selection_rule": (
                "For every physical native channel from the scenario's bounded pre-onset "
                "support acquisition, select the first causal native row whose absolute raw "
                "interval starts at or after 95.0 s; retain its PRN and exact action-used "
                "state; map selected rows to contiguous output channels; require at least "
                "four rows and require every selected row strictly before frozen onset."
            ),
            "selection_inputs_allowed": [
                "scenario identity",
                "native channel presence",
                "absolute raw interval start",
                "PRN identity",
                "exact action-used receiver state",
            ],
            "selection_inputs_prohibited": [
                "TRACE score",
                "quality-filter outcome",
                "four-PRN block outcome",
                "alarm or detection outcome",
                "post-onset rows",
            ],
            "phase_a_replays": [
                "TEXBAT.cleanStatic.rep3",
                "TEXBAT.cleanStatic.rep4",
                "TEXBAT.DS3.smoke",
                "OAKBAT.OS3.smoke",
            ],
            "phase_a_contract": "Inherited unchanged byte-for-byte in semantics from R2d/R2c.",
            "phase_b_scope_if_authorized": [
                "TEXBAT.cleanStatic",
                "TEXBAT.DS3",
                "TEXBAT.DS7",
                "OAKBAT.cleanStatic",
                "OAKBAT.OS3",
                "OAKBAT.OS4",
            ],
            "phase_b_authorization_rule": (
                "Unchanged Phase A must PASS and repaired DS7/OS4 support validation must "
                "show at least one frozen four-PRN block before and after onset."
            ),
            "phase_b_stop_rule": (
                "Fail closed without a performance claim if either repaired attack path "
                "lacks a pre-onset or post-onset four-PRN block under frozen support rules."
            ),
            "frozen_phase_b_scorer": {
                "implementation": "scripts/evaluate_trace_r2_phase_b.py",
                "sha256": sha256(ROOT / "scripts/evaluate_trace_r2_phase_b.py"),
                "allowed_adapter_scope": "R2e artifact and SSD paths only",
            },
            "attack_scores_read_or_computed": False,
            "performance_claimed": False,
        },
    )
    write(
        "source_commit.json",
        {
            "schema": "gnss-doppler-lab.trace-r2e-source-freeze.v1",
            "research_branch": branch,
            "task_base_commit": BASE,
            "parent_r2d_commit": BASE,
            "preregistration_commit": "TO_BE_RECORDED_AFTER_PREREGISTRATION_COMMIT",
            "support_freeze_commit": "TO_BE_RECORDED_AFTER_SUPPORT_FREEZE_COMMIT",
            "final_commit": "TO_BE_RECORDED_AT_FINALIZATION",
        },
    )
    (ARTIFACT / "README.md").write_text(
        "# TRACE Stage-0 R2e Attack Support Repair\n\n"
        "R2e repairs only the frozen TEXBAT DS7 and OAKBAT OS4 receiver/handoff "
        "support identified by R2d. The repair is preregistered before support "
        "acquisition and metric evaluation. No performance claim is available while "
        "the frozen reruns and verification remain in progress.\n"
    )
    print(json.dumps({"status": "SEALED", "artifact_root": str(ARTIFACT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
