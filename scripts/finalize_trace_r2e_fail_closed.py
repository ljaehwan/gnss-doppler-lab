#!/usr/bin/env python3
"""Seal an artifact-backed R2e support failure without a performance claim."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2e_attack_support_repair"


def read(name: str):
    return json.loads((ARTIFACT / name).read_text())


def write(name: str, payload: object) -> None:
    (ARTIFACT / name).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> int:
    phase_a = read("rep3_rep4_reproduction_metrics.json")
    audits = {
        "TEXBAT.DS7": read("ds7_attack_support_audit.json"),
        "OAKBAT.OS4": read("os4_attack_support_audit.json"),
    }
    failed = [name for name, audit in audits.items() if audit["status"] != "PASS"]
    if phase_a["phase_a_status"] != "PASS" or not phase_a["phase_b_authorized"]:
        raise ValueError("this finalizer is only for an attack-support failure after Phase A PASS")
    if not failed:
        raise ValueError("attack support is complete; frozen metric evaluation is required")
    next_action = (
        "Acquire an independently validated pre-onset receiver state for "
        + " and ".join(failed)
        + " without changing frozen TRACE rules, then repeat unchanged Phase A and Phase B."
    )
    failure = {
        "schema": "gnss-doppler-lab.trace-r2e-phase-b-support-failure.v1",
        "status": "FAIL_CLOSED",
        "failure_label": "FROZEN_ATTACK_SUPPORT_INCOMPLETE_AFTER_REPAIR",
        "failed_scenarios": failed,
        "engineering_repair_status": "ATTACK_SUPPORT_INCOMPLETE",
        "phase_a_status": "PASS",
        "frozen_contract_unchanged": True,
        "scenario_evidence": audits,
        "interpretation": "At least one preregistered attack receiver path still lacks frozen pre/post-onset four-PRN support; complete Phase B metrics are unavailable.",
        "performance_claimed": False,
        "next_action": next_action,
    }
    verdict = {
        "schema": "gnss-doppler-lab.trace-r2e-final-verdict.v1",
        "verdict": "INCONCLUSIVE_INPUT_OR_RECEIVER",
        "failure_label": failure["failure_label"],
        "reason": failure["interpretation"],
        "engineering_repair_status": failure["engineering_repair_status"],
        "phase_a_passed": True,
        "phase_b_authorized": True,
        "phase_b_run": True,
        "attack_scores_computed": False,
        "attack_metrics_computed": False,
        "complete_phase_b_metrics_computed": False,
        "performance_claimed": False,
        "sci_wcl_claimable": False,
        "science_claim": None,
        "support_failure_audit": "phase_b_support_failure_audit.json",
        "recommended_next_action": next_action,
    }
    write("phase_b_support_failure_audit.json", failure)
    write("final_verdict.json", verdict)
    print(json.dumps(verdict, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
