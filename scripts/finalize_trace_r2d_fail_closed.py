#!/usr/bin/env python3
"""Seal the frozen Phase-B receiver-support failure without a performance claim."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_trace_r2d_phase_b as adapter

ARTIFACT = ROOT / "artifacts/trace_stage0_r2d_oakbat_clean_support_repair"
RUN = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "trace-stage0-r2d-oakbat-clean-support-repair/runner-runs/"
    "20260816T145339Z-r2d-phase-b-finalize"
)
FROZEN = adapter.inherited.frozen


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scenario_support(name: str) -> dict[str, object]:
    pairs = FROZEN.load_pairs(name)
    blocks = FROZEN.robust_epoch_blocks(
        pairs,
        np.zeros(len(pairs.time_s), dtype=np.float64),
        block_s=FROZEN.CONFIG["block_s"],
        minimum_prns=FROZEN.CONFIG["minimum_prns"],
    )
    onset_s = FROZEN.SCENARIOS[name]["timeline"][0]
    attack_blocks = blocks[blocks["block_start_s"] >= onset_s]
    metrics = json.loads(
        (FROZEN.WORK / FROZEN.SCENARIOS[name]["slug"] / "metrics.json").read_text()
    )
    full = next(row for row in metrics if row["model"] == "TRACE Full")
    return {
        "selected_quality_common_support_pair_count": int(len(pairs.time_s)),
        "selected_unique_prn_count": int(len(np.unique(pairs.prn))),
        "selected_unique_prns": sorted(map(int, np.unique(pairs.prn))),
        "selected_time_start_s": float(pairs.time_s.min()),
        "selected_time_end_s": float(pairs.time_s.max()),
        "minimum_prns_per_block": FROZEN.CONFIG["minimum_prns"],
        "valid_block_count": int(len(blocks)),
        "attack_onset_s": onset_s,
        "post_onset_block_count": int(len(attack_blocks)),
        "frozen_metric_valid_epochs": full["valid_epochs"],
        "frozen_metric_pre_onset_fpr": full["pre_onset_fpr"],
        "frozen_metric_attack_detection_rate": full["attack_detection_rate"],
    }


def main() -> int:
    phase_a = json.loads((ARTIFACT / "rep3_rep4_reproduction_metrics.json").read_text())
    clean = json.loads((ARTIFACT / "oakbat_clean_support_audit.json").read_text())
    ds7 = scenario_support("TEXBAT.DS7")
    os4 = scenario_support("OAKBAT.OS4")
    assert phase_a["phase_a_status"] == "PASS" and phase_a["phase_b_authorized"] is True
    assert clean["status"] == "PASS"
    assert ds7["valid_block_count"] == 0 and ds7["selected_unique_prn_count"] < 4
    assert os4["post_onset_block_count"] == 0
    assert json.loads((RUN / "status.json").read_text())["status"] == "failed"

    next_action = (
        "Preregister and repair the frozen TEXBAT DS7 and OAKBAT OS4 attack "
        "receiver/handoff support without changing TRACE scoring or gates, then "
        "repeat unchanged Phase A and frozen Phase B."
    )
    failure = {
        "schema": "gnss-doppler-lab.trace-r2d-phase-b-support-failure.v1",
        "status": "FAIL_CLOSED",
        "failure_label": "FROZEN_ATTACK_SUPPORT_INCOMPLETE",
        "failure_labels": [
            "TEXBAT_DS7_FOUR_PRN_BLOCK_SUPPORT_EMPTY",
            "OAKBAT_OS4_POST_ONSET_BLOCK_SUPPORT_EMPTY",
        ],
        "engineering_repair_status": "OAKBAT_CLEAN_SUPPORT_REPAIRED",
        "phase_a_status": "PASS",
        "oakbat_clean_support_status": "PASS",
        "frozen_contract_unchanged": True,
        "scenario_evidence": {"TEXBAT.DS7": ds7, "OAKBAT.OS4": os4},
        "frozen_finalizer_failure": {
            "run_id": RUN.name,
            "exit_code": 1,
            "exception": "TypeError comparing null and finite pre_onset_fpr while computing the worst external-static FPR",
            "stderr_sha256": sha256(RUN / "stderr.log"),
        },
        "interpretation": (
            "The OAKBAT clean repair succeeded, but the complete frozen Phase-B "
            "metric set is unavailable because two unchanged attack receiver/handoff "
            "paths do not cover their preregistered evaluation windows."
        ),
        "performance_claimed": False,
        "next_action": next_action,
    }
    verdict = {
        "schema": "gnss-doppler-lab.trace-r2d-final-verdict.v1",
        "verdict": "INCONCLUSIVE_INPUT_OR_RECEIVER",
        "failure_label": failure["failure_label"],
        "failure_labels": failure["failure_labels"],
        "reason": failure["interpretation"],
        "engineering_repair_status": "OAKBAT_CLEAN_SUPPORT_REPAIRED",
        "phase_a_passed": True,
        "phase_b_authorized": True,
        "phase_b_run": True,
        "attack_scores_computed": False,
        "attack_metrics_computed": False,
        "complete_phase_b_metrics_computed": False,
        "partial_scenario_diagnostics_computed": True,
        "normal_fpr": {"status": "UNAVAILABLE_INCOMPLETE_PHASE_B"},
        "actual_action_vs_shuffled_no_action": {
            "status": "UNAVAILABLE_INCOMPLETE_PHASE_B"
        },
        "performance_claimed": False,
        "sci_wcl_claimable": False,
        "science_claim": None,
        "support_failure_audit": "phase_b_support_failure_audit.json",
        "recommended_next_action": next_action,
    }
    (ARTIFACT / "phase_b_support_failure_audit.json").write_text(
        json.dumps(failure, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (ARTIFACT / "final_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(verdict, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
