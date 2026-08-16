#!/usr/bin/env python3
"""Emit structured Phase-B NOT_AUTHORIZED artifacts after a failed Phase A."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2a_reproducibility_repair"
REASON = "Phase A did not pass every preregistered source/causal/semantic/support gate; Phase B is NOT_AUTHORIZED and no attack performance was read or computed."


def dump(name: str, value: object) -> None:
    (ARTIFACT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> int:
    phase_a = json.loads((ARTIFACT / "rep3_rep4_reproduction_metrics.json").read_text())
    if phase_a["phase_b_authorized"]:
        raise RuntimeError("Phase A PASS authorizes Phase B; refusing to emit NOT_AUTHORIZED placeholders")
    unavailable = {"status": "NOT_AUTHORIZED", "reason": REASON, "metrics_computed": False}
    dump("clean_split_audit.json", {"schema": "gnss-doppler-lab.trace-r2a-clean-split-audit.v1", **unavailable})
    dump("thresholds.json", {"schema": "gnss-doppler-lab.trace-r2a-thresholds.v1", **unavailable, "thresholds": {}})
    dump("action_shuffle_metrics.json", {"schema": "gnss-doppler-lab.trace-r2a-action-shuffle.v1", **unavailable, "scenarios": {}})
    dump("physical_controls.json", {"schema": "gnss-doppler-lab.trace-r2a-physical-controls.v1", **unavailable, "controls": {}})
    for name, columns in (
        ("scenario_metrics.csv", ["dataset", "scenario", "model", "status", "reason"]),
        ("ablation_metrics.csv", ["dataset", "scenario", "model", "status", "reason"]),
        ("bootstrap_intervals.csv", ["dataset", "scenario", "comparison", "status", "reason"]),
    ):
        with (ARTIFACT / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerow({columns[0]: "UNAVAILABLE", columns[1]: "UNAVAILABLE", columns[2]: "UNAVAILABLE", "status": "NOT_AUTHORIZED", "reason": REASON})
    with gzip.open(ARTIFACT / "per_epoch_scores.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("status", "reason"))
        writer.writerow(("NOT_AUTHORIZED", REASON))
    semantic = phase_a["semantic_reproduction_gate"]
    failure_label = (
        "NATIVE_DUMP_PHYSICAL_VALUES_NONREPRODUCIBLE"
        if semantic["status"] == "FAIL"
        else "INCONCLUSIVE_RECEIVER_REPRODUCIBILITY"
    )
    dump(
        "final_verdict.json",
        {
            "schema": "gnss-doppler-lab.trace-r2a-final-verdict.v1",
            "verdict": "INCONCLUSIVE_RECEIVER_REPRODUCIBILITY",
            "phase_a_passed": False,
            "phase_b_authorized": False,
            "phase_b_run": False,
            "attack_metrics_computed": False,
            "failure_label": failure_label,
            "reason": REASON,
            "r2_prior_verdict_preserved": "INCONCLUSIVE_INPUT_OR_RECEIVER",
            "normal_fpr": {"status": "UNAVAILABLE", "reason": "Phase B not authorized; no scientific threshold/FPR evaluation."},
            "actual_action_vs_shuffled_no_action": {"status": "UNAVAILABLE", "reason": "Phase B not authorized."},
            "next_action": "Repair or replace the receiver handoff implementation, then repeat the same frozen Phase A without changing TRACE scoring or tolerances.",
        },
    )
    print(json.dumps({"status": "PASS_STRUCTURED_NOT_AUTHORIZED", "failure_label": failure_label}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
