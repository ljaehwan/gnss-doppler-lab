#!/usr/bin/env python3
"""Finalize the fail-closed TRACE Stage-0 R1 Route-B result.

The Phase-A gate is authoritative. This script refuses to create performance
metrics when retained 20 ms row-to-row action mapping is invalid.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r1_native_cadence"
PRIOR = ROOT / "artifacts/trace_stage0_static"


def dump_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]] | None = None) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows or [])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_files() -> list[Path]:
    return sorted(path for path in ARTIFACT.rglob("*") if path.is_file() and path.name != "artifact_manifest_sha256.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-commit", required=True)
    parser.add_argument("--runner-run-id", action="append", default=[])
    args = parser.parse_args()
    mapping_path = ARTIFACT / "action_mapping_validation.json"
    cadence_path = ARTIFACT / "cadence_contract.json"
    if not mapping_path.is_file() or not cadence_path.is_file():
        raise FileNotFoundError("Phase-A cadence and mapping artifacts must exist")
    mapping = json.loads(mapping_path.read_text())
    cadence = json.loads(cadence_path.read_text())
    if mapping.get("retained_row_t_to_t_plus_1") is not False:
        raise RuntimeError("This fail-closed finalizer is only valid for Route B")
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = {
        "schema": "gnss-doppler-lab.trace-r1-config.v1",
        "actual_dt_from_sample_count": True,
        "alarm_consecutive_blocks": 3,
        "block_s": 0.5,
        "clean_split_ratios": {"train": 0.45, "covariance_validation": 0.20, "threshold_calibration": 0.15, "holdout": 0.20},
        "cn0_min_db_hz": 28.0,
        "covariance": "LedoitWolf",
        "guard_s": 5.0,
        "lock_min": 0.85,
        "minimum_prns": 4,
        "pooling": "median",
        "prompt_epsilon": 1e-12,
        "ridge_alpha": 10.0,
        "seed": 23017,
        "threshold_quantiles": [0.99, 0.995],
        "carrier_global_doppler_warp": False,
        "route_gate": "mapping before scoring",
    }
    preregistration = {
        "schema": "gnss-doppler-lab.trace-r1-preregistration.v1",
        "wording": "Configuration frozen before this TRACE-R1 evaluation.",
        "attack_data_used_for_selection": False,
        "attack_scores_computed": False,
        "normal_training": {"TEXBAT": ["cleanStatic"], "OAKBAT": ["cleanStatic"]},
        "core_scenarios": ["DS3", "DS7", "OS3", "OS4"],
        "boundary_scenarios": ["DS1", "OS1"],
        "baselines": ["A0", "A1", "A2", "A3", "A4", "Full", "B0-static-retrained"],
        "route_decision": "Route B because retained 20 ms row mapping is invalid",
        "fail_closed_verdict": "NEEDS_TRACE_SPECIFIC_RECEIVER_DUMP",
    }
    dump_json(ARTIFACT / "config.json", config)
    dump_json(ARTIFACT / "preregistration.json", preregistration)
    dump_json(ARTIFACT / "source_commit.json", {
        "schema": "gnss-doppler-lab.trace-r1-source-commit.v1",
        "base_sha": "ab8770b021020062d86fcd240ce7ecee76466072",
        "preregistration_freeze_commit": args.freeze_commit,
        "finalizer_started_at_commit": current,
        "branch": "research/trace-stage0-r1-native-cadence",
    })
    unavailable = {
        "status": "UNAVAILABLE",
        "reason": "RETAINED_20MS_ROW_ACTION_MAPPING_INVALID_AND_NATIVE_1MS_DUMP_NOT_GENERATED",
        "attack_scores_computed": False,
        "performance_claimed": False,
    }
    dump_json(ARTIFACT / "clean_split_audit.json", {**unavailable, "attack_data_excluded": True, "split_not_instantiated": True})
    dump_json(ARTIFACT / "normal_model_summary.json", {**unavailable, "model_fit": False})
    dump_json(ARTIFACT / "thresholds.json", {**unavailable, "thresholds": {}})
    dump_json(ARTIFACT / "action_shuffle_metrics.json", {**unavailable, "action_shuffle_run": False})
    prior_synthetic = json.loads((PRIOR / "synthetic_physics_metrics.json").read_text())
    dump_json(ARTIFACT / "synthetic_physics_metrics.json", {
        "status": "ANALYTIC_SANITY_ONLY",
        "r1_statistical_synthetic_run": False,
        "reason": "Route-B input gate failed before model instantiation; no measured-reference R1 predictor exists to score.",
        "preserved_prior_trace_result": prior_synthetic,
        "prior_artifact_path": "artifacts/trace_stage0_static/synthetic_physics_metrics.json",
        "prior_artifact_sha256": sha256(PRIOR / "synthetic_physics_metrics.json"),
        "performance_claimed": False,
    })
    dump_json(ARTIFACT / "physical_controls.json", {
        **unavailable,
        "controls_run": False,
        "required_controls": ["gain", "phase", "navigation_bit", "residual_noise", "single_prn", "prn_drop_add", "gap_reacquisition", "multipath_like", "normal_drift"],
        "note": "Controls require a fitted predictor and valid action/target pairs; copying scores or metadata-only changes is prohibited.",
    })
    dump_json(ARTIFACT / "b0_lineage.json", {
        **unavailable,
        "label": "B0-static-retrained",
        "rerun": False,
        "reason_baseline_not_run": "No TRACE common-support attack evaluation exists after the input gate failed.",
        "does_not_force_trace_no_go": True,
    })
    metric_fields = ("dataset", "scenario", "model", "status", "reason", "valid_blocks", "valid_prns", "roc_auc", "pauc_fpr_le_0p05", "pr_auc", "pre_onset_fpr", "delay_s")
    scenarios = [("TEXBAT", name) for name in ("DS1", "DS3", "DS7")] + [("OAKBAT", name) for name in ("OS1", "OS3", "OS4")]
    scenario_rows = [
        {"dataset": dataset, "scenario": scenario, "model": "Full", "status": "UNAVAILABLE", "reason": unavailable["reason"],
         "valid_blocks": "", "valid_prns": "", "roc_auc": "", "pauc_fpr_le_0p05": "", "pr_auc": "", "pre_onset_fpr": "", "delay_s": ""}
        for dataset, scenario in scenarios
    ]
    write_csv(ARTIFACT / "scenario_metrics.csv", metric_fields, scenario_rows)
    ablation_rows = [
        {"dataset": dataset, "scenario": scenario, "model": model, "status": "UNAVAILABLE", "reason": unavailable["reason"],
         "valid_blocks": "", "valid_prns": "", "roc_auc": "", "pauc_fpr_le_0p05": "", "pr_auc": "", "pre_onset_fpr": "", "delay_s": ""}
        for dataset, scenario in scenarios for model in ("A0", "A1", "A2", "A3", "A4", "Full", "B0-static-retrained")
    ]
    write_csv(ARTIFACT / "ablation_metrics.csv", metric_fields, ablation_rows)
    write_csv(ARTIFACT / "external_static_fpr.csv", ("dataset", "scenario", "model", "status", "fpr", "false_alarms_per_hour", "longest_false_alarm_run"))
    write_csv(ARTIFACT / "bootstrap_intervals.csv", ("comparison", "metric", "estimate", "ci_low", "ci_high", "status", "reason"))
    with gzip.open(ARTIFACT / "per_epoch_scores.csv.gz", "wt", newline="") as stream:
        stream.write("dataset,scenario,model,block_start_s,score,alarm,tracked_prn_count,status\n")
    core = {}
    for name in ("DS3", "DS7", "OS3", "OS4"):
        key = ("TEXBAT." if name.startswith("DS") else "OAKBAT.") + name
        payload = cadence["scenarios"][key]
        core[name] = {
            "post_onset_valid_20ms_blocks_ge4_prns": payload["post_onset_valid_20ms_blocks_ge4_prns"],
            "post_onset_max_valid_prns_per_0p5s_block": payload["post_onset_max_valid_prns_per_0p5s_block"],
            "scores_computed": False,
        }
    verdict = {
        "schema": "gnss-doppler-lab.trace-r1-final-verdict.v1",
        "verdict": "NEEDS_TRACE_SPECIFIC_RECEIVER_DUMP",
        "route": "B",
        "actual_20ms_action_mapping_held": False,
        "attack_scores_computed": False,
        "performance_claimed": False,
        "core_support_audit": core,
        "clean_holdout_fpr": None,
        "external_static_fpr": None,
        "full_a1_a2_b0_comparison": "UNAVAILABLE",
        "action_shuffle": "UNAVAILABLE",
        "physical_controls": "UNAVAILABLE",
        "detection_delay": None,
        "trace_hypothesis_worth_pursuing": True,
        "reason": "The hypothesis remains untested: retained post-sync rows omit 19 intermediate 1 ms receiver actions.",
        "recommended_next_action": "Generate authenticated native-1ms TRACE receiver dumps with complex nine taps, applied next-buffer actions, and loop-boundary flags, then rerun unchanged R1 from Phase A.",
    }
    dump_json(ARTIFACT / "final_verdict.json", verdict)
    dump_json(ARTIFACT / "runner_runs.json", {
        "schema": "gnss-doppler-lab.trace-r1-runner-provenance.v1",
        "runner_root": "/home/ubuntu/research-context/tasks/trace-stage0-r1-native-cadence/runner",
        "run_ids": args.runner_run_id,
    })
    readme = f"""# TRACE Stage-0 R1 native-cadence result

Configuration frozen before this TRACE-R1 evaluation.

## Outcome

Route: **B**. Final verdict: `NEEDS_TRACE_SPECIFIC_RECEIVER_DUMP`.

The retained post-synchronization rows are about 20 ms apart, but the actual
receiver loop and NCO update every 1 ms. A retained row's `code_freq_chips` and
`carrier_doppler_hz` apply only to the next 1 ms input buffer; the next retained
complex correlator follows 19 unobserved loop updates. Therefore the required
row-t action to row-(t+1) correlator mapping did not hold.

All four core scenarios have audited post-onset 20 ms row support, but it is not
causal action support and was not scored:

| Scenario | >=4-PRN post-onset blocks | Maximum PRNs/block |
|---|---:|---:|
"""
    for name, payload in core.items():
        readme += f"| {name} | {payload['post_onset_valid_20ms_blocks_ge4_prns']} | {payload['post_onset_max_valid_prns_per_0p5s_block']} |\n"
    readme += """

Clean holdout/external FPR, Full/A1/A2/B0 comparison, action shuffle, physical
controls, and detection delays are unavailable. No model, threshold, attack
score, ROC/PR result, or performance claim was produced.

Prompt referencing removes common carrier phase, so full Doppler phase rotation
was not applied to normalized taps. Doing so would double-apply a removed global
phase and still would not recover the omitted actions.

The TRACE hypothesis remains worth pursuing because this result is an input
observability failure, not evidence for or against action equivariance. The one
recommended next action is to generate authenticated native-1ms receiver dumps
with complex nine taps, applied next-buffer code/carrier actions, sample stamps,
C/N0, lock, integration interval, and loop-boundary flags, then rerun the frozen
R1 protocol from Phase A.
"""
    (ARTIFACT / "README.md").write_text(readme)
    payload = {str(path.relative_to(ARTIFACT)): sha256(path) for path in manifest_files()}
    dump_json(ARTIFACT / "artifact_manifest_sha256.json", {"schema": "gnss-doppler-lab.trace-r1-artifact-manifest.v1", "files": payload})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
