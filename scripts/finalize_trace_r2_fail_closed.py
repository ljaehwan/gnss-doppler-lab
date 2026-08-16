#!/usr/bin/env python3
"""Create the complete TRACE-R2 fail-closed artifact bundle without metrics."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2_native_1ms_dump"


def dump_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--reason")
    args = parser.parse_args()
    smoke_path = ARTIFACT / "smoke_replay_results.json"
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
    labels = sorted(set(args.label or smoke.get("failure_labels", [])))
    if not labels:
        raise ValueError("at least one implementation/source-binding failure label is required")
    allowed = {
        "NATIVE_1MS_DUMP_IMPLEMENTATION_INVALID",
        "ACTION_MAPPING_UNRESOLVED",
        "INSUFFICIENT_MULTI_PRN_SUPPORT",
        "RAW_SOURCE_BINDING_FAILED",
    }
    if not set(labels) <= allowed:
        raise ValueError(f"invalid fail-closed labels: {sorted(set(labels) - allowed)}")
    reason = args.reason or "Phase A did not authenticate all required native receiver inputs."
    unavailable = {
        "status": "UNAVAILABLE",
        "reason": reason,
        "failure_labels": labels,
        "attack_scores_computed": False,
        "performance_claimed": False,
    }
    dump_json(
        ARTIFACT / "clean_split_audit.json",
        {**unavailable, "split_instantiated": False, "raw_sample_or_byte_overlap": None, "attack_data_used": False},
    )
    dump_json(ARTIFACT / "thresholds.json", {**unavailable, "thresholds": {}})
    dump_json(ARTIFACT / "action_shuffle_metrics.json", {**unavailable, "action_shuffle_run": False})
    dump_json(ARTIFACT / "physical_controls.json", {**unavailable, "controls_run": False})
    metric_fields = (
        "dataset", "scenario", "model", "status", "reason", "valid_epochs", "valid_prns",
        "roc_auc", "pauc_fpr_le_0p05", "pr_auc", "clean_holdout_fpr", "pre_onset_fpr",
        "attack_detection_rate", "transition_detection_rate", "established_detection_rate",
        "onset_delay_s", "pull_off_delay_s", "persistent_alarm_ratio",
    )
    scenarios = (("TEXBAT", "DS3"), ("TEXBAT", "DS7"), ("OAKBAT", "OS3"), ("OAKBAT", "OS4"))
    models = (
        "TRACE Full", "no-action predictor", "zero-action counterfactual",
        "wrong shifted-action negative control", "shuffled-action TRACE", "action norm only",
        "complex residual norm only", "fixed complex 9-tap detector", "B0 exact",
    )
    rows = [
        {
            **{field: "" for field in metric_fields},
            "dataset": dataset,
            "scenario": scenario,
            "model": model,
            "status": "UNAVAILABLE",
            "reason": reason,
        }
        for dataset, scenario in scenarios
        for model in models
    ]
    write_csv(ARTIFACT / "scenario_metrics.csv", metric_fields, [row for row in rows if row["model"] == "TRACE Full"])
    write_csv(ARTIFACT / "ablation_metrics.csv", metric_fields, rows)
    write_csv(
        ARTIFACT / "external_static_fpr.csv",
        ("dataset", "scenario", "model", "status", "fpr", "reason"),
    )
    write_csv(
        ARTIFACT / "bootstrap_intervals.csv",
        ("dataset", "scenario", "comparison", "metric", "estimate", "ci_low", "ci_high", "status", "reason"),
    )
    with gzip.open(ARTIFACT / "per_epoch_scores.csv.gz", "wt", newline="") as stream:
        stream.write("dataset,scenario,model,receiver_time_s,raw_sample,score,alarm,tracked_prn_count,status,reason\n")
    with gzip.open(ARTIFACT / "per_prn_action_response.csv.gz", "wt", newline="") as stream:
        stream.write("dataset,scenario,channel,prn,loop_sequence,receiver_time_s,action_norm,response_norm,score,status,reason\n")
    verdict = {
        "schema": "gnss-doppler-lab.trace-r2-final-verdict.v1",
        "verdict": "INCONCLUSIVE_INPUT_OR_RECEIVER",
        "failure_labels": labels,
        "reason": reason,
        "phase_a_passed": False,
        "phase_b_run": False,
        "attack_scores_computed": False,
        "performance_claimed": False,
        "science_claim": "No claim for or against the TRACE physical hypothesis is supported.",
        "sci_wcl_claimable": False,
        "recommended_next_action": "Resolve the named receiver/source-input failure and rerun Phase A from the frozen preregistration.",
    }
    dump_json(ARTIFACT / "final_verdict.json", verdict)
    readme = (ARTIFACT / "README.md").read_text()
    readme += f"""

## Final outcome

Verdict: `INCONCLUSIVE_INPUT_OR_RECEIVER`.

Failure labels: `{', '.join(labels)}`.

{reason} Phase B and frozen TRACE-R1 scoring were not run. All performance,
FPR, delay, comparison, shuffle, bootstrap, and physical-control outputs are
explicitly unavailable; no placeholder performance plot was created.
"""
    (ARTIFACT / "README.md").write_text(readme)
    files = {
        str(path.relative_to(ARTIFACT)): sha256(path)
        for path in sorted(ARTIFACT.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest_sha256.json"
    }
    dump_json(
        ARTIFACT / "artifact_manifest_sha256.json",
        {"schema": "gnss-doppler-lab.trace-r2-artifact-manifest.v1", "files": files},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
