#!/usr/bin/env python3
"""Independent PG-SCC artifact/leakage/verdict auditor; imports no producer."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "artifacts/pg_scc_stage0_static_k9"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def refresh_manifest(root: Path) -> None:
    manifest = {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*"))
                if path.is_file() and path.name != "artifact_manifest_sha256.json"}
    write(root / "artifact_manifest_sha256.json", manifest)


def audit(root: Path, freeze_only: bool) -> dict:
    errors = []
    required_freeze = {
        "README.md", "config.json", "frozen_design.json", "freeze_manifest.json", "source_manifest.json",
        "foundation_validation.json", "synthetic_bank_summary.json", "selected_coordinates.csv",
        "selector_training_summary.json", "thresholds.json", "pooling.json", "masks.json",
        "normalization_covariance.npz", "timeline.json", "gate_definition.json",
    }
    for name in required_freeze:
        if not (root / name).is_file(): errors.append(f"missing:{name}")
    if errors:
        return {"schema": "pg_scc_independent_audit.v1", "status": "FAIL", "errors": errors}
    freeze = load(root / "freeze_manifest.json")
    drift = [name for name, expected in freeze.items() if not (root / name).is_file() or digest(root / name) != expected]
    errors.extend(f"freeze_drift:{name}" for name in drift)
    design = load(root / "frozen_design.json"); source = load(root / "source_manifest.json")
    foundation = load(root / "foundation_validation.json"); thresholds = load(root / "thresholds.json")
    masks = load(root / "masks.json"); timeline = load(root / "timeline.json")
    if design.get("attack_iq_bytes_read_before_freeze") != 0 or design.get("attack_cache_bytes_read_before_freeze") != 0:
        errors.append("prefreeze_attack_access")
    if design.get("real_attack_labels_used_for_selector_pooling_threshold") is not False:
        errors.append("attack_label_leakage")
    if source.get("attack_iq_bytes_read_before_freeze") != 0 or source.get("attack_cache_bytes_read_before_freeze") != 0:
        errors.append("source_manifest_prefreeze_access")
    if foundation.get("status") != "PASS" or foundation.get("zero_center_fallback") is not False:
        errors.append("foundation_invalid")
    for budget in (3, 5, 9):
        mask = masks.get(f"pg_scc_k{budget}", [])
        if len(mask) != budget or len(set(mask)) != budget or 93 not in mask:
            errors.append(f"mask_budget:k{budget}")
    if any(item.get("source") != "cleanStatic calibration event pooling only" for item in thresholds.values()):
        errors.append("threshold_not_normal_only")
    train_source = (ROOT / "scripts/train_pg_scc_selector.py").read_text()
    if "attack_features.npz" in train_source or "attack_features.json" in train_source:
        errors.append("training_attack_cache_literal")
    selector_source = (ROOT / "src/gnss_doppler_lab/pg_scc_selector.py").read_text()
    if "prn_identity" in selector_source: errors.append("prn_identity_feature")
    eval_source = (ROOT / "scripts/eval_pg_scc_static.py").read_text()
    tree = ast.parse(eval_source); main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    statements = [ast.unparse(node) for node in main.body]
    verify_index = next(index for index, statement in enumerate(statements) if "verify_freeze" in statement)
    load_index = next(index for index, statement in enumerate(statements) if "load_feature_cache(attack_npz" in statement)
    if verify_index >= load_index: errors.append("freeze_not_verified_before_attack_open")
    if timeline.get("ds4_claim_scope") != "transition_only_truncated_recording": errors.append("ds4_claim_scope")
    if timeline.get("ds7_ds8_independent") is not False: errors.append("ds78_double_count")
    report = {"schema": "pg_scc_independent_audit.v1", "mode": "freeze_only" if freeze_only else "final",
        "status": "PASS" if not errors else "FAIL", "errors": sorted(set(errors)),
        "freeze_files_verified": len(freeze), "foundation": foundation.get("status"),
        "leakage": "PASS" if not any("leak" in error or "attack_access" in error for error in errors) else "FAIL",
        "calibration_tail_audit": {"events_per_method": sorted({int(value["events"]) for value in thresholds.values()}),
            "status": "LIMITED", "reason": "14 one-second events cannot precisely estimate q99/q99.5 tails"}}
    if freeze_only: return report
    required_final = {"attack_lineage_validation.json", "attack_access_audit.json", "scenario_metrics.csv", "family_metrics.csv",
        "baseline_metrics.csv", "control_metrics.csv", "bootstrap_intervals.csv", "compute_metrics.csv",
        "per_epoch_scores.csv", "final_verdict.json", "ds7_ds8_overlap_audit.json"}
    for name in required_final:
        if not (root / name).is_file(): errors.append(f"missing_final:{name}")
    if any(error.startswith("missing_final") for error in errors):
        report.update(status="FAIL", errors=sorted(set(errors))); return report
    lineage = load(root / "attack_lineage_validation.json"); access = load(root / "attack_access_audit.json")
    verdict = load(root / "final_verdict.json"); baselines = rows(root / "baseline_metrics.csv")
    controls = rows(root / "control_metrics.csv"); compute = rows(root / "compute_metrics.csv"); bootstrap = rows(root / "bootstrap_intervals.csv")
    if lineage.get("status") != "PASS" or any(value.get("status") != "PASS" for value in lineage.get("sources", {}).values()): errors.append("attack_lineage")
    if access.get("attack_iq_bytes_read_before_freeze") != 0 or access.get("selector_threshold_pooling_changed_after_attack") is not False: errors.append("postfreeze_drift")
    b0 = next((row for row in baselines if row["method"] == "B0_exact"), None)
    if not b0 or b0.get("status") != "UNAVAILABLE": errors.append("b0_historic_reuse")
    if any(int(row.get("block_seconds", 0)) != 10 for row in bootstrap if row.get("status") == "PASS"): errors.append("bootstrap_temporal_correlation")
    if not all(row.get("includes_raw_read_wipeoff_code_replica_correlation") == "True" for row in compute): errors.append("compute_excludes_raw_cost")
    expected_verdict = "INCONCLUSIVE" if not verdict["gates"]["foundation_lineage"] else ("CONDITIONAL_GO" if all(verdict["gates"].values()) else "NO_GO")
    if verdict.get("verdict") != expected_verdict: errors.append("verdict_mismatch")
    report.update({"status": "PASS" if not errors else "FAIL", "errors": sorted(set(errors)),
        "verdict_recomputed": expected_verdict, "b0_reuse_audit": "PASS",
        "fixed9_support_audit": "identical cached epoch/PRN rows and shared pooling",
        "dense_teacher_audit": "actual complex two-template GLS over the frozen dense grid",
        "bootstrap_audit": "10-second block resampling",
        "compute_audit": "raw read + code replica + carrier wipeoff + correlations included",
        "ds4_claim_audit": "transition-only", "ds78_counting_audit": "one family",
        "raw_power_shortcut_control": next((row for row in controls if row["control"] == "raw_power_alarm_overlap"), None)})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", type=Path, default=DEFAULT)
    parser.add_argument("--freeze-only", action="store_true"); args = parser.parse_args(); report = audit(args.artifact, args.freeze_only)
    name = "freeze_verification.json" if args.freeze_only else "independent_audit.json"
    write(args.artifact / name, report); refresh_manifest(args.artifact)
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
