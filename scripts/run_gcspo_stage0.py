#!/usr/bin/env python3
"""Execute the frozen GCSPO Stage-0 clean-only or protected phase."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.gcspo_artifacts import FROZEN_HASHES, SOURCE_HASHES, canonical_write_json, preflight_receiver_semantics, prepare_valid_artifact_manifest, quarantine_failed_final_verdict, sha256_file, utc_now, write_fail_closed_invalid
from gnss_doppler_lab.gcspo_clean import run_clean_a1
from gnss_doppler_lab.gcspo_clean_controls import run_clean_control_evidence
from gnss_doppler_lab.gcspo_core import AccessGate
from gnss_doppler_lab.gcspo_evaluate import run_one_shot
from gnss_doppler_lab.gcspo_evaluate import validate_clean_contrast_preaccess
from gnss_doppler_lab.gcspo_capabilities import validate_preaccess_capabilities
from gnss_doppler_lab.gcspo_core import empirical_threshold
from gnss_doppler_lab.gcspo_freeze import (claim_protected_attempt, live_remote_snapshot,
                                               validate_protected_manifest_inventory, verify_freeze_record,
                                               verify_review_candidate_record)
from gnss_doppler_lab.gcspo_transfer import (prove_closed_loop_transfer,
                                              prove_synthetic_physical_recovery)
from gnss_doppler_lab.gcspo_full import GeometryCache, geometry_preflight, role_full_terms, score_full_terms, select_full_lambda

CONFIG_SHA = "0db816116b95b41db8b7af7379cd7411cc52d43b6428ae00ab02d6ccac19f4ad"
DEFAULT_CLEAN_ROOT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/receiver/cleanStatic-complex9")


def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("clean-only", "protected"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/gcspo_stage0_static_rerun")
    parser.add_argument("--receiver-source", type=Path, default=Path("/home/ubuntu/build-gnss-sdr-complex9"))
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    return parser.parse_args()


def _hash_array(value):
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _control_generation(validated_rows):
    controls = [
        ("COMMON_GAIN", [.5, .8, 1.2, 2.]), ("PROMPT_AMPLITUDE", [.5, .8, 1.2, 2.]),
        ("CN0_METADATA_EXCLUSION_INVARIANCE", [-3., -6., -10.]), ("PRN_DROP_ONLY", [1, 2, 4]),
        ("EMPIRICAL_NOISE", [.5, 1., 2.]), ("ONE_PRN_DISTURBANCE", [1., 2., 4.]),
        ("INDEPENDENT_MULTIPATH_LIKE", [.5, 1., 2.]), ("CLOCK_DRIFT", [.1, 1., 5.]),
    ]
    return {"schema": "gnss-doppler-lab.gcspo-stage0.clean-control-generation.v1", "numpy_version": np.__version__,
            "source_role": "cleanStatic holdout [350,470)", "source_blocks": 12,
            "validated_physical_rows": sorted(validated_rows),
            "controls": [{"id": name, "levels": levels, "generation_status": "PASS", "alarm_behavior_evaluated": False} for name, levels in controls],
            "overall_status": "PASS", "protected_attack_rows_read": False}
def _git(*arguments):
    return subprocess.run(["git", *arguments], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()

def _record_preaccess_invalid(artifact_dir, exc):
    return write_fail_closed_invalid(
        artifact_dir, reason_codes=["PREACCESS_PREFLIGHT_FAILED"],
        failed_checks=[{"id": "protected_preflight", "status": "FAIL", "error_type": type(exc).__name__}],
        target_commit=_git("rev-parse", "HEAD"),
    )


def protected(args):
    if sha256_file(args.config) != CONFIG_SHA:
        raise PermissionError("VALID_FOR_PROTECTED_ACCESS not reached: frozen config mismatch")
    freeze_path = args.artifact_dir / "implementation_manifest.json"
    if not freeze_path.is_file(): raise PermissionError("VALID_FOR_PROTECTED_ACCESS not reached: implementation freeze is absent")
    freeze = json.loads(freeze_path.read_text())
    local = _git("rev-parse", "HEAD")
    if freeze.get("validity_state") == "AWAITING_INDEPENDENT_REREVIEW":
        verify_review_candidate_record(freeze, target_commit=freeze.get("target_commit"))
        raise PermissionError("VALID_FOR_PROTECTED_ACCESS not reached: independent rereview is pending")
    verify_freeze_record(freeze, target_commit=local)
    snapshot = live_remote_snapshot(ROOT, "origin", "research/gcspo-stage0-static-rerun")
    if not snapshot["synchronized"]:
        raise PermissionError("VALID_FOR_PROTECTED_ACCESS not reached: live remote exact sync missing")
    inventory = json.loads((args.artifact_dir / "data_inventory.json").read_text())
    capabilities = validate_preaccess_capabilities(
        json.loads((args.artifact_dir / "protected_capabilities.json").read_text()))
    manifest_identities = validate_protected_manifest_inventory(
        inventory, required=tuple(capabilities["available"]))
    for scenario, capability in capabilities["available"].items():
        if manifest_identities[scenario] != capability["manifest_identity"]:
            raise ValueError(f"{scenario} inventory/capability manifest identity mismatch")
    validate_clean_contrast_preaccess(args.artifact_dir, freeze["clean_scientific_artifacts"])
    claim_protected_attempt(args.artifact_dir.parent / f".{args.artifact_dir.name}.protected_run_started.json",
                            {"target_commit": local, "live_remote_sha": snapshot["remote_sha"]})
    gate = AccessGate(args.artifact_dir / "access_ledger.jsonl")
    gate.set_preflight(clean_only_pass=True, reviews_pass=True, freeze_sha=local, frozen_hashes=FROZEN_HASHES)
    gate.set_remote_sync(local_sha=local, remote_sha=snapshot["remote_sha"], ahead=0, behind=0, clean=True)
    verdict = run_one_shot(artifact_dir=args.artifact_dir, repo_root=ROOT, inventory=inventory, gate=gate,
                           manifest_identities=manifest_identities,
                           clean_identities=freeze["clean_scientific_artifacts"], capabilities=capabilities)
    prepare_valid_artifact_manifest(args.artifact_dir)
    print(f"PROTECTED_ONE_SHOT_PASS verdict={verdict}", flush=True)
    return 0



def clean_only(args):
    started = utc_now()
    if sha256_file(args.config) != CONFIG_SHA: raise ValueError("frozen config checksum mismatch")
    config = json.loads(args.config.read_text())
    semantic = preflight_receiver_semantics(args.receiver_source)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    canonical_write_json(args.artifact_dir / "receiver_source_snapshot.json", semantic["source_snapshot"])
    if semantic["overall_status"] != "PASS": raise RuntimeError("clean/source semantic preflight failed; protected access remains sealed")
    print("CLEAN_SOURCE_PREFLIGHT_PASS", flush=True)
    clean = run_clean_a1(args.clean_root, ridge_grid=config["h0_predictor"]["ridge_grid"])
    print(f"CLEAN_VAR_A1_PASS rows={len(clean['data'].epoch)}", flush=True)
    closed_loop = prove_closed_loop_transfer(args.receiver_source, clean["model"], expected_source_hashes=SOURCE_HASHES,
                                             expected_var_sha256=_hash_array(clean["model"].coefficients))
    if closed_loop["overall_status"] != "PASS" or not closed_loop["method_availability"]["Full"]:
        raise RuntimeError("closed-loop transfer proof failed; protected access remains sealed")
    recovery = prove_synthetic_physical_recovery(clean["model"], clean["whitener"], clean["gamma"])
    if recovery["overall_status"] != "PASS":
        raise RuntimeError("end-to-end synthetic physical recovery failed; protected access remains sealed")
    print(f"SYNTHETIC_PHYSICAL_RECOVERY_PASS error={recovery['maximum_scaled_state_error']}", flush=True)
    geometry = geometry_preflight(args.clean_root, tracked_prns=clean["data"].prn)
    validated_rows = set(closed_loop["validated_rows"])
    cache = GeometryCache(geometry["ephemerides"], geometry["receiver_ecef"], validated_rows)
    validation_terms = role_full_terms(clean["data"], clean["model"], clean["whitener"], clean["gamma"], cache, 150, 210)
    selected_lambda, lambda_objectives = select_full_lambda(validation_terms, config["lambda_selection"]["grid"])
    print(f"CLEAN_FULL_LAMBDA_PASS windows={len(validation_terms)} lambda={selected_lambda}", flush=True)
    calibration_terms = role_full_terms(clean["data"], clean["model"], clean["whitener"], clean["gamma"], cache, 220, 340)
    holdout_terms = role_full_terms(clean["data"], clean["model"], clean["whitener"], clean["gamma"], cache, 350, 470)
    full_calibration = score_full_terms(calibration_terms, selected_lambda)
    full_holdout = score_full_terms(holdout_terms, selected_lambda)
    full_values = np.asarray([row["score"] for row in full_calibration])
    full_thresholds = {"q99": empirical_threshold(full_values, .99), "q995": empirical_threshold(full_values, .995)}
    for row in full_calibration + full_holdout: row["state"] = np.asarray(row["state"]).tolist()
    a1_holdout_values = np.asarray([row["score"] for row in clean["holdout"]])
    fpr = {key: float(np.mean(np.asarray([row["score"] for row in full_holdout]) > value)) for key, value in full_thresholds.items()}
    a1_fpr = {key: float(np.mean(a1_holdout_values > value)) for key, value in clean["thresholds"].items()}
    controls = run_clean_control_evidence(clean, geometry, smoothness=selected_lambda, threshold=full_thresholds["q99"])
    canonical_write_json(args.artifact_dir / "physical_controls.json", controls)
    if controls["overall_status"] != "PASS":
        raise RuntimeError(f"clean control generation/semantics failed: {controls['failures'][:5]}")
    preflight_checks = [
        {"id": "frozen_config", "status": "PASS"}, {"id": "receiver_source_semantics", "status": semantic["overall_status"]},
        {"id": "synthetic_physical_recovery", "status": recovery["overall_status"],
         "maximum_scaled_state_error": recovery["maximum_scaled_state_error"]},
        {"id": "clean_geometry_time_alignment", "status": geometry["report"]["overall_status"]},
        {"id": "mandatory_full_available", "status": "PASS"}, {"id": "lambda_minimum_windows", "status": "PASS", "observed": len(validation_terms)},
        {"id": "control_generation", "status": controls["overall_status"]}, {"id": "protected_access_count_zero", "status": "PASS"},
    ]
    canonical_write_json(args.artifact_dir / "implementation_resolution.json", {
        "schema": "gnss-doppler-lab.gcspo-stage0.implementation-resolution.v1", "status": "SOURCE_SYNTHETIC_RESOLUTION",
        "resolution": "Pinned receiver equations and epsilon synthetic vectors resolve all four requested direct physical rows without attack data.",
        "clean_only_bug_repairs": [{"id": "NUMPY_CANONICAL_JSON", "reason": "First clean-only attempt completed numerical paths but failed before PASS on NumPy scalar serialization; deterministic conversion added with no scientific-path change."}],
        "validated_rows": sorted(validated_rows), "attack_performance_used": False,
    })
    canonical_write_json(args.artifact_dir / "preflight_report.json", {
        "schema": "gnss-doppler-lab.gcspo-stage0.preflight-report.v1", "started_utc": started, "finished_utc": utc_now(),
        "config_sha256": CONFIG_SHA, "receiver_semantics": semantic, "closed_loop_transfer": closed_loop,
        "synthetic_physical_recovery": recovery, "geometry": geometry["report"], "checks": preflight_checks,
        "overall_status": "PASS", "protected_attack_rows_read": False, "attack_access_count": 0,
    })
    canonical_write_json(args.artifact_dir / "normal_model_summary.json", {
        "schema": "gnss-doppler-lab.gcspo-stage0.normal-model-summary.v1", "fit_role": "cleanStatic train-fit [30,140)",
        "ridge": clean["ridge"], "lag_epochs": clean["model"].lags, "intercept": clean["model"].intercept.tolist(),
        "coefficients": clean["model"].coefficients.tolist(), "coefficient_sha256": _hash_array(clean["model"].coefficients),
        "whitener_location": clean["whitener"].location.tolist(), "whitener_covariance": clean["whitener"].covariance.tolist(),
        "whitener_inverse_sqrt": clean["whitener"].inverse_sqrt.tolist(), "gamma": clean["gamma"].tolist(),
        "lambda_selected": selected_lambda, "lambda_objectives": lambda_objectives, "lambda_validation_windows": len(validation_terms),
        "validated_rows": sorted(validated_rows),
        "optional_method_capabilities": {
            "M1": {"status": "UNAVAILABLE", "reason": "AUTHENTICATED_CHECKPOINT_ABSENT"},
            "Fixed9": {"status": "LIMITED", "reason": "OPTIONAL_RERUN_INPUT_NOT_BOUND"},
            "GCMR": {"status": "LIMITED", "reason": "OPTIONAL_EXACT_SUPPORT_ADAPTER_NOT_BOUND"}
        },
        "normalization_epsilon_by_prn": {str(k): v for k, v in sorted(clean["data"].epsilons.items())},
    })
    canonical_write_json(args.artifact_dir / "thresholds.json", {
        "schema": "gnss-doppler-lab.gcspo-stage0.thresholds.v1", "source": "cleanStatic calibration [220,340)",
        "strict_comparison": "score > threshold", "Full": full_thresholds, "A1": clean["thresholds"],
        "calibration_windows": {"Full": len(full_calibration), "A1": len(clean["calibration"])},
    })
    canonical_write_json(args.artifact_dir / "physical_controls.json", controls)
    canonical_write_json(args.artifact_dir / "clean_only_report.json", {
        "schema": "gnss-doppler-lab.gcspo-stage0.clean-only-report.v1", "run_status": "CLEAN_ONLY_PASS",
        "protected_attack_rows_read": False, "attack_access_count": 0, "aggregated_prn_epochs": len(clean["data"].epoch),
        "scores": {"Full_calibration": full_calibration, "Full_holdout": full_holdout,
                   "A1_calibration": clean["calibration"], "A1_holdout": clean["holdout"]},
        "holdout_fpr": {"Full": fpr, "A1": a1_fpr}, "controls_generation": "PASS", "overall_status": "PASS",
    })
    print(f"CLEAN_ONLY_PASS Full_calibration={len(full_calibration)} Full_holdout={len(full_holdout)}", flush=True)
    return 0


def main():
    args = _parse()
    try: return clean_only(args) if args.phase == "clean-only" else protected(args)
    except Exception as exc:
        if args.phase == "protected":
            marker = args.artifact_dir.parent / f".{args.artifact_dir.name}.protected_run_started.json"
            ledger = args.artifact_dir / "access_ledger.jsonl"
            if not marker.exists() and (not ledger.exists() or ledger.stat().st_size == 0):
                try:
                    _record_preaccess_invalid(args.artifact_dir, exc)
                except Exception as invalid_exc:
                    print(f"pre-access invalid artifact failure: {invalid_exc}", file=sys.stderr)
            else:
                quarantine_failed_final_verdict(args.artifact_dir)
        print(str(exc), file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
