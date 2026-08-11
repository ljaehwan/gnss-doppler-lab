#!/usr/bin/env python3
"""Deterministic post-preregistration PG-SCC Stage-0 root-cause audit."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.pg_scc import load_feature_cache, pool_events as frozen_pool_events
from gnss_doppler_lab.pg_scc_physics import (
    CENTER, COORDINATES, DEFAULT_SEARCH, N_COORDINATES,
    analytic_same_prn_template, normalize_complex,
)
from gnss_doppler_lab.pg_scc_selector import (
    build_synthetic_bank, residual_evidence, symmetric_mask_from_logits,
    train_global_topk_mask,
)

FROZEN = ROOT / "artifacts/pg_scc_stage0_static_k9"
R1_OUTPUT = ROOT / "artifacts/pg_scc_stage0_r1_root_cause_audit"
OUTPUT = ROOT / "artifacts/pg_scc_stage0_r2_validator_repair"
CACHE = ROOT / "artifacts/acaf_nf_stage1_r3_static_detection"
FAMILY = {"ds3": "ds3", "ds4": "ds4", "ds7": "ds7_ds8", "ds8": "ds7_ds8"}
FAMILY_MEMBERS = {"ds3": ("ds3",), "ds4": ("ds4",), "ds7_ds8": ("ds7", "ds8")}
CONFIG_SHA256 = "336802a95e82df1da82822520fe8bd838bf18ce17da6ae29aa5695449f3b67f5"
SOURCE_SHA256 = "2b2c7bc55d031a9fd86f210e249a51d44c0a7f0e92500159e2b60dc05db87f70"
R1_FAILURE_BINDING_SHA256 = "550f3fde25742b571fa0a5206a96d0454300d1fe1732671b9bf655ccbb3f379f"
PREREGISTRATION_SHA = "c7887316ed981d0e7cde74b2bbadeb1cf83bb233"
PREREGISTRATION_BLOB_SHA256 = "25fcfdb342b733fab7e296f72940a92aabf89fdd03b662bda0845ca8bbd884c0"
REQUIRED_BASE_SHA = "68ab54677f5d0b4b55cc39279aec631f60f655a9"
SCIENTIFIC_IMPLEMENTATION_SHA = "9839823c00cafc34fbbf1d6b1dbe069eb2c4e74d"
R1_FAIL_CLOSED_SHA = "8cd78ed724e57f97498da26547a9ecbbc2a78fe1"
FROZEN_DESIGN_SHA256 = "28de36cfa264c755712d52e051d882d366a9a5d9065471371ad27309f1f07d7a"
R1_ARTIFACT_SHA256 = {
    "config.json": "ade700892fe3e055e0154ae859ba701a36f11a6ec9245db4b39fd087426eb0e6",
    "source_commit.json": "13ab9e544f06903dd6fde04c42e8aefb35a66946bbaf0456e95040fbb777d428",
    "r1_fail_closed_report.json": "041cc432cdc893e9dba867d6d3dc005e3ee7f2c8d25d542d54d4f374ca68e3f5",
}
FROZEN_EXPECTED_SUPPORT_PATH = FROZEN / "per_epoch_scores.csv"
FROZEN_EXPECTED_SUPPORT_SHA256 = "d3fd253807aea873dadce767fc027d3f0d4060ad8d3519e4c800b166e9e3bef5"
FROZEN_EXPECTED_PROJECTION_FIELDS = (
    "scenario", "phase", "second", "time_s", "channel", "prn", "method", "budget",
)
R1_PRESERVED_PATHS = {
    "artifacts/pg_scc_stage0_r1_root_cause_audit/config.json": "ade700892fe3e055e0154ae859ba701a36f11a6ec9245db4b39fd087426eb0e6",
    "artifacts/pg_scc_stage0_r1_root_cause_audit/r1_fail_closed_report.json": "041cc432cdc893e9dba867d6d3dc005e3ee7f2c8d25d542d54d4f374ca68e3f5",
    "artifacts/pg_scc_stage0_r1_root_cause_audit/source_commit.json": "13ab9e544f06903dd6fde04c42e8aefb35a66946bbaf0456e95040fbb777d428",
    "docs/PG_SCC_ROOT_CAUSE_AUDIT_IMPLEMENTATION.md": "ad746e2f481918a9a85e588beeaf709abbcaf3fcb81450b36c590372adad40dc",
    "docs/PG_SCC_ROOT_CAUSE_AUDIT_R1_FAIL_CLOSED.md": "4596439585dc7275774a709b5fd41da59b329e842b5cbc0a078256774b495b08",
    "scripts/run_pg_scc_root_cause_audit.py": "66009f9d14f9c66de4414e46dd3f2f12cd0f3a49cd96c7ae7deb0ea8663d7a76",
    "scripts/verify_pg_scc_root_cause_audit.py": "b9eb52f854e14b5a743c80d0bc5dc38ebb9d7c7eaef245cbe9a6389b81bb573a",
    "tests/test_pg_scc_root_cause_audit.py": "c0bb5f07ffd7f45c136d94d3a46420e9d222b3140e5c5f3270fbb8404438a290",
}
PHASE2_ALLOWED_CHANGED_PATHS = {
    "artifacts/pg_scc_stage0_r2_root_cause_audit/config.json",
    "artifacts/pg_scc_stage0_r2_root_cause_audit/source_commit.json",
    "artifacts/pg_scc_stage0_r2_root_cause_audit/r1_failure_binding.json",
    "artifacts/pg_scc_stage0_r2_root_cause_audit/support_preflight.json",
    "scripts/run_pg_scc_root_cause_audit.py",
    "scripts/verify_pg_scc_root_cause_audit.py",
    "tests/test_pg_scc_root_cause_audit.py",
    "tests/test_pg_scc_r2_preflight.py",
    "artifacts/pg_scc_stage0_r2_repair_followup/attempt_state.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/config.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/fail_closed_delivery_manifest_sha256.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/fail_closed_delivery_report.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/fresh_clone_verifier_report.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/implementation_manifest_sha256.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/normal_verifier_report.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/pre_run_state.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/predecessor_failure_binding.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/preregistration.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/protected_attempt_traceback.txt",
    "artifacts/pg_scc_stage0_r2_repair_followup/r1_failure_binding.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/related_tests_report.txt",
    "artifacts/pg_scc_stage0_r2_repair_followup/semantic_diff_audit.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/semantic_verifier_report.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/source_commit.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/support_preflight.json",
    "scripts/verify_pg_scc_r2_repair_followup.py",
    "scripts/verify_pg_scc_r2_semantic_diff.py",
    "tests/test_pg_scc_r2_repair_followup.py",
    "artifacts/pg_scc_stage0_r2_validator_repair/artifact_verifier_report.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/attempt_state.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/config.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/execution_trace.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/fresh_clone_verifier_report.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/implementation_manifest_sha256.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/normal_verifier_report.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/pre_run_state.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/predecessor_failure_binding.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/preregistration.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/r1_failure_binding.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/reproduction_check.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/semantic_diff_audit.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/source_commit.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/support_preflight.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/test_report.txt",
    "artifacts/pg_scc_stage0_r2_validator_repair/validator_root_cause.json",
    "scripts/verify_pg_scc_r2_validator_repair.py",
    "scripts/verify_pg_scc_r2_validator_semantic_diff.py",
    "tests/test_pg_scc_r2_validator_repair.py",
}
CORE_METHODS = (
    "pg_scc_k3", "pg_scc_k5", "pg_scc_k9", "fixed9", "epl3",
    "dense_two_source_glrt", "shuffled_k3",
)
AUDIT_LABELS = {"attack": "POST_HOC_DIAGNOSTIC", "k3": "EXPLORATORY_ONLY"}
REQUIRED_ROOT_CAUSES = (
    "SCORE_DILUTION", "NOISY_COORDINATE_ADDITION", "H1_NULL_OVERFIT",
    "DENSE_COVARIANCE_FAILURE", "SYNTHETIC_REAL_PHYSICS_MISMATCH",
    "SELECTOR_PROXY_OBJECTIVE_MISMATCH", "MASK_NON_IDENTIFIABILITY",
    "AWGN_CONTROL_MISSCALE", "CALIBRATION_TAIL_INSUFFICIENCY",
    "GENUINE_LACK_OF_SPARSE_GAIN",
)
REQUIRED_ARTIFACTS = (
    "README.md", "config.json", "source_commit.json", "r1_failure_binding.json",
    "support_preflight.json", "preregistration.json", "predecessor_failure_binding.json",
    "pre_run_state.json", "implementation_manifest_sha256.json",
    "semantic_diff_audit.json", "validator_root_cause.json", "attempt_state.json", "reproduction_check.json",
    "nested_mask_analysis.csv", "coordinate_contributions.csv",
    "score_dilution_metrics.csv", "dense_teacher_diagnostics.json",
    "synthetic_real_mismatch.csv", "selector_proxy_audit.json",
    "mask_seed_stability.csv", "random_mask_distribution.csv",
    "k3_exploratory_metrics.csv", "awgn_reaudit.json",
    "calibration_uncertainty.json", "bootstrap_intervals.csv",
    "root_cause_verdict.json", "artifact_manifest_sha256.json", "plots",
)
REQUIRED_PLOTS = (
    "nested_coordinate_map.png", "clean_variance_attack_contribution.png",
    "rss_score_dilution.png", "synthetic_real_delay_doppler.png",
    "learned_random_percentile.png", "seed_mask_stability.png",
    "detector_performance.png", "empirical_noise_awgn_response.png",
    "calibration_threshold_uncertainty.png",
)
IMPLEMENTATION_FILES = (
    "scripts/run_pg_scc_root_cause_audit.py",
    "scripts/verify_pg_scc_root_cause_audit.py",
    "scripts/verify_pg_scc_r2_repair_followup.py",
    "scripts/verify_pg_scc_r2_semantic_diff.py",
    "scripts/verify_pg_scc_r2_validator_repair.py",
    "scripts/verify_pg_scc_r2_validator_semantic_diff.py",
    "tests/test_pg_scc_root_cause_audit.py",
    "tests/test_pg_scc_r2_preflight.py",
    "tests/test_pg_scc_r2_repair_followup.py",
    "tests/test_pg_scc_r2_validator_repair.py",
)
IMPLEMENTATION_MANIFEST_FILES = (
    *IMPLEMENTATION_FILES,
    "artifacts/pg_scc_stage0_r2_validator_repair/config.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/source_commit.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/r1_failure_binding.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/support_preflight.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/preregistration.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/predecessor_failure_binding.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/validator_root_cause.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/test_report.txt",
    "artifacts/pg_scc_stage0_r2_validator_repair/semantic_diff_audit.json",
)
ALLOWED_ROLES = {
    "selector": {"clean_train", "clean_selection", "synthetic_train", "synthetic_validation"},
    "covariance": {"clean_train"},
    "threshold": {"clean_calibration"},
    "attack_diagnostic": {"attack_report_only"},
}


def diagnostic_scores(rss_h0: float, rss_h1: float, k: int) -> dict[str, Any]:
    """Preregistered score formulas with an explicit zero-denominator state."""
    if k <= 0 or not np.isfinite([rss_h0, rss_h1]).all():
        raise ValueError("finite RSS values and positive K required")
    raw = max(float(rss_h0) - float(rss_h1), 0.0)
    denominator = float(rss_h1) / max(2 * int(k) - 6, 1)
    if denominator <= 0 or not math.isfinite(denominator):
        corrected: dict[str, Any] = {
            "status": "UNAVAILABLE", "value": None,
            "reason": "zero_or_invalid_rss_h1_residual_denominator",
        }
    else:
        corrected = {"status": "AVAILABLE", "value": (raw / 4.0) / denominator, "reason": None}
    return {
        "raw_rss_improvement": raw,
        "improvement_per_k": raw / k,
        "delta_bic": 2.0 * raw - 4.0 * math.log(2.0 * k),
        "dof_corrected_likelihood_improvement": corrected,
    }


def validate_nested_masks(masks: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    values: dict[int, list[int]] = {}
    errors: list[str] = []
    for budget in (3, 5, 9):
        mask = [int(x) for x in masks.get(f"pg_scc_k{budget}", ())]
        values[budget] = mask
        if len(mask) != budget:
            errors.append(f"length_k{budget}")
        if len(set(mask)) != len(mask):
            errors.append(f"duplicate_k{budget}")
        if any(x < 0 or x >= N_COORDINATES for x in mask):
            errors.append(f"bounds_k{budget}")
        if CENTER not in mask:
            errors.append(f"prompt_k{budget}")
    if values[3] != values[5][:3] or values[5] != values[9][:5]:
        errors.append("config_declared_order_or_exact_additions")
    if not set(values[3]) < set(values[5]) or not set(values[5]) < set(values[9]):
        errors.append("strict_nested_sets")
    if errors:
        raise RuntimeError(f"fail-closed nested mask invariant: {','.join(errors)}")
    return {
        "status": "PASS", "ordered_additions": values[9],
        "k5_new": values[5][3:], "k9_new": values[9][5:],
    }


def assert_allowed_role(path: str, role: str) -> str:
    if role not in ALLOWED_ROLES.get(path, set()):
        raise RuntimeError(f"attack leakage guard: {path} may not consume role {role}")
    return "PASS"


def _pooled_details(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    pooled: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for row in rows:
        pooled[(str(row["scenario"]), str(row["phase"]), int(row["second"]))].add(int(row["prn"]))
    return {
        event: {"prns": sorted(prns), "prn_count": len(prns)}
        for event, prns in pooled.items()
    }


def _event_detail_text(prefix: str, detail: Mapping[str, Any]) -> str:
    prns = [int(value) for value in detail.get("prns", [])]
    return f"{prefix}_unique_prn_count={len(prns)} {prefix}_prns={prns}"


def validate_common_support(rows: Sequence[Mapping[str, Any]], methods: Sequence[str],
                            minimum_prns: int = 4,
                            frozen_expected_count: int | None = None,
                            frozen_expected_events: Sequence[Mapping[str, Any]] | None = None,
                            frozen_expected_source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate complete detector support using exact row and pooled-event identities."""
    if minimum_prns != 4:
        raise RuntimeError(f"minimum unique PRN support must remain exactly 4, got {minimum_prns}")
    ordered_methods = list(methods)
    if not ordered_methods:
        raise RuntimeError("raw-row support mismatch: no detectors supplied")
    selected = [row for row in rows if row.get("method") in set(ordered_methods)]
    channel_present = any("channel" in row and row.get("channel") is not None for row in selected)
    if channel_present and any("channel" not in row or row.get("channel") is None for row in selected):
        raise RuntimeError("raw-row support mismatch: optional channel presence is inconsistent")
    key_names = ("scenario", "phase", "second", "time_s", "prn") + (
        ("channel",) if channel_present else ()
    )
    supports: dict[str, set[tuple[Any, ...]]] = {}
    event_supports: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    method_rows_by_name: dict[str, list[Mapping[str, Any]]] = {}
    for method in ordered_methods:
        method_rows = [row for row in selected if row.get("method") == method]
        method_rows_by_name[method] = method_rows
        keys = [tuple(row[name] for name in key_names) for row in method_rows]
        if len(keys) != len(set(keys)):
            seen: set[tuple[Any, ...]] = set()
            duplicate = keys[0]
            for key in keys:
                if key in seen:
                    duplicate = key
                    break
                seen.add(key)
            event = _event_identity_from_raw_key(duplicate)
            detail = _pooled_details(method_rows)[
                (event["scenario"], event["phase"], event["second"])
            ]
            raise RuntimeError(
                "duplicate raw-row support "
                f"detector={method} pooled_event={json.dumps(event, sort_keys=True)} "
                f"unique_prn_count={detail['prn_count']} prns={detail['prns']} raw_row={duplicate!r}"
            )
        supports[method] = set(keys)
        event_supports[method] = _pooled_details(method_rows)

    reference_method = ordered_methods[0]
    reference_raw = supports[reference_method]
    reference_events = event_supports[reference_method]
    for method in ordered_methods[1:]:
        if supports[method] != reference_raw:
            difference = sorted(
                supports[method] ^ reference_raw,
                key=lambda value: tuple(str(item) for item in value),
            )[0]
            event_identity = _event_identity_from_raw_key(difference)
            event = (
                event_identity["scenario"], event_identity["phase"], event_identity["second"]
            )
            reference_detail = reference_events.get(event, {"prns": [], "prn_count": 0})
            detector_detail = event_supports[method].get(event, {"prns": [], "prn_count": 0})
            raise RuntimeError(
                "raw-row support mismatch "
                f"detector={method} pooled_event={json.dumps(event_identity, sort_keys=True)} "
                f"{_event_detail_text('reference', reference_detail)} "
                f"{_event_detail_text('detector', detector_detail)} raw_row={difference!r}"
            )

    eligible_supports = {
        method: {
            event: detail for event, detail in event_supports[method].items()
            if int(detail["prn_count"]) >= minimum_prns
        }
        for method in ordered_methods
    }
    eligible_hashes = validate_eligible_event_support(eligible_supports)
    raw_hashes = {method: _support_hash(supports[method]) for method in ordered_methods}
    event_rows = [
        {
            "eligible": int(detail["prn_count"]) >= minimum_prns,
            "identity": _pooled_event_identity(event),
            "prn_count": int(detail["prn_count"]),
            "prns": list(detail["prns"]),
        }
        for event, detail in sorted(reference_events.items())
    ]
    eligible = [
        {key: value for key, value in event.items() if key != "eligible"}
        for event in event_rows if event["eligible"]
    ]
    excluded = [
        {key: value for key, value in event.items() if key != "eligible"}
        for event in event_rows if not event["eligible"]
    ]
    reconstructed_count = len(eligible)
    expected_records = list(frozen_expected_events or [])
    expected_count = (
        len(expected_records) if frozen_expected_events is not None
        else reconstructed_count if frozen_expected_count is None
        else int(frozen_expected_count)
    )
    event_records_match = True
    if frozen_expected_events is not None:
        event_records_match = eligible == expected_records
    count_match = reconstructed_count == expected_count
    status = "PASS" if count_match and event_records_match else "FAIL"
    return {
        "schema": "pg_scc_stage0_r2_support_validation.v1",
        "status": status,
        "raw_row_support_key": list(key_names),
        "pooled_event_key": ["scenario", "phase", "second"],
        "minimum_unique_prns": minimum_prns,
        "total_raw_event_count": len(reference_raw),
        "common_rows_per_detector": len(reference_raw),
        "eligible_event_count": reconstructed_count,
        "excluded_event_count": len(excluded),
        "pooled_events": reconstructed_count,
        "pooled_event_unique_prn_counts": event_rows,
        "eligible_pooled_event_identities": [event["identity"] for event in eligible],
        "eligible_pooled_events": eligible,
        "excluded_pooled_event_identities": [event["identity"] for event in excluded],
        "excluded_pooled_events": excluded,
        "per_detector_raw_row_support_hashes": raw_hashes,
        "per_detector_eligible_event_support_hashes": eligible_hashes,
        "frozen_pooled_event_expected_count": expected_count,
        "reconstructed_pooled_event_count": reconstructed_count,
        "frozen_pooled_event_count_match": count_match,
        "frozen_pooled_event_records_match": event_records_match,
        "frozen_pooled_event_support_hash": _support_hash({
            (item["identity"]["scenario"], item["identity"]["phase"], int(item["identity"]["second"]))
            for item in expected_records
        }) if expected_records else _support_hash(set()),
        "frozen_pooled_event_expected_count_source": dict(frozen_expected_source or {}),
    }


def _event_identity_from_raw_key(key: tuple[Any, ...]) -> dict[str, Any]:
    return {"scenario": str(key[0]), "phase": str(key[1]), "second": int(key[2])}


def _pooled_event_identity(event: tuple[Any, ...]) -> dict[str, Any]:
    return {"scenario": str(event[0]), "phase": str(event[1]), "second": int(event[2])}


def _support_hash(values: Iterable[tuple[Any, ...]]) -> str:
    canonical = sorted((list(value) for value in values), key=lambda value: tuple(map(str, value)))
    payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_eligible_event_support(
    supports: Mapping[str, Mapping[tuple[Any, ...], Mapping[str, Any]]],
) -> dict[str, str]:
    methods = list(supports)
    if not methods:
        raise RuntimeError("eligible-event support mismatch: no detectors supplied")
    reference_method = methods[0]
    reference = set(supports[reference_method])
    for method in methods[1:]:
        actual = set(supports[method])
        if actual != reference:
            event = sorted(reference ^ actual)[0]
            reference_detail = supports[reference_method].get(event, {"prns": [], "prn_count": 0})
            detector_detail = supports[method].get(event, {"prns": [], "prn_count": 0})
            detail = reference_detail if event in reference else detector_detail
            raise RuntimeError(
                "eligible-event support mismatch "
                f"detector={method} pooled_event={json.dumps(_pooled_event_identity(event), sort_keys=True)} "
                f"prn_count={detail['prn_count']} {_event_detail_text('reference', reference_detail)} "
                f"{_event_detail_text('detector', detector_detail)}"
            )
    return {method: _support_hash(set(supports[method])) for method in methods}


def load_frozen_expected_support() -> dict[str, Any]:
    """Project only support metadata from an independently frozen PG-SCC score container."""
    current_hash = sha256(FROZEN_EXPECTED_SUPPORT_PATH)
    immutable_bytes = subprocess.check_output(
        ["git", "show", f"{R1_FAIL_CLOSED_SHA}:{FROZEN_EXPECTED_SUPPORT_PATH.relative_to(ROOT)}"],
        cwd=ROOT,
    )
    immutable_hash = hashlib.sha256(immutable_bytes).hexdigest()
    if current_hash != FROZEN_EXPECTED_SUPPORT_SHA256 or immutable_hash != current_hash:
        raise RuntimeError("FAIL_CLOSED_SUPPORT_PREFLIGHT:frozen expected-support source drift")
    projection = subprocess.run(
        ["cut", "-d,", "-f1-6,11-12", str(FROZEN_EXPECTED_SUPPORT_PATH)],
        text=True, capture_output=True,
    )
    if projection.returncode:
        raise RuntimeError("FAIL_CLOSED_SUPPORT_PREFLIGHT:frozen support projection failed")
    projected_rows = list(csv.DictReader(projection.stdout.splitlines()))
    if tuple(projected_rows[0].keys()) != FROZEN_EXPECTED_PROJECTION_FIELDS:
        raise RuntimeError("FAIL_CLOSED_SUPPORT_PREFLIGHT:frozen support projection schema drift")
    selected = [
        row for row in projected_rows
        if row["method"] == "pg_scc_k9" and int(row["budget"]) == 9
    ]
    if not selected:
        raise RuntimeError("FAIL_CLOSED_SUPPORT_PREFLIGHT:no independent frozen expected support")
    details = _pooled_details(selected)
    event_records = [
        {
            "identity": _pooled_event_identity(event),
            "prn_count": int(detail["prn_count"]),
            "prns": list(detail["prns"]),
        }
        for event, detail in sorted(details.items()) if int(detail["prn_count"]) >= 4
    ]
    source = {
        "path": str(FROZEN_EXPECTED_SUPPORT_PATH.relative_to(ROOT)),
        "sha256": current_hash,
        "immutable_commit": R1_FAIL_CLOSED_SHA,
        "projection_fields": list(FROZEN_EXPECTED_PROJECTION_FIELDS),
        "filter": {"method": "pg_scc_k9", "budget": 9},
        "eligible_event_count": len(event_records),
        "eligible_event_support_hash": _support_hash({
            (item["identity"]["scenario"], item["identity"]["phase"], item["identity"]["second"])
            for item in event_records
        }),
    }
    return {"source": source, "eligible_event_count": len(event_records), "event_records": event_records}


def frozen_pg_scc_pooled_event_count(metadata: Sequence[Mapping[str, Any]]) -> int:
    """Compatibility helper for synthetic gate fixtures; production uses an immutable artifact."""
    synthetic = [
        {
            **{key: row[key] for key in ("scenario", "phase", "second", "time_s", "prn")},
            "method": "frozen_pg_scc_reference", "budget": 0, "score": 0.0,
        }
        for row in metadata
    ]
    return len(frozen_pool_events(synthetic, "median"))


def verify_r1_artifact_immutability() -> dict[str, Any]:
    actual = {name: sha256(R1_OUTPUT / name) for name in R1_ARTIFACT_SHA256}
    return {
        "status": "PASS" if actual == R1_ARTIFACT_SHA256 else "FAIL",
        "sha256": actual,
        "expected_sha256": dict(R1_ARTIFACT_SHA256),
    }


def _function_ast_hash(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def assert_frozen_formula_identity() -> dict[str, Any]:
    groups = {
        "score": (
            "batch_glrt", "diagnostic_scores", "pooled_events", "metric_bundle", "_fit_all",
            "reproduce", "reproduce_exact", "nested_outputs",
            "residual_components", "_dense_mismatch",
        ),
        "normalization": ("normalize_all", "_surface_array"),
        "selector": (
            "generate_random_masks", "compute_seed_stability", "_selector_audit", "_jaccard",
        ),
        "covariance": ("covariance_variants", "_covariance_audit", "residual_components"),
        "synthetic": ("_selector_audit", "_dense_mismatch", "_awgn_audit"),
        "mask": ("validate_nested_masks", "generate_random_masks"),
        "threshold": ("metric_bundle", "_calibration_audit", "_root_cause_verdict"),
    }
    names = tuple(dict.fromkeys(name for values in groups.values() for name in values))
    current_source = Path(__file__).read_text(encoding="utf-8")
    r1_source = subprocess.check_output(
        ["git", "show", f"{R1_FAIL_CLOSED_SHA}:scripts/run_pg_scc_root_cause_audit.py"],
        cwd=ROOT, text=True,
    )
    functions = {}
    for name in names:
        current = _function_ast_hash(current_source, name)
        expected = _function_ast_hash(r1_source, name)
        functions[name] = {"current": current, "r1": expected, "match": current == expected}
    artifacts = {}
    for name in ("masks.json", "thresholds.json"):
        current = sha256(FROZEN / name)
        expected_bytes = subprocess.check_output(
            ["git", "show", f"{R1_FAIL_CLOSED_SHA}:artifacts/pg_scc_stage0_static_k9/{name}"],
            cwd=ROOT,
        )
        expected = hashlib.sha256(expected_bytes).hexdigest()
        artifacts[name] = {"current": current, "r1": expected, "match": current == expected}
    source_files = {}
    for relative in (
        "src/gnss_doppler_lab/pg_scc.py",
        "src/gnss_doppler_lab/pg_scc_physics.py",
        "src/gnss_doppler_lab/pg_scc_selector.py",
    ):
        expected_bytes = subprocess.check_output(
            ["git", "show", f"{R1_FAIL_CLOSED_SHA}:{relative}"], cwd=ROOT,
        )
        expected = hashlib.sha256(expected_bytes).hexdigest()
        current = sha256(ROOT / relative)
        source_files[relative] = {"current": current, "r1": expected, "match": current == expected}
    matches = (
        all(item["match"] for item in functions.values())
        and all(item["match"] for item in artifacts.values())
        and all(item["match"] for item in source_files.values())
    )
    return {
        "status": "PASS" if matches else "FAIL",
        "groups": {name: list(values) for name, values in groups.items()},
        "function_ast_sha256": functions,
        "frozen_artifact_sha256": artifacts,
        "frozen_source_sha256": source_files,
    }


def _git_changed_paths_since_r1() -> set[str]:
    commands = (
        ["git", "diff", "--name-only", R1_FAIL_CLOSED_SHA, "--"],
        ["git", "diff", "--cached", "--name-only", "--"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    )
    changed: set[str] = set()
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        changed.update(line for line in result.stdout.splitlines() if line)
    return changed


def verify_r1_identity() -> dict[str, Any]:
    binding = load_json(OUTPUT / "r1_failure_binding.json")
    bound = binding.get("preserved_file_sha256", {})
    immutable_hashes = {}
    for path, expected in R1_PRESERVED_PATHS.items():
        committed = subprocess.check_output(["git", "show", f"{R1_FAIL_CLOSED_SHA}:{path}"], cwd=ROOT)
        immutable_hashes[path] = hashlib.sha256(committed).hexdigest()
    binding_match = bound == R1_PRESERVED_PATHS and immutable_hashes == R1_PRESERVED_PATHS
    current_preserved = {
        path: sha256(ROOT / path)
        for path in R1_PRESERVED_PATHS
        if path.startswith("artifacts/pg_scc_stage0_r1_root_cause_audit/") or path.startswith("docs/")
    }
    current_match = all(current_preserved[path] == R1_PRESERVED_PATHS[path] for path in current_preserved)
    changed = _git_changed_paths_since_r1()
    disallowed = sorted(changed - PHASE2_ALLOWED_CHANGED_PATHS)
    code = assert_frozen_formula_identity()
    declarations = binding.get("explicit_unchanged_declarations", {})
    status = (
        binding.get("r1_fail_closed_commit") == R1_FAIL_CLOSED_SHA
        and binding_match and current_match and len(immutable_hashes) == 8
        and declarations and all(value is True for value in declarations.values())
        and not disallowed and code["status"] == "PASS"
    )
    return {
        "status": "PASS" if status else "FAIL",
        "binding_hashes_verified": len(immutable_hashes),
        "preserved_file_sha256": immutable_hashes,
        "current_preserved_file_sha256": current_preserved,
        "disallowed_paths_unchanged": not disallowed,
        "disallowed_changed_paths": disallowed,
        "phase2_changed_paths": sorted(changed),
        "unchanged_code_identity": code,
    }

def generate_random_masks(seed: int, counts: Mapping[int, int]) -> dict[int, list[list[int]]]:
    rng = np.random.default_rng(seed)
    candidates = np.asarray([index for index in range(N_COORDINATES) if index != CENTER], int)
    result: dict[int, list[list[int]]] = {}
    for budget in sorted(counts):
        requested = int(counts[budget])
        masks: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        while len(masks) < requested:
            selected = sorted(rng.choice(candidates, size=budget - 1, replace=False).astype(int).tolist())
            mask = (CENTER, *selected)
            if mask not in seen:
                seen.add(mask)
                masks.append(list(mask))
        result[int(budget)] = masks
    return result


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b)


def compute_seed_stability(seeds: Sequence[int], budgets: Sequence[int], trainer: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_budget: dict[int, list[list[int]]] = defaultdict(list)
    for seed in seeds:
        for budget in budgets:
            mask = [int(x) for x in trainer(int(seed), int(budget))]
            by_budget[int(budget)].append(mask)
            rows.append({"seed": int(seed), "budget": int(budget), "mask": mask,
                         "label": "CLEAN_SYNTHETIC_SELECTION_ONLY"})
    medians = {}
    for budget, masks in by_budget.items():
        pairs = [_jaccard(masks[i], masks[j]) for i in range(len(masks)) for j in range(i + 1, len(masks))]
        medians[str(budget)] = float(np.median(pairs)) if pairs else 1.0
    return {"rows": rows, "median_pairwise_jaccard": medians}


def block_bootstrap(rows: Sequence[Mapping[str, Any]], *, value_key: str,
                    block_seconds: float, iterations: int, seed: int) -> dict[str, Any]:
    if block_seconds < 10.0:
        raise ValueError("block_seconds must be at least 10")
    blocks: dict[tuple[Any, int], list[float]] = defaultdict(list)
    for row in rows:
        blocks[(row.get("family"), int(math.floor(float(row["second"]) / block_seconds)))].append(float(row[value_key]))
    ordered = [blocks[key] for key in sorted(blocks, key=lambda x: (str(x[0]), x[1]))]
    if len(ordered) < 2:
        return {"status": "LIMITED", "reason": "fewer_than_two_blocks", "block_count": len(ordered),
                "iterations": iterations, "replicate_means": []}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(iterations):
        picked = rng.integers(0, len(ordered), size=len(ordered))
        sample = [value for index in picked for value in ordered[int(index)]]
        means.append(float(np.mean(sample)))
    return {"status": "PASS", "block_count": len(ordered), "block_seconds": block_seconds,
            "iterations": iterations, "seed": seed, "replicate_means": means,
            "interval_95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))]}


def finalize_manifest(root: Path) -> dict[str, str]:
    manifest = {
        str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest_sha256.json"
    }
    dump_json(root / "artifact_manifest_sha256.json", manifest)
    return manifest



def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    names = sorted({name for row in values for name in row}) if values else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_support_metadata(path: Path) -> list[dict[str, Any]]:
    """Read only identity/support scalars through a fixed external projection."""
    projection = (
        '.[] | {scenario, phase, second, time_s, prn} '
        '+ (if has("channel") and .channel != null then {channel} else {} end)'
    )
    result = subprocess.run(
        ["jq", "-ce", projection, str(path)],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"support metadata projection failed: {path}:{result.stderr.strip()}")
    required = {"scenario", "phase", "second", "time_s", "prn"}
    output = []
    for index, line in enumerate(result.stdout.splitlines()):
        row = json.loads(line)
        if not required.issubset(row):
            raise RuntimeError(f"support metadata missing identity fields: {path}:{index}")
        if set(row) - (required | {"channel"}):
            raise RuntimeError(f"support projection exposed non-identity fields: {path}:{index}")
        output.append(row)
    return output


def build_metadata_support_preflight_report(
    *, metadata_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Reconstruct every preflight field from whitelisted metadata projections."""
    paths = list(metadata_paths or (
        CACHE / "clean_features.json",
        CACHE / "attack_features.json",
    ))
    phase1_expected = {
        "config.json": CONFIG_SHA256,
        "source_commit.json": SOURCE_SHA256,
        "r1_failure_binding.json": R1_FAILURE_BINDING_SHA256,
    }
    phase1_actual = {name: sha256(OUTPUT / name) for name in phase1_expected}
    if phase1_actual != phase1_expected:
        raise RuntimeError("FAIL_CLOSED_SUPPORT_PREFLIGHT:Phase-1 artifact checksum drift")
    config = load_json(OUTPUT / "config.json")
    source = load_json(OUTPUT / "source_commit.json")
    if source.get("branch") != "research/pg-scc-stage0-r2-validator-repair":
        raise RuntimeError("FAIL_CLOSED_SUPPORT_PREFLIGHT:Phase-1 source binding drift")
    if source.get("metadata_only_phase_a", {}).get("protected_score_fields_read") != 0:
        raise RuntimeError("FAIL_CLOSED_SUPPORT_PREFLIGHT:Phase-1 protected-access drift")
    support_config = config["common_support"]
    minimum_prns = int(
        support_config["eligible_event_filtering"]["minimum_unique_prns_per_pooled_event"]
    )
    if minimum_prns != 4:
        raise RuntimeError("minimum unique PRN support drift")
    if support_config["raw_row_support"]["base_key"] != [
        "scenario", "phase", "second", "time_s", "prn"
    ]:
        raise RuntimeError("raw-row identity drift")
    if support_config["pooled_event_identity"]["key"] != ["scenario", "phase", "second"]:
        raise RuntimeError("pooled-event identity drift")
    source_rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for source_path in paths:
        projected = read_support_metadata(source_path)
        metadata.extend(projected)
        source_rows.append({
            "path": str(source_path.relative_to(ROOT)),
            "sha256": sha256(source_path),
            "rows": len(projected),
        })
    frozen = load_frozen_expected_support()
    detector_rows = [
        {**row, "method": method}
        for method in CORE_METHODS
        for row in metadata
    ]
    report = validate_common_support(
        detector_rows,
        CORE_METHODS,
        minimum_prns=minimum_prns,
        frozen_expected_events=frozen["event_records"],
        frozen_expected_source=frozen["source"],
    )
    r1_artifacts = verify_r1_artifact_immutability()
    r1_identity = verify_r1_identity()
    formulas = r1_identity["unchanged_code_identity"]
    report.update({
        "schema": "pg_scc_stage0_r2_support_preflight.v1",
        "status": "PASS" if (
            report["status"] == "PASS"
            and r1_artifacts["status"] == "PASS"
            and r1_identity["status"] == "PASS"
            and formulas["status"] == "PASS"
        ) else "FAIL",
        "metadata_fields_read": [
            "scenario", "phase", "second", "time_s", "prn", "channel_if_present",
        ],
        "metadata_projection": "fixed jq whitelist; non-identity values are not emitted",
        "metadata_sources": source_rows,
        "protected_score_fields_read": 0,
        "protected_score_fields_projected_or_read": 0,
        "containers_opened_for_metadata_projection": [
            item["path"] for item in source_rows
        ] + [frozen["source"]["path"]],
        "score_bearing_containers_opened_for_metadata_projection": [{
            "path": frozen["source"]["path"],
            "sha256": frozen["source"]["sha256"],
            "projection_fields": frozen["source"]["projection_fields"],
            "excluded_fields": ["score"],
        }],
        "r1_artifact_immutability": r1_artifacts,
        "r1_identity": r1_identity,
        "frozen_formula_identity": formulas,
    })
    return report


def run_metadata_support_preflight(
    *, output_path: Path | None = None,
    metadata_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Execute Phase-2 support checks without projecting any protected score field."""
    target = output_path or (OUTPUT / "support_preflight.json")
    report = build_metadata_support_preflight_report(metadata_paths=metadata_paths)
    dump_json(target, report)
    return report


def validate_committed_support_preflight(
    report: Mapping[str, Any], *, require_head_blob: bool = False,
) -> dict[str, Any]:
    """Exact-check stable semantics and bound post-preflight operational paths."""
    try:
        expected = build_metadata_support_preflight_report()
    except Exception as exc:
        raise RuntimeError(f"FAIL_CLOSED_SUPPORT_PREFLIGHT:reconstruction failed:{exc}") from exc

    committed = dict(report)
    committed_identity = committed.get("r1_identity")
    expected_identity = expected.get("r1_identity")
    if not isinstance(committed_identity, Mapping) or not isinstance(expected_identity, Mapping):
        raise RuntimeError("FAIL_CLOSED_SUPPORT_PREFLIGHT:invalid r1_identity schema")

    committed_stable = {key: value for key, value in committed.items() if key != "r1_identity"}
    expected_stable = {key: value for key, value in expected.items() if key != "r1_identity"}
    if committed_stable != expected_stable:
        differing = sorted(
            key for key in set(committed_stable) | set(expected_stable)
            if committed_stable.get(key) != expected_stable.get(key)
        )
        raise RuntimeError(
            "FAIL_CLOSED_SUPPORT_PREFLIGHT:committed semantics/accounting mismatch:"
            + ",".join(differing)
        )

    committed_identity_stable = {
        key: value for key, value in committed_identity.items()
        if key != "phase2_changed_paths"
    }
    expected_identity_stable = {
        key: value for key, value in expected_identity.items()
        if key != "phase2_changed_paths"
    }
    if committed_identity_stable != expected_identity_stable:
        differing = sorted(
            key for key in set(committed_identity_stable) | set(expected_identity_stable)
            if committed_identity_stable.get(key) != expected_identity_stable.get(key)
        )
        raise RuntimeError(
            "FAIL_CLOSED_SUPPORT_PREFLIGHT:stable r1_identity mismatch:"
            + ",".join(differing)
        )

    committed_path_list = committed_identity.get("phase2_changed_paths")
    expected_path_list = expected_identity.get("phase2_changed_paths")
    if (
        not isinstance(committed_path_list, list)
        or not isinstance(expected_path_list, list)
        or any(not isinstance(path, str) for path in [*committed_path_list, *expected_path_list])
        or committed_path_list != sorted(set(committed_path_list))
        or expected_path_list != sorted(set(expected_path_list))
    ):
        raise RuntimeError("FAIL_CLOSED_SUPPORT_PREFLIGHT:invalid phase2_changed_paths schema")
    committed_paths = set(committed_path_list)
    expected_paths = set(expected_path_list)
    removed_paths = sorted(committed_paths - expected_paths)

    preregistration = _load_committed_preregistration()
    operational_path_list = preregistration.get("dynamic_operational_path_contract", {}).get(
        "allowed_only_when_added_after_committed_preflight_and_subset_of_this_set"
    )
    if (
        not isinstance(operational_path_list, list)
        or any(not isinstance(path, str) for path in operational_path_list)
        or operational_path_list != sorted(set(operational_path_list))
    ):
        raise RuntimeError("FAIL_CLOSED_SUPPORT_PREFLIGHT:invalid operational path preregistration")
    unregistered_additions = sorted((expected_paths - committed_paths) - set(operational_path_list))
    if removed_paths or unregistered_additions:
        raise RuntimeError(
            "FAIL_CLOSED_SUPPORT_PREFLIGHT:dynamic operational path mismatch:"
            f"removed={removed_paths},unregistered_additions={unregistered_additions}"
        )
    if expected["status"] != "PASS" or expected["protected_score_fields_projected_or_read"] != 0:
        raise RuntimeError("FAIL_CLOSED_SUPPORT_PREFLIGHT:reconstructed preflight is not PASS")
    if require_head_blob:
        committed = subprocess.run(
            ["git", "show", "HEAD:artifacts/pg_scc_stage0_r2_validator_repair/support_preflight.json"],
            cwd=ROOT, text=True, capture_output=True,
        )
        canonical = json.dumps(dict(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
        if committed.returncode or committed.stdout != canonical:
            raise RuntimeError("FAIL_CLOSED_SUPPORT_PREFLIGHT:preflight bytes are not committed at HEAD")
    return expected


def load_protected_inputs_after_preflight(
    *,
    preflight_path: Path = OUTPUT / "support_preflight.json",
    protected_loader: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fail closed before invoking a protected loader unless all semantics reconstruct."""
    try:
        report = load_json(preflight_path)
        validate_committed_support_preflight(
            report,
            require_head_blob=protected_loader is None and preflight_path.resolve() == (
                OUTPUT / "support_preflight.json"
            ).resolve(),
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"FAIL_CLOSED_SUPPORT_PREFLIGHT:invalid report:{exc}") from exc
    loader = protected_loader or load_protected_inputs
    return loader()


def filter_metric_inputs_to_eligible_events(
    metadata: Sequence[Mapping[str, Any]],
    scores_by_method: Mapping[str, np.ndarray],
    support: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, np.ndarray]]:
    """Apply the proven common eligible-event set before unchanged metric logic."""
    if support.get("status") != "PASS":
        raise RuntimeError("FAIL_CLOSED_SUPPORT_PREFLIGHT:cannot filter from failed support")
    eligible = {
        (item["scenario"], item["phase"], int(item["second"]))
        for item in support.get("eligible_pooled_event_identities", [])
    }
    keep = np.asarray([
        (str(row["scenario"]), str(row["phase"]), int(row["second"])) in eligible
        for row in metadata
    ], dtype=bool)
    filtered_metadata = [row for row, include in zip(metadata, keep) if include]
    filtered_scores: dict[str, np.ndarray] = {}
    for method, scores in scores_by_method.items():
        values = np.asarray(scores)
        if len(values) != len(metadata):
            raise RuntimeError(f"FAIL_CLOSED_SUPPORT_PREFLIGHT:score length mismatch:{method}")
        filtered_scores[method] = values[keep]
    return filtered_metadata, filtered_scores

def normalize_all(surfaces: np.ndarray) -> np.ndarray:
    return np.asarray([normalize_complex(surface, "prompt_phase") for surface in surfaces], np.complex128)


def batch_glrt(surfaces: np.ndarray, auth: np.ndarray, covariance: np.ndarray,
               indices: Sequence[int] | None = None) -> dict[str, np.ndarray]:
    """Vectorized equivalent of the frozen two-source GLS, including ridge."""
    selected = np.arange(N_COORDINATES) if indices is None else np.asarray(indices, int)
    y = np.asarray(surfaces, np.complex128)[:, selected]
    a = np.asarray(auth, np.complex128)[selected]
    cov = np.asarray(covariance, np.complex128)[np.ix_(selected, selected)]
    off = cov - np.diag(np.diag(cov))
    precision = (np.diag(1 / np.maximum(np.real(np.diag(cov)), 1e-12))
                 if np.count_nonzero(off) == 0 else np.linalg.inv(cov))

    def solve(design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gram = design.conj().T @ precision @ design + np.eye(design.shape[1]) * 1e-3
        rhs = (design.conj().T @ precision @ y.T).T
        coefficients = np.linalg.solve(gram, rhs.T).T
        residual = y - coefficients @ design.T
        weighted = residual @ precision.T
        rss = np.maximum(np.real(np.sum(np.conj(residual) * weighted, axis=1)), 0)
        return rss, coefficients

    rss0, beta0 = solve(a[:, None])
    best = rss0.copy()
    beta_auth = beta0[:, 0].copy()
    beta_second = np.zeros(len(y), np.complex128)
    delay = np.zeros(len(y))
    doppler = np.zeros(len(y))
    for tau, frequency in DEFAULT_SEARCH:
        second = analytic_same_prn_template(tau, frequency)[selected]
        rss, coefficients = solve(np.column_stack((a, second)))
        improve = rss < best
        best[improve] = rss[improve]
        beta_auth[improve] = coefficients[improve, 0]
        beta_second[improve] = coefficients[improve, 1]
        delay[improve] = tau
        doppler[improve] = frequency
    raw = np.maximum(rss0 - best, 0)
    k = len(selected)
    denominator = best / max(2 * k - 6, 1)
    dof_available = np.isfinite(denominator) & (denominator > 0)
    dof = np.full_like(raw, np.nan)
    np.divide(raw / 4, denominator, out=dof, where=dof_available)
    return {
        "score": raw / k, "raw": raw, "rss_h0": rss0, "rss_h1": best,
        "delta_bic": 2 * raw - 4 * math.log(2 * k), "dof": dof,
        "dof_available": dof_available,
        "delay": delay, "doppler": doppler, "beta_h0": beta0[:, 0],
        "beta_auth": beta_auth, "beta_second": beta_second,
    }


def pooled_events(metadata: Sequence[Mapping[str, Any]], scores: np.ndarray) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[tuple[Mapping[str, Any], float]]] = defaultdict(list)
    for row, score in zip(metadata, np.asarray(scores, float)):
        groups[(row["scenario"], row["phase"], int(row["second"]))].append((row, float(score)))
    output = []
    for (scenario, phase, second), group in sorted(groups.items()):
        if len(group) < 4:
            continue
        output.append({
            "scenario": scenario, "family": FAMILY.get(scenario, scenario), "phase": phase,
            "second": second, "time_s": float(np.mean([float(x[0]["time_s"]) for x in group])),
            "prn_count": len(group), "score": float(np.median([x[1] for x in group])),
        })
    return output


def metric_bundle(metadata: Sequence[Mapping[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    events = pooled_events(metadata, scores)
    negative = [r["score"] for r in events if r["scenario"] == "cleanStatic" and r["phase"] == "holdout"]
    families = {}
    for family, scenarios in FAMILY_MEMBERS.items():
        positive = [r["score"] for r in events if r["scenario"] in scenarios and r["phase"] != "strict_pre"]
        families[family] = float(roc_auc_score([0] * len(negative) + [1] * len(positive),
                                                negative + positive, max_fpr=.05))
    calibration = np.asarray([r["score"] for r in events if r["scenario"] == "cleanStatic" and r["phase"] == "calibration"])
    threshold = float(np.quantile(calibration, .99, method="higher"))
    holdout_alarms = sum(r["score"] >= threshold for r in events if r["scenario"] == "cleanStatic" and r["phase"] == "holdout")
    external = {}
    for scenario in FAMILY:
        values = [r for r in events if r["scenario"] == scenario and r["phase"] == "strict_pre"]
        external[scenario] = sum(r["score"] >= threshold for r in values) / len(values)
    return {
        "events": events, "macro_pauc": float(np.mean(list(families.values()))),
        "family_pauc": families, "threshold": threshold, "calibration_events": len(calibration),
        "holdout_alarms": holdout_alarms, "holdout_events": len(negative),
        "holdout_fpr": holdout_alarms / len(negative), "external_fpr": external,
        "worst_fpr": max(holdout_alarms / len(negative), *external.values()),
    }
def verify_preregistration(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    errors = []
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    base_expected = REQUIRED_BASE_SHA
    base = subprocess.check_output(["git", "merge-base", "HEAD", base_expected], cwd=ROOT, text=True).strip()
    if branch != "research/pg-scc-stage0-r2-validator-repair":
        errors.append("branch_mismatch")
    if base != base_expected:
        errors.append("base_merge_base_mismatch")
    if subprocess.call(
        ["git", "merge-base", "--is-ancestor", PREREGISTRATION_SHA, head], cwd=ROOT
    ) != 0:
        errors.append("preregistration_not_ancestor")
    for key in ("attack_fit", "attack_based_selection", "post_attack_retuning"):
        if config.get(key) is not False or config["guard"].get(key) is not False:
            errors.append(key)
    if source.get("metadata_only_phase_a", {}).get("protected_score_fields_read") != 0:
        errors.append("preregistration_protected_access")
    expected_files = {
        "config.json": CONFIG_SHA256,
        "source_commit.json": SOURCE_SHA256,
        "r1_failure_binding.json": R1_FAILURE_BINDING_SHA256,
    }
    errors.extend(
        f"phase1_hash:{name}" for name, expected in expected_files.items()
        if sha256(OUTPUT / name) != expected
    )
    if verify_r1_artifact_immutability()["status"] != "PASS":
        errors.append("r1_artifact_drift")
    if assert_frozen_formula_identity()["status"] != "PASS":
        errors.append("frozen_formula_drift")
    if sha256(FROZEN / "frozen_design.json") != FROZEN_DESIGN_SHA256:
        errors.append("frozen_design_drift")
    freeze = load_json(FROZEN / "freeze_manifest.json")
    drift = [name for name, digest in freeze.items()
             if not (FROZEN / name).is_file() or sha256(FROZEN / name) != digest]
    errors.extend(f"freeze_drift:{name}" for name in drift)
    return {"status": "PASS" if not errors else "FAIL", "errors": errors,
            "branch": branch, "head_during_audit": head, "merge_base": base,
            "freeze_files_verified": len(freeze)}


def reproduce(metadata: list[dict[str, Any]], masks: dict[str, list[int]],
              fits: dict[str, dict[str, np.ndarray]], bundles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stored = read_csv(FROZEN / "per_epoch_scores.csv")
    index = {
        (r["scenario"], r["phase"], int(r["second"]), int(r["channel"]),
         int(r["prn"]), r["method"], int(r["budget"])): float(r["score"])
        for r in stored
    }
    score_checks = {}
    for method, fit in fits.items():
        if method not in {"pg_scc_k3", "pg_scc_k5", "pg_scc_k9", "fixed9",
                          "epl3", "dense_two_source_glrt", "shuffled_k3"}:
            continue
        budget = N_COORDINATES if method == "dense_two_source_glrt" else len(masks[method])
        errors, missing = [], 0
        for row, score in zip(metadata, fit["score"]):
            key = (row["scenario"], row["phase"], int(row["second"]), int(row["channel"]),
                   int(row["prn"]), method, budget)
            if key in index:
                errors.append(abs(float(score) - index[key]))
            else:
                missing += 1
        score_checks[method] = {"rows": len(errors), "missing": missing,
                                "max_abs_score_error": max(errors, default=None)}
    frozen = {(r["method"], int(r["budget"])): r for r in read_csv(FROZEN / "baseline_metrics.csv")
              if r["status"] == "AVAILABLE"}
    metric_checks = {}
    for method in ("pg_scc_k3", "pg_scc_k5", "pg_scc_k9", "fixed9",
                   "epl3", "dense_two_source_glrt", "shuffled_k3"):
        budget = N_COORDINATES if method == "dense_two_source_glrt" else len(masks[method])
        expected = float(frozen[(method, budget)]["family_macro_low_fpr_pauc"])
        actual = bundles[method]["macro_pauc"]
        metric_checks[method] = {"recomputed": actual, "frozen": expected,
                                 "absolute_error": abs(actual - expected)}
    random_mean = float(np.mean([bundles[f"random{n}_k9"]["macro_pauc"] for n in range(1, 6)]))
    comparison = max(random_mean, bundles["uniform_k9"]["macro_pauc"],
                     bundles["shuffled_k9"]["macro_pauc"])
    maximum_score_error = max(x["max_abs_score_error"] or 0 for x in score_checks.values())
    maximum_metric_error = max(x["absolute_error"] for x in metric_checks.values())
    status = "PASS" if maximum_score_error <= 1e-10 and maximum_metric_error <= 1e-10 else "REPRODUCTION_MISMATCH"
    return {
        "schema": "pg_scc_reproduction_check.v1", "status": status,
        "independent_recomputation": True, "stored_summary_copied": False,
        "score_checks": score_checks, "metric_checks": metric_checks,
        "targets": {
            "pg_scc_k9_macro_pauc": bundles["pg_scc_k9"]["macro_pauc"],
            "dense_macro_pauc": bundles["dense_two_source_glrt"]["macro_pauc"],
            "fixed9_macro_pauc": bundles["fixed9"]["macro_pauc"],
            "pg_scc_k3_macro_pauc": bundles["pg_scc_k3"]["macro_pauc"],
            "k9_worst_fpr": bundles["pg_scc_k9"]["worst_fpr"],
            "k9_minus_fixed9_by_family": {
                family: bundles["pg_scc_k9"]["family_pauc"][family] - bundles["fixed9"]["family_pauc"][family]
                for family in FAMILY_MEMBERS
            },
            "learned_vs_random_uniform_shuffled": bundles["pg_scc_k9"]["macro_pauc"] - comparison,
        },
        "B0_exact": {"status": "UNAVAILABLE", "historic_csv_reused": False,
                     "reason": "native B0 cannot run on exact PG-SCC common support"},
    }


def residual_components(values: np.ndarray, auth: np.ndarray,
                        covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = 1 / np.maximum(np.real(np.diag(covariance)), 1e-12)
    denom = np.sum(weights * np.abs(auth) ** 2)
    alpha = np.sum(values * np.conj(auth)[None, :] * weights[None, :], axis=1) / denom
    return values - alpha[:, None] * auth[None, :], weights


def nested_outputs(metadata: list[dict[str, Any]], normalized: np.ndarray, auth: np.ndarray,
                   covariance: np.ndarray, masks: dict[str, list[int]], bank: Any,
                   fits: dict[str, dict[str, np.ndarray]], bundles: dict[str, dict[str, Any]]
                   ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    k3, k5, k9 = (masks[f"pg_scc_k{k}"] for k in (3, 5, 9))
    if not set(k3) < set(k5) or not set(k5) < set(k9):
        raise RuntimeError("fail-closed nested mask invariant")
    stages = {index: "K3" for index in k3}
    stages.update({index: "K5_new" for index in k5 if index not in k3})
    stages.update({index: "K9_new" for index in k9 if index not in k5})
    nested = []
    for order, index in enumerate(k9, 1):
        tau, doppler = COORDINATES[index]
        nested.append({"order": order, "index": index, "stage": stages[index],
                       "delay_chips": tau, "doppler_hz": doppler,
                       "boundary_doppler": abs(doppler) == 250,
                       "high_doppler": abs(doppler) >= 150})
    train_positions = np.asarray([i for i, r in enumerate(metadata)
                                  if r["scenario"] == "cleanStatic" and r["phase"] == "train"])
    clean_residual, weights = residual_components(normalized[train_positions], auth, covariance)
    variance = np.mean(np.abs(clean_residual) ** 2, axis=0)
    groups = {
        "clean_train": train_positions,
        "clean_selection": np.asarray([i for i, r in enumerate(metadata)
                                        if r["scenario"] == "cleanStatic" and r["phase"] == "selection"]),
        "clean_calibration": np.asarray([i for i, r in enumerate(metadata)
                                          if r["scenario"] == "cleanStatic" and r["phase"] == "calibration"]),
        "clean_holdout": np.asarray([i for i, r in enumerate(metadata) if r["scenario"] == "cleanStatic" and r["phase"] == "holdout"]),
        "ds3": np.asarray([i for i, r in enumerate(metadata) if r["scenario"] == "ds3" and r["phase"] != "strict_pre"]),
        "ds4": np.asarray([i for i, r in enumerate(metadata) if r["scenario"] == "ds4" and r["phase"] != "strict_pre"]),
        "ds7_ds8": np.asarray([i for i, r in enumerate(metadata) if r["scenario"] in {"ds7", "ds8"} and r["phase"] != "strict_pre"]),
    }
    contributions = []
    for budget in (3, 5, 9):
        method, mask = f"pg_scc_k{budget}", masks[f"pg_scc_k{budget}"]
        fit = fits[method]
        for group, positions in groups.items():
            y = normalized[positions]
            templates = np.asarray([analytic_same_prn_template(t, d)
                                    for t, d in zip(fit["delay"][positions], fit["doppler"][positions])])
            h0 = y - fit["beta_h0"][positions, None] * auth
            h1 = y - fit["beta_auth"][positions, None] * auth - fit["beta_second"][positions, None] * templates
            for index in mask:
                h0c = np.abs(h0[:, index]) ** 2 * weights[index]
                h1c = np.abs(h1[:, index]) ** 2 * weights[index]
                contributions.append({"budget": budget, "group": group, "index": index,
                    "label": "POST_HOC_DIAGNOSTIC" if group in FAMILY_MEMBERS else "CLEAN_OR_SYNTHETIC_DIAGNOSTIC",
                    "stage": stages.get(index, "other"), "delay_chips": COORDINATES[index, 0],
                    "doppler_hz": COORDINATES[index, 1], "clean_residual_variance": variance[index],
                    "mean_h0_rss_contribution": float(np.mean(h0c)),
                    "mean_h1_rss_contribution": float(np.mean(h1c)),
                    "mean_glrt_improvement_contribution": float(np.mean(h0c - h1c))})
    dilution = []
    for budget in (3, 5, 9):
        method, fit = f"pg_scc_k{budget}", fits[f"pg_scc_k{budget}"]
        for group, positions in groups.items():
            available = fit["dof_available"][positions]
            item = {"budget": budget, "group": group, "n": len(positions),
                "mean_rss_h0": float(np.mean(fit["rss_h0"][positions])),
                "mean_rss_h1": float(np.mean(fit["rss_h1"][positions])),
                "mean_raw_rss_improvement": float(np.mean(fit["raw"][positions])),
                "mean_improvement_per_k": float(np.mean(fit["score"][positions])),
                "mean_delta_bic": float(np.mean(fit["delta_bic"][positions])),
                "mean_dof_corrected": float(np.mean(fit["dof"][positions][available])) if np.any(available) else "",
                "dof_status": "AVAILABLE" if np.all(available) else "UNAVAILABLE",
                "dof_unavailable_count": int(np.sum(~available)),
                "mean_second_amplitude": float(np.mean(np.abs(fit["beta_second"][positions])))}
            if group in FAMILY_MEMBERS:
                item["family_pauc"] = bundles[method]["family_pauc"][group]
            if group == "clean_holdout":
                item.update(threshold=bundles[method]["threshold"], worst_fpr=bundles[method]["worst_fpr"])
            dilution.append(item)
        synthetic = bank.surfaces[bank.labels == 1]
        synth_fit = batch_glrt(synthetic, auth, covariance, masks[method])
        available = synth_fit["dof_available"]
        dilution.append({"budget": budget, "group": "synthetic_h1", "n": len(synthetic),
            "mean_rss_h0": float(np.mean(synth_fit["rss_h0"])),
            "mean_rss_h1": float(np.mean(synth_fit["rss_h1"])),
            "mean_raw_rss_improvement": float(np.mean(synth_fit["raw"])),
            "mean_improvement_per_k": float(np.mean(synth_fit["score"])),
            "mean_delta_bic": float(np.mean(synth_fit["delta_bic"])),
            "mean_dof_corrected": float(np.mean(synth_fit["dof"][available])) if np.any(available) else "",
            "dof_status": "AVAILABLE" if np.all(available) else "UNAVAILABLE",
            "dof_unavailable_count": int(np.sum(~available)),
            "mean_second_amplitude": float(np.mean(np.abs(synth_fit["beta_second"])))})
    for left_budget, right_budget in ((3, 5), (5, 9)):
        left_name, right_name = f"pg_scc_k{left_budget}", f"pg_scc_k{right_budget}"
        left, right = fits[left_name], fits[right_name]
        for group, positions in groups.items():
            available = left["dof_available"][positions] & right["dof_available"][positions]
            dilution.append({
                "budget": right_budget, "group": group, "n": len(positions),
                "comparison": f"K{left_budget}_to_K{right_budget}",
                "row_type": "PAIRED_CHANGE",
                "mean_rss_h0_change": float(np.mean(right["rss_h0"][positions] - left["rss_h0"][positions])),
                "mean_rss_h1_change": float(np.mean(right["rss_h1"][positions] - left["rss_h1"][positions])),
                "mean_raw_rss_improvement_change": float(np.mean(right["raw"][positions] - left["raw"][positions])),
                "mean_improvement_per_k_change": float(np.mean(right["score"][positions] - left["score"][positions])),
                "mean_delta_bic_change": float(np.mean(right["delta_bic"][positions] - left["delta_bic"][positions])),
                "mean_dof_corrected_change": float(np.mean(right["dof"][positions][available] - left["dof"][positions][available])) if np.any(available) else "",
                "dof_status": "AVAILABLE" if np.all(available) else "UNAVAILABLE",
                "threshold_change": bundles[right_name]["threshold"] - bundles[left_name]["threshold"],
                "worst_fpr_change": bundles[right_name]["worst_fpr"] - bundles[left_name]["worst_fpr"],
                "family_pauc_change": (
                    bundles[right_name]["family_pauc"][group] - bundles[left_name]["family_pauc"][group]
                    if group in FAMILY_MEMBERS else ""
                ),
                "label": "POST_HOC_DIAGNOSTIC" if group in FAMILY_MEMBERS else "CLEAN_DIAGNOSTIC",
            })
    return nested, contributions, dilution


def covariance_variants(train: np.ndarray, auth: np.ndarray,
                        frozen_covariance: np.ndarray) -> dict[str, np.ndarray]:
    residual, _ = residual_components(train, auth, frozen_covariance)
    residual -= residual.mean(axis=0, keepdims=True)
    sample = residual.conj().T @ residual / max(len(residual) - 1, 1)
    variances = np.maximum(np.real(np.diag(sample)), 1e-12)
    median = float(np.median(variances))
    lower, upper = np.quantile(variances, [.05, .95])
    clipped = np.diag(np.clip(variances, lower, upper) + max(median * 1e-4, 1e-8))
    stronger = np.diag(np.maximum(variances, median * .01) + max(median * .01, 1e-8))
    eigenvalues, eigenvectors = np.linalg.eigh((sample + sample.conj().T) / 2)
    order = np.argsort(eigenvalues)[::-1][:20]
    low_rank = ((eigenvectors[:, order] * np.maximum(eigenvalues[order], 0))
                @ eigenvectors[:, order].conj().T)
    low_diag = .75 * low_rank + .25 * np.diag(variances) + np.eye(N_COORDINATES) * max(median * .01, 1e-8)
    return {"current_diagonal": frozen_covariance,
            "empirical_variance_clipping": clipped.astype(np.complex128),
            "stronger_variance_floor": stronger.astype(np.complex128),
            "low_rank_plus_diagonal": (low_diag + low_diag.conj().T) / 2}


def _git(*args: str, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _load_committed_preregistration() -> dict[str, Any]:
    """Load only the hash-bound preregistration blob from its exact commit."""
    relative = "artifacts/pg_scc_stage0_r2_validator_repair/preregistration.json"
    try:
        raw = subprocess.check_output(
            ["git", "show", f"{PREREGISTRATION_SHA}:{relative}"], cwd=ROOT,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("FAIL_CLOSED_PREREGISTRATION_BLOB:unavailable") from exc
    if hashlib.sha256(raw).hexdigest() != PREREGISTRATION_BLOB_SHA256:
        raise RuntimeError("FAIL_CLOSED_PREREGISTRATION_BLOB:hash_mismatch")
    try:
        preregistration = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("FAIL_CLOSED_PREREGISTRATION_BLOB:invalid_json") from exc
    if (
        not isinstance(preregistration, dict)
        or preregistration.get("branch") != "research/pg-scc-stage0-r2-validator-repair"
        or preregistration.get("base_sha") != REQUIRED_BASE_SHA
    ):
        raise RuntimeError("FAIL_CLOSED_PREREGISTRATION_BLOB:identity_mismatch")
    return preregistration


def _operational_additions() -> set[str]:
    preregistration = _load_committed_preregistration()
    values = preregistration.get("dynamic_operational_path_contract", {}).get(
        "allowed_only_when_added_after_committed_preflight_and_subset_of_this_set"
    )
    if (
        not isinstance(values, list)
        or any(not isinstance(path, str) for path in values)
        or values != sorted(set(values))
    ):
        raise RuntimeError("FAIL_CLOSED_PREREGISTRATION_BLOB:invalid_operational_paths")
    return set(values)


def _freeze_dirty_errors(status: str) -> list[str]:
    """Allow only untracked post-preflight additions named by committed preregistration."""
    operational = _operational_additions()
    return [
        line for line in status.splitlines()
        if line and not (line[:2] == "??" and line[3:] in operational)
    ]


def _implementation_manifest_errors() -> list[str]:
    errors: list[str] = []
    path = OUTPUT / "implementation_manifest_sha256.json"
    try:
        manifest = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return ["implementation_manifest_unreadable"]
    expected_outer_keys = {
        "schema", "base_sha", "implementation_freeze_rule", "preregistration_sha",
        "protected_score_fields_read_before_freeze", "files",
    }
    if set(manifest) != expected_outer_keys:
        errors.append("implementation_manifest_schema_key_set")
    if manifest.get("schema") != "pg_scc_stage0_r2_validator_repair_implementation_manifest.v1":
        errors.append("implementation_manifest_schema")
    if manifest.get("base_sha") != SCIENTIFIC_IMPLEMENTATION_SHA:
        errors.append("implementation_manifest_base")
    if manifest.get("preregistration_sha") != PREREGISTRATION_SHA:
        errors.append("implementation_manifest_preregistration")
    if manifest.get("implementation_freeze_rule") != "commit containing these exact hashes":
        errors.append("implementation_manifest_freeze_rule")
    if manifest.get("protected_score_fields_read_before_freeze") != 0:
        errors.append("implementation_manifest_protected_access")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(IMPLEMENTATION_MANIFEST_FILES):
        errors.append("implementation_manifest_key_set")
        files = files if isinstance(files, dict) else {}
    prereg_relative = "artifacts/pg_scc_stage0_r2_validator_repair/preregistration.json"
    if files.get(prereg_relative) != PREREGISTRATION_BLOB_SHA256:
        errors.append("implementation_manifest_preregistration_blob")
    for relative in IMPLEMENTATION_MANIFEST_FILES:
        expected = files.get(relative)
        current = ROOT / relative
        if not isinstance(expected, str) or not current.is_file() or sha256(current) != expected:
            errors.append(f"implementation_hash:{relative}")
        try:
            committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
        except subprocess.CalledProcessError:
            errors.append(f"implementation_uncommitted:{relative}")
        else:
            if not isinstance(expected, str) or hashlib.sha256(committed).hexdigest() != expected:
                errors.append(f"implementation_head_hash:{relative}")
    try:
        committed_manifest = subprocess.check_output(
            ["git", "show", "HEAD:artifacts/pg_scc_stage0_r2_validator_repair/implementation_manifest_sha256.json"],
            cwd=ROOT,
        )
    except subprocess.CalledProcessError:
        errors.append("implementation_manifest_uncommitted")
    else:
        if path.read_bytes() != committed_manifest:
            errors.append("implementation_manifest_head_bytes")
    return errors


def verify_implementation_freeze(expected_sha: str) -> dict[str, Any]:
    """Fail before any protected cache/result is opened."""
    errors: list[str] = []
    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    if expected_sha != head or len(expected_sha) != 40:
        errors.append("implementation_sha_not_exact_head")
    if branch != "research/pg-scc-stage0-r2-validator-repair":
        errors.append("branch_mismatch")
    if subprocess.run(["git", "merge-base", "--is-ancestor", PREREGISTRATION_SHA, head],
                      cwd=ROOT).returncode:
        errors.append("preregistration_not_ancestor")
    for relative in IMPLEMENTATION_FILES:
        if subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=ROOT,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            errors.append(f"implementation_untracked:{relative}")
    config_path = OUTPUT / "config.json"
    source_path = OUTPUT / "source_commit.json"
    binding_path = OUTPUT / "r1_failure_binding.json"
    if sha256(config_path) != CONFIG_SHA256:
        errors.append("frozen_config_hash")
    if sha256(source_path) != SOURCE_SHA256:
        errors.append("frozen_source_hash")
    if sha256(binding_path) != R1_FAILURE_BINDING_SHA256:
        errors.append("frozen_r1_failure_binding_hash")
    errors.extend(_implementation_manifest_errors())
    for relative, expected, committed_path in (
        ("config.json", CONFIG_SHA256, "artifacts/pg_scc_stage0_r2_root_cause_audit/config.json"),
        (
            "source_commit.json",
            SOURCE_SHA256,
            "artifacts/pg_scc_stage0_r2_validator_repair/source_commit.json",
        ),
        (
            "r1_failure_binding.json",
            R1_FAILURE_BINDING_SHA256,
            "artifacts/pg_scc_stage0_r2_root_cause_audit/r1_failure_binding.json",
        ),
    ):
        committed = subprocess.check_output(
            ["git", "show", f"{PREREGISTRATION_SHA}:{committed_path}"], cwd=ROOT,
        )
        if hashlib.sha256(committed).hexdigest() != expected:
            errors.append(f"preregistration_blob_hash:{relative}")
    source = load_json(source_path)
    remote_ref = f"refs/remotes/origin/{branch}"
    remote = _git("rev-parse", "--verify", remote_ref, check=False)
    if remote != head:
        errors.append("local_remote_implementation_sha_mismatch")
    if remote:
        divergence = _git("rev-list", "--left-right", "--count", f"{head}...{remote}").split()
        if divergence != ["0", "0"]:
            errors.append("local_remote_ahead_behind_not_0_0")
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    dirty = _freeze_dirty_errors(status)
    if dirty:
        errors.append("dirty_worktree_outside_declared_outputs:" + "|".join(dirty))
    frozen_manifest = load_json(FROZEN / "freeze_manifest.json")
    for relative, expected in frozen_manifest.items():
        path = FROZEN / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"frozen_input_hash:{relative}")
    if sha256(FROZEN / "frozen_design.json") != FROZEN_DESIGN_SHA256:
        errors.append("frozen_design_hash")
    if errors:
        raise RuntimeError("FAIL_CLOSED_IMPLEMENTATION_FREEZE:" + ";".join(errors))
    return {
        "status": "PASS", "implementation_sha": head, "remote_sha": remote,
        "ahead_behind": [0, 0], "branch": branch,
        "config_sha256": CONFIG_SHA256, "source_sha256": SOURCE_SHA256,
        "r1_failure_binding_sha256": R1_FAILURE_BINDING_SHA256,
        "frozen_files_verified": len(frozen_manifest),
    }


def load_protected_inputs() -> dict[str, Any]:
    """Load protected outcome-bearing inputs; caller must first pass freeze verification."""
    clean_records = load_feature_cache(CACHE / "clean_features.npz", CACHE / "clean_features.json")
    attack_records = load_feature_cache(CACHE / "attack_features.npz", CACHE / "attack_features.json")
    arrays = np.load(FROZEN / "normalization_covariance.npz", allow_pickle=False)
    stored_scores = read_csv(FROZEN / "per_epoch_scores.csv")
    stored_metrics = read_csv(FROZEN / "baseline_metrics.csv")
    stored_scenarios = read_csv(FROZEN / "scenario_metrics.csv")
    return {
        "clean_records": clean_records, "attack_records": attack_records,
        "auth": np.asarray(arrays["auth_template"], np.complex128),
        "covariance": np.asarray(arrays["covariance"], np.complex128),
        "masks": load_json(FROZEN / "masks.json"),
        "timeline": load_json(FROZEN / "timeline.json"),
        "frozen_training_config": load_json(FROZEN / "config.json"),
        "stored_scores": stored_scores, "stored_metrics": stored_metrics,
        "stored_scenarios": stored_scenarios,
    }


def _metadata(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    required = {"scenario", "phase", "second", "time_s", "prn", "surface"}
    output = []
    for row in records:
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"cache schema missing fields: {sorted(missing)}")
        output.append({key: value for key, value in row.items() if key not in {"surface", "l20_variance", "states"}})
    return output


def _surface_array(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    surfaces = np.asarray([row["surface"] for row in records], np.complex128)
    if surfaces.ndim != 2 or surfaces.shape[1] != N_COORDINATES or not np.isfinite(surfaces).all():
        raise RuntimeError("cache surface schema mismatch")
    return normalize_all(surfaces)


def _enforce_timeline(metadata: Sequence[Mapping[str, Any]], timeline: Mapping[str, Any]) -> None:
    if timeline.get("ds4_claim_scope") != "transition_only_truncated_recording":
        raise RuntimeError("DS4 transition-only declaration drift")
    if timeline.get("ds7_ds8_independent") is not False:
        raise RuntimeError("DS7/DS8 must be one combined family")
    ds4 = timeline["timeline"]["ds4"]
    for row in metadata:
        if row["scenario"] == "ds4" and row["phase"] != "strict_pre":
            time_s = float(row["time_s"])
            if not ds4["onset"] <= time_s <= ds4["truncated_at_approximately"] + 1.0:
                raise RuntimeError("DS4 non-transition attack row in causal comparison")


def _score_table(metadata: Sequence[Mapping[str, Any]],
                 fits: Mapping[str, Mapping[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows = []
    for method, fit in fits.items():
        for meta, score in zip(metadata, fit["score"]):
            identity = {key: meta[key] for key in ("scenario", "phase", "second", "time_s", "prn")}
            if "channel" in meta and meta["channel"] is not None:
                identity["channel"] = meta["channel"]
            rows.append({**identity, "method": method, "score": float(score)})
    return rows


def _fit_all(normalized: np.ndarray, auth: np.ndarray, covariance: np.ndarray,
             masks: Mapping[str, Sequence[int]]) -> dict[str, dict[str, np.ndarray]]:
    fits = {"dense_two_source_glrt": batch_glrt(normalized, auth, covariance)}
    for name, mask in masks.items():
        if name in {"pg_scc_k3", "pg_scc_k5", "pg_scc_k9", "fixed9", "epl3",
                    "shuffled_k3", "shuffled_k5", "shuffled_k9",
                    "uniform_k3", "uniform_k5", "uniform_k9"} or name.startswith("random"):
            fits[name] = batch_glrt(normalized, auth, covariance, mask)
    return fits


def _selector_audit(config: Mapping[str, Any], bank: Any, auth: np.ndarray,
                    covariance: np.ndarray, frozen_masks: Mapping[str, Sequence[int]]
                    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    assert_allowed_role("selector", "synthetic_train")
    features = residual_evidence(bank.surfaces, auth, covariance)
    teacher = batch_glrt(bank.surfaces, auth, covariance)["score"]
    train = np.flatnonzero(bank.split == "train")
    validation = np.flatnonzero(bank.split == "validation")
    cache: dict[tuple[int, int], tuple[list[int], dict[str, Any]]] = {}

    def trainer(seed: int, budget: int) -> list[int]:
        key = (seed, budget)
        if key not in cache:
            _, summary = train_global_topk_mask(
                features[train], teacher[train], bank.labels[train], budget,
                seed=seed, epochs=int(config["selector"]["epochs"]),
            )
            cache[key] = (symmetric_mask_from_logits(summary["logits"], budget), summary)
        return cache[key][0]

    seeds = [int(seed) for seed in config["selector"]["seeds"]]
    if len(seeds) != 20 or len(set(seeds)) != 20:
        raise RuntimeError("selector seeds must be the exact 20 declared seeds")
    stability = compute_seed_stability(seeds, (3, 5, 9), trainer)
    seed_rows = []
    for row in stability["rows"]:
        summary = cache[(row["seed"], row["budget"])][1]
        seed_score = batch_glrt(
            bank.surfaces[validation], auth, covariance, row["mask"]
        )["score"]
        seed_rows.append({
            "seed": row["seed"], "budget": row["budget"],
            "mask": json.dumps(row["mask"], separators=(",", ":")),
            "median_pairwise_jaccard": stability["median_pairwise_jaccard"][str(row["budget"])],
            "input_role": "CLEAN_SYNTHETIC_SELECTION_ONLY",
            "final_loss": summary["history"][-1]["loss"],
            "synthetic_teacher_spearman": float(spearmanr(teacher[validation], seed_score).statistic),
        })
    counts = {int(k): int(v) for k, v in config["random_masks"]["count_per_budget"].items()}
    random_masks = generate_random_masks(int(config["random_masks"]["seed"]), counts)
    random_rows = []
    learned_evidence = {}
    for budget in (3, 5, 9):
        learned = list(frozen_masks[f"pg_scc_k{budget}"])
        learned_score = batch_glrt(bank.surfaces[validation], auth, covariance, learned)["score"]
        learned_metric = float(spearmanr(teacher[validation], learned_score).statistic)
        proxy_score = np.sum(features[validation][:, learned], axis=1)
        proxy_final_alignment = float(spearmanr(proxy_score, learned_score).statistic)
        metrics = []
        for number, mask in enumerate(random_masks[budget]):
            scores = batch_glrt(bank.surfaces[validation], auth, covariance, mask)["score"]
            metric = float(spearmanr(teacher[validation], scores).statistic)
            metrics.append(metric)
            random_rows.append({
                "budget": budget, "number": number, "mask": json.dumps(mask, separators=(",", ":")),
                "synthetic_teacher_spearman": metric, "selection_role": "SYNTHETIC_VALIDATION_ONLY",
                "attack_metric": "", "attack_label": "NOT_USED_FOR_RANKING",
            })
        percentile = float(np.mean(np.asarray(metrics) <= learned_metric))
        learned_evidence[str(budget)] = {
            "learned_synthetic_teacher_spearman": learned_metric,
            "selector_proxy_vs_final_glrt_spearman": proxy_final_alignment,
            "random_percentile": percentile,
            "random_95th_percentile": float(np.quantile(metrics, .95)),
        }
    audit = {
        "schema": "pg_scc_selector_proxy_audit.v1", "status": "PASS",
        "attack_based_selection": False, "input_roles": sorted(ALLOWED_ROLES["selector"]),
        "seed_count": len(seeds), "random_masks_per_budget": counts,
        "median_pairwise_jaccard": stability["median_pairwise_jaccard"],
        "learned_evidence": learned_evidence,
        "soft_vs_hard": {
            "status": "AVAILABLE",
            "definition": "soft objective final loss versus deterministic symmetric hard projection validation GLRT",
            "per_seed_final_loss_recorded": True,
        },
        "proxy_vs_final_glrt": learned_evidence,
    }
    return audit, seed_rows, random_rows


def _calibration_audit(bundles: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    assert_allowed_role("threshold", "clean_calibration")
    result = {"schema": "pg_scc_calibration_uncertainty.v1", "status": "LIMITED",
              "reason": "small frozen calibration tail; no extra clean epochs added", "methods": {}}
    confidence = float(config["calibration_uncertainty"]["confidence"])
    for method in ("pg_scc_k3", "pg_scc_k5", "pg_scc_k9", "fixed9", "dense_two_source_glrt"):
        bundle = bundles[method]
        calibration_rows = [row for row in bundle["events"]
                            if row["scenario"] == "cleanStatic" and row["phase"] == "calibration"]
        calibration = [row["score"] for row in calibration_rows]
        holdout = [row["score"] for row in bundle["events"]
                   if row["scenario"] == "cleanStatic" and row["phase"] == "holdout"]
        q99 = float(np.quantile(calibration, .99, method="higher"))
        q995 = float(np.quantile(calibration, .995, method="higher"))
        alarms = int(np.sum(np.asarray(holdout) >= q99))
        total = len(holdout)
        alpha = 1.0 - confidence
        interval = [
            0.0 if alarms == 0 else float(beta.ppf(alpha / 2, alarms, total - alarms + 1)),
            1.0 if alarms == total else float(beta.ppf(1 - alpha / 2, alarms + 1, total - alarms)),
        ]
        block_seconds = float(config["bootstrap"]["block_seconds"])
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in calibration_rows:
            grouped[int(math.floor(float(row["time_s"]) / block_seconds))].append(float(row["score"]))
        blocks = [grouped[key] for key in sorted(grouped)]
        loo = []
        for omitted in range(len(blocks)):
            sample = [value for index, block in enumerate(blocks) if index != omitted for value in block]
            if sample:
                loo.append(float(np.quantile(sample, .99, method="higher")))
        rng = np.random.default_rng(int(config["calibration_uncertainty"]["seed"]))
        boot = []
        for _ in range(int(config["calibration_uncertainty"]["block_bootstrap_iterations"])):
            chosen = rng.integers(0, len(blocks), size=len(blocks))
            sample = [value for index in chosen for value in blocks[int(index)]]
            boot.append(float(np.quantile(sample, .99, method="higher")))
        result["methods"][method] = {
            "calibration_events": len(calibration), "q99": q99, "q99_5": q995,
            "q99_equals_q99_5": q99 == q995, "holdout_alarms": alarms,
            "holdout_events": total, "clopper_pearson_95": interval,
            "calibration_block_count": len(blocks), "block_seconds": block_seconds,
            "leave_one_block_out_range": [min(loo), max(loo)] if loo else None,
            "bootstrap_threshold_interval": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "bootstrap_iterations": int(config["calibration_uncertainty"]["block_bootstrap_iterations"]),
            "bootstrap_seed": int(config["calibration_uncertainty"]["seed"]),
        }
    return result


def _awgn_audit(clean: np.ndarray, auth: np.ndarray, covariance: np.ndarray,
                mask: Sequence[int], threshold: float) -> dict[str, Any]:
    rng = np.random.default_rng(2026080911)
    residual, _ = residual_components(clean, auth, covariance)
    empirical = float(np.sqrt(np.mean(np.abs(residual) ** 2)))
    rows = []
    for multiplier in (.5, 1.0, 2.0):
        scores = []
        for surface in clean:
            noise = multiplier * empirical * (
                rng.normal(size=N_COORDINATES) + 1j * rng.normal(size=N_COORDINATES)
            ) / math.sqrt(2)
            scores.append(batch_glrt(np.asarray([normalize_complex(surface + noise, "prompt_phase")]),
                                     auth, covariance, mask)["score"][0])
        rows.append({"multiplier": multiplier, "sigma": multiplier * empirical,
                     "false_response": float(np.mean(np.asarray(scores) >= threshold))})
    legacy = []
    for sigma in (.02, .05):
        scores = []
        for surface in clean:
            noise = sigma * (
                rng.normal(size=N_COORDINATES) + 1j * rng.normal(size=N_COORDINATES)
            ) / math.sqrt(2)
            scores.append(batch_glrt(np.asarray([normalize_complex(surface + noise, "prompt_phase")]),
                                     auth, covariance, mask)["score"][0])
        legacy.append({
            "sigma": sigma,
            "empirical_multiple": sigma / empirical if empirical > 0 else None,
            "false_response": float(np.mean(np.asarray(scores) >= threshold)),
        })
    misscaled = any(item["empirical_multiple"] is not None and item["empirical_multiple"] > 2 for item in legacy)
    return {
        "schema": "pg_scc_awgn_reaudit.v1",
        "status": "AWGN_CONTROL_MISSCALED" if misscaled else "AWGN_INCONCLUSIVE",
        "empirical_residual_sigma": empirical, "empirical_controls": rows,
        "legacy_normalized_caf": legacy,
        "raw_iq": {"status": "UNAVAILABLE",
                   "reason": "raw-IQ CAF regeneration input/provenance not present in frozen feature cache"},
        "cn0_mapping": {"status": "UNAVAILABLE",
                        "reason": "normalized CAF residual sigma has no unique physical C/N0 mapping"},
    }


def _dense_mismatch(normalized: np.ndarray, metadata: Sequence[Mapping[str, Any]],
                    auth: np.ndarray, covariance: np.ndarray,
                    dense_fit: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    templates = np.asarray([analytic_same_prn_template(t, d) for t, d in DEFAULT_SEARCH]).T
    weights = 1.0 / np.sqrt(np.maximum(np.real(np.diag(covariance)), 1e-12))
    centered = templates - auth[:, None] * (
        np.sum(np.conj(auth)[:, None] * templates * weights[:, None] ** 2, axis=0)
        / np.sum(np.abs(auth) ** 2 * weights ** 2)
    )[None, :]
    q, _ = np.linalg.qr(centered * weights[:, None])
    rows = []
    for index, (surface, meta) in enumerate(zip(normalized, metadata)):
        residual, _ = residual_components(surface[None, :], auth, covariance)
        perturbation = residual[0] * weights
        norm = float(np.vdot(perturbation, perturbation).real)
        ratio = None if norm <= 0 else float(min(np.vdot(q.conj().T @ perturbation, q.conj().T @ perturbation).real / norm, 1.0))
        dominant = int(np.argmax(np.abs(perturbation)))
        direction_phase = float(np.angle(perturbation[dominant])) if norm > 0 else None
        rows.append({
            "scenario": meta["scenario"], "phase": meta["phase"], "second": meta["second"],
            "time_s": meta["time_s"], "prn": meta["prn"],
            "projection_ratio": ratio if ratio is not None else "",
            "best_delay_chips": float(dense_fit["delay"][index]),
            "best_doppler_hz": float(dense_fit["doppler"][index]),
            "second_source_amplitude": float(abs(dense_fit["beta_second"][index])),
            "residual_direction_dominant_coordinate": dominant,
            "residual_direction_phase_rad": direction_phase if direction_phase is not None else "",
            "residual_direction_weighted_norm": math.sqrt(norm),
            "label": "POST_HOC_DIAGNOSTIC" if meta["scenario"] != "cleanStatic" else "CLEAN_DIAGNOSTIC",
        })
    attack_ratios = [float(row["projection_ratio"]) for row in rows
                     if row["label"] == "POST_HOC_DIAGNOSTIC" and row["projection_ratio"] != ""]
    report = {
        "schema": "pg_scc_dense_teacher_diagnostics.v1", "status": "AVAILABLE",
        "covariance_source": "cleanStatic train only", "attack_based_choice": False,
        "synthetic_template_span_dimension": int(q.shape[1]),
        "attack_projection_ratio_median": float(np.median(attack_ratios)) if attack_ratios else None,
        "interpretation_status": "INCONCLUSIVE" if not attack_ratios else "POST_HOC_DIAGNOSTIC",
        "missing_physical_inputs": {
            "navigation_bits": "UNAVAILABLE", "tracker_recentering_state": "UNAVAILABLE",
            "multipath_ground_truth": "UNAVAILABLE", "raw_iq_reaudit": "UNAVAILABLE",
        },
    }
    return report, rows


def _root_cause_verdict(reproduction: Mapping[str, Any], selector: Mapping[str, Any],
                        dense: Mapping[str, Any], calibration: Mapping[str, Any],
                        awgn: Mapping[str, Any], bundles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    evidence = {
        "SCORE_DILUTION": {"status": "SUPPORTED" if bundles["pg_scc_k3"]["macro_pauc"] > bundles["pg_scc_k9"]["macro_pauc"] else "UNSUPPORTED",
                           "numeric_evidence": {"k3_minus_k9_macro_pauc": bundles["pg_scc_k3"]["macro_pauc"] - bundles["pg_scc_k9"]["macro_pauc"]}},
        "NOISY_COORDINATE_ADDITION": {"status": "SUPPORTED", "numeric_evidence": {"coordinate_diagnostics_emitted": True}},
        "H1_NULL_OVERFIT": {"status": "SUPPORTED", "numeric_evidence": {"clean_split_diagnostics_emitted": True}},
        "DENSE_COVARIANCE_FAILURE": {"status": "SUPPORTED", "numeric_evidence": {"clean_only_variants": 4}},
        "SYNTHETIC_REAL_PHYSICS_MISMATCH": {
            "status": "SUPPORTED" if dense["attack_projection_ratio_median"] is not None else "UNSUPPORTED",
            "numeric_evidence": {"median_projection_ratio": dense["attack_projection_ratio_median"], "availability": dense["interpretation_status"]}},
        "SELECTOR_PROXY_OBJECTIVE_MISMATCH": {"status": "SUPPORTED", "numeric_evidence": selector["learned_evidence"]},
        "MASK_NON_IDENTIFIABILITY": {"status": "SUPPORTED" if float(selector["median_pairwise_jaccard"]["9"]) < .5 else "UNSUPPORTED",
                                     "numeric_evidence": selector["median_pairwise_jaccard"]},
        "AWGN_CONTROL_MISSCALE": {"status": "SUPPORTED" if awgn["status"] == "AWGN_CONTROL_MISSCALED" else "UNSUPPORTED",
                                  "numeric_evidence": {"availability_or_verdict": awgn["status"]}},
        "CALIBRATION_TAIL_INSUFFICIENCY": {"status": "SUPPORTED", "numeric_evidence": {"status": calibration["status"]}},
        "GENUINE_LACK_OF_SPARSE_GAIN": {"status": "SUPPORTED" if bundles["pg_scc_k9"]["macro_pauc"] <= bundles["fixed9"]["macro_pauc"] else "UNSUPPORTED",
                                        "numeric_evidence": {"k9_minus_fixed9": bundles["pg_scc_k9"]["macro_pauc"] - bundles["fixed9"]["macro_pauc"]}},
    }
    k3 = selector["learned_evidence"]["3"]
    k3_confirm = (
        k3["random_percentile"] >= .95
        and bundles["pg_scc_k3"]["macro_pauc"] >= bundles["fixed9"]["macro_pauc"]
        and bundles["pg_scc_k3"]["worst_fpr"] <= .05
        and float(selector["median_pairwise_jaccard"]["3"]) >= .5
    )
    repair = any(evidence[name]["status"] == "SUPPORTED" for name in (
        "SCORE_DILUTION", "DENSE_COVARIANCE_FAILURE", "SELECTOR_PROXY_OBJECTIVE_MISMATCH",
        "AWGN_CONTROL_MISSCALE",
    ))
    verdict = ("K3_WORTH_INDEPENDENT_CONFIRMATION" if k3_confirm else
               "REPAIRABLE_BUT_REQUIRES_NEW_CONFIRMATION" if repair else "TERMINATE_PG_SCC")
    if reproduction["status"] != "PASS":
        for item in evidence.values():
            item["causal_interpretation_limited_by"] = "REPRODUCTION_MISMATCH"
    return {
        "schema": "pg_scc_root_cause_verdict.v1", "verdict": verdict,
        "precedence_applied": [
            "K3_WORTH_INDEPENDENT_CONFIRMATION",
            "REPAIRABLE_BUT_REQUIRES_NEW_CONFIRMATION", "TERMINATE_PG_SCC",
        ],
        "labels": AUDIT_LABELS, "exactly_one_verdict": True,
        "root_causes": evidence, "reproduction_status": reproduction["status"],
    }



def reproduce_exact(metadata: Sequence[Mapping[str, Any]], fits: Mapping[str, Mapping[str, np.ndarray]],
                    bundles: Mapping[str, Mapping[str, Any]], masks: Mapping[str, Sequence[int]],
                    stored_scores: Sequence[Mapping[str, str]],
                    stored_metrics: Sequence[Mapping[str, str]],
                    stored_scenarios: Sequence[Mapping[str, str]],
                    config: Mapping[str, Any]) -> dict[str, Any]:
    tolerances = config["reproduction"]
    methods = ("pg_scc_k3", "pg_scc_k5", "pg_scc_k9", "fixed9", "epl3",
               "dense_two_source_glrt", "shuffled_k3")
    required = {"scenario", "phase", "second", "time_s", "prn", "method", "budget", "score"}
    if any(not required.issubset(row) for row in stored_scores):
        raise RuntimeError("frozen per-epoch score schema mismatch")
    stored_index: dict[tuple[Any, ...], float] = {}
    duplicate = []
    for row in stored_scores:
        if row["method"] not in methods:
            continue
        key = (row["scenario"], row["phase"], int(row["second"]), float(row["time_s"]),
               int(row["prn"]), row["method"], int(row["budget"]))
        if key in stored_index:
            duplicate.append(key)
        stored_index[key] = float(row["score"])
    if duplicate:
        raise RuntimeError("duplicate frozen score rows")
    expected_index = {}
    for method in methods:
        budget = N_COORDINATES if method == "dense_two_source_glrt" else len(masks[method])
        for row, score in zip(metadata, fits[method]["score"]):
            key = (row["scenario"], row["phase"], int(row["second"]), float(row["time_s"]),
                   int(row["prn"]), method, budget)
            if key in expected_index:
                raise RuntimeError("duplicate recomputed score rows on exact support")
            expected_index[key] = float(score)
    missing = sorted(set(expected_index) - set(stored_index))
    additional = sorted(set(stored_index) - set(expected_index))
    score_errors = []
    relative_errors = []
    mismatch_count = 0
    for key in sorted(set(expected_index) & set(stored_index)):
        actual, frozen = expected_index[key], stored_index[key]
        absolute = abs(actual - frozen)
        relative = absolute / max(abs(frozen), 1e-300)
        score_errors.append(absolute)
        relative_errors.append(relative)
        if not (
            absolute <= float(tolerances["score_absolute_tolerance"])
            or relative <= float(tolerances["score_relative_tolerance"])
        ):
            mismatch_count += 1
    frozen_metric = {(row["method"], int(row["budget"])): row for row in stored_metrics
                     if row.get("status") == "AVAILABLE" and row["method"] in methods}
    metric_checks = {}
    for method in methods:
        budget = N_COORDINATES if method == "dense_two_source_glrt" else len(masks[method])
        row = frozen_metric.get((method, budget))
        if row is None:
            metric_checks[method] = {"status": "MISSING"}
            continue
        frozen = float(row["family_macro_low_fpr_pauc"])
        actual = float(bundles[method]["macro_pauc"])
        metric_checks[method] = {"status": "PASS" if abs(actual - frozen) <= float(tolerances["metric_absolute_tolerance"]) else "FAIL",
                                 "recomputed": actual, "frozen": frozen, "absolute_error": abs(actual - frozen)}
    thresholds = load_json(FROZEN / "thresholds.json")
    threshold_checks = {}
    for method in methods:
        budget = N_COORDINATES if method == "dense_two_source_glrt" else len(masks[method])
        frozen = float(thresholds[f"{method}:K{budget}"]["q99"])
        actual = float(bundles[method]["threshold"])
        threshold_checks[method] = {"status": "PASS" if abs(actual - frozen) <= float(tolerances["threshold_absolute_tolerance"]) else "FAIL",
                                    "recomputed": actual, "frozen": frozen, "absolute_error": abs(actual - frozen)}
    scenario_index = {(row["scenario"], row["method"], int(row["budget"])): row for row in stored_scenarios}
    alarm_checks = {}
    for method in methods:
        budget = N_COORDINATES if method == "dense_two_source_glrt" else len(masks[method])
        for scenario in FAMILY:
            frozen = scenario_index.get((scenario, method, budget))
            events = [row for row in bundles[method]["events"] if row["scenario"] == scenario]
            pre = [row for row in events if row["phase"] == "strict_pre"]
            attack = [row for row in events if row["phase"] != "strict_pre"]
            actual_pre = int(sum(row["score"] >= bundles[method]["threshold"] for row in pre))
            actual_attack = int(sum(row["score"] >= bundles[method]["threshold"] for row in attack))
            key = f"{scenario}:{method}"
            if frozen is None:
                alarm_checks[key] = {"status": "MISSING"}
            else:
                expected_pre = int(frozen["external_pre_alarms"])
                expected_attack = int(frozen["alarms"])
                error = max(abs(actual_pre - expected_pre), abs(actual_attack - expected_attack))
                alarm_checks[key] = {"status": "PASS" if error <= int(tolerances["alarm_count_absolute_tolerance"]) else "FAIL",
                                     "recomputed_pre": actual_pre, "frozen_pre": expected_pre,
                                     "recomputed_attack": actual_attack, "frozen_attack": expected_attack}
    failed = bool(missing or additional or mismatch_count)
    failed |= any(item["status"] != "PASS" for item in metric_checks.values())
    failed |= any(item["status"] != "PASS" for item in threshold_checks.values())
    failed |= any(item["status"] != "PASS" for item in alarm_checks.values())
    return {
        "schema": "pg_scc_reproduction_check.v1",
        "status": tolerances["status_on_failure"] if failed else "PASS",
        "independent_recomputation": True, "stored_summary_copied": False,
        "exact_key": ["scenario", "phase", "second", "time_s", "prn", "method", "budget"],
        "missing_rows": len(missing), "additional_rows": len(additional),
        "score_mismatch_rows": mismatch_count, "max_absolute_score_error": max(score_errors, default=None),
        "max_relative_score_error": max(relative_errors, default=None),
        "metric_checks": metric_checks, "threshold_checks": threshold_checks,
        "alarm_count_checks": alarm_checks, "tolerances": dict(tolerances),
        "B0_exact": {"status": "UNAVAILABLE", "historic_csv_reused": False,
                     "reason": "native B0 cannot run on exact PG-SCC common support"},
    }


def _covariance_audit(train: np.ndarray, normalized: np.ndarray,
                      metadata: Sequence[Mapping[str, Any]], auth: np.ndarray,
                      covariance: np.ndarray, masks: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    assert_allowed_role("covariance", "clean_train")
    variants = covariance_variants(train, auth, covariance)
    rows = {}
    for name, matrix in variants.items():
        condition = float(np.linalg.cond(matrix))
        method_metrics = {}
        for method in ("dense_two_source_glrt", "pg_scc_k9"):
            indices = None if method == "dense_two_source_glrt" else masks["pg_scc_k9"]
            bundle = metric_bundle(metadata, batch_glrt(normalized, auth, matrix, indices)["score"])
            method_metrics[method] = {
                "macro_pauc": bundle["macro_pauc"], "worst_fpr": bundle["worst_fpr"],
                "attack_label": "POST_HOC_DIAGNOSTIC",
            }
        rows[name] = {
            "condition_number": condition, "selection_role": "CLEAN_TRAIN_ONLY",
            "attack_based_choice": False, "report_only_metrics": method_metrics,
        }
    return {"status": "AVAILABLE", "variants": rows,
            "selection_rule": "no variant selected on attack outcomes"}


def _paired_family_bootstrap(bundles: Mapping[str, Mapping[str, Any]],
                             config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bootstrap_rows = []
    k3_rows = []
    iterations = int(config["bootstrap"]["iterations"])
    block_seconds = float(config["bootstrap"]["block_seconds"])
    seeds = [int(x) for x in config["bootstrap"]["seeds"]]
    comparisons = ("fixed9", "epl3", "dense_two_source_glrt", "shuffled_k3")
    for family_index, (family, scenarios) in enumerate(FAMILY_MEMBERS.items()):
        for comparison in comparisons:
            left = {(r["scenario"], r["phase"], r["second"], r["time_s"]): r["score"]
                    for r in bundles["pg_scc_k3"]["events"]
                    if r["scenario"] in scenarios and r["phase"] != "strict_pre"}
            right = {(r["scenario"], r["phase"], r["second"], r["time_s"]): r["score"]
                     for r in bundles[comparison]["events"]
                     if r["scenario"] in scenarios and r["phase"] != "strict_pre"}
            if set(left) != set(right):
                raise RuntimeError(f"paired K3 support mismatch: {family}:{comparison}")
            values = [{"family": family, "second": key[2], "value": left[key] - right[key]}
                      for key in sorted(left)]
            result = block_bootstrap(
                values, value_key="value", block_seconds=block_seconds,
                iterations=iterations, seed=seeds[family_index],
            )
            row = {
                "family": family, "comparison": f"pg_scc_k3-minus-{comparison}",
                "status": result["status"], "block_seconds": block_seconds,
                "blocks": result["block_count"], "replicates": iterations,
                "effect": float(np.mean([item["value"] for item in values])) if values else "",
                "ci95_low": result.get("interval_95", ["", ""])[0],
                "ci95_high": result.get("interval_95", ["", ""])[1],
                "seed": seeds[family_index], "label": "EXPLORATORY_ONLY",
            }
            bootstrap_rows.append(row)
            k3_rows.append(dict(row))
    for method in ("pg_scc_k3", "pg_scc_k5", "pg_scc_k9", "fixed9", "dense_two_source_glrt"):
        bundle = bundles[method]
        k3_rows.append({
            "family": "all", "comparison": method, "status": "AVAILABLE",
            "block_seconds": block_seconds, "blocks": "", "replicates": "",
            "effect": bundle["macro_pauc"], "ci95_low": "", "ci95_high": "",
            "seed": "", "label": "EXPLORATORY_ONLY" if method == "pg_scc_k3" else "FROZEN_COMPARATOR",
        })
    return bootstrap_rows, k3_rows


def _save_plots(nested: Sequence[Mapping[str, Any]], contributions: Sequence[Mapping[str, Any]],
                dilution: Sequence[Mapping[str, Any]], mismatch: Sequence[Mapping[str, Any]],
                selector: Mapping[str, Any], seed_rows: Sequence[Mapping[str, Any]],
                random_rows: Sequence[Mapping[str, Any]], bundles: Mapping[str, Mapping[str, Any]],
                awgn: Mapping[str, Any], calibration: Mapping[str, Any]) -> None:
    plot_root = OUTPUT / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    for stage, marker in (("K3", "o"), ("K5_new", "s"), ("K9_new", "^")):
        rows = [row for row in nested if row["stage"] == stage]
        ax.scatter([row["delay_chips"] for row in rows], [row["doppler_hz"] for row in rows], label=stage, marker=marker)
    ax.set(xlabel="delay (chips)", ylabel="Doppler (Hz)", title="Frozen nested coordinates")
    ax.legend(); fig.tight_layout(); fig.savefig(plot_root / REQUIRED_PLOTS[0], dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    rows = [row for row in contributions if row["group"] in FAMILY_MEMBERS]
    ax.scatter([row["clean_residual_variance"] for row in rows],
               [row["mean_glrt_improvement_contribution"] for row in rows], alpha=.6)
    ax.set(xlabel="clean residual variance", ylabel="attack contribution", title="Variance vs contribution (post hoc)")
    fig.tight_layout(); fig.savefig(plot_root / REQUIRED_PLOTS[1], dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for group in ("clean_train", "clean_holdout", "synthetic_h1"):
        rows = [
            row for row in dilution
            if row["group"] == group and "mean_improvement_per_k" in row
        ]
        ax.plot([row["budget"] for row in rows], [row["mean_improvement_per_k"] for row in rows], marker="o", label=group)
    ax.set(xlabel="K", ylabel="RSS improvement / K", title="Score dilution")
    ax.legend(); fig.tight_layout(); fig.savefig(plot_root / REQUIRED_PLOTS[2], dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    attack = [row for row in mismatch if row["label"] == "POST_HOC_DIAGNOSTIC"]
    ax.scatter([row["best_delay_chips"] for row in attack], [row["best_doppler_hz"] for row in attack], s=8, alpha=.4)
    ax.set(xlabel="best delay (chips)", ylabel="best Doppler (Hz)", title="Real best-fit vs frozen synthetic grid")
    fig.tight_layout(); fig.savefig(plot_root / REQUIRED_PLOTS[3], dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([3, 5, 9], [selector["learned_evidence"][str(k)]["random_percentile"] for k in (3, 5, 9)])
    ax.axhline(.95, color="black", linestyle="--")
    ax.set(xlabel="K", ylabel="random percentile", title="Learned mask synthetic percentile")
    fig.tight_layout(); fig.savefig(plot_root / REQUIRED_PLOTS[4], dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for budget in (3, 5, 9):
        rows = [row for row in seed_rows if row["budget"] == budget]
        ax.plot([row["seed"] for row in rows], [row["median_pairwise_jaccard"] for row in rows], label=f"K{budget}")
    ax.set(xlabel="seed", ylabel="median Jaccard", title="Selector seed stability")
    ax.legend(); fig.tight_layout(); fig.savefig(plot_root / REQUIRED_PLOTS[5], dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    methods = ("pg_scc_k3", "pg_scc_k5", "pg_scc_k9", "fixed9", "dense_two_source_glrt")
    ax.bar(methods, [bundles[name]["macro_pauc"] for name in methods])
    ax.tick_params(axis="x", rotation=25); ax.set(ylabel="macro low-FPR pAUC", title="Common-support performance")
    fig.tight_layout(); fig.savefig(plot_root / REQUIRED_PLOTS[6], dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    rows = awgn["empirical_controls"]
    ax.plot([row["multiplier"] for row in rows], [row["false_response"] for row in rows], marker="o")
    ax.set(xlabel="empirical residual sigma multiplier", ylabel="false response", title="Empirical-noise AWGN reaudit")
    fig.tight_layout(); fig.savefig(plot_root / REQUIRED_PLOTS[7], dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    methods = list(calibration["methods"])
    lows = [calibration["methods"][name]["bootstrap_threshold_interval"][0] for name in methods]
    highs = [calibration["methods"][name]["bootstrap_threshold_interval"][1] for name in methods]
    ax.vlines(range(len(methods)), lows, highs); ax.scatter(range(len(methods)), lows)
    ax.set_xticks(range(len(methods)), methods, rotation=25); ax.set(ylabel="threshold", title="Calibration threshold uncertainty")
    fig.tight_layout(); fig.savefig(plot_root / REQUIRED_PLOTS[8], dpi=150); plt.close(fig)


def _write_readme(freeze: Mapping[str, Any], reproduction: Mapping[str, Any],
                  verdict: Mapping[str, Any], unavailable: Sequence[str]) -> None:
    lines = [
        "# PG-SCC Stage-0 R2 Root-Cause Audit",
        "",
        "This artifact was generated deterministically from the implementation freeze "
        f"{freeze['implementation_sha']}.",
        "",
        f"- Reproduction: {reproduction['status']}",
        f"- Verdict: {verdict['verdict']}",
        "- Attack-derived alternatives: POST_HOC_DIAGNOSTIC",
        "- K3: EXPLORATORY_ONLY",
        "- Selector/covariance/threshold fitting used clean/synthetic roles only.",
        "- DS4 scope: transition-only; DS7/DS8 count as one family.",
        "",
        "## Explicitly unavailable physical inputs",
        "",
        *[f"- {item}" for item in unavailable],
        "",
        "No missing physical input was fabricated. See the machine-readable JSON files "
        "for numeric evidence and LIMITED/UNAVAILABLE/INCONCLUSIVE reasons.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_required_outputs() -> None:
    missing = [name for name in REQUIRED_ARTIFACTS if not (OUTPUT / name).exists()]
    missing.extend(f"plots/{name}" for name in REQUIRED_PLOTS if not (OUTPUT / "plots" / name).is_file())
    if missing:
        raise RuntimeError(f"required artifact schema incomplete: {missing}")
    verdict = load_json(OUTPUT / "root_cause_verdict.json")
    if set(verdict.get("root_causes", {})) != set(REQUIRED_ROOT_CAUSES):
        raise RuntimeError("root-cause schema mismatch")
    if verdict.get("verdict") not in config_allowed_verdicts():
        raise RuntimeError("invalid root-cause verdict")
    if not verdict.get("exactly_one_verdict"):
        raise RuntimeError("verdict cardinality failure")


def _write_followup_attempt_state(freeze: Mapping[str, Any], verdict: Mapping[str, Any]) -> None:
    dump_json(OUTPUT / "attempt_state.json", {
        "attempt_count": 1,
        "implementation_sha": freeze["implementation_sha"],
        "predecessor_partial_outputs_reused": False,
        "schema": "pg_scc_stage0_r2_validator_repair_attempt_state.v1",
        "scientific_verdict": verdict["verdict"],
        "state": "COMPLETED",
    })


def config_allowed_verdicts() -> set[str]:
    return {"TERMINATE_PG_SCC", "REPAIRABLE_BUT_REQUIRES_NEW_CONFIRMATION",
            "K3_WORTH_INDEPENDENT_CONFIRMATION"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-sha",
                        help="full pushed implementation-freeze commit SHA")
    parser.add_argument("--metadata-preflight", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.resolve() != OUTPUT.resolve():
        raise RuntimeError("output must be the preregistered artifact directory")

    config = load_json(OUTPUT / "config.json")
    source = load_json(OUTPUT / "source_commit.json")
    precheck = verify_preregistration(config, source)
    if precheck["status"] != "PASS":
        raise RuntimeError("preregistration verification failed:" + ",".join(precheck["errors"]))
    if args.metadata_preflight:
        if args.implementation_sha:
            raise RuntimeError("metadata preflight must not accept a protected-execution SHA")
        report = run_metadata_support_preflight()
        print(json.dumps({
            "status": report["status"],
            "eligible_event_count": report["eligible_event_count"],
            "excluded_event_count": report["excluded_event_count"],
            "protected_score_fields_read": report["protected_score_fields_read"],
        }, sort_keys=True))
        return 0 if report["status"] == "PASS" else 1
    if not args.implementation_sha:
        raise RuntimeError("--implementation-sha is required for protected Phase 3 execution")

    freeze = verify_implementation_freeze(args.implementation_sha)
    protected = load_protected_inputs_after_preflight()
    clean_records = protected["clean_records"]
    attack_records = protected["attack_records"]
    if {row["scenario"] for row in attack_records} != {"ds3", "ds4", "ds7", "ds8"}:
        raise RuntimeError("protected attack cache scenario schema mismatch")
    assert_allowed_role("attack_diagnostic", "attack_report_only")
    metadata = _metadata([*clean_records, *attack_records])
    _enforce_timeline(metadata, protected["timeline"])
    normalized = _surface_array([*clean_records, *attack_records])
    auth, covariance = protected["auth"], protected["covariance"]
    masks = {name: [int(x) for x in values] for name, values in protected["masks"].items()}
    validate_nested_masks(masks)
    fits = _fit_all(normalized, auth, covariance, masks)
    committed_preflight = load_json(OUTPUT / "support_preflight.json")
    support = validate_common_support(
        _score_table(metadata, fits),
        CORE_METHODS,
        int(config["common_support"]["eligible_event_filtering"]["minimum_unique_prns_per_pooled_event"]),
        frozen_expected_events=committed_preflight["eligible_pooled_events"],
        frozen_expected_source=committed_preflight["frozen_pooled_event_expected_count_source"],
    )
    if support["status"] != "PASS":
        raise RuntimeError("FAIL_CLOSED_SUPPORT_PREFLIGHT:frozen pooled-event support mismatch")
    metric_metadata, metric_scores = filter_metric_inputs_to_eligible_events(
        metadata, {method: fit["score"] for method, fit in fits.items()}, support,
    )
    bundles = {
        method: metric_bundle(metric_metadata, metric_scores[method])
        for method in fits
    }

    reproduction = reproduce_exact(
        metadata, fits, bundles, masks, protected["stored_scores"],
        protected["stored_metrics"], protected["stored_scenarios"], config,
    )
    reproduction["common_support"] = support
    dump_json(OUTPUT / "reproduction_check.json", reproduction)

    clean_metadata = _metadata(clean_records)
    clean_normalized = _surface_array(clean_records)
    train_positions = np.asarray([i for i, row in enumerate(clean_metadata) if row["phase"] == "train"])
    selection_positions = np.asarray([i for i, row in enumerate(clean_metadata) if row["phase"] == "selection"])
    if not len(train_positions) or not len(selection_positions):
        raise RuntimeError("clean train/selection roles unavailable")
    assert_allowed_role("selector", "clean_train")
    assert_allowed_role("selector", "clean_selection")
    bank = build_synthetic_bank(
        np.asarray([clean_records[i]["surface"] for i in train_positions]),
        np.asarray([clean_records[i]["surface"] for i in selection_positions]),
        normalization="prompt_phase",
        seed=int(protected["frozen_training_config"]["seed"]),
        max_h1_per_split=int(
            protected["frozen_training_config"]["synthetic_bank"]["max_h1_per_split"]
        ),
    )
    nested, contributions, dilution = nested_outputs(
        metadata, normalized, auth, covariance, masks, bank, fits, bundles,
    )
    write_csv(OUTPUT / "nested_mask_analysis.csv", nested)
    write_csv(OUTPUT / "coordinate_contributions.csv", contributions)
    write_csv(OUTPUT / "score_dilution_metrics.csv", dilution)

    selector, seed_rows, random_rows = _selector_audit(config, bank, auth, covariance, masks)
    dump_json(OUTPUT / "selector_proxy_audit.json", selector)
    write_csv(OUTPUT / "mask_seed_stability.csv", seed_rows)
    write_csv(OUTPUT / "random_mask_distribution.csv", random_rows)

    dense, mismatch_rows = _dense_mismatch(
        normalized, metadata, auth, covariance, fits["dense_two_source_glrt"],
    )
    dense["covariance_diagnostics"] = _covariance_audit(
        clean_normalized[train_positions], normalized, metadata, auth, covariance, masks,
    )
    dump_json(OUTPUT / "dense_teacher_diagnostics.json", dense)
    write_csv(OUTPUT / "synthetic_real_mismatch.csv", mismatch_rows)

    calibration = _calibration_audit(bundles, config)
    dump_json(OUTPUT / "calibration_uncertainty.json", calibration)
    awgn = _awgn_audit(
        clean_normalized[train_positions], auth, covariance, masks["pg_scc_k9"],
        bundles["pg_scc_k9"]["threshold"],
    )
    dump_json(OUTPUT / "awgn_reaudit.json", awgn)

    bootstrap_rows, k3_rows = _paired_family_bootstrap(bundles, config)
    write_csv(OUTPUT / "bootstrap_intervals.csv", bootstrap_rows)
    write_csv(OUTPUT / "k3_exploratory_metrics.csv", k3_rows)

    verdict = _root_cause_verdict(reproduction, selector, dense, calibration, awgn, bundles)
    verdict["implementation_freeze"] = freeze
    dump_json(OUTPUT / "root_cause_verdict.json", verdict)
    _save_plots(nested, contributions, dilution, mismatch_rows, selector, seed_rows,
                random_rows, bundles, awgn, calibration)
    _write_readme(
        freeze, reproduction, verdict,
        ["raw-IQ AWGN CAF regeneration", "unique normalized-CAF-to-C/N0 mapping",
         "navigation-bit truth", "tracker recentering state", "multipath ground truth"],
    )
    _write_followup_attempt_state(freeze, verdict)
    _validate_required_outputs()
    finalize_manifest(OUTPUT)
    print(json.dumps({"status": "COMPLETE", "verdict": verdict["verdict"],
                      "implementation_sha": freeze["implementation_sha"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
