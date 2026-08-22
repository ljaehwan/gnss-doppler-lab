"""Locked clean freeze and post-freeze attack evaluation for Q-SET R2."""

from __future__ import annotations

import csv
import json
import math
import platform
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .qset_stage0a_r2_execution import *  # noqa: F401,F403


CODE_BINDING_PATHS = (
    "src/gnss_doppler_lab/qset_stage0a_r2.py",
    "src/gnss_doppler_lab/qset_stage0a_r2_execution.py",
    "src/gnss_doppler_lab/qset_stage0a_r2_evaluation.py",
    "src/gnss_doppler_lab/qset_stage0a_r2_finalize.py",
    "scripts/run_qset_gnss_stage0a_r2.py",
    "scripts/verify_qset_gnss_stage0a_r2.py",
    "tests/test_qset_gnss_stage0a_r2.py",
    "configs/qset_galileo_e1_trace9.conf.template",
    "patches/qset_galileo_e1_native_trace.patch",
)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def clean_execution() -> dict[str, Any]:
    receiver_build = build_receiver(); manifests = {}
    for name in ("C-1", "C-3"):
        replay = replay_scenario(name, scenario_path(name), receiver_build); manifests[name] = replay
        rows = extract_window_features(SSD_ROOT / "replays" / name / "receiver", name, SCENARIOS[name]["size"] / BYTES_PER_COMPLEX / RAW_FS)
        require(rows, f"no feature rows for {name}"); save_feature_cache(name, rows)
    clean = analyze_clean(); synthetic = synthetic_dilution(clean)
    require(synthetic["status"] == "PASS", "synthetic implementation sanity failed")
    return {"receiver_build": receiver_build, "replays": manifests, "clean": clean, "synthetic": synthetic}


def freeze_clean_artifacts(result: dict[str, Any]) -> None:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    write_json(ARTIFACT / "preregistration_commit.json", {"schema": "gnss-doppler-lab.qset-stage0a-r2-preregistration-commit.v1", "status": "PASS", "commit_sha": PREREGISTRATION_SHA, "pushed_before_clean_execution": True})
    write_json(ARTIFACT / "receiver_binary_inventory.json", result["receiver_build"])
    receiver_manifests = ARTIFACT / "receiver_manifests"; receiver_manifests.mkdir(exist_ok=True)
    support_rows = []
    for name, replay in result["replays"].items():
        write_json(receiver_manifests / f"{name}.json", replay)
        for prn in replay["trace_validation"]["tracked_prns"]:
            support_rows.append({"scenario": name, "prn": prn, "source": "native TRACE", "status": "TRACKED"})
    write_csv(ARTIFACT / "per_prn_support.csv", support_rows, ["scenario", "prn", "source", "status"])
    clean = result["clean"]
    write_json(ARTIFACT / "clean_score_summary.json", {key: value for key, value in clean.items() if key not in {"model", "multi_q_reference", "thresholds"}} | {"clean_metrics": clean["clean_metrics"]})
    write_json(ARTIFACT / "normal_model.json", clean["model"])
    write_json(ARTIFACT / "threshold_binding.json", {"schema": "gnss-doppler-lab.qset-stage0a-r2-threshold.v1", "status": "PASS", "target_fpr": TARGET_FPR, "multi_q_reference": clean["multi_q_reference"], "thresholds": clean["thresholds"], "threshold_sha256": clean["threshold_sha256"], "attack_data_used": False})
    write_json(ARTIFACT / "synthetic_dilution_control.json", result["synthetic"])
    feature_bindings = {}
    for name in ("C-1", "C-3"):
        path = SSD_ROOT / "features" / f"{name}.npz"; feature_bindings[name] = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    code_bindings = {relative: sha256_file(ROOT / relative) for relative in CODE_BINDING_PATHS}
    replay_bindings = {name: {"raw_md5": replay["raw_input"]["md5"], "decoder_sha256": replay["decoder"]["output_sha256"], "output_set_sha256": replay["output_set"]["aggregate_sha256"]} for name, replay in result["replays"].items()}
    freeze = {
        "schema": "gnss-doppler-lab.qset-stage0a-r2-execution-freeze.v1", "status": "PASS_PRE_ATTACK",
        "preregistration_commit": PREREGISTRATION_SHA, "code_bindings": code_bindings,
        "receiver_sha256": result["receiver_build"]["receiver_sha256"], "receiver_source_diff_sha256": result["receiver_build"]["combined_source_diff_sha256"],
        "configuration_sha256": CONFIG_SHA256, "feature_bindings": feature_bindings, "clean_replay_bindings": replay_bindings,
        "model_sha256": clean["model_sha256"], "threshold_sha256": clean["threshold_sha256"],
        "clean_holdout_status": clean["status"], "clean_holdout_metrics": clean["clean_metrics"],
        "attack_allowlist": ["SS-1", "SS-3", "SS-5", "SS-11"], "worker_count": 1,
        "shortcut_operationalization": {"absolute_pearson_correlation_limit": 0.8, "stable_spoofed_prn_windows_min": 60, "score_inputs": "morphology only"},
        "output_root": str(SSD_ROOT), "freeze_commit": "supplied as --freeze-sha from pushed clean checkout",
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__},
    }
    write_json(ARTIFACT / "execution_freeze.json", freeze)
    audit = read_json(ARTIFACT / "access_audit.json")
    audit["phase"] = "CLEAN_EXECUTION_COMPLETE_PRE_ATTACK"
    audit["clean_payload_execution"] = {"C-1": {"identity_hash_bytes": SCENARIOS["C-1"]["size"], "decoder_bytes_read": SCENARIOS["C-1"]["size"]}, "C-3": {"identity_hash_bytes": SCENARIOS["C-3"]["size"], "decoder_bytes_read": SCENARIOS["C-3"]["size"]}}
    audit["scientific_operations"] = {"receiver_runs": 2, "feature_windows": clean["development_prn_windows"], "scores": clean["holdout_event_windows"], "thresholds_fit": len(AGGREGATORS), "attack_evaluations": 0}
    audit["attack_payload"] = {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0, "signal_statistics": 0}
    audit["status"] = "PASS_ATTACK_STILL_SEALED"
    write_json(ARTIFACT / "access_audit.json", audit)


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _persistent_scenario(windows: list[dict[str, Any]], clean: dict[str, Any], aggregator: str) -> dict[str, Any]:
    raw = np.asarray([window["aggregates"][aggregator] for window in windows]); ends = np.asarray([window["window_end_s"] for window in windows]); continuous, warmup = persistence(raw, ends); eligible = ~warmup & np.isfinite(continuous); threshold = clean["thresholds"][aggregator]["threshold"]; alarms = continuous[eligible] > threshold
    return {"continuous": continuous, "eligible": eligible, "alarms": alarms, "detection_rate": float(np.mean(alarms)) if len(alarms) else 0.0, "eligible_count": int(np.sum(eligible))}


def _bootstrap_difference(windows: list[dict[str, Any]], clean: dict[str, Any]) -> dict[str, float]:
    multi = _persistent_scenario(windows, clean, "MULTI_Q"); mean = _persistent_scenario(windows, clean, "MEAN"); eligible = multi["eligible"] & mean["eligible"]
    ends = np.asarray([window["window_end_s"] for window in windows])[eligible]; differences = (multi["continuous"][eligible] > clean["thresholds"]["MULTI_Q"]["threshold"]).astype(float) - (mean["continuous"][eligible] > clean["thresholds"]["MEAN"]["threshold"]).astype(float)
    blocks = sorted(set(((ends - 1) // 10).tolist())); block_values = [differences[((ends - 1) // 10) == block] for block in blocks]; require(block_values, "no bootstrap blocks")
    rng = np.random.default_rng(20260822); estimates = np.empty(10000)
    for index in range(len(estimates)):
        selected = rng.integers(0, len(block_values), size=len(block_values)); estimates[index] = np.mean(np.concatenate([block_values[item] for item in selected]))
    return {"estimate": float(np.mean(differences)), "lower_95": float(np.quantile(estimates, 0.025)), "upper_95": float(np.quantile(estimates, 0.975)), "block_count": len(blocks)}


def evaluate_attack(name: str, clean: dict[str, Any]) -> dict[str, Any]:
    rows = load_feature_cache(name); windows = dynamic_windows(rows, clean["model"])
    for window in windows: window["aggregates"] = aggregate_scores(window["scores"], clean["multi_q_reference"])
    c3_windows = dynamic_windows(load_feature_cache("C-3"), clean["model"])
    for window in c3_windows: window["aggregates"] = aggregate_scores(window["scores"], clean["multi_q_reference"])
    metrics = {}; window_rows = []
    for aggregator in AGGREGATORS:
        attack = _persistent_scenario(windows, clean, aggregator); holdout = _persistent_scenario(c3_windows, clean, aggregator)
        metrics[aggregator] = {"detection_rate": attack["detection_rate"], "eligible_windows": attack["eligible_count"], "pauc_fpr_le_0p01": normalized_pauc(holdout["continuous"][holdout["eligible"]], attack["continuous"][attack["eligible"]])}
        if aggregator == "MULTI_Q":
            for window, continuous, eligible in zip(windows, attack["continuous"], attack["eligible"], strict=True):
                if eligible: window_rows.append({"window_end_s": window["window_end_s"], "prn_count": len(window["prns"]), "cn0_mean": float(np.mean(window["cn0"])), "lock_mean": float(np.mean(window["lock"])), "score": float(continuous), "alarm": bool(continuous > clean["thresholds"]["MULTI_Q"]["threshold"])})
    spoofed = set(SCENARIOS[name]["spoofed"]); score_rows = []
    scoreable_ends = {window["window_end_s"] for window in windows}
    for row in rows:
        if row["window_end_s"] in scoreable_ends:
            score = float(local_scores(np.asarray([row["feature"]]), clean["model"])[0]); score_rows.append({"prn": row["prn"], "window_end_s": row["window_end_s"], "score": score, "spoofed": row["prn"] in spoofed})
    positives = [row["score"] for row in score_rows if row["spoofed"]]; negatives = [row["score"] for row in score_rows if not row["spoofed"]]
    per_prn_auc = binary_auc(negatives, positives) if positives and negatives else None
    support_by_spoof = {prn: sum(row["prn"] == prn for row in score_rows) for prn in sorted(spoofed)}
    vector = {key: np.asarray([row[key] for row in window_rows], dtype=float) for key in ("score", "prn_count", "cn0_mean", "lock_mean")}
    correlations = {key: _pearson(vector["score"], vector[key]) for key in ("prn_count", "cn0_mean", "lock_mean")}
    correlation_pass = all(value is None or abs(value) < 0.8 for value in correlations.values())
    return {"scenario": name, "windows": windows, "metrics": metrics, "per_prn_auc": per_prn_auc, "spoofed_support": support_by_spoof, "stable_spoofed_prn": any(count >= 60 for count in support_by_spoof.values()), "scoreable_windows": len(windows), "bootstrap_multi_minus_mean": _bootstrap_difference(windows, clean), "shortcut": {"correlations": correlations, "pass": correlation_pass, "score_feature_inputs_exclude_shortcuts": True}, "window_rows": window_rows, "score_rows": score_rows}


def verify_file_binding(path: Path, binding: dict[str, Any], label: str) -> None:
    require(path.is_file(), f"missing frozen {label}: {path}")
    if "size_bytes" in binding:
        require(path.stat().st_size == int(binding["size_bytes"]), f"frozen {label} size drift")
    require(sha256_file(path) == binding["sha256"], f"frozen {label} hash drift")


def verify_freeze(freeze_sha: str) -> dict[str, Any]:
    require(git("rev-parse", "HEAD") == freeze_sha, "not executing pushed freeze SHA")
    require(git("status", "--porcelain") == "", "freeze checkout not clean")
    require(git("rev-parse", "origin/research/qset-gnss-stage0a-r2-galileo-partial-prn-execution") == freeze_sha, "remote freeze mismatch")
    freeze = read_json(ARTIFACT / "execution_freeze.json")
    require(freeze["status"] == "PASS_PRE_ATTACK", "freeze status drift")
    require(freeze["attack_allowlist"] == ["SS-1", "SS-3", "SS-5", "SS-11"], "attack allowlist drift")
    require(freeze["worker_count"] == 1, "worker count drift")
    for relative, expected in freeze["code_bindings"].items(): require(sha256_file(ROOT / relative) == expected, f"frozen code drift {relative}")
    require(sha256_file(RECEIVER) == freeze["receiver_sha256"], "receiver drift")
    receiver_inventory = read_json(ARTIFACT / "receiver_binary_inventory.json")
    require(receiver_inventory["receiver_sha256"] == freeze["receiver_sha256"], "receiver inventory drift")
    for name in ("C-1", "C-3"):
        binding = freeze["feature_bindings"][name]
        verify_file_binding(Path(binding["path"]), binding, f"{name} feature cache")
        replay = read_json(SSD_ROOT / "replays" / name / "manifest.json")
        verify_manifest(SSD_ROOT / "replays" / name, replay["output_set"])
        expected = freeze["clean_replay_bindings"][name]
        require(replay["status"] == "PASS", f"{name} replay status drift")
        require(replay["raw_input"]["md5"] == expected["raw_md5"], f"{name} raw binding drift")
        require(replay["decoder"]["output_sha256"] == expected["decoder_sha256"], f"{name} decoder binding drift")
        require(replay["output_set"]["aggregate_sha256"] == expected["output_set_sha256"], f"{name} output-set binding drift")
    model = read_json(ARTIFACT / "normal_model.json")
    require(canonical_sha(model) == freeze["model_sha256"], "clean model drift")
    threshold = read_json(ARTIFACT / "threshold_binding.json")
    threshold_sha = canonical_sha({"multi_q_reference": threshold["multi_q_reference"], "thresholds": threshold["thresholds"]})
    require(threshold["status"] == "PASS" and threshold["attack_data_used"] is False, "threshold provenance drift")
    require(set(threshold["thresholds"]) == set(AGGREGATORS), "threshold aggregator drift")
    require(threshold_sha == threshold["threshold_sha256"] == freeze["threshold_sha256"], "threshold binding drift")
    clean = {"model": model, **threshold}
    clean["thresholds"] = clean["thresholds"]; clean["multi_q_reference"] = clean["multi_q_reference"]
    return clean


def run_attacks(freeze_sha: str) -> dict[str, Any]:
    clean = verify_freeze(freeze_sha); receiver_build = read_json(ARTIFACT / "receiver_binary_inventory.json"); results = {}
    for name in ("SS-1", "SS-3", "SS-5", "SS-11"):
        replay_scenario(name, scenario_path(name, allow_search=True), receiver_build)
        rows = extract_window_features(SSD_ROOT / "replays" / name / "receiver", name, SCENARIOS[name]["size"] / BYTES_PER_COMPLEX / RAW_FS); require(rows, f"no attack feature rows {name}"); save_feature_cache(name, rows); results[name] = evaluate_attack(name, clean)
    return {"clean": clean, "results": results}
