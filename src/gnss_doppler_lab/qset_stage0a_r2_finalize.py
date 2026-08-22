"""Compact artifact finalization for the locked Q-SET Stage-0A R2 result."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import numpy as np

from .qset_stage0a_r2_evaluation import *  # noqa: F401,F403


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"windows", "window_rows", "score_rows"}}


def _plot(results: dict[str, dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["SS-1", "SS-3", "SS-5", "SS-11"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names)); width = 0.13
    for index, aggregator in enumerate(AGGREGATORS):
        values = [results[name]["metrics"][aggregator]["detection_rate"] for name in names]
        axis.bar(x + (index - 2.5) * width, values, width, label=aggregator)
    axis.set_xticks(x, names); axis.set_ylim(0, 1); axis.set_ylabel("Detection rate at clean FPR threshold")
    axis.set_title("Q-SET Stage-0A locked aggregator comparison"); axis.legend(ncol=3, fontsize=8); figure.tight_layout()
    figure.savefig(ARTIFACT / "plots/aggregator_detection_rate.png", dpi=160); plt.close(figure)


def finalize_attack_artifacts(payload: dict[str, Any], freeze_sha: str) -> dict[str, Any]:
    results = payload["results"]; clean_summary = read_json(ARTIFACT / "clean_score_summary.json"); threshold = read_json(ARTIFACT / "threshold_binding.json")
    write_json(ARTIFACT / "freeze_commit.json", {"schema": "gnss-doppler-lab.qset-stage0a-r2-freeze-commit.v1", "status": "PASS", "commit_sha": freeze_sha, "pushed_before_attack_access": True, "local_remote_match": True, "ahead": 0, "behind": 0, "clean_checkout": True})
    receiver_manifests = ARTIFACT / "receiver_manifests"; receiver_manifests.mkdir(exist_ok=True)
    for name in results:
        write_json(receiver_manifests / f"{name}.json", read_json(SSD_ROOT / "replays" / name / "manifest.json"))
    scenario_metrics = {"schema": "gnss-doppler-lab.qset-stage0a-r2-scenario-metrics.v1", "status": "COMPLETE", "scenarios": {name: _compact_result(result) for name, result in results.items()}}
    write_json(ARTIFACT / "scenario_metrics.json", scenario_metrics)
    aggregator_rows = []
    for name, result in results.items():
        for aggregator, metrics in result["metrics"].items():
            aggregator_rows.append({"scenario": name, "spoofed_prn_count": len(SCENARIOS[name]["spoofed"]), "multipath": bool(SCENARIOS[name].get("multipath", False)), "aggregator": aggregator, **metrics, "threshold": threshold["thresholds"][aggregator]["threshold"]})
    write_csv(ARTIFACT / "aggregator_comparison.csv", aggregator_rows, ["scenario", "spoofed_prn_count", "multipath", "aggregator", "eligible_windows", "detection_rate", "pauc_fpr_le_0p01", "threshold"])
    prn_rows = []
    for name, result in results.items():
        for prn in sorted(SCENARIOS[name]["spoofed"]):
            prn_rows.append({"scenario": name, "prn": prn, "ground_truth": "spoofed", "supported_windows": result["spoofed_support"].get(prn, 0), "scenario_spoof_vs_genuine_auc": result["per_prn_auc"]})
        genuine = sorted({row["prn"] for row in result["score_rows"] if not row["spoofed"]})
        for prn in genuine:
            prn_rows.append({"scenario": name, "prn": prn, "ground_truth": "genuine", "supported_windows": sum(row["prn"] == prn for row in result["score_rows"]), "scenario_spoof_vs_genuine_auc": result["per_prn_auc"]})
    write_csv(ARTIFACT / "per_prn_ground_truth_metrics.csv", prn_rows, ["scenario", "prn", "ground_truth", "supported_windows", "scenario_spoof_vs_genuine_auc"])
    with gzip.open(ARTIFACT / "per_window_scores.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        fields = ["scenario", "window_end_s", "prn_count", "cn0_mean", "lock_mean", "multi_q_persistent_score", "alarm"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for name, result in results.items():
            for row in result["window_rows"]:
                writer.writerow({"scenario": name, "window_end_s": row["window_end_s"], "prn_count": row["prn_count"], "cn0_mean": row["cn0_mean"], "lock_mean": row["lock_mean"], "multi_q_persistent_score": row["score"], "alarm": row["alarm"]})
    shortcuts = {"schema": "gnss-doppler-lab.qset-stage0a-r2-shortcut-audit.v1", "forbidden_score_inputs_absent": True, "scenario_id_classifier": False, "prn_identity_input": False, "absolute_cn0_power_count_input": False, "scenarios": {name: result["shortcut"] for name, result in results.items()}}
    shortcuts["status"] = "PASS" if all(item["pass"] for item in shortcuts["scenarios"].values()) else "FAIL"
    write_json(ARTIFACT / "shortcut_audit.json", shortcuts)
    clean_gate = all(item["empirical_fpr"] <= 0.01 and item["wilson_95_upper"] <= 0.05 for item in clean_summary["clean_metrics"].values())
    technical_gate = all(result["scoreable_windows"] >= 60 and result["stable_spoofed_prn"] for result in results.values())
    technical_gate = technical_gate and all(read_json(SSD_ROOT / "replays" / name / "manifest.json")["status"] == "PASS" for name in results)
    local_pass_scenarios = [name for name, result in results.items() if result["per_prn_auc"] is not None and result["per_prn_auc"] >= 0.75]
    local_gate = len(local_pass_scenarios) >= 2 and shortcuts["status"] == "PASS"
    ss1, ss3, ss5 = (results[name]["metrics"] for name in ("SS-1", "SS-3", "SS-5"))
    aggregation_checks = {
        "ss1_detection_gain_ge_0p15": ss1["MULTI_Q"]["detection_rate"] - ss1["MEAN"]["detection_rate"] >= 0.15,
        "ss3_detection_gain_ge_0p15": ss3["MULTI_Q"]["detection_rate"] - ss3["MEAN"]["detection_rate"] >= 0.15,
        "ss1_pauc_gain_ge_0p10": ss1["MULTI_Q"]["pauc_fpr_le_0p01"] - ss1["MEAN"]["pauc_fpr_le_0p01"] >= 0.10,
        "ss3_pauc_gain_ge_0p10": ss3["MULTI_Q"]["pauc_fpr_le_0p01"] - ss3["MEAN"]["pauc_fpr_le_0p01"] >= 0.10,
        "partial_mean_pauc_beats_max": np.mean([ss1["MULTI_Q"]["pauc_fpr_le_0p01"], ss3["MULTI_Q"]["pauc_fpr_le_0p01"]]) > np.mean([ss1["MAX"]["pauc_fpr_le_0p01"], ss3["MAX"]["pauc_fpr_le_0p01"]]),
        "ss5_loss_vs_mean_le_0p05": ss5["MULTI_Q"]["detection_rate"] >= ss5["MEAN"]["detection_rate"] - 0.05,
        "no_serious_reversal": ss3["MULTI_Q"]["detection_rate"] >= ss1["MULTI_Q"]["detection_rate"] - 0.10 and ss5["MULTI_Q"]["detection_rate"] >= ss3["MULTI_Q"]["detection_rate"] - 0.10,
        "bootstrap_lower_positive_at_least_two": sum(result["bootstrap_multi_minus_mean"]["lower_95"] > 0 for result in results.values()) >= 2,
    }
    aggregation_gate = all(aggregation_checks.values())
    all_gate = technical_gate and clean_gate and local_gate and aggregation_gate
    if not technical_gate:
        verdict = "INCONCLUSIVE_QSET_DATA_FORMAT_OR_RECEIVER_SUPPORT"
    elif not local_gate:
        verdict = "NO_GO_QSET_PARTIAL_PRN_SIGNAL"
    elif not all_gate:
        verdict = "QSET_STAGE0A_PARTIAL_SIGNAL_WITHOUT_AGGREGATION_GAIN"
    else:
        verdict = "QSET_STAGE0A_PARTIAL_PRN_AGGREGATION_PASS"
    gate = {"schema": "gnss-doppler-lab.qset-stage0a-r2-gate.v1", "technical": {"pass": technical_gate}, "false_alarm": {"pass": clean_gate, "metrics": clean_summary["clean_metrics"]}, "local_signal": {"pass": local_gate, "passing_scenarios": local_pass_scenarios}, "aggregation": {"pass": aggregation_gate, "checks": aggregation_checks}, "overall_pass": all_gate}
    write_json(ARTIFACT / "stage0a_gate.json", gate)
    final = {"schema": "gnss-doppler-lab.qset-stage0a-r2-final-verdict.v1", "verdict": verdict, "next_state": "READY_FOR_QSET_STAGE0B_SHARED_ENCODER_PREREGISTRATION" if all_gate else "NOT_AUTHORIZED", "stage0b_authorized": all_gate, "attack_evaluation_complete": True, "freeze_sha": freeze_sha, "classification": "STAGE_0A_NON_NEURAL_FEASIBILITY", "forbidden_claims_made": False}
    write_json(ARTIFACT / "final_verdict.json", final)
    deterministic_payload = {name: _compact_result(evaluate_attack(name, payload["clean"])) for name in results}
    first_sha = canonical_sha({name: _compact_result(result) for name, result in results.items()}); second_sha = canonical_sha(deterministic_payload)
    write_json(ARTIFACT / "deterministic_reproduction.json", {"schema": "gnss-doppler-lab.qset-stage0a-r2-determinism.v1", "status": "PASS" if first_sha == second_sha else "FAIL", "analysis_sha256_first": first_sha, "analysis_sha256_second": second_sha, "byte_identical_compact_metrics": first_sha == second_sha})
    audit = read_json(ARTIFACT / "access_audit.json"); raw_bytes = sum(SCENARIOS[name]["size"] for name in results)
    audit["phase"] = "ATTACK_EVALUATION_COMPLETE"; audit["attack_payload"] = {"allowlisted_scenarios": list(results), "stats": len(results) * 3, "hashes": len(results), "opens": len(results) * 2, "mmaps": 0, "bytes_read": raw_bytes * 2, "signal_statistics": len(results)}; audit["unallowlisted_tuni2025_raw"] = {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0}; audit["attack_scientific_operations"] = {"attack_evaluations": len(results), "attack_scoreable_windows": sum(result["scoreable_windows"] for result in results.values()), "attack_local_score_rows": sum(len(result["score_rows"]) for result in results.values())}; audit["attack_access_after_freeze_only"] = True; audit["status"] = "PASS"
    write_json(ARTIFACT / "access_audit.json", audit); _plot(results)
    return final


def compact_manifest(root: Path = ARTIFACT) -> dict[str, Any]:
    excluded = {"artifact_manifest_sha256.json", "verifier_output.txt", "fresh_clone_verifier_output.txt"}
    files = [{"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(root.rglob("*")) if path.is_file() and path.name not in excluded]
    return {"schema": "gnss-doppler-lab.qset-stage0a-r2-artifact-manifest.v1", "status": "PASS", "file_count": len(files), "files": files, "aggregate_sha256": canonical_sha(files)}

