"""Strict outcome/control helpers for the protected PG-SCC R2 producer."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

from gnss_doppler_lab.pg_scc_physics import normalize_complex
from gnss_doppler_lab.pg_scc_r2_support import (
    common_support_by_event, event_key, family_is_eligible, method_support,
    require_identical_paired_support, support_stratum, universe_support,
)


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "artifacts/pg_scc_stage0_static_k9"
R1_CONFIG = ROOT / "artifacts/pg_scc_stage0_r1_root_cause_audit/config.json"
COMPARISONS = {
    "K9": ("pg_scc_k9", "fixed9", "shuffled_k9"),
    "K5": ("pg_scc_k5", "uniform_k5", "shuffled_k5"),
    "K3": ("pg_scc_k3", "epl3", "shuffled_k3"),
    "DENSE": ("dense_two_source_glrt",),
}


def _fingerprint(values: Mapping[tuple[str, str, str, int], set[int]]) -> str:
    canonical = [[*event, sorted(values[event])] for event in sorted(values)]
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()


def _pooled(method_rows: Sequence[Mapping[str, Any]],
            common: Mapping[tuple[str, str, str, int], set[int]]) -> dict[str, dict[tuple[str, str, str, int], float]]:
    nodes: dict[tuple[str, tuple[str, str, str, int], int], list[float]] = defaultdict(list)
    for row in method_rows:
        nodes[(str(row["method"]), event_key(row), int(row["prn"]))].append(float(row["score"]))
    methods = sorted({str(row["method"]) for row in method_rows})
    output = {method: {} for method in methods}
    for method in methods:
        for event, prns in common.items():
            values = [float(np.median(nodes[(method, event, prn)])) for prn in sorted(prns)
                      if nodes.get((method, event, prn))]
            if len(values) != len(prns):
                raise RuntimeError("method score support differs from common support")
            if values:
                output[method][event] = float(np.median(values))
    return output


def _exact_ci(alarms: int, total: int, confidence: float = .95) -> list[float] | None:
    if total <= 0:
        return None
    alpha = 1 - confidence
    return [
        0.0 if alarms == 0 else float(beta.ppf(alpha / 2, alarms, total - alarms + 1)),
        1.0 if alarms == total else float(beta.ppf(1 - alpha / 2, alarms + 1, total - alarms)),
    ]


def _block_ci(differences: Mapping[tuple[str, str, str, int], float],
              config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    seconds = float(config["minimum_gates"]["block_seconds"])
    blocks: dict[tuple[str, int], list[float]] = defaultdict(list)
    for event, value in differences.items():
        blocks[(event[1], int(event[3] // seconds))].append(value)
    if len(blocks) < int(config["minimum_gates"]["temporal_blocks_per_ci"]):
        return {"status": "LIMITED", "blocks": len(blocks), "ci95": None}
    ordered = [blocks[key] for key in sorted(blocks)]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(int(config["minimum_gates"]["bootstrap_iterations"])):
        choice = rng.integers(0, len(ordered), len(ordered))
        means.append(float(np.mean([value for index in choice for value in ordered[int(index)]])))
    return {"status": "AVAILABLE", "blocks": len(blocks),
            "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))]}


def calibration_and_pairs(method_rows: Sequence[Mapping[str, Any]],
                          config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute fixed-stratum calibration and paired relational estimands."""
    universe = universe_support(method_rows)
    supports = method_support(method_rows, sorted({m for values in COMPARISONS.values() for m in values}))
    calibration: dict[str, Any] = {"schema": "pg_scc_stage0_r2_calibration.v1", "cells": []}
    paired: dict[str, Any] = {"schema": "pg_scc_stage0_r2_paired_results.v1", "cells": [],
                              "relationship_permutation_required": True}
    calibration_index: dict[tuple[str, str], dict[str, Any]] = {}
    outcome_families = {"ds3": {"ds3"}, "ds4": {"ds4"}, "ds7_ds8": {"ds7", "ds8"}}
    seeds = [int(value) for value in config["minimum_gates"]["bootstrap_seeds"]]
    block_seconds = float(config["minimum_gates"]["block_seconds"])
    for family, methods in COMPARISONS.items():
        common = common_support_by_event(universe, supports, methods)
        scores = _pooled([row for row in method_rows if row["method"] in methods], common)
        for stratum in ("K9", "K5", "K3", "DENSE_ONLY"):
            selected = {event for event, prns in common.items()
                        if support_stratum(len(prns)) == stratum and family_is_eligible(family, len(prns))}
            for method_index, method in enumerate(methods):
                cal_events = [event for event in selected if event[0] == "clean" and event[1] == "cleanStatic" and event[2] == "calibration"]
                cal_values = [scores[method][event] for event in cal_events]
                minimum = int(config["minimum_gates"]["calibration_events_per_support_stratum"])
                status = "AVAILABLE" if len(cal_values) >= minimum else "UNAVAILABLE"
                threshold = (float(np.quantile(cal_values, config["calibration"]["quantile"], method="higher"))
                             if status == "AVAILABLE" else None)
                holdout = [scores[method][event] for event in selected
                           if event[0] == "clean" and event[1] == "cleanStatic" and event[2] == "holdout"]
                external = [scores[method][event] for event in selected
                            if event[0] == "attack" and event[2] == "strict_pre"]
                holdout_alarms = sum(value >= threshold for value in holdout) if threshold is not None else None
                external_alarms = sum(value >= threshold for value in external) if threshold is not None else None
                holdout_fpr = holdout_alarms / len(holdout) if holdout_alarms is not None and holdout else None
                external_fpr = external_alarms / len(external) if external_alarms is not None and external else None
                fprs = [value for value in (holdout_fpr, external_fpr) if value is not None]
                false_alarm_gate = bool(fprs) and max(fprs) <= float(config["calibration"]["clean_false_alarm_gate"])
                grouped: dict[int, list[float]] = defaultdict(list)
                for event in cal_events:
                    grouped[int(event[3] // block_seconds)].append(scores[method][event])
                blocks = [grouped[key] for key in sorted(grouped)]
                leave_one_out = []
                for omitted in range(len(blocks)):
                    sample = [value for index, block in enumerate(blocks) if index != omitted for value in block]
                    if sample:
                        leave_one_out.append(float(np.quantile(sample, config["calibration"]["quantile"], method="higher")))
                boot = []
                if status == "AVAILABLE" and blocks:
                    rng = np.random.default_rng(seeds[method_index % len(seeds)])
                    for _ in range(int(config["minimum_gates"]["bootstrap_iterations"])):
                        choice = rng.integers(0, len(blocks), len(blocks))
                        sample = [value for index in choice for value in blocks[int(index)]]
                        boot.append(float(np.quantile(sample, config["calibration"]["quantile"], method="higher")))
                cell = {
                    "family": family, "support_stratum": stratum, "method": method,
                    "status": status, "calibration_events": len(cal_values),
                    "calibration_blocks": len(blocks), "threshold_q99": threshold,
                    "threshold_leave_one_block_out_range": [min(leave_one_out), max(leave_one_out)] if leave_one_out else None,
                    "threshold_block_bootstrap_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))] if boot else None,
                    "clean_holdout_events": len(holdout), "clean_holdout_alarms": holdout_alarms,
                    "clean_holdout_fpr": holdout_fpr,
                    "clean_holdout_clopper_pearson_95": _exact_ci(holdout_alarms, len(holdout)) if holdout_alarms is not None else None,
                    "strict_external_pre_events": len(external), "strict_external_pre_alarms": external_alarms,
                    "strict_external_pre_fpr": external_fpr,
                    "strict_external_pre_clopper_pearson_95": _exact_ci(external_alarms, len(external)) if external_alarms is not None else None,
                    "false_alarm_gate": false_alarm_gate, "eligible_event_denominator": len(selected),
                }
                calibration["cells"].append(cell)
                calibration_index[(method, stratum)] = cell
            if len(methods) < 3:
                continue
            focal, ablation, permutation = methods
            require_identical_paired_support(common, common)
            for family_index, (outcome_family, scenarios) in enumerate(outcome_families.items()):
                events = {event for event in selected
                          if event[0] == "attack" and event[1] in scenarios and event[2] != "strict_pre"}
                for role, comparator in (("RELATIONAL_ABLATION", ablation), ("RELATIONSHIP_PERMUTATION", permutation)):
                    differences = {event: scores[focal][event] - scores[comparator][event] for event in events}
                    ci = _block_ci(differences, config, seeds[family_index])
                    negative_focal = [scores[focal][event] for event in selected
                                      if event[0] == "clean" and event[1] == "cleanStatic" and event[2] == "holdout"]
                    negative_comparator = [scores[comparator][event] for event in selected
                                           if event[0] == "clean" and event[1] == "cleanStatic" and event[2] == "holdout"]
                    positive_focal = [scores[focal][event] for event in events]
                    positive_comparator = [scores[comparator][event] for event in events]
                    enough = (len(positive_focal) >= int(config["minimum_gates"]["positive_events_per_family_support_stratum"])
                              and len(negative_focal) >= int(config["minimum_gates"]["clean_holdout_events_per_support_stratum"]))
                    focal_pauc = (float(roc_auc_score([0] * len(negative_focal) + [1] * len(positive_focal),
                                                      negative_focal + positive_focal, max_fpr=.05)) if enough else None)
                    comparator_pauc = (float(roc_auc_score([0] * len(negative_comparator) + [1] * len(positive_comparator),
                                                           negative_comparator + positive_comparator, max_fpr=.05)) if enough else None)
                    false_alarm_gate = calibration_index[(focal, stratum)]["false_alarm_gate"]
                    status = "AVAILABLE" if enough and ci["status"] == "AVAILABLE" and false_alarm_gate else "LIMITED"
                    support = {event: common[event] for event in events}
                    paired["cells"].append({
                        "k_family": family, "support_stratum": stratum,
                        "outcome_family": outcome_family, "focal_method": focal,
                        "comparator": comparator, "control_role": role, "status": status,
                        "false_alarm_gate": false_alarm_gate,
                        "eligible_event_denominator": len(selected), "paired_events": len(events),
                        "mean_paired_score_effect": float(np.mean(list(differences.values()))) if differences else None,
                        "ci95": ci["ci95"], "temporal_blocks": ci["blocks"],
                        "focal_normalized_low_fpr_pauc": focal_pauc,
                        "comparator_normalized_low_fpr_pauc": comparator_pauc,
                        "paired_pauc_difference": focal_pauc - comparator_pauc if focal_pauc is not None else None,
                        "support_fingerprint_left": _fingerprint(support),
                        "support_fingerprint_right": _fingerprint(support),
                    })
    available = [cell for cell in paired["cells"] if cell["status"] == "AVAILABLE"]
    effects = [float(cell["mean_paired_score_effect"]) for cell in available]
    weights = [int(cell["paired_events"]) for cell in available]
    by_stratum: dict[str, list[float]] = defaultdict(list)
    by_outcome_family: dict[str, list[float]] = defaultdict(list)
    for cell in available:
        by_stratum[str(cell["support_stratum"])].append(float(cell["mean_paired_score_effect"]))
        by_outcome_family[str(cell["outcome_family"])].append(float(cell["mean_paired_score_effect"]))
    stratum_means = {key: float(np.mean(values)) for key, values in sorted(by_stratum.items())}
    outcome_family_means = {key: float(np.mean(values)) for key, values in sorted(by_outcome_family.items())}
    paired["aggregate_estimands"] = {
        "available_cells": len(available), "total_preregistered_cells": len(paired["cells"]),
        "support_stratum_mean_paired_effects": stratum_means,
        "equal_stratum_mean_paired_effect": float(np.mean(list(stratum_means.values()))) if stratum_means else None,
        "outcome_family_mean_paired_effects": outcome_family_means,
        "equal_outcome_family_mean_paired_effect": float(np.mean(list(outcome_family_means.values()))) if outcome_family_means else None,
        "equal_cell_mean_paired_effect": float(np.mean(effects)) if effects else None,
        "event_count_weighted_mean_paired_effect": float(np.average(effects, weights=weights)) if effects and sum(weights) else None,
        "included_event_weight": sum(weights), "raw_scores_pooled_across_k_or_strata": False,
    }
    return calibration, paired


def _r1_module() -> Any:
    path = ROOT / "scripts/run_pg_scc_root_cause_audit.py"
    spec = importlib.util.spec_from_file_location("frozen_r1_control_kernel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen R1 control kernel")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def full_control_results(records: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray],
                         masks: Mapping[str, Sequence[int]], calibration: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute clean/synthetic selector stability, AWGN, and covariance controls."""
    r1 = _r1_module()
    normalized = np.asarray([normalize_complex(row["surface"], "prompt_phase") for row in records])
    train = np.asarray([i for i, row in enumerate(records) if row["source_role"] == "clean" and row["phase"] == "train"])
    selection = np.asarray([i for i, row in enumerate(records) if row["source_role"] == "clean" and row["phase"] == "selection"])
    if not len(train) or not len(selection):
        raise RuntimeError("clean train/selection roles unavailable")
    frozen_config = json.loads((FROZEN / "config.json").read_text(encoding="utf-8"))
    r1_config = json.loads(R1_CONFIG.read_text(encoding="utf-8"))
    bank = r1.build_synthetic_bank(
        np.asarray([records[i]["surface"] for i in train]),
        np.asarray([records[i]["surface"] for i in selection]),
        normalization="prompt_phase", seed=int(frozen_config["seed"]),
        max_h1_per_split=int(frozen_config["synthetic_bank"]["max_h1_per_split"]),
    )
    selector, seed_rows, random_rows = r1._selector_audit(
        r1_config, bank, arrays["auth_template"], arrays["covariance"], masks,
    )
    thresholds = {(cell["method"], cell["support_stratum"]): cell["threshold_q99"]
                  for cell in calibration["cells"] if cell["status"] == "AVAILABLE"}
    awgn = {}
    for method, stratum in (("pg_scc_k9", "K9"), ("pg_scc_k5", "K5"), ("pg_scc_k3", "K3")):
        threshold = thresholds.get((method, stratum))
        if threshold is not None:
            awgn[f"{method}:{stratum}"] = r1._awgn_audit(
                normalized[train], arrays["auth_template"], arrays["covariance"], masks[method], threshold,
            )
    variants = r1.covariance_variants(normalized[train], arrays["auth_template"], arrays["covariance"])
    covariance = {name: {"condition_number": float(np.linalg.cond(value)),
                         "source": "cleanStatic train only", "attack_based_choice": False}
                  for name, value in variants.items()}
    return {
        "schema": "pg_scc_stage0_r2_control_results.v1",
        "awgn": awgn, "awgn_status": "AVAILABLE" if awgn else "UNAVAILABLE",
        "selector_seed_stability": {
            "status": selector["status"], "seed_count": selector["seed_count"],
            "median_pairwise_jaccard": selector["median_pairwise_jaccard"],
            "learned_evidence": selector["learned_evidence"],
            "seed_rows": len(seed_rows), "random_control_rows": len(random_rows),
            "attack_based_selection": selector["attack_based_selection"],
        },
        "covariance_controls": covariance,
        "leakage_guards": {"attack_fit": False, "attack_based_selection": False, "post_attack_retuning": False},
        "relationship_controls": ["RELATIONAL_ABLATION", "RELATIONSHIP_PERMUTATION"],
    }
