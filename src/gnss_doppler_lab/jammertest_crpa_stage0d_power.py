"""Received-power audit and frozen fold-local matching for Stage-0D."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from gnss_doppler_lab.jammertest_crpa_stage0b import observe_npy_schema, sha256_file
from gnss_doppler_lab.jammertest_crpa_stage0d import (
    BASE_COMMIT,
    CALIPERS_DB,
    PRIMARY_CALIPER_DB,
    SEED,
    write_csv,
    write_json,
)


EXPECTED_BYTES = 1_398_308_992
EXPECTED_SHA256 = "d869fa20d552288002e4d2a5b6c5d1300083a6348c01a956cd6a34ff232e0a3f"
DESIGN_FREEZE_COMMIT = "33c3c6924f2a7f42ca964e1bd136239cacaea04c"


@dataclass(frozen=True)
class PowerPair:
    spoof_sample_index: int
    prn_sample_index: int
    absolute_power_difference_db: float


def load_split(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for name in ("fold", "sample_index", "binary_label", "area", "transmit_power_dbm", "block_size", "class_block"):
            row[name] = int(row[name])
    return rows


def _better(candidate: tuple[int, float, int], incumbent: tuple[int, float, int]) -> bool:
    """Lexicographic max-cardinality, min-cost, deterministic priority."""

    return (-candidate[0], candidate[1], candidate[2]) < (-incumbent[0], incumbent[1], incumbent[2])


def maximum_cardinality_minimum_cost_match(
    spoof: list[tuple[int, float]],
    prn: list[tuple[int, float]],
    caliper_db: float,
) -> list[PowerPair]:
    """Solve one-dimensional caliper matching without replacement.

    Absolute distance is Monge, so an optimal non-crossing solution exists;
    dynamic programming therefore gives maximum cardinality and then minimum
    total cost. Input and output tie-breaking are sample-index deterministic.
    """

    left = sorted(spoof, key=lambda item: (item[1], item[0]))
    right = sorted(prn, key=lambda item: (item[1], item[0]))
    n, m = len(left), len(right)
    counts = np.zeros((n + 1, m + 1), dtype=np.int32)
    costs = np.zeros((n + 1, m + 1), dtype=np.float64)
    choice = np.zeros((n + 1, m + 1), dtype=np.int8)
    # choice: 1 skip left, 2 skip right, 3 match. Priority on exact ties is
    # match, then skip left, then skip right.
    for i in range(1, n + 1):
        choice[i, 0] = 1
    for j in range(1, m + 1):
        choice[0, j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = [
                (int(counts[i - 1, j]), float(costs[i - 1, j]), 1),
                (int(counts[i, j - 1]), float(costs[i, j - 1]), 2),
            ]
            difference = abs(left[i - 1][1] - right[j - 1][1])
            if difference <= caliper_db + 1e-12:
                candidates.append((
                    int(counts[i - 1, j - 1]) + 1,
                    float(costs[i - 1, j - 1]) + difference,
                    0,
                ))
            best = candidates[0]
            for candidate in candidates[1:]:
                if _better(candidate, best):
                    best = candidate
            counts[i, j] = best[0]
            costs[i, j] = best[1]
            choice[i, j] = 3 if best[2] == 0 else best[2]

    pairs: list[PowerPair] = []
    i, j = n, m
    while i > 0 or j > 0:
        selected = int(choice[i, j])
        if selected == 3:
            difference = abs(left[i - 1][1] - right[j - 1][1])
            pairs.append(PowerPair(left[i - 1][0], right[j - 1][0], difference))
            i -= 1
            j -= 1
        elif selected == 1:
            i -= 1
        elif selected == 2:
            j -= 1
        else:
            break
    pairs.reverse()
    return pairs


def received_power_db(values: np.ndarray) -> np.ndarray:
    power = np.mean(np.abs(np.asarray(values, dtype=np.complex128)) ** 2, axis=(1, 2))
    return 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny))


def standardized_mean_difference(labels: np.ndarray, values: np.ndarray) -> float:
    positive = values[labels == 1]
    negative = values[labels == 0]
    pooled = np.sqrt((np.var(positive, ddof=1) + np.var(negative, ddof=1)) / 2)
    return float((np.mean(positive) - np.mean(negative)) / pooled) if pooled > 0 else 0.0


def power_summary(labels: np.ndarray, values: np.ndarray) -> dict:
    result = {}
    for label, name in ((1, "Spoof"), (0, "Prn")):
        selected = values[labels == label]
        result[name] = {
            "count": int(len(selected)),
            "min_db": float(np.min(selected)),
            "max_db": float(np.max(selected)),
            "mean_db": float(np.mean(selected)),
            "std_db": float(np.std(selected, ddof=1)),
            "quantiles_db": {str(q): float(np.quantile(selected, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)},
        }
    result["standardized_mean_difference_spoof_minus_prn"] = standardized_mean_difference(labels, values)
    support_low = max(result["Spoof"]["min_db"], result["Prn"]["min_db"])
    support_high = min(result["Spoof"]["max_db"], result["Prn"]["max_db"])
    result["common_support_db"] = [support_low, support_high] if support_low <= support_high else None
    result["common_support_exists"] = support_low <= support_high
    result["disjoint_power_range_gap_db"] = max(0.0, support_low - support_high)
    raw_auc = float(roc_auc_score(labels, values))
    result["raw_received_power_auc"] = raw_auc
    result["raw_received_power_orientation_free_auc"] = max(raw_auc, 1.0 - raw_auc)
    return result


def fit_m0(train_power: np.ndarray, train_labels: np.ndarray):
    scaler = StandardScaler().fit(train_power[:, None])
    classifier = LogisticRegression(
        penalty="l2", C=1.0, solver="liblinear", class_weight="balanced",
        max_iter=2_000, random_state=SEED,
    ).fit(scaler.transform(train_power[:, None]), train_labels)
    return scaler, classifier


def metric_values(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities >= 0.5
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "count": int(len(labels)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "confusion_matrix_tn_fp_fn_tp": [int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1])],
        "threshold": 0.5,
    }


def matched_indices(pairs: list[PowerPair]) -> list[int]:
    return [index for pair in pairs for index in (pair.spoof_sample_index, pair.prn_sample_index)]


def build_power_match_freeze(raw_path: Path, artifact: Path) -> dict:
    if raw_path.stat().st_size != EXPECTED_BYTES or sha256_file(raw_path) != EXPECTED_SHA256:
        raise SystemExit("RAW_INTEGRITY_FAILURE")
    raw, schema = observe_npy_schema(raw_path)
    if not schema["schema_valid"]:
        raise SystemExit("RAW_SCHEMA_FAILURE")
    split_rows = load_split(artifact / "split_manifest.csv")
    sample_indices = sorted({row["sample_index"] for row in split_rows})
    values = np.asarray(raw[np.asarray(sample_indices, dtype=np.int64)]).copy()
    del raw
    powers = received_power_db(values)
    power_by_sample = dict(zip(sample_indices, powers, strict=True))
    class_by_sample = {
        row["sample_index"]: row["class_name"]
        for row in split_rows
    }
    labels = np.asarray([int(class_by_sample[index] == "Spoof") for index in sample_indices])
    write_json(artifact / "data_integrity.json", {
        "size_bytes": EXPECTED_BYTES,
        "sha256": EXPECTED_SHA256,
        "shape": list(values.shape[:1]) + [4, 1024],
        "source_shape": schema["shape"],
        "dtype": str(values.dtype),
        "read_mode": "np.load(path, mmap_mode='r', allow_pickle=False)",
        "selected_snapshot_count": len(sample_indices),
        "source_reused_without_redownload_or_copy": True,
    })

    full_summary = power_summary(labels, powers)
    pre_predictions = []
    for fold in range(5):
        current = [row for row in split_rows if row["fold"] == fold]
        train = [row for row in current if row["role"] == "train"]
        test = [row for row in current if row["role"] == "test"]
        train_power = np.asarray([power_by_sample[row["sample_index"]] for row in train])
        train_labels = np.asarray([row["binary_label"] for row in train])
        test_power = np.asarray([power_by_sample[row["sample_index"]] for row in test])
        scaler, classifier = fit_m0(train_power, train_labels)
        probability = classifier.predict_proba(scaler.transform(test_power[:, None]))[:, 1]
        pre_predictions.extend(
            (row["sample_index"], row["binary_label"], probability)
            for row, probability in zip(test, probability, strict=True)
        )
    pre_predictions.sort()
    pre_labels = np.asarray([row[1] for row in pre_predictions])
    pre_probability = np.asarray([row[2] for row in pre_predictions])
    full_summary["track_a_m0_oof"] = metric_values(pre_labels, pre_probability)
    write_json(artifact / "power_distribution.json", full_summary)

    manifest_rows: list[dict] = []
    caliper_results = {}
    for caliper in CALIPERS_DB:
        post_predictions = []
        per_fold = []
        for fold in range(5):
            current = [row for row in split_rows if row["fold"] == fold]
            role_pairs = {}
            for role in ("train", "test"):
                role_rows = [row for row in current if row["role"] == role]
                spoof = [(row["sample_index"], power_by_sample[row["sample_index"]]) for row in role_rows if row["class_name"] == "Spoof"]
                prn = [(row["sample_index"], power_by_sample[row["sample_index"]]) for row in role_rows if row["class_name"] == "Prn"]
                pairs = maximum_cardinality_minimum_cost_match(spoof, prn, caliper)
                role_pairs[role] = pairs
                for pair_number, pair in enumerate(pairs):
                    manifest_rows.append({
                        "caliper_db": f"{caliper:.2f}", "fold": fold, "role": role,
                        "pair_number": pair_number,
                        "spoof_sample_index": pair.spoof_sample_index,
                        "spoof_class_block": pair.spoof_sample_index // 32,
                        "spoof_received_power_db": power_by_sample[pair.spoof_sample_index],
                        "prn_sample_index": pair.prn_sample_index,
                        "prn_class_block": pair.prn_sample_index // 32,
                        "prn_received_power_db": power_by_sample[pair.prn_sample_index],
                        "absolute_power_difference_db": pair.absolute_power_difference_db,
                    })
            train_indices = matched_indices(role_pairs["train"])
            test_indices = matched_indices(role_pairs["test"])
            if role_pairs["train"] and role_pairs["test"]:
                train_power = np.asarray([power_by_sample[index] for index in train_indices])
                train_labels = np.asarray([int(class_by_sample[index] == "Spoof") for index in train_indices])
                test_power = np.asarray([power_by_sample[index] for index in test_indices])
                test_labels = np.asarray([int(class_by_sample[index] == "Spoof") for index in test_indices])
                scaler, classifier = fit_m0(train_power, train_labels)
                probability = classifier.predict_proba(scaler.transform(test_power[:, None]))[:, 1]
                post_predictions.extend(zip(test_indices, test_labels, probability, strict=True))
            all_pairs = role_pairs["test"]
            pair_differences = [pair.absolute_power_difference_db for pair in all_pairs]
            per_fold.append({
                "fold": fold,
                "train_pair_count": len(role_pairs["train"]),
                "test_pair_count": len(role_pairs["test"]),
                "test_mean_pair_difference_db": float(np.mean(pair_differences)) if pair_differences else None,
                "test_max_pair_difference_db": float(np.max(pair_differences)) if pair_differences else None,
            })
        post_predictions.sort()
        post_labels = np.asarray([row[1] for row in post_predictions], dtype=int)
        post_probability = np.asarray([row[2] for row in post_predictions], dtype=float)
        test_manifest = [row for row in manifest_rows if row["caliper_db"] == f"{caliper:.2f}" and row["role"] == "test"]
        matched_samples = [
            (row["spoof_sample_index"], 1, row["spoof_received_power_db"], row["spoof_class_block"])
            for row in test_manifest
        ] + [
            (row["prn_sample_index"], 0, row["prn_received_power_db"], row["prn_class_block"])
            for row in test_manifest
        ]
        matched_labels = np.asarray([row[1] for row in matched_samples], dtype=int)
        matched_power = np.asarray([row[2] for row in matched_samples], dtype=float)
        differences = [row["absolute_power_difference_db"] for row in test_manifest]
        caliper_results[f"{caliper:.2f}"] = {
            "caliper_db": caliper,
            "per_fold": per_fold,
            "total_test_pair_count": len(test_manifest),
            "total_train_pair_count_across_folds": sum(row["train_pair_count"] for row in per_fold),
            "retained_spoof_blocks": sorted({row["spoof_class_block"] for row in test_manifest}),
            "retained_prn_blocks": sorted({row["prn_class_block"] for row in test_manifest}),
            "retained_by_class_block": dict(sorted(Counter(
                f"{'Spoof' if label else 'Prn'}:{block}"
                for _, label, _, block in matched_samples
            ).items())),
            "mean_pair_difference_db": float(np.mean(differences)) if differences else None,
            "max_pair_difference_db": float(np.max(differences)) if differences else None,
            "matched_power_standardized_mean_difference": standardized_mean_difference(matched_labels, matched_power) if len(matched_power) else None,
            "matched_common_support_db": power_summary(matched_labels, matched_power)["common_support_db"] if len(matched_power) else None,
            "track_b_m0_oof": metric_values(post_labels, post_probability) if len(np.unique(post_labels)) == 2 else None,
        }

    write_csv(
        artifact / "power_match_manifest.csv",
        ["caliper_db", "fold", "role", "pair_number", "spoof_sample_index", "spoof_class_block", "spoof_received_power_db", "prn_sample_index", "prn_class_block", "prn_received_power_db", "absolute_power_difference_db"],
        manifest_rows,
    )
    primary = caliper_results[f"{PRIMARY_CALIPER_DB:.2f}"]
    gate_conditions = {
        "total_test_matched_pairs_at_least_50": primary["total_test_pair_count"] >= 50,
        "at_least_4_spoof_blocks_retained": len(primary["retained_spoof_blocks"]) >= 4,
        "at_least_4_prn_blocks_retained": len(primary["retained_prn_blocks"]) >= 4,
        "matched_m0_auroc_at_most_0_60": primary["track_b_m0_oof"] is not None and primary["track_b_m0_oof"]["auroc"] <= 0.60,
    }
    freeze = {
        "status": "POWER_MATCH_FREEZE_PRE_SPATIAL_SCORING",
        "base_commit": BASE_COMMIT,
        "design_freeze_commit": DESIGN_FREEZE_COMMIT,
        "source_binding": {
            "raw_size_bytes": EXPECTED_BYTES,
            "raw_sha256": EXPECTED_SHA256,
            "design_freeze_sha256": sha256_file(artifact / "design_freeze.json"),
            "split_manifest_sha256": sha256_file(artifact / "split_manifest.csv"),
        },
        "algorithm": "fold-local one-dimensional dynamic programming; maximum cardinality then minimum total absolute difference; no replacement",
        "calipers": caliper_results,
        "primary_caliper_db": PRIMARY_CALIPER_DB,
        "primary_gate_conditions": gate_conditions,
        "primary_gate_pass": all(gate_conditions.values()),
        "spatial_feature_bytes_computed": 0,
        "matching_inputs": ["class label", "received_power_db"],
        "train_test_cross_pairing": False,
    }
    write_json(artifact / "power_match_freeze.json", freeze)
    return freeze
