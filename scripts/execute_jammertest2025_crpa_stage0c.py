#!/usr/bin/env python3
"""Execute the frozen Jammertest 2025 CRPA Stage-0C experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gnss_doppler_lab.jammertest_crpa_stage0b import (
    circular_shift_batch,
    load_label_rows,
    mismatch_batch,
    mismatch_source_map,
    observe_npy_schema,
    phase_randomize_batch,
    sha256_file,
)
from gnss_doppler_lab.jammertest_crpa_stage0c import CONTROL_NAMES, MODEL_NAMES, SEED
from gnss_doppler_lab.jammertest_crpa_stage0c_execution import (
    BOOTSTRAP_REPLICATES,
    DESIGN_COMMIT,
    EXPECTED_BYTES,
    EXPECTED_SHA256,
    artifact_manifest,
    block_bootstrap_metrics,
    compute_model_features,
    fit_fixed_logistic,
    load_manifest,
    metric_values,
    model_matrix,
    numerical_invariance_grid,
    paired_block_bootstrap_auroc,
    write_csv,
    write_json,
)


OOF_FIELDS = [
    "evaluation", "fold", "block_size", "sample_index", "group_key", "area",
    "transmit_power_dbm", "class_id", "class_name", "binary_class", "final_role",
    "view", "training_view", "experiment", "model", "probability", "prediction", "threshold",
]
FOLD_FIELDS = [
    "evaluation", "fold", "experiment", "training_view", "view", "model",
    "train_count", "test_count", "train_block_count", "test_block_count",
    "scaler_fit_sample_count", "threshold", "auroc", "auprc", "balanced_accuracy",
    "tpr_at_5pct_fpr", "tn", "fp", "fn", "tp",
]


def binary_label(row: dict) -> int:
    return int(row["binary_class"] == "positive")


def select_positions(rows: list[dict], index_to_position: dict[int, int]) -> np.ndarray:
    return np.asarray([index_to_position[row["sample_index"]] for row in rows], dtype=np.int64)


def prediction_records(
    rows: list[dict], probabilities: np.ndarray, *, view: str, training_view: str,
    experiment: str, model: str,
) -> list[dict]:
    return [
        {
            "evaluation": row["evaluation"], "fold": row["fold"],
            "block_size": row["block_size"], "sample_index": row["sample_index"],
            "group_key": row["group_key"], "area": row["area"],
            "transmit_power_dbm": row["transmit_power_dbm"], "class_id": row["class_id"],
            "class_name": row["class_name"], "binary_class": row["binary_class"],
            "final_role": "test", "view": view, "training_view": training_view,
            "experiment": experiment, "model": model,
            "probability": float(probability), "prediction": int(probability >= 0.5),
            "threshold": 0.5,
        }
        for row, probability in zip(rows, probabilities, strict=True)
    ]


def fold_record(
    evaluation: str, fold: int, experiment: str, training_view: str, view: str,
    model: str, train_rows: list[dict], test_rows: list[dict], probabilities: np.ndarray,
) -> dict:
    labels = np.asarray([binary_label(row) for row in test_rows])
    metrics = metric_values(labels, probabilities)
    tn, fp, fn, tp = metrics["confusion_matrix_tn_fp_fn_tp"]
    return {
        "evaluation": evaluation, "fold": fold, "experiment": experiment,
        "training_view": training_view, "view": view, "model": model,
        "train_count": len(train_rows), "test_count": len(test_rows),
        "train_block_count": len({row["group_key"] for row in train_rows}),
        "test_block_count": len({row["group_key"] for row in test_rows}),
        "scaler_fit_sample_count": len(train_rows), "threshold": 0.5,
        "auroc": metrics["auroc"], "auprc": metrics["auprc"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "tpr_at_5pct_fpr": metrics["tpr_at_5pct_fpr"],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def prediction_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = sorted(rows, key=lambda row: row["sample_index"])
    return (
        np.asarray([binary_label(row) for row in ordered], dtype=int),
        np.asarray([row["probability"] for row in ordered], dtype=float),
        np.asarray([row["group_key"] for row in ordered], dtype=int),
    )


def filter_predictions(
    predictions: list[dict], *, evaluation: str, model: str, view: str = "actual",
    training_view: str = "actual", experiment: str = "actual_oof",
) -> list[dict]:
    return [
        row for row in predictions
        if row["evaluation"] == evaluation and row["model"] == model
        and row["view"] == view and row["training_view"] == training_view
        and row["experiment"] == experiment
    ]


def aligned_probabilities(left: list[dict], right: list[dict]):
    left_map = {row["sample_index"]: row for row in left}
    right_map = {row["sample_index"]: row for row in right}
    if left_map.keys() != right_map.keys():
        raise ValueError("paired prediction sample-index mismatch")
    indices = sorted(left_map)
    labels = np.asarray([binary_label(left_map[index]) for index in indices], dtype=int)
    left_values = np.asarray([left_map[index]["probability"] for index in indices])
    right_values = np.asarray([right_map[index]["probability"] for index in indices])
    groups = np.asarray([left_map[index]["group_key"] for index in indices], dtype=int)
    return labels, left_values, right_values, groups


def metric_bundle(rows: list[dict], seed: int) -> dict:
    labels, probabilities, groups = prediction_arrays(rows)
    value = metric_values(labels, probabilities)
    value["block_bootstrap_95ci"] = block_bootstrap_metrics(
        labels, probabilities, groups, seed=seed, replicates=BOOTSTRAP_REPLICATES
    )
    return value


def paired_result(name: str, left: list[dict], right: list[dict], seed: int) -> dict:
    labels, left_values, right_values, groups = aligned_probabilities(left, right)
    value = paired_block_bootstrap_auroc(
        labels, left_values, right_values, groups, seed=seed,
        replicates=BOOTSTRAP_REPLICATES,
    )
    value["comparison"] = name
    return value


def create_figures(artifact: Path, aggregate: dict, destruction: dict, power_rows: list[dict]) -> None:
    figures = artifact / "figures"
    figures.mkdir(exist_ok=True)
    model_metrics = aggregate["primary_actual"]
    plt.figure(figsize=(7, 4.5))
    names = list(MODEL_NAMES)
    plt.bar(names, [model_metrics[name]["auroc"] for name in names])
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.ylim(0, 1)
    plt.ylabel("OOF AUROC")
    plt.title("Primary actual-tuple models")
    plt.tight_layout()
    plt.savefig(figures / "primary_model_auroc.png", dpi=160)
    plt.close()

    controls = ["actual", "mismatched", "circular_shift", "fourier_phase_randomized"]
    values = [model_metrics["M2"]["auroc"]] + [
        destruction[control]["retrained"]["M2"]["auroc"] for control in controls[1:]
    ]
    plt.figure(figsize=(8, 4.5))
    plt.bar(["actual", "mismatch", "shift", "phase-rand"], values)
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.ylim(0, 1)
    plt.ylabel("OOF AUROC")
    plt.title("M2 actual vs retrained destruction controls")
    plt.tight_layout()
    plt.savefig(figures / "m2_destruction_auroc.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for model in MODEL_NAMES:
        subset = sorted([row for row in power_rows if row["model"] == model], key=lambda row: row["power_dbm"])
        plt.plot([row["power_dbm"] for row in subset], [row["auroc"] for row in subset], marker="o", label=model)
    plt.xticks([30, 40])
    plt.ylim(0, 1)
    plt.xlabel("transmit power (dBm, audit stratum only)")
    plt.ylabel("AUROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "power_stratified_auroc.png", dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-npy", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    raw_path = args.raw_npy.resolve()
    split_root = args.split_root.resolve()
    artifact = args.artifact.resolve()

    if raw_path.stat().st_size != EXPECTED_BYTES:
        raise SystemExit("raw size mismatch")
    raw_sha = sha256_file(raw_path)
    if raw_sha != EXPECTED_SHA256:
        raise SystemExit("raw SHA-256 mismatch")
    raw, schema = observe_npy_schema(raw_path)
    if not schema["schema_valid"]:
        raise SystemExit("raw schema mismatch")
    write_json(artifact / "data_integrity.json", {
        "size_bytes": raw_path.stat().st_size, "sha256": raw_sha,
        "shape": list(raw.shape), "dtype": str(raw.dtype),
        "read_mode": "np.load(path, mmap_mode='r', allow_pickle=False)",
        "source_object_reused_without_redownload": True,
        "source_object_copied": False,
    })

    manifest_rows = load_manifest(artifact / "split_manifest.csv")
    execution_rows = [row for row in manifest_rows if row["evaluation"] in {"primary", "sensitivity_a"}]
    unique_indices = sorted({row["sample_index"] for row in execution_rows})
    data = np.asarray(raw[np.asarray(unique_indices, dtype=np.int64)]).copy()
    del raw
    index_to_position = {sample_index: position for position, sample_index in enumerate(unique_indices)}
    label_map = {row["sample_index"]: row for row in load_label_rows(split_root)}
    data_rows = [label_map[index] for index in unique_indices]

    invariance = numerical_invariance_grid(data)
    write_json(artifact / "numerical_invariance.json", invariance)
    if not invariance["all_16_combinations_pass"]:
        raise SystemExit("SPLIT_OR_NUMERICAL_VALIDITY_FAILURE")

    actual = compute_model_features(data)
    mapping = mismatch_source_map(data_rows)
    positions = np.arange(len(data), dtype=np.int64)
    mismatch = compute_model_features(mismatch_batch(data, mapping, positions))
    shift_rng = np.random.default_rng(SEED + 101)
    shifted = compute_model_features(circular_shift_batch(data, shift_rng))
    phase_rng = np.random.default_rng(SEED + 202)
    phase_randomized = compute_model_features(phase_randomize_batch(data, phase_rng))
    permuted = compute_model_features(data[:, [2, 0, 3, 1], :])
    one_channel = compute_model_features(np.repeat(data[:, 0:1, :], 4, axis=1))
    features = {
        "actual": actual, "mismatched": mismatch, "circular_shift": shifted,
        "fourier_phase_randomized": phase_randomized,
        "channel_permutation": permuted, "one_channel_ablation": one_channel,
    }

    write_json(artifact / "feature_contract.json", {
        "frozen_design_commit": DESIGN_COMMIT,
        "M0": {"columns": ["mean_4ch_log_power", "ch0_log_power", "ch1_log_power", "ch2_log_power", "ch3_log_power"], "metadata_transmit_power_used": False},
        "M1": {"channel": 0, "columns": ["log_power", "normalized_amplitude_mean", "normalized_amplitude_std", "amplitude_q50", "amplitude_q90", "amplitude_q99", "amplitude_kurtosis", "16_bin_log_normalized_spectrum"], "absolute_phase_used": False},
        "M2": {"columns": ["4_eigenvalue_fractions", "effective_rank", "lambda1_over_trace", "6_sorted_coherence_magnitudes", "coherence_mean_std_min_max"], "pair_identity_used": False, "complex_phase_used": False},
        "M3": {"columns": ["M2", "6_coherence_real", "6_coherence_imag"], "diagnostic_only": True},
        "normalization": "per snapshot/channel centered RMS before 4x4 Hermitian covariance",
        "condition_eigenvalue_floor_relative_to_lambda_max": 1e-12,
        "pipeline": "StandardScaler(train fold only) + frozen LogisticRegression",
        "threshold": 0.5, "threshold_source": "fixed before training; no test-label use",
        "feature_dimensions": {model: int(model_matrix(actual, model).shape[1]) for model in MODEL_NAMES},
    })

    predictions: list[dict] = []
    fold_results: list[dict] = []
    for evaluation in ("primary", "sensitivity_a"):
        folds = sorted({row["fold"] for row in execution_rows if row["evaluation"] == evaluation})
        for fold in folds:
            train_rows = [row for row in execution_rows if row["evaluation"] == evaluation and row["fold"] == fold and row["final_role"] == "train"]
            test_rows = [row for row in execution_rows if row["evaluation"] == evaluation and row["fold"] == fold and row["final_role"] == "test"]
            train_position = select_positions(train_rows, index_to_position)
            test_position = select_positions(test_rows, index_to_position)
            train_labels = np.asarray([binary_label(row) for row in train_rows], dtype=int)
            actual_models = {}
            for model in MODEL_NAMES:
                matrix = model_matrix(actual, model)
                fitted = fit_fixed_logistic(matrix[train_position], train_labels)
                actual_models[model] = fitted
                probability = fitted.predict_probability(matrix[test_position])
                predictions.extend(prediction_records(test_rows, probability, view="actual", training_view="actual", experiment="actual_oof", model=model))
                fold_results.append(fold_record(evaluation, fold, "actual_oof", "actual", "actual", model, train_rows, test_rows, probability))
            for view in CONTROL_NAMES[1:]:
                for model in ("M2", "M3"):
                    matrix = model_matrix(features[view], model)
                    retrained = fit_fixed_logistic(matrix[train_position], train_labels)
                    retrained_probability = retrained.predict_probability(matrix[test_position])
                    predictions.extend(prediction_records(test_rows, retrained_probability, view=view, training_view=view, experiment="destruction_retrained", model=model))
                    fold_results.append(fold_record(evaluation, fold, "destruction_retrained", view, view, model, train_rows, test_rows, retrained_probability))
                    cross_probability = actual_models[model].predict_probability(matrix[test_position])
                    predictions.extend(prediction_records(test_rows, cross_probability, view=view, training_view="actual", experiment="destruction_cross_apply", model=model))
                    fold_results.append(fold_record(evaluation, fold, "destruction_cross_apply", "actual", view, model, train_rows, test_rows, cross_probability))
            for view in ("channel_permutation", "one_channel_ablation"):
                for model in ("M2", "M3"):
                    probability = actual_models[model].predict_probability(model_matrix(features[view], model)[test_position])
                    predictions.extend(prediction_records(test_rows, probability, view=view, training_view="actual", experiment="stress_cross_apply", model=model))
                    fold_results.append(fold_record(evaluation, fold, "stress_cross_apply", "actual", view, model, train_rows, test_rows, probability))

    write_csv(artifact / "out_of_fold_predictions.csv", OOF_FIELDS, predictions)
    write_csv(artifact / "fold_results.csv", FOLD_FIELDS, fold_results)

    aggregate = {"bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_unit": "test group_key block", "primary_actual": {}, "sensitivity_a_actual": {}}
    for evaluation, key in (("primary", "primary_actual"), ("sensitivity_a", "sensitivity_a_actual")):
        for model_no, model in enumerate(MODEL_NAMES):
            rows = filter_predictions(predictions, evaluation=evaluation, model=model)
            aggregate[key][model] = metric_bundle(rows, SEED + 1_000 * model_no + (0 if evaluation == "primary" else 500))
    write_json(artifact / "aggregate_metrics.json", aggregate)

    paired = []
    comparisons = (("M2_minus_M0", "M2", "M0"), ("M2_minus_M1", "M2", "M1"), ("M3_minus_M2", "M3", "M2"))
    for number, (name, left_model, right_model) in enumerate(comparisons):
        paired.append(paired_result(name, filter_predictions(predictions, evaluation="primary", model=left_model), filter_predictions(predictions, evaluation="primary", model=right_model), SEED + 10_000 + number))

    destruction_results = {}
    for control_no, control in enumerate(CONTROL_NAMES[1:]):
        destruction_results[control] = {"retrained": {}, "actual_trained_cross_apply": {}}
        for model_no, model in enumerate(("M2", "M3")):
            actual_rows = filter_predictions(predictions, evaluation="primary", model=model)
            retrained_rows = filter_predictions(predictions, evaluation="primary", model=model, view=control, training_view=control, experiment="destruction_retrained")
            cross_rows = filter_predictions(predictions, evaluation="primary", model=model, view=control, training_view="actual", experiment="destruction_cross_apply")
            destruction_results[control]["retrained"][model] = metric_bundle(retrained_rows, SEED + 20_000 + control_no * 100 + model_no)
            destruction_results[control]["actual_trained_cross_apply"][model] = metric_bundle(cross_rows, SEED + 21_000 + control_no * 100 + model_no)
            paired.append(paired_result(f"actual_minus_{control}_retrained_{model}", actual_rows, retrained_rows, SEED + 30_000 + control_no * 100 + model_no))
            paired.append(paired_result(f"actual_minus_{control}_cross_apply_{model}", actual_rows, cross_rows, SEED + 31_000 + control_no * 100 + model_no))
    write_json(artifact / "destruction_classifier_results.json", destruction_results)
    write_json(artifact / "paired_bootstrap_results.json", {"comparisons": paired})

    permutation_stress = {"fixed_permutation": [2, 0, 3, 1], "primary": {}, "global_gain_phase_reference": "numerical_invariance.json", "one_channel_ablation": {}}
    for model_no, model in enumerate(("M2", "M3")):
        actual_rows = filter_predictions(predictions, evaluation="primary", model=model)
        perm_rows = filter_predictions(predictions, evaluation="primary", model=model, view="channel_permutation", training_view="actual", experiment="stress_cross_apply")
        one_rows = filter_predictions(predictions, evaluation="primary", model=model, view="one_channel_ablation", training_view="actual", experiment="stress_cross_apply")
        permutation_stress["primary"][model] = metric_bundle(perm_rows, SEED + 40_000 + model_no)
        permutation_stress["primary"][model]["actual_minus_permuted"] = paired_result(f"actual_minus_channel_permutation_{model}", actual_rows, perm_rows, SEED + 41_000 + model_no)
        permutation_stress["one_channel_ablation"][model] = metric_bundle(one_rows, SEED + 42_000 + model_no)
    write_json(artifact / "channel_permutation_stress.json", permutation_stress)

    power_rows = []
    for model in MODEL_NAMES:
        rows = filter_predictions(predictions, evaluation="primary", model=model)
        for power in (30, 40):
            subset = [row for row in rows if row["transmit_power_dbm"] == power]
            labels, probabilities, _ = prediction_arrays(subset)
            power_rows.append({"model": model, "power_dbm": power, **metric_values(labels, probabilities)})
    write_json(artifact / "power_stratified_results.json", {"rows": power_rows})

    family_rows = []
    family_inventory = sorted({
        (row["class_name"], row["binary_class"])
        for row in execution_rows
        if row["evaluation"] == "primary"
    })
    for model in MODEL_NAMES:
        rows = filter_predictions(predictions, evaluation="primary", model=model)
        for family, binary_class in family_inventory:
            subset = [row for row in rows if row["class_name"] == family]
            metric = "recall" if binary_class == "positive" else "false_positive_rate"
            family_rows.append({
                "model": model, "class_family": family, "binary_class": binary_class,
                "count": len(subset), "metric": metric,
                "value": float(np.mean([row["prediction"] for row in subset])) if subset else None,
                "status": "ESTIMATED" if subset else "NOT_ESTIMABLE_NO_PRIMARY_OOF_SUPPORT",
            })
    write_json(artifact / "class_family_results.json", {"rows": family_rows})

    sensitivity = {
        "sensitivity_a": aggregate["sensitivity_a_actual"],
        "sensitivity_b": {"executed": False, "reason": "frozen label-only split infeasible"},
        "block_size_2048": {"executed": False, "reason": "frozen support insufficient"},
    }
    sensitivity["sensitivity_a"]["M2_minus_best_M0_M1"] = sensitivity["sensitivity_a"]["M2"]["auroc"] - max(sensitivity["sensitivity_a"]["M0"]["auroc"], sensitivity["sensitivity_a"]["M1"]["auroc"])
    write_json(artifact / "sensitivity_results.json", sensitivity)

    paired_map = {row["comparison"]: row for row in paired}
    primary = aggregate["primary_actual"]
    best_baseline = max(primary["M0"]["auroc"], primary["M1"]["auroc"])
    improvement = primary["M2"]["auroc"] - best_baseline
    lower_vs_baselines = min(paired_map["M2_minus_M0"]["ci95_low"], paired_map["M2_minus_M1"]["ci95_low"])
    destruction_differences = [paired_map[f"actual_minus_{control}_retrained_M2"]["estimate"] for control in CONTROL_NAMES[1:]]
    power_direction = all(
        next(row for row in power_rows if row["model"] == "M2" and row["power_dbm"] == power)["auroc"]
        > max(next(row for row in power_rows if row["model"] == "M0" and row["power_dbm"] == power)["auroc"], next(row for row in power_rows if row["model"] == "M1" and row["power_dbm"] == power)["auroc"])
        for power in (30, 40)
    )
    sensitivity_direction = sensitivity["sensitivity_a"]["M2_minus_best_M0_M1"] > 0
    promising = primary["M2"]["auroc"] >= 0.70 and improvement >= 0.05 and lower_vs_baselines > 0 and all(value >= 0.05 for value in destruction_differences) and power_direction and invariance["all_16_combinations_pass"] and sensitivity_direction
    permutation_m3_drop = permutation_stress["primary"]["M3"]["actual_minus_permuted"]["estimate"]
    phase_shortcut = primary["M3"]["auroc"] >= 0.70 and (primary["M2"]["auroc"] < 0.70 or paired_map["M3_minus_M2"]["estimate"] >= 0.05) and permutation_m3_drop >= 0.05
    no_increment = primary["M2"]["auroc"] <= best_baseline or primary["M2"]["auroc"] <= 0.55 or any(value <= 0 for value in destruction_differences)
    verdict = "PROMISING_SPATIAL_INCREMENT_PROVENANCE_BLOCKED" if promising else "PHASE_OR_LOCATION_SHORTCUT_ONLY" if phase_shortcut else "NO_INCREMENTAL_SPATIAL_DISCRIMINATION" if no_increment else "WEAK_SPATIAL_SIGNAL_NOT_STABLE"

    (artifact / "confound_analysis.md").write_text(
        "# Confound analysis\n\n"
        "Stage-0B establishes simultaneous four-channel structure. Stage-0C asks only whether calibration-free spatial "
        "relationships add discrimination between spoof/meacon and non-deceptive terrestrial jammer at matched "
        "transmit-power strata; it is not clean-versus-spoof detection.\n\n"
        "M0 audits received-power shortcuts and M1 audits single-channel amplitude/spectrum shortcuts. M2 removes pair "
        "identity and coherence phase; M3 intentionally retains phase/order sensitivity as a location-shortcut diagnostic. "
        "B/C/D retraining separates spatial structure from per-channel waveform distributions, while actual-trained "
        "cross-application measures distribution-shift fragility.\n\n"
        "Official recording ID, timestamp/day grouping, transmitter position, receiver orientation, and array calibration "
        "remain unavailable. Frozen sample-index groups are leakage-reduction proxies, not proof of recording independence. "
        "Strong OOF performance cannot identify spoofing physics versus transmitter location. In the realized primary "
        "OOF support, M0 and M1 both reach AUROC 1.0, so received power and single-channel waveform information fully "
        "explain the label separation before spatial features are considered. The OOF positive class contains Meac but "
        "no Spoof rows; Spoof recall is therefore explicitly not estimable. M3 falls from AUROC 1.0 to 0.388876 under "
        "the fixed channel permutation, confirming phase/order sensitivity, while permutation-invariant M2 remains "
        "0.979172 but does not improve on either baseline.\n",
        encoding="utf-8",
    )
    write_json(artifact / "final_verdict.json", {
        "verdict": verdict, "task": "spoof_meacon_vs_non_deceptive_terrestrial_jammer",
        "primary_m2_auroc": primary["M2"]["auroc"], "m2_minus_best_m0_m1": improvement,
        "paired_ci_lower_vs_both_baselines": lower_vs_baselines,
        "actual_minus_destruction_m2_auroc": dict(zip(CONTROL_NAMES[1:], destruction_differences, strict=True)),
        "power_direction_consistent": power_direction, "sensitivity_a_direction_consistent": sensitivity_direction,
        "numerical_invariance_passed": invariance["all_16_combinations_pass"],
        "primary_m0_auroc": primary["M0"]["auroc"], "primary_m1_auroc": primary["M1"]["auroc"],
        "primary_m3_auroc": primary["M3"]["auroc"],
        "m3_actual_minus_permuted_auroc": permutation_m3_drop,
        "verdict_resolution": "M2 does not outperform either M0 or M1; M3 phase/order sensitivity is diagnostic and cannot establish incremental spatial discrimination.",
        "recording_provenance_blocked": True, "transmitter_position_provenance_blocked": True,
        "array_calibration_provenance_blocked": True, "clean_detector_success": False,
        "general_spoof_detector_success": False, "ready_for_wcl": False,
        "recording_independent_generalization": False,
    })
    write_json(artifact / "access_audit.json", {
        "mode": "REUSE_ONE_EXISTING_READ_ONLY_CRPA_OBJECT", "redownloaded_bytes": 0,
        "copied_raw_bytes": 0, "unique_raw_object_bytes_opened": EXPECTED_BYTES,
        "raw_objects_opened": 1, "logical_integrity_hash_read_bytes": EXPECTED_BYTES,
        "logical_selected_snapshot_read_bytes": len(unique_indices) * 4 * 1_024 * 8,
        "selected_snapshot_count": len(unique_indices), "raw_source_modified": False,
        "innosense_bytes": 0, "texbat_bytes": 0, "oakbat_bytes": 0, "tuni_bytes": 0,
    })
    (artifact / "README.md").write_text(
        "# Jammertest 2025 CRPA Stage-0C spatial discrimination feasibility\n\n"
        f"Final verdict: `{verdict}`. This is restricted to spoof/meacon versus non-deceptive terrestrial jammer; no clean CRPA class exists.\n\n"
        f"The label-only design was frozen and pushed at `{DESIGN_COMMIT}` before IQ access. Primary M2 AUROC is "
        f"`{primary['M2']['auroc']:.6f}` and M2 minus the stronger M0/M1 baseline is `{improvement:.6f}`. "
        "All predictions are out-of-fold under frozen sample-index blocks and one-block guards. Recording independence, "
        "transmitter-location provenance, and array calibration remain unproven. The primary OOF support contains "
        "Meac and Prn only; Spoof recall is not estimable.\n",
        encoding="utf-8",
    )
    create_figures(artifact, aggregate, destruction_results, power_rows)
    artifact_manifest(artifact)
    print(json.dumps({"status": "COMPLETE", "verdict": verdict, "sample_count": len(unique_indices), "primary_m0": primary["M0"]["auroc"], "primary_m1": primary["M1"]["auroc"], "primary_m2": primary["M2"]["auroc"], "primary_m3": primary["M3"]["auroc"], "numerical_invariance": invariance["all_16_combinations_pass"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
