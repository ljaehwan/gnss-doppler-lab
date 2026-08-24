#!/usr/bin/env python3
"""Execute frozen Stage-0D Track A after both prerequisite freezes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gnss_doppler_lab.jammertest_crpa_stage0b import (
    circular_shift_batch, mismatch_batch, mismatch_source_map,
    observe_npy_schema, phase_randomize_batch, sha256_file,
)
from gnss_doppler_lab.jammertest_crpa_stage0d import CONTROL_NAMES, MODEL_NAMES, SEED, SPATIAL_MODELS
from gnss_doppler_lab.jammertest_crpa_stage0d_execution import (
    BOOTSTRAP_REPLICATES, DESIGN_FREEZE_COMMIT, POWER_MATCH_FREEZE_COMMIT,
    artifact_manifest, block_bootstrap_metrics, compute_features, correlations,
    feature_error, fit_pipeline, hash_indices, metric_values,
    paired_block_bootstrap, predict_pipeline, read_json, write_csv, write_json,
)
from gnss_doppler_lab.jammertest_crpa_stage0d_power import (
    EXPECTED_BYTES, EXPECTED_SHA256, load_split, received_power_db,
)


PREDICTION_FIELDS = [
    "track", "caliper_db", "fold", "sample_index", "class_name", "binary_label",
    "class_block", "class_block_key", "role", "experiment", "model", "view",
    "training_view", "probability", "prediction", "threshold",
]
FOLD_FIELDS = [
    "track", "fold", "experiment", "model", "view", "training_view", "train_count",
    "test_count", "train_spoof_count", "train_prn_count", "test_spoof_count",
    "test_prn_count", "train_block_count", "test_block_count", "scaler_fit_sample_count",
    "threshold", "auroc", "auprc", "balanced_accuracy", "tpr_at_5pct_fpr",
    "spoof_recall", "prn_false_positive_rate", "tn", "fp", "fn", "tp",
]


def positions(rows, index_to_position):
    return np.asarray([index_to_position[row["sample_index"]] for row in rows], dtype=np.int64)


def prediction_rows(rows, probabilities, *, model, experiment, view, training_view):
    return [{
        "track": "A", "caliper_db": "", "fold": row["fold"],
        "sample_index": row["sample_index"], "class_name": row["class_name"],
        "binary_label": row["binary_label"], "class_block": row["class_block"],
        "class_block_key": row["class_block_key"], "role": "test",
        "experiment": experiment, "model": model, "view": view,
        "training_view": training_view, "probability": float(probability),
        "prediction": int(probability >= 0.5), "threshold": 0.5,
    } for row, probability in zip(rows, probabilities, strict=True)]


def fold_row(fold, train, test, probabilities, *, model, experiment, view, training_view):
    labels = np.asarray([row["binary_label"] for row in test])
    result = metric_values(labels, probabilities)
    tn, fp, fn, tp = result["confusion_matrix_tn_fp_fn_tp"]
    return {
        "track": "A", "fold": fold, "experiment": experiment, "model": model,
        "view": view, "training_view": training_view, "train_count": len(train),
        "test_count": len(test), "train_spoof_count": sum(row["binary_label"] == 1 for row in train),
        "train_prn_count": sum(row["binary_label"] == 0 for row in train),
        "test_spoof_count": sum(row["binary_label"] == 1 for row in test),
        "test_prn_count": sum(row["binary_label"] == 0 for row in test),
        "train_block_count": len({row["class_block_key"] for row in train}),
        "test_block_count": len({row["class_block_key"] for row in test}),
        "scaler_fit_sample_count": len(train), "threshold": 0.5,
        "auroc": result["auroc"], "auprc": result["auprc"],
        "balanced_accuracy": result["balanced_accuracy"],
        "tpr_at_5pct_fpr": result["tpr_at_5pct_fpr"], "spoof_recall": result["spoof_recall"],
        "prn_false_positive_rate": result["prn_false_positive_rate"],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def select_predictions(predictions, *, model, experiment="actual_oof", view="actual", training_view="actual"):
    return [row for row in predictions if row["model"] == model and row["experiment"] == experiment
            and row["view"] == view and row["training_view"] == training_view]


def prediction_arrays(rows):
    ordered = sorted(rows, key=lambda row: row["sample_index"])
    return (
        np.asarray([row["binary_label"] for row in ordered], dtype=int),
        np.asarray([row["probability"] for row in ordered], dtype=float),
        np.asarray([row["class_block_key"] for row in ordered]),
        [row["sample_index"] for row in ordered],
    )


def metric_bundle(rows, seed):
    labels, probabilities, groups, _ = prediction_arrays(rows)
    return metric_values(labels, probabilities) | {
        "block_bootstrap_95ci": block_bootstrap_metrics(labels, probabilities, groups, seed=seed)
    }


def paired_result(name, left, right, seed):
    left_labels, left_probability, groups, left_indices = prediction_arrays(left)
    right_labels, right_probability, _, right_indices = prediction_arrays(right)
    if left_indices != right_indices or not np.array_equal(left_labels, right_labels):
        raise ValueError(f"unaligned paired comparison: {name}")
    return {"comparison": name} | paired_block_bootstrap(
        left_labels, left_probability, right_probability, groups, seed=seed
    )


def residual_audit_record(fold, pipeline, features, power, train_position, test_position, train, test):
    residualizer = pipeline.residualizer
    train_before, test_before = features.m2[train_position], features.m2[test_position]
    train_after = residualizer.transform(power[train_position], train_before)
    test_after = residualizer.transform(power[test_position], test_before)
    before_train = correlations(train_before, power[train_position])
    after_train = correlations(train_after, power[train_position])
    before_test = correlations(test_before, power[test_position])
    after_test = correlations(test_after, power[test_position])

    def maximum(values):
        finite = [abs(value) for value in values if value is not None and np.isfinite(value)]
        return max(finite) if finite else None

    return {
        "fold": fold, "fit_scope": "train fold only", "power_mean_db": residualizer.power_mean,
        "power_scale_db": residualizer.power_scale, "polynomial_basis": ["1", "z", "z^2", "z^3"],
        "ridge_alpha": 1.0, "ridge_fit_intercept": False,
        "coefficient_shape": list(residualizer.ridge.coef_.shape),
        "coefficient_sha256": hashlib.sha256(np.asarray(residualizer.ridge.coef_, dtype=np.float64).tobytes()).hexdigest(),
        "train_sample_index_sha256": hash_indices(np.asarray([row["sample_index"] for row in train])),
        "test_sample_index_sha256": hash_indices(np.asarray([row["sample_index"] for row in test])),
        "train_max_abs_feature_power_correlation_before": maximum(before_train),
        "train_max_abs_feature_power_correlation_after": maximum(after_train),
        "test_max_abs_feature_power_correlation_before": maximum(before_test),
        "test_max_abs_feature_power_correlation_after": maximum(after_test),
    }


def make_figures(artifact, aggregate, destruction, power):
    figures = artifact / "figures"
    figures.mkdir(exist_ok=True)
    actual = aggregate["track_a_actual"]
    plt.figure(figsize=(8, 4.5))
    plt.bar(list(MODEL_NAMES), [actual[model]["auroc"] for model in MODEL_NAMES])
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.ylim(0, 1.02); plt.ylabel("Track A OOF AUROC"); plt.tight_layout()
    plt.savefig(figures / "track_a_model_auroc.png", dpi=160); plt.close()
    plt.figure(figsize=(8, 4.5))
    plt.bar(["actual", "mismatch", "shift", "phase-rand"], [actual["M2"]["auroc"]] + [destruction[name]["retrained"]["M2"]["auroc"] for name in CONTROL_NAMES])
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.ylim(0, 1.02); plt.ylabel("M2 Track A OOF AUROC"); plt.tight_layout()
    plt.savefig(figures / "m2_destruction_auroc.png", dpi=160); plt.close()
    plt.figure(figsize=(7, 4.5))
    plt.scatter([0] * 7, list(power["Spoof"]["quantiles_db"].values()), label="Spoof quantiles")
    plt.scatter([1] * 7, list(power["Prn"]["quantiles_db"].values()), label="Prn quantiles")
    plt.xticks([0, 1], ["Spoof", "Prn"]); plt.ylabel("received power (dB)"); plt.tight_layout()
    plt.savefig(figures / "received_power_disjoint_support.png", dpi=160); plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-npy", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    raw_path, artifact = args.raw_npy.resolve(), args.artifact.resolve()
    if raw_path.stat().st_size != EXPECTED_BYTES or sha256_file(raw_path) != EXPECTED_SHA256:
        raise SystemExit("RAW_INTEGRITY_FAILURE")
    raw, schema = observe_npy_schema(raw_path)
    if not schema["schema_valid"]:
        raise SystemExit("RAW_SCHEMA_FAILURE")
    power_freeze = read_json(artifact / "power_match_freeze.json")
    if read_json(artifact / "power_match_freeze_commit.json")["commit_sha"] != POWER_MATCH_FREEZE_COMMIT:
        raise SystemExit("POWER_MATCH_FREEZE_COMMIT_MISMATCH")
    split = load_split(artifact / "split_manifest.csv")
    sample_indices = sorted({row["sample_index"] for row in split})
    data = np.asarray(raw[np.asarray(sample_indices, dtype=np.int64)]).copy(); del raw
    index_to_position = {index: position for position, index in enumerate(sample_indices)}
    metadata_by_sample = {}
    for row in split:
        metadata_by_sample.setdefault(row["sample_index"], row)
    metadata = [metadata_by_sample[index] for index in sample_indices]

    actual_power = received_power_db(data)
    actual = compute_features(data, actual_power)
    mismatch_metadata = [{"area": 1, "class_id": row["binary_label"], "transmit_power_dbm": 40} for row in metadata]
    mapping = mismatch_source_map(mismatch_metadata)
    all_positions = np.arange(len(data), dtype=np.int64)
    view_data = {
        "actual": data,
        "mismatched": mismatch_batch(data, mapping, all_positions),
        "circular_shift": circular_shift_batch(data, np.random.default_rng(SEED + 101)),
        "fourier_phase_randomized": phase_randomize_batch(data, np.random.default_rng(SEED + 202)),
        "channel_permutation": data[:, [2, 0, 3, 1], :],
        "one_channel_ablation": np.repeat(data[:, 0:1, :], 4, axis=1),
    }
    view_power = {name: received_power_db(values) for name, values in view_data.items()}
    features = {name: compute_features(values, view_power[name]) for name, values in view_data.items()}

    grid = []
    invariance_input = np.asarray(data, dtype=np.complex128)
    for gain in (0.1, 1.0, 3.7, 10.0):
        for phase in (0.0, 0.37, 1.123, 2.9):
            grid.append({"gain": gain, "phase_radians": phase, "m2": feature_error(actual.m2, compute_features(invariance_input * (gain * np.exp(1j * phase))).m2)})
    permutation_error = feature_error(actual.m2, features["channel_permutation"].m2)
    invariance = {
        "rtol": 1e-10, "atol": 1e-12, "global_gain_phase": grid,
        "fixed_channel_permutation": [2, 0, 3, 1],
        "m2_channel_permutation_error": permutation_error,
        "all_contracts_pass": all(row["m2"]["allclose_pass"] for row in grid) and permutation_error["allclose_pass"],
    }
    write_json(artifact / "invariance_results.json", invariance)
    if not invariance["all_contracts_pass"]:
        raise SystemExit("INVARIANCE_CONTRACT_FAILURE")
    write_json(artifact / "feature_contract.json", {
        "design_freeze_commit": DESIGN_FREEZE_COMMIT, "power_match_freeze_commit": POWER_MATCH_FREEZE_COMMIT,
        "M0": {"dimension": 1, "input": "received_power_db"},
        "M1": {"dimension": int(actual.m1.shape[1]), "absolute_received_power_excluded": True, "absolute_phase_excluded": True},
        "M2": {"dimension": int(actual.m2.shape[1]), "channel_identity_excluded": True, "coherence_phase_excluded": True},
        "M2R": {"dimension": int(actual.m2.shape[1]), "power_standardization": "train mean/std only", "basis": ["1", "z", "z^2", "z^3"], "ridge_alpha": 1.0, "ridge_fit_intercept": False},
        "M3": {"dimension": int(actual.m3.shape[1]), "diagnostic_only": True},
        "classifier": {"scaler": "StandardScaler train only", "penalty": "l2", "C": 1.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000, "random_state": SEED},
        "threshold": 0.5, "test_label_threshold_use": False,
    })

    predictions, fold_results, residual_audits = [], [], []
    for fold in range(5):
        current = [row for row in split if row["fold"] == fold]
        train = [row for row in current if row["role"] == "train"]
        test = [row for row in current if row["role"] == "test"]
        train_position, test_position = positions(train, index_to_position), positions(test, index_to_position)
        train_labels = np.asarray([row["binary_label"] for row in train], dtype=int)
        pipelines = {}
        for model in MODEL_NAMES:
            pipeline = fit_pipeline(model, actual, actual_power, train_position, train_labels)
            pipelines[model] = pipeline
            probability = predict_pipeline(pipeline, actual, actual_power, test_position)
            predictions.extend(prediction_rows(test, probability, model=model, experiment="actual_oof", view="actual", training_view="actual"))
            fold_results.append(fold_row(fold, train, test, probability, model=model, experiment="actual_oof", view="actual", training_view="actual"))
            if model == "M2R":
                residual_audits.append(residual_audit_record(fold, pipeline, actual, actual_power, train_position, test_position, train, test))
        for view in CONTROL_NAMES:
            for model in SPATIAL_MODELS:
                retrained = fit_pipeline(model, features[view], view_power[view], train_position, train_labels)
                probability = predict_pipeline(retrained, features[view], view_power[view], test_position)
                predictions.extend(prediction_rows(test, probability, model=model, experiment="destruction_retrained", view=view, training_view=view))
                fold_results.append(fold_row(fold, train, test, probability, model=model, experiment="destruction_retrained", view=view, training_view=view))
                probability = predict_pipeline(pipelines[model], features[view], view_power[view], test_position)
                predictions.extend(prediction_rows(test, probability, model=model, experiment="destruction_cross_apply", view=view, training_view="actual"))
                fold_results.append(fold_row(fold, train, test, probability, model=model, experiment="destruction_cross_apply", view=view, training_view="actual"))
        for view in ("channel_permutation", "one_channel_ablation"):
            for model in ("M2", "M2R", "M3"):
                probability = predict_pipeline(pipelines[model], features[view], view_power[view], test_position)
                predictions.extend(prediction_rows(test, probability, model=model, experiment="stress_cross_apply", view=view, training_view="actual"))
                fold_results.append(fold_row(fold, train, test, probability, model=model, experiment="stress_cross_apply", view=view, training_view="actual"))
    write_csv(artifact / "out_of_fold_predictions.csv", PREDICTION_FIELDS, predictions)
    write_csv(artifact / "fold_results.csv", FOLD_FIELDS, fold_results)
    write_json(artifact / "power_residualization_audit.json", {"fit_scope": "each Track-A train fold only", "folds": residual_audits, "test_refit_count": 0})

    aggregate = {"track_a_actual": {}, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_unit": "original test class_block_key"}
    for number, model in enumerate(MODEL_NAMES):
        aggregate["track_a_actual"][model] = metric_bundle(select_predictions(predictions, model=model), SEED + 1_000 + number)
    write_json(artifact / "aggregate_metrics.json", aggregate)
    paired = []
    for number, (name, left, right) in enumerate((("M2_minus_M0", "M2", "M0"), ("M2_minus_M1", "M2", "M1"), ("M2R_minus_M0", "M2R", "M0"), ("M2R_minus_M1", "M2R", "M1"), ("M3_minus_M2", "M3", "M2"))):
        paired.append(paired_result(name, select_predictions(predictions, model=left), select_predictions(predictions, model=right), SEED + 10_000 + number))
    destruction = {}
    for control_number, control in enumerate(CONTROL_NAMES):
        destruction[control] = {"retrained": {}, "actual_trained_cross_apply": {}}
        for model_number, model in enumerate(SPATIAL_MODELS):
            actual_rows = select_predictions(predictions, model=model)
            retrained_rows = select_predictions(predictions, model=model, experiment="destruction_retrained", view=control, training_view=control)
            cross_rows = select_predictions(predictions, model=model, experiment="destruction_cross_apply", view=control, training_view="actual")
            destruction[control]["retrained"][model] = metric_bundle(retrained_rows, SEED + 20_000 + control_number * 100 + model_number)
            destruction[control]["actual_trained_cross_apply"][model] = metric_bundle(cross_rows, SEED + 21_000 + control_number * 100 + model_number)
            paired.append(paired_result(f"actual_minus_{control}_retrained_{model}", actual_rows, retrained_rows, SEED + 30_000 + control_number * 100 + model_number))
            paired.append(paired_result(f"actual_minus_{control}_cross_apply_{model}", actual_rows, cross_rows, SEED + 31_000 + control_number * 100 + model_number))
    write_json(artifact / "destruction_results.json", destruction)
    write_json(artifact / "paired_bootstrap_results.json", {"comparisons": paired})
    stress = {"fixed_channel_permutation": [2, 0, 3, 1], "channel_permutation": {}, "one_channel_ablation": {}}
    for view_number, view in enumerate(("channel_permutation", "one_channel_ablation")):
        for model_number, model in enumerate(("M2", "M2R", "M3")):
            rows = select_predictions(predictions, model=model, experiment="stress_cross_apply", view=view, training_view="actual")
            stress[view][model] = metric_bundle(rows, SEED + 40_000 + view_number * 100 + model_number)
            stress[view][model]["actual_minus_stress"] = paired_result(f"actual_minus_{view}_{model}", select_predictions(predictions, model=model), rows, SEED + 41_000 + view_number * 100 + model_number)
    write_json(artifact / "channel_permutation_stress.json", stress)

    write_json(artifact / "matched_metrics.json", {
        "track_b_executed": False, "primary_caliper_db": 0.25,
        "reason": "NOT_EXECUTED: primary power-match gate failed with zero train/test pairs",
        "matched_m0": None, "matched_models": {model: None for model in MODEL_NAMES},
        "power_match_gate": power_freeze["primary_gate_conditions"],
    })
    write_json(artifact / "caliper_sensitivity.json", {
        "spatial_scoring_executed": False, "reason": "all frozen calipers have zero pairs",
        "calipers": {key: {"caliper_db": value["caliper_db"], "test_pair_count": value["total_test_pair_count"], "train_pair_count_across_folds": value["total_train_pair_count_across_folds"], "track_b_m0_oof": value["track_b_m0_oof"], "spatial_metrics": None} for key, value in power_freeze["calipers"].items()},
    })
    verdict = "SPOOF_EVALUATION_INVALID_NO_RECEIVED_POWER_OVERLAP"
    track_a = aggregate["track_a_actual"]
    power_distribution = read_json(artifact / "power_distribution.json")
    write_json(artifact / "final_verdict.json", {
        "verdict": verdict, "primary_power_match_gate_pass": False,
        "primary_matched_pair_count": 0, "received_power_range_gap_db": power_distribution["disjoint_power_range_gap_db"],
        "track_a_auroc": {model: track_a[model]["auroc"] for model in MODEL_NAMES},
        "track_a_results_diagnostic_only": True, "track_b_metrics": None,
        "matched_m0_before_auroc": track_a["M0"]["auroc"], "matched_m0_after_auroc": None,
        "track_b_executed": False, "verdict_precedence": "power-overlap gate failure is terminal for the conditional spatial question",
        "largest_provenance_limit": "No received-power common support; recording ID, transmitter position, and array calibration are unavailable.",
        "clean_versus_spoof_success": False, "general_spoof_detector_success": False,
        "recording_independent_generalization": False, "ready_for_wcl": False,
    })
    (artifact / "confound_analysis.md").write_text(
        "# Confound analysis\n\nThe exact Spoof and Prn classes have disjoint received-power support: the nearest class ranges are separated "
        f"by {power_distribution['disjoint_power_range_gap_db']:.6f} dB, and Track-A M0 achieves AUROC 1.0. No frozen caliper yields a pair, so the effect of spatial relationships conditional on received power is not identifiable.\n\n"
        "Track-A M1/M2/M2R/M3 and destruction controls are diagnostic only and cannot rescue the failed overlap gate. All 124 Spoof snapshots occupy five consecutive sample-index blocks, while official recording IDs, transmitter position, receiver orientation, and array calibration are absent.\n",
        encoding="utf-8",
    )
    write_json(artifact / "access_audit.json", {
        "mode": "ONE_EXISTING_READ_ONLY_CRPA_OBJECT", "unique_raw_object_bytes": EXPECTED_BYTES,
        "raw_integrity_full_scan_count": 4, "logical_raw_integrity_bytes_read": EXPECTED_BYTES * 4,
        "selected_snapshot_read_passes": 3, "selected_snapshot_count_per_pass": len(sample_indices),
        "logical_selected_snapshot_bytes_read": len(sample_indices) * 4 * 1024 * 8 * 3,
        "total_logical_raw_bytes_read": EXPECTED_BYTES * 4 + len(sample_indices) * 4 * 1024 * 8 * 3,
        "raw_source_modified": False, "redownloaded_bytes": 0, "copied_raw_bytes": 0,
        "texbat_bytes": 0, "oakbat_bytes": 0, "tuni_bytes": 0, "innosense_bytes": 0,
    })
    (artifact / "README.md").write_text(
        f"# Jammertest 2025 CRPA Stage-0D true-Spoof discrimination\n\nFinal verdict: `{verdict}`. The 124 Spoof and 164 Prn snapshots have a `{power_distribution['disjoint_power_range_gap_db']:.6f} dB` received-power gap. Primary 0.25 dB matching retains zero pairs, so Track B and all matched spatial scores are not executed. Track-A results and destruction controls are diagnostic only.\n",
        encoding="utf-8",
    )
    make_figures(artifact, aggregate, destruction, power_distribution)
    artifact_manifest(artifact)
    print(json.dumps({"status": "COMPLETE", "verdict": verdict, "track_b_executed": False, "track_a": {model: track_a[model]["auroc"] for model in MODEL_NAMES}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
