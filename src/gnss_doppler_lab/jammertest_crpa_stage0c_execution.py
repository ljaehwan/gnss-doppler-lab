"""Frozen Stage-0C features, classifiers, statistics, and artifact verifier."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from gnss_doppler_lab.jammertest_crpa_stage0b import PAIR_INDICES, sha256_file
from gnss_doppler_lab.jammertest_crpa_stage0c import SEED, VERDICTS


EXPECTED_BYTES = 1_398_308_992
EXPECTED_SHA256 = "d869fa20d552288002e4d2a5b6c5d1300083a6348c01a956cd6a34ff232e0a3f"
DESIGN_COMMIT = "67495be08486b5479fbf09ee1b03c9faddb2a077"
BOOTSTRAP_REPLICATES = 2_000
INVARIANCE_RTOL = 1e-10
INVARIANCE_ATOL = 1e-12

REQUIRED_ARTIFACTS = {
    "README.md", "design_freeze.json", "design_freeze_commit.json",
    "split_manifest.csv", "exclusion_manifest.csv", "data_integrity.json",
    "numerical_invariance.json", "feature_contract.json", "fold_results.csv",
    "out_of_fold_predictions.csv", "aggregate_metrics.json",
    "paired_bootstrap_results.json", "destruction_classifier_results.json",
    "channel_permutation_stress.json", "power_stratified_results.json",
    "class_family_results.json", "sensitivity_results.json",
    "confound_analysis.md", "final_verdict.json", "access_audit.json",
    "artifact_manifest_sha256.json", "test_output.txt", "verifier_output.txt",
}


@dataclass
class ModelFeatureSet:
    m0: np.ndarray
    m1: np.ndarray
    m2: np.ndarray
    m3: np.ndarray
    normalized_covariance: np.ndarray
    eigenvalue_fractions: np.ndarray
    coherences: np.ndarray
    effective_rank: np.ndarray
    condition_feature: np.ndarray


@dataclass
class FixedLogistic:
    scaler: StandardScaler
    classifier: LogisticRegression

    def predict_probability(self, values: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self.scaler.transform(values))[:, 1]


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for name in ("fold", "block_size", "sample_index", "group_key", "area", "transmit_power_dbm", "class_id"):
            row[name] = int(row[name])
        if "selected" in row:
            row["selected"] = row["selected"] == "true"
    return rows


def compute_model_features(x: np.ndarray) -> ModelFeatureSet:
    values = np.asarray(x, dtype=np.complex128)
    raw_channel_power = np.mean(np.abs(values) ** 2, axis=-1)
    log_channel_power = 10 * np.log10(np.maximum(raw_channel_power, np.finfo(float).tiny))
    m0 = np.column_stack((10 * np.log10(np.maximum(raw_channel_power.mean(axis=1), np.finfo(float).tiny)), log_channel_power))

    centered = values - values.mean(axis=-1, keepdims=True)
    centered_power = np.mean(np.abs(centered) ** 2, axis=-1)
    normalized = centered / np.sqrt(np.maximum(centered_power, np.finfo(float).tiny))[:, :, None]
    covariance = np.einsum("bct,bdt->bcd", normalized, normalized.conj(), optimize=True) / normalized.shape[-1]
    covariance = (covariance + covariance.conj().transpose(0, 2, 1)) / 2
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
    fractions = eigenvalues[:, ::-1] / np.maximum(eigenvalues.sum(axis=1, keepdims=True), np.finfo(float).tiny)
    positive = np.maximum(fractions, np.finfo(float).tiny)
    effective_rank = np.exp(-np.sum(np.where(fractions > 0, fractions * np.log(positive), 0), axis=1))
    condition_floor = np.maximum(eigenvalues[:, -1] * 1e-12, np.finfo(float).eps)
    condition = eigenvalues[:, -1] / np.maximum(eigenvalues[:, 0], condition_floor)
    condition_feature = np.log10(np.maximum(condition, 1.0))
    coherences = np.stack([covariance[:, left, right] for left, right in PAIR_INDICES], axis=1)
    coherence_magnitude = np.abs(coherences)
    sorted_magnitude = np.sort(coherence_magnitude, axis=1)[:, ::-1]
    sorted_statistics = np.column_stack((
        coherence_magnitude.mean(axis=1), coherence_magnitude.std(axis=1),
        coherence_magnitude.min(axis=1), coherence_magnitude.max(axis=1),
    ))
    m2 = np.column_stack((fractions, effective_rank, fractions[:, 0], sorted_magnitude, sorted_statistics))
    m3 = np.column_stack((m2, coherences.real, coherences.imag))

    amplitude = np.abs(values[:, 0, :])
    amplitude_scale = np.sqrt(np.maximum(raw_channel_power[:, 0], np.finfo(float).tiny))
    normalized_amplitude = amplitude / amplitude_scale[:, None]
    spectrum = np.abs(np.fft.fft(values[:, 0, :], axis=-1)) ** 2
    band_energy = spectrum.reshape((-1, 16, 64)).sum(axis=2)
    band_fraction = band_energy / np.maximum(band_energy.sum(axis=1, keepdims=True), np.finfo(float).tiny)
    centered_amp = normalized_amplitude - normalized_amplitude.mean(axis=1, keepdims=True)
    amp_std = np.maximum(normalized_amplitude.std(axis=1), np.finfo(float).tiny)
    amp_kurtosis = np.mean(centered_amp ** 4, axis=1) / amp_std ** 4
    m1 = np.column_stack((
        log_channel_power[:, 0], normalized_amplitude.mean(axis=1), amp_std,
        np.quantile(normalized_amplitude, 0.5, axis=1),
        np.quantile(normalized_amplitude, 0.9, axis=1),
        np.quantile(normalized_amplitude, 0.99, axis=1),
        amp_kurtosis,
        np.log10(np.maximum(band_fraction, np.finfo(float).tiny)),
    ))
    return ModelFeatureSet(
        m0=m0, m1=m1, m2=m2, m3=m3,
        normalized_covariance=covariance,
        eigenvalue_fractions=fractions,
        coherences=coherences,
        effective_rank=effective_rank,
        condition_feature=condition_feature,
    )


def model_matrix(features: ModelFeatureSet, model: str) -> np.ndarray:
    return getattr(features, model.lower())


def fit_fixed_logistic(values: np.ndarray, labels: np.ndarray) -> FixedLogistic:
    scaler = StandardScaler().fit(values)
    classifier = LogisticRegression(
        penalty="l2", C=1.0, solver="liblinear", max_iter=2_000,
        random_state=SEED,
    ).fit(scaler.transform(values), labels)
    return FixedLogistic(scaler=scaler, classifier=classifier)


def metric_values(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities >= threshold
    fpr, tpr, _ = roc_curve(labels, probabilities)
    allowed = tpr[fpr <= 0.05]
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "tpr_at_5pct_fpr": float(allowed.max()) if len(allowed) else 0.0,
        "confusion_matrix_tn_fp_fn_tp": [int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1])],
        "threshold": threshold,
        "count": int(len(labels)),
    }


def _bootstrap_indices(groups: np.ndarray, rng: np.random.Generator):
    unique = np.unique(groups)
    group_positions = {group: np.flatnonzero(groups == group) for group in unique}
    draw = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([group_positions[group] for group in draw])


def block_bootstrap_metrics(
    labels: np.ndarray, probabilities: np.ndarray, groups: np.ndarray,
    *, seed: int = SEED, replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    keys = ("auroc", "auprc", "balanced_accuracy", "tpr_at_5pct_fpr")
    samples = {key: [] for key in keys}
    rng = np.random.default_rng(seed)
    attempts = 0
    while len(samples["auroc"]) < replicates:
        attempts += 1
        select = _bootstrap_indices(groups, rng)
        if len(np.unique(labels[select])) < 2:
            continue
        value = metric_values(labels[select], probabilities[select])
        for key in keys:
            samples[key].append(value[key])
        if attempts > replicates * 20:
            raise ValueError("insufficient two-class bootstrap draws")
    return {
        key: {
            "estimate": metric_values(labels, probabilities)[key],
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for key, values in samples.items()
    } | {"replicates": replicates, "unit": "test group_key block", "unique_block_count": int(len(np.unique(groups)))}


def paired_block_bootstrap_auroc(
    labels: np.ndarray, left: np.ndarray, right: np.ndarray, groups: np.ndarray,
    *, seed: int = SEED, replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    rng = np.random.default_rng(seed)
    values = []
    attempts = 0
    while len(values) < replicates:
        attempts += 1
        select = _bootstrap_indices(groups, rng)
        if len(np.unique(labels[select])) < 2:
            continue
        values.append(
            roc_auc_score(labels[select], left[select])
            - roc_auc_score(labels[select], right[select])
        )
        if attempts > replicates * 20:
            raise ValueError("insufficient paired bootstrap draws")
    estimate = roc_auc_score(labels, left) - roc_auc_score(labels, right)
    return {
        "metric": "paired_auroc_difference",
        "estimate": float(estimate),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "replicates": replicates,
        "unit": "test group_key block",
        "unique_block_count": int(len(np.unique(groups))),
    }


def feature_error(before: np.ndarray, after: np.ndarray) -> dict:
    difference = np.abs(before - after)
    scale = INVARIANCE_ATOL + INVARIANCE_RTOL * np.abs(after)
    denominator = np.maximum(np.maximum(np.abs(before), np.abs(after)), np.finfo(float).tiny)
    return {
        "max_absolute_error": float(np.max(difference)),
        "max_relative_error": float(np.max(difference / denominator)),
        "max_allclose_scaled_error": float(np.max(difference / scale)),
        "allclose_pass": bool(np.allclose(before, after, rtol=INVARIANCE_RTOL, atol=INVARIANCE_ATOL)),
    }


def numerical_invariance_grid(data: np.ndarray) -> dict:
    values = np.asarray(data, dtype=np.complex128)
    before = compute_model_features(values)
    combinations = []
    for gain in (0.1, 1.0, 3.7, 10.0):
        for phase in (0.0, 0.37, 1.123, 2.9):
            after = compute_model_features(values * (gain * np.exp(1j * phase)))
            feature_results = {
                "normalized_covariance": feature_error(before.normalized_covariance, after.normalized_covariance),
                "eigenvalue_fractions": feature_error(before.eigenvalue_fractions, after.eigenvalue_fractions),
                "coherence": feature_error(before.coherences, after.coherences),
                "effective_rank": feature_error(before.effective_rank, after.effective_rank),
                "condition_derived_normalized_feature": feature_error(before.condition_feature, after.condition_feature),
            }
            combinations.append({
                "gain": gain, "phase_radians": phase, "features": feature_results,
                "all_features_pass": all(item["allclose_pass"] for item in feature_results.values()),
            })
    return {
        "comparison": "np.allclose(before, after, rtol=1e-10, atol=1e-12)",
        "rtol": INVARIANCE_RTOL, "atol": INVARIANCE_ATOL,
        "sample_count": int(len(data)), "combinations": combinations,
        "all_16_combinations_pass": all(item["all_features_pass"] for item in combinations),
    }


def artifact_manifest(artifact: Path) -> dict:
    excluded = {"artifact_manifest_sha256.json", "test_output.txt", "verifier_output.txt"}
    value = {
        "algorithm": "sha256",
        "excluded_self_and_mutable_logs": sorted(excluded),
        "files": {
            path.relative_to(artifact).as_posix(): sha256_file(path)
            for path in sorted(artifact.rglob("*"))
            if path.is_file() and path.name not in excluded
        },
    }
    write_json(artifact / "artifact_manifest_sha256.json", value)
    return value


def verify_artifact(artifact: Path) -> list[str]:
    errors = []
    missing = sorted(name for name in REQUIRED_ARTIFACTS if not (artifact / name).is_file())
    if missing:
        errors.append(f"missing required artifacts: {missing}")
    figures = artifact / "figures"
    if not figures.is_dir() or not any(figures.glob("*.png")):
        errors.append("missing figures")
    try:
        manifest = read_json(artifact / "artifact_manifest_sha256.json")
        for relative, expected in manifest["files"].items():
            path = artifact / relative
            if not path.is_file():
                errors.append(f"manifest target missing: {relative}")
            elif sha256_file(path) != expected:
                errors.append(f"manifest hash mismatch: {relative}")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"invalid manifest: {exc}")
    try:
        commit = read_json(artifact / "design_freeze_commit.json")
        if commit["commit_sha"] != DESIGN_COMMIT or commit["local_sha"] != commit["remote_sha"]:
            errors.append("design-freeze commit binding mismatch")
        design = read_json(artifact / "design_freeze.json")
        if design["status"] != "LABEL_ONLY_DESIGN_FREEZE_PRE_FEATURE" or design["iq_feature_bytes_read"] != 0:
            errors.append("design freeze was not label-only")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"invalid design binding: {exc}")
    try:
        integrity = read_json(artifact / "data_integrity.json")
        if integrity["size_bytes"] != EXPECTED_BYTES or integrity["sha256"] != EXPECTED_SHA256:
            errors.append("raw integrity binding mismatch")
        if integrity["shape"] != [42_673, 4, 1_024] or integrity["dtype"] != "complex64":
            errors.append("raw schema binding mismatch")
        invariance = read_json(artifact / "numerical_invariance.json")
        if len(invariance["combinations"]) != 16 or not invariance["all_16_combinations_pass"]:
            errors.append("numerical invariance gate failed")
        verdict = read_json(artifact / "final_verdict.json")
        if verdict["verdict"] not in VERDICTS:
            errors.append("invalid verdict")
        for field in ("clean_detector_success", "general_spoof_detector_success", "ready_for_wcl", "recording_independent_generalization"):
            if verdict[field]:
                errors.append(f"forbidden claim true: {field}")
        access = read_json(artifact / "access_audit.json")
        for field in ("redownloaded_bytes", "copied_raw_bytes", "innosense_bytes", "texbat_bytes", "oakbat_bytes", "tuni_bytes"):
            if access[field] != 0:
                errors.append(f"forbidden access nonzero: {field}")
        if access["raw_objects_opened"] != 1 or access["selected_snapshot_count"] != 3_588:
            errors.append("raw access accounting mismatch")
        contract = read_json(artifact / "feature_contract.json")
        if contract["M0"]["metadata_transmit_power_used"] or contract["threshold"] != 0.5:
            errors.append("feature or threshold contract mismatch")
        aggregate = read_json(artifact / "aggregate_metrics.json")["primary_actual"]
        baseline = max(aggregate["M0"]["auroc"], aggregate["M1"]["auroc"])
        if aggregate["M2"]["auroc"] <= baseline and verdict["verdict"] != "NO_INCREMENTAL_SPATIAL_DISCRIMINATION":
            errors.append("verdict does not match no-increment gate")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"invalid scientific artifact: {exc}")
    try:
        predictions = load_manifest(artifact / "out_of_fold_predictions.csv")
        actual = [row for row in predictions if row["view"] == "actual" and row["training_view"] == "actual"]
        keys = [(row["evaluation"], row["model"], row["sample_index"]) for row in actual]
        if len(keys) != len(set(keys)):
            errors.append("duplicate actual OOF prediction")
        if any(row["final_role"] != "test" for row in actual):
            errors.append("non-test row in OOF predictions")
        expected_counts = {"primary": 260 * 4, "sensitivity_a": 324 * 4}
        for evaluation, expected in expected_counts.items():
            if sum(row["evaluation"] == evaluation for row in actual) != expected:
                errors.append(f"incomplete actual OOF inventory: {evaluation}")
        if any(float(row["threshold"]) != 0.5 for row in predictions):
            errors.append("non-frozen prediction threshold")
        if any(int(row["prediction"]) != int(float(row["probability"]) >= 0.5) for row in predictions):
            errors.append("prediction/threshold inconsistency")
        family = read_json(artifact / "class_family_results.json")["rows"]
        spoof = [row for row in family if row["class_family"] == "Spoof"]
        if len(spoof) != 4 or any(row["count"] != 0 or row["value"] is not None for row in spoof):
            errors.append("Spoof no-support disclosure missing")
    except (OSError, KeyError, ValueError) as exc:
        errors.append(f"invalid OOF predictions: {exc}")
    return errors
