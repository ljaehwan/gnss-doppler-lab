"""Frozen features, models, statistics, and verification for Stage-0D."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from gnss_doppler_lab.jammertest_crpa_stage0b import PAIR_INDICES, sha256_file
from gnss_doppler_lab.jammertest_crpa_stage0d import MODEL_NAMES, SEED, VERDICTS
from gnss_doppler_lab.jammertest_crpa_stage0d_power import EXPECTED_BYTES, EXPECTED_SHA256


DESIGN_FREEZE_COMMIT = "33c3c6924f2a7f42ca964e1bd136239cacaea04c"
POWER_MATCH_FREEZE_COMMIT = "aa1859b411b65a2599325642b7c0d4a9abf6558c"
BOOTSTRAP_REPLICATES = 2_000
INVARIANCE_RTOL = 1e-10
INVARIANCE_ATOL = 1e-12

REQUIRED_ARTIFACTS = {
    "README.md", "design_freeze.json", "design_freeze_commit.json",
    "complete_oof_contract.json", "split_manifest.csv", "guard_manifest.csv",
    "power_distribution.json", "power_match_freeze.json",
    "power_match_freeze_commit.json", "power_match_manifest.csv",
    "data_integrity.json", "feature_contract.json", "fold_results.csv",
    "out_of_fold_predictions.csv", "aggregate_metrics.json",
    "matched_metrics.json", "caliper_sensitivity.json",
    "power_residualization_audit.json", "paired_bootstrap_results.json",
    "destruction_results.json", "channel_permutation_stress.json",
    "invariance_results.json", "confound_analysis.md", "final_verdict.json",
    "access_audit.json", "artifact_manifest_sha256.json", "test_output.txt",
    "verifier_output.txt",
}


@dataclass
class FeatureSet:
    m0: np.ndarray
    m1: np.ndarray
    m2: np.ndarray
    m3: np.ndarray
    normalized_covariance: np.ndarray
    coherences: np.ndarray


@dataclass
class PowerResidualizer:
    power_mean: float
    power_scale: float
    ridge: Ridge

    @staticmethod
    def basis(power: np.ndarray, mean: float, scale: float) -> np.ndarray:
        z = (np.asarray(power, dtype=float) - mean) / scale
        return np.column_stack((np.ones(len(z)), z, z ** 2, z ** 3))

    @classmethod
    def fit(cls, power: np.ndarray, values: np.ndarray) -> "PowerResidualizer":
        mean = float(np.mean(power))
        scale = float(np.std(power))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        ridge = Ridge(alpha=1.0, fit_intercept=False).fit(
            cls.basis(power, mean, scale), values
        )
        return cls(mean, scale, ridge)

    def transform(self, power: np.ndarray, values: np.ndarray) -> np.ndarray:
        predicted = self.ridge.predict(self.basis(power, self.power_mean, self.power_scale))
        return np.asarray(values) - predicted


@dataclass
class FixedPipeline:
    model: str
    scaler: StandardScaler
    classifier: LogisticRegression
    residualizer: PowerResidualizer | None = None


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compute_features(values: np.ndarray, power_db: np.ndarray | None = None) -> FeatureSet:
    x = np.asarray(values, dtype=np.complex128)
    raw_channel_power = np.mean(np.abs(x) ** 2, axis=-1)
    if power_db is None:
        power_db = 10 * np.log10(np.maximum(raw_channel_power.mean(axis=1), np.finfo(float).tiny))
    m0 = np.asarray(power_db, dtype=float)[:, None]

    channel_zero = x[:, 0, :]
    amplitude = np.abs(channel_zero)
    rms = np.sqrt(np.maximum(np.mean(amplitude ** 2, axis=1), np.finfo(float).tiny))
    normalized_amplitude = amplitude / rms[:, None]
    centered_amplitude = normalized_amplitude - normalized_amplitude.mean(axis=1, keepdims=True)
    amplitude_std = np.maximum(normalized_amplitude.std(axis=1), np.finfo(float).tiny)
    kurtosis = np.mean(centered_amplitude ** 4, axis=1) / amplitude_std ** 4
    spectrum = np.abs(np.fft.fft(channel_zero, axis=-1)) ** 2
    bands = spectrum.reshape((-1, 16, 64)).sum(axis=2)
    band_fraction = bands / np.maximum(bands.sum(axis=1, keepdims=True), np.finfo(float).tiny)
    m1 = np.column_stack((
        normalized_amplitude.mean(axis=1), amplitude_std,
        np.quantile(normalized_amplitude, 0.5, axis=1),
        np.quantile(normalized_amplitude, 0.9, axis=1),
        np.quantile(normalized_amplitude, 0.99, axis=1),
        kurtosis,
        np.log10(np.maximum(band_fraction, np.finfo(float).tiny)),
    ))

    centered = x - x.mean(axis=-1, keepdims=True)
    centered_power = np.mean(np.abs(centered) ** 2, axis=-1)
    normalized = centered / np.sqrt(np.maximum(centered_power, np.finfo(float).tiny))[:, :, None]
    covariance = np.einsum("bct,bdt->bcd", normalized, normalized.conj(), optimize=True) / normalized.shape[-1]
    covariance = (covariance + covariance.conj().transpose(0, 2, 1)) / 2
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
    fractions = eigenvalues[:, ::-1] / np.maximum(eigenvalues.sum(axis=1, keepdims=True), np.finfo(float).tiny)
    positive = np.maximum(fractions, np.finfo(float).tiny)
    effective_rank = np.exp(-np.sum(np.where(fractions > 0, fractions * np.log(positive), 0), axis=1))
    coherences = np.stack([covariance[:, left, right] for left, right in PAIR_INDICES], axis=1)
    magnitudes = np.abs(coherences)
    sorted_magnitudes = np.sort(magnitudes, axis=1)[:, ::-1]
    summaries = np.column_stack((magnitudes.mean(axis=1), magnitudes.std(axis=1), magnitudes.min(axis=1), magnitudes.max(axis=1)))
    m2 = np.column_stack((fractions, fractions[:, 0], effective_rank, sorted_magnitudes, summaries))
    m3 = np.column_stack((m2, coherences.real, coherences.imag))
    return FeatureSet(m0=m0, m1=m1, m2=m2, m3=m3, normalized_covariance=covariance, coherences=coherences)


def base_matrix(features: FeatureSet, model: str) -> np.ndarray:
    return features.m2 if model == "M2R" else getattr(features, model.lower())


def fit_pipeline(
    model: str,
    features: FeatureSet,
    power: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
) -> FixedPipeline:
    matrix = base_matrix(features, model)[positions]
    residualizer = None
    if model == "M2R":
        residualizer = PowerResidualizer.fit(power[positions], matrix)
        matrix = residualizer.transform(power[positions], matrix)
    scaler = StandardScaler().fit(matrix)
    classifier = LogisticRegression(
        penalty="l2", C=1.0, solver="liblinear", class_weight="balanced",
        max_iter=2_000, random_state=SEED,
    ).fit(scaler.transform(matrix), labels)
    return FixedPipeline(model, scaler, classifier, residualizer)


def predict_pipeline(
    pipeline: FixedPipeline,
    features: FeatureSet,
    power: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    matrix = base_matrix(features, pipeline.model)[positions]
    if pipeline.residualizer is not None:
        matrix = pipeline.residualizer.transform(power[positions], matrix)
    return pipeline.classifier.predict_proba(pipeline.scaler.transform(matrix))[:, 1]


def metric_values(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities >= 0.5
    fpr, tpr, _ = roc_curve(labels, probabilities)
    allowed = tpr[fpr <= 0.05]
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "count": int(len(labels)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "tpr_at_5pct_fpr": float(np.max(allowed)) if len(allowed) else 0.0,
        "confusion_matrix_tn_fp_fn_tp": [int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 0]), int(matrix[1, 1])],
        "spoof_recall": float(np.mean(predictions[labels == 1])),
        "prn_false_positive_rate": float(np.mean(predictions[labels == 0])),
        "threshold": 0.5,
    }


def bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    draw = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([positions[group] for group in draw])


def block_bootstrap_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    keys = ("auroc", "auprc", "balanced_accuracy", "tpr_at_5pct_fpr")
    samples = {key: [] for key in keys}
    rng = np.random.default_rng(seed)
    rejected = 0
    attempts = 0
    while len(samples["auroc"]) < replicates:
        attempts += 1
        selected = bootstrap_indices(groups, rng)
        if len(np.unique(labels[selected])) < 2:
            rejected += 1
            continue
        result = metric_values(labels[selected], probabilities[selected])
        for key in keys:
            samples[key].append(result[key])
        if attempts > replicates * 100:
            raise ValueError("bootstrap could not obtain two-class replicates")
    estimate = metric_values(labels, probabilities)
    return {
        "accepted_replicates": replicates,
        "single_class_rejected_replicates": rejected,
        "single_class_rejection_ratio": rejected / attempts,
        "total_draws": attempts,
        "unit": "original test class_block_key",
        "unique_block_count": int(len(np.unique(groups))),
        "metrics": {
            key: {
                "estimate": estimate[key],
                "ci95_low": float(np.quantile(samples[key], 0.025)),
                "ci95_high": float(np.quantile(samples[key], 0.975)),
            }
            for key in keys
        },
    }


def paired_block_bootstrap(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    rng = np.random.default_rng(seed)
    values = []
    rejected = 0
    attempts = 0
    while len(values) < replicates:
        attempts += 1
        selected = bootstrap_indices(groups, rng)
        if len(np.unique(labels[selected])) < 2:
            rejected += 1
            continue
        values.append(
            roc_auc_score(labels[selected], left[selected])
            - roc_auc_score(labels[selected], right[selected])
        )
        if attempts > replicates * 100:
            raise ValueError("paired bootstrap could not obtain two-class replicates")
    estimate = roc_auc_score(labels, left) - roc_auc_score(labels, right)
    return {
        "metric": "paired_auroc_difference",
        "estimate": float(estimate),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "accepted_replicates": replicates,
        "single_class_rejected_replicates": rejected,
        "single_class_rejection_ratio": rejected / attempts,
        "total_draws": attempts,
        "unit": "original test class_block_key",
    }


def feature_error(before: np.ndarray, after: np.ndarray) -> dict:
    difference = np.abs(before - after)
    denominator = np.maximum(np.maximum(np.abs(before), np.abs(after)), np.finfo(float).tiny)
    return {
        "max_absolute_error": float(np.max(difference)),
        "max_relative_error": float(np.max(difference / denominator)),
        "allclose_pass": bool(np.allclose(before, after, rtol=INVARIANCE_RTOL, atol=INVARIANCE_ATOL)),
    }


def hash_indices(indices: np.ndarray) -> str:
    payload = ",".join(str(int(value)) for value in indices).encode()
    return hashlib.sha256(payload).hexdigest()


def correlations(values: np.ndarray, power: np.ndarray) -> list[float | None]:
    result = []
    for column in values.T:
        if np.std(column) == 0 or np.std(power) == 0:
            result.append(None)
        else:
            result.append(float(np.corrcoef(column, power)[0, 1]))
    return result


def artifact_manifest(artifact: Path) -> dict:
    excluded = {"artifact_manifest_sha256.json", "test_output.txt", "verifier_output.txt"}
    result = {
        "algorithm": "sha256",
        "excluded_self_and_mutable_logs": sorted(excluded),
        "files": {
            path.relative_to(artifact).as_posix(): sha256_file(path)
            for path in sorted(artifact.rglob("*"))
            if path.is_file() and path.name not in excluded
        },
    }
    write_json(artifact / "artifact_manifest_sha256.json", result)
    return result


def verify_artifact(artifact: Path) -> list[str]:
    errors: list[str] = []
    missing = sorted(name for name in REQUIRED_ARTIFACTS if not (artifact / name).is_file())
    if missing:
        errors.append(f"missing required artifacts: {missing}")
    if not (artifact / "figures").is_dir() or not any((artifact / "figures").glob("*.png")):
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
        design_commit = read_json(artifact / "design_freeze_commit.json")
        power_commit = read_json(artifact / "power_match_freeze_commit.json")
        if design_commit["commit_sha"] != DESIGN_FREEZE_COMMIT or design_commit["local_sha"] != design_commit["remote_sha"]:
            errors.append("design freeze commit mismatch")
        if power_commit["commit_sha"] != POWER_MATCH_FREEZE_COMMIT or power_commit["local_sha"] != power_commit["remote_sha"]:
            errors.append("power freeze commit mismatch")
        oof = read_json(artifact / "complete_oof_contract.json")
        if oof["status"] != "PASS" or oof["oof_missing_count"] != 0 or oof["oof_duplicate_count"] != 0:
            errors.append("complete OOF contract failure")
        if oof["oof_test_class_counts"] != {"Prn": 164, "Spoof": 124} or oof["enumerated_unique_class_block_count"] != 11:
            errors.append("complete OOF inventory mismatch")
        integrity = read_json(artifact / "data_integrity.json")
        if integrity["size_bytes"] != EXPECTED_BYTES or integrity["sha256"] != EXPECTED_SHA256:
            errors.append("raw source binding mismatch")
        if integrity["source_shape"] != [42673, 4, 1024] or integrity["dtype"] != "complex64":
            errors.append("raw schema binding mismatch")
        power = read_json(artifact / "power_match_freeze.json")
        if power["primary_gate_pass"] or power["calipers"]["0.25"]["total_test_pair_count"] != 0:
            errors.append("power overlap gate mismatch")
        verdict = read_json(artifact / "final_verdict.json")
        if verdict["verdict"] not in VERDICTS or verdict["verdict"] != "SPOOF_EVALUATION_INVALID_NO_RECEIVED_POWER_OVERLAP":
            errors.append("final verdict mismatch")
        for field in ("clean_versus_spoof_success", "general_spoof_detector_success", "recording_independent_generalization", "ready_for_wcl"):
            if verdict[field]:
                errors.append(f"forbidden claim true: {field}")
        access = read_json(artifact / "access_audit.json")
        for field in ("redownloaded_bytes", "copied_raw_bytes", "texbat_bytes", "oakbat_bytes", "tuni_bytes", "innosense_bytes"):
            if access[field] != 0:
                errors.append(f"forbidden access nonzero: {field}")
        matched = read_json(artifact / "matched_metrics.json")
        if matched["track_b_executed"] or any(value is not None for value in matched["matched_models"].values()):
            errors.append("Track B improperly executed without power overlap")
        feature = read_json(artifact / "feature_contract.json")
        if feature["design_freeze_commit"] != DESIGN_FREEZE_COMMIT or feature["power_match_freeze_commit"] != POWER_MATCH_FREEZE_COMMIT:
            errors.append("feature execution not bound to pushed freezes")
        if feature["test_label_threshold_use"] or feature["threshold"] != 0.5:
            errors.append("feature/threshold leakage contract mismatch")
        invariance = read_json(artifact / "invariance_results.json")
        if not invariance["all_contracts_pass"]:
            errors.append("invariance contract failure")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"invalid scientific artifact: {exc}")
    try:
        predictions = load_csv(artifact / "out_of_fold_predictions.csv")
        actual = [row for row in predictions if row["experiment"] == "actual_oof" and row["track"] == "A"]
        for model in MODEL_NAMES:
            selected = [row for row in actual if row["model"] == model]
            indices = [int(row["sample_index"]) for row in selected]
            if len(indices) != 288 or len(indices) != len(set(indices)):
                errors.append(f"incomplete OOF predictions: {model}")
            classes = Counter(row["class_name"] for row in selected)
            if classes != {"Spoof": 124, "Prn": 164}:
                errors.append(f"OOF class inventory mismatch: {model}")
        if any(float(row["threshold"]) != 0.5 for row in predictions):
            errors.append("non-frozen threshold")
        if any(int(row["prediction"]) != int(float(row["probability"]) >= 0.5) for row in predictions):
            errors.append("prediction threshold inconsistency")
    except (OSError, KeyError, ValueError) as exc:
        errors.append(f"invalid predictions: {exc}")
    return errors
