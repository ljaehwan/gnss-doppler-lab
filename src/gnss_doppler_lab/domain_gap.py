"""Leakage-aware diagnostics for simulation-to-real tracking-feature gaps."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


DEFAULT_FEATURE_COLUMNS = (
    "near_sym_mean",
    "near_sym_std",
    "sharp_narrow_mean",
    "sharp_narrow_std",
    "sharp_narrow_slope",
    "doppler_std",
    "doppler_slope",
    "cn0_std",
    "code_err_abs_mean",
    "code_err_std",
    "prompt_mag_cv",
)


def select_rows(
    rows: Iterable[Mapping[str, str]],
    selectors: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Select rows matching any exact selector without duplicating overlaps."""
    selected: list[dict[str, str]] = []
    for row in rows:
        if any(all(row.get(key) == value for key, value in selector.items()) for selector in selectors):
            selected.append(dict(row))
    return selected


def finite_feature_matrix(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the numeric matrix and a mask excluding any non-finite row."""
    if not feature_columns:
        raise ValueError("feature_columns must not be empty")
    try:
        values = np.asarray(
            [[float(row[column]) for column in feature_columns] for row in rows],
            dtype=np.float64,
        )
    except KeyError as exc:
        raise ValueError(f"missing feature column: {exc.args[0]}") from exc
    if values.ndim != 2:
        values = values.reshape((len(rows), len(feature_columns)))
    mask = np.isfinite(values).all(axis=1)
    return values[mask], mask


def deterministic_group_cap(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_rows_per_group: int,
    group_columns: Sequence[str],
    order_column: str = "window_mid_s",
) -> list[dict[str, Any]]:
    """Evenly retain at most N ordered rows from each leakage group."""
    if max_rows_per_group < 1:
        raise ValueError("max_rows_per_group must be positive")
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row[column]) for column in group_columns)
        grouped[key].append(dict(row))
    retained: list[dict[str, Any]] = []
    for key in sorted(grouped):
        ordered = sorted(
            grouped[key],
            key=lambda row: (float(row[order_column]), int(float(row.get("window_index", 0)))),
        )
        if len(ordered) <= max_rows_per_group:
            retained.extend(ordered)
            continue
        indices = np.linspace(0, len(ordered) - 1, max_rows_per_group, dtype=int)
        retained.extend(ordered[index] for index in indices)
    return retained


def _real_robust_scale(values: np.ndarray) -> float:
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float((q75 - q25) / 1.349)
    floor = max(abs(float(np.median(values))) * 1e-6, 1e-12)
    return max(scale, floor)


def compare_feature_distributions(
    simulation_rows: Sequence[Mapping[str, Any]],
    real_rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
) -> tuple[list[dict[str, float | str]], dict[str, float | int]]:
    """Compute real-scale robust shifts, Wasserstein distances, and KS tests."""
    simulation, simulation_mask = finite_feature_matrix(simulation_rows, feature_columns)
    real, real_mask = finite_feature_matrix(real_rows, feature_columns)
    if not len(simulation) or not len(real):
        raise ValueError("both domains need at least one finite feature row")
    metrics: list[dict[str, float | str]] = []
    for index, feature in enumerate(feature_columns):
        sim_values = simulation[:, index]
        real_values = real[:, index]
        scale = _real_robust_scale(real_values)
        ks = ks_2samp(sim_values, real_values, alternative="two-sided", method="auto")
        metrics.append({
            "feature": feature,
            "simulation_median": float(np.median(sim_values)),
            "simulation_iqr": float(np.subtract(*np.quantile(sim_values, [0.75, 0.25]))),
            "real_median": float(np.median(real_values)),
            "real_iqr": float(np.subtract(*np.quantile(real_values, [0.75, 0.25]))),
            "real_robust_scale": scale,
            "robust_median_shift": abs(float(np.median(sim_values) - np.median(real_values))) / scale,
            "real_scaled_wasserstein": float(wasserstein_distance(sim_values / scale, real_values / scale)),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
        })
    return metrics, {
        "simulation_input_rows": len(simulation_rows),
        "simulation_finite_rows": int(simulation_mask.sum()),
        "real_input_rows": len(real_rows),
        "real_finite_rows": int(real_mask.sum()),
        "median_robust_median_shift": float(np.median([row["robust_median_shift"] for row in metrics])),
        "max_robust_median_shift": float(max(row["robust_median_shift"] for row in metrics)),
        "median_ks_statistic": float(np.median([row["ks_statistic"] for row in metrics])),
        "max_ks_statistic": float(max(row["ks_statistic"] for row in metrics)),
        "median_real_scaled_wasserstein": float(np.median([row["real_scaled_wasserstein"] for row in metrics])),
    }


def _domain_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    domain: str,
    source_column: str,
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        copied = dict(row)
        copied["_domain"] = domain
        copied["_cv_group"] = f"{domain}:{copied[source_column]}:{copied['prn']}"
        result.append(copied)
    return result


def domain_classifier_audit(
    simulation_rows: Sequence[Mapping[str, Any]],
    real_rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    *,
    simulation_source_column: str = "paired_group_id",
    real_source_column: str = "domain_source",
    max_rows_per_group: int = 128,
    n_splits: int = 5,
    random_state: int = 20260825,
) -> dict[str, Any]:
    """Estimate separability while keeping each source/PRN group in one fold."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    combined = _domain_rows(
        simulation_rows,
        domain="simulation",
        source_column=simulation_source_column,
    ) + _domain_rows(real_rows, domain="real", source_column=real_source_column)
    capped = deterministic_group_cap(
        combined,
        max_rows_per_group=max_rows_per_group,
        group_columns=("_cv_group",),
    )
    matrix, finite_mask = finite_feature_matrix(capped, feature_columns)
    valid_rows = [row for row, valid in zip(capped, finite_mask, strict=True) if valid]
    labels = np.asarray([1 if row["_domain"] == "real" else 0 for row in valid_rows], dtype=int)
    groups = np.asarray([row["_cv_group"] for row in valid_rows], dtype=object)
    if len(np.unique(labels)) != 2:
        raise ValueError("domain classifier requires both simulation and real rows")
    domain_group_counts = {
        domain: len({row["_cv_group"] for row in valid_rows if row["_domain"] == domain})
        for domain in ("simulation", "real")
    }
    if min(domain_group_counts.values()) < n_splits:
        raise ValueError("each domain needs at least n_splits independent source/PRN groups")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(matrix, labels, groups)):
        if len(np.unique(labels[test_indices])) != 2:
            raise ValueError(f"fold {fold} does not contain both domains")
        model = Pipeline([
            ("scale", RobustScaler(quantile_range=(25.0, 75.0))),
            ("classify", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                solver="liblinear",
                random_state=random_state,
            )),
        ])
        model.fit(matrix[train_indices], labels[train_indices])
        fold_probabilities = model.predict_proba(matrix[test_indices])[:, 1]
        probabilities[test_indices] = fold_probabilities
        raw_auc = float(roc_auc_score(labels[test_indices], fold_probabilities))
        fold_rows.append({
            "fold": fold,
            "auc": raw_auc,
            "separability_auc": max(raw_auc, 1.0 - raw_auc),
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "train_groups": int(len(np.unique(groups[train_indices]))),
            "test_groups": int(len(np.unique(groups[test_indices]))),
            "simulation_test_rows": int(np.sum(labels[test_indices] == 0)),
            "real_test_rows": int(np.sum(labels[test_indices] == 1)),
        })
    if not np.isfinite(probabilities).all():
        raise RuntimeError("not every domain-classifier row received an out-of-fold prediction")
    pooled_auc = float(roc_auc_score(labels, probabilities))

    final_model = Pipeline([
        ("scale", RobustScaler(quantile_range=(25.0, 75.0))),
        ("classify", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
            random_state=random_state,
        )),
    ])
    final_model.fit(matrix, labels)
    coefficients = final_model.named_steps["classify"].coef_[0]
    ranked = sorted(
        (
            {"feature": feature, "coefficient": float(value), "absolute_coefficient": abs(float(value))}
            for feature, value in zip(feature_columns, coefficients, strict=True)
        ),
        key=lambda row: row["absolute_coefficient"],
        reverse=True,
    )
    return {
        "input_rows": len(combined),
        "sampled_rows": len(capped),
        "finite_rows": len(valid_rows),
        "class_rows": {
            "simulation": int(np.sum(labels == 0)),
            "real": int(np.sum(labels == 1)),
        },
        "class_groups": domain_group_counts,
        "n_splits": n_splits,
        "max_rows_per_group": max_rows_per_group,
        "random_state": random_state,
        "pooled_auc": pooled_auc,
        "pooled_separability_auc": max(pooled_auc, 1.0 - pooled_auc),
        "mean_fold_separability_auc": float(np.mean([row["separability_auc"] for row in fold_rows])),
        "folds": fold_rows,
        "ranked_coefficients": ranked,
    }


def assign_gate_status(
    distribution_summary: Mapping[str, float],
    classifier_summary: Mapping[str, float],
    thresholds: Mapping[str, Mapping[str, float]],
) -> tuple[str, list[str]]:
    """Apply predeclared engineering screen thresholds, strictest first."""
    observed = {
        "domain_auc": float(classifier_summary["pooled_separability_auc"]),
        "median_ks": float(distribution_summary["median_ks_statistic"]),
        "median_robust_shift": float(distribution_summary["median_robust_median_shift"]),
    }
    required = ("domain_auc_max", "median_ks_max", "median_robust_shift_max")
    for level in ("pass", "conditional"):
        limits = thresholds[level]
        missing = set(required) - set(limits)
        if missing:
            raise ValueError(f"{level} gate is missing thresholds: {sorted(missing)}")
        checks = {
            "domain_auc": observed["domain_auc"] <= float(limits["domain_auc_max"]),
            "median_ks": observed["median_ks"] <= float(limits["median_ks_max"]),
            "median_robust_shift": observed["median_robust_shift"] <= float(limits["median_robust_shift_max"]),
        }
        if all(checks.values()):
            return level, []
    conditional = thresholds["conditional"]
    reasons = [
        f"domain_auc={observed['domain_auc']:.6f}>{float(conditional['domain_auc_max']):.6f}"
        if observed["domain_auc"] > float(conditional["domain_auc_max"]) else "",
        f"median_ks={observed['median_ks']:.6f}>{float(conditional['median_ks_max']):.6f}"
        if observed["median_ks"] > float(conditional["median_ks_max"]) else "",
        f"median_robust_shift={observed['median_robust_shift']:.6f}>{float(conditional['median_robust_shift_max']):.6f}"
        if observed["median_robust_shift"] > float(conditional["median_robust_shift_max"]) else "",
    ]
    return "stop", [reason for reason in reasons if reason]


def worst_gate_status(statuses: Iterable[str]) -> str:
    ranking = {"pass": 0, "conditional": 1, "stop": 2}
    values = list(statuses)
    if not values or any(value not in ranking for value in values):
        raise ValueError("statuses must be non-empty pass/conditional/stop values")
    return max(values, key=ranking.__getitem__)
