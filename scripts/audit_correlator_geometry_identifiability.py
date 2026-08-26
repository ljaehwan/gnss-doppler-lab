#!/usr/bin/env python3
"""Run the train-only matched-control correlator/geometry identifiability audit."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import plan_simulation_v4_paired_split as splitter  # noqa: E402
from gnss_doppler_lab.correlator_geometry import (  # noqa: E402
    TemplateDelayEstimator,
    build_complex_template_bank,
    build_template_bank,
    complex_profile_features,
    fit_common_geometry,
    profile_width_variance,
    random_derangement,
    two_path_complex_profile,
)
from gnss_doppler_lab.peak_mixture_law import parse_gps_sdr_sim_los_table  # noqa: E402

DEFAULT_CONFIG = Path("configs/experiments/correlator_geometry_identifiability_train_v1.json")
EXPECTED_TAPS = np.asarray([-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5])
MODALITIES = ("oracle", "magnitude", "complex")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(document), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError(f"cannot write empty frame: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False, lineterminator="\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _load_pinned_json(source: dict[str, Any], name: str) -> tuple[Path, dict[str, Any]]:
    path = _repo_path(source["path"])
    observed = _sha256(path)
    if observed != source["sha256"]:
        raise ValueError(f"{name} SHA-256 mismatch: {observed}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _inclusive_axis(low: float, high: float, step: float) -> np.ndarray:
    values = [float(low), float(high), float(step)]
    if not all(math.isfinite(value) for value in values) or step <= 0 or high < low:
        raise ValueError("template axis bounds and step are invalid")
    intervals = (high - low) / step
    if not math.isclose(intervals, round(intervals), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("template axis range is not divisible by its step")
    return np.linspace(low, high, int(round(intervals)) + 1, dtype=np.float64)


def validate_config(
    config: dict[str, Any], *, verify_source_artifacts: bool = False
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    if config.get("version") != 1:
        raise ValueError("unsupported correlator-geometry config version")
    experiment = config.get("experiment", {})
    splitter._safe_name(str(experiment.get("name", "")), "experiment.name")
    if experiment.get("role") != "exploratory train-only matched-control mechanism audit":
        raise ValueError("experiment role must remain exploratory and train-only")

    source_path, source = _load_pinned_json(config["source_artifact_summary"], "source artifact summary")
    split_path, split = _load_pinned_json(config["split_config"], "split config")
    splitter.validate_config(split)
    train_ids = [
        str(pair["paired_group_id"]) for pair in split["pairs"] if pair["split"] == "train"
    ]
    boundary = config.get("data_boundary", {})
    if boundary.get("allowed_partition") != "train" or boundary.get("allowed_pair_ids") != train_ids:
        raise ValueError("allowed pair roster differs from the frozen train split")
    if boundary.get("validation_pairs_accessed") is not False:
        raise ValueError("validation access must remain false")
    if boundary.get("test_pairs_accessed") is not False:
        raise ValueError("test access must remain false")
    if boundary.get("texbat_recordings_accessed") != []:
        raise ValueError("this mechanism audit may not access TEXBAT")
    if set(config.get("los_sources", {})) != set(train_ids):
        raise ValueError("LOS source roster differs from train")

    correlator = config.get("correlator", {})
    taps = np.asarray(correlator.get("tap_offsets_chips", []), dtype=np.float64)
    if taps.shape != EXPECTED_TAPS.shape or not np.array_equal(taps, EXPECTED_TAPS):
        raise ValueError("correlator audit requires the exact frozen nine-tap layout")
    if correlator.get("prompt_index") != 4:
        raise ValueError("the fifth tap must remain the prompt")
    if set(correlator.get("observation_modes", {})) != {"magnitude_9tap", "complex_9tap"}:
        raise ValueError("magnitude and complex observation ablations are both required")

    template = config.get("template_estimator", {})
    delay_axis = _inclusive_axis(template["delay_min_chips"], template["delay_max_chips"], template["delay_step_chips"])
    center_axis = _inclusive_axis(template["center_min_chips"], template["center_max_chips"], template["center_step_chips"])
    ratio_axis = _inclusive_axis(template["amplitude_min"], template["amplitude_max"], template["amplitude_step"])
    if int(template.get("phase_count", 0)) < 4 or template.get("normalization") != "l2":
        raise ValueError("invalid physical template estimator")
    if template.get("observation_modes") != ["magnitude_9tap", "complex_9tap"]:
        raise ValueError("template observation ladder drifted")
    if len(delay_axis) * len(center_axis) * len(ratio_axis) * int(template["phase_count"]) != 19680:
        raise ValueError("template bank must remain the frozen 19,680-profile dictionary")

    generator = config.get("generator", {})
    if int(generator.get("event_count_per_geometry", 0)) != 300:
        raise ValueError("official audit requires 300 events per geometry")
    if not (
        0 < float(generator["displacement_norm_min_chips"]) < float(generator["displacement_norm_max_chips"])
        and 0 <= float(generator["vertical_direction_abs_max"]) <= 1
        and 0 <= float(generator["authentic_center_abs_max_chips"]) <= float(template["center_max_chips"])
        and 0 < float(generator["secondary_amplitude_min"]) < float(generator["secondary_amplitude_max"])
        and float(generator["maximum_absolute_delay_chips"]) <= float(template["delay_max_chips"])
    ):
        raise ValueError("invalid controlled two-path generator")
    maximum_possible = float(generator["displacement_norm_max_chips"]) + float(generator["clock_bias_abs_max_chips"])
    if maximum_possible > float(generator["maximum_absolute_delay_chips"]):
        raise ValueError("generator can exceed the configured delay support")

    evaluation = config.get("evaluation", {})
    if int(evaluation.get("minimum_prns", 0)) < 8 or int(evaluation.get("bootstrap_repetitions", 0)) < 100:
        raise ValueError("evaluation support is too small")
    support = config.get("exploratory_support_rule", {})
    expected_support_keys = {
        "single_prn_profile_multiset_exact_match",
        "single_prn_width_auc_absolute_distance_from_half_max",
        "oracle_geometry_auc_min",
        "complex_geometry_auc_min",
        "each_geometry_complex_auc_min",
        "complex_delay_mae_max_chips",
        "complex_delay_sign_accuracy_min",
        "magnitude_only_is_reported_ablation_without_support_gate",
        "requires_validation_confirmation",
    }
    if set(support) != expected_support_keys or not all(
        support.get(key) is True
        for key in (
            "single_prn_profile_multiset_exact_match",
            "magnitude_only_is_reported_ablation_without_support_gate",
            "requires_validation_confirmation",
        )
    ):
        raise ValueError("exploratory support rule drifted")
    claim = config.get("claim_boundary", {})
    if any(claim.get(key) is not False for key in (
        "actual_multipath_rf_generated", "actual_receiver_tracking_evaluated", "actual_receiver_complex_taps_evaluated"
    )):
        raise ValueError("claim boundary cannot assert unperformed RF or receiver evaluation")

    if source.get("campaign", {}).get("partition") != "train" or set(source.get("artifacts", {})) != set(train_ids):
        raise ValueError("source artifact summary is not the frozen train roster")
    source_boundary = source.get("data_boundary", {})
    if source_boundary.get("validation_pairs_accessed") is not False or source_boundary.get("test_pairs_accessed") is not False:
        raise ValueError("source artifact summary crossed the train boundary")
    if verify_source_artifacts:
        for pair_id in train_ids:
            source_pair = source["artifacts"][pair_id]
            manifest_path = Path(source_pair["pair_manifest"]).resolve()
            if _sha256(manifest_path) != source_pair["pair_manifest_sha256"]:
                raise ValueError(f"pair manifest integrity failure: {pair_id}")
            los_source = config["los_sources"][pair_id]
            los_path = _repo_path(los_source["path"])
            if _sha256(los_path) != los_source["sha256"]:
                raise ValueError(f"LOS log integrity failure: {pair_id}")
    return source_path, source, split_path, split


def _template_axes(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    template = config["template_estimator"]
    return (
        _inclusive_axis(template["delay_min_chips"], template["delay_max_chips"], template["delay_step_chips"]),
        _inclusive_axis(template["center_min_chips"], template["center_max_chips"], template["center_step_chips"]),
        _inclusive_axis(template["amplitude_min"], template["amplitude_max"], template["amplitude_step"]),
        np.linspace(-np.pi, np.pi, int(template["phase_count"]), endpoint=False, dtype=np.float64),
    )


def _load_los(config: dict[str, Any]) -> tuple[dict[str, tuple[list[str], np.ndarray]], dict[str, Any]]:
    result: dict[str, tuple[list[str], np.ndarray]] = {}
    provenance: dict[str, Any] = {}
    minimum = int(config["evaluation"]["minimum_prns"])
    for pair_id in config["data_boundary"]["allowed_pair_ids"]:
        source = config["los_sources"][pair_id]
        path = _repo_path(source["path"])
        observed_sha = _sha256(path)
        if observed_sha != source["sha256"]:
            raise ValueError(f"LOS log integrity failure: {pair_id}")
        table = parse_gps_sdr_sim_los_table(path.read_text(encoding="utf-8"))
        prns = sorted(table)
        los = np.asarray([table[prn] for prn in prns], dtype=np.float64)
        design_rank = int(np.linalg.matrix_rank(np.column_stack((-los, np.ones(len(los))))))
        if len(prns) < minimum or design_rank != 4:
            raise ValueError(f"insufficient LOS geometry: {pair_id}")
        result[pair_id] = (prns, los)
        provenance[pair_id] = {
            "path": str(path), "sha256": observed_sha, "prns": prns,
            "prn_count": len(prns), "design_rank": design_rank,
        }
    return result, provenance


def _exact_row_multiset_match(left: np.ndarray, right: np.ndarray) -> bool:
    first = np.ascontiguousarray(left)
    second = np.ascontiguousarray(right)
    if first.shape != second.shape or first.dtype != second.dtype:
        return False
    return Counter(row.tobytes() for row in first) == Counter(row.tobytes() for row in second)


def _row_fingerprint(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()


def _auc(labels: np.ndarray | pd.Series, scores: np.ndarray | pd.Series) -> float:
    label_values = np.asarray(labels, dtype=np.int64)
    score_values = np.asarray(scores, dtype=np.float64)
    if set(np.unique(label_values)) != {0, 1} or not np.isfinite(score_values).all():
        raise ValueError("AUC requires finite scores from both classes")
    return float(roc_auc_score(label_values, score_values))


def _bootstrap_auc(
    events: pd.DataFrame, score_column: str, repetitions: int, rng: np.random.Generator
) -> dict[str, float]:
    pivot = events.pivot(index="event_key", columns="label", values=score_column)
    if set(pivot.columns) != {0, 1} or pivot.isna().any().any():
        raise ValueError("paired event bootstrap requires one row per class")
    negative = pivot[0].to_numpy(dtype=np.float64)
    positive = pivot[1].to_numpy(dtype=np.float64)
    estimates = np.empty(repetitions, dtype=np.float64)
    labels = np.concatenate((np.ones(len(pivot), dtype=np.int64), np.zeros(len(pivot), dtype=np.int64)))
    for index in range(repetitions):
        sample = rng.integers(0, len(pivot), size=len(pivot))
        scores = np.concatenate((positive[sample], negative[sample]))
        estimates[index] = roc_auc_score(labels, scores)
    return {
        "estimate": _auc(events["label"], events[score_column]),
        "bootstrap_ci95_low": float(np.quantile(estimates, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_repetitions": int(repetitions),
    }


def _paired_bootstrap_auc_delta(
    events: pd.DataFrame, candidate_column: str, baseline_column: str,
    repetitions: int, rng: np.random.Generator,
) -> dict[str, float]:
    candidate = events.pivot(index="event_key", columns="label", values=candidate_column)
    baseline = events.pivot(index="event_key", columns="label", values=baseline_column)
    if not candidate.index.equals(baseline.index) or set(candidate.columns) != {0, 1}:
        raise ValueError("paired AUC comparison requires aligned event pairs")
    labels = np.concatenate((np.ones(len(candidate), dtype=np.int64), np.zeros(len(candidate), dtype=np.int64)))
    differences = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = rng.integers(0, len(candidate), size=len(candidate))
        candidate_scores = np.concatenate((candidate[1].to_numpy()[sample], candidate[0].to_numpy()[sample]))
        baseline_scores = np.concatenate((baseline[1].to_numpy()[sample], baseline[0].to_numpy()[sample]))
        differences[index] = roc_auc_score(labels, candidate_scores) - roc_auc_score(labels, baseline_scores)
    estimate = _auc(events["label"], events[candidate_column]) - _auc(events["label"], events[baseline_column])
    return {
        "estimate": float(estimate),
        "paired_bootstrap_ci95_low": float(np.quantile(differences, 0.025)),
        "paired_bootstrap_ci95_high": float(np.quantile(differences, 0.975)),
        "bootstrap_repetitions": int(repetitions),
    }


def _delay_diagnostic(delays: pd.DataFrame, estimate_column: str, threshold: float) -> dict[str, Any]:
    truth = delays["truth_delay_chips"].to_numpy(dtype=np.float64)
    estimate = delays[estimate_column].to_numpy(dtype=np.float64)
    eligible = np.abs(truth) >= threshold
    return {
        "mae_chips": float(np.mean(np.abs(truth - estimate))),
        "median_absolute_error_chips": float(np.median(np.abs(truth - estimate))),
        "sign_accuracy": float(np.mean(np.sign(truth[eligible]) == np.sign(estimate[eligible]))),
        "sign_eligible_count": int(eligible.sum()),
        "row_count": int(len(truth)),
    }


def _simulate(
    config: dict[str, Any], los_sets: dict[str, tuple[list[str], np.ndarray]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    taps = np.asarray(config["correlator"]["tap_offsets_chips"], dtype=np.float64)
    prompt_index = int(config["correlator"]["prompt_index"])
    delay_axis, center_axis, ratio_axis, phase_axis = _template_axes(config)
    magnitude_bank = build_template_bank(
        taps, delays_chips=delay_axis, centers_chips=center_axis,
        amplitude_ratios=ratio_axis, phases_rad=phase_axis,
    )
    complex_bank = build_complex_template_bank(
        taps, prompt_index=prompt_index, delays_chips=delay_axis,
        centers_chips=center_axis, amplitude_ratios=ratio_axis, phases_rad=phase_axis,
    )
    magnitude_estimator = TemplateDelayEstimator(magnitude_bank)
    complex_estimator = TemplateDelayEstimator(complex_bank)
    generator = config["generator"]
    rng = np.random.default_rng(int(generator["seed"]))
    event_rows: list[dict[str, Any]] = []
    delay_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    exact_matches = 0

    for pair_id, (prns, los) in los_sets.items():
        design = np.column_stack((-los, np.ones(len(los), dtype=np.float64)))
        for event_index in range(int(generator["event_count_per_geometry"])):
            azimuth = rng.uniform(-np.pi, np.pi)
            vertical = rng.uniform(-float(generator["vertical_direction_abs_max"]), float(generator["vertical_direction_abs_max"]))
            horizontal = math.sqrt(1.0 - vertical**2)
            direction = np.asarray([horizontal * math.cos(azimuth), horizontal * math.sin(azimuth), vertical])
            displacement_norm = rng.uniform(float(generator["displacement_norm_min_chips"]), float(generator["displacement_norm_max_chips"]))
            displacement = displacement_norm * direction
            clock_bias = rng.uniform(-float(generator["clock_bias_abs_max_chips"]), float(generator["clock_bias_abs_max_chips"]))
            theta = np.concatenate((displacement, [clock_bias]))
            truth_delays = design @ theta
            if np.max(np.abs(truth_delays)) > float(generator["maximum_absolute_delay_chips"]) + 1e-12:
                raise ValueError("generated delay exceeded the frozen support")

            centers = np.clip(
                rng.normal(0.0, float(generator["authentic_center_std_chips"]), len(los)),
                -float(generator["authentic_center_abs_max_chips"]),
                float(generator["authentic_center_abs_max_chips"]),
            )
            ratios = rng.uniform(float(generator["secondary_amplitude_min"]), float(generator["secondary_amplitude_max"]), len(los))
            phases = rng.uniform(float(generator["relative_phase_min_rad"]), float(generator["relative_phase_max_rad"]), len(los))
            noise_std = rng.uniform(float(generator["complex_noise_std_min"]), float(generator["complex_noise_std_max"]), len(los))
            complex_profiles = np.stack([
                two_path_complex_profile(
                    taps, authentic_center_chips=centers[index],
                    secondary_delay_chips=truth_delays[index],
                    secondary_amplitude_ratio=ratios[index], relative_phase_rad=phases[index],
                    complex_noise=noise_std[index] * (rng.normal(size=len(taps)) + 1j * rng.normal(size=len(taps))),
                )
                for index in range(len(los))
            ])
            magnitudes = np.abs(complex_profiles)
            magnitude_delays, magnitude_distance, _ = magnitude_estimator.estimate(magnitudes)
            complex_features = complex_profile_features(complex_profiles, prompt_index=prompt_index)
            complex_delays, complex_distance, _ = complex_estimator.estimate(complex_features)
            widths = profile_width_variance(magnitudes, taps)
            permutation = random_derangement(len(los), rng)
            matched_profiles = complex_profiles[permutation]
            exact = _exact_row_multiset_match(complex_profiles, matched_profiles)
            exact_matches += int(exact)
            if not exact:
                raise AssertionError("matched control profile multiset is not exact")
            event_key = f"{pair_id}-e{event_index:04d}"

            assignments = (
                (1, "coherent_spoof", np.arange(len(los))),
                (0, "independent_multipath_control", permutation),
            )
            for label, event_class, source_indices in assignments:
                oracle_fit = fit_common_geometry(los, truth_delays[source_indices])
                magnitude_fit = fit_common_geometry(los, magnitude_delays[source_indices])
                complex_fit = fit_common_geometry(los, complex_delays[source_indices])
                event_rows.append({
                    "event_key": event_key, "pair_id": pair_id, "event_index": event_index,
                    "event_class": event_class, "label": label, "prn_count": len(los),
                    "profile_multiset_exact_match": exact,
                    "oracle_geometry_residual": oracle_fit.normalized_residual,
                    "magnitude_geometry_residual": magnitude_fit.normalized_residual,
                    "complex_geometry_residual": complex_fit.normalized_residual,
                    "oracle_score": -oracle_fit.normalized_residual,
                    "magnitude_score": -magnitude_fit.normalized_residual,
                    "complex_score": -complex_fit.normalized_residual,
                    "oracle_fit_rank": oracle_fit.rank,
                    "magnitude_fit_rank": magnitude_fit.rank,
                    "complex_fit_rank": complex_fit.rank,
                    "truth_displacement_e_chips": displacement[0],
                    "truth_displacement_n_chips": displacement[1],
                    "truth_displacement_u_chips": displacement[2],
                    "truth_clock_bias_chips": clock_bias,
                })
                for assigned_index, source_index in enumerate(source_indices):
                    assignment_rows.append({
                        "event_key": event_key, "pair_id": pair_id, "event_index": event_index,
                        "event_class": event_class, "label": label,
                        "assigned_prn": prns[assigned_index], "source_prn": prns[source_index],
                        "profile_sha256": _row_fingerprint(complex_profiles[source_index]),
                        "profile_width_chips_squared": widths[source_index],
                    })

            for index, prn in enumerate(prns):
                row = {
                    "event_key": event_key, "pair_id": pair_id, "event_index": event_index,
                    "prn": prn, "los_e": los[index, 0], "los_n": los[index, 1], "los_u": los[index, 2],
                    "truth_delay_chips": truth_delays[index],
                    "magnitude_estimate_chips": magnitude_delays[index],
                    "complex_estimate_chips": complex_delays[index],
                    "magnitude_template_distance": magnitude_distance[index],
                    "complex_template_distance": complex_distance[index],
                    "authentic_center_chips": centers[index], "secondary_amplitude_ratio": ratios[index],
                    "relative_phase_rad": phases[index], "complex_noise_std": noise_std[index],
                }
                for tap_index, offset in enumerate(taps):
                    tag = f"m{abs(int(round(offset * 1000))):03d}" if offset < 0 else f"p{int(round(offset * 1000)):03d}"
                    row[f"magnitude_tap_{tag}"] = magnitudes[index, tap_index]
                delay_rows.append(row)

    events = pd.DataFrame(event_rows)
    delays = pd.DataFrame(delay_rows)
    assignments = pd.DataFrame(assignment_rows)
    template_manifest = {
        "tap_offsets_chips": taps.tolist(), "prompt_index": prompt_index,
        "delay_axis_chips": delay_axis.tolist(), "center_axis_chips": center_axis.tolist(),
        "amplitude_axis": ratio_axis.tolist(), "phase_axis_rad": phase_axis.tolist(),
        "magnitude_profile_count": int(len(magnitude_bank.profiles)),
        "complex_profile_count": int(len(complex_bank.profiles)),
        "complex_feature_dimension": int(complex_bank.profiles.shape[1]),
        "matched_event_count": int(exact_matches),
    }
    return events, delays, assignments, template_manifest


def _summarize(
    config: dict[str, Any], events: pd.DataFrame, delays: pd.DataFrame,
    assignments: pd.DataFrame, template_manifest: dict[str, Any]
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    repetitions = int(evaluation["bootstrap_repetitions"])
    bootstrap_rng = np.random.default_rng(int(evaluation["bootstrap_seed"]))
    aucs = {
        modality: _bootstrap_auc(events, f"{modality}_score", repetitions, bootstrap_rng)
        for modality in MODALITIES
    }
    complex_gain = _paired_bootstrap_auc_delta(
        events, "complex_score", "magnitude_score", repetitions,
        np.random.default_rng(int(evaluation["bootstrap_seed"]) + 1),
    )
    width_auc = _auc(assignments["label"], assignments["profile_width_chips_squared"])
    per_geometry: dict[str, Any] = {}
    for pair_id, frame in events.groupby("pair_id", sort=True):
        per_geometry[pair_id] = {
            f"{modality}_geometry_auc": _auc(frame["label"], frame[f"{modality}_score"])
            for modality in MODALITIES
        }
        per_geometry[pair_id]["prn_count"] = int(frame["prn_count"].iloc[0])
    threshold = float(evaluation["delay_sign_accuracy_min_abs_truth_chips"])
    delay_diagnostics = {
        "magnitude": _delay_diagnostic(delays, "magnitude_estimate_chips", threshold),
        "complex": _delay_diagnostic(delays, "complex_estimate_chips", threshold),
    }
    residuals: dict[str, Any] = {}
    for modality in MODALITIES:
        residuals[modality] = {
            event_class: float(frame[f"{modality}_geometry_residual"].median())
            for event_class, frame in events.groupby("event_class", sort=True)
        }
    support = config["exploratory_support_rule"]
    gates = {
        "single_prn_profile_multiset_exact_match": bool(events["profile_multiset_exact_match"].all()),
        "single_prn_width_auc_is_half": abs(width_auc - 0.5) <= float(support["single_prn_width_auc_absolute_distance_from_half_max"]),
        "oracle_geometry_auc": aucs["oracle"]["estimate"] >= float(support["oracle_geometry_auc_min"]),
        "complex_geometry_auc": aucs["complex"]["estimate"] >= float(support["complex_geometry_auc_min"]),
        "each_geometry_complex_auc": min(item["complex_geometry_auc"] for item in per_geometry.values()) >= float(support["each_geometry_complex_auc_min"]),
        "complex_delay_mae": delay_diagnostics["complex"]["mae_chips"] <= float(support["complex_delay_mae_max_chips"]),
        "complex_delay_sign_accuracy": delay_diagnostics["complex"]["sign_accuracy"] >= float(support["complex_delay_sign_accuracy_min"]),
    }
    passed = all(gates.values())
    return {
        "event_pair_count": int(events["event_key"].nunique()),
        "classified_event_row_count": int(len(events)),
        "delay_row_count": int(len(delays)),
        "profile_assignment_row_count": int(len(assignments)),
        "single_prn_width_auc": width_auc,
        "geometry_auc": aucs,
        "complex_minus_magnitude_auc": complex_gain,
        "per_geometry": per_geometry,
        "delay_diagnostics": delay_diagnostics,
        "median_geometry_residual": residuals,
        "support_gates": gates,
        "all_exploratory_support_gates_passed": passed,
        "exploratory_status": "supported_on_train_requires_validation" if passed else "not_supported_on_train",
        "template_manifest": template_manifest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    started = time.time()
    config_path = _repo_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_path, source, split_path, split = validate_config(config, verify_source_artifacts=True)
    output_root = _repo_path(config["output_root"])
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    los_sets, los_provenance = _load_los(config)
    events, delays, assignments, template_manifest = _simulate(config, los_sets)
    diagnostic = _summarize(config, events, delays, assignments, template_manifest)

    event_path = output_root / "event_scores.csv"
    delay_path = output_root / "delay_estimates.csv"
    assignment_path = output_root / "profile_assignments.csv"
    template_path = output_root / "template_manifest.json"
    _atomic_frame(event_path, events)
    _atomic_frame(delay_path, delays)
    _atomic_frame(assignment_path, assignments)
    _atomic_json(template_path, template_manifest)
    summary = {
        "schema": "gnss-doppler-lab.correlator-geometry-identifiability-audit",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config["experiment"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
        "physics_module": {
            "path": str((REPO_ROOT / "src/gnss_doppler_lab/correlator_geometry.py").resolve()),
            "sha256": _sha256(REPO_ROOT / "src/gnss_doppler_lab/correlator_geometry.py"),
        },
        "source_artifact_summary": {"path": str(source_path), "sha256": _sha256(source_path)},
        "split_config": {"path": str(split_path), "sha256": _sha256(split_path)},
        "data_boundary": {
            **config["data_boundary"], "observed_pair_ids": list(los_sets),
            "validation_pair_ids_accessed": [], "test_pair_ids_accessed": [],
        },
        "los_provenance": los_provenance,
        "physical_hypothesis": {
            "law": "delta_j = -u_j^T (Delta r / L_chip) + clock_bias",
            "spoof_model_dimension": 4,
            "matched_control": config["correlator"]["single_prn_control"],
            "identifiability_argument": "single-PRN empirical distributions are exactly matched; only the profile-to-LOS association differs",
        },
        "diagnostic": diagnostic,
        "candidate_status": diagnostic["exploratory_status"],
        "claim_boundary": [
            "This is an ideal triangular-autocorrelation two-path mechanism audit, not RF multipath generation.",
            "The multipath control is an independent PRN derangement, not a ray-traced environment.",
            "The complex-tap candidate was not evaluated through the current receiver, which stores magnitudes for extra taps.",
            "No validation pair, test pair, or TEXBAT recording was accessed.",
            "No deployable threshold, field false-alarm rate, or WCL-level generalization claim follows from this train-only audit.",
        ],
        "next_gate": "add complex I/Q export for all nine correlator taps, freeze the estimator, then run RF multipath and untouched validation pairs 007-009",
        "outputs": {
            "event_scores": {"path": str(event_path), "sha256": _sha256(event_path)},
            "delay_estimates": {"path": str(delay_path), "sha256": _sha256(delay_path)},
            "profile_assignments": {"path": str(assignment_path), "sha256": _sha256(assignment_path)},
            "template_manifest": {"path": str(template_path), "sha256": _sha256(template_path)},
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    summary_path = output_root / "summary.json"
    _atomic_json(summary_path, summary)
    print(json.dumps(_jsonable({
        "summary": str(summary_path), "candidate_status": summary["candidate_status"],
        "diagnostic": diagnostic, "data_boundary": summary["data_boundary"],
    }), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
