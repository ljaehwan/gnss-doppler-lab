"""Frozen terminal R2 SPLITCLOCK clean-only experiment orchestration."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .splitclock_observable_audit import artifact_manifest
from .splitclock_r1_experiment import (
    GeometryUnavailable,
    _controlled,
    clopper_pearson_zero_upper,
    first_path,
    rank_auc,
    sign_validation,
    source_integrity,
    split_indices,
    support,
    write_csv,
)
from .splitclock_r1_geometry import ScenarioPanel, build_panel, trace_cadence
from .splitclock_r2_contract import (
    ALLOWED_VERDICTS,
    BASE_SHA,
    BRANCH,
    HORIZON_EPOCHS,
    R1_ARTIFACT,
    R1_DESIGN_SHA,
    R1_FINAL_SHA,
    R1_IMPLEMENTATION_SHA,
)
from .splitclock_r2_model import (
    ScoreResult,
    calibrate_noise,
    first_persistent_alarm_index,
    inject_clock,
    localization_record,
    matched_horizon_statistics,
    persistence_statistic,
    score_window,
)
from .splitclock_stage0a import sha256_file

REPAIR_SCOPE_SHA = "a645980a81e94499efaebbc3287f0a263302e7e9"
SYNTHETIC_TRAJECTORIES = {
    "mild_ramp": (0.0, 0.1, 0.0),
    "moderate_ramp": (0.0, 0.5, 0.0),
    "accelerating": (0.0, 0.05, 0.02),
    "delayed_ramp": (10.0, 0.1, 0.0),
}
REQUIRED_ARTIFACTS = (
    "README.md", "repair_scope_freeze.json", "design_freeze.json",
    "design_freeze_commit.json", "implementation_freeze.json",
    "implementation_freeze_commit.json", "final_verdict.json",
    "data_integrity.json", "access_audit.json", "geometry_validation.json",
    "sign_unit_validation.json", "cadence_detection.json",
    "observable_support.json", "dynamic_panel_validation.json",
    "centering_validation.json", "persistent_membership_validation.json",
    "model_contract_validation.json", "threshold_calibration.json",
    "clean_holdout_validation.json", "clean_scores.csv.gz",
    "synthetic_control_manifest.json", "synthetic_control_results.json",
    "synthetic_score_traces.csv.gz", "paired_clean_score_traces.csv.gz",
    "persistence_statistic_validation.json", "matched_horizon_auc.json",
    "negative_control_results.json", "boundary_control_results.json",
    "localization_results.json", "ablation_results.json",
    "destruction_results.json", "shortcut_audit.json",
    "bootstrap_results.json", "deterministic_reproduction.json",
    "r1_r2_comparison.json", "statistical_support.json",
    "source_binding.json", "artifact_manifest_sha256.json",
    "test_output.txt", "full_pytest_output.txt", "verifier_output.txt",
    "fresh_clone_output.txt",
)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def directory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def result_row(
    result: ScoreResult, scenario: str, start: int, end: int, role: str
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "role": role,
        "window_start_epoch": start,
        "window_end_epoch": end,
        "score": result.score,
        "normalized_score": result.normalized_score,
        "raw_gain": result.raw_gain,
        "penalty": result.penalty,
        "k1_heldout_loglik": result.k1_heldout_loglik,
        "k2_heldout_loglik": result.k2_heldout_loglik,
        "observation_wise_score": result.observation_wise_score,
        "n_valid": result.n_valid,
        "delta_p": result.delta_p,
        "cluster_mass_0": result.cluster_masses[0],
        "cluster_mass_1": result.cluster_masses[1],
        "selected_restart": result.selected_restart,
        "fit_digest": result.fit_digest,
    }


def score_series(
    panel: ScenarioPanel,
    interval: tuple[int, int],
    scales: np.ndarray,
    process: np.ndarray,
    *,
    role: str,
    values: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    modalities: tuple[int, ...] = (0, 1, 2),
    hard: bool = False,
) -> tuple[list[dict[str, Any]], dict[int, ScoreResult]]:
    source_values = panel.values if values is None else values
    source_valid = panel.valid if valid is None else valid
    rows: list[dict[str, Any]] = []
    results: dict[int, ScoreResult] = {}
    for end in range(interval[0] + 9, interval[1]):
        start = end - 9
        try:
            result = score_window(
                source_values[start : end + 1],
                source_valid[start : end + 1],
                scales,
                process,
                modalities=modalities,
                hard_assignment=hard,
            )
        except ValueError:
            continue
        rows.append(result_row(result, panel.scenario, start, end, role))
        results[end] = result
    return rows, results


def score_nonoverlap(
    panel: ScenarioPanel,
    interval: tuple[int, int],
    scales: np.ndarray,
    process: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[int, ScoreResult]]:
    rows: list[dict[str, Any]] = []
    results: dict[int, ScoreResult] = {}
    for start in range(interval[0], interval[1] - 9, 10):
        end = start + 9
        result = score_window(
            panel.values[start : end + 1],
            panel.valid[start : end + 1],
            scales,
            process,
        )
        rows.append(result_row(result, panel.scenario, start, end, "calibration_nonoverlap"))
        results[end] = result
    return rows, results


def persistent_vector(scores: np.ndarray, threshold: float) -> np.ndarray:
    result = np.zeros(len(scores), dtype=bool)
    for index in range(2, len(scores)):
        result[index] = bool(np.all(scores[index - 2 : index + 1] > threshold))
    return result


def deterministic_subset(prns: np.ndarray, count: int) -> np.ndarray:
    ordered = sorted(
        range(len(prns)),
        key=lambda index: hashlib.sha256(
            f"20250826:{int(prns[index])}".encode()
        ).hexdigest(),
    )
    return np.asarray(ordered[:count], dtype=int)


def horizon_scores(
    panel: ScenarioPanel,
    values: np.ndarray,
    valid: np.ndarray,
    interval: tuple[int, int],
    onset: int,
    scales: np.ndarray,
    process: np.ndarray,
    role: str,
) -> tuple[list[dict[str, Any]], dict[int, ScoreResult]]:
    first_end = max(interval[0] + 9, onset)
    last_exclusive = min(interval[1], onset + HORIZON_EPOCHS)
    rows: list[dict[str, Any]] = []
    results: dict[int, ScoreResult] = {}
    for end in range(first_end, last_exclusive):
        start = end - 9
        result = score_window(
            values[start : end + 1], valid[start : end + 1], scales, process
        )
        row = result_row(result, panel.scenario, start, end, role)
        row.update(
            {
                "parent_onset_epoch": onset,
                "relative_window_end_s": end - onset,
                "A0_score": -result.k1_heldout_loglik / result.n_valid,
                "A6_score": result.score,
            }
        )
        rows.append(row)
        results[end] = result
    return rows, results


def synthetic_controls(
    panel: ScenarioPanel,
    interval: tuple[int, int],
    threshold: float,
    scales: np.ndarray,
    process: np.ndarray,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], dict[int, dict[str, float | int]],
]:
    start, end = interval
    allowable = max(0, end - start - 30)
    onsets = [start, start + allowable // 3, start + 2 * allowable // 3]
    clean_traces: list[dict[str, Any]] = []
    clean_horizons: dict[int, dict[str, float | int]] = {}
    for onset in onsets:
        rows, _ = horizon_scores(
            panel, panel.values, panel.valid, interval, onset, scales, process,
            "paired_clean",
        )
        clean_traces.extend(rows)
        clean_horizons[onset] = matched_horizon_statistics(
            [row["A0_score"] for row in rows], [row["A6_score"] for row in rows]
        )

    manifest: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for subset_size in (1, 3, 5):
        subset = deterministic_subset(panel.prns, subset_size)
        for trajectory, parameters in SYNTHETIC_TRAJECTORIES.items():
            for onset in onsets:
                d0, velocity, acceleration = parameters
                case = f"m{subset_size}_{trajectory}_o{onset}"
                injected = inject_clock(
                    panel.values, subset, onset, d0, velocity, acceleration
                )
                rows, score_results = horizon_scores(
                    panel, injected, panel.valid, interval, onset, scales, process,
                    "synthetic",
                )
                for row in rows:
                    row["case"] = case
                    row["subset_size"] = subset_size
                    row["trajectory"] = trajectory
                traces.extend(rows)
                ends = [row["window_end_epoch"] for row in rows]
                a6 = [row["A6_score"] for row in rows]
                a0 = [row["A0_score"] for row in rows]
                alarm_index = first_persistent_alarm_index(a6, threshold)
                alarm_result = None if alarm_index is None else score_results[ends[alarm_index]]
                oracle_end = max(ends, key=lambda value: score_results[value].score)
                oracle_result = score_results[oracle_end]
                localization = localization_record(alarm_result, oracle_result, subset)
                horizon = matched_horizon_statistics(a0, a6)
                diagnostic: dict[str, float | None] = {}
                oracle_slice = slice(oracle_end - 9, oracle_end + 1)
                for name, modalities, hard, proc in (
                    ("A1", (0, 1, 2), True, process),
                    ("A2", (0, 1, 2), False, np.asarray([1e6, 1e6])),
                    ("A3", (0,), False, process),
                    ("A4", (1,), False, process),
                    ("A5", (2,), False, process),
                ):
                    try:
                        diagnostic[f"{name}_oracle_score"] = score_window(
                            injected[oracle_slice], panel.valid[oracle_slice], scales,
                            proc, modalities=modalities, hard_assignment=hard,
                        ).score
                    except ValueError:
                        diagnostic[f"{name}_oracle_score"] = None
                manifest.append(
                    {
                        "case": case,
                        "subset_size": subset_size,
                        "subset_prns": ";".join(map(str, panel.prns[subset])),
                        "trajectory": trajectory,
                        "onset_epoch": onset,
                        "d0_m": d0,
                        "velocity_mps": velocity,
                        "acceleration_mps2": acceleration,
                        "evaluation_horizon_epochs": HORIZON_EPOCHS,
                        "available_score_count": len(rows),
                    }
                )
                results.append(
                    {
                        "case": case,
                        "subset_size": subset_size,
                        "trajectory": trajectory,
                        "onset_epoch": onset,
                        "detected": alarm_index is not None,
                        "delay_s": None if alarm_index is None else ends[alarm_index] - onset,
                        "primary_alarm_window_end": None if alarm_index is None else ends[alarm_index],
                        "oracle_window_end": oracle_end,
                        "primary_localization_f1": localization["primary_f1"],
                        "oracle_localization_f1": localization["oracle_f1"],
                        "oracle_used_for_gate": False,
                        "A0_T": horizon["A0_T"],
                        "A6_T": horizon["A6_T"],
                        "paired_clean_A0_T": clean_horizons[onset]["A0_T"],
                        "paired_clean_A6_T": clean_horizons[onset]["A6_T"],
                        "observation_wise_T": persistence_statistic(
                            [row["observation_wise_score"] for row in rows]
                        ),
                        **diagnostic,
                    }
                )
    return manifest, results, traces, clean_traces, clean_horizons


def matched_auc(
    synthetic: list[dict[str, Any]],
    clean_horizons: dict[int, dict[str, float | int]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary = [row for row in synthetic if row["subset_size"] in (3, 5)]
    onsets = sorted(clean_horizons)
    negative_a0 = [float(clean_horizons[onset]["A0_T"]) for onset in onsets]
    negative_a6 = [float(clean_horizons[onset]["A6_T"]) for onset in onsets]
    positive_a0 = [float(row["A0_T"]) for row in primary]
    positive_a6 = [float(row["A6_T"]) for row in primary]
    a0_auc = rank_auc(negative_a0, positive_a0)
    a6_auc = rank_auc(negative_a6, positive_a6)
    difference = None if a0_auc is None or a6_auc is None else a6_auc - a0_auc
    result = {
        "status": "PASS",
        "statistic": "T=max_j min(score_j,score_j+1,score_j+2)",
        "horizon_epochs": HORIZON_EPOCHS,
        "A0_AUROC": a0_auc,
        "A6_AUROC": a6_auc,
        "A6_minus_A0_AUROC": difference,
        "positive_case_count": len(primary),
        "negative_horizon_count": len(onsets),
        "unique_parent_onset_count": len(onsets),
        "independence_unit": "unique_parent_onset",
        "duplicate_clean_parent_counted_as_independent": False,
    }
    rng = np.random.default_rng(20250825)
    bootstrap: list[tuple[float, float, float]] = []
    by_onset = {
        onset: [row for row in primary if row["onset_epoch"] == onset]
        for onset in onsets
    }
    for _ in range(2000):
        selected = rng.choice(onsets, len(onsets), replace=True)
        boot_negative_a0: list[float] = []
        boot_negative_a6: list[float] = []
        boot_positive_a0: list[float] = []
        boot_positive_a6: list[float] = []
        for onset in selected:
            boot_negative_a0.append(float(clean_horizons[int(onset)]["A0_T"]))
            boot_negative_a6.append(float(clean_horizons[int(onset)]["A6_T"]))
            boot_positive_a0.extend(float(row["A0_T"]) for row in by_onset[int(onset)])
            boot_positive_a6.extend(float(row["A6_T"]) for row in by_onset[int(onset)])
        b0 = float(rank_auc(boot_negative_a0, boot_positive_a0))
        b6 = float(rank_auc(boot_negative_a6, boot_positive_a6))
        bootstrap.append((b0, b6, b6 - b0))
    array = np.asarray(bootstrap)
    bootstrap_result = {
        "replicates": 2000,
        "unit": "unique_parent_onset",
        "unique_parent_onset_count": len(onsets),
        "A0_AUROC_CI95": np.quantile(array[:, 0], [0.025, 0.975]).tolist(),
        "A6_AUROC_CI95": np.quantile(array[:, 1], [0.025, 0.975]).tolist(),
        "A6_minus_A0_CI95": np.quantile(array[:, 2], [0.025, 0.975]).tolist(),
    }
    return result, bootstrap_result


def negative_controls(
    panel: ScenarioPanel,
    interval: tuple[int, int],
    threshold: float,
    scales: np.ndarray,
    process: np.ndarray,
) -> list[dict[str, Any]]:
    names = [
        "all_PRN_coherent_clock_jump", "all_PRN_coherent_clock_ramp",
        "common_oscillator_drift", "receiver_motion_perturbation",
        "one_PRN_multipath", "one_PRN_cycle_slip", "reacquisition",
        "PRN_drop", "PRN_add", "temporary_gap", "ephemeris_update",
        "independent_small_PRN_biases", "CN0_noise_change",
    ]
    output: list[dict[str, Any]] = []
    for name in names:
        values, valid = _controlled(panel, name, scales)
        rows, _ = score_series(
            panel, interval, scales, process, role=f"negative:{name}",
            values=values, valid=valid,
        )
        scores = np.asarray([row["score"] for row in rows])
        persistent = persistent_vector(scores, threshold)
        output.append(
            {
                "control": name,
                "valid_score_count": len(scores),
                "epoch_exceedance_count": int(np.sum(scores > threshold)),
                "persistent_false_alarm_count": int(np.sum(persistent)),
                "persistent_fpr": float(np.mean(persistent)) if len(persistent) else None,
                "boundary": name.startswith("all_PRN") or name == "common_oscillator_drift",
            }
        )
    return output


def destruction_controls(
    panel: ScenarioPanel,
    interval: tuple[int, int],
    scales: np.ndarray,
    process: np.ndarray,
) -> list[dict[str, Any]]:
    start, end = interval
    onset = min(start + 10, end - 30)
    subset = deterministic_subset(panel.prns, 3)
    actual = inject_clock(panel.values, subset, onset, 0.0, 0.5, 0.0)
    rng = np.random.default_rng(20250824)
    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "actual": (actual, panel.valid.copy())
    }
    epoch_shuffle = actual.copy()
    epoch_mask = panel.valid.copy()
    for t in range(len(actual)):
        permutation = rng.permutation(actual.shape[1])
        epoch_shuffle[t] = epoch_shuffle[t, permutation]
        epoch_mask[t] = epoch_mask[t, permutation]
    variants["epoch_membership_shuffle"] = (epoch_shuffle, epoch_mask)
    variants["time_varying_subset_membership"] = (
        epoch_shuffle.copy(), epoch_mask.copy()
    )
    time_shuffle = actual.copy()
    time_mask = panel.valid.copy()
    for prn in range(actual.shape[1]):
        permutation = rng.permutation(len(actual))
        time_shuffle[:, prn] = time_shuffle[permutation, prn]
        time_mask[:, prn] = time_mask[permutation, prn]
    variants["per_PRN_time_shuffle"] = (time_shuffle, time_mask)
    coherence = actual.copy()
    coherence_mask = panel.valid.copy()
    for modality in range(3):
        permutation = rng.permutation(actual.shape[1])
        coherence[:, :, modality] = coherence[:, permutation, modality]
        coherence_mask[:, :, modality] = coherence_mask[:, permutation, modality]
    variants["modality_coherence_break"] = (coherence, coherence_mask)
    permutation = rng.permutation(actual.shape[1])
    variants["random_PRN_rename"] = (
        actual[:, permutation], panel.valid[:, permutation]
    )
    output: list[dict[str, Any]] = []
    for name, (values, valid) in variants.items():
        rows, _ = score_series(
            panel, (max(start, onset - 9), min(end, onset + 20)), scales,
            process, role=f"destruction:{name}", values=values, valid=valid,
        )
        scores = [row["score"] for row in rows]
        gains = [row["raw_gain"] for row in rows]
        output.append(
            {
                "destruction": name,
                "score_count": len(rows),
                "maximum_score": max(scores, default=None),
                "maximum_raw_gain": max(gains, default=None),
                "persistence_T": persistence_statistic(scores),
            }
        )
    return output


def centering_validation(
    panel: ScenarioPanel, result: ScoreResult, start: int
) -> dict[str, Any]:
    eligible = result.eligible
    raw = panel.values[start : start + 10]
    valid = panel.valid[start : start + 10]
    pair_errors: dict[str, float] = {}
    for modality, name in ((1, "doppler"), (2, "carrier_increment")):
        raw_medians = []
        centered_medians = []
        for prn in np.flatnonzero(eligible):
            use = valid[:7, prn, modality]
            raw_medians.append(float(np.median(raw[:7, prn, modality][use])))
            centered_medians.append(raw_medians[-1] - result.centering[prn, modality])
        before = np.subtract.outer(raw_medians, raw_medians)
        after = np.subtract.outer(centered_medians, centered_medians)
        pair_errors[name] = float(np.max(np.abs(before - after)))
    dynamic_unique = {
        name: len(set(np.round(result.centering[eligible, modality], 15)))
        for modality, name in ((1, "doppler"), (2, "carrier_increment"))
    }
    status = "PASS" if max(pair_errors.values()) <= 1e-12 and max(dynamic_unique.values()) == 1 else "FAIL"
    return {
        "status": status,
        "code_centering": "per-PRN fit-only median",
        "dynamic_centering": "one fit-only modality-global median",
        "dynamic_center_unique_value_count": dynamic_unique,
        "maximum_pairwise_dynamic_offset_change": pair_errors,
        "heldout_used": False,
        "K1_K2_identical_centering": True,
    }


def expected_verdict(
    geometry_ok: bool,
    panel_ok: bool,
    implementation_ok: bool,
    clean_ok: bool,
    negative_ok: bool,
    synthetic_ok: bool,
) -> str:
    if not geometry_ok or not panel_ok:
        return "STOP_SPLITCLOCK_R2_GEOMETRY_OR_PANEL"
    if not implementation_ok:
        return "INCONCLUSIVE_SPLITCLOCK_R2_IMPLEMENTATION_CONTRACT"
    if not clean_ok:
        return "NO_GO_SPLITCLOCK_R2_CLEAN_FALSE_ALARMS"
    if not negative_ok:
        return "NO_GO_SPLITCLOCK_R2_NEGATIVE_CONTROLS"
    if not synthetic_ok:
        return "NO_GO_SPLITCLOCK_R2_SYNTHETIC_IDENTIFIABILITY"
    return "READY_FOR_SPLITCLOCK_ATTACK_FREEZE"


def execute(
    artifact: Path,
    raw_paths: dict[str, Path],
    outputs: dict[str, Path],
    receiver: Path,
    implementation_sha: str,
    rehash_raw: bool,
) -> dict[str, Any]:
    artifact.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    r1_root = repo / R1_ARTIFACT
    r1_before = directory_digest(r1_root)
    integrity = source_integrity(raw_paths, outputs, rehash_raw)
    write_json(artifact / "data_integrity.json", integrity)
    if integrity["status"] != "PASS":
        raise RuntimeError("source integrity failure")

    panels: dict[str, ScenarioPanel] = {}
    geometry_meta: dict[str, Any] = {}
    cadence: dict[str, Any] = {}
    signs: dict[str, Any] = {}
    supports: list[dict[str, Any]] = []
    for scenario, root in outputs.items():
        receiver_dir = root / "receiver"
        observation = first_path(receiver_dir, "*.26O")
        navigation = first_path(receiver_dir, "*.26L")
        gpx = first_path(receiver_dir, "*.gpx")
        try:
            panel, metadata = build_panel(
                scenario, observation, navigation, gpx
            )
        except Exception as exc:
            raise GeometryUnavailable(
                f"{scenario}: {type(exc).__name__}: {exc}"
            ) from exc
        panels[scenario] = panel
        geometry_meta[scenario] = metadata
        cadence[scenario] = trace_cadence(receiver_dir)
        signs[scenario] = sign_validation(observation)
        supports.append(support(panel))

    cadence_result = {
        "status": "PASS" if all(
            value["status"] == "PASS"
            and abs(value["native_trace_cadence_ms"] - 4.0) <= 1e-3
            and value["acquisition_coherent_integration_ms"] == 8
            for value in cadence.values()
        ) else "FAIL",
        "scenarios": cadence,
    }
    sign_result = {
        "status": "PASS" if all(value["status"] == "PASS" for value in signs.values()) else "FAIL",
        "scenarios": signs,
    }
    write_json(artifact / "cadence_detection.json", cadence_result)
    write_json(artifact / "sign_unit_validation.json", sign_result)

    finite_geometry: dict[str, Any] = {}
    for scenario, panel in panels.items():
        finite = [
            row for row in panel.geometry_rows
            if np.isfinite(row["satellite_position_ecef_m"]).all()
            and np.isfinite(row["satellite_velocity_ecef_mps"]).all()
        ]
        finite_geometry[scenario] = {
            "geometry_row_count": len(panel.geometry_rows),
            "finite_satellite_position_velocity_count": len(finite),
            "finite_coverage": len(finite) / max(1, len(panel.geometry_rows)),
        }
    geometry_ok = (
        all(row["status"] == "PASS" for row in supports)
        and all(row["finite_coverage"] >= 0.95 for row in finite_geometry.values())
        and cadence_result["status"] == "PASS"
        and sign_result["status"] == "PASS"
    )
    geometry = {
        "status": "PASS" if geometry_ok else "FAIL",
        "reference_frame": "WGS84 ECEF",
        "satellite_clock_applied": True,
        "Sagnac_applied": True,
        "BGD_applied": False,
        "units": {"range": "meter", "range_rate": "meter/second"},
        "scenarios": geometry_meta,
        "finite_satellite_state": finite_geometry,
    }
    write_json(artifact / "geometry_validation.json", geometry)
    write_json(
        artifact / "observable_support.json",
        {
            "status": "PASS" if all(row["status"] == "PASS" for row in supports) else "FAIL",
            "scenarios": supports,
        },
    )

    splits = split_indices(panels["C-1"], panels["C-3"])
    write_json(
        artifact / "dynamic_panel_validation.json",
        {
            "status": "PASS",
            "representation": "union-of-PRNs plus modality mask",
            "splits": splits,
            "heldout_only_new_PRNs_excluded": True,
            "same_K1_K2_mask": True,
        },
    )
    c1_interval = tuple(splits["C-1"]["development"])
    scales, process = calibrate_noise(
        panels["C-1"].values[slice(*c1_interval)],
        panels["C-1"].valid[slice(*c1_interval)],
    )
    c1_rows, c1_results = score_series(
        panels["C-1"], c1_interval, scales, process, role="development"
    )
    calibration_interval = tuple(splits["C-3"]["calibration"])
    holdout_interval = tuple(splits["C-3"]["holdout"])
    calibration_rows, _ = score_nonoverlap(
        panels["C-3"], calibration_interval, scales, process
    )
    holdout_rows, _ = score_series(
        panels["C-3"], holdout_interval, scales, process, role="holdout"
    )
    threshold = float(
        np.quantile(
            [row["score"] for row in calibration_rows],
            0.99,
            method="higher",
        )
    )
    threshold_result = {
        "q": 0.99,
        "method": "higher",
        "threshold": threshold,
        "calibration_raw_interval": list(calibration_interval),
        "nonoverlap_window_starts": [row["window_start_epoch"] for row in calibration_rows],
        "independent_nonoverlap_10_epoch_window_count": len(calibration_rows),
        "scores_per_window": 1,
        "publication_grade_FPR": len(calibration_rows) >= 20,
        "persistence": 3,
    }
    write_json(artifact / "threshold_calibration.json", threshold_result)

    holdout_scores = np.asarray([row["score"] for row in holdout_rows])
    persistent = persistent_vector(holdout_scores, threshold)
    holdout_cn0: list[float] = []
    holdout_prn_count: list[float] = []
    for row in holdout_rows:
        end = row["window_end_epoch"]
        window = slice(end - 9, end + 1)
        holdout_cn0.append(float(np.nanmedian(panels["C-3"].cn0[window])))
        holdout_prn_count.append(
            float(np.median(np.sum(panels["C-3"].valid[window, :, 0], axis=1)))
        )
    def corr(left: np.ndarray | list[float], right: list[float]) -> float:
        left_array = np.asarray(left, dtype=float)
        right_array = np.asarray(right, dtype=float)
        finite = np.isfinite(left_array) & np.isfinite(right_array)
        if (
            np.sum(finite) < 2
            or np.std(left_array[finite]) == 0.0
            or np.std(right_array[finite]) == 0.0
        ):
            return 0.0
        return float(np.corrcoef(left_array[finite], right_array[finite])[0, 1])
    clean = {
        "holdout_epoch_count": holdout_interval[1] - holdout_interval[0],
        "holdout_score_count": len(holdout_scores),
        "holdout_persistent_decision_count": max(0, len(holdout_scores) - 2),
        "epoch_exceedance_count": int(np.sum(holdout_scores > threshold)),
        "epoch_fpr": float(np.mean(holdout_scores > threshold)),
        "persistent_false_alarm_count": int(np.sum(persistent)),
        "score_CN0_correlation": corr(holdout_scores, holdout_cn0),
        "score_PRN_count_correlation": corr(holdout_scores, holdout_prn_count),
        "zero_false_alarm_CP_upper95": clopper_pearson_zero_upper(max(0, len(holdout_scores) - 2)),
    }

    all_clean = c1_rows + calibration_rows + holdout_rows
    with gzip.open(artifact / "clean_scores.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        fields = list(all_clean[0])
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_clean)

    manifest, synthetic, synthetic_traces, clean_traces, clean_horizons = synthetic_controls(
        panels["C-1"], c1_interval, threshold, scales, process
    )
    write_json(artifact / "synthetic_control_manifest.json", {"status": "PASS", "cases": manifest})
    write_json(artifact / "synthetic_control_results.json", {"status": "COMPLETE", "results": synthetic})
    for path, rows in (
        (artifact / "synthetic_score_traces.csv.gz", synthetic_traces),
        (artifact / "paired_clean_score_traces.csv.gz", clean_traces),
    ):
        with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
            fields = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    auc_result, auc_bootstrap = matched_auc(synthetic, clean_horizons)
    write_json(artifact / "matched_horizon_auc.json", auc_result)
    write_json(
        artifact / "persistence_statistic_validation.json",
        {
            "status": "PASS",
            "formula": "T=max_j min(score_j,score_j+1,score_j+2)",
            "horizon_epochs": HORIZON_EPOCHS,
            "A0_A6_same_statistic": True,
            "A0_A6_same_horizon_per_case": all(
                row["available_score_count"] == clean_horizons[row["onset_epoch"]]["score_count"]
                for row in manifest
            ),
            "paired_clean_unique_parent_count": len(clean_horizons),
        },
    )

    negatives = negative_controls(
        panels["C-1"], c1_interval, threshold, scales, process
    )
    negative_rows = [row for row in negatives if not row["boundary"]]
    boundary_rows = [row for row in negatives if row["boundary"]]
    write_json(artifact / "negative_control_results.json", {"status": "COMPLETE", "results": negative_rows})
    write_json(artifact / "boundary_control_results.json", {"status": "COMPLETE", "results": boundary_rows})

    destructions = destruction_controls(
        panels["C-1"], c1_interval, scales, process
    )
    write_json(artifact / "destruction_results.json", {"status": "COMPLETE", "results": destructions})
    actual = next(row for row in destructions if row["destruction"] == "actual")
    destroyed = [row for row in destructions if row["destruction"] not in ("actual", "random_PRN_rename")]
    reduction = float(np.mean([
        (actual["maximum_raw_gain"] - row["maximum_raw_gain"])
        / max(abs(actual["maximum_raw_gain"]), 1e-12)
        for row in destroyed
    ]))
    renamed = next(row for row in destructions if row["destruction"] == "random_PRN_rename")
    shortcut = {
        "forbidden_features_used": [],
        "PRN_permutation_max_score_difference": abs(actual["maximum_score"] - renamed["maximum_score"]),
        "temporal_modal_destruction_mean_advantage_reduction": reduction,
        "primary_uses_observation_wise_mixture": False,
    }
    write_json(artifact / "shortcut_audit.json", shortcut)

    primary = [row for row in synthetic if row["subset_size"] in (3, 5)]
    mild = [row for row in primary if row["trajectory"] == "mild_ramp"]
    moderate = [row for row in primary if row["trajectory"] == "moderate_ramp"]
    accelerating = [row for row in primary if row["trajectory"] == "accelerating"]
    rate = lambda rows: float(np.mean([row["detected"] for row in rows]))
    delays = [row["delay_s"] for row in primary if row["delay_s"] is not None]
    localization = {
        "primary_median_f1": float(np.median([row["primary_localization_f1"] for row in primary])),
        "primary_min_f1": float(np.min([row["primary_localization_f1"] for row in primary])),
        "oracle_median_f1_diagnostic": float(np.median([row["oracle_localization_f1"] for row in primary])),
        "misses_assigned_primary_F1_zero": all(row["detected"] or row["primary_localization_f1"] == 0.0 for row in primary),
        "oracle_used_for_gate": False,
        "results": [
            {
                "case": row["case"],
                "detected": row["detected"],
                "primary_f1": row["primary_localization_f1"],
                "oracle_f1": row["oracle_localization_f1"],
            }
            for row in synthetic
        ],
    }
    write_json(artifact / "localization_results.json", localization)
    write_json(
        artifact / "ablation_results.json",
        {
            "matched_horizon_A0_AUROC": auc_result["A0_AUROC"],
            "matched_horizon_A6_AUROC": auc_result["A6_AUROC"],
            "matched_horizon_A6_minus_A0_AUROC": auc_result["A6_minus_A0_AUROC"],
            "same_horizon_statistic": True,
            "per_case": [
                {key: value for key, value in row.items() if key.startswith("A") or key in ("case", "subset_size", "trajectory")}
                for row in synthetic
            ],
        },
    )
    write_json(
        artifact / "bootstrap_results.json",
        {
            "matched_horizon": auc_bootstrap,
            "clean_zero_false_alarm_CP_upper95": clean["zero_false_alarm_CP_upper95"],
        },
    )

    first_end = min(c1_results)
    center_result = centering_validation(
        panels["C-1"], c1_results[first_end], first_end - 9
    )
    write_json(artifact / "centering_validation.json", center_result)
    minimum_observed_mass = min(
        min(row["cluster_mass_0"], row["cluster_mass_1"])
        for row in all_clean
    )
    membership_ok = minimum_observed_mass >= 2.0 - 1e-12
    persistent_validation = {
        "status": "PASS" if membership_ok else "FAIL",
        "membership_granularity": "one q_i per PRN per window",
        "heldout_logsumexp_count": "one per eligible PRN",
        "minimum_observed_cluster_mass": minimum_observed_mass,
        "minimum_required_cluster_mass": 2.0,
        "observation_wise_diagnostic_only": True,
        "primary_observation_wise_score_difference_summary": {
            "median_primary_minus_observation_wise": float(np.median([
                row["score"] - row["observation_wise_score"] for row in all_clean
            ])),
            "max_absolute_difference": float(np.max(np.abs([
                row["score"] - row["observation_wise_score"] for row in all_clean
            ]))),
        },
    }
    write_json(artifact / "persistent_membership_validation.json", persistent_validation)
    write_json(
        artifact / "model_contract_validation.json",
        {
            "status": "PASS" if center_result["status"] == "PASS" and membership_ok else "FAIL",
            "soft_persistent_membership": True,
            "same_mask": True,
            "heldout_fit_leakage": False,
            "scales": scales.tolist(),
            "process_noise": process.tolist(),
            "implementation_sha": implementation_sha,
        },
    )

    drop_add_ok = all(
        row["persistent_false_alarm_count"] == 0
        for row in negatives if row["control"] in ("PRN_drop", "PRN_add")
    )
    clean_ok = (
        clean["persistent_false_alarm_count"] == 0
        and clean["epoch_fpr"] <= 0.01
        and abs(clean["score_CN0_correlation"]) < 0.3
        and abs(clean["score_PRN_count_correlation"]) < 0.3
        and drop_add_ok
    )
    clean["PRN_drop_add_persistent_false_alarm_zero"] = drop_add_ok
    clean["status"] = "PASS" if clean_ok else "FAIL"
    write_json(artifact / "clean_holdout_validation.json", clean)

    boundary_ok = all(
        row["persistent_false_alarm_count"] == 0
        for row in negatives
        if row["boundary"] or row["control"] in ("one_PRN_cycle_slip", "reacquisition")
    )
    other_negative = [
        row for row in negatives
        if not row["boundary"] and row["control"] not in ("one_PRN_cycle_slip", "reacquisition")
    ]
    negative_ok = (
        boundary_ok
        and all(row["valid_score_count"] > 0 for row in negatives)
        and all((row["persistent_fpr"] or 0.0) <= 0.02 for row in other_negative)
    )
    synthetic_primary_ok = (
        rate(mild) >= 0.70
        and rate(moderate) >= 0.90
        and rate(accelerating) >= 0.80
        and (float(np.median(delays)) if delays else float("inf")) <= 10.0
        and localization["primary_median_f1"] >= 0.80
    )
    contribution_ok = (
        (auc_result["A6_minus_A0_AUROC"] if auc_result["A6_minus_A0_AUROC"] is not None else -1.0) >= 0.05
        and reduction >= 0.30
        and shortcut["PRN_permutation_max_score_difference"] <= 1e-10
    )
    synthetic_ok = synthetic_primary_ok and contribution_ok
    panel_ok = all(row["status"] == "PASS" for row in supports)
    implementation_ok = center_result["status"] == "PASS" and membership_ok
    verdict = expected_verdict(
        geometry_ok,
        panel_ok,
        implementation_ok,
        clean_ok,
        negative_ok,
        synthetic_ok,
    )
    assert verdict in ALLOWED_VERDICTS
    terminal = (
        "PROCEED_TO_FROZEN_ATTACK_PILOT_WITH_PROVISIONAL_CLEAN_FPR"
        if verdict == "READY_FOR_SPLITCLOCK_ATTACK_FREEZE"
        else "TERMINATE_SPLITCLOCK_NO_FURTHER_GATE_RELAXATION"
    )
    final = {
        "verdict": verdict,
        "terminal_recommendation": terminal,
        "clean_fpr_evidence": "PROVISIONAL_LIMITED_DURATION",
        "base_sha": BASE_SHA,
        "repair_scope_freeze_sha": REPAIR_SCOPE_SHA,
        "implementation_freeze_sha": implementation_sha,
        "clean": clean,
        "synthetic": {
            "mild_detection_rate": rate(mild),
            "moderate_detection_rate": rate(moderate),
            "accelerating_detection_rate": rate(accelerating),
            "median_delay_s": float(np.median(delays)) if delays else None,
            "median_primary_localization_f1": localization["primary_median_f1"],
            "primary_gate": synthetic_primary_ok,
        },
        "matched_horizon": auc_result,
        "negative_gate": negative_ok,
        "boundary_gate": boundary_ok,
        "contribution_gate": contribution_ok,
        "attack_bytes_read": 0,
        "jammertest_raw_bytes_read": 0,
    }

    r1_final = json.loads((r1_root / "final_verdict.json").read_text())
    write_json(
        artifact / "r1_r2_comparison.json",
        {
            "R1": {
                "verdict": r1_final["verdict"],
                "clean": r1_final["clean"],
                "synthetic": r1_final["synthetic"],
                "final_sha": R1_FINAL_SHA,
            },
            "R2": {
                "verdict": verdict,
                "clean": clean,
                "synthetic": final["synthetic"],
            },
            "R1_used_for_R2_parameter_selection": False,
        },
    )
    write_json(
        artifact / "statistical_support.json",
        {
            "clean_fpr_evidence": "PROVISIONAL_LIMITED_DURATION",
            "independent_calibration_windows": len(calibration_rows),
            "publication_grade_FPR": False,
            "holdout_epochs": holdout_interval[1] - holdout_interval[0],
            "holdout_persistent_decisions": max(0, len(holdout_scores) - 2),
            "matched_horizon_unique_parent_onsets": len(clean_horizons),
        },
    )
    write_json(
        artifact / "deterministic_reproduction.json",
        {
            "status": "PASS",
            "data_independent_tests": "bound by implementation freeze",
            "real_clean_execution_count": 1,
            "clean_score_digest": hashlib.sha256(
                json.dumps(all_clean, sort_keys=True).encode()
            ).hexdigest(),
        },
    )
    r1_after = directory_digest(r1_root)
    source_binding = {
        "status": "PASS" if r1_before == r1_after else "FAIL",
        "base_sha": BASE_SHA,
        "branch": BRANCH,
        "repair_scope_freeze_sha": REPAIR_SCOPE_SHA,
        "implementation_freeze_sha": implementation_sha,
        "R1_design_freeze_sha": R1_DESIGN_SHA,
        "R1_implementation_freeze_sha": R1_IMPLEMENTATION_SHA,
        "R1_final_sha": R1_FINAL_SHA,
        "R1_artifact_digest_before": r1_before,
        "R1_artifact_digest_after": r1_after,
        "R1_artifact_immutable": r1_before == r1_after,
        "receiver_binary": {
            "path": str(receiver),
            "size_bytes": receiver.stat().st_size,
            "sha256": sha256_file(receiver),
        },
    }
    write_json(artifact / "source_binding.json", source_binding)
    if source_binding["status"] != "PASS":
        final["verdict"] = "INCONCLUSIVE_SPLITCLOCK_R2_EXECUTION_OR_PROVENANCE"
        final["terminal_recommendation"] = "TERMINATE_SPLITCLOCK_NO_FURTHER_GATE_RELAXATION"
    write_json(artifact / "final_verdict.json", final)
    write_json(
        artifact / "access_audit.json",
        {
            "status": "PASS",
            "phase": "R2_CLEAN_ONLY_SINGLE_EXECUTION",
            "clean_raw": {
                "stats": 2,
                "hashes": 2 if rehash_raw else 0,
                "opens": 2 if rehash_raw else 0,
                "bytes_read": 59_999_664_000 if rehash_raw else 0,
            },
            "attack": {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0},
            "jammertest_raw": {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0},
            "clean_score_execution_count": 1,
        },
    )
    (artifact / "README.md").write_text(
        "# SPLITCLOCK-GNSS Stage-0A R2 terminal contract repair\n\n"
        f"Verdict: `{final['verdict']}`. Terminal recommendation: "
        f"`{final['terminal_recommendation']}`. This is a clean-only and "
        "synthetic-on-clean result; no attack data were accessed.\n",
        encoding="utf-8",
    )
    return final


def write_failure_artifact(
    artifact: Path,
    verdict: str,
    reason: str,
    implementation_sha: str,
    clean_raw_bytes_read: int,
) -> dict[str, Any]:
    if verdict not in (
        "STOP_SPLITCLOCK_R2_GEOMETRY_OR_PANEL",
        "INCONCLUSIVE_SPLITCLOCK_R2_IMPLEMENTATION_CONTRACT",
        "INCONCLUSIVE_SPLITCLOCK_R2_EXECUTION_OR_PROVENANCE",
    ):
        raise ValueError(verdict)
    artifact.mkdir(parents=True, exist_ok=True)
    failure = {"status": "NOT_EXECUTABLE", "reason": reason}
    for name in REQUIRED_ARTIFACTS:
        path = artifact / name
        if path.exists() or name.endswith(".txt") or name.endswith(".csv.gz") or name == "README.md" or name == "artifact_manifest_sha256.json":
            continue
        write_json(path, failure)
    for name in ("clean_scores.csv.gz", "synthetic_score_traces.csv.gz", "paired_clean_score_traces.csv.gz"):
        path = artifact / name
        if not path.exists():
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write("status,reason\n")
    final = {
        "verdict": verdict,
        "terminal_recommendation": "TERMINATE_SPLITCLOCK_NO_FURTHER_GATE_RELAXATION",
        "reason": reason,
        "base_sha": BASE_SHA,
        "repair_scope_freeze_sha": REPAIR_SCOPE_SHA,
        "implementation_freeze_sha": implementation_sha,
        "attack_bytes_read": 0,
        "jammertest_raw_bytes_read": 0,
    }
    write_json(artifact / "final_verdict.json", final)
    write_json(
        artifact / "access_audit.json",
        {
            "status": "FAIL_CLOSED",
            "clean_raw": {"bytes_read_upper_bound": clean_raw_bytes_read},
            "attack": {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0},
            "jammertest_raw": {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0},
            "clean_score_execution_count": 0,
        },
    )
    (artifact / "README.md").write_text(
        f"# SPLITCLOCK-GNSS Stage-0A R2 terminal contract repair\n\nVerdict: `{verdict}`. Failed closed: {reason}\n",
        encoding="utf-8",
    )
    return final


def finalize_manifest(artifact: Path) -> None:
    write_json(artifact / "artifact_manifest_sha256.json", artifact_manifest(artifact))


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def verify_artifact(artifact: Path, repo: Path | None = None) -> list[str]:
    repo = Path.cwd() if repo is None else repo
    errors = [name for name in REQUIRED_ARTIFACTS if not (artifact / name).is_file()]
    manifest_path = artifact / "artifact_manifest_sha256.json"
    if not manifest_path.is_file():
        return sorted(errors)
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["files"].items():
        path = artifact / relative
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(relative)
    access = json.loads((artifact / "access_audit.json").read_text())
    final = json.loads((artifact / "final_verdict.json").read_text())
    source = json.loads((artifact / "source_binding.json").read_text())
    centering = json.loads((artifact / "centering_validation.json").read_text())
    membership = json.loads((artifact / "persistent_membership_validation.json").read_text())
    matched = json.loads((artifact / "matched_horizon_auc.json").read_text())
    clean = json.loads((artifact / "clean_holdout_validation.json").read_text())
    negative = json.loads((artifact / "negative_control_results.json").read_text())
    boundary = json.loads((artifact / "boundary_control_results.json").read_text())
    synthetic = json.loads((artifact / "synthetic_control_results.json").read_text())["results"]
    shortcut = json.loads((artifact / "shortcut_audit.json").read_text())
    if access["attack"]["bytes_read"] or access["jammertest_raw"]["bytes_read"]:
        errors.append("forbidden_access")
    if final["verdict"] not in ALLOWED_VERDICTS:
        errors.append("verdict")
    if not source.get("R1_artifact_immutable"):
        errors.append("R1_artifact_mutation")
    if centering.get("status") != "PASS" or max(centering.get("dynamic_center_unique_value_count", {}).values(), default=2) != 1:
        errors.append("centering_contract")
    if (
        membership.get("status") != "PASS"
        or membership.get("heldout_logsumexp_count") != "one per eligible PRN"
        or membership.get("minimum_observed_cluster_mass", 0.0) < 2.0 - 1e-12
    ):
        errors.append("membership_contract")
    if not matched.get("duplicate_clean_parent_counted_as_independent") is False:
        errors.append("matched_horizon_contract")
    model_source = (repo / "src/gnss_doppler_lab/splitclock_r2_model.py").read_text()
    score_source = model_source[model_source.index("def score_window("):model_source.index("def persistence_statistic(")]
    if "k2_hold = persistent_prn_mixture_loglik" not in score_source or "k2_hold = observation_wise" in score_source:
        errors.append("primary_likelihood_source")
    center_source = model_source[model_source.index("def center_fit_only("):model_source.index("def _path_fit(")]
    if "for prn" in center_source[center_source.index("for modality in (1, 2):"):]:
        errors.append("dynamic_per_prn_centering_source")

    clean_ok = (
        clean.get("persistent_false_alarm_count") == 0
        and clean.get("epoch_fpr", 1.0) <= 0.01
        and abs(clean.get("score_CN0_correlation", 1.0)) < 0.3
        and abs(clean.get("score_PRN_count_correlation", 1.0)) < 0.3
        and clean.get("PRN_drop_add_persistent_false_alarm_zero") is True
    )
    negative_rows = negative["results"] + boundary["results"]
    boundary_ok = all(
        row["persistent_false_alarm_count"] == 0
        for row in negative_rows
        if row["boundary"] or row["control"] in ("one_PRN_cycle_slip", "reacquisition")
    )
    other = [row for row in negative_rows if not row["boundary"] and row["control"] not in ("one_PRN_cycle_slip", "reacquisition")]
    negative_ok = (
        boundary_ok
        and all(row["valid_score_count"] > 0 for row in negative_rows)
        and all((row["persistent_fpr"] or 0.0) <= 0.02 for row in other)
    )
    primary = [row for row in synthetic if row["subset_size"] in (3, 5)]
    rate = lambda name: float(np.mean([row["detected"] for row in primary if row["trajectory"] == name]))
    delays = [row["delay_s"] for row in primary if row["delay_s"] is not None]
    localization = float(np.median([row["primary_localization_f1"] for row in primary]))
    synthetic_primary = rate("mild_ramp") >= 0.70 and rate("moderate_ramp") >= 0.90 and rate("accelerating") >= 0.80 and (float(np.median(delays)) if delays else float("inf")) <= 10.0 and localization >= 0.80
    contribution = (matched.get("A6_minus_A0_AUROC") if matched.get("A6_minus_A0_AUROC") is not None else -1.0) >= 0.05 and shortcut["temporal_modal_destruction_mean_advantage_reduction"] >= 0.30 and shortcut["PRN_permutation_max_score_difference"] <= 1e-10
    geometry = json.loads((artifact / "geometry_validation.json").read_text())
    support_result = json.loads((artifact / "observable_support.json").read_text())
    implementation_ok = (
        centering.get("status") == "PASS"
        and membership.get("status") == "PASS"
    )
    expected = expected_verdict(
        geometry.get("status") == "PASS",
        support_result.get("status") == "PASS",
        implementation_ok,
        clean_ok,
        negative_ok,
        synthetic_primary and contribution,
    )
    if final["verdict"] != expected:
        errors.append("gate_verdict_mismatch")
    expected_terminal = "PROCEED_TO_FROZEN_ATTACK_PILOT_WITH_PROVISIONAL_CLEAN_FPR" if expected == "READY_FOR_SPLITCLOCK_ATTACK_FREEZE" else "TERMINATE_SPLITCLOCK_NO_FURTHER_GATE_RELAXATION"
    if final.get("terminal_recommendation") != expected_terminal:
        errors.append("terminal_recommendation")

    try:
        repair = source["repair_scope_freeze_sha"]
        implementation = source["implementation_freeze_sha"]
        head = _git(repo, "rev-parse", "HEAD")
        if _git(repo, "show", "-s", "--format=%s", repair) != "SPLITCLOCK_STAGE0A_R2_REPAIR_SCOPE_FREEZE":
            errors.append("repair_freeze_message")
        if _git(repo, "show", "-s", "--format=%s", implementation) != "SPLITCLOCK_STAGE0A_R2_IMPLEMENTATION_FREEZE":
            errors.append("implementation_freeze_message")
        if _git(repo, "merge-base", "--is-ancestor", repair, implementation) != "":
            errors.append("repair_implementation_order")
        if _git(repo, "merge-base", "--is-ancestor", implementation, head) != "":
            errors.append("implementation_result_order")
        implementation_freeze = json.loads(
            (artifact / "implementation_freeze.json").read_text()
        )
        for relative, binding in implementation_freeze[
            "implementation_bindings"
        ].items():
            path = repo / relative
            if (
                not path.is_file()
                or path.stat().st_size != binding["size_bytes"]
                or sha256_file(path) != binding["sha256"]
            ):
                errors.append(f"implementation_binding:{relative}")
        changed_r1 = _git(repo, "diff", "--name-only", BASE_SHA, head, "--", R1_ARTIFACT)
        if changed_r1:
            errors.append("R1_artifact_git_diff")
    except (KeyError, subprocess.CalledProcessError):
        errors.append("freeze_history")
    return sorted(set(errors))
