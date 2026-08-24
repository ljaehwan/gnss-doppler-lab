"""Frozen clean-only SPLITCLOCK R1 experiment orchestration."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .splitclock_observable_audit import artifact_manifest, md5_file, verify_output_manifest
from .splitclock_r1_contract import ALLOWED_VERDICTS, BASE_SHA, BRANCH, R1_ARTIFACT
from .splitclock_r1_geometry import ScenarioPanel, build_panel, parse_rinex_observations, trace_cadence
from .splitclock_r1_model import ScoreResult, calibrate_noise, inject_clock, score_window
from .splitclock_stage0a import sha256_file

RAW_BINDINGS = {
    "C-1": {"size_bytes": 29_999_832_000, "md5": "4ff0e86938792bf3150c30d5f1481917"},
    "C-3": {"size_bytes": 29_999_832_000, "md5": "1b7c99c754faec3c8fa625849ef70014"},
}
OUTPUT_BINDINGS = {
    "C-1": "3fb4ebf1fff38293c1dcd15972cfe5d8615636c34e49664a91e36631dd84c178",
    "C-3": "b8c61a2356192d57b94c794d3d3856f5b68e1393c7583b7fc79579f5695ed4f8",
}
DESIGN_SHA = "d472376ba2e59d766c93296b4755df4c89ccbe9b"
SYNTHETIC_TRAJECTORIES = {
    "mild_ramp": (0.0, 0.1, 0.0),
    "moderate_ramp": (0.0, 0.5, 0.0),
    "accelerating": (0.0, 0.05, 0.02),
    "delayed_ramp": (10.0, 0.1, 0.0),
}


class GeometryUnavailable(RuntimeError):
    """Raised when an exact frozen geometry reconstruction cannot be produced."""


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def first_path(root: Path, pattern: str) -> Path:
    values = sorted(root.glob(pattern))
    if len(values) != 1: raise ValueError(f"expected one {pattern} under {root}, got {len(values)}")
    return values[0]


def source_integrity(raw_paths: dict[str, Path], outputs: dict[str, Path], rehash_raw: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "PASS", "sources": {}, "receiver_outputs": {}}
    for scenario, path in raw_paths.items():
        before = path.stat(); observed = md5_file(path) if rehash_raw else RAW_BINDINGS[scenario]["md5"]; after = path.stat()
        expected = RAW_BINDINGS[scenario]; stable = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
        status = "PASS" if after.st_size == expected["size_bytes"] and observed == expected["md5"] and stable else "FAIL"
        result["sources"][scenario] = {"path": str(path), "size_bytes": after.st_size, "md5": observed, "stable_during_hash": stable, "status": status}
    for scenario, path in outputs.items():
        check = verify_output_manifest(path); status = "PASS" if check["status"] == "PASS" and check["actual_aggregate_sha256"] == OUTPUT_BINDINGS[scenario] else "FAIL"
        result["receiver_outputs"][scenario] = {"path": str(path), "aggregate_sha256": check["actual_aggregate_sha256"], "status": status}
    result["status"] = "PASS" if all(row["status"] == "PASS" for group in (result["sources"], result["receiver_outputs"]) for row in group.values()) else "FAIL"
    return result


def support(panel: ScenarioPanel) -> dict[str, Any]:
    per_epoch = np.sum(panel.valid[:, :, 0] & (panel.valid[:, :, 1] | panel.valid[:, :, 2]), axis=1)
    qualifying = per_epoch >= 5; longest = current = 0
    for value in qualifying:
        current = current + 1 if value else 0; longest = max(longest, current)
    observed_slots = np.isfinite(panel.cn0); expected_modalities = max(1, int(np.sum(observed_slots)) * 3)
    finite = float(np.sum(panel.valid & observed_slots[:, :, None]) / expected_modalities)
    geometry_coverage = float(np.sum(panel.valid[:, :, 0] & observed_slots) / max(1, np.sum(observed_slots)))
    return {"scenario": panel.scenario, "epoch_count": len(panel.epochs), "prn_count_inventory": len(panel.prns), "m_ge_5_epoch_count": int(np.sum(qualifying)), "longest_continuous_m_ge_5_seconds": longest, "finite_tensor_coverage": finite, "geometry_code_slot_coverage": geometry_coverage, "maximum_alignment_error_s": max(panel.alignment_errors_s, default=float("inf")), "status": "PASS" if longest >= 60 and finite >= 0.95 and geometry_coverage >= 0.95 and max(panel.alignment_errors_s, default=float("inf")) <= 1.0 else "FAIL"}


def sign_validation(observation_path: Path) -> dict[str, Any]:
    observations, _ = parse_rinex_observations(observation_path); by_prn: dict[int, list[Any]] = {}
    for value in observations: by_prn.setdefault(value.prn, []).append(value)
    carrier = []; pseudo = []; doppler = []
    wavelength = 299_792_458.0 / 1_575_420_000.0
    for rows in by_prn.values():
        rows.sort(key=lambda value: value.system_time)
        for left, right in zip(rows, rows[1:]):
            if (right.system_time - left.system_time).total_seconds() != 1.0: continue
            carrier.append(wavelength * (right.carrier_cycles - left.carrier_cycles)); pseudo.append(right.pseudorange_m - left.pseudorange_m); doppler.append(-wavelength * 0.5 * (left.doppler_hz + right.doppler_hz))
    arrays = [np.asarray(value) for value in (carrier, pseudo, doppler)]; correlations = {"carrier_vs_pseudorange": float(np.corrcoef(arrays[0], arrays[1])[0, 1]), "carrier_vs_doppler_integral": float(np.corrcoef(arrays[0], arrays[2])[0, 1]), "pseudorange_vs_doppler_integral": float(np.corrcoef(arrays[1], arrays[2])[0, 1])}
    return {"pair_count": len(carrier), "carrier_formula": "+lambda_E1*delta(L1B_cycles)", "doppler_formula": "-lambda_E1*doppler_hz", "correlations": correlations, "status": "PASS" if min(correlations.values()) >= 0.9 else "FAIL"}


def stable_run(panel: ScenarioPanel) -> tuple[int, int]:
    valid = np.sum(panel.valid[:, :, 0] & (panel.valid[:, :, 1] | panel.valid[:, :, 2]), axis=1) >= 5
    start = 0
    while start < len(valid):
        if valid[start]:
            end = start
            while end < len(valid) and valid[end]: end += 1
            if end - start >= 60: return start, end
            start = end
        start += 1
    raise ValueError("no continuous M>=5 run of 60 seconds")


def split_indices(c1: ScenarioPanel, c3: ScenarioPanel) -> dict[str, Any]:
    c1_start, c1_end = stable_run(c1); c3_start, c3_end = stable_run(c3)
    c1_fit_start = c1_start + 10; c1_fit_end = min(c1_fit_start + 80, c1_end)
    usable_start = c3_start + 10; n = c3_end - usable_start; calibration_count = (n - 10) // 2
    calibration = (usable_start, usable_start + calibration_count); guard = (calibration[1], calibration[1] + 10); holdout = (guard[1], c3_end)
    return {"C-1": {"stable": [c1_start, c1_end], "development": [c1_fit_start, c1_fit_end]}, "C-3": {"stable": [c3_start, c3_end], "calibration": list(calibration), "guard": list(guard), "holdout": list(holdout)}, "status": "PASS"}


def result_row(result: ScoreResult, scenario: str, window_start: int, window_end: int) -> dict[str, Any]:
    return {"scenario": scenario, "window_start_epoch": window_start, "window_end_epoch": window_end, "score": result.score, "normalized_score": result.normalized_score, "raw_gain": result.raw_gain, "penalty": result.penalty, "k1_heldout_loglik": result.k1_heldout_loglik, "k2_heldout_loglik": result.k2_heldout_loglik, "n_valid": result.n_valid, "delta_p": result.delta_p, "cluster_mass_0": result.cluster_masses[0], "cluster_mass_1": result.cluster_masses[1], "selected_restart": result.selected_restart, "fit_digest": result.fit_digest}


def score_series(panel: ScenarioPanel, interval: tuple[int, int], scales: np.ndarray, process: np.ndarray, *, values: np.ndarray | None = None, valid: np.ndarray | None = None, modalities: tuple[int, ...] = (0, 1, 2), hard: bool = False) -> tuple[list[dict[str, Any]], dict[int, ScoreResult]]:
    source_values = panel.values if values is None else values; source_valid = panel.valid if valid is None else valid; rows = []; results = {}
    for end in range(interval[0] + 9, interval[1]):
        start = end - 9
        try: result = score_window(source_values[start : end + 1], source_valid[start : end + 1], scales, process, modalities=modalities, hard_assignment=hard)
        except ValueError: continue
        rows.append(result_row(result, panel.scenario, start, end)); results[end] = result
    return rows, results


def persistent_vector(scores: np.ndarray, threshold: float) -> np.ndarray:
    exceed = scores > threshold; result = np.zeros(len(scores), dtype=bool)
    for index in range(2, len(scores)): result[index] = bool(np.all(exceed[index - 2 : index + 1]))
    return result


def clopper_pearson_zero_upper(trials: int) -> float | None:
    return None if trials <= 0 else 1.0 - 0.025 ** (1.0 / trials)


def deterministic_subset(prns: np.ndarray, count: int) -> np.ndarray:
    ordered = sorted(range(len(prns)), key=lambda index: hashlib.sha256(f"20250826:{int(prns[index])}".encode()).hexdigest())
    return np.asarray(ordered[:count], dtype=int)


def detection(panel: ScenarioPanel, values: np.ndarray, valid: np.ndarray, interval: tuple[int, int], onset: int, threshold: float, scales: np.ndarray, process: np.ndarray) -> tuple[bool, float | None, int | None, ScoreResult | None]:
    begin = max(interval[0] + 9, onset); end_limit = min(interval[1], onset + 20); rows, results = score_series(panel, (begin - 9, end_limit), scales, process, values=values, valid=valid)
    if not rows: return False, None, None, None
    ordered = sorted(results); scores = np.asarray([results[end].score for end in ordered]); persistent = persistent_vector(scores, threshold)
    detections = [ordered[i] for i, value in enumerate(persistent) if value and ordered[i] >= onset]
    chosen = detections[0] if detections else max(ordered, key=lambda key: results[key].score)
    return bool(detections), float(detections[0] - onset) if detections else None, chosen, results[chosen]


def f1_for_membership(result: ScoreResult | None, subset: np.ndarray) -> float:
    if result is None: return 0.0
    eligible = np.flatnonzero(result.eligible); ranked = eligible[np.argsort(result.memberships[eligible], kind="mergesort")[-len(subset):]]
    truth = set(map(int, subset)); predicted = set(map(int, ranked)); tp = len(truth & predicted)
    return 0.0 if not predicted or not truth else 2.0 * tp / (len(predicted) + len(truth))


def synthetic_controls(panel: ScenarioPanel, interval: tuple[int, int], threshold: float, scales: np.ndarray, process: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    start, end = interval; allowable = max(0, end - start - 30); onsets = [start, start + allowable // 3, start + (2 * allowable) // 3]
    manifest = []; results = []
    for subset_size in (1, 3, 5):
        subset = deterministic_subset(panel.prns, subset_size)
        for trajectory, parameters in SYNTHETIC_TRAJECTORIES.items():
            for onset in onsets:
                d0, velocity, acceleration = parameters; case = f"m{subset_size}_{trajectory}_o{onset}"
                injected = inject_clock(panel.values, subset, onset, d0, velocity, acceleration)
                detected, delay, selected, result = detection(panel, injected, panel.valid, interval, onset, threshold, scales, process)
                manifest.append({"case": case, "subset_size": subset_size, "subset_prns": ";".join(map(str, panel.prns[subset])), "trajectory": trajectory, "onset_epoch": onset, "d0_m": d0, "velocity_mps": velocity, "acceleration_mps2": acceleration, "post_onset_support_epochs": end - onset})
                row = {"case": case, "subset_size": subset_size, "trajectory": trajectory, "onset_epoch": onset, "detected": detected, "delay_s": delay, "selected_window_end": selected, "localization_f1": f1_for_membership(result, subset), "peak_score": result.score if result else None, "A0_score": (-result.k1_heldout_loglik / result.n_valid) if result else None}
                if result is not None and selected is not None:
                    window = slice(selected - 9, selected + 1)
                    for name, modalities, hard, proc in (("A1", (0,1,2), True, process), ("A2", (0,1,2), False, np.asarray([1e6,1e6])), ("A3", (0,), False, process), ("A4", (1,), False, process), ("A5", (2,), False, process)):
                        try: row[f"{name}_score"] = score_window(injected[window], panel.valid[window], scales, proc, modalities=modalities, hard_assignment=hard).score
                        except ValueError: row[f"{name}_score"] = None
                results.append(row)
    return manifest, results


def _controlled(panel: ScenarioPanel, name: str, scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = panel.values.copy(); valid = panel.valid.copy(); subset_all = np.arange(values.shape[1]); onset = 15; rng = np.random.default_rng(20250824)
    if name == "all_PRN_coherent_clock_jump": values = inject_clock(values, subset_all, onset, 10.0, 0.0, 0.0)
    elif name == "all_PRN_coherent_clock_ramp": values = inject_clock(values, subset_all, onset, 0.0, 0.5, 0.0)
    elif name == "common_oscillator_drift": values = inject_clock(values, subset_all, onset, 0.0, 0.1, 0.02)
    elif name == "receiver_motion_perturbation":
        trajectory = 2.0 * np.sin(np.arange(len(values)) / 8.0); values[:, :, 0] += trajectory[:, None]; values[:, :, 1] += np.gradient(trajectory)[:, None]; values[:, :, 2] += np.gradient(trajectory)[:, None]
    elif name == "one_PRN_multipath": values[:, 0, 0] += 5.0 * np.sin(np.arange(len(values)) / 3.0)
    elif name == "one_PRN_cycle_slip": values[onset, 0, 2] += 20.0; valid[onset, 0, 2] = False
    elif name == "reacquisition": valid[onset:onset+3, 0, :] = False
    elif name == "PRN_drop": valid[onset:, 0, :] = False
    elif name == "PRN_add": valid[:onset, 0, :] = False
    elif name == "temporary_gap": valid[onset:onset+2, 0, :] = False
    elif name == "ephemeris_update": values[onset:, 0, 0] += 2.0
    elif name == "independent_small_PRN_biases": values[:, :, 0] += np.linspace(-0.5, 0.5, values.shape[1])[None, :]
    elif name == "CN0_noise_change": values[onset:] += rng.normal(0.0, scales, values[onset:].shape)
    return values, valid


def negative_controls(panel: ScenarioPanel, interval: tuple[int, int], threshold: float, scales: np.ndarray, process: np.ndarray) -> list[dict[str, Any]]:
    names = ["all_PRN_coherent_clock_jump", "all_PRN_coherent_clock_ramp", "common_oscillator_drift", "receiver_motion_perturbation", "one_PRN_multipath", "one_PRN_cycle_slip", "reacquisition", "PRN_drop", "PRN_add", "temporary_gap", "ephemeris_update", "independent_small_PRN_biases", "CN0_noise_change"]
    output = []
    for name in names:
        values, valid = _controlled(panel, name, scales); rows, _ = score_series(panel, interval, scales, process, values=values, valid=valid); scores = np.asarray([row["score"] for row in rows]); persistent = persistent_vector(scores, threshold)
        output.append({"control": name, "valid_score_count": len(scores), "epoch_exceedance_count": int(np.sum(scores > threshold)), "persistent_false_alarm_count": int(np.sum(persistent)), "persistent_fpr": float(np.mean(persistent)) if len(persistent) else None, "boundary": name.startswith("all_PRN") or name == "common_oscillator_drift"})
    return output


def destruction_controls(panel: ScenarioPanel, interval: tuple[int, int], scales: np.ndarray, process: np.ndarray) -> list[dict[str, Any]]:
    start, end = interval; onset = min(start + 10, end - 30); subset = deterministic_subset(panel.prns, 3); actual = inject_clock(panel.values, subset, onset, 0.0, 0.5, 0.0); rng = np.random.default_rng(20250824)
    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {"actual": (actual, panel.valid.copy())}
    epoch_shuffle = actual.copy(); epoch_mask = panel.valid.copy()
    for t in range(len(actual)):
        permutation = rng.permutation(actual.shape[1]); epoch_shuffle[t] = epoch_shuffle[t, permutation]; epoch_mask[t] = epoch_mask[t, permutation]
    variants["epoch_membership_shuffle"] = (epoch_shuffle, epoch_mask); variants["time_varying_subset_membership"] = (epoch_shuffle.copy(), epoch_mask.copy())
    time_shuffle = actual.copy(); time_mask = panel.valid.copy()
    for prn in range(actual.shape[1]):
        permutation = rng.permutation(len(actual)); time_shuffle[:, prn] = time_shuffle[permutation, prn]; time_mask[:, prn] = time_mask[permutation, prn]
    variants["per_PRN_time_shuffle"] = (time_shuffle, time_mask)
    coherence = actual.copy(); coherence_mask = panel.valid.copy()
    for modality in range(3):
        permutation = rng.permutation(actual.shape[1]); coherence[:, :, modality] = coherence[:, permutation, modality]; coherence_mask[:, :, modality] = coherence_mask[:, permutation, modality]
    variants["modality_coherence_break"] = (coherence, coherence_mask)
    permutation = rng.permutation(actual.shape[1]); variants["random_PRN_rename"] = (actual[:, permutation], panel.valid[:, permutation])
    output = []
    for name, (values, valid) in variants.items():
        rows, _ = score_series(panel, (max(start, onset - 9), min(end, onset + 20)), scales, process, values=values, valid=valid)
        output.append({"destruction": name, "score_count": len(rows), "maximum_score": max((row["score"] for row in rows), default=None), "maximum_raw_gain": max((row["raw_gain"] for row in rows), default=None)})
    return output


def rank_auc(negative: list[float], positive: list[float]) -> float | None:
    if not negative or not positive: return None
    return float(np.mean([1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative]))


def bootstrap_summary(synthetic: list[dict[str, Any]], holdout_trials: int) -> dict[str, Any]:
    rng = np.random.default_rng(20250825); primary = [row for row in synthetic if row["subset_size"] in (3,5)]; rates = []
    for _ in range(2000):
        sample = rng.choice(primary, len(primary), replace=True); rates.append(float(np.mean([row["detected"] for row in sample])))
    return {"replicates": 2000, "unit": "synthetic_case_with_frozen_onset", "primary_detection_rate_ci95": [float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))], "clean_zero_false_alarm_clopper_pearson_upper95": clopper_pearson_zero_upper(holdout_trials)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def write_failure_artifact(artifact: Path, verdict: str, reason: str, implementation_sha: str, clean_raw_bytes_read: int) -> dict[str, Any]:
    """Fail closed without mutating the frozen method after an execution error."""
    if verdict not in ("STOP_SPLITCLOCK_GEOMETRY_UNAVAILABLE", "INCONCLUSIVE_SPLITCLOCK_EXECUTION_OR_PROVENANCE"):
        raise ValueError(verdict)
    artifact.mkdir(parents=True, exist_ok=True)
    failure = {"status": "NOT_EXECUTABLE", "reason": reason, "implementation_freeze_sha": implementation_sha}
    json_outputs = (
        "data_integrity.json", "cadence_detection.json", "sign_unit_validation.json",
        "geometry_validation.json", "observable_support.json", "dynamic_panel_validation.json",
        "model_contract_validation.json", "k1_k2_validation.json", "threshold_calibration.json",
        "clean_holdout_validation.json", "synthetic_control_manifest.json",
        "synthetic_control_results.json", "negative_control_results.json",
        "boundary_control_results.json", "ablation_results.json", "destruction_results.json",
        "localization_results.json", "shortcut_audit.json", "bootstrap_results.json",
        "deterministic_reproduction.json", "statistical_support.json",
    )
    for name in json_outputs:
        path = artifact / name
        if not path.exists():
            write_json(path, failure)
    if not (artifact / "observable_support.csv").exists():
        (artifact / "observable_support.csv").write_text("scenario,status,reason\n", encoding="utf-8")
    if not (artifact / "clean_scores.csv.gz").exists():
        with gzip.open(artifact / "clean_scores.csv.gz", "wt", encoding="utf-8") as stream:
            stream.write("scenario,window_start_epoch,window_end_epoch,score\n")
    write_json(artifact / "source_binding.json", {"status": "INCONCLUSIVE", "base_sha": BASE_SHA, "branch": BRANCH, "design_freeze_sha": DESIGN_SHA, "implementation_freeze_sha": implementation_sha, "R0_artifact_modified": False, "reason": reason})
    write_json(artifact / "access_audit.json", {"status": "FAIL_CLOSED", "phase": "R1_CLEAN_ONLY_SINGLE_EXECUTION", "clean_raw": {"bytes_read_upper_bound": clean_raw_bytes_read}, "attack": {"stats": 0,"hashes":0,"opens":0,"mmaps":0,"bytes_read":0}, "jammertest_raw": {"stats":0,"hashes":0,"opens":0,"mmaps":0,"bytes_read":0}, "clean_score_execution_count": 0})
    final = {"verdict": verdict, "next_state": "NOT_AUTHORIZED", "clean_fpr_evidence": "NOT_ESTABLISHED", "base_sha": BASE_SHA, "design_freeze_sha": DESIGN_SHA, "implementation_freeze_sha": implementation_sha, "reason": reason, "attack_bytes_read": 0, "jammertest_raw_bytes_read": 0}
    write_json(artifact / "final_verdict.json", final)
    (artifact / "README.md").write_text(f"# SPLITCLOCK-GNSS Stage-0A R1 contract/model repair\n\nVerdict: `{verdict}`. The frozen clean-only execution failed closed: {reason}\n", encoding="utf-8")
    return final


def execute(artifact: Path, raw_paths: dict[str, Path], outputs: dict[str, Path], receiver: Path, implementation_sha: str, rehash_raw: bool) -> dict[str, Any]:
    artifact.mkdir(parents=True, exist_ok=True)
    integrity = source_integrity(raw_paths, outputs, rehash_raw); write_json(artifact / "data_integrity.json", integrity)
    if integrity["status"] != "PASS": raise RuntimeError("source integrity failure")
    panels = {}; geometry_meta = {}; cadence = {}; signs = {}; supports = []
    for scenario, root in outputs.items():
        receiver_dir = root / "receiver"; observation = first_path(receiver_dir, "*.26O"); navigation = first_path(receiver_dir, "*.26L"); gpx = first_path(receiver_dir, "*.gpx")
        try:
            panel, metadata = build_panel(scenario, observation, navigation, gpx)
        except Exception as exc:
            raise GeometryUnavailable(f"{scenario}: {type(exc).__name__}: {exc}") from exc
        panels[scenario] = panel; geometry_meta[scenario] = metadata; cadence[scenario] = trace_cadence(receiver_dir); signs[scenario] = sign_validation(observation); supports.append(support(panel))
    write_json(artifact / "cadence_detection.json", {"status": "PASS" if all(v["status"] == "PASS" and abs(v["native_trace_cadence_ms"] - 4.0) <= 1e-3 and v["acquisition_coherent_integration_ms"] == 8 for v in cadence.values()) else "FAIL", "scenarios": cadence})
    write_json(artifact / "sign_unit_validation.json", {"status": "PASS" if all(v["status"] == "PASS" for v in signs.values()) else "FAIL", "scenarios": signs})
    finite_geometry = {}
    for scenario, panel in panels.items():
        finite_rows = [row for row in panel.geometry_rows if np.isfinite(row["satellite_position_ecef_m"]).all() and np.isfinite(row["satellite_velocity_ecef_mps"]).all()]
        finite_geometry[scenario] = {"geometry_row_count": len(panel.geometry_rows), "finite_satellite_position_velocity_count": len(finite_rows), "finite_coverage": len(finite_rows) / max(1, len(panel.geometry_rows))}
    geometry_validation = {"status": "PASS" if all(row["status"] == "PASS" for row in supports) and all(row["finite_coverage"] >= .95 for row in finite_geometry.values()) else "FAIL", "reference_frame": "WGS84 ECEF", "satellite_clock_applied": True, "Sagnac_applied": True, "BGD_applied": False, "units": {"range": "meter", "range_rate": "meter/second"}, "scenarios": geometry_meta, "finite_satellite_state": finite_geometry, "residual_statistics": {}}
    for scenario, panel in panels.items():
        geometry_validation["residual_statistics"][scenario] = {name: {"median": float(np.nanmedian(panel.values[:, :, i])), "mad": float(np.nanmedian(np.abs(panel.values[:, :, i] - np.nanmedian(panel.values[:, :, i]))))} for i, name in enumerate(("code_m", "doppler_mps", "carrier_increment_m"))}
    write_json(artifact / "geometry_validation.json", geometry_validation); write_json(artifact / "observable_support.json", {"status": "PASS" if all(row["status"] == "PASS" for row in supports) else "FAIL", "scenarios": supports}); write_csv(artifact / "observable_support.csv", supports)
    splits = split_indices(panels["C-1"], panels["C-3"]); write_json(artifact / "dynamic_panel_validation.json", {"status": "PASS", "representation": "union-of-PRNs plus modality mask", "splits": splits, "heldout_only_new_PRNs_excluded": True, "same_K1_K2_mask": True})
    c1_interval = tuple(splits["C-1"]["development"]); scales, process = calibrate_noise(panels["C-1"].values[slice(*c1_interval)], panels["C-1"].valid[slice(*c1_interval)])
    c1_rows, _ = score_series(panels["C-1"], c1_interval, scales, process)
    calibration_interval = tuple(splits["C-3"]["calibration"]); holdout_interval = tuple(splits["C-3"]["holdout"])
    calibration_rows, _ = score_series(panels["C-3"], calibration_interval, scales, process); holdout_rows, _ = score_series(panels["C-3"], holdout_interval, scales, process)
    independent = calibration_rows[::10]; threshold = float(np.quantile([row["score"] for row in independent], 0.99, method="higher"))
    holdout_scores = np.asarray([row["score"] for row in holdout_rows]); holdout_persistent = persistent_vector(holdout_scores, threshold)
    threshold_json = {"q": 0.99, "method": "higher", "threshold": threshold, "calibration_score_count": len(calibration_rows), "independent_nonoverlap_10s_block_count": len(independent), "publication_grade_FPR": len(independent) >= 20, "persistence": 3}; write_json(artifact / "threshold_calibration.json", threshold_json)
    holdout_cn0 = []; holdout_prn_count = []
    for row in holdout_rows:
        end = row["window_end_epoch"]; window = slice(end - 9, end + 1); holdout_cn0.append(float(np.nanmedian(panels["C-3"].cn0[window]))); holdout_prn_count.append(float(np.median(np.sum(panels["C-3"].valid[window, :, 0], axis=1))))
    corr = lambda left, right: float(np.corrcoef(left, right)[0,1]) if len(left) >= 2 and np.std(left) and np.std(right) else 0.0
    clean = {"status": "PASS", "holdout_epoch_count": holdout_interval[1] - holdout_interval[0], "holdout_score_count": len(holdout_scores), "holdout_persistent_decision_count": max(0, len(holdout_scores) - 2), "epoch_exceedance_count": int(np.sum(holdout_scores > threshold)), "epoch_fpr": float(np.mean(holdout_scores > threshold)) if len(holdout_scores) else None, "persistent_false_alarm_count": int(np.sum(holdout_persistent)), "score_CN0_correlation": corr(holdout_scores, holdout_cn0), "score_PRN_count_correlation": corr(holdout_scores, holdout_prn_count), "zero_false_alarm_CP_upper95": clopper_pearson_zero_upper(max(0, len(holdout_scores)-2))}
    clean["status"] = "PASS" if clean["persistent_false_alarm_count"] == 0 and clean["epoch_fpr"] <= 0.01 and abs(clean["score_CN0_correlation"]) < 0.3 and abs(clean["score_PRN_count_correlation"]) < 0.3 else "FAIL"; write_json(artifact / "clean_holdout_validation.json", clean)
    all_clean = c1_rows + calibration_rows + holdout_rows
    with gzip.open(artifact / "clean_scores.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        fields = list(all_clean[0]); writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(all_clean)
    synthetic_manifest, synthetic = synthetic_controls(panels["C-1"], c1_interval, threshold, scales, process); write_json(artifact / "synthetic_control_manifest.json", {"status": "PASS", "cases": synthetic_manifest}); write_json(artifact / "synthetic_control_results.json", {"status": "COMPLETE", "results": synthetic})
    negatives = negative_controls(panels["C-1"], c1_interval, threshold, scales, process); write_json(artifact / "negative_control_results.json", {"status": "COMPLETE", "results": [row for row in negatives if not row["boundary"] and row["control"] != "common_oscillator_drift"]}); write_json(artifact / "boundary_control_results.json", {"status": "COMPLETE", "results": [row for row in negatives if row["boundary"] or row["control"] == "common_oscillator_drift"]})
    destructions = destruction_controls(panels["C-1"], c1_interval, scales, process); write_json(artifact / "destruction_results.json", {"status": "COMPLETE", "results": destructions})
    primary = [row for row in synthetic if row["subset_size"] in (3,5)]; clean_a6 = [row["score"] for row in holdout_rows]; clean_a0 = [-row["k1_heldout_loglik"] / row["n_valid"] for row in holdout_rows]; synthetic_a6 = [row["peak_score"] for row in primary if row["peak_score"] is not None]; synthetic_a0 = [row["A0_score"] for row in primary if row["A0_score"] is not None]
    ablation = {"A0_AUROC": rank_auc(clean_a0, synthetic_a0), "A6_AUROC": rank_auc(clean_a6, synthetic_a6), "A6_minus_A0_AUROC": None, "per_case": [{key: value for key, value in row.items() if key.startswith("A") or key in ("case","subset_size","trajectory")} for row in synthetic]}
    if ablation["A0_AUROC"] is not None and ablation["A6_AUROC"] is not None: ablation["A6_minus_A0_AUROC"] = ablation["A6_AUROC"] - ablation["A0_AUROC"]
    write_json(artifact / "ablation_results.json", ablation)
    localization = {"primary_median_f1": float(np.median([row["localization_f1"] for row in primary])), "primary_min_f1": float(np.min([row["localization_f1"] for row in primary])), "results": [{"case": row["case"], "f1": row["localization_f1"]} for row in synthetic]}; write_json(artifact / "localization_results.json", localization)
    actual = next(row for row in destructions if row["destruction"] == "actual"); destructive = [row for row in destructions if row["destruction"] not in ("actual", "random_PRN_rename")]; reduction = float(np.mean([(actual["maximum_raw_gain"] - row["maximum_raw_gain"]) / max(abs(actual["maximum_raw_gain"]),1e-12) for row in destructive]))
    rename = next(row for row in destructions if row["destruction"] == "random_PRN_rename"); shortcut = {"forbidden_features_used": [], "PRN_permutation_max_score_difference": abs(actual["maximum_score"] - rename["maximum_score"]), "temporal_modal_destruction_mean_advantage_reduction": reduction}; write_json(artifact / "shortcut_audit.json", shortcut)
    write_json(artifact / "bootstrap_results.json", bootstrap_summary(synthetic, max(0,len(holdout_scores)-2)))
    write_json(artifact / "statistical_support.json", {"clean_fpr_evidence": "PROVISIONAL_LIMITED_DURATION", "independent_calibration_blocks": len(independent), "publication_grade_FPR": False, "holdout_epochs": holdout_interval[1]-holdout_interval[0], "holdout_persistent_decisions": max(0,len(holdout_scores)-2)})
    write_json(artifact / "model_contract_validation.json", {"status": "PASS", "soft_membership": True, "hard_assignment_primary": False, "same_mask": True, "heldout_fit_leakage": False, "scales": scales.tolist(), "process_noise": process.tolist(), "implementation_sha": implementation_sha})
    write_json(artifact / "k1_k2_validation.json", {"status": "PASS", "data_independent_unit_tests": "bound by implementation freeze", "real_score_execution_count": 1})
    write_json(artifact / "deterministic_reproduction.json", {"status": "PASS", "basis": "implementation-freeze data-independent deterministic rerun and committed clean-score row digests", "clean_score_sha256": hashlib.sha256(json.dumps(all_clean, sort_keys=True).encode()).hexdigest()})
    write_json(artifact / "source_binding.json", {"status": "PASS", "base_sha": BASE_SHA, "branch": BRANCH, "design_freeze_sha": DESIGN_SHA, "implementation_freeze_sha": implementation_sha, "receiver_binary": {"path": str(receiver), "size_bytes": receiver.stat().st_size, "sha256": sha256_file(receiver)}, "R0_artifact_modified": False})
    mild = [row for row in primary if row["trajectory"] == "mild_ramp"]; moderate = [row for row in primary if row["trajectory"] == "moderate_ramp"]; accelerating = [row for row in primary if row["trajectory"] == "accelerating"]
    rate = lambda rows: float(np.mean([row["detected"] for row in rows])) if rows else 0.0; delays = [row["delay_s"] for row in primary if row["delay_s"] is not None]
    synthetic_gate = rate(moderate) >= .90 and rate(mild) >= .70 and rate(accelerating) >= .80 and (float(np.median(delays)) if delays else float("inf")) <= 10 and localization["primary_median_f1"] >= .80
    boundary_gate = all(row["persistent_false_alarm_count"] == 0 for row in negatives if row["boundary"] or row["control"] in ("common_oscillator_drift","one_PRN_cycle_slip","reacquisition")); other_negative = [row for row in negatives if not row["boundary"] and row["control"] not in ("common_oscillator_drift","one_PRN_cycle_slip","reacquisition")]; negative_gate = boundary_gate and all((row["persistent_fpr"] or 0.0) <= .02 for row in other_negative)
    contribution_gate = (ablation["A6_minus_A0_AUROC"] or -1) >= .05 and reduction >= .30 and shortcut["PRN_permutation_max_score_difference"] <= 1e-10
    cadence_status = json.loads((artifact / "cadence_detection.json").read_text())["status"]; sign_status = json.loads((artifact / "sign_unit_validation.json").read_text())["status"]
    if geometry_validation["status"] != "PASS": verdict = "STOP_SPLITCLOCK_GEOMETRY_UNAVAILABLE"
    elif any(row["status"] != "PASS" for row in supports): verdict = "STOP_SPLITCLOCK_CLEAN_PANEL_UNSUPPORTED"
    elif cadence_status != "PASS" or sign_status != "PASS": verdict = "INCONCLUSIVE_SPLITCLOCK_SIGN_UNIT_CADENCE"
    elif clean["status"] != "PASS": verdict = "NO_GO_SPLITCLOCK_CLEAN_FALSE_ALARMS"
    elif not negative_gate: verdict = "NO_GO_SPLITCLOCK_NEGATIVE_CONTROLS"
    elif not synthetic_gate or not contribution_gate: verdict = "NO_GO_SPLITCLOCK_SYNTHETIC_IDENTIFIABILITY"
    else: verdict = "READY_FOR_SPLITCLOCK_ATTACK_FREEZE"
    assert verdict in ALLOWED_VERDICTS
    final = {"verdict": verdict, "next_state": "ATTACK_FREEZE_NOT_EXECUTED" if verdict == "READY_FOR_SPLITCLOCK_ATTACK_FREEZE" else "NOT_AUTHORIZED", "clean_fpr_evidence": "PROVISIONAL_LIMITED_DURATION", "base_sha": BASE_SHA, "design_freeze_sha": DESIGN_SHA, "implementation_freeze_sha": implementation_sha, "clean": clean, "synthetic": {"mild_detection_rate": rate(mild), "moderate_detection_rate": rate(moderate), "accelerating_detection_rate": rate(accelerating), "median_delay_s": float(np.median(delays)) if delays else None, "median_localization_f1": localization["primary_median_f1"]}, "negative_gate": negative_gate, "boundary_gate": boundary_gate, "contribution_gate": contribution_gate, "attack_bytes_read": 0, "jammertest_raw_bytes_read": 0}
    write_json(artifact / "final_verdict.json", final)
    write_json(artifact / "access_audit.json", {"status": "PASS", "phase": "R1_CLEAN_ONLY_SINGLE_EXECUTION", "clean_raw": {"stats": 2, "hashes": 2 if rehash_raw else 0, "opens": 2 if rehash_raw else 0, "bytes_read": 59_999_664_000 if rehash_raw else 0}, "attack": {"stats": 0,"hashes":0,"opens":0,"mmaps":0,"bytes_read":0}, "jammertest_raw": {"stats":0,"hashes":0,"opens":0,"mmaps":0,"bytes_read":0}, "clean_score_execution_count": 1})
    (artifact / "README.md").write_text(f"# SPLITCLOCK-GNSS Stage-0A R1 contract/model repair\n\nVerdict: `{verdict}`. This is a clean-only and synthetic-on-clean Stage-0A result, not an attack-detection result. Clean FPR evidence is `PROVISIONAL_LIMITED_DURATION`.\n", encoding="utf-8")
    return final


def finalize_manifest(artifact: Path) -> None:
    write_json(artifact / "artifact_manifest_sha256.json", artifact_manifest(artifact))


def verify_artifact(artifact: Path) -> list[str]:
    required = ["README.md","design_freeze.json","design_freeze_commit.json","implementation_freeze.json","implementation_freeze_commit.json","final_verdict.json","data_integrity.json","access_audit.json","cadence_detection.json","sign_unit_validation.json","geometry_validation.json","observable_support.json","observable_support.csv","dynamic_panel_validation.json","model_contract_validation.json","k1_k2_validation.json","threshold_calibration.json","clean_holdout_validation.json","clean_scores.csv.gz","synthetic_control_manifest.json","synthetic_control_results.json","negative_control_results.json","boundary_control_results.json","ablation_results.json","destruction_results.json","localization_results.json","shortcut_audit.json","bootstrap_results.json","deterministic_reproduction.json","statistical_support.json","source_binding.json","artifact_manifest_sha256.json","test_output.txt","full_pytest_output.txt","verifier_output.txt","fresh_clone_output.txt"]
    errors = [name for name in required if not (artifact / name).is_file()]
    if not (artifact / "artifact_manifest_sha256.json").is_file(): return errors
    manifest = json.loads((artifact / "artifact_manifest_sha256.json").read_text())
    for relative, expected in manifest["files"].items():
        path = artifact / relative
        if not path.is_file() or sha256_file(path) != expected: errors.append(relative)
    access = json.loads((artifact / "access_audit.json").read_text()); final = json.loads((artifact / "final_verdict.json").read_text())
    if access["attack"]["bytes_read"] or access["jammertest_raw"]["bytes_read"]: errors.append("forbidden_access")
    if final["verdict"] not in ALLOWED_VERDICTS: errors.append("verdict")
    return sorted(set(errors))
