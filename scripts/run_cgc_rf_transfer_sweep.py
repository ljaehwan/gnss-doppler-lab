#!/usr/bin/env python3
"""Map the fixed-receiver CGC transfer curve over 20--240 m."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_cgc_rf_challenge_pilot as pilot  # noqa: E402
import run_cgc_rf_observability_anchors as anchor  # noqa: E402
import run_simulation_v4_paired_train_generation as source  # noqa: E402
from gnss_doppler_lab.peak_mixture_law import parse_gps_sdr_sim_los_table  # noqa: E402
from gnss_doppler_lab.simulation_v4 import SimulationScenario, compose_paired_iq  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/cgc_rf_transfer_sweep_v1.json"
PROTOCOL_PATH = REPO_ROOT / "docs/results/cgc_rf_transfer_sweep_protocol_v1.md"
RELEASE_TOKEN = "RELEASE-CGC-RF-TRANSFER-SWEEP-V1"
RELEASE_INPUTS = (
    "configs/experiments/cgc_rf_transfer_sweep_v1.json",
    "docs/results/cgc_rf_transfer_sweep_protocol_v1.md",
    "scripts/run_cgc_rf_transfer_sweep.py",
)
DISTANCES_M = (20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 160.0, 200.0, 240.0)
POWERS_DB = (-6.0, 3.0)


def sha256(path: str | Path) -> str:
    return anchor.sha256(path)


def repo_path(value: str | Path) -> Path:
    return anchor.repo_path(value)


def write_json(path: Path, document: dict[str, Any]) -> None:
    anchor.write_json(path, document)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    anchor.write_csv(path, rows)


def power_label(power_db: float) -> str:
    mapping = {-6.0: "pneg6", 3.0: "ppos3"}
    if float(power_db) not in mapping:
        raise ValueError(f"unsupported power: {power_db}")
    return mapping[float(power_db)]


def condition_id(power_db: float, distance_m: float) -> str:
    return f"{power_label(power_db)}-d{int(distance_m):03d}"


def condition_specs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for distance in DISTANCES_M:
        for power in POWERS_DB:
            rows.append({
                "condition_id": condition_id(power, distance),
                "distance_m": distance,
                "final_advantage_db": power,
                "target_offset_enu_m": [0.8 * distance, 0.6 * distance, 0.0],
                "transition_seconds": distance / 20.0,
            })
    return rows


def contiguous_intervals(values: list[float], selected: list[bool]) -> list[list[float]]:
    if len(values) != len(selected):
        raise ValueError("value and selection lengths differ")
    intervals: list[list[float]] = []
    start: float | None = None
    previous: float | None = None
    for value, keep in zip(values, selected):
        if keep and start is None:
            start = value
        if not keep and start is not None:
            assert previous is not None
            intervals.append([start, previous])
            start = None
        previous = value
    if start is not None:
        assert previous is not None
        intervals.append([start, previous])
    return intervals


def evaluate_curve(
    rows: list[dict[str, Any]], *, threshold: float, minimum_bins: int
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["distance_m"]))
    distances = [float(row["distance_m"]) for row in ordered]
    if distances != list(DISTANCES_M):
        raise ValueError("curve requires the complete frozen distance grid")
    aucs = [float(row["serial_bin_auc"]) for row in ordered]
    above = [auc >= threshold for auc in aucs]
    peak_index = int(np.argmax(aucs))
    correlation = float(spearmanr(distances, aucs).statistic)
    return {
        "auc_by_distance_m": {str(int(d)): auc for d, auc in zip(distances, aucs)},
        "peak_auc": aucs[peak_index],
        "peak_distance_m": distances[peak_index],
        "first_tested_threshold_crossing_m": next((d for d, flag in zip(distances, above) if flag), None),
        "above_threshold_grid_intervals_m": contiguous_intervals(distances, above),
        "strictly_increasing_over_full_grid": all(left < right for left, right in zip(aucs, aucs[1:])),
        "post_peak_decline_at_final_cell": peak_index < len(aucs) - 1 and aucs[-1] < aucs[peak_index],
        "final_minus_peak_auc": aucs[-1] - aucs[peak_index],
        "spearman_distance_auc": correlation,
        "minimum_comparison_bins_passed": all(
            int(row["spoof_bin_count"]) >= minimum_bins
            and int(row["multipath_bin_count"]) >= minimum_bins
            for row in ordered
        ),
        "template_edge_fraction_by_distance_m": {
            str(int(row["distance_m"])): float(row["template_delay_edge_fraction"])
            for row in ordered
        },
    }


def verify_record(record: dict[str, str], label: str) -> Path:
    return anchor.verify_record(record, label)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "gnss-doppler-lab.cgc-rf-transfer-sweep-config" or config.get("schema_version") != 1:
        raise ValueError("unsupported RF transfer sweep config")
    if config.get("experiment", {}).get("name") != "cgc-rf-transfer-sweep-v1":
        raise ValueError("experiment identity drifted")
    paths: dict[str, Path] = {}
    for key in ("prior_negative_result", "base_pair_config", "normal_profile", "controlled_template"):
        paths[key] = verify_record(config[key], key)
    for key, record in config["pinned_code"].items():
        paths[key] = verify_record(record, key)
    for key, record in config["shared_inputs"].items():
        paths[key] = verify_record(record, key)
    for key in ("simulator", "receiver", "receiver_patch"):
        paths[key] = verify_record(config["rf_tools"][key], key)
    prior = json.loads(paths["prior_negative_result"].read_text(encoding="utf-8"))
    if prior.get("decision") != "not_reproduced" or prior.get("profile_boundary_reproduced_in_receiver_rf") is not False:
        raise ValueError("prior negative result drifted")
    fresh = json.loads(paths["base_pair_config"].read_text(encoding="utf-8"))
    matches = [row for row in fresh["pairs"] if row["paired_group_id"] == config["base_pair_config"]["pair_id"]]
    if len(matches) != 1 or matches[0]["domain"] != "static":
        raise ValueError("base static pair drifted")
    base_pair = matches[0]
    if base_pair["duration_seconds"] != 30 or base_pair["receiver_seed"] != 20261001:
        raise ValueError("base duration or receiver seed drifted")
    normal_profile = json.loads(paths["normal_profile"].read_text(encoding="utf-8"))
    controlled = json.loads(paths["controlled_template"].read_text(encoding="utf-8"))
    normal_manifest = json.loads(paths["normal_rf_manifest"].read_text(encoding="utf-8"))
    multipath_receiver = json.loads(paths["multipath_receiver_manifest"].read_text(encoding="utf-8"))
    if int(normal_profile["rf_profile"]["rf_sample_rate_hz"]) != 25_000_000:
        raise ValueError("normal profile is not 25 MHz")
    if normal_manifest["iq"]["sha256"] != config["shared_inputs"]["normal_rf_iq"]["sha256"]:
        raise ValueError("normal RF pins disagree")
    if multipath_receiver["source"]["iq_sha256"] != config["shared_inputs"]["multipath_rf_iq"]["sha256"]:
        raise ValueError("multipath receiver and RF pins disagree")
    tools = config["rf_tools"]
    if (
        tools["channel_count"] != 11
        or tools["tracking_tap_count"] != 9
        or tools["tracking_tap_spacing_chips"] != 0.125
        or tools["tracking_half_aperture_chips"] != 0.5
        or multipath_receiver["tracking"]["tap_count"] != 9
        or multipath_receiver["tracking"]["tap_spacing_chips"] != 0.125
    ):
        raise ValueError("fixed receiver aperture drifted")
    if controlled["correlator"]["tap_offsets_chips"] != [-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5]:
        raise ValueError("template tap aperture drifted")
    sweep = config["sweep"]
    if (
        tuple(float(value) for value in sweep["distances_m"]) != DISTANCES_M
        or tuple(float(value) for value in sweep["final_advantages_db"]) != POWERS_DB
        or sweep["target_direction_enu_unit"] != [0.8, 0.6, 0.0]
        or sweep["start_seconds"] != 5.0
        or sweep["pull_off_rate_mps"] != 20.0
        or sweep["power_ramp_seconds"] != 5.0
        or sweep["latest_settle_seconds"] != 17.0
        or sweep["comparison_start_seconds"] != 18.0
    ):
        raise ValueError("distance, speed, power, or interval grid drifted")
    if not math.isclose(float(sweep["chip_length_m"]), 299_792_458.0 / 1_023_000.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("chip length drifted")
    for spec in condition_specs():
        if not math.isclose(spec["transition_seconds"], spec["distance_m"] / sweep["pull_off_rate_mps"]):
            raise ValueError("pull-off rate drifted")
        if sweep["start_seconds"] + spec["transition_seconds"] > sweep["latest_settle_seconds"]:
            raise ValueError("condition settles after frozen latest time")
    analysis = config["analysis"]
    if analysis["minimum_prns"] != 8 or analysis["minimum_comparison_bins_per_stream"] != 8 or analysis["descriptive_auc_threshold"] != 0.8:
        raise ValueError("analysis contract drifted")
    if config["retention"]["shared_inputs_removed"] is not False:
        raise ValueError("shared input deletion must remain forbidden")
    output_root = repo_path(config["output_root"])
    if output_root != REPO_ROOT / "artifacts/cgc_rf_transfer_sweep_v1":
        raise ValueError("output root drifted")
    return {
        "paths": paths,
        "base_pair": base_pair,
        "normal_profile": normal_profile,
        "controlled": controlled,
        "normal_manifest": normal_manifest,
        "output_root": output_root,
    }


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def committed_release() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        git("ls-files", "--error-unmatch", relative)
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=REPO_ROOT).returncode:
            raise ValueError(f"release input is not committed and clean: {relative}")
    return {
        "head_commit": git("rev-parse", "HEAD"),
        "input_commits": {relative: git("log", "-1", "--format=%H", "--", relative) for relative in RELEASE_INPUTS},
        "runner_sha256": sha256(Path(__file__).resolve()),
    }


def component_pair(base_pair: dict[str, Any], distance_m: float, config: dict[str, Any]) -> dict[str, Any]:
    pair = deepcopy(base_pair)
    sweep = config["sweep"]
    pair["paired_group_id"] = f"transfer-d{int(distance_m):03d}-source"
    pair["split"] = "rf_transfer_sweep"
    pair["spoofing"] = {
        "start_seconds": float(sweep["start_seconds"]),
        "transition_seconds": float(distance_m / sweep["pull_off_rate_mps"]),
        "target_offset_enu_m": [0.8 * distance_m, 0.6 * distance_m, 0.0],
        "initial_advantage_db": 0.0,
        "final_advantage_db": 0.0,
        "power_ramp_seconds": 0.0,
    }
    return pair


def scenario_pair(base_pair: dict[str, Any], spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    pair = deepcopy(base_pair)
    sweep = config["sweep"]
    pair["paired_group_id"] = spec["condition_id"]
    pair["split"] = "rf_transfer_sweep"
    pair["spoofing"] = {
        "start_seconds": float(sweep["start_seconds"]),
        "transition_seconds": float(spec["transition_seconds"]),
        "target_offset_enu_m": list(spec["target_offset_enu_m"]),
        "initial_advantage_db": float(sweep["initial_advantage_db"]),
        "final_advantage_db": float(spec["final_advantage_db"]),
        "power_ramp_seconds": float(sweep["power_ramp_seconds"]),
    }
    return pair


def ensure_spoof_rf(
    root: Path, pair: dict[str, Any], config: dict[str, Any], context: dict[str, Any],
    counterfeit_path: Path, counterfeit_manifest_path: Path, counterfeit_manifest: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    rf_root = root / "rf"
    iq_path = rf_root / "gps_l1ca_s8_iq.bin"
    manifest_path = rf_root / "manifest.json"
    if manifest_path.is_file():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not iq_path.is_file() or sha256(iq_path) != document["iq"]["sha256"]:
            raise ValueError("composed sweep RF integrity failure")
        return iq_path, manifest_path, document
    if rf_root.exists() and any(rf_root.iterdir()):
        raise FileExistsError(f"partial sweep RF: {rf_root}")
    rf_root.mkdir(parents=True, exist_ok=True)
    event = source._spoof_event(pair)
    scenario = SimulationScenario(pair["paired_group_id"], "carryoff_spoof", spoofing=event)
    receiver = source.normal._run_impairment(context["normal_profile"], pair)
    composition = compose_paired_iq(
        context["paths"]["authentic_component"], counterfeit_path,
        {scenario.name: iq_path}, (scenario,),
        sample_rate_hz=int(context["normal_profile"]["rf_profile"]["rf_sample_rate_hz"]),
        receiver=receiver,
        normal_target_rms=float(context["normal_profile"]["normal_target_rms"]),
        reference_override=context["normal_manifest"]["simulation_v4"]["receiver"]["reference"],
    )
    report = composition["scenarios"][scenario.name]
    sample_rate = int(context["normal_profile"]["rf_profile"]["rf_sample_rate_hz"])
    prefix = source.compare_prefix(
        context["paths"]["normal_rf_iq"], iq_path,
        int(round(event.start_seconds * sample_rate)),
    )
    if prefix.get("byte_identical") is not True:
        raise RuntimeError("sweep pre-onset prefix differs from pinned normal RF")
    truth = {"class": "spoofing", "event": "constant_rate_carryoff", "is_spoofing": True, "spoofing": asdict(event)}
    document = {
        "schema_version": 4,
        "run_id": f"cgc-rf-transfer-{pair['paired_group_id']}",
        "scenario": {
            "name": scenario.name, "campaign": config["experiment"]["name"],
            "paired_group_id": pair["paired_group_id"], "split": "rf_transfer_sweep",
            "utc": pair["utc"], "duration_seconds": pair["duration_seconds"],
            "position": pair["position"], "motion": None, "domain": "static", **truth,
        },
        "iq": {
            "path": iq_path.name, "sha256": report["sha256"], "actual_bytes": report["bytes"],
            "complex_samples": report["complex_samples"], "actual_duration_seconds": report["actual_duration_seconds"],
            "rf_sample_rate_hz": sample_rate, "sample_format": "s8_iq", "channels": 2,
        },
        "simulation_v4": {
            "truth": truth,
            "pair_contract": {"reference_member": "pinned-normal", "paired_prefix_check": prefix},
            "receiver": {"requested": receiver.manifest(), "reference": composition["reference"], "processing": composition["processing"]},
            "measurements": report,
            "sources": {"authentic": config["shared_inputs"]["authentic_component"], "counterfeit": counterfeit_manifest["counterfeit"]},
            "scope": "offline exploratory receiver transfer sweep; no transmission",
        },
        "generation": {
            "config_sha256": sha256(DEFAULT_CONFIG),
            "counterfeit_manifest": str(counterfeit_manifest_path.resolve()),
            "counterfeit_manifest_sha256": sha256(counterfeit_manifest_path),
        },
    }
    write_json(manifest_path, document)
    return iq_path, manifest_path, document


def condition_paths(root: Path, condition: str) -> dict[str, Path]:
    base = root / "conditions" / condition
    return {
        "root": base,
        "spoof_iq": base / "rf/gps_l1ca_s8_iq.bin",
        "result": base / "condition_result.json",
        "retention": base / "retention.json",
    }


def distance_paths(root: Path, distance_m: float) -> dict[str, Path]:
    base = root / "distances" / f"d{int(distance_m):03d}"
    return {
        "root": base,
        "counterfeit_iq": base / "component/counterfeit_gps_l1ca_s8_iq.bin",
        "component_manifest": base / "component/manifest.json",
        "retention": base / "retention.json",
    }


def remove_single_iq(path: Path, identity: dict[str, Any], retention: Path, kind: str) -> dict[str, Any]:
    if retention.is_file() and not path.exists():
        return json.loads(retention.read_text(encoding="utf-8"))
    if not path.is_file():
        raise FileNotFoundError(f"intermediate disappeared before retention record: {path}")
    if sha256(path) != identity["sha256"] or path.stat().st_size != int(identity["bytes"]):
        raise ValueError(f"refusing to remove changed intermediate: {path}")
    path.unlink()
    document = {
        "schema": "gnss-doppler-lab.cgc-rf-transfer-retention",
        "schema_version": 1,
        "removed_intermediate": {"kind": kind, **identity, "removed": True},
        "shared_inputs_removed": False,
        "receiver_outputs_retained": True,
        "deterministic_regeneration_inputs_retained": True,
    }
    write_json(retention, document)
    return document


def summarize_condition(
    spec: dict[str, Any], spoof_delays: list[dict[str, Any]], spoof_geometry: list[dict[str, Any]],
    multipath_geometry: list[dict[str, Any]], config: dict[str, Any],
) -> dict[str, Any]:
    start = float(config["sweep"]["comparison_start_seconds"])
    spoof = [row for row in spoof_geometry if float(row["bin_start_s"]) >= start]
    multipath = [row for row in multipath_geometry if float(row["bin_start_s"]) >= start]
    delays = [row for row in spoof_delays if float(row["bin_start_s"]) >= start]
    if not spoof or not multipath or not delays:
        raise ValueError("empty sweep comparison stream")
    labels = np.r_[np.ones(len(spoof), dtype=int), np.zeros(len(multipath), dtype=int)]
    scores = -np.asarray([row["clock_centered_geometry_residual"] for row in spoof + multipath], dtype=float)
    theta = np.asarray([
        [row["estimated_displacement_e_chips"], row["estimated_displacement_n_chips"], row["estimated_displacement_u_chips"]]
        for row in spoof
    ], dtype=float)
    norms = np.linalg.norm(theta, axis=1)
    direction = np.asarray(config["sweep"]["target_direction_enu_unit"], dtype=float)
    valid = norms > 1e-12
    direction_cosines = np.abs(theta[valid] @ direction / norms[valid])
    delay_values = np.asarray([row["estimated_delay_chips"] for row in delays], dtype=float)
    template_distances = np.asarray([row["median_template_distance"] for row in delays], dtype=float)
    edge = float(config["analysis"]["template_delay_edge_absolute_chips"])
    spoof_residual = np.asarray([row["clock_centered_geometry_residual"] for row in spoof], dtype=float)
    multipath_residual = np.asarray([row["clock_centered_geometry_residual"] for row in multipath], dtype=float)
    return {
        **spec,
        "distance_chips": float(spec["distance_m"] / config["sweep"]["chip_length_m"]),
        "pull_off_rate_mps": float(config["sweep"]["pull_off_rate_mps"]),
        "comparison_start_seconds": start,
        "spoof_bin_count": len(spoof),
        "multipath_bin_count": len(multipath),
        "minimum_spoof_prn_count": min(int(row["prn_count"]) for row in spoof),
        "minimum_multipath_prn_count": min(int(row["prn_count"]) for row in multipath),
        "serial_bin_auc": float(roc_auc_score(labels, scores)),
        "spoof_median_clock_centered_residual": float(np.median(spoof_residual)),
        "multipath_median_clock_centered_residual": float(np.median(multipath_residual)),
        "multipath_minus_spoof_median_residual": float(np.median(multipath_residual) - np.median(spoof_residual)),
        "median_estimated_displacement_norm_chips": float(np.median(norms)),
        "median_absolute_direction_cosine": float(np.median(direction_cosines)) if len(direction_cosines) else 0.0,
        "template_delay_edge_fraction": float(np.mean(np.abs(delay_values) >= edge)),
        "median_template_distance": float(np.median(template_distances)),
        "delay_estimate_count": len(delays),
    }


def run_condition(
    spec: dict[str, Any], config: dict[str, Any], context: dict[str, Any], estimator: Any,
    los: dict[str, tuple[float, float, float]], multipath_geometry: list[dict[str, Any]],
    counterfeit: Path, counterfeit_manifest_path: Path, counterfeit_manifest: dict[str, Any],
) -> dict[str, Any]:
    paths = condition_paths(context["output_root"], spec["condition_id"])
    if paths["result"].is_file():
        result = json.loads(paths["result"].read_text(encoding="utf-8"))
        remove_single_iq(paths["spoof_iq"], result["intermediate_spoof_iq"], paths["retention"], "composed_spoof_iq")
        return result
    pair = scenario_pair(context["base_pair"], spec, config)
    print(f"[transfer] {spec['condition_id']} composing paired spoof RF", flush=True)
    spoof_iq, spoof_manifest_path, spoof_manifest = ensure_spoof_rf(
        paths["root"], pair, config, context, counterfeit,
        counterfeit_manifest_path, counterfeit_manifest,
    )
    receiver_config = {
        "executable": str(context["paths"]["receiver"]),
        "channel_count": config["rf_tools"]["channel_count"],
        "timeout_seconds": config["rf_tools"]["timeout_seconds"],
        "tracking_tap_spacing_chips": config["rf_tools"]["tracking_tap_spacing_chips"],
    }
    expected_receiver = paths["root"] / "receiver" / spoof_manifest["run_id"] / "manifest.json"
    print(f"[transfer] {spec['condition_id']} running fixed GNSS-SDR", flush=True)
    receiver_manifest = pilot._ensure_receiver(
        spoof_manifest_path, paths["root"] / "receiver", receiver_config,
        resume=expected_receiver.is_file(),
    )
    print(f"[transfer] {spec['condition_id']} scoring transfer diagnostics", flush=True)
    spoof_delays, spoof_geometry = anchor.analyze_stream(
        spec["condition_id"], receiver_manifest, estimator, los, config
    )
    enriched_delays = [{"condition_id": spec["condition_id"], **row} for row in spoof_delays]
    enriched_geometry = [{"condition_id": spec["condition_id"], **row} for row in spoof_geometry]
    delay_path = paths["root"] / "spoof_delay_estimates.csv"
    geometry_path = paths["root"] / "spoof_geometry_scores.csv"
    write_csv(delay_path, enriched_delays)
    write_csv(geometry_path, enriched_geometry)
    summary = summarize_condition(spec, spoof_delays, spoof_geometry, multipath_geometry, config)
    identity = {"path": str(spoof_iq.resolve()), "sha256": spoof_manifest["iq"]["sha256"], "bytes": spoof_manifest["iq"]["actual_bytes"]}
    result = {
        "schema": "gnss-doppler-lab.cgc-rf-transfer-condition-result",
        "schema_version": 1,
        "condition": spec,
        "pair": pair,
        "summary": summary,
        "prefix": spoof_manifest["simulation_v4"]["pair_contract"]["paired_prefix_check"],
        "counterfeit_manifest": {"path": str(counterfeit_manifest_path.resolve()), "sha256": sha256(counterfeit_manifest_path)},
        "spoof_rf_manifest": {"path": str(spoof_manifest_path.resolve()), "sha256": sha256(spoof_manifest_path)},
        "receiver_manifest": {"path": str(receiver_manifest.resolve()), "sha256": sha256(receiver_manifest)},
        "intermediate_spoof_iq": identity,
        "score_artifacts": {
            "delays": {"path": str(delay_path.resolve()), "sha256": sha256(delay_path), "row_count": len(enriched_delays)},
            "geometry": {"path": str(geometry_path.resolve()), "sha256": sha256(geometry_path), "row_count": len(enriched_geometry)},
        },
    }
    write_json(paths["result"], result)
    remove_single_iq(paths["spoof_iq"], identity, paths["retention"], "composed_spoof_iq")
    print(f"[transfer] {spec['condition_id']} complete; composed IQ removed", flush=True)
    return result


def start_release(config_path: Path, context: dict[str, Any], resume: bool) -> tuple[Path, dict[str, Any]]:
    root = context["output_root"]
    state_path = root / "release_state.json"
    commits = committed_release()
    if not resume:
        if root.exists():
            raise FileExistsError(root)
        root.mkdir(parents=True)
        state = {
            "schema": "gnss-doppler-lab.cgc-rf-transfer-release-state",
            "schema_version": 1,
            "phase": "released_before_transfer_outcomes",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)},
            "commits": commits,
            "condition_ids": [row["condition_id"] for row in condition_specs()],
            "metrics_emitted": False,
        }
        write_json(state_path, state)
        return state_path, state
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["config"]["sha256"] != sha256(config_path) or state["commits"]["runner_sha256"] != commits["runner_sha256"]:
        raise ValueError("resume release provenance mismatch")
    if state.get("metrics_emitted") is not False:
        raise ValueError("completed release cannot be resumed")
    return state_path, state


def run(config_path: Path, *, resume: bool) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    context = validate_config(config)
    state_path, state = start_release(config_path, context, resume)
    estimator = pilot._estimator(context["controlled"])
    los = parse_gps_sdr_sim_los_table(context["paths"]["authentic_los_log"].read_text(encoding="utf-8"))
    state["phase"] = "common_multipath_analysis"
    write_json(state_path, state)
    multipath_delays, multipath_geometry = anchor.analyze_stream(
        "independent_multipath", context["paths"]["multipath_receiver_manifest"], estimator, los, config
    )
    write_csv(context["output_root"] / "common_multipath_delay_estimates.csv", multipath_delays)
    write_csv(context["output_root"] / "common_multipath_geometry_scores.csv", multipath_geometry)
    results: list[dict[str, Any]] = []
    specs = condition_specs()
    for distance in DISTANCES_M:
        distance_specs = [row for row in specs if row["distance_m"] == distance]
        dpaths = distance_paths(context["output_root"], distance)
        if dpaths["retention"].is_file() and not dpaths["counterfeit_iq"].exists():
            for spec in distance_specs:
                result_path = condition_paths(context["output_root"], spec["condition_id"])["result"]
                if not result_path.is_file():
                    raise FileNotFoundError("distance component removed before both conditions completed")
                results.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        state["phase"] = f"distance:{int(distance)}:counterfeit"
        write_json(state_path, state)
        print(f"[transfer] d{int(distance):03d} generating shared counterfeit component", flush=True)
        cpair = component_pair(context["base_pair"], distance, config)
        counterfeit, counterfeit_manifest_path, counterfeit_manifest = anchor.ensure_counterfeit(
            dpaths["root"], cpair, context["normal_profile"], context["paths"]["simulator"]
        )
        for spec in distance_specs:
            state["phase"] = f"condition:{spec['condition_id']}"
            write_json(state_path, state)
            results.append(run_condition(
                spec, config, context, estimator, los, multipath_geometry,
                counterfeit, counterfeit_manifest_path, counterfeit_manifest,
            ))
        component_identity = {
            "path": str(counterfeit.resolve()),
            "sha256": counterfeit_manifest["counterfeit"]["sha256"],
            "bytes": counterfeit_manifest["counterfeit"]["bytes"],
        }
        remove_single_iq(counterfeit, component_identity, dpaths["retention"], "shared_distance_counterfeit_iq")
        print(f"[transfer] d{int(distance):03d} both powers complete; shared component removed", flush=True)
    summaries = [result["summary"] for result in results]
    curves: dict[str, Any] = {}
    for power in POWERS_DB:
        rows = [row for row in summaries if float(row["final_advantage_db"]) == power]
        curves[f"{power:+g}_db"] = evaluate_curve(
            rows,
            threshold=float(config["analysis"]["descriptive_auc_threshold"]),
            minimum_bins=int(config["analysis"]["minimum_comparison_bins_per_stream"]),
        )
    write_csv(context["output_root"] / "condition_summary.csv", summaries)
    state["phase"] = "metrics_emitted"
    state["metrics_emitted"] = True
    state["metrics_emitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, state)
    summary = {
        "schema": "gnss-doppler-lab.cgc-rf-transfer-sweep-result",
        "schema_version": 1,
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "release_state": {"path": str(state_path.resolve()), "sha256": sha256(state_path)},
        "condition_summaries": summaries,
        "transfer_curves": curves,
        "full_grid_reported": len(summaries) == len(specs) and {row["condition_id"] for row in summaries} == {row["condition_id"] for row in specs},
        "post_release_condition_or_interval_substitution": False,
        "retention": config["retention"],
        "claim_boundary": config["claim_boundary"],
        "artifacts": {
            "condition_summary": {"path": str((context["output_root"] / "condition_summary.csv").resolve()), "sha256": sha256(context["output_root"] / "condition_summary.csv"), "row_count": len(summaries)},
            "multipath_delays": {"path": str((context["output_root"] / "common_multipath_delay_estimates.csv").resolve()), "sha256": sha256(context["output_root"] / "common_multipath_delay_estimates.csv"), "row_count": len(multipath_delays)},
            "multipath_geometry": {"path": str((context["output_root"] / "common_multipath_geometry_scores.csv").resolve()), "sha256": sha256(context["output_root"] / "common_multipath_geometry_scores.csv"), "row_count": len(multipath_geometry)},
        },
    }
    summary_path = context["output_root"] / "summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), "transfer_curves": curves}, indent=2, sort_keys=True))
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--release-token", choices=[RELEASE_TOKEN])
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config_path = DEFAULT_CONFIG.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if args.validate_only:
        print("RF transfer sweep config, 18 conditions, constant rate, fixed receiver, and shared inputs verified")
        return 0
    run(config_path, resume=bool(args.resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
