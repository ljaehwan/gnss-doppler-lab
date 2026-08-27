#!/usr/bin/env python3
"""Validate the CGC receiver state map on two additional RF geometries."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_cgc_rf_challenge_pilot as pilot  # noqa: E402
import run_cgc_rf_observability_anchors as anchor  # noqa: E402
import run_cgc_rf_transfer_sweep as transfer  # noqa: E402
import run_simulation_v4_paired_train_generation as source  # noqa: E402
from gnss_doppler_lab.peak_mixture_law import parse_gps_sdr_sim_los_table  # noqa: E402
from gnss_doppler_lab.simulation_v4 import SimulationScenario, compose_paired_iq  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/cgc_rf_state_validation_v1.json"
PROTOCOL_PATH = REPO_ROOT / "docs/results/cgc_rf_state_validation_protocol_v1.md"
RELEASE_TOKEN = "RELEASE-CGC-RF-STATE-VALIDATION-V1"
RELEASE_INPUTS = (
    "configs/experiments/cgc_rf_state_validation_v1.json",
    "docs/results/cgc_rf_state_validation_protocol_v1.md",
    "scripts/run_cgc_rf_state_validation.py",
)
GEOMETRY_IDS = ("straight", "sweep")
DISTANCES_M = (40.0, 60.0, 80.0, 100.0, 160.0, 240.0)
POWERS_DB = (-6.0, 3.0)


def receiver_run_id(condition_id: str) -> str:
    """Keep GNSS-SDR INI lines below its fixed 200-character parser limit."""
    match = re.fullmatch(r"(straight|sweep)-p(neg|pos)(\d+)-d(\d{3})", condition_id)
    if match is None:
        raise ValueError(f"unsupported state-validation condition id: {condition_id}")
    geometry, sign, magnitude, distance = match.groups()
    geometry_short = {"straight": "st", "sweep": "sw"}[geometry]
    sign_short = {"neg": "n", "pos": "p"}[sign]
    return f"v-{geometry_short}-{sign_short}{int(magnitude)}-{int(distance)}"


def sha256(path: str | Path) -> str:
    return anchor.sha256(path)


def repo_path(value: str | Path) -> Path:
    return anchor.repo_path(value)


def write_json(path: Path, document: dict[str, Any]) -> None:
    anchor.write_json(path, document)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    anchor.write_csv(path, rows)


def condition_specs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for geometry in GEOMETRY_IDS:
        for distance in DISTANCES_M:
            for power in POWERS_DB:
                short = transfer.condition_id(power, distance)
                rows.append({
                    "condition_id": f"{geometry}-{short}",
                    "geometry_id": geometry,
                    "distance_m": distance,
                    "final_advantage_db": power,
                    "target_offset_enu_m": [0.8 * distance, 0.6 * distance, 0.0],
                    "transition_seconds": distance / 20.0,
                })
    return rows


def relative_displacement_error(row: dict[str, Any]) -> float:
    truth = float(row["distance_chips"])
    return (float(row["median_estimated_displacement_norm_chips"]) - truth) / truth


def evaluate_state_group(
    rows: list[dict[str, Any]], gates: dict[str, Any], *, minimum_bins: int
) -> dict[str, Any]:
    by_distance = {float(row["distance_m"]): row for row in rows}
    if set(by_distance) != set(DISTANCES_M) or len(rows) != len(DISTANCES_M):
        raise ValueError("state group requires the complete six-distance grid")
    row40, row60, row240 = by_distance[40.0], by_distance[60.0], by_distance[240.0]
    metric_rows = [by_distance[float(distance)] for distance in gates["metric_distances_m"]]
    decisions = {
        "minimum_support": all(
            int(row["spoof_bin_count"]) >= minimum_bins
            and int(row["multipath_bin_count"]) >= minimum_bins
            and int(row["minimum_spoof_prn_count"]) >= 8
            and int(row["minimum_multipath_prn_count"]) >= 8
            for row in rows
        ),
        "40m_unresolved_auc": float(row40["serial_bin_auc"]) < float(gates["unresolved_40m_max_auc"]),
        "60m_onset_auc": float(row60["serial_bin_auc"]) < float(gates["onset_60m_max_auc"]),
        "60m_onset_direction": float(row60["median_absolute_direction_cosine"]) >= float(gates["onset_60m_min_absolute_direction_cosine"]),
        "direction_improves_40m_to_60m": float(row60["median_absolute_direction_cosine"]) > float(row40["median_absolute_direction_cosine"]),
        "metric_auc": all(float(row["serial_bin_auc"]) >= float(gates["metric_min_auc"]) for row in metric_rows),
        "metric_direction": all(float(row["median_absolute_direction_cosine"]) >= float(gates["metric_min_absolute_direction_cosine"]) for row in metric_rows),
        "metric_displacement_error": all(abs(relative_displacement_error(row)) <= float(gates["metric_max_absolute_relative_displacement_error"]) for row in metric_rows),
        "240m_saturation_auc": float(row240["serial_bin_auc"]) >= float(gates["saturation_min_auc"]),
        "240m_saturation_edge": float(row240["template_delay_edge_fraction"]) >= float(gates["saturation_min_edge_fraction"]),
        "240m_saturation_bias": relative_displacement_error(row240) <= float(gates["saturation_max_relative_displacement_error"]),
    }
    return {
        "gates": decisions,
        "state_map_reproduced": bool(all(decisions.values())),
        "auc_by_distance_m": {str(int(distance)): float(by_distance[distance]["serial_bin_auc"]) for distance in DISTANCES_M},
        "direction_by_distance_m": {str(int(distance)): float(by_distance[distance]["median_absolute_direction_cosine"]) for distance in DISTANCES_M},
        "relative_displacement_error_by_distance_m": {str(int(distance)): relative_displacement_error(by_distance[distance]) for distance in DISTANCES_M},
        "edge_fraction_by_distance_m": {str(int(distance)): float(by_distance[distance]["template_delay_edge_fraction"]) for distance in DISTANCES_M},
        "transition_80m": {
            "auc": float(by_distance[80.0]["serial_bin_auc"]),
            "direction": float(by_distance[80.0]["median_absolute_direction_cosine"]),
            "relative_displacement_error": relative_displacement_error(by_distance[80.0]),
        },
    }


def verify_record(record: dict[str, str], label: str) -> Path:
    return anchor.verify_record(record, label)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "gnss-doppler-lab.cgc-rf-state-validation-config" or config.get("schema_version") != 1:
        raise ValueError("unsupported RF state validation config")
    if config.get("experiment", {}).get("name") != "cgc-rf-state-validation-v1":
        raise ValueError("experiment identity drifted")
    paths: dict[str, Path] = {}
    for key in ("discovery_result", "base_pair_config", "normal_profile", "controlled_template"):
        paths[key] = verify_record(config[key], key)
    for key, record in config["pinned_code"].items():
        paths[key] = verify_record(record, key)
    for key in ("simulator", "receiver", "receiver_patch"):
        paths[key] = verify_record(config["rf_tools"][key], key)
    discovery = json.loads(paths["discovery_result"].read_text(encoding="utf-8"))
    if discovery.get("decision") != "exploratory_map_completed" or discovery.get("condition_count") != 18:
        raise ValueError("discovery state map drifted")
    fresh = json.loads(paths["base_pair_config"].read_text(encoding="utf-8"))
    normal_profile = json.loads(paths["normal_profile"].read_text(encoding="utf-8"))
    controlled = json.loads(paths["controlled_template"].read_text(encoding="utf-8"))
    if int(normal_profile["rf_profile"]["rf_sample_rate_hz"]) != 25_000_000:
        raise ValueError("normal profile is not 25 MHz")
    geometries = config.get("geometries", [])
    if [row.get("geometry_id") for row in geometries] != list(GEOMETRY_IDS):
        raise ValueError("held-out geometry roster drifted")
    expected_motion = {"straight": "straight", "sweep": "parallel-sweep"}
    contexts: dict[str, dict[str, Any]] = {}
    for geometry in geometries:
        geometry_id = geometry["geometry_id"]
        geometry_paths: dict[str, Path] = {}
        for key in (
            "authentic_component", "authentic_los_log", "normal_rf_manifest", "normal_rf_iq",
            "multipath_rf_manifest", "multipath_rf_iq", "multipath_receiver_manifest",
        ):
            geometry_paths[key] = verify_record(geometry[key], f"{geometry_id}.{key}")
        matches = [pair for pair in fresh["pairs"] if pair["paired_group_id"] == geometry["pair_id"]]
        if len(matches) != 1:
            raise ValueError(f"base pair missing: {geometry_id}")
        pair = matches[0]
        if pair["domain"] != "dynamic" or pair["motion"]["kind"] != expected_motion[geometry_id] or geometry["motion_kind"] != expected_motion[geometry_id]:
            raise ValueError(f"motion contract drifted: {geometry_id}")
        if pair["duration_seconds"] != 30 or pair["receiver_seed"] not in (20261004, 20261007):
            raise ValueError(f"duration or seed drifted: {geometry_id}")
        normal_manifest = json.loads(geometry_paths["normal_rf_manifest"].read_text(encoding="utf-8"))
        multipath_receiver = json.loads(geometry_paths["multipath_receiver_manifest"].read_text(encoding="utf-8"))
        if normal_manifest["iq"]["sha256"] != geometry["normal_rf_iq"]["sha256"]:
            raise ValueError(f"normal pins disagree: {geometry_id}")
        if multipath_receiver["source"]["iq_sha256"] != geometry["multipath_rf_iq"]["sha256"]:
            raise ValueError(f"multipath pins disagree: {geometry_id}")
        if multipath_receiver["tracking"]["tap_count"] != 9 or multipath_receiver["tracking"]["tap_spacing_chips"] != 0.125:
            raise ValueError(f"receiver aperture drifted: {geometry_id}")
        los = parse_gps_sdr_sim_los_table(geometry_paths["authentic_los_log"].read_text(encoding="utf-8"))
        if len(los) < 8:
            raise ValueError(f"insufficient LOS support: {geometry_id}")
        contexts[geometry_id] = {
            "definition": geometry,
            "paths": geometry_paths,
            "base_pair": pair,
            "normal_manifest": normal_manifest,
            "los": los,
        }
    tools = config["rf_tools"]
    if tools["channel_count"] != 11 or tools["tracking_tap_count"] != 9 or tools["tracking_tap_spacing_chips"] != 0.125:
        raise ValueError("fixed receiver contract drifted")
    if controlled["correlator"]["tap_offsets_chips"] != [-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5]:
        raise ValueError("template aperture drifted")
    sweep = config["sweep"]
    if (
        tuple(float(value) for value in sweep["distances_m"]) != DISTANCES_M
        or tuple(float(value) for value in sweep["final_advantages_db"]) != POWERS_DB
        or sweep["target_direction_enu_unit"] != [0.8, 0.6, 0.0]
        or sweep["start_seconds"] != 5.0
        or sweep["pull_off_rate_mps"] != 20.0
        or sweep["comparison_start_seconds"] != 18.0
    ):
        raise ValueError("validation sweep drifted")
    if not math.isclose(float(sweep["chip_length_m"]), 299_792_458.0 / 1_023_000.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("chip length drifted")
    gates = config["state_gates"]
    if gates != {
        "unresolved_40m_max_auc": 0.8,
        "onset_60m_max_auc": 0.8,
        "onset_60m_min_absolute_direction_cosine": 0.7,
        "onset_direction_must_improve_over_40m": True,
        "metric_distances_m": [100.0, 160.0],
        "metric_min_auc": 0.8,
        "metric_min_absolute_direction_cosine": 0.85,
        "metric_max_absolute_relative_displacement_error": 0.15,
        "saturation_distance_m": 240.0,
        "saturation_min_auc": 0.8,
        "saturation_min_edge_fraction": 0.1,
        "saturation_max_relative_displacement_error": -0.05,
        "overall_rule": "every gate passes independently in all four geometry-power groups",
    }:
        raise ValueError("state gates drifted")
    if config["retention"]["shared_inputs_removed"] is not False:
        raise ValueError("shared input deletion must remain forbidden")
    output_root = repo_path(config["output_root"])
    if output_root != REPO_ROOT / "artifacts/cgc_rf_state_validation_v1":
        raise ValueError("output root drifted")
    return {
        "paths": paths,
        "contexts": contexts,
        "normal_profile": normal_profile,
        "controlled": controlled,
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


def component_pair(base_pair: dict[str, Any], geometry_id: str, distance_m: float, config: dict[str, Any]) -> dict[str, Any]:
    pair = deepcopy(base_pair)
    sweep = config["sweep"]
    pair["paired_group_id"] = f"state-{geometry_id}-d{int(distance_m):03d}-source"
    pair["split"] = "rf_state_validation"
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
    pair["split"] = "rf_state_validation"
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
            raise ValueError("composed state-validation RF integrity failure")
        return iq_path, manifest_path, document
    if rf_root.exists() and any(rf_root.iterdir()):
        raise FileExistsError(f"partial state-validation RF: {rf_root}")
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
    prefix = source.compare_prefix(context["paths"]["normal_rf_iq"], iq_path, int(round(event.start_seconds * sample_rate)))
    if prefix.get("byte_identical") is not True:
        raise RuntimeError("state-validation pre-onset prefix differs from pinned normal RF")
    truth = {"class": "spoofing", "event": "constant_rate_carryoff", "is_spoofing": True, "spoofing": asdict(event)}
    document = {
        "schema_version": 4,
        "run_id": receiver_run_id(pair["paired_group_id"]),
        "scenario": {
            "name": scenario.name, "campaign": config["experiment"]["name"],
            "paired_group_id": pair["paired_group_id"], "split": "rf_state_validation",
            "utc": pair["utc"], "duration_seconds": pair["duration_seconds"],
            "position": pair["position"], "motion": pair.get("motion"), "domain": pair["domain"], **truth,
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
            "sources": {"authentic": context["definition"]["authentic_component"], "counterfeit": counterfeit_manifest["counterfeit"]},
            "scope": "offline confirmatory receiver state validation; no transmission",
        },
        "generation": {
            "config_sha256": sha256(DEFAULT_CONFIG),
            "counterfeit_manifest": str(counterfeit_manifest_path.resolve()),
            "counterfeit_manifest_sha256": sha256(counterfeit_manifest_path),
        },
    }
    write_json(manifest_path, document)
    return iq_path, manifest_path, document


def run_condition(
    spec: dict[str, Any], config: dict[str, Any], context: dict[str, Any], estimator: Any,
    multipath_geometry: list[dict[str, Any]], counterfeit: Path,
    counterfeit_manifest_path: Path, counterfeit_manifest: dict[str, Any],
) -> dict[str, Any]:
    paths = transfer.condition_paths(context["output_root"], spec["condition_id"])
    if paths["result"].is_file():
        result = json.loads(paths["result"].read_text(encoding="utf-8"))
        transfer.remove_single_iq(paths["spoof_iq"], result["intermediate_spoof_iq"], paths["retention"], "composed_spoof_iq")
        return result
    pair = scenario_pair(context["base_pair"], spec, config)
    print(f"[state-validation] {spec['condition_id']} composing RF", flush=True)
    spoof_iq, spoof_manifest_path, spoof_manifest = ensure_spoof_rf(
        paths["root"], pair, config, context, counterfeit, counterfeit_manifest_path, counterfeit_manifest
    )
    receiver_config = {
        "executable": str(context["global_paths"]["receiver"]),
        "channel_count": config["rf_tools"]["channel_count"],
        "timeout_seconds": config["rf_tools"]["timeout_seconds"],
        "tracking_tap_spacing_chips": config["rf_tools"]["tracking_tap_spacing_chips"],
    }
    expected = paths["root"] / "receiver" / spoof_manifest["run_id"] / "manifest.json"
    print(f"[state-validation] {spec['condition_id']} running GNSS-SDR", flush=True)
    receiver_manifest = pilot._ensure_receiver(
        spoof_manifest_path, paths["root"] / "receiver", receiver_config, resume=expected.is_file()
    )
    print(f"[state-validation] {spec['condition_id']} scoring states", flush=True)
    spoof_delays, spoof_geometry = anchor.analyze_stream(
        spec["condition_id"], receiver_manifest, estimator, context["los"], config
    )
    enriched_delays = [{"condition_id": spec["condition_id"], **row} for row in spoof_delays]
    enriched_geometry = [{"condition_id": spec["condition_id"], **row} for row in spoof_geometry]
    delay_path = paths["root"] / "spoof_delay_estimates.csv"
    geometry_path = paths["root"] / "spoof_geometry_scores.csv"
    write_csv(delay_path, enriched_delays)
    write_csv(geometry_path, enriched_geometry)
    summary = transfer.summarize_condition(spec, spoof_delays, spoof_geometry, multipath_geometry, config)
    identity = {"path": str(spoof_iq.resolve()), "sha256": spoof_manifest["iq"]["sha256"], "bytes": spoof_manifest["iq"]["actual_bytes"]}
    result = {
        "schema": "gnss-doppler-lab.cgc-rf-state-validation-condition-result",
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
    transfer.remove_single_iq(paths["spoof_iq"], identity, paths["retention"], "composed_spoof_iq")
    print(f"[state-validation] {spec['condition_id']} complete; composed IQ removed", flush=True)
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
            "schema": "gnss-doppler-lab.cgc-rf-state-validation-release-state",
            "schema_version": 1,
            "phase": "released_before_validation_outcomes",
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
    global_context = validate_config(config)
    state_path, state = start_release(config_path, global_context, resume)
    estimator = pilot._estimator(global_context["controlled"])
    specs = condition_specs()
    results: list[dict[str, Any]] = []
    common_artifacts: dict[str, Any] = {}
    for geometry_id in GEOMETRY_IDS:
        base = global_context["contexts"][geometry_id]
        geometry_root = global_context["output_root"] / "geometries" / geometry_id
        context = {
            **base,
            "normal_profile": global_context["normal_profile"],
            "global_paths": global_context["paths"],
            "output_root": geometry_root,
        }
        state["phase"] = f"geometry:{geometry_id}:common_multipath"
        write_json(state_path, state)
        multipath_delays, multipath_geometry = anchor.analyze_stream(
            f"{geometry_id}-independent-multipath", context["paths"]["multipath_receiver_manifest"],
            estimator, context["los"], config,
        )
        delay_path = geometry_root / "common_multipath_delay_estimates.csv"
        geometry_path = geometry_root / "common_multipath_geometry_scores.csv"
        write_csv(delay_path, multipath_delays)
        write_csv(geometry_path, multipath_geometry)
        common_artifacts[geometry_id] = {
            "delays": {"path": str(delay_path.resolve()), "sha256": sha256(delay_path), "row_count": len(multipath_delays)},
            "geometry": {"path": str(geometry_path.resolve()), "sha256": sha256(geometry_path), "row_count": len(multipath_geometry)},
        }
        geometry_specs = [row for row in specs if row["geometry_id"] == geometry_id]
        for distance in DISTANCES_M:
            distance_specs = [row for row in geometry_specs if row["distance_m"] == distance]
            dpaths = transfer.distance_paths(geometry_root, distance)
            if dpaths["retention"].is_file() and not dpaths["counterfeit_iq"].exists():
                for spec in distance_specs:
                    result_path = transfer.condition_paths(geometry_root, spec["condition_id"])["result"]
                    if not result_path.is_file():
                        raise FileNotFoundError("distance component removed before both conditions completed")
                    results.append(json.loads(result_path.read_text(encoding="utf-8")))
                continue
            state["phase"] = f"geometry:{geometry_id}:distance:{int(distance)}:counterfeit"
            write_json(state_path, state)
            print(f"[state-validation] {geometry_id} d{int(distance):03d} generating shared component", flush=True)
            cpair = component_pair(context["base_pair"], geometry_id, distance, config)
            counterfeit, counterfeit_manifest_path, counterfeit_manifest = anchor.ensure_counterfeit(
                dpaths["root"], cpair, context["normal_profile"], context["global_paths"]["simulator"]
            )
            for spec in distance_specs:
                state["phase"] = f"condition:{spec['condition_id']}"
                write_json(state_path, state)
                results.append(run_condition(
                    spec, config, context, estimator, multipath_geometry,
                    counterfeit, counterfeit_manifest_path, counterfeit_manifest,
                ))
            component_identity = {
                "path": str(counterfeit.resolve()),
                "sha256": counterfeit_manifest["counterfeit"]["sha256"],
                "bytes": counterfeit_manifest["counterfeit"]["bytes"],
            }
            transfer.remove_single_iq(counterfeit, component_identity, dpaths["retention"], "shared_distance_counterfeit_iq")
            print(f"[state-validation] {geometry_id} d{int(distance):03d} both powers complete; component removed", flush=True)
    summaries = [result["summary"] for result in results]
    evaluations: dict[str, Any] = {}
    for geometry_id in GEOMETRY_IDS:
        evaluations[geometry_id] = {}
        for power in POWERS_DB:
            rows = [
                row for row in summaries
                if row["geometry_id"] == geometry_id and float(row["final_advantage_db"]) == power
            ]
            evaluations[geometry_id][f"{power:+g}_db"] = evaluate_state_group(
                rows, config["state_gates"], minimum_bins=int(config["analysis"]["minimum_comparison_bins_per_stream"])
            )
    full_grid = len(summaries) == len(specs) and {row["condition_id"] for row in summaries} == {row["condition_id"] for row in specs}
    overall = full_grid and all(
        group["state_map_reproduced"]
        for geometry in evaluations.values() for group in geometry.values()
    )
    write_csv(global_context["output_root"] / "condition_summary.csv", summaries)
    state["phase"] = "metrics_emitted"
    state["metrics_emitted"] = True
    state["metrics_emitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, state)
    summary = {
        "schema": "gnss-doppler-lab.cgc-rf-state-validation-result",
        "schema_version": 1,
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "release_state": {"path": str(state_path.resolve()), "sha256": sha256(state_path)},
        "condition_summaries": summaries,
        "state_evaluations": evaluations,
        "full_grid_reported": full_grid,
        "all_four_geometry_power_groups_reproduced": overall,
        "post_release_geometry_cell_or_gate_substitution": False,
        "retention": config["retention"],
        "claim_boundary": config["claim_boundary"],
        "artifacts": {
            "condition_summary": {
                "path": str((global_context["output_root"] / "condition_summary.csv").resolve()),
                "sha256": sha256(global_context["output_root"] / "condition_summary.csv"),
                "row_count": len(summaries),
            },
            "common_multipath": common_artifacts,
        },
    }
    summary_path = global_context["output_root"] / "summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), "all_four_geometry_power_groups_reproduced": overall, "state_evaluations": evaluations}, indent=2, sort_keys=True))
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
        print("Two held-out geometries, 24 RF cells, state gates, tools, and shared inputs verified")
        return 0
    run(config_path, resume=bool(args.resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
