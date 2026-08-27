#!/usr/bin/env python3
"""Validate CGC state boundaries across five geometries and three apertures."""
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
import run_simulation_v4_normal_independent_validation as normal  # noqa: E402
import run_simulation_v4_paired_train_generation as source  # noqa: E402
from gnss_doppler_lab.clock_centered_geometry import fit_clock_centered_geometry  # noqa: E402
from gnss_doppler_lab.correlator_geometry import complex_profile_features  # noqa: E402
from gnss_doppler_lab.peak_mixture_law import parse_gps_sdr_sim_los_table  # noqa: E402
from gnss_doppler_lab.rf_config import (  # noqa: E402
    InputConfig,
    OutputConfig,
    RFGenerationConfig,
    Scenario,
    SimulatorConfig,
    StaticPosition,
    TrajectoryPosition,
)
from gnss_doppler_lab.rf_impairments import ImpairmentConfig  # noqa: E402
from gnss_doppler_lab.satellite_multipath import (  # noqa: E402
    PrnMultipathGpsSdrSimRunner,
    independent_echoes,
)
from gnss_doppler_lab.simulation_v4 import SimulationScenario, compose_paired_iq  # noqa: E402
from gnss_doppler_lab.tracking_peaks import (  # noqa: E402
    available_tracking_prns,
    load_receiver_tracking_peak_series_segments,
)
from gnss_doppler_lab.trajectory import read_trajectory  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/cgc_rf_geometry_aperture_validation_v1.json"
PROTOCOL_PATH = REPO_ROOT / "docs/results/cgc_rf_geometry_aperture_validation_protocol_v1.md"
RELEASE_TOKEN = "RELEASE-CGC-RF-GEOMETRY-APERTURE-VALIDATION-V1"
RELEASE_INPUTS = (
    "configs/experiments/cgc_rf_geometry_aperture_validation_v1.json",
    "docs/results/cgc_rf_geometry_aperture_validation_protocol_v1.md",
    "scripts/run_cgc_rf_geometry_aperture_validation.py",
)
GEOMETRY_IDS = ("denver-static", "seoul-static", "tokyo-straight", "london-circle", "sydney-sweep")
DISTANCES_M = (40.0, 60.0, 100.0, 240.0)
POWERS_DB = (-6.0, 3.0)
APERTURE_TAPS = (3, 5, 9)
APERTURE_INDICES = {3: (3, 4, 5), 5: (2, 3, 4, 5, 6), 9: tuple(range(9))}
FULL_TAPS = (-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5)


def sha256(path: str | Path) -> str:
    return anchor.sha256(path)


def repo_path(path: str | Path) -> Path:
    return anchor.repo_path(path)


def write_json(path: Path, document: dict[str, Any]) -> None:
    anchor.write_json(path, document)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    anchor.write_csv(path, rows)


def receiver_run_id(condition_id: str) -> str:
    match = re.fullmatch(r"(denver-static|seoul-static|tokyo-straight|london-circle|sydney-sweep)-p(neg|pos)(\d+)-d(\d{3})", condition_id)
    if match is None:
        raise ValueError(f"unsupported geometry-aperture condition: {condition_id}")
    geometry, sign, magnitude, distance = match.groups()
    short = {
        "denver-static": "ds", "seoul-static": "ss", "tokyo-straight": "ts",
        "london-circle": "lc", "sydney-sweep": "sw",
    }[geometry]
    return f"ga-{short}-{'n' if sign == 'neg' else 'p'}{int(magnitude)}-{int(distance)}"


def condition_specs() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for geometry_id in GEOMETRY_IDS:
        for distance_m in DISTANCES_M:
            for power_db in POWERS_DB:
                result.append({
                    "condition_id": f"{geometry_id}-{transfer.condition_id(power_db, distance_m)}",
                    "geometry_id": geometry_id,
                    "distance_m": distance_m,
                    "final_advantage_db": power_db,
                    "target_offset_enu_m": [0.8 * distance_m, 0.6 * distance_m, 0.0],
                    "transition_seconds": distance_m / 20.0,
                })
    return result


def relative_displacement_error(row: dict[str, Any]) -> float:
    truth = float(row["distance_chips"])
    return (float(row["median_estimated_displacement_norm_chips"]) - truth) / truth


def evaluate_geometry_group(rows: list[dict[str, Any]], gates: dict[str, Any], *, minimum_bins: int) -> dict[str, Any]:
    by_distance = {float(row["distance_m"]): row for row in rows}
    if len(rows) != len(DISTANCES_M) or set(by_distance) != set(DISTANCES_M):
        raise ValueError("geometry group requires the complete four-distance grid")
    metric = by_distance[100.0]
    saturation = by_distance[240.0]
    decisions = {
        "minimum_support": all(
            int(row["spoof_bin_count"]) >= minimum_bins
            and int(row["multipath_bin_count"]) >= minimum_bins
            and int(row["minimum_spoof_prn_count"]) >= 8
            and int(row["minimum_multipath_prn_count"]) >= 8
            for row in rows
        ),
        "100m_metric_auc": float(metric["serial_bin_auc"]) >= float(gates["metric_min_auc"]),
        "100m_metric_direction": float(metric["median_absolute_direction_cosine"]) >= float(gates["metric_min_absolute_direction_cosine"]),
        "100m_metric_error": abs(relative_displacement_error(metric)) <= float(gates["metric_max_absolute_relative_displacement_error"]),
        "240m_saturation_auc": float(saturation["serial_bin_auc"]) >= float(gates["saturation_min_auc"]),
        "240m_saturation_direction": float(saturation["median_absolute_direction_cosine"]) >= float(gates["saturation_min_absolute_direction_cosine"]),
        "240m_saturation_edge": float(saturation["template_delay_edge_fraction"]) >= float(gates["saturation_min_edge_fraction"]),
        "240m_saturation_bias": relative_displacement_error(saturation) <= float(gates["saturation_max_relative_displacement_error"]),
    }
    return {
        "gates": decisions,
        "mechanism_reproduced": bool(all(decisions.values())),
        "auc_by_distance_m": {str(int(d)): float(by_distance[d]["serial_bin_auc"]) for d in DISTANCES_M},
        "direction_by_distance_m": {str(int(d)): float(by_distance[d]["median_absolute_direction_cosine"]) for d in DISTANCES_M},
        "relative_error_by_distance_m": {str(int(d)): relative_displacement_error(by_distance[d]) for d in DISTANCES_M},
        "edge_fraction_by_distance_m": {str(int(d)): float(by_distance[d]["template_delay_edge_fraction"]) for d in DISTANCES_M},
        "early_boundary_descriptive_only": {
            str(int(d)): {
                "auc": float(by_distance[d]["serial_bin_auc"]),
                "direction": float(by_distance[d]["median_absolute_direction_cosine"]),
            }
            for d in (40.0, 60.0)
        },
    }


def evaluate_aperture_mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if float(row["distance_m"]) in (100.0, 240.0)]
    aggregates: dict[str, dict[str, float]] = {}
    for taps in APERTURE_TAPS:
        tap_rows = [row for row in selected if int(row["aperture_taps"]) == taps]
        if len(tap_rows) != len(GEOMETRY_IDS) * len(POWERS_DB) * 2:
            raise ValueError("aperture audit requires all geometry-power cells at 100 and 240 m")
        at100 = [row for row in tap_rows if float(row["distance_m"]) == 100.0]
        at240 = [row for row in tap_rows if float(row["distance_m"]) == 240.0]
        aggregates[str(taps)] = {
            "median_100m_absolute_relative_error": float(np.median([abs(relative_displacement_error(row)) for row in at100])),
            "median_240m_absolute_relative_error": float(np.median([abs(relative_displacement_error(row)) for row in at240])),
            "median_240m_recovered_norm_chips": float(np.median([row["median_estimated_displacement_norm_chips"] for row in at240])),
            "median_240m_edge_fraction": float(np.median([row["template_delay_edge_fraction"] for row in at240])),
        }
    a3, a5, a9 = (aggregates[str(taps)] for taps in APERTURE_TAPS)
    decisions = {
        "240m_absolute_error_nonincreasing_3_to_5_to_9": a3["median_240m_absolute_relative_error"] >= a5["median_240m_absolute_relative_error"] >= a9["median_240m_absolute_relative_error"],
        "240m_recovered_norm_nondecreasing_3_to_5_to_9": a3["median_240m_recovered_norm_chips"] <= a5["median_240m_recovered_norm_chips"] <= a9["median_240m_recovered_norm_chips"],
        "240m_edge_fraction_nonincreasing_3_to_5_to_9": a3["median_240m_edge_fraction"] >= a5["median_240m_edge_fraction"] >= a9["median_240m_edge_fraction"],
        "100m_9tap_error_not_worse_than_3tap": a9["median_100m_absolute_relative_error"] <= a3["median_100m_absolute_relative_error"],
    }
    return {"aggregates": aggregates, "gates": decisions, "aperture_mechanism_supported": bool(all(decisions.values()))}


def verify_record(record: dict[str, Any], label: str) -> Path:
    path = repo_path(record["path"])
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"pinned input mismatch: {label}")
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"pinned byte count mismatch: {label}")
    return path


def _asset_paths(config: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Path]:
    return {key: verify_record(record, f"{geometry['geometry_id']}.{key}") for key, record in geometry["source_assets"].items()}


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "gnss-doppler-lab.cgc-rf-geometry-aperture-validation-config" or config.get("schema_version") != 1:
        raise ValueError("unsupported geometry-aperture validation config")
    if config.get("experiment", {}).get("name") != "cgc-rf-geometry-aperture-validation-v1":
        raise ValueError("experiment identity drifted")
    normal_path = verify_record(config["normal_profile"], "normal_profile")
    controlled_path = verify_record(config["controlled_template"], "controlled_template")
    for key, record in config["rf_tools"].items():
        if isinstance(record, dict) and "path" in record:
            verify_record(record, f"rf_tools.{key}")
    normal_profile = json.loads(normal_path.read_text(encoding="utf-8"))
    controlled = json.loads(controlled_path.read_text(encoding="utf-8"))
    if int(normal_profile["rf_profile"]["rf_sample_rate_hz"]) != 25_000_000:
        raise ValueError("normal profile must remain 25 MHz")
    if tuple(controlled["correlator"]["tap_offsets_chips"]) != FULL_TAPS:
        raise ValueError("controlled nine-tap aperture drifted")
    if tuple(float(x) for x in config["sweep"]["distances_m"]) != DISTANCES_M or tuple(float(x) for x in config["sweep"]["final_advantages_db"]) != POWERS_DB:
        raise ValueError("geometry-aperture sweep drifted")
    if config["sweep"]["target_direction_enu_unit"] != [0.8, 0.6, 0.0] or float(config["sweep"]["comparison_start_seconds"]) != 18.0:
        raise ValueError("pull-off geometry or comparison time drifted")
    if tuple(config["aperture_ablation"]["tap_counts"]) != APERTURE_TAPS or config["aperture_ablation"]["policy"] != "central subsets of the same raw nine-tap receiver output":
        raise ValueError("aperture ablation drifted")
    geometries = config["geometries"]
    if tuple(row["geometry_id"] for row in geometries) != GEOMETRY_IDS:
        raise ValueError("five-geometry roster drifted")
    source_runs = {run["name"]: run for run in normal_profile["runs"]}
    contexts: dict[str, dict[str, Any]] = {}
    for geometry in geometries:
        run = source_runs.get(geometry["source_run"])
        if run is None or run != geometry["run"]:
            raise ValueError(f"source run mismatch: {geometry['geometry_id']}")
        paths = _asset_paths(config, geometry)
        component = json.loads(paths["component_manifest"].read_text(encoding="utf-8"))
        normal_manifest = json.loads(paths["normal_rf_manifest"].read_text(encoding="utf-8"))
        if component["run"] != run or normal_manifest["validation"]["run"] != run:
            raise ValueError(f"source metadata mismatch: {geometry['geometry_id']}")
        if component["iq_sha256"] != geometry["source_assets"]["authentic_component"]["sha256"]:
            raise ValueError("authentic component pins disagree")
        if normal_manifest["iq"]["sha256"] != geometry["source_assets"]["normal_rf_iq"]["sha256"]:
            raise ValueError("normal RF pins disagree")
        los = parse_gps_sdr_sim_los_table(paths["authentic_los_log"].read_text(encoding="utf-8"))
        if len(los) != int(geometry["startup_los_prns"]) or len(los) < int(config["analysis"]["minimum_prns"]):
            raise ValueError(f"LOS support mismatch: {geometry['geometry_id']}")
        contexts[geometry["geometry_id"]] = {
            "definition": geometry, "run": run, "paths": paths,
            "component_manifest": component, "normal_manifest": normal_manifest, "los": los,
        }
    if config["retention"]["shared_source_inputs_removed"] is not False:
        raise ValueError("source inputs may not be removed")
    output_root = repo_path(config["output_root"])
    if output_root != REPO_ROOT / "artifacts/cgc_rf_geometry_aperture_validation_v1":
        raise ValueError("output root drifted")
    return {"normal_profile": normal_profile, "controlled": controlled, "contexts": contexts, "output_root": output_root}


def _scenario_position(run: dict[str, Any], component: dict[str, Any]) -> StaticPosition | TrajectoryPosition:
    position = run["position"]
    if run["domain"] == "static":
        if component.get("trajectory") is not None:
            raise ValueError("static source unexpectedly has a trajectory")
        return StaticPosition(float(position["latitude_deg"]), float(position["longitude_deg"]), float(position["altitude_m"]))
    record = component.get("trajectory")
    if not isinstance(record, dict):
        raise ValueError("dynamic source lacks trajectory provenance")
    path, metadata = Path(record["path"]).resolve(), Path(record["metadata_path"]).resolve()
    if sha256(path) != record["sha256"] or sha256(metadata) != record["metadata_sha256"]:
        raise ValueError("source trajectory integrity failure")
    rows = tuple(read_trajectory(path, float(run["duration_seconds"]), "llh"))
    if len(rows) != int(record["row_count"]):
        raise ValueError("source trajectory row count mismatch")
    return TrajectoryPosition(path=path, coordinate_system="llh", rows=rows, csv_sha256=record["sha256"], metadata_path=metadata, metadata_sha256=record["metadata_sha256"])


def ensure_multipath_component(root: Path, config: dict[str, Any], context: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    directory = root / "component"
    iq_path, log_path, manifest_path = directory / "multipath_gps_l1ca_s8_iq.bin", directory / "gps-sdr-sim.log", directory / "manifest.json"
    if manifest_path.is_file():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not iq_path.is_file() or sha256(iq_path) != document["iq"]["sha256"]:
            raise ValueError("multipath component resume integrity failure")
        return iq_path, manifest_path, document
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"partial multipath component: {directory}")
    geometry = context["definition"]
    run = context["run"]
    multipath = config["multipath"]
    prns = [int(prn[1:]) for prn in context["los"]]
    echoes = independent_echoes(prns, seed=int(geometry["multipath_seed"]), delay_chips_range=tuple(multipath["delay_chips_range"]), amplitude_range=tuple(multipath["amplitude_range"]))
    simulator = verify_record(config["rf_tools"]["multipath_simulator"], "multipath_simulator")
    sample_rate = int(context["normal_profile"]["rf_profile"]["rf_sample_rate_hz"])
    rf_config = RFGenerationConfig(
        version=1,
        scenario=Scenario(f"{geometry['geometry_id']}-independent-multipath", "GPS", "L1CA", datetime.fromisoformat(run["utc"].replace("Z", "+00:00")), int(run["duration_seconds"]), _scenario_position(run, context["component_manifest"])),
        input=InputConfig(repo_path(context["normal_profile"]["input"]["rinex_nav"])),
        output=OutputConfig(directory, sample_rate, "s8_iq"),
        simulator=SimulatorConfig(str(simulator)),
        impairments=ImpairmentConfig(),
    )
    runner = PrnMultipathGpsSdrSimRunner(str(simulator), echoes)
    result = runner.run(rf_config, iq_path, log_path)
    if result["actual_bytes"] != runner.expected_output_bytes(rf_config):
        raise ValueError("multipath component byte contract failed")
    component_los = parse_gps_sdr_sim_los_table(log_path.read_text(encoding="utf-8"))
    if component_los != context["los"]:
        raise ValueError("multipath component LOS changed")
    document = {
        "schema": "gnss-doppler-lab.cgc-rf-geometry-aperture-multipath-component", "schema_version": 1,
        "geometry": geometry, "run": run,
        "source_component_manifest": {"path": str(context["paths"]["component_manifest"].resolve()), "sha256": sha256(context["paths"]["component_manifest"])},
        "simulator": {"path": str(simulator), "sha256": sha256(simulator), "patch": config["rf_tools"]["multipath_patch"], "command": result["command"]},
        "multipath": result["multipath"],
        "iq": {"path": str(iq_path.resolve()), "sha256": sha256(iq_path), "bytes": iq_path.stat().st_size, "rf_sample_rate_hz": sample_rate},
        "log": {"path": str(log_path.resolve()), "sha256": sha256(log_path)},
    }
    write_json(manifest_path, document)
    return iq_path, manifest_path, document


def ensure_multipath_rf(root: Path, config: dict[str, Any], context: dict[str, Any], component: Path, component_manifest: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    directory = root / "rf"
    iq_path, manifest_path = directory / "gps_l1ca_s8_iq.bin", directory / "manifest.json"
    if manifest_path.is_file():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not iq_path.is_file() or sha256(iq_path) != document["iq"]["sha256"]:
            raise ValueError("multipath RF resume integrity failure")
        return iq_path, manifest_path, document
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"partial multipath RF: {directory}")
    run, profile = context["run"], context["normal_profile"]
    scenario = SimulationScenario("independent_multipath", "steady_normal")
    receiver = normal._run_impairment(profile, run)
    composition = compose_paired_iq(
        component, component, {scenario.name: iq_path}, (scenario,),
        sample_rate_hz=int(profile["rf_profile"]["rf_sample_rate_hz"]), receiver=receiver,
        normal_target_rms=float(profile["normal_target_rms"]),
        reference_override=context["normal_manifest"]["validation"]["composition_reference"],
    )
    report = composition["scenarios"][scenario.name]
    document = {
        "schema_version": 4, "run_id": f"ga-mp-{context['definition']['geometry_id']}",
        "scenario": {"name": "independent_multipath", "campaign": config["experiment"]["name"], "class": "multipath", "event": "steady_prn_specific_multipath", "is_spoofing": False, "split": "geometry_aperture_validation", "paired_group_id": context["definition"]["geometry_id"], "utc": run["utc"], "duration_seconds": run["duration_seconds"], "position": run["position"], "motion": run.get("motion"), "domain": run["domain"]},
        "iq": {"path": iq_path.name, "sha256": report["sha256"], "actual_bytes": report["bytes"], "complex_samples": report["complex_samples"], "actual_duration_seconds": report["actual_duration_seconds"], "rf_sample_rate_hz": int(profile["rf_profile"]["rf_sample_rate_hz"]), "sample_format": "s8_iq", "channels": 2},
        "simulation_v4": {"truth": {"class": "multipath", "is_spoofing": False, "echoes": component_manifest["multipath"]["echoes"]}, "receiver": {"requested": receiver.manifest(), "reference": composition["reference"], "processing": composition["processing"]}, "measurements": report, "source_component": component_manifest["iq"], "scope": "offline receiver-RF geometry/aperture validation; no transmission"},
    }
    write_json(manifest_path, document)
    return iq_path, manifest_path, document


def _estimator(controlled: dict[str, Any], taps: int) -> Any:
    modified = deepcopy(controlled)
    indices = APERTURE_INDICES[taps]
    modified["correlator"]["tap_offsets_chips"] = [FULL_TAPS[index] for index in indices]
    modified["correlator"]["prompt_index"] = taps // 2
    return pilot._estimator(modified)


def analyze_stream(name: str, receiver_manifest: Path, estimator: Any, los: dict[str, tuple[float, float, float]], config: dict[str, Any], taps: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = receiver_manifest.parent
    bin_seconds = float(config["analysis"]["bin_seconds"])
    minimum_prns = int(config["analysis"]["minimum_prns"])
    indices = APERTURE_INDICES[taps]
    delays: list[dict[str, Any]] = []
    by_bin: dict[int, list[tuple[str, float, float]]] = {}
    for prn in available_tracking_prns(run_dir):
        if prn not in los:
            continue
        segments = load_receiver_tracking_peak_series_segments(run_dir, prn, tap_count=9, require_complex_taps=True)
        times = np.concatenate([segment.time_s for segment in segments])
        profiles = np.concatenate([segment.complex_taps[:, indices] for segment in segments])
        features = complex_profile_features(profiles, prompt_index=taps // 2)
        estimates, distances, _ = estimator.estimate(features)
        bins = np.floor(times / bin_seconds).astype(np.int64)
        for bin_index in np.unique(bins):
            mask = bins == bin_index
            delay = float(np.median(estimates[mask]))
            distance = float(np.median(distances[mask]))
            delays.append({"scenario": name, "aperture_taps": taps, "bin_index": int(bin_index), "bin_start_s": float(bin_index * bin_seconds), "prn": prn, "epoch_count": int(np.count_nonzero(mask)), "estimated_delay_chips": delay, "median_template_distance": distance})
            by_bin.setdefault(int(bin_index), []).append((prn, delay, distance))
    geometry_rows: list[dict[str, Any]] = []
    for bin_index, entries in sorted(by_bin.items()):
        if len(entries) < minimum_prns:
            continue
        prns = [entry[0] for entry in entries]
        estimated = np.asarray([entry[1] for entry in entries], dtype=np.float64)
        fit = fit_clock_centered_geometry(np.asarray([los[prn] for prn in prns], dtype=np.float64), estimated)
        geometry_rows.append({
            "scenario": name, "aperture_taps": taps, "bin_index": bin_index, "bin_start_s": float(bin_index * bin_seconds), "prn_count": len(prns),
            "complex_geometry_residual": fit.normalized_residual, "complex_geometry_coherence": fit.coherence,
            "clock_centered_geometry_residual": fit.clock_centered_normalized_residual, "directional_geometry_coherence": fit.directional_coherence,
            "clock_only_bias_chips": fit.clock_only_bias_chips, "fit_rank": fit.rank,
            "estimated_displacement_e_chips": float(fit.theta[0]), "estimated_displacement_n_chips": float(fit.theta[1]), "estimated_displacement_u_chips": float(fit.theta[2]), "estimated_clock_bias_chips": float(fit.theta[3]),
        })
    return delays, geometry_rows


def _receiver_config(config: dict[str, Any]) -> dict[str, Any]:
    return {"executable": str(repo_path(config["rf_tools"]["receiver"]["path"])), "channel_count": int(config["rf_tools"]["channel_count"]), "timeout_seconds": int(config["rf_tools"]["timeout_seconds"]), "tracking_tap_spacing_chips": float(config["rf_tools"]["tracking_tap_spacing_chips"])}


def prepare_multipath_control(root: Path, config: dict[str, Any], context: dict[str, Any], estimators: dict[int, Any]) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    expected = root / "receiver" / f"ga-mp-{context['definition']['geometry_id']}" / "manifest.json"
    rf_manifest = root / "rf/manifest.json"
    component_iq: Path | None = None
    component_document: dict[str, Any] | None = None
    if expected.is_file():
        receiver_manifest = pilot._ensure_receiver(rf_manifest, root / "receiver", _receiver_config(config), resume=True)
    else:
        component_iq, _, component_document = ensure_multipath_component(root, config, context)
        rf_iq, rf_manifest, rf_document = ensure_multipath_rf(root, config, context, component_iq, component_document)
        receiver_manifest = pilot._ensure_receiver(rf_manifest, root / "receiver", _receiver_config(config), resume=expected.is_file())
    by_aperture: dict[int, list[dict[str, Any]]] = {}
    artifacts: dict[str, Any] = {"receiver_manifest": {"path": str(receiver_manifest.resolve()), "sha256": sha256(receiver_manifest)}, "apertures": {}}
    for taps in APERTURE_TAPS:
        delay_rows, geometry_rows = analyze_stream("independent_multipath", receiver_manifest, estimators[taps], context["los"], config, taps)
        delay_path, geometry_path = root / f"multipath_delay_estimates_{taps}tap.csv", root / f"multipath_geometry_scores_{taps}tap.csv"
        write_csv(delay_path, delay_rows)
        write_csv(geometry_path, geometry_rows)
        by_aperture[taps] = geometry_rows
        artifacts["apertures"][str(taps)] = {"delays": {"path": str(delay_path.resolve()), "sha256": sha256(delay_path), "row_count": len(delay_rows)}, "geometry": {"path": str(geometry_path.resolve()), "sha256": sha256(geometry_path), "row_count": len(geometry_rows)}}
    rf_doc = json.loads(rf_manifest.read_text(encoding="utf-8"))
    identities = [{"path": str((rf_manifest.parent / rf_doc["iq"]["path"]).resolve()), "sha256": rf_doc["iq"]["sha256"], "bytes": rf_doc["iq"]["actual_bytes"], "kind": "multipath_rf"}]
    if component_document is None:
        component_manifest_path = root / "component/manifest.json"
        if component_manifest_path.is_file():
            component_document = json.loads(component_manifest_path.read_text(encoding="utf-8"))
    if component_document is not None:
        identities.append({"path": component_document["iq"]["path"], "sha256": component_document["iq"]["sha256"], "bytes": component_document["iq"]["bytes"], "kind": "multipath_component"})
    removed = []
    for identity in identities:
        path = Path(identity["path"])
        if path.is_file():
            if sha256(path) != identity["sha256"] or path.stat().st_size != int(identity["bytes"]):
                raise ValueError("refusing to remove changed multipath intermediate")
            path.unlink()
            removed.append({**identity, "removed": True})
    write_json(root / "retention.json", {"schema": "gnss-doppler-lab.cgc-rf-geometry-aperture-retention", "schema_version": 1, "removed_intermediates": removed, "receiver_outputs_retained": True, "source_inputs_removed": False})
    return by_aperture, artifacts


def component_pair(run: dict[str, Any], geometry_id: str, distance_m: float, config: dict[str, Any]) -> dict[str, Any]:
    pair = deepcopy(run)
    pair["paired_group_id"] = f"ga-{geometry_id}-d{int(distance_m):03d}-source"
    pair["split"] = "geometry_aperture_validation"
    pair["spoofing"] = {"start_seconds": float(config["sweep"]["start_seconds"]), "transition_seconds": distance_m / float(config["sweep"]["pull_off_rate_mps"]), "target_offset_enu_m": [0.8 * distance_m, 0.6 * distance_m, 0.0], "initial_advantage_db": 0.0, "final_advantage_db": 0.0, "power_ramp_seconds": 0.0}
    return pair


def scenario_pair(run: dict[str, Any], spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    pair = deepcopy(run)
    pair["paired_group_id"] = spec["condition_id"]
    pair["split"] = "geometry_aperture_validation"
    pair["spoofing"] = {"start_seconds": float(config["sweep"]["start_seconds"]), "transition_seconds": float(spec["transition_seconds"]), "target_offset_enu_m": spec["target_offset_enu_m"], "initial_advantage_db": float(config["sweep"]["initial_advantage_db"]), "final_advantage_db": float(spec["final_advantage_db"]), "power_ramp_seconds": float(config["sweep"]["power_ramp_seconds"])}
    return pair


def ensure_spoof_rf(root: Path, pair: dict[str, Any], config: dict[str, Any], context: dict[str, Any], counterfeit: Path, counterfeit_manifest_path: Path, counterfeit_manifest: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    directory = root / "rf"
    iq_path, manifest_path = directory / "gps_l1ca_s8_iq.bin", directory / "manifest.json"
    if manifest_path.is_file():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not iq_path.is_file() or sha256(iq_path) != document["iq"]["sha256"]:
            raise ValueError("spoof RF resume integrity failure")
        return iq_path, manifest_path, document
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"partial spoof RF: {directory}")
    event = source._spoof_event(pair)
    scenario = SimulationScenario(pair["paired_group_id"], "carryoff_spoof", spoofing=event)
    receiver = normal._run_impairment(context["normal_profile"], pair)
    composition = compose_paired_iq(
        context["paths"]["authentic_component"], counterfeit, {scenario.name: iq_path}, (scenario,),
        sample_rate_hz=int(context["normal_profile"]["rf_profile"]["rf_sample_rate_hz"]), receiver=receiver,
        normal_target_rms=float(context["normal_profile"]["normal_target_rms"]),
        reference_override=context["normal_manifest"]["validation"]["composition_reference"],
    )
    report = composition["scenarios"][scenario.name]
    sample_rate = int(context["normal_profile"]["rf_profile"]["rf_sample_rate_hz"])
    prefix = source.compare_prefix(context["paths"]["normal_rf_iq"], iq_path, int(round(event.start_seconds * sample_rate)))
    if prefix.get("byte_identical") is not True:
        raise RuntimeError("spoof pre-onset prefix differs from pinned normal RF")
    truth = {"class": "spoofing", "event": "constant_rate_carryoff", "is_spoofing": True, "spoofing": asdict(event)}
    document = {
        "schema_version": 4, "run_id": receiver_run_id(pair["paired_group_id"]),
        "scenario": {"name": scenario.name, "campaign": config["experiment"]["name"], "paired_group_id": pair["paired_group_id"], "split": "geometry_aperture_validation", "utc": pair["utc"], "duration_seconds": pair["duration_seconds"], "position": pair["position"], "motion": pair.get("motion"), "domain": pair["domain"], **truth},
        "iq": {"path": iq_path.name, "sha256": report["sha256"], "actual_bytes": report["bytes"], "complex_samples": report["complex_samples"], "actual_duration_seconds": report["actual_duration_seconds"], "rf_sample_rate_hz": sample_rate, "sample_format": "s8_iq", "channels": 2},
        "simulation_v4": {"truth": truth, "pair_contract": {"reference_member": "pinned-normal", "paired_prefix_check": prefix}, "receiver": {"requested": receiver.manifest(), "reference": composition["reference"], "processing": composition["processing"]}, "measurements": report, "sources": {"authentic": context["definition"]["source_assets"]["authentic_component"], "counterfeit": counterfeit_manifest["counterfeit"]}, "scope": "offline receiver-RF geometry/aperture validation; no transmission"},
        "generation": {"counterfeit_manifest": str(counterfeit_manifest_path.resolve()), "counterfeit_manifest_sha256": sha256(counterfeit_manifest_path)},
    }
    write_json(manifest_path, document)
    return iq_path, manifest_path, document


def run_condition(spec: dict[str, Any], config: dict[str, Any], context: dict[str, Any], estimators: dict[int, Any], multipath_by_aperture: dict[int, list[dict[str, Any]]], counterfeit: Path, counterfeit_manifest_path: Path, counterfeit_manifest: dict[str, Any]) -> dict[str, Any]:
    paths = transfer.condition_paths(context["output_root"], spec["condition_id"])
    if paths["result"].is_file():
        result = json.loads(paths["result"].read_text(encoding="utf-8"))
        transfer.remove_single_iq(paths["spoof_iq"], result["intermediate_spoof_iq"], paths["retention"], "composed_spoof_iq")
        return result
    pair = scenario_pair(context["run"], spec, config)
    spoof_iq, spoof_manifest_path, spoof_manifest = ensure_spoof_rf(paths["root"], pair, config, context, counterfeit, counterfeit_manifest_path, counterfeit_manifest)
    expected = paths["root"] / "receiver" / spoof_manifest["run_id"] / "manifest.json"
    receiver_manifest = pilot._ensure_receiver(spoof_manifest_path, paths["root"] / "receiver", _receiver_config(config), resume=expected.is_file())
    summaries: dict[str, dict[str, Any]] = {}
    score_artifacts: dict[str, Any] = {}
    for taps in APERTURE_TAPS:
        delays, geometry = analyze_stream(spec["condition_id"], receiver_manifest, estimators[taps], context["los"], config, taps)
        delay_path, geometry_path = paths["root"] / f"spoof_delay_estimates_{taps}tap.csv", paths["root"] / f"spoof_geometry_scores_{taps}tap.csv"
        write_csv(delay_path, delays)
        write_csv(geometry_path, geometry)
        summary = transfer.summarize_condition(spec, delays, geometry, multipath_by_aperture[taps], config)
        summaries[str(taps)] = {**summary, "aperture_taps": taps}
        score_artifacts[str(taps)] = {"delays": {"path": str(delay_path.resolve()), "sha256": sha256(delay_path), "row_count": len(delays)}, "geometry": {"path": str(geometry_path.resolve()), "sha256": sha256(geometry_path), "row_count": len(geometry)}}
    identity = {"path": str(spoof_iq.resolve()), "sha256": spoof_manifest["iq"]["sha256"], "bytes": spoof_manifest["iq"]["actual_bytes"]}
    result = {
        "schema": "gnss-doppler-lab.cgc-rf-geometry-aperture-condition-result", "schema_version": 1,
        "condition": spec, "pair": pair, "summary_by_aperture": summaries,
        "prefix": spoof_manifest["simulation_v4"]["pair_contract"]["paired_prefix_check"],
        "counterfeit_manifest": {"path": str(counterfeit_manifest_path.resolve()), "sha256": sha256(counterfeit_manifest_path)},
        "spoof_rf_manifest": {"path": str(spoof_manifest_path.resolve()), "sha256": sha256(spoof_manifest_path)},
        "receiver_manifest": {"path": str(receiver_manifest.resolve()), "sha256": sha256(receiver_manifest)},
        "intermediate_spoof_iq": identity, "score_artifacts": score_artifacts,
    }
    write_json(paths["result"], result)
    transfer.remove_single_iq(paths["spoof_iq"], identity, paths["retention"], "composed_spoof_iq")
    print(f"[geometry-aperture] {spec['condition_id']} complete", flush=True)
    return result


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def committed_release() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        git("ls-files", "--error-unmatch", relative)
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=REPO_ROOT).returncode:
            raise ValueError(f"release input is not committed and clean: {relative}")
    return {"head_commit": git("rev-parse", "HEAD"), "input_commits": {path: git("log", "-1", "--format=%H", "--", path) for path in RELEASE_INPUTS}, "runner_sha256": sha256(Path(__file__).resolve())}


def start_release(config_path: Path, output_root: Path, resume: bool) -> tuple[Path, dict[str, Any]]:
    state_path = output_root / "release_state.json"
    commits = committed_release()
    if not resume:
        if output_root.exists():
            raise FileExistsError(output_root)
        output_root.mkdir(parents=True)
        state = {"schema": "gnss-doppler-lab.cgc-rf-geometry-aperture-release-state", "schema_version": 1, "phase": "released_before_outcomes", "started_at_utc": datetime.now(timezone.utc).isoformat(), "config": {"path": str(config_path), "sha256": sha256(config_path)}, "protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)}, "commits": commits, "condition_ids": [row["condition_id"] for row in condition_specs()], "metrics_emitted": False}
        write_json(state_path, state)
        return state_path, state
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["config"]["sha256"] != sha256(config_path) or state["commits"]["runner_sha256"] != commits["runner_sha256"] or state.get("metrics_emitted") is not False:
        raise ValueError("resume release provenance mismatch")
    return state_path, state


def run(config_path: Path, *, resume: bool) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    global_context = validate_config(config)
    state_path, state = start_release(config_path, global_context["output_root"], resume)
    estimators = {taps: _estimator(global_context["controlled"], taps) for taps in APERTURE_TAPS}
    results: list[dict[str, Any]] = []
    common_artifacts: dict[str, Any] = {}
    specs = condition_specs()
    for geometry_id in GEOMETRY_IDS:
        state["phase"] = f"geometry:{geometry_id}:multipath"
        write_json(state_path, state)
        base = global_context["contexts"][geometry_id]
        geometry_root = global_context["output_root"] / "geometries" / geometry_id
        context = {**base, "normal_profile": global_context["normal_profile"], "output_root": geometry_root}
        multipath, common_artifacts[geometry_id] = prepare_multipath_control(geometry_root / "common_multipath", config, context, estimators)
        geometry_specs = [spec for spec in specs if spec["geometry_id"] == geometry_id]
        for distance_m in DISTANCES_M:
            distance_specs = [spec for spec in geometry_specs if spec["distance_m"] == distance_m]
            dpaths = transfer.distance_paths(geometry_root, distance_m)
            if dpaths["retention"].is_file() and not dpaths["counterfeit_iq"].exists():
                for spec in distance_specs:
                    result_path = transfer.condition_paths(geometry_root, spec["condition_id"])["result"]
                    if not result_path.is_file():
                        raise FileNotFoundError("counterfeit removed before both powers completed")
                    results.append(json.loads(result_path.read_text(encoding="utf-8")))
                continue
            state["phase"] = f"geometry:{geometry_id}:distance:{int(distance_m)}"
            write_json(state_path, state)
            pair = component_pair(context["run"], geometry_id, distance_m, config)
            counterfeit, counterfeit_manifest_path, counterfeit_manifest = anchor.ensure_counterfeit(dpaths["root"], pair, context["normal_profile"], repo_path(config["rf_tools"]["simulator"]["path"]))
            for spec in distance_specs:
                state["phase"] = f"condition:{spec['condition_id']}"
                write_json(state_path, state)
                results.append(run_condition(spec, config, context, estimators, multipath, counterfeit, counterfeit_manifest_path, counterfeit_manifest))
            identity = {"path": str(counterfeit.resolve()), "sha256": counterfeit_manifest["counterfeit"]["sha256"], "bytes": counterfeit_manifest["counterfeit"]["bytes"]}
            transfer.remove_single_iq(counterfeit, identity, dpaths["retention"], "shared_distance_counterfeit_iq")
    summaries = [summary for result in results for summary in result["summary_by_aperture"].values()]
    primary = [row for row in summaries if int(row["aperture_taps"]) == 9]
    geometry_evaluations: dict[str, dict[str, Any]] = {}
    for geometry_id in GEOMETRY_IDS:
        geometry_evaluations[geometry_id] = {}
        for power_db in POWERS_DB:
            rows = [row for row in primary if row["geometry_id"] == geometry_id and float(row["final_advantage_db"]) == power_db]
            geometry_evaluations[geometry_id][f"{power_db:+g}_db"] = evaluate_geometry_group(rows, config["geometry_gates"], minimum_bins=int(config["analysis"]["minimum_comparison_bins_per_stream"]))
    full_grid = len(results) == len(specs) and len(summaries) == len(specs) * len(APERTURE_TAPS)
    geometry_supported = full_grid and all(group["mechanism_reproduced"] for geometry in geometry_evaluations.values() for group in geometry.values())
    aperture_evaluation = evaluate_aperture_mechanism(summaries)
    write_csv(global_context["output_root"] / "condition_aperture_summary.csv", summaries)
    state["phase"] = "metrics_emitted"
    state["metrics_emitted"] = True
    state["metrics_emitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, state)
    summary = {
        "schema": "gnss-doppler-lab.cgc-rf-geometry-aperture-validation-result", "schema_version": 1,
        "config": {"path": str(config_path), "sha256": sha256(config_path)}, "protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)}, "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "release_state": {"path": str(state_path.resolve()), "sha256": sha256(state_path)}, "condition_count": len(results), "condition_aperture_summary_count": len(summaries), "full_grid_reported": full_grid,
        "geometry_evaluations_9tap": geometry_evaluations, "all_ten_geometry_power_groups_reproduced": geometry_supported,
        "aperture_evaluation": aperture_evaluation, "aperture_mechanism_supported": aperture_evaluation["aperture_mechanism_supported"],
        "early_boundary_role": "descriptive geometry-dependent outcome; no universal 40/60 m pass gate", "post_release_cell_gate_or_aperture_substitution": False,
        "retention": config["retention"], "claim_boundary": config["claim_boundary"], "common_multipath_artifacts": common_artifacts,
        "artifact": {"path": str((global_context["output_root"] / "condition_aperture_summary.csv").resolve()), "sha256": sha256(global_context["output_root"] / "condition_aperture_summary.csv"), "row_count": len(summaries)},
    }
    summary_path = global_context["output_root"] / "summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), "full_grid_reported": full_grid, "geometry_supported": geometry_supported, "aperture_supported": aperture_evaluation["aperture_mechanism_supported"]}, indent=2, sort_keys=True))
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release-token", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.release_token != RELEASE_TOKEN:
        raise ValueError("explicit release token is required")
    run(args.config.resolve(), resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
