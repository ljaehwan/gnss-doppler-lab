#!/usr/bin/env python3
"""Run six frozen 25 MHz RF anchors around the CGC observability boundary."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_cgc_rf_challenge_pilot as pilot  # noqa: E402
import run_simulation_v4_paired_train_generation as source  # noqa: E402
from gnss_doppler_lab.peak_mixture_law import parse_gps_sdr_sim_los_table  # noqa: E402
from gnss_doppler_lab.simulation_v4 import SimulationScenario, compose_paired_iq  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/cgc_rf_observability_anchors_v1.json"
PROTOCOL_PATH = REPO_ROOT / "docs/results/cgc_rf_observability_anchors_protocol_v1.md"
RELEASE_TOKEN = "RELEASE-CGC-RF-OBSERVABILITY-ANCHORS-V1"
RELEASE_INPUTS = (
    "configs/experiments/cgc_rf_observability_anchors_v1.json",
    "docs/results/cgc_rf_observability_anchors_protocol_v1.md",
    "scripts/run_cgc_rf_observability_anchors.py",
)


def expected_condition_ids() -> list[str]:
    return [
        "pneg6-d040", "pneg6-d060", "pneg6-d080",
        "ppos3-d040", "ppos3-d060", "ppos3-d080",
    ]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def verify_record(record: dict[str, str], label: str) -> Path:
    path = repo_path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    observed = sha256(path)
    if observed != record["sha256"]:
        raise ValueError(f"{label} hash mismatch: {observed}")
    return path


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    fields = list(rows[0])
    extras = sorted({key for row in rows for key in row} - set(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields + extras, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def comparison_start_seconds(event: dict[str, Any]) -> float:
    return float(event["start_seconds"]) + max(
        float(event["transition_seconds"]), float(event["power_ramp_seconds"])
    ) + 1.0


def evaluate_power_regime(
    rows: list[dict[str, Any]], *, threshold: float, minimum_bins: int
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["separation_m"]))
    if [float(row["separation_m"]) for row in ordered] != [40.0, 60.0, 80.0]:
        raise ValueError("power regime requires exactly 40, 60, and 80 m")
    aucs = [float(row["serial_bin_auc"]) for row in ordered]
    gates = {
        "minimum_comparison_bins": all(
            int(row["multipath_bin_count"]) >= minimum_bins
            and int(row["spoof_bin_count"]) >= minimum_bins
            for row in ordered
        ),
        "40m_auc_below_threshold": aucs[0] < threshold,
        "60m_auc_at_or_above_threshold": aucs[1] >= threshold,
        "80m_auc_at_or_above_threshold": aucs[2] >= threshold,
        "strict_auc_ordering": aucs[0] < aucs[1] < aucs[2],
    }
    return {
        "aucs_by_separation_m": {str(int(row["separation_m"])): float(row["serial_bin_auc"]) for row in ordered},
        "gates": gates,
        "ordering_reproduced": bool(all(gates.values())),
    }


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "gnss-doppler-lab.cgc-rf-observability-anchors-config" or config.get("schema_version") != 1:
        raise ValueError("unsupported RF anchor config")
    experiment = config.get("experiment", {})
    if experiment.get("name") != "cgc-rf-observability-anchors-v1":
        raise ValueError("experiment identity drifted")
    paths: dict[str, Path] = {}
    for key in ("screen_record", "base_pair_config", "normal_profile", "controlled_template"):
        paths[key] = verify_record(config[key], key)
    for key, record in config["pinned_code"].items():
        paths[key] = verify_record(record, key)
    for key, record in config["shared_inputs"].items():
        paths[key] = verify_record(record, key)
    for key in ("simulator", "receiver", "receiver_patch"):
        paths[key] = verify_record(config["rf_tools"][key], key)
    if {
        key: config["rf_tools"][key]
        for key in ("channel_count", "tracking_tap_count", "tracking_tap_spacing_chips", "timeout_seconds")
    } != {
        "channel_count": 11, "tracking_tap_count": 9,
        "tracking_tap_spacing_chips": 0.125, "timeout_seconds": 1200,
    }:
        raise ValueError("RF receiver contract drifted")
    fresh = json.loads(paths["base_pair_config"].read_text(encoding="utf-8"))
    matches = [row for row in fresh["pairs"] if row["paired_group_id"] == config["base_pair_config"]["pair_id"]]
    if len(matches) != 1 or matches[0]["domain"] != "static":
        raise ValueError("base static pair drifted")
    base_pair = matches[0]
    if (
        base_pair["utc"] != "2022-01-01T00:30:00Z"
        or base_pair["receiver_seed"] != 20261001
        or base_pair["duration_seconds"] != 30
    ):
        raise ValueError("base pair time, seed, or duration drifted")
    conditions = config.get("conditions", [])
    if [row.get("condition_id") for row in conditions] != expected_condition_ids():
        raise ValueError("anchor condition roster drifted")
    expected = [
        (-6.0, 40.0, "below_boundary"), (-6.0, 60.0, "at_boundary"), (-6.0, 80.0, "above_boundary"),
        (3.0, 40.0, "below_boundary"), (3.0, 60.0, "at_boundary"), (3.0, 80.0, "above_boundary"),
    ]
    for row, contract in zip(conditions, expected):
        if (row["final_advantage_db"], row["separation_m"], row["screen_role"]) != contract:
            raise ValueError("anchor definition drifted")
        target = np.asarray(row["target_offset_enu_m"], dtype=float)
        if not math.isclose(float(np.linalg.norm(target)), float(row["separation_m"]), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("anchor target norm drifted")
    event = config.get("spoof_event", {})
    if event != {
        "start_seconds": 10.0, "transition_seconds": 5.0,
        "initial_advantage_db": -30.0, "power_ramp_seconds": 5.0,
        "comparison_start_seconds": 16.0,
    } or comparison_start_seconds(event) != 16.0:
        raise ValueError("spoof timing drifted")
    analysis = config.get("analysis", {})
    if analysis.get("primary_serial_bin_auc_threshold") != 0.8 or analysis.get("minimum_comparison_bins_per_stream") != 8:
        raise ValueError("RF evaluation rule drifted")
    if config.get("retention", {}).get("shared_authentic_or_existing_multipath_deleted") is not False:
        raise ValueError("shared input deletion must remain forbidden")
    normal_profile = json.loads(paths["normal_profile"].read_text(encoding="utf-8"))
    controlled = json.loads(paths["controlled_template"].read_text(encoding="utf-8"))
    normal_manifest = json.loads(paths["normal_rf_manifest"].read_text(encoding="utf-8"))
    if int(normal_profile["rf_profile"]["rf_sample_rate_hz"]) != 25_000_000:
        raise ValueError("normal profile is not 25 MHz")
    if normal_manifest["iq"]["sha256"] != config["shared_inputs"]["normal_rf_iq"]["sha256"]:
        raise ValueError("normal manifest and IQ pin disagree")
    output_root = repo_path(config["output_root"])
    if output_root != REPO_ROOT / "artifacts/cgc_rf_observability_anchors_v1":
        raise ValueError("output root drifted")
    return {
        "paths": paths,
        "fresh": fresh,
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


def condition_pair(base_pair: dict[str, Any], condition: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    pair = deepcopy(base_pair)
    pair["paired_group_id"] = condition["condition_id"]
    pair["split"] = "rf_anchor_pilot"
    pair["spoofing"] = {
        "start_seconds": float(event["start_seconds"]),
        "transition_seconds": float(event["transition_seconds"]),
        "target_offset_enu_m": [float(value) for value in condition["target_offset_enu_m"]],
        "initial_advantage_db": float(event["initial_advantage_db"]),
        "final_advantage_db": float(condition["final_advantage_db"]),
        "power_ramp_seconds": float(event["power_ramp_seconds"]),
    }
    return pair


def ensure_counterfeit(
    root: Path, pair: dict[str, Any], normal_profile: dict[str, Any], simulator_path: Path
) -> tuple[Path, Path, dict[str, Any]]:
    component_root = root / "component"
    component_path = component_root / "counterfeit_gps_l1ca_s8_iq.bin"
    log_path = component_root / "counterfeit-gps-sdr-sim.log"
    manifest_path = component_root / "manifest.json"
    if manifest_path.is_file():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if document.get("pair") != pair:
            raise ValueError("counterfeit pair contract changed")
        if not component_path.is_file():
            raise FileNotFoundError("counterfeit IQ was removed before receiver completion")
        if sha256(component_path) != document["counterfeit"]["sha256"]:
            raise ValueError("counterfeit IQ integrity failure")
        return component_path, manifest_path, document
    if component_root.exists() and any(component_root.iterdir()):
        raise FileExistsError(f"partial counterfeit component: {component_root}")
    component_root.mkdir(parents=True, exist_ok=True)
    _, authentic_rows, _ = source._trajectory_position(component_root, pair)
    event = source._spoof_event(pair)
    counterfeit_position, trajectory = source._counterfeit_position(component_root, authentic_rows, event)
    position = pair["position"]
    base = source.RFGenerationConfig(
        version=1,
        scenario=source.Scenario(
            f"{pair['condition_id'] if 'condition_id' in pair else pair['paired_group_id']}-counterfeit-source",
            "GPS", "L1CA", source.normal._run_datetime(pair), int(pair["duration_seconds"]),
            source.StaticPosition(float(position["latitude_deg"]), float(position["longitude_deg"]), float(position["altitude_m"])),
        ),
        input=source.InputConfig(repo_path(normal_profile["input"]["rinex_nav"])),
        output=source.OutputConfig(component_root, int(normal_profile["rf_profile"]["rf_sample_rate_hz"]), "s8_iq"),
        simulator=source.SimulatorConfig(str(simulator_path)),
        impairments=source.normal.ImpairmentConfig(),
    )
    counterfeit_config = replace(base, scenario=replace(base.scenario, position=counterfeit_position))
    runner = source.GpsSdrSimRunner(str(simulator_path))
    result = runner.run(counterfeit_config, component_path, log_path)
    expected = runner.expected_output_bytes(counterfeit_config)
    if component_path.stat().st_size != expected:
        raise RuntimeError("counterfeit byte contract failed")
    document = {
        "schema": "gnss-doppler-lab.cgc-rf-anchor-counterfeit-component",
        "schema_version": 1,
        "pair": pair,
        "trajectory": trajectory,
        "counterfeit": {"path": str(component_path.resolve()), "sha256": sha256(component_path), "bytes": component_path.stat().st_size},
        "log": {"path": str(log_path.resolve()), "sha256": sha256(log_path)},
        "simulator": {"path": str(simulator_path), "sha256": sha256(simulator_path), "command": result["command"]},
    }
    write_json(manifest_path, document)
    return component_path, manifest_path, document


def ensure_spoof_rf(
    root: Path,
    pair: dict[str, Any],
    config: dict[str, Any],
    context: dict[str, Any],
    counterfeit_path: Path,
    counterfeit_manifest_path: Path,
    counterfeit_manifest: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    rf_root = root / "rf"
    iq_path = rf_root / "gps_l1ca_s8_iq.bin"
    manifest_path = rf_root / "manifest.json"
    if manifest_path.is_file():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not iq_path.is_file() or sha256(iq_path) != document["iq"]["sha256"]:
            raise ValueError("composed spoof RF integrity failure")
        return iq_path, manifest_path, document
    if rf_root.exists() and any(rf_root.iterdir()):
        raise FileExistsError(f"partial spoof RF: {rf_root}")
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
        raise RuntimeError("composed spoof pre-onset prefix differs from pinned normal RF")
    truth = {"class": "spoofing", "event": "carryoff", "is_spoofing": True, "spoofing": asdict(event)}
    document = {
        "schema_version": 4,
        "run_id": f"cgc-rf-anchor-{pair['paired_group_id']}",
        "scenario": {
            "name": scenario.name, "campaign": config["experiment"]["name"],
            "paired_group_id": pair["paired_group_id"], "split": "rf_anchor_pilot",
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
            "sources": {
                "authentic": config["shared_inputs"]["authentic_component"],
                "counterfeit": counterfeit_manifest["counterfeit"],
            },
            "scope": "offline receiver-RF boundary pilot; no transmission",
        },
        "generation": {
            "config_sha256": sha256(DEFAULT_CONFIG),
            "counterfeit_manifest": str(counterfeit_manifest_path.resolve()),
            "counterfeit_manifest_sha256": sha256(counterfeit_manifest_path),
        },
    }
    write_json(manifest_path, document)
    return iq_path, manifest_path, document


def condition_paths(root: Path, condition_id: str) -> dict[str, Path]:
    base = root / "conditions" / condition_id
    return {
        "root": base,
        "counterfeit_iq": base / "component/counterfeit_gps_l1ca_s8_iq.bin",
        "spoof_iq": base / "rf/gps_l1ca_s8_iq.bin",
        "result": base / "condition_result.json",
        "retention": base / "retention.json",
    }


def analyze_stream(
    name: str, receiver_manifest: Path, estimator: Any, los: dict[str, tuple[float, float, float]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return pilot._scenario_geometry(
        name, receiver_manifest, estimator, los,
        bin_seconds=float(config["analysis"]["bin_seconds"]),
        minimum_prns=int(config["analysis"]["minimum_prns"]),
    )


def summarize_condition(
    condition: dict[str, Any], spoof_geometry: list[dict[str, Any]], multipath_geometry: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    start = float(config["spoof_event"]["comparison_start_seconds"])
    spoof = [row for row in spoof_geometry if float(row["bin_start_s"]) >= start]
    multipath = [row for row in multipath_geometry if float(row["bin_start_s"]) >= start]
    if not spoof or not multipath:
        raise ValueError("empty RF comparison stream")
    labels = np.r_[np.ones(len(spoof), dtype=int), np.zeros(len(multipath), dtype=int)]
    scores = -np.asarray([
        row["clock_centered_geometry_residual"] for row in spoof + multipath
    ], dtype=float)
    auc = float(roc_auc_score(labels, scores))
    spoof_residual = np.asarray([row["clock_centered_geometry_residual"] for row in spoof])
    multipath_residual = np.asarray([row["clock_centered_geometry_residual"] for row in multipath])
    return {
        **condition,
        "comparison_start_seconds": start,
        "spoof_bin_count": len(spoof),
        "multipath_bin_count": len(multipath),
        "minimum_spoof_prn_count": min(int(row["prn_count"]) for row in spoof),
        "minimum_multipath_prn_count": min(int(row["prn_count"]) for row in multipath),
        "spoof_median_clock_centered_residual": float(np.median(spoof_residual)),
        "multipath_median_clock_centered_residual": float(np.median(multipath_residual)),
        "multipath_minus_spoof_median_residual": float(np.median(multipath_residual) - np.median(spoof_residual)),
        "serial_bin_auc": auc,
    }


def retain_receiver_and_remove_intermediate_iq(
    paths: dict[str, Path], identities: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if (
        paths["retention"].is_file()
        and not paths["counterfeit_iq"].exists()
        and not paths["spoof_iq"].exists()
    ):
        return json.loads(paths["retention"].read_text(encoding="utf-8"))
    removed: list[dict[str, Any]] = []
    for key in ("counterfeit_iq", "spoof_iq"):
        path = paths[key]
        identity = identities[key]
        if path.is_file():
            if sha256(path) != identity["sha256"] or path.stat().st_size != int(identity["bytes"]):
                raise ValueError(f"refusing to remove changed intermediate: {path}")
            path.unlink()
            removed.append({"kind": key, **identity, "removed": True})
        elif not paths["retention"].is_file():
            raise FileNotFoundError(f"intermediate disappeared before retention record: {path}")
    document = {
        "schema": "gnss-doppler-lab.cgc-rf-anchor-retention",
        "schema_version": 1,
        "removed_intermediates": removed,
        "shared_inputs_removed": False,
        "receiver_outputs_retained": True,
        "deterministic_regeneration_inputs_retained": True,
    }
    write_json(paths["retention"], document)
    return document


def run_condition(
    condition: dict[str, Any], config: dict[str, Any], context: dict[str, Any],
    estimator: Any, los: dict[str, tuple[float, float, float]],
    multipath_delays: list[dict[str, Any]], multipath_geometry: list[dict[str, Any]],
) -> dict[str, Any]:
    paths = condition_paths(context["output_root"], condition["condition_id"])
    if paths["result"].is_file():
        result = json.loads(paths["result"].read_text(encoding="utf-8"))
        retain_receiver_and_remove_intermediate_iq(paths, result["intermediate_iq"])
        return result
    pair = condition_pair(context["base_pair"], condition, config["spoof_event"])
    print(f"[rf-anchor] {condition['condition_id']} generating counterfeit", flush=True)
    counterfeit, counterfeit_manifest_path, counterfeit_manifest = ensure_counterfeit(
        paths["root"], pair, context["normal_profile"], context["paths"]["simulator"]
    )
    print(f"[rf-anchor] {condition['condition_id']} composing paired spoof RF", flush=True)
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
    print(f"[rf-anchor] {condition['condition_id']} running GNSS-SDR", flush=True)
    expected_receiver_manifest = (
        paths["root"] / "receiver" / spoof_manifest["run_id"] / "manifest.json"
    )
    receiver_manifest = pilot._ensure_receiver(
        spoof_manifest_path, paths["root"] / "receiver", receiver_config,
        resume=expected_receiver_manifest.is_file(),
    )
    print(f"[rf-anchor] {condition['condition_id']} scoring CGC", flush=True)
    spoof_delays, spoof_geometry = analyze_stream(
        condition["condition_id"], receiver_manifest, estimator, los, config
    )
    enriched_geometry = [{"condition_id": condition["condition_id"], **row} for row in spoof_geometry]
    enriched_delays = [{"condition_id": condition["condition_id"], **row} for row in spoof_delays]
    write_csv(paths["root"] / "spoof_delay_estimates.csv", enriched_delays)
    write_csv(paths["root"] / "spoof_geometry_scores.csv", enriched_geometry)
    summary = summarize_condition(condition, spoof_geometry, multipath_geometry, config)
    identities = {
        "counterfeit_iq": {
            "path": str(counterfeit.resolve()), "sha256": counterfeit_manifest["counterfeit"]["sha256"],
            "bytes": counterfeit_manifest["counterfeit"]["bytes"],
        },
        "spoof_iq": {
            "path": str(spoof_iq.resolve()), "sha256": spoof_manifest["iq"]["sha256"],
            "bytes": spoof_manifest["iq"]["actual_bytes"],
        },
    }
    result = {
        "schema": "gnss-doppler-lab.cgc-rf-anchor-condition-result",
        "schema_version": 1,
        "condition": condition,
        "pair": pair,
        "summary": summary,
        "prefix": spoof_manifest["simulation_v4"]["pair_contract"]["paired_prefix_check"],
        "counterfeit_manifest": {"path": str(counterfeit_manifest_path.resolve()), "sha256": sha256(counterfeit_manifest_path)},
        "spoof_rf_manifest": {"path": str(spoof_manifest_path.resolve()), "sha256": sha256(spoof_manifest_path)},
        "receiver_manifest": {"path": str(receiver_manifest.resolve()), "sha256": sha256(receiver_manifest)},
        "intermediate_iq": identities,
        "score_artifacts": {
            "delays": {"path": str((paths["root"] / "spoof_delay_estimates.csv").resolve()), "sha256": sha256(paths["root"] / "spoof_delay_estimates.csv"), "row_count": len(enriched_delays)},
            "geometry": {"path": str((paths["root"] / "spoof_geometry_scores.csv").resolve()), "sha256": sha256(paths["root"] / "spoof_geometry_scores.csv"), "row_count": len(enriched_geometry)},
        },
    }
    write_json(paths["result"], result)
    retain_receiver_and_remove_intermediate_iq(paths, identities)
    print(f"[rf-anchor] {condition['condition_id']} complete; temporary IQ removed", flush=True)
    return result


def start_release(config_path: Path, config: dict[str, Any], context: dict[str, Any], resume: bool) -> tuple[Path, dict[str, Any]]:
    root = context["output_root"]
    state_path = root / "release_state.json"
    commits = committed_release()
    if not resume:
        if root.exists():
            raise FileExistsError(root)
        root.mkdir(parents=True)
        state = {
            "schema": "gnss-doppler-lab.cgc-rf-observability-anchors-release-state",
            "schema_version": 1,
            "phase": "released_before_anchor_outcomes",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)},
            "commits": commits,
            "condition_ids": expected_condition_ids(),
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
    state_path, state = start_release(config_path, config, context, resume)
    estimator = pilot._estimator(context["controlled"])
    los = parse_gps_sdr_sim_los_table(context["paths"]["authentic_los_log"].read_text(encoding="utf-8"))
    state["phase"] = "common_multipath_analysis"
    write_json(state_path, state)
    multipath_delays, multipath_geometry = analyze_stream(
        "independent_multipath", context["paths"]["multipath_receiver_manifest"], estimator, los, config
    )
    write_csv(context["output_root"] / "common_multipath_delay_estimates.csv", multipath_delays)
    write_csv(context["output_root"] / "common_multipath_geometry_scores.csv", multipath_geometry)
    results: list[dict[str, Any]] = []
    for condition in config["conditions"]:
        state["phase"] = f"condition:{condition['condition_id']}"
        write_json(state_path, state)
        results.append(run_condition(
            condition, config, context, estimator, los, multipath_delays, multipath_geometry
        ))
    summaries = [result["summary"] for result in results]
    evaluations: dict[str, Any] = {}
    for power in (-6.0, 3.0):
        rows = [row for row in summaries if float(row["final_advantage_db"]) == power]
        evaluations[f"{power:+g}_db"] = evaluate_power_regime(
            rows,
            threshold=float(config["analysis"]["primary_serial_bin_auc_threshold"]),
            minimum_bins=int(config["analysis"]["minimum_comparison_bins_per_stream"]),
        )
    write_csv(context["output_root"] / "condition_summary.csv", summaries)
    state["phase"] = "metrics_emitted"
    state["metrics_emitted"] = True
    state["metrics_emitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, state)
    summary = {
        "schema": "gnss-doppler-lab.cgc-rf-observability-anchors-result",
        "schema_version": 1,
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "release_state": {"path": str(state_path.resolve()), "sha256": sha256(state_path)},
        "condition_summaries": summaries,
        "power_regime_evaluations": evaluations,
        "all_power_regime_orderings_reproduced": all(value["ordering_reproduced"] for value in evaluations.values()),
        "retention": config["retention"],
        "claim_boundary": config["claim_boundary"],
        "post_release_tuning_or_anchor_substitution": False,
        "artifacts": {
            "condition_summary": {"path": str((context["output_root"] / "condition_summary.csv").resolve()), "sha256": sha256(context["output_root"] / "condition_summary.csv"), "row_count": len(summaries)},
            "multipath_delays": {"path": str((context["output_root"] / "common_multipath_delay_estimates.csv").resolve()), "sha256": sha256(context["output_root"] / "common_multipath_delay_estimates.csv"), "row_count": len(multipath_delays)},
            "multipath_geometry": {"path": str((context["output_root"] / "common_multipath_geometry_scores.csv").resolve()), "sha256": sha256(context["output_root"] / "common_multipath_geometry_scores.csv"), "row_count": len(multipath_geometry)},
        },
    }
    summary_path = context["output_root"] / "summary.json"
    write_json(summary_path, summary)
    print(json.dumps({
        "summary": str(summary_path),
        "condition_summaries": summaries,
        "power_regime_evaluations": evaluations,
        "all_power_regime_orderings_reproduced": summary["all_power_regime_orderings_reproduced"],
    }, indent=2, sort_keys=True))
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
        print("RF anchor config, six conditions, shared authentic, and common multipath inputs verified")
        return 0
    run(config_path, resume=bool(args.resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
