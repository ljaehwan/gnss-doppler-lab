#!/usr/bin/env python3
"""Generate and receive the frozen simulation-v4 train paired partition only."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for search_root in (SCRIPT_ROOT, SRC_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import plan_simulation_v4_paired_split as splitter  # noqa: E402
import run_simulation_v4_normal_independent_validation as normal  # noqa: E402
from gnss_doppler_lab.domain_gap import worst_gate_status  # noqa: E402
from gnss_doppler_lab.gnss_sdr import run_receiver  # noqa: E402
from gnss_doppler_lab.gps_sdr_sim import GpsSdrSimRunner  # noqa: E402
from gnss_doppler_lab.rf_config import (  # noqa: E402
    InputConfig,
    OutputConfig,
    RFGenerationConfig,
    Scenario,
    SimulatorConfig,
    StaticPosition,
    TrajectoryPosition,
)
from gnss_doppler_lab.simulation_v4 import (  # noqa: E402
    SimulationScenario,
    SpoofEvent,
    build_carryoff_rows,
    compare_prefix,
    compose_paired_iq,
)
from gnss_doppler_lab.tracking_feature_windows import (  # noqa: E402
    export_receiver_run_tracking_feature_csv,
)
from gnss_doppler_lab.trajectory import (  # noqa: E402
    generate_trajectory,
    llh_to_enu,
    read_trajectory,
)

DEFAULT_CONFIG = Path(
    "configs/experiments/simulation_v4_paired_train_generation_v1.json"
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0])
    extras = sorted({key for row in rows for key in row} - set(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields + extras, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_hashed_json(source: dict[str, Any], name: str) -> tuple[Path, dict[str, Any]]:
    path = _repo_path(source["path"])
    observed = _sha256(path)
    if observed != source["sha256"]:
        raise ValueError(f"{name} hash mismatch: {observed}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _validate_generation_config(
    config: dict[str, Any],
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    if config.get("version") != 1:
        raise ValueError("unsupported paired train generation config version")
    campaign = config["campaign"]
    if campaign.get("partition") != "train":
        raise ValueError("this runner is hard-limited to the train partition")
    splitter._safe_name(campaign["name"], "campaign.name")
    if config.get("keep_components") is not True:
        raise ValueError("train pilot must retain authentic and counterfeit components")
    minimum = config.get("minimum_feature_rows_per_scenario")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("minimum_feature_rows_per_scenario must be a positive integer")

    split_config_path, split_config = _load_hashed_json(
        config["split_config"], "split config"
    )
    splitter.validate_config(split_config)
    split_manifest_path, split_manifest = _load_hashed_json(
        config["split_record"], "split record"
    )
    if split_manifest.get("config", {}).get("sha256") != config["split_config"]["sha256"]:
        raise ValueError("split record does not pin the requested split config")
    canonical_split_sha = str(config.get("canonical_split_manifest_sha256", ""))
    if (
        split_manifest.get("source_split_manifest", {}).get("sha256") != canonical_split_sha
    ):
        raise ValueError("split record does not pin the canonical split manifest")
    if split_manifest.get("test_release", {}).get("status") != "locked":
        raise ValueError("test partition must remain locked")
    normal_path, normal_config = _load_hashed_json(
        config["normal_profile"], "normal profile"
    )
    normal._validate_config(normal_config)

    profile = split_config["fixed_rf_profile"]
    if (
        int(normal_config["rf_profile"]["rf_sample_rate_hz"])
        != int(profile["rf_sample_rate_hz"])
        or float(normal_config["rf_profile"]["frontend_cutoff_hz"])
        != float(profile["frontend_cutoff_hz"])
        or int(normal_config["receiver"]["frontend_order"])
        != int(profile["frontend_order"])
        or float(normal_config["normal_target_rms"])
        != float(profile["normal_target_rms"])
        or int(normal_config["gnss_sdr"]["tracking_tap_count"])
        != int(profile["tracking_tap_count"])
        or float(normal_config["gnss_sdr"]["tracking_tap_spacing_chips"])
        != float(profile["tracking_tap_spacing_chips"])
        or int(normal_config["features"]["tap_count"])
        != int(profile["feature_tap_count"])
    ):
        raise ValueError("normal generation profile differs from the frozen split profile")

    boundary = config["data_boundary"]
    if boundary.get("allowed_partition") != "train":
        raise ValueError("data boundary may allow only train")
    if boundary.get("validation_pairs_accessed") is not False:
        raise ValueError("validation access flag must remain false")
    if boundary.get("test_pairs_accessed") is not False:
        raise ValueError("test access flag must remain false")
    if set(boundary.get("allowed_texbat_recordings", [])) != {
        "cleanStatic",
        "cleanDynamic",
    }:
        raise ValueError("only cleanStatic/cleanDynamic may be normal fidelity references")
    if set(boundary.get("forbidden_texbat_recordings", [])) != {
        f"ds{index}" for index in range(1, 9)
    }:
        raise ValueError("TEXBAT ds1-ds8 must remain forbidden")

    train_ids = {
        str(pair["paired_group_id"])
        for pair in split_config["pairs"]
        if pair["split"] == "train"
    }
    manifest_ids = set(split_manifest["partitions"]["train"]["paired_group_ids"])
    if train_ids != manifest_ids or len(train_ids) != 6:
        raise ValueError("train pair roster differs between split config and manifest")
    if any(pair["split"] != "train" for pair in split_config["pairs"] if pair["paired_group_id"] in train_ids):
        raise ValueError("non-train pair entered the train roster")
    return (
        split_config_path,
        split_config,
        split_manifest_path,
        split_manifest,
        normal_path,
        normal_config,
    )


def _spoof_event(pair: dict[str, Any]) -> SpoofEvent:
    document = pair["spoofing"]
    return SpoofEvent(
        start_seconds=float(document["start_seconds"]),
        transition_seconds=float(document["transition_seconds"]),
        target_offset_enu_m=tuple(float(value) for value in document["target_offset_enu_m"]),
        initial_advantage_db=float(document["initial_advantage_db"]),
        final_advantage_db=float(document["final_advantage_db"]),
        power_ramp_seconds=float(document["power_ramp_seconds"]),
    )


def _trajectory_position(
    component_dir: Path,
    pair: dict[str, Any],
) -> tuple[StaticPosition | TrajectoryPosition, tuple[tuple[float, float, float, float], ...], dict[str, Any] | None]:
    position = pair["position"]
    duration = int(pair["duration_seconds"])
    if pair["domain"] == "static":
        rows = tuple(
            (
                index / 10.0,
                float(position["latitude_deg"]),
                float(position["longitude_deg"]),
                float(position["altitude_m"]),
            )
            for index in range(duration * 10)
        )
        return (
            StaticPosition(
                float(position["latitude_deg"]),
                float(position["longitude_deg"]),
                float(position["altitude_m"]),
            ),
            rows,
            None,
        )
    motion = pair["motion"]
    path = component_dir / "authentic_trajectory.csv"
    kwargs: dict[str, Any] = {
        "latitude_deg": float(position["latitude_deg"]),
        "longitude_deg": float(position["longitude_deg"]),
        "altitude_m": float(position["altitude_m"]),
        "duration_seconds": duration,
        "speed_mps": float(motion["speed_mps"]),
        "heading_deg": float(motion["heading_deg"]),
    }
    for key in ("radius_m", "leg_length_m", "lane_spacing_m"):
        if key in motion:
            kwargs[key] = float(motion[key])
    metadata = generate_trajectory(str(motion["kind"]), path, **kwargs)
    rows = tuple(read_trajectory(path, duration, "llh"))
    sidecar = path.with_suffix(".json")
    record = {
        "path": str(path.resolve()),
        "sha256": str(metadata["csv_sha256"]),
        "metadata_path": str(sidecar.resolve()),
        "metadata_sha256": _sha256(sidecar),
        "row_count": len(rows),
        "effective": metadata["effective"],
    }
    return (
        TrajectoryPosition(
            path.resolve(),
            "llh",
            rows,
            record["sha256"],
            sidecar.resolve(),
            record["metadata_sha256"],
        ),
        rows,
        record,
    )


def _counterfeit_position(
    component_dir: Path,
    authentic_rows: tuple[tuple[float, float, float, float], ...],
    event: SpoofEvent,
) -> tuple[TrajectoryPosition, dict[str, Any]]:
    rows = build_carryoff_rows(authentic_rows, event)
    path = component_dir / "counterfeit_trajectory.csv"
    payload = "".join(
        f"{time_s:.1f},{latitude:.9f},{longitude:.9f},{altitude:.4f}\n"
        for time_s, latitude, longitude, altitude in rows
    ).encode("ascii")
    _atomic_bytes(path, payload)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = path.with_suffix(".json")
    relative_final = llh_to_enu(*rows[-1][1:], *authentic_rows[-1][1:])
    metadata = {
        "schema": "gnss-doppler-lab.simulation-v4-dynamic-carryoff-trajectory",
        "schema_version": 1,
        "sample_rate_hz": 10,
        "row_count": len(rows),
        "csv_sha256": digest,
        "attack": asdict(event),
        "realized_final_spoof_offset_from_authentic_enu_m": list(relative_final),
    }
    _atomic_json(sidecar, metadata)
    record = {
        "path": str(path.resolve()),
        "sha256": digest,
        "metadata_path": str(sidecar.resolve()),
        "metadata_sha256": _sha256(sidecar),
        "row_count": len(rows),
        "realized_final_spoof_offset_from_authentic_enu_m": list(relative_final),
    }
    return (
        TrajectoryPosition(
            path.resolve(),
            "llh",
            rows,
            digest,
            sidecar.resolve(),
            record["metadata_sha256"],
        ),
        record,
    )


def _component_pair(
    generation_config: dict[str, Any],
    generation_config_sha256: str,
    split_config_sha256: str,
    split_manifest_sha256: str,
    normal_config: dict[str, Any],
    root: Path,
    pair: dict[str, Any],
    simulator: GpsSdrSimRunner,
    *,
    resume: bool,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    pair_id = str(pair["paired_group_id"])
    component_dir = root / "pairs" / pair_id / "components"
    authentic_path = component_dir / "authentic_gps_l1ca_s8_iq.bin"
    counterfeit_path = component_dir / "counterfeit_gps_l1ca_s8_iq.bin"
    manifest_path = component_dir / "manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(manifest_path)
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if document.get("generation_config_sha256") != generation_config_sha256:
            raise ValueError(f"component generation config changed: {pair_id}")
        if document.get("pair") != pair:
            raise ValueError(f"component pair definition changed: {pair_id}")
        for name, path in (("authentic", authentic_path), ("counterfeit", counterfeit_path)):
            if not path.is_file() or _sha256(path) != document["components"][name]["sha256"]:
                raise ValueError(f"component integrity failure: {pair_id}/{name}")
        for trajectory in document["trajectories"].values():
            if trajectory and (
                _sha256(trajectory["path"]) != trajectory["sha256"]
                or _sha256(trajectory["metadata_path"]) != trajectory["metadata_sha256"]
            ):
                raise ValueError(f"trajectory integrity failure: {pair_id}")
        return authentic_path, counterfeit_path, manifest_path, document
    if component_dir.exists() and any(component_dir.iterdir()):
        raise FileExistsError(f"partial component directory: {component_dir}")
    component_dir.mkdir(parents=True, exist_ok=True)

    authentic_position, authentic_rows, authentic_trajectory = _trajectory_position(
        component_dir, pair
    )
    event = _spoof_event(pair)
    counterfeit_position, counterfeit_trajectory = _counterfeit_position(
        component_dir, authentic_rows, event
    )
    sample_rate = int(normal_config["rf_profile"]["rf_sample_rate_hz"])
    base_scenario = Scenario(
        f"{pair_id}-authentic-source",
        "GPS",
        "L1CA",
        normal._run_datetime(pair),
        int(pair["duration_seconds"]),
        authentic_position,
    )
    base_config = RFGenerationConfig(
        version=1,
        scenario=base_scenario,
        input=InputConfig(_repo_path(normal_config["input"]["rinex_nav"])),
        output=OutputConfig(component_dir, sample_rate, "s8_iq"),
        simulator=SimulatorConfig(str(_repo_path(normal_config["simulator"]["executable"]))),
        impairments=normal.ImpairmentConfig(),
    )
    counterfeit_config = replace(
        base_config,
        scenario=replace(
            base_scenario,
            name=f"{pair_id}-counterfeit-source",
            position=counterfeit_position,
        ),
    )
    authentic_log = component_dir / "authentic-gps-sdr-sim.log"
    counterfeit_log = component_dir / "counterfeit-gps-sdr-sim.log"
    authentic_result = simulator.run(base_config, authentic_path, authentic_log)
    counterfeit_result = simulator.run(
        counterfeit_config, counterfeit_path, counterfeit_log
    )
    expected = simulator.expected_output_bytes(base_config)
    if authentic_path.stat().st_size != expected or counterfeit_path.stat().st_size != expected:
        raise RuntimeError(f"component byte contract failed: {pair_id}")
    document = {
        "schema": "gnss-doppler-lab.simulation-v4-paired-components",
        "schema_version": 1,
        "generation_config_sha256": generation_config_sha256,
        "split_config_sha256": split_config_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "pair": pair,
        "components": {
            "authentic": {
                "path": str(authentic_path.resolve()),
                "sha256": _sha256(authentic_path),
                "bytes": authentic_path.stat().st_size,
            },
            "counterfeit": {
                "path": str(counterfeit_path.resolve()),
                "sha256": _sha256(counterfeit_path),
                "bytes": counterfeit_path.stat().st_size,
            },
        },
        "trajectories": {
            "authentic": authentic_trajectory,
            "counterfeit": counterfeit_trajectory,
        },
        "simulator": {
            "identity": simulator.identity,
            "executable": simulator.executable,
            "provenance": simulator.provenance,
            "cli_contract": simulator.cli_contract,
            "authentic_command": authentic_result["command"],
            "counterfeit_command": counterfeit_result["command"],
        },
        "scope": "offline paired train generation only; no RF transmission",
    }
    _atomic_json(manifest_path, document)
    return authentic_path, counterfeit_path, manifest_path, document


def _receiver_run_id(pair: dict[str, Any], member: str) -> str:
    number = str(pair["paired_group_id"]).rsplit("-", 1)[-1]
    suffix = "n" if member == "normal" else "s"
    timestamp = normal._run_datetime(pair).strftime("%Y%m%dT%H%M%SZ")
    return f"simv4pt-p{number}-{suffix}_{timestamp}"


def _compose_pair(
    generation_config: dict[str, Any],
    generation_config_sha256: str,
    split_config_sha256: str,
    split_manifest_sha256: str,
    normal_config: dict[str, Any],
    root: Path,
    pair: dict[str, Any],
    authentic_path: Path,
    counterfeit_path: Path,
    component_manifest_path: Path,
    component_manifest: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    pair_id = str(pair["paired_group_id"])
    pair_root = root / "pairs" / pair_id
    pair_manifest_path = pair_root / "pair_manifest.json"
    members = {
        "normal": f"{pair_id}-normal",
        "spoof": f"{pair_id}-spoof",
    }
    rf_manifests = {
        member: pair_root / "rf" / member / "manifest.json" for member in members
    }
    iq_paths = {
        member: pair_root / "rf" / member / "gps_l1ca_s8_iq.bin"
        for member in members
    }
    if pair_manifest_path.is_file():
        if not resume:
            raise FileExistsError(pair_manifest_path)
        document = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
        if document.get("generation_config_sha256") != generation_config_sha256:
            raise ValueError(f"paired RF generation config changed: {pair_id}")
        if document.get("pair") != pair:
            raise ValueError(f"paired RF pair definition changed: {pair_id}")
        if document.get("component_manifest_sha256") != _sha256(component_manifest_path):
            raise ValueError(f"paired RF component provenance mismatch: {pair_id}")
        for member in members:
            if (
                not rf_manifests[member].is_file()
                or _sha256(rf_manifests[member])
                != document["rf_manifests"][member]["sha256"]
                or not iq_paths[member].is_file()
                or _sha256(iq_paths[member]) != document["iq"][member]["sha256"]
            ):
                raise ValueError(f"paired RF integrity failure: {pair_id}/{member}")
            normal.calibration._receiver_iq_alias(
                root, members[member], iq_paths[member]
            )
        if document.get("paired_prefix_check", {}).get("byte_identical") is not True:
            raise ValueError(f"paired prefix contract failed: {pair_id}")
        return {
            "pair_manifest": pair_manifest_path,
            "rf_manifests": rf_manifests,
            "iq_paths": iq_paths,
            "document": document,
        }
    rf_root = pair_root / "rf"
    if rf_root.exists() and any(rf_root.iterdir()):
        raise FileExistsError(f"partial paired RF directory: {rf_root}")
    for path in iq_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    event = _spoof_event(pair)
    scenarios = (
        SimulationScenario(members["normal"], "steady_normal"),
        SimulationScenario(members["spoof"], "carryoff_spoof", spoofing=event),
    )
    impairment = normal._run_impairment(normal_config, pair)
    composition = compose_paired_iq(
        authentic_path,
        counterfeit_path,
        {scenario.name: iq_paths[member] for member, scenario in zip(members, scenarios, strict=True)},
        scenarios,
        sample_rate_hz=int(normal_config["rf_profile"]["rf_sample_rate_hz"]),
        receiver=impairment,
        normal_target_rms=float(normal_config["normal_target_rms"]),
    )
    sample_rate = int(normal_config["rf_profile"]["rf_sample_rate_hz"])
    prefix = compare_prefix(
        iq_paths["normal"],
        iq_paths["spoof"],
        int(round(event.start_seconds * sample_rate)),
    )
    if not prefix["byte_identical"]:
        raise RuntimeError(f"paired prefix differs before spoof onset: {pair_id}")

    for member, scenario in zip(members, scenarios, strict=True):
        report = composition["scenarios"][scenario.name]
        alias = normal.calibration._receiver_iq_alias(
            root, scenario.name, iq_paths[member]
        )
        truth = (
            {"class": "normal", "event": "steady", "is_spoofing": False}
            if member == "normal"
            else {
                "class": "spoofing",
                "event": "carryoff",
                "is_spoofing": True,
                "spoofing": asdict(event),
            }
        )
        rf_document = {
            "schema_version": 4,
            "run_id": _receiver_run_id(pair, member),
            "scenario": {
                "name": scenario.name,
                "campaign": generation_config["campaign"]["name"],
                "paired_group_id": pair_id,
                "split": "train",
                "utc": pair["utc"],
                "duration_seconds": pair["duration_seconds"],
                "position": pair["position"],
                "motion": pair.get("motion"),
                "domain": pair["domain"],
                **truth,
            },
            "iq": {
                "path": str(alias),
                "canonical_storage_path": str(iq_paths[member].resolve()),
                "sha256": report["sha256"],
                "actual_bytes": report["bytes"],
                "complex_samples": report["complex_samples"],
                "actual_duration_seconds": report["actual_duration_seconds"],
                "rf_sample_rate_hz": sample_rate,
                "sample_format": "s8_iq",
                "channels": 2,
            },
            "simulation_v4": {
                "truth": truth,
                "pair_contract": {
                    "paired_group_id": pair_id,
                    "reference_member": members["normal"],
                    "paired_prefix_check": None if member == "normal" else prefix,
                },
                "receiver": {
                    "requested": impairment.manifest(),
                    "reference": composition["reference"],
                    "processing": composition["processing"],
                },
                "measurements": report,
                "sources": component_manifest["components"],
                "scope": "offline train-partition baseband only; no RF transmission",
            },
            "generation": {
                "generation_config_sha256": generation_config_sha256,
                "split_config_sha256": split_config_sha256,
                "split_manifest_sha256": split_manifest_sha256,
                "component_manifest": str(component_manifest_path.resolve()),
                "component_manifest_sha256": _sha256(component_manifest_path),
                "runner_script_sha256": _sha256(Path(__file__)),
            },
        }
        _atomic_json(rf_manifests[member], rf_document)

    pair_document = {
        "schema": "gnss-doppler-lab.simulation-v4-paired-train-pair",
        "schema_version": 1,
        "generation_config_sha256": generation_config_sha256,
        "split_config_sha256": split_config_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "pair": pair,
        "component_manifest": str(component_manifest_path.resolve()),
        "component_manifest_sha256": _sha256(component_manifest_path),
        "rf_manifests": {
            member: {
                "path": str(rf_manifests[member].resolve()),
                "sha256": _sha256(rf_manifests[member]),
            }
            for member in members
        },
        "iq": {
            member: {
                "path": str(iq_paths[member].resolve()),
                "sha256": composition["scenarios"][scenarios[index].name]["sha256"],
                "bytes": composition["scenarios"][scenarios[index].name]["bytes"],
            }
            for index, member in enumerate(members)
        },
        "paired_prefix_check": {
            "reference": members["normal"],
            "candidate": members["spoof"],
            "onset_seconds": event.start_seconds,
            **prefix,
        },
        "composition": {
            "reference": composition["reference"],
            "processing": composition["processing"],
        },
    }
    _atomic_json(pair_manifest_path, pair_document)
    return {
        "pair_manifest": pair_manifest_path,
        "rf_manifests": rf_manifests,
        "iq_paths": iq_paths,
        "document": pair_document,
    }


def _receiver_features(
    normal_config: dict[str, Any],
    root: Path,
    pair: dict[str, Any],
    member: str,
    scenario_id: str,
    rf_manifest: Path,
    *,
    resume: bool,
) -> tuple[Path, Path, dict[str, Any], list[dict[str, str]]]:
    receiver_root = root / "receiver"
    feature_root = root / "features"
    receiver_root.mkdir(exist_ok=True)
    feature_root.mkdir(exist_ok=True)
    rf_document = json.loads(rf_manifest.read_text(encoding="utf-8"))
    receiver_dir = receiver_root / str(rf_document["run_id"])
    receiver_manifest = receiver_dir / "manifest.json"
    feature_path = feature_root / f"{scenario_id}_tracking_features.csv"
    feature_manifest = feature_path.with_suffix(".manifest.json")
    required = (receiver_manifest, feature_path, feature_manifest)
    if all(path.is_file() for path in required):
        if not resume:
            raise FileExistsError(receiver_manifest)
        receiver_document = json.loads(receiver_manifest.read_text(encoding="utf-8"))
        if receiver_document.get("source", {}).get("rf_manifest_sha256") != _sha256(rf_manifest):
            raise ValueError(f"receiver provenance mismatch: {scenario_id}")
        expected_taps = int(normal_config["gnss_sdr"]["tracking_tap_count"])
        if int(receiver_document.get("tracking", {}).get("tap_count", 3)) != expected_taps:
            raise ValueError(f"receiver tap-count mismatch: {scenario_id}")
        expected_executable = normal.calibration._executable_path(
            normal_config["gnss_sdr"]["executable"]
        )
        command = receiver_document.get("receiver", {}).get("command", [])
        if command and str(Path(command[0]).resolve()) != str(Path(expected_executable).resolve()):
            raise ValueError(f"receiver executable mismatch: {scenario_id}")
        feature_document = json.loads(feature_manifest.read_text(encoding="utf-8"))
        if (
            feature_document.get("receiver_manifest_sha256") != _sha256(receiver_manifest)
            or feature_document.get("feature_csv_sha256") != _sha256(feature_path)
            or int(feature_document.get("tap_count", -1))
            != int(normal_config["features"]["tap_count"])
            or float(feature_document.get("window_s", -1.0))
            != float(normal_config["features"]["window_s"])
            or float(feature_document.get("stride_s", -1.0))
            != float(normal_config["features"]["stride_s"])
            or int(feature_document.get("min_epochs", -1))
            != int(normal_config["features"]["min_epochs"])
        ):
            raise ValueError(f"feature integrity failure: {scenario_id}")
    else:
        if any(path.exists() for path in required) or receiver_dir.exists():
            raise FileExistsError(f"partial receiver/feature output: {scenario_id}")
        print(f"[receiver] {scenario_id}", flush=True)
        receiver_manifest = run_receiver(
            rf_manifest,
            receiver_root,
            executable=normal.calibration._executable_path(
                normal_config["gnss_sdr"]["executable"]
            ),
            channel_count=int(normal_config["gnss_sdr"]["channel_count"]),
            timeout_seconds=int(normal_config["gnss_sdr"]["timeout_seconds"]),
            tracking_tap_count=int(normal_config["gnss_sdr"]["tracking_tap_count"]),
            tracking_tap_spacing_chips=float(
                normal_config["gnss_sdr"]["tracking_tap_spacing_chips"]
            ),
        )
        export_receiver_run_tracking_feature_csv(
            receiver_manifest.parent,
            output_path=feature_path,
            tap_count=int(normal_config["features"]["tap_count"]),
            window_s=float(normal_config["features"]["window_s"]),
            stride_s=float(normal_config["features"]["stride_s"]),
            min_epochs=int(normal_config["features"]["min_epochs"]),
            label="normal" if member == "normal" else "spoofing",
        )
        rows = _read_csv(feature_path)
        _atomic_json(feature_manifest, {
            "schema": "gnss-doppler-lab.simulation-v4-paired-train-features",
            "schema_version": 1,
            "paired_group_id": pair["paired_group_id"],
            "split": "train",
            "member": member,
            "scenario_id": scenario_id,
            "receiver_manifest": str(receiver_manifest.resolve()),
            "receiver_manifest_sha256": _sha256(receiver_manifest),
            "feature_csv": str(feature_path.resolve()),
            "feature_csv_sha256": _sha256(feature_path),
            "row_count": len(rows),
            "tap_count": int(normal_config["features"]["tap_count"]),
            "window_s": float(normal_config["features"]["window_s"]),
            "stride_s": float(normal_config["features"]["stride_s"]),
            "min_epochs": int(normal_config["features"]["min_epochs"]),
        })
    rows = _read_csv(feature_path)
    if not rows:
        raise ValueError(f"zero feature rows: {scenario_id}")
    state = normal.calibration._receiver_state_summary(receiver_manifest.parent)
    return receiver_manifest, feature_path, state, rows


def _event_label(
    member: str,
    event: SpoofEvent,
    window_mid_s: float,
) -> tuple[str, str, int]:
    if member == "normal":
        return "normal", "steady_normal", 0
    if window_mid_s < event.start_seconds:
        return "normal", "pre_event_normal", 0
    if window_mid_s < event.start_seconds + event.transition_seconds:
        return "spoofing", "carryoff_transition", 1
    return "spoofing", "carryoff_final", 1


def _combine_features(
    records: list[dict[str, Any]],
    output: Path,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    combined: list[dict[str, str]] = []
    state_counts: dict[str, int] = {}
    for record in records:
        event = _spoof_event(record["pair"])
        for source in record["rows"]:
            row = dict(source)
            label, event_state, is_spoofing = _event_label(
                record["member"], event, float(row["window_mid_s"])
            )
            row.update({
                "label": label,
                "scenario_name": record["scenario_id"],
                "scenario_kind": (
                    "steady_normal" if record["member"] == "normal" else "carryoff_spoof"
                ),
                "paired_group_id": record["pair"]["paired_group_id"],
                "base_run_fingerprint": record["base_run_fingerprint"],
                "event_state": event_state,
                "is_spoofing": str(is_spoofing),
                "dataset_role": "simulation_v4_paired_train_only",
                "split": "train",
            })
            combined.append(row)
            state_counts[event_state] = state_counts.get(event_state, 0) + 1
    combined.sort(key=lambda row: (
        row["paired_group_id"],
        row["scenario_name"],
        row["run_id"],
        row["prn"],
        int(row["channel"]),
        int(row["segment_index"]),
        int(row["window_index"]),
    ))
    _atomic_csv(output, combined)
    return combined, state_counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args(argv)
    started = time.time()
    config_path = _repo_path(args.config)
    generation_config = json.loads(config_path.read_text(encoding="utf-8"))
    (
        split_config_path,
        split_config,
        split_manifest_path,
        split_manifest,
        normal_config_path,
        normal_config,
    ) = _validate_generation_config(generation_config)
    generation_config_sha = _sha256(config_path)
    split_config_sha = _sha256(split_config_path)
    split_record_sha = _sha256(split_manifest_path)
    split_manifest_sha = generation_config["canonical_split_manifest_sha256"]
    normal_config_sha = _sha256(normal_config_path)
    root = _repo_path(generation_config["output_root"])
    if root.exists() and not args.resume:
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    simulator = GpsSdrSimRunner(
        str(_repo_path(normal_config["simulator"]["executable"]))
    )
    train_pairs = [pair for pair in split_config["pairs"] if pair["split"] == "train"]
    fingerprints = splitter.validate_config(split_config)

    paired_outputs: dict[str, dict[str, Any]] = {}
    for pair in train_pairs:
        pair_id = str(pair["paired_group_id"])
        print(f"[components] {pair_id}", flush=True)
        authentic, counterfeit, component_manifest, component_document = _component_pair(
            generation_config,
            generation_config_sha,
            split_config_sha,
            split_manifest_sha,
            normal_config,
            root,
            pair,
            simulator,
            resume=args.resume,
        )
        print(f"[paired-rf] {pair_id}", flush=True)
        paired_outputs[pair_id] = _compose_pair(
            generation_config,
            generation_config_sha,
            split_config_sha,
            split_manifest_sha,
            normal_config,
            root,
            pair,
            authentic,
            counterfeit,
            component_manifest,
            component_document,
            resume=args.resume,
        )
    if args.generate_only:
        print(root)
        return 0

    feature_records: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    normal_states: dict[str, dict[str, Any]] = {}
    all_states: dict[str, dict[str, Any]] = {}
    minimum_rows = int(generation_config["minimum_feature_rows_per_scenario"])
    for pair in train_pairs:
        pair_id = str(pair["paired_group_id"])
        output = paired_outputs[pair_id]
        artifacts[pair_id] = {
            "pair_manifest": str(output["pair_manifest"]),
            "pair_manifest_sha256": _sha256(output["pair_manifest"]),
            "members": {},
        }
        for member in ("normal", "spoof"):
            scenario_id = f"{pair_id}-{member}"
            receiver_manifest, feature_path, state, rows = _receiver_features(
                normal_config,
                root,
                pair,
                member,
                scenario_id,
                output["rf_manifests"][member],
                resume=args.resume,
            )
            if len(rows) < minimum_rows:
                raise ValueError(
                    f"{scenario_id} has {len(rows)} feature rows below minimum {minimum_rows}"
                )
            all_states[scenario_id] = state
            if member == "normal":
                normal_states[pair_id] = state
            feature_records.append({
                "pair": pair,
                "base_run_fingerprint": fingerprints[pair_id],
                "member": member,
                "scenario_id": scenario_id,
                "rows": rows,
            })
            artifacts[pair_id]["members"][member] = {
                "rf_manifest": str(output["rf_manifests"][member]),
                "rf_manifest_sha256": _sha256(output["rf_manifests"][member]),
                "receiver_manifest": str(receiver_manifest),
                "receiver_manifest_sha256": _sha256(receiver_manifest),
                "feature_csv": str(feature_path),
                "feature_csv_sha256": _sha256(feature_path),
                "feature_rows": len(rows),
                "iq_sha256": output["document"]["iq"][member]["sha256"],
                "iq_bytes": output["document"]["iq"][member]["bytes"],
            }

    dataset_path = root / "train_tracking_features_labeled.csv"
    combined_rows, state_counts = _combine_features(feature_records, dataset_path)
    split_validation = splitter.validate_dataset_rows(
        combined_rows,
        {"pairs": train_pairs},
    )
    group_to_pair = {str(pair["paired_group_id"]): pair for pair in train_pairs}
    normal_rows = [row for row in combined_rows if row["scenario_kind"] == "steady_normal"]
    static_rows = [
        row for row in normal_rows
        if group_to_pair[row["paired_group_id"]]["domain"] == "static"
    ]
    dynamic_rows = [
        row for row in normal_rows
        if group_to_pair[row["paired_group_id"]]["domain"] == "dynamic"
    ]
    real = normal.calibration._load_real_clean(normal_config)
    comparisons = {
        "cleanStatic": normal._comparison(static_rows, real["cleanStatic"], normal_config),
        "cleanDynamic": normal._comparison(dynamic_rows, real["cleanDynamic"], normal_config),
        "cleanCombined": normal._comparison(
            normal_rows,
            real["cleanStatic"] + real["cleanDynamic"],
            normal_config,
        ),
    }
    receiver_state = normal.receiver_state_gate(normal_states, normal_config)
    domain_status = worst_gate_status(
        result["gate_status"] for result in comparisons.values()
    )
    overall_status = worst_gate_status((domain_status, receiver_state["gate_status"]))
    prefix_all = all(
        output["document"]["paired_prefix_check"]["byte_identical"]
        for output in paired_outputs.values()
    )
    if not prefix_all:
        overall_status = "stop"

    pair_rows: list[dict[str, Any]] = []
    for pair in train_pairs:
        pair_id = str(pair["paired_group_id"])
        normal_artifact = artifacts[pair_id]["members"]["normal"]
        spoof_artifact = artifacts[pair_id]["members"]["spoof"]
        pair_rows.append({
            "paired_group_id": pair_id,
            "domain": pair["domain"],
            "motion_kind": splitter._motion_kind(pair),
            "normal_feature_rows": normal_artifact["feature_rows"],
            "spoof_feature_rows": spoof_artifact["feature_rows"],
            "normal_iq_sha256": normal_artifact["iq_sha256"],
            "spoof_iq_sha256": spoof_artifact["iq_sha256"],
            "prefix_byte_identical": paired_outputs[pair_id]["document"]["paired_prefix_check"]["byte_identical"],
            "normal_lock_fraction": normal_states[pair_id]["carrier_lock_above_0_5_fraction"],
            "normal_cn0_median_db_hz": normal_states[pair_id]["cn0_db_hz_median"],
        })
    _atomic_csv(root / "pair_scores.csv", pair_rows)

    summary = {
        "schema": "gnss-doppler-lab.simulation-v4-paired-train-generation",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": generation_config["campaign"],
        "generation_config": {"path": str(config_path), "sha256": generation_config_sha},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
        "split_config": {"path": str(split_config_path), "sha256": split_config_sha},
        "split_record": {"path": str(split_manifest_path), "sha256": split_record_sha},
        "canonical_split_manifest": {
            "path": split_manifest["source_split_manifest"]["path"],
            "sha256": split_manifest_sha,
        },
        "normal_profile": {"path": str(normal_config_path), "sha256": normal_config_sha},
        "data_boundary": {
            **generation_config["data_boundary"],
            "generated_pair_ids": [pair["paired_group_id"] for pair in train_pairs],
            "validation_pair_ids_accessed": [],
            "test_pair_ids_accessed": [],
            "forbidden_texbat_scenarios_accessed": False,
        },
        "pair_count": len(train_pairs),
        "scenario_count": 2 * len(train_pairs),
        "artifacts": artifacts,
        "paired_prefix_all_byte_identical": prefix_all,
        "receiver_states": all_states,
        "normal_receiver_state_gate": receiver_state,
        "normal_fidelity": comparisons,
        "normal_domain_gate_status": domain_status,
        "overall_generation_gate_status": overall_status,
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": _sha256(dataset_path),
            "row_count": len(combined_rows),
            "event_state_counts": state_counts,
            "split_validation": split_validation,
            "role": "train only; normalization and model fitting allowed, threshold selection forbidden",
        },
        "interpretation": (
            "qualified for exploratory train-only model fitting"
            if overall_status in {"pass", "conditional"}
            else "not qualified for model fitting"
        ),
        "elapsed_seconds": round(time.time() - started, 3),
        "limitations": [
            "Only the six preassigned train pairs were generated.",
            "TEXBAT clean recordings remain development normal-fidelity references.",
            "No validation or test pair was generated or accessed.",
            "No detector, preprocessing transform, or decision threshold was fit in this stage.",
        ],
    }
    summary_path = root / "summary.json"
    _atomic_json(summary_path, summary)
    print(json.dumps({
        "summary": str(summary_path),
        "overall_generation_gate_status": overall_status,
        "paired_prefix_all_byte_identical": prefix_all,
        "dataset_rows": len(combined_rows),
        "event_state_counts": state_counts,
        "normal_fidelity": {
            name: {
                "status": result["gate_status"],
                "auc": result["domain_classifier"]["pooled_separability_auc"],
                "median_ks": result["distribution"]["median_ks_statistic"],
            }
            for name, result in comparisons.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
