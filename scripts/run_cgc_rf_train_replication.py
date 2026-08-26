#!/usr/bin/env python3
"""Run the preregistered receiver/RF CGC replication on train pairs 002--006."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    sys.path.insert(0, str(search_root))

import run_cgc_rf_challenge_pilot as pilot  # noqa: E402
from gnss_doppler_lab.peak_mixture_law import (  # noqa: E402
    parse_gps_sdr_sim_los_table,
)
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
from gnss_doppler_lab.trajectory import read_trajectory  # noqa: E402


DEFAULT_CONFIG = (
    REPO_ROOT / "configs/experiments/cgc_rf_train_replication_v1.json"
)
EXPECTED_PAIR_IDS = [f"pv1-pair-{index:03d}" for index in range(2, 7)]
EXPECTED_GATES = {
    "required_pair_count": 5,
    "positive_clock_centered_separation_pair_count": 5,
    "minimum_pair_block_auc": 0.80,
    "minimum_clock_centered_improvement_over_legacy_pair_count": 4,
    "minimum_comparison_bins_per_scenario_per_pair": 5,
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    return (REPO_ROOT / value).resolve()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_pinned(record: dict[str, str], label: str) -> tuple[Path, dict[str, Any]]:
    path = _repo_path(record["path"])
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"{label} hash mismatch")
    return path, json.loads(path.read_text(encoding="utf-8"))


def validate_config(
    config: dict[str, Any], *, verify_pair_inputs: bool = False
) -> dict[str, Any]:
    if config.get("schema") != "gnss-doppler-lab.cgc-rf-train-replication-config":
        raise ValueError("unsupported replication config schema")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported replication config version")

    experiment = config.get("experiment", {})
    if experiment.get("candidate_commit") != (
        "aa7c5bfefd6ef0102146d1f22d04b47b517044a6"
    ):
        raise ValueError("frozen candidate commit drifted")
    if experiment.get("execution_policy") != (
        "one deterministic execution on pairs 002--006; no score, estimator, "
        "aggregation, seed, or gate changes after outcome inspection"
    ):
        raise ValueError("replication execution policy drifted")
    if experiment.get("runner_path") != "scripts/run_cgc_rf_train_replication.py":
        raise ValueError("replication runner path drifted")

    frozen = config.get("frozen_candidate", {})
    pilot_config_path, _ = _load_pinned(
        frozen["pilot_config"], "pilot config"
    )
    pilot_result_path, pilot_result = _load_pinned(
        frozen["pilot_result"], "pilot result"
    )
    clock_path = _repo_path(frozen["clock_centered_module"]["path"])
    if _sha256(clock_path) != frozen["clock_centered_module"]["sha256"]:
        raise ValueError("clock-centered candidate module hash mismatch")
    if frozen.get("residual") != (
        "SSE_full / sum(weight * (delay - weighted_mean(delay))^2)"
    ):
        raise ValueError("clock-centered score law drifted")
    if frozen.get("detection_score") != "negative residual":
        raise ValueError("detection score direction drifted")
    if pilot_result.get("primary_candidate") != "clock_centered_directional":
        raise ValueError("pilot did not freeze the expected primary candidate")

    split_path, split = _load_pinned(config["split_config"], "split config")
    normal_path, normal_profile = _load_pinned(
        config["source_generation"]["normal_profile"], "normal profile"
    )
    generation_path, _ = _load_pinned(
        config["source_generation"]["paired_generation_config"],
        "paired generation config",
    )
    paired_runner = REPO_ROOT / "scripts/run_simulation_v4_paired_train_generation.py"
    if _sha256(paired_runner) != config["source_generation"][
        "paired_generator_sha256"
    ]:
        raise ValueError("paired generation runner hash mismatch")

    controlled_path, controlled = _load_pinned(
        config["analysis"]["controlled_template_config"],
        "controlled template config",
    )
    boundary = config.get("data_boundary", {})
    if boundary.get("authorized_partition") != "train":
        raise ValueError("replication must remain train-only")
    if boundary.get("allowed_pair_ids") != EXPECTED_PAIR_IDS:
        raise ValueError("authorized pair roster drifted")
    if boundary.get("pair_001_access") != "frozen pilot records only":
        raise ValueError("pair-001 access boundary drifted")
    if boundary.get("validation_pairs_accessed") is not False:
        raise ValueError("validation access must remain false")
    if boundary.get("test_pairs_accessed") is not False:
        raise ValueError("test access must remain false")
    if boundary.get("texbat_attack_recordings_accessed") != []:
        raise ValueError("TEXBAT attack access is forbidden")

    selected_pairs = [
        pair for pair in split["pairs"]
        if pair["paired_group_id"] in EXPECTED_PAIR_IDS
    ]
    if [pair["paired_group_id"] for pair in selected_pairs] != EXPECTED_PAIR_IDS:
        raise ValueError("split config does not contain the frozen pair order")
    if any(pair.get("split") != "train" for pair in selected_pairs):
        raise ValueError("authorized replication pair is not train")
    if [pair.get("domain") for pair in selected_pairs] != [
        "static", "dynamic", "dynamic", "dynamic", "dynamic"
    ]:
        raise ValueError("static/dynamic replication roster drifted")

    multipath = config.get("multipath", {})
    if list(multipath.get("seed_by_pair", {})) != EXPECTED_PAIR_IDS:
        raise ValueError("multipath seed roster drifted")
    expected_seeds = {
        pair_id: 2026091200 + int(pair_id[-3:])
        for pair_id in EXPECTED_PAIR_IDS
    }
    if multipath.get("seed_by_pair") != expected_seeds:
        raise ValueError("multipath seeds drifted")
    if multipath.get("delay_chips_range") != [0.12, 0.45]:
        raise ValueError("multipath delay range drifted")
    if multipath.get("amplitude_range") != [0.20, 0.70]:
        raise ValueError("multipath amplitude range drifted")
    simulator = multipath["simulator_executable"]
    simulator_path = _repo_path(simulator["path"])
    if _sha256(simulator_path) != simulator["sha256"]:
        raise ValueError("multipath simulator executable hash mismatch")
    patch = multipath["simulator_patch"]
    if _sha256(_repo_path(patch["path"])) != patch["sha256"]:
        raise ValueError("multipath simulator patch hash mismatch")

    receiver = config.get("gnss_sdr", {})
    receiver_path = _repo_path(receiver["executable"]["path"])
    if _sha256(receiver_path) != receiver["executable"]["sha256"]:
        raise ValueError("GNSS-SDR executable hash mismatch")
    if _sha256(_repo_path(receiver["patch"]["path"])) != receiver["patch"]["sha256"]:
        raise ValueError("GNSS-SDR patch hash mismatch")
    if {
        "channel_count": receiver.get("channel_count"),
        "tracking_tap_count": receiver.get("tracking_tap_count"),
        "tracking_tap_spacing_chips": receiver.get(
            "tracking_tap_spacing_chips"
        ),
    } != {
        "channel_count": 11,
        "tracking_tap_count": 9,
        "tracking_tap_spacing_chips": 0.125,
    }:
        raise ValueError("receiver configuration drifted")

    analysis = config.get("analysis", {})
    if analysis.get("bin_seconds") != 1.0 or analysis.get("minimum_prns") != 8:
        raise ValueError("analysis binning drifted")
    if analysis.get("comparison_start_policy") != (
        "spoof start + max(transition, power ramp) + 1 second"
    ):
        raise ValueError("comparison boundary drifted")
    evaluation = config.get("evaluation", {})
    if evaluation.get("support_gates") != EXPECTED_GATES:
        raise ValueError("support gates drifted")
    if evaluation.get("bootstrap_seed") != 2026091299:
        raise ValueError("bootstrap seed drifted")
    if evaluation.get("bootstrap_repetitions") != 10000:
        raise ValueError("bootstrap repetitions drifted")
    if evaluation.get("threshold_fitting") is not False:
        raise ValueError("threshold fitting is forbidden")

    output_root = _repo_path(config["output_root"])
    if output_root.parent != REPO_ROOT / "artifacts":
        raise ValueError("replication output must stay under artifacts")

    if verify_pair_inputs:
        for pair in selected_pairs:
            paths = source_paths(config, pair["paired_group_id"])
            for name, path in paths.items():
                exists = path.is_dir() if name == "pair_root" else path.is_file()
                if not exists:
                    raise FileNotFoundError(f"missing {name}: {path}")

    return {
        "pilot_config_path": pilot_config_path,
        "pilot_result_path": pilot_result_path,
        "split_path": split_path,
        "split": split,
        "normal_profile_path": normal_path,
        "normal_profile": normal_profile,
        "paired_generation_path": generation_path,
        "controlled_path": controlled_path,
        "controlled": controlled,
        "pairs": selected_pairs,
        "simulator_path": simulator_path,
        "receiver_path": receiver_path,
        "output_root": output_root,
    }


def source_paths(config: dict[str, Any], pair_id: str) -> dict[str, Path]:
    source = config["source_generation"]
    root = _repo_path(source["pair_root_template"].format(pair_id=pair_id))
    return {
        "pair_root": root,
        "normal_manifest": root / source["normal_manifest_relative"],
        "spoof_manifest": root / source["spoof_manifest_relative"],
        "component_manifest": root / source["component_manifest_relative"],
        "los_log": root / source["los_log_relative"],
    }


def _trajectory_position(
    pair: dict[str, Any],
    source_component: dict[str, Any],
) -> tuple[StaticPosition | TrajectoryPosition, dict[str, Any] | None]:
    position = pair["position"]
    if pair["domain"] == "static":
        if source_component["trajectories"]["authentic"] is not None:
            raise ValueError("static pair unexpectedly records a trajectory")
        return StaticPosition(
            float(position["latitude_deg"]),
            float(position["longitude_deg"]),
            float(position["altitude_m"]),
        ), None

    record = source_component["trajectories"]["authentic"]
    if not isinstance(record, dict):
        raise ValueError("dynamic pair is missing authentic trajectory provenance")
    path = Path(record["path"]).resolve()
    metadata_path = Path(record["metadata_path"]).resolve()
    if _sha256(path) != record["sha256"]:
        raise ValueError("authentic trajectory CSV hash mismatch")
    if _sha256(metadata_path) != record["metadata_sha256"]:
        raise ValueError("authentic trajectory metadata hash mismatch")
    rows = tuple(read_trajectory(path, float(pair["duration_seconds"]), "llh"))
    if len(rows) != int(record["row_count"]):
        raise ValueError("authentic trajectory row count mismatch")
    trajectory = TrajectoryPosition(
        path=path,
        coordinate_system="llh",
        rows=rows,
        csv_sha256=record["sha256"],
        metadata_path=metadata_path,
        metadata_sha256=record["metadata_sha256"],
    )
    return trajectory, record


def _ensure_component(
    pair_root: Path,
    config: dict[str, Any],
    config_path: Path,
    pair: dict[str, Any],
    normal_profile: dict[str, Any],
    *,
    resume: bool,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    pair_id = pair["paired_group_id"]
    sources = source_paths(config, pair_id)
    source_component_path = sources["component_manifest"]
    source_component = json.loads(
        source_component_path.read_text(encoding="utf-8")
    )
    if source_component.get("pair") != pair:
        raise ValueError(f"source component pair mismatch: {pair_id}")
    source_los_text = sources["los_log"].read_text(encoding="utf-8")
    source_los = parse_gps_sdr_sim_los_table(source_los_text)
    prns = [int(prn[1:]) for prn in source_los]
    minimum_prns = int(config["analysis"]["minimum_prns"])
    if len(prns) < minimum_prns:
        raise ValueError(f"too few startup LOS PRNs for {pair_id}")

    multipath = config["multipath"]
    echoes = independent_echoes(
        prns,
        seed=int(multipath["seed_by_pair"][pair_id]),
        delay_chips_range=tuple(multipath["delay_chips_range"]),
        amplitude_range=tuple(multipath["amplitude_range"]),
    )
    simulator = _repo_path(multipath["simulator_executable"]["path"])
    component_dir = pair_root / "component"
    iq_path = component_dir / "multipath_gps_l1ca_s8_iq.bin"
    log_path = component_dir / "gps-sdr-sim.log"
    manifest_path = component_dir / "manifest.json"
    config_sha256 = _sha256(config_path)

    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(manifest_path)
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks = (
            document.get("config_sha256") == config_sha256,
            document.get("pair") == pair,
            document.get("source_component_manifest", {}).get("sha256")
            == _sha256(source_component_path),
            document.get("source_los", {}).get("sha256")
            == _sha256(sources["los_log"]),
            document.get("simulator", {}).get("executable_sha256")
            == _sha256(simulator),
            document.get("simulator", {}).get("patch_sha256")
            == multipath["simulator_patch"]["sha256"],
            iq_path.is_file()
            and _sha256(iq_path) == document.get("iq", {}).get("sha256"),
            log_path.is_file()
            and _sha256(log_path) == document.get("log", {}).get("sha256"),
        )
        if not all(checks):
            raise ValueError(f"component resume provenance mismatch: {pair_id}")
        return iq_path, log_path, manifest_path, document
    if component_dir.exists() and any(component_dir.iterdir()):
        raise FileExistsError(f"partial component directory: {component_dir}")

    position, trajectory_record = _trajectory_position(pair, source_component)
    sample_rate = int(normal_profile["rf_profile"]["rf_sample_rate_hz"])
    rf_config = RFGenerationConfig(
        version=1,
        scenario=Scenario(
            name=f"{pair_id}-independent-multipath",
            constellation="GPS",
            signal="L1CA",
            utc=datetime.fromisoformat(pair["utc"].replace("Z", "+00:00")),
            duration_seconds=int(pair["duration_seconds"]),
            position=position,
        ),
        input=InputConfig(_repo_path(normal_profile["input"]["rinex_nav"])),
        output=OutputConfig(component_dir, sample_rate, "s8_iq"),
        simulator=SimulatorConfig(str(simulator)),
        impairments=ImpairmentConfig(),
    )
    runner = PrnMultipathGpsSdrSimRunner(str(simulator), echoes)
    result = runner.run(rf_config, iq_path, log_path)
    if result["actual_bytes"] != runner.expected_output_bytes(rf_config):
        raise ValueError(f"component byte contract failed: {pair_id}")
    component_los = parse_gps_sdr_sim_los_table(
        log_path.read_text(encoding="utf-8")
    )
    if set(source_los) != set(component_los) or any(
        not np.allclose(
            source_los[prn], component_los[prn], rtol=0.0, atol=1e-12
        )
        for prn in source_los
    ):
        raise ValueError(f"startup LOS mismatch: {pair_id}")

    document = {
        "schema": "gnss-doppler-lab.cgc-rf-train-replication-component",
        "schema_version": 1,
        "config": str(config_path),
        "config_sha256": config_sha256,
        "pair": pair,
        "source_component_manifest": {
            "path": str(source_component_path.resolve()),
            "sha256": _sha256(source_component_path),
        },
        "source_los": {
            "path": str(sources["los_log"].resolve()),
            "sha256": _sha256(sources["los_log"]),
            "prns": list(source_los),
        },
        "trajectory": trajectory_record,
        "simulator": {
            "executable": str(simulator),
            "executable_sha256": _sha256(simulator),
            "upstream_commit": multipath["simulator_upstream_commit"],
            "patch": str(
                _repo_path(multipath["simulator_patch"]["path"])
            ),
            "patch_sha256": multipath["simulator_patch"]["sha256"],
            "cli_contract": runner.cli_contract,
            "command": result["command"],
        },
        "multipath": result["multipath"],
        "iq": {
            "path": str(iq_path.resolve()),
            "sha256": _sha256(iq_path),
            "bytes": iq_path.stat().st_size,
            "rf_sample_rate_hz": sample_rate,
        },
        "log": {
            "path": str(log_path.resolve()),
            "sha256": _sha256(log_path),
        },
        "scope": "unused-train pair receiver/RF replication only",
    }
    _write_json(manifest_path, document)
    return iq_path, log_path, manifest_path, document


def _validate_rf_resume(
    rf_manifest: Path,
    component_document: dict[str, Any],
    normal_manifest: Path,
) -> dict[str, Any]:
    document = json.loads(rf_manifest.read_text(encoding="utf-8"))
    source = document["simulation_v4"]["source_component"]
    receiver = document["simulation_v4"]["receiver"]
    if source["sha256"] != component_document["iq"]["sha256"]:
        raise ValueError("RF source component hash mismatch")
    if source["bytes"] != component_document["iq"]["bytes"]:
        raise ValueError("RF source component byte count mismatch")
    if receiver["reference_source_manifest_sha256"] != _sha256(normal_manifest):
        raise ValueError("RF reference source manifest hash mismatch")
    iq_path = rf_manifest.parent / document["iq"]["path"]
    if _sha256(iq_path) != document["iq"]["sha256"]:
        raise ValueError("RF IQ hash mismatch")
    return document


def _ensure_pair(
    config: dict[str, Any],
    config_path: Path,
    context: dict[str, Any],
    pair: dict[str, Any],
    *,
    resume: bool,
) -> Path:
    pair_id = pair["paired_group_id"]
    pair_root = context["output_root"] / "pairs" / pair_id
    sources = source_paths(config, pair_id)
    for name in ("normal_manifest", "spoof_manifest", "component_manifest", "los_log"):
        if not sources[name].is_file():
            raise FileNotFoundError(sources[name])

    component, component_log, component_manifest, component_document = (
        _ensure_component(
            pair_root,
            config,
            config_path,
            pair,
            context["normal_profile"],
            resume=resume,
        )
    )
    runtime_config = {
        "source_normal_rf_manifest": str(
            sources["normal_manifest"].resolve()
        )
    }
    _, multipath_rf_manifest, _ = pilot._ensure_rf(
        pair_root,
        runtime_config,
        pair,
        context["normal_profile"],
        component,
        component_document,
        resume=resume,
    )
    _validate_rf_resume(
        multipath_rf_manifest, component_document, sources["normal_manifest"]
    )

    receiver_config = {
        "executable": config["gnss_sdr"]["executable"]["path"],
        "channel_count": config["gnss_sdr"]["channel_count"],
        "timeout_seconds": config["gnss_sdr"]["timeout_seconds"],
        "tracking_tap_spacing_chips": config["gnss_sdr"][
            "tracking_tap_spacing_chips"
        ],
    }
    receiver_root = pair_root / "receiver"
    multipath_receiver = pilot._ensure_receiver(
        multipath_rf_manifest, receiver_root, receiver_config, resume=resume
    )
    spoof_receiver = pilot._ensure_receiver(
        sources["spoof_manifest"], receiver_root, receiver_config, resume=resume
    )
    runtime = {
        "schema": "gnss-doppler-lab.cgc-rf-train-replication-runtime",
        "schema_version": 1,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "pair": pair,
        "sources": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for name, path in sources.items()
            if name != "pair_root"
        },
        "component_manifest": {
            "path": str(component_manifest.resolve()),
            "sha256": _sha256(component_manifest),
        },
        "component_log": {
            "path": str(component_log.resolve()),
            "sha256": _sha256(component_log),
        },
        "multipath_rf_manifest": {
            "path": str(multipath_rf_manifest.resolve()),
            "sha256": _sha256(multipath_rf_manifest),
        },
        "multipath_receiver_manifest": {
            "path": str(multipath_receiver.resolve()),
            "sha256": _sha256(multipath_receiver),
        },
        "spoof_receiver_manifest": {
            "path": str(spoof_receiver.resolve()),
            "sha256": _sha256(spoof_receiver),
        },
        "data_boundary": config["data_boundary"],
    }
    runtime_path = pair_root / "runtime_manifest.json"
    _write_json(runtime_path, runtime)
    return runtime_path


def _load_runtime(
    config_path: Path, pair_root: Path, pair: dict[str, Any]
) -> dict[str, Any]:
    path = pair_root / "runtime_manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("config_sha256") != _sha256(config_path):
        raise ValueError("runtime config hash mismatch")
    if document.get("pair") != pair:
        raise ValueError("runtime pair mismatch")
    for section in (
        "sources",
        "component_manifest",
        "component_log",
        "multipath_rf_manifest",
        "multipath_receiver_manifest",
        "spoof_receiver_manifest",
    ):
        record = document[section]
        records = record.values() if section == "sources" else (record,)
        for item in records:
            if _sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"runtime provenance mismatch: {section}")
    return document


def comparison_start_seconds(pair: dict[str, Any]) -> float:
    event = pair["spoofing"]
    return float(event["start_seconds"]) + max(
        float(event["transition_seconds"]),
        float(event["power_ramp_seconds"]),
    ) + 1.0


def _analyze_pair(
    config: dict[str, Any],
    config_path: Path,
    context: dict[str, Any],
    pair: dict[str, Any],
    estimator,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    pair_id = pair["paired_group_id"]
    pair_root = context["output_root"] / "pairs" / pair_id
    runtime = _load_runtime(config_path, pair_root, pair)
    source_los_path = Path(runtime["sources"]["los_log"]["path"])
    component_log = Path(runtime["component_log"]["path"])
    source_los = parse_gps_sdr_sim_los_table(
        source_los_path.read_text(encoding="utf-8")
    )
    component_los = parse_gps_sdr_sim_los_table(
        component_log.read_text(encoding="utf-8")
    )
    if set(source_los) != set(component_los) or any(
        not np.allclose(
            source_los[prn], component_los[prn], rtol=0.0, atol=1e-12
        )
        for prn in source_los
    ):
        raise ValueError(f"analysis startup LOS mismatch: {pair_id}")

    analysis = config["analysis"]
    delay_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    receiver_records = (
        (
            "independent_multipath",
            Path(runtime["multipath_receiver_manifest"]["path"]),
        ),
        ("carryoff_spoof", Path(runtime["spoof_receiver_manifest"]["path"])),
    )
    for scenario, receiver_manifest in receiver_records:
        delays, geometry = pilot._scenario_geometry(
            scenario,
            receiver_manifest,
            estimator,
            source_los,
            bin_seconds=float(analysis["bin_seconds"]),
            minimum_prns=int(analysis["minimum_prns"]),
        )
        delay_rows.extend(
            {"pair_id": pair_id, "domain": pair["domain"], **row}
            for row in delays
        )
        geometry_rows.extend(
            {"pair_id": pair_id, "domain": pair["domain"], **row}
            for row in geometry
        )

    start = comparison_start_seconds(pair)
    for row in geometry_rows:
        row["comparison_eligible"] = int(float(row["bin_start_s"]) >= start)
    comparison = [row for row in geometry_rows if row["comparison_eligible"]]
    scenario_rows = {
        scenario: [
            row for row in comparison if row["scenario"] == scenario
        ]
        for scenario, _ in receiver_records
    }
    if any(not rows for rows in scenario_rows.values()):
        raise ValueError(f"empty comparison scenario: {pair_id}")

    def median(scenario: str, field: str) -> float:
        return float(np.median([
            float(row[field]) for row in scenario_rows[scenario]
        ]))

    legacy_mp = median("independent_multipath", "complex_geometry_residual")
    legacy_sp = median("carryoff_spoof", "complex_geometry_residual")
    centered_mp = median(
        "independent_multipath", "clock_centered_geometry_residual"
    )
    centered_sp = median(
        "carryoff_spoof", "clock_centered_geometry_residual"
    )
    summary = {
        "pair_id": pair_id,
        "domain": pair["domain"],
        "comparison_start_seconds": start,
        "startup_los_prn_count": len(source_los),
        "multipath_comparison_bin_count": len(
            scenario_rows["independent_multipath"]
        ),
        "spoof_comparison_bin_count": len(scenario_rows["carryoff_spoof"]),
        "legacy_multipath_median_residual": legacy_mp,
        "legacy_spoof_median_residual": legacy_sp,
        "legacy_separation": legacy_mp - legacy_sp,
        "clock_centered_multipath_median_residual": centered_mp,
        "clock_centered_spoof_median_residual": centered_sp,
        "clock_centered_separation": centered_mp - centered_sp,
        "clock_centered_improvement_over_legacy": (
            centered_mp - centered_sp - (legacy_mp - legacy_sp)
        ),
        "direction_supported": centered_mp > centered_sp,
        "runtime_manifest_sha256": _sha256(
            pair_root / "runtime_manifest.json"
        ),
    }
    pair_scenario_rows = [
        {
            "pair_id": pair_id,
            "domain": pair["domain"],
            "scenario": scenario,
            "label_spoof": int(scenario == "carryoff_spoof"),
            "legacy_median_residual": median(
                scenario, "complex_geometry_residual"
            ),
            "clock_centered_median_residual": median(
                scenario, "clock_centered_geometry_residual"
            ),
            "comparison_bin_count": len(scenario_rows[scenario]),
        }
        for scenario, _ in receiver_records
    ]
    pair_analysis = pair_root / "analysis"
    _write_csv(pair_analysis / "delay_estimates.csv", delay_rows)
    _write_csv(pair_analysis / "geometry_scores.csv", geometry_rows)
    _write_json(pair_analysis / "summary.json", summary)
    return summary, delay_rows, geometry_rows, pair_scenario_rows


def _pair_block_auc(
    multipath_residual: np.ndarray, spoof_residual: np.ndarray
) -> float:
    count = len(multipath_residual)
    labels = np.concatenate((
        np.zeros(count, dtype=np.int64),
        np.ones(count, dtype=np.int64),
    ))
    scores = -np.concatenate((multipath_residual, spoof_residual))
    return float(roc_auc_score(labels, scores))


def evaluate_pair_summaries(
    pair_summaries: list[dict[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_repetitions: int,
    gates: dict[str, Any],
) -> dict[str, Any]:
    multipath = np.asarray([
        row["clock_centered_multipath_median_residual"]
        for row in pair_summaries
    ], dtype=np.float64)
    spoof = np.asarray([
        row["clock_centered_spoof_median_residual"]
        for row in pair_summaries
    ], dtype=np.float64)
    separation = multipath - spoof
    legacy_separation = np.asarray([
        row["legacy_separation"] for row in pair_summaries
    ], dtype=np.float64)
    pair_auc = _pair_block_auc(multipath, spoof)

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_separation = np.empty(bootstrap_repetitions, dtype=np.float64)
    bootstrap_auc = np.empty(bootstrap_repetitions, dtype=np.float64)
    count = len(pair_summaries)
    for repetition in range(bootstrap_repetitions):
        indices = rng.integers(0, count, size=count)
        bootstrap_separation[repetition] = float(
            np.median(separation[indices])
        )
        bootstrap_auc[repetition] = _pair_block_auc(
            multipath[indices], spoof[indices]
        )

    positive_count = int(np.count_nonzero(separation > 0.0))
    improvement_count = int(np.count_nonzero(
        separation > legacy_separation
    ))
    minimum_bins = min(
        min(
            int(row["multipath_comparison_bin_count"]),
            int(row["spoof_comparison_bin_count"]),
        )
        for row in pair_summaries
    )
    gate_records = {
        "required_pair_count": {
            "observed": count,
            "required": int(gates["required_pair_count"]),
            "passed": count == int(gates["required_pair_count"]),
        },
        "positive_clock_centered_separation_pair_count": {
            "observed": positive_count,
            "required": int(
                gates["positive_clock_centered_separation_pair_count"]
            ),
            "passed": positive_count >= int(
                gates["positive_clock_centered_separation_pair_count"]
            ),
        },
        "minimum_pair_block_auc": {
            "observed": pair_auc,
            "required": float(gates["minimum_pair_block_auc"]),
            "passed": pair_auc >= float(gates["minimum_pair_block_auc"]),
        },
        "minimum_clock_centered_improvement_over_legacy_pair_count": {
            "observed": improvement_count,
            "required": int(
                gates[
                    "minimum_clock_centered_improvement_over_legacy_pair_count"
                ]
            ),
            "passed": improvement_count >= int(
                gates[
                    "minimum_clock_centered_improvement_over_legacy_pair_count"
                ]
            ),
        },
        "minimum_comparison_bins_per_scenario_per_pair": {
            "observed": minimum_bins,
            "required": int(
                gates["minimum_comparison_bins_per_scenario_per_pair"]
            ),
            "passed": minimum_bins >= int(
                gates["minimum_comparison_bins_per_scenario_per_pair"]
            ),
        },
    }
    all_passed = all(record["passed"] for record in gate_records.values())
    return {
        "pair_count": count,
        "pair_block_auc": pair_auc,
        "pair_level_clock_centered_separations": separation.tolist(),
        "median_pair_separation": float(np.median(separation)),
        "median_pair_separation_bootstrap_95_percentile_interval": [
            float(value) for value in np.percentile(
                bootstrap_separation, [2.5, 97.5]
            )
        ],
        "pair_block_auc_bootstrap_95_percentile_interval": [
            float(value) for value in np.percentile(
                bootstrap_auc, [2.5, 97.5]
            )
        ],
        "positive_separation_pair_count": positive_count,
        "clock_centered_improvement_over_legacy_pair_count": (
            improvement_count
        ),
        "gates": gate_records,
        "all_support_gates_passed": all_passed,
        "status": (
            "supported_on_unused_train_replication_requires_locked_test"
            if all_passed
            else "not_supported_on_unused_train_replication"
        ),
    }


def _analyze_all(
    config: dict[str, Any],
    config_path: Path,
    context: dict[str, Any],
) -> dict[str, Any]:
    estimator = pilot._estimator(context["controlled"])
    pair_summaries: list[dict[str, Any]] = []
    all_delays: list[dict[str, Any]] = []
    all_geometry: list[dict[str, Any]] = []
    pair_scenario_rows: list[dict[str, Any]] = []
    for pair in context["pairs"]:
        summary, delays, geometry, scenarios = _analyze_pair(
            config, config_path, context, pair, estimator
        )
        pair_summaries.append(summary)
        all_delays.extend(delays)
        all_geometry.extend(geometry)
        pair_scenario_rows.extend(scenarios)

    analysis_root = context["output_root"] / "analysis"
    delay_path = analysis_root / "delay_estimates.csv"
    geometry_path = analysis_root / "geometry_scores.csv"
    pair_path = analysis_root / "pair_summary.csv"
    scenario_path = analysis_root / "pair_scenario_medians.csv"
    _write_csv(delay_path, all_delays)
    _write_csv(geometry_path, all_geometry)
    _write_csv(pair_path, pair_summaries)
    _write_csv(scenario_path, pair_scenario_rows)

    eligible = [row for row in all_geometry if row["comparison_eligible"]]
    labels = np.asarray([
        int(row["scenario"] == "carryoff_spoof") for row in eligible
    ])
    pooled_auc = {
        "legacy_zero_referenced": float(roc_auc_score(
            labels,
            -np.asarray([
                float(row["complex_geometry_residual"]) for row in eligible
            ]),
        )),
        "clock_centered_directional": float(roc_auc_score(
            labels,
            -np.asarray([
                float(row["clock_centered_geometry_residual"])
                for row in eligible
            ]),
        )),
    }
    evaluation_config = config["evaluation"]
    primary = evaluate_pair_summaries(
        pair_summaries,
        bootstrap_seed=int(evaluation_config["bootstrap_seed"]),
        bootstrap_repetitions=int(
            evaluation_config["bootstrap_repetitions"]
        ),
        gates=evaluation_config["support_gates"],
    )
    result = {
        "schema": "gnss-doppler-lab.cgc-rf-train-replication-result",
        "schema_version": 1,
        "role": config["experiment"]["role"],
        "execution_policy": config["experiment"]["execution_policy"],
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "frozen_candidate": config["frozen_candidate"],
        "pairs": pair_summaries,
        "primary_pair_block_evaluation": primary,
        "secondary_serial_bin_auc": pooled_auc,
        "artifacts": {
            "delay_estimates": {
                "path": str(delay_path.resolve()),
                "sha256": _sha256(delay_path),
                "row_count": len(all_delays),
            },
            "geometry_scores": {
                "path": str(geometry_path.resolve()),
                "sha256": _sha256(geometry_path),
                "row_count": len(all_geometry),
            },
            "pair_summary": {
                "path": str(pair_path.resolve()),
                "sha256": _sha256(pair_path),
                "row_count": len(pair_summaries),
            },
            "pair_scenario_medians": {
                "path": str(scenario_path.resolve()),
                "sha256": _sha256(scenario_path),
                "row_count": len(pair_scenario_rows),
            },
        },
        "data_boundary": config["data_boundary"],
        "threshold_fitted": False,
        "claim_boundary": config["claim_boundary"],
        "next_gate": (
            "if supported, freeze a still-locked test protocol before any "
            "test access; the previously inspected validation partition "
            "cannot validate this post-hoc candidate"
        ),
    }
    summary_path = context["output_root"] / "summary.json"
    _write_json(summary_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--pair-id", choices=EXPECTED_PAIR_IDS, action="append"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.generate_only and args.analysis_only:
        parser.error("--generate-only and --analysis-only are mutually exclusive")
    if args.pair_id and not args.generate_only:
        parser.error("--pair-id is allowed only with --generate-only")

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    context = validate_config(config, verify_pair_inputs=True)
    if args.verify_only:
        print("replication config and authorized inputs verified")
        return 0

    selected = set(args.pair_id or EXPECTED_PAIR_IDS)
    if not args.analysis_only:
        for pair in context["pairs"]:
            if pair["paired_group_id"] in selected:
                print(f"[generate] {pair['paired_group_id']}", flush=True)
                _ensure_pair(
                    config, config_path, context, pair, resume=args.resume
                )
    if args.generate_only:
        return 0

    _analyze_all(config, config_path, context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
