#!/usr/bin/env python3
"""Generate and gate independent static/dynamic normal simulation-v4 runs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for search_root in (SCRIPT_ROOT, SRC_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_simulation_v4_normal_calibration as calibration  # noqa: E402
from gnss_doppler_lab.domain_gap import (  # noqa: E402
    assign_gate_status,
    compare_feature_distributions,
    domain_classifier_audit,
    worst_gate_status,
)
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
from gnss_doppler_lab.rf_impairments import ImpairmentConfig  # noqa: E402
from gnss_doppler_lab.simulation_v4 import (  # noqa: E402
    SimulationScenario,
    compose_paired_iq,
)
from gnss_doppler_lab.tracking_feature_windows import (  # noqa: E402
    export_receiver_run_tracking_feature_csv,
)
from gnss_doppler_lab.trajectory import generate_trajectory, read_trajectory  # noqa: E402

DEFAULT_CONFIG = Path(
    "configs/experiments/simulation_v4_normal_independent_validation_v1.json"
)
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _sha256(path: str | Path) -> str:
    return calibration._sha256(Path(path))


def _repo_path(value: str | Path) -> Path:
    return calibration._repo_path(value)


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    calibration._atomic_json(path, document)


def _read_csv(path: Path) -> list[dict[str, str]]:
    return calibration._read_csv(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    calibration._write_csv(path, rows)


def _run_datetime(run: dict[str, Any]) -> datetime:
    value = datetime.fromisoformat(str(run["utc"]).replace("Z", "+00:00"))
    if value.tzinfo is None or value.microsecond:
        raise ValueError(f"run UTC must be timezone-aware with integer seconds: {run['name']}")
    return value.astimezone(timezone.utc)


def _increasing_bounds(bounds: Any, name: str) -> tuple[float, float]:
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError(f"{name} must contain [low, high]")
    low, high = map(float, bounds)
    if not math.isfinite(low) or not math.isfinite(high) or not low < high:
        raise ValueError(f"{name} bounds must be finite and increasing")
    return low, high


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("version") != 1:
        raise ValueError("unsupported independent-validation config version")
    profile = config["rf_profile"]
    sample_rate = int(profile["rf_sample_rate_hz"])
    cutoff = float(profile["frontend_cutoff_hz"])
    if sample_rate < 1_000_000 or not 0 < cutoff < sample_rate / 2:
        raise ValueError("invalid RF sample rate or front-end cutoff")
    runs = config.get("runs")
    if not isinstance(runs, list) or len(runs) < 2:
        raise ValueError("runs must contain at least two independent runs")
    names = [str(run.get("name", "")) for run in runs]
    if len(names) != len(set(names)) or any(not SAFE_NAME.fullmatch(name) for name in names):
        raise ValueError("run names must be unique safe identifiers")
    seeds = [run.get("receiver_seed") for run in runs]
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("receiver seeds must be integers")
    if any(not 0 <= seed < 2**64 for seed in seeds):
        raise ValueError("receiver seeds must be unsigned 64-bit integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("receiver seeds must be unique")
    domains: set[str] = set()
    for run in runs:
        domain = str(run.get("domain"))
        domains.add(domain)
        if domain not in {"static", "dynamic"}:
            raise ValueError(f"unsupported run domain: {domain}")
        _run_datetime(run)
        raw_duration = run["duration_seconds"]
        if isinstance(raw_duration, bool) or not isinstance(raw_duration, int):
            raise ValueError("run duration must be an integer number of seconds")
        duration = raw_duration
        if not 10 <= duration <= 300:
            raise ValueError("run duration must be in [10, 300] seconds")
        position = run["position"]
        latitude = float(position["latitude_deg"])
        longitude = float(position["longitude_deg"])
        altitude = float(position["altitude_m"])
        if not -89.9 <= latitude <= 89.9 or not -180 <= longitude <= 180:
            raise ValueError(f"invalid run position: {run['name']}")
        if not all(math.isfinite(value) for value in (latitude, longitude, altitude)):
            raise ValueError(f"non-finite run position: {run['name']}")
        motion = run.get("motion")
        if domain == "static" and motion is not None:
            raise ValueError("static runs must not define motion")
        if domain == "dynamic":
            if not isinstance(motion, dict):
                raise ValueError("dynamic runs must define motion")
            if motion.get("kind") not in {"straight", "circle", "parallel-sweep"}:
                raise ValueError("unsupported dynamic motion kind")
            if float(motion["speed_mps"]) <= 0:
                raise ValueError("dynamic speed must be positive")
        target_cn0 = float(run.get(
            "target_composite_cn0_db_hz",
            profile["target_composite_cn0_db_hz"],
        ))
        sample_snr = calibration.equivalent_sample_snr_db(target_cn0, sample_rate)
        if not -30 <= sample_snr <= 100:
            raise ValueError(f"realized sample SNR is out of range: {run['name']}")
    if domains != {"static", "dynamic"}:
        raise ValueError("independent validation requires both static and dynamic runs")

    gnss_sdr = config["gnss_sdr"]
    features = config["features"]
    if int(gnss_sdr["tracking_tap_count"]) != 9:
        raise ValueError("independent validation requires the selected 9-tap receiver")
    if not math.isclose(
        float(gnss_sdr["tracking_tap_spacing_chips"]),
        0.125,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("independent validation requires 0.125-chip tap spacing")
    if int(features["tap_count"]) != 3:
        raise ValueError("independent validation requires inner E/P/L feature extraction")

    allowed = set(config["data_boundary"]["allowed_texbat_recordings"])
    if allowed != {"cleanStatic", "cleanDynamic"} or set(config["real_clean"]) != allowed:
        raise ValueError("only cleanStatic and cleanDynamic may be calibration references")
    serialized = json.dumps(config["real_clean"]).lower()
    for scenario in config["data_boundary"]["forbidden_texbat_recordings"]:
        if f"/{str(scenario).lower()}/" in serialized:
            raise ValueError(f"forbidden TEXBAT scenario configured: {scenario}")

    state_gate = config["receiver_state_gate"]
    for metric in ("carrier_lock_above_0_5_fraction", "cn0_db_hz_median"):
        pass_bounds = _increasing_bounds(state_gate["pass"][metric], f"pass.{metric}")
        conditional_bounds = _increasing_bounds(
            state_gate["conditional"][metric],
            f"conditional.{metric}",
        )
        if conditional_bounds[0] > pass_bounds[0] or conditional_bounds[1] < pass_bounds[1]:
            raise ValueError("conditional receiver-state bounds must contain pass bounds")


def _trajectory_position(
    root: Path,
    run: dict[str, Any],
) -> tuple[TrajectoryPosition, dict[str, Any]]:
    motion = run["motion"]
    path = root / "trajectories" / f"{run['name']}.csv"
    position = run["position"]
    kwargs: dict[str, Any] = {
        "latitude_deg": float(position["latitude_deg"]),
        "longitude_deg": float(position["longitude_deg"]),
        "altitude_m": float(position["altitude_m"]),
        "duration_seconds": int(run["duration_seconds"]),
        "speed_mps": float(motion["speed_mps"]),
        "heading_deg": float(motion.get("heading_deg", 0.0)),
    }
    for key in ("radius_m", "leg_length_m", "lane_spacing_m", "laps"):
        if key in motion:
            kwargs[key] = motion[key]
    metadata = generate_trajectory(str(motion["kind"]), path, **kwargs)
    rows = tuple(read_trajectory(path, float(run["duration_seconds"]), "llh"))
    sidecar = path.with_suffix(".json")
    trajectory = TrajectoryPosition(
        path=path.resolve(),
        coordinate_system="llh",
        rows=rows,
        csv_sha256=str(metadata["csv_sha256"]),
        metadata_path=sidecar.resolve(),
        metadata_sha256=_sha256(sidecar),
    )
    return trajectory, {
        "path": str(path.resolve()),
        "sha256": trajectory.csv_sha256,
        "metadata_path": str(sidecar.resolve()),
        "metadata_sha256": trajectory.metadata_sha256,
        "motion": motion,
        "effective": metadata["effective"],
        "row_count": len(rows),
    }


def _authentic_component(
    config: dict[str, Any],
    config_sha256: str,
    root: Path,
    run: dict[str, Any],
    simulator: GpsSdrSimRunner,
    *,
    resume: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    run_dir = root / "components" / str(run["name"])
    iq_path = run_dir / "authentic_gps_l1ca_s8_iq.bin"
    log_path = run_dir / "gps-sdr-sim.log"
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != config_sha256 or manifest.get("run") != run:
            raise ValueError(f"component provenance mismatch: {run['name']}")
        if not iq_path.is_file() or _sha256(iq_path) != manifest["iq_sha256"]:
            raise ValueError(f"component IQ integrity failure: {run['name']}")
        trajectory = manifest.get("trajectory")
        if trajectory and _sha256(trajectory["path"]) != trajectory["sha256"]:
            raise ValueError(f"trajectory integrity failure: {run['name']}")
        return iq_path, manifest_path, manifest
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"partial component directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    position_doc = run["position"]
    trajectory_record = None
    if run["domain"] == "dynamic":
        scenario_position, trajectory_record = _trajectory_position(root, run)
    else:
        scenario_position = StaticPosition(
            float(position_doc["latitude_deg"]),
            float(position_doc["longitude_deg"]),
            float(position_doc["altitude_m"]),
        )
    sample_rate = int(config["rf_profile"]["rf_sample_rate_hz"])
    rf_config = RFGenerationConfig(
        version=1,
        scenario=Scenario(
            str(run["name"]),
            "GPS",
            "L1CA",
            _run_datetime(run),
            int(run["duration_seconds"]),
            scenario_position,
        ),
        input=InputConfig(_repo_path(config["input"]["rinex_nav"])),
        output=OutputConfig(run_dir, sample_rate, "s8_iq"),
        simulator=SimulatorConfig(str(_repo_path(config["simulator"]["executable"]))),
        impairments=ImpairmentConfig(),
    )
    result = simulator.run(rf_config, iq_path, log_path)
    expected_bytes = simulator.expected_output_bytes(rf_config)
    if iq_path.stat().st_size != expected_bytes:
        raise RuntimeError(f"component byte contract failed: {run['name']}")
    manifest = {
        "schema": "gnss-doppler-lab.normal-independent-component",
        "schema_version": 1,
        "config_sha256": config_sha256,
        "run": run,
        "iq_path": str(iq_path.resolve()),
        "iq_sha256": _sha256(iq_path),
        "bytes": iq_path.stat().st_size,
        "complex_samples": iq_path.stat().st_size // 2,
        "actual_duration_seconds": (iq_path.stat().st_size // 2) / sample_rate,
        "trajectory": trajectory_record,
        "simulator": {
            "identity": simulator.identity,
            "executable": simulator.executable,
            "provenance": simulator.provenance,
            "cli_contract": simulator.cli_contract,
            "command": result["command"],
            "time": result["time"],
        },
    }
    _atomic_json(manifest_path, manifest)
    return iq_path, manifest_path, manifest


def _run_impairment(config: dict[str, Any], run: dict[str, Any]) -> ImpairmentConfig:
    profile = config["rf_profile"]
    receiver = config["receiver"]
    sample_rate = int(profile["rf_sample_rate_hz"])
    reference_rate = int(receiver["reference_sample_rate_hz"])
    phase_noise = float(receiver["phase_noise_std_rad_per_sqrt_sample"]) * math.sqrt(
        reference_rate / sample_rate
    )
    target_cn0 = float(run.get(
        "target_composite_cn0_db_hz",
        profile["target_composite_cn0_db_hz"],
    ))
    return ImpairmentConfig(
        enabled=True,
        profile="explicit",
        seed=int(run["receiver_seed"]),
        sample_snr_db=calibration.equivalent_sample_snr_db(target_cn0, sample_rate),
        carrier_offset_hz=float(receiver["carrier_offset_hz"]),
        frequency_drift_hz_per_s=float(receiver["frequency_drift_hz_per_s"]),
        phase_noise_std_rad_per_sqrt_sample=phase_noise,
        frontend_cutoff_hz=float(profile["frontend_cutoff_hz"]),
        frontend_order=int(receiver["frontend_order"]),
        iq_gain_imbalance_db=float(receiver["iq_gain_imbalance_db"]),
        iq_phase_imbalance_deg=float(receiver["iq_phase_imbalance_deg"]),
        dc_i=float(receiver["dc_i"]),
        dc_q=float(receiver["dc_q"]),
        gain=1.0,
        agc_target_rms=None,
        clip_level=float(receiver["clip_level"]),
        chunk_samples=int(receiver["chunk_samples"]),
    )


def _receiver_run_id(run: dict[str, Any]) -> str:
    timestamp = _run_datetime(run).strftime("%Y%m%dT%H%M%SZ")
    return f"simv4iv-{run['name']}_{timestamp}"


def _normal_rf(
    config: dict[str, Any],
    config_sha256: str,
    root: Path,
    run: dict[str, Any],
    component_path: Path,
    component_manifest_path: Path,
    component_manifest: dict[str, Any],
    *,
    resume: bool,
) -> Path:
    run_dir = root / "rf" / str(run["name"])
    iq_path = run_dir / "gps_l1ca_s8_iq.bin"
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("validation", {}).get("config_sha256") != config_sha256:
            raise ValueError(f"RF config provenance mismatch: {run['name']}")
        if manifest.get("validation", {}).get("run") != run:
            raise ValueError(f"RF run definition changed: {run['name']}")
        if not iq_path.is_file() or _sha256(iq_path) != manifest["iq"]["sha256"]:
            raise ValueError(f"RF IQ integrity failure: {run['name']}")
        calibration._receiver_iq_alias(root, str(run["name"]), iq_path)
        return manifest_path
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"partial RF directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    impairment = _run_impairment(config, run)
    scenario = SimulationScenario(str(run["name"]), "steady_normal")
    composition = compose_paired_iq(
        component_path,
        component_path,
        {str(run["name"]): iq_path},
        (scenario,),
        sample_rate_hz=int(config["rf_profile"]["rf_sample_rate_hz"]),
        receiver=impairment,
        normal_target_rms=float(config["normal_target_rms"]),
    )
    report = composition["scenarios"][run["name"]]
    alias = calibration._receiver_iq_alias(root, str(run["name"]), iq_path)
    manifest = {
        "schema_version": 4,
        "run_id": _receiver_run_id(run),
        "scenario": {
            "name": run["name"],
            "campaign": config["campaign"]["name"],
            "utc": run["utc"],
            "duration_seconds": int(run["duration_seconds"]),
            "position": run["position"],
            "motion": run.get("motion"),
            "domain": run["domain"],
            "class": "normal",
            "event": "steady",
            "is_spoofing": False,
        },
        "iq": {
            "path": str(alias),
            "canonical_storage_path": str(iq_path.resolve()),
            "sha256": report["sha256"],
            "actual_bytes": report["bytes"],
            "complex_samples": report["complex_samples"],
            "actual_duration_seconds": report["actual_duration_seconds"],
            "rf_sample_rate_hz": int(config["rf_profile"]["rf_sample_rate_hz"]),
            "sample_format": "s8_iq",
            "channels": 2,
        },
        "validation": {
            "config_sha256": config_sha256,
            "run": run,
            "runner_script_sha256": _sha256(Path(__file__)),
            "realized_sample_snr_db": impairment.sample_snr_db,
            "target_composite_cn0_db_hz": float(run.get(
                "target_composite_cn0_db_hz",
                config["rf_profile"]["target_composite_cn0_db_hz"],
            )),
            "receiver_impairment": impairment.manifest(),
            "authentic_component_manifest": str(component_manifest_path.resolve()),
            "authentic_component_manifest_sha256": _sha256(component_manifest_path),
            "authentic_component_sha256": component_manifest["iq_sha256"],
            "trajectory": component_manifest.get("trajectory"),
            "composition_reference": composition["reference"],
            "composition_processing": composition["processing"],
            "measurements": report,
            "scope": "independent normal-only static/dynamic validation; no spoofing input",
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _receiver_and_features(
    config: dict[str, Any],
    root: Path,
    run: dict[str, Any],
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
    feature_path = feature_root / f"{run['name']}_tracking_features.csv"
    feature_manifest = feature_path.with_suffix(".manifest.json")
    required = (receiver_manifest, feature_path, feature_manifest)
    if all(path.is_file() for path in required):
        if not resume:
            raise FileExistsError(receiver_manifest)
        receiver_document = json.loads(receiver_manifest.read_text(encoding="utf-8"))
        if receiver_document.get("source", {}).get("rf_manifest_sha256") != _sha256(rf_manifest):
            raise ValueError(f"receiver provenance mismatch: {run['name']}")
        expected_taps = int(config["gnss_sdr"]["tracking_tap_count"])
        if int(receiver_document.get("tracking", {}).get("tap_count", 3)) != expected_taps:
            raise ValueError(f"receiver tap-count mismatch: {run['name']}")
        expected_executable = calibration._executable_path(config["gnss_sdr"]["executable"])
        command = receiver_document.get("receiver", {}).get("command", [])
        if command and str(Path(command[0]).resolve()) != str(Path(expected_executable).resolve()):
            raise ValueError(f"receiver executable mismatch: {run['name']}")
        feature_document = json.loads(feature_manifest.read_text(encoding="utf-8"))
        if feature_document.get("receiver_manifest_sha256") != _sha256(receiver_manifest):
            raise ValueError(f"feature receiver provenance mismatch: {run['name']}")
        if _sha256(feature_path) != feature_document.get("feature_csv_sha256"):
            raise ValueError(f"feature CSV integrity failure: {run['name']}")
    else:
        if any(path.exists() for path in required) or receiver_dir.exists():
            raise FileExistsError(f"partial receiver/feature output: {run['name']}")
        print(f"[receiver] {run['name']}", flush=True)
        receiver_manifest = run_receiver(
            rf_manifest,
            receiver_root,
            executable=calibration._executable_path(config["gnss_sdr"]["executable"]),
            channel_count=int(config["gnss_sdr"]["channel_count"]),
            timeout_seconds=int(config["gnss_sdr"]["timeout_seconds"]),
            tracking_tap_count=int(config["gnss_sdr"]["tracking_tap_count"]),
            tracking_tap_spacing_chips=float(
                config["gnss_sdr"]["tracking_tap_spacing_chips"]
            ),
        )
        export_receiver_run_tracking_feature_csv(
            receiver_manifest.parent,
            output_path=feature_path,
            tap_count=int(config["features"]["tap_count"]),
            window_s=float(config["features"]["window_s"]),
            stride_s=float(config["features"]["stride_s"]),
            min_epochs=int(config["features"]["min_epochs"]),
            label="normal",
        )
        rows = _read_csv(feature_path)
        _atomic_json(feature_manifest, {
            "schema": "gnss-doppler-lab.normal-independent-features",
            "schema_version": 1,
            "run": run,
            "receiver_manifest": str(receiver_manifest.resolve()),
            "receiver_manifest_sha256": _sha256(receiver_manifest),
            "feature_csv": str(feature_path.resolve()),
            "feature_csv_sha256": _sha256(feature_path),
            "row_count": len(rows),
            "tap_count": int(config["features"]["tap_count"]),
            "window_s": float(config["features"]["window_s"]),
            "stride_s": float(config["features"]["stride_s"]),
            "min_epochs": int(config["features"]["min_epochs"]),
        })
    rows = _read_csv(feature_path)
    if not rows:
        raise ValueError(f"zero feature rows: {run['name']}")
    state = calibration._receiver_state_summary(receiver_manifest.parent)
    return receiver_manifest, feature_path, state, rows


def _comparison(
    simulation_rows: list[dict[str, str]],
    real_rows: list[dict[str, str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    metrics, distribution = compare_feature_distributions(
        simulation_rows,
        real_rows,
        config["features"]["columns"],
    )
    classifier = domain_classifier_audit(
        simulation_rows,
        real_rows,
        config["features"]["columns"],
        simulation_source_column="run_id",
        real_source_column="domain_source",
        max_rows_per_group=int(config["classifier"]["max_windows_per_source_prn_group"]),
        n_splits=int(config["classifier"]["n_splits"]),
        random_state=int(config["classifier"]["random_state"]),
    )
    status, reasons = assign_gate_status(distribution, classifier, config["gate"])
    return {
        "gate_status": status,
        "stop_reasons": reasons,
        "distribution": distribution,
        "domain_classifier": classifier,
        "per_feature": metrics,
    }


def _value_gate(value: float, metric: str, config: dict[str, Any]) -> str:
    gate = config["receiver_state_gate"]
    pass_low, pass_high = _increasing_bounds(gate["pass"][metric], f"pass.{metric}")
    conditional_low, conditional_high = _increasing_bounds(
        gate["conditional"][metric],
        f"conditional.{metric}",
    )
    if pass_low <= value <= pass_high:
        return "pass"
    if conditional_low <= value <= conditional_high:
        return "conditional"
    return "stop"


def receiver_state_gate(
    states: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    for name, state in states.items():
        metrics = {
            metric: {
                "value": float(state[metric]),
                "gate_status": _value_gate(float(state[metric]), metric, config),
            }
            for metric in ("carrier_lock_above_0_5_fraction", "cn0_db_hz_median")
        }
        runs[name] = {
            "metrics": metrics,
            "gate_status": worst_gate_status(
                metric["gate_status"] for metric in metrics.values()
            ),
        }
    return {
        "runs": runs,
        "gate_status": worst_gate_status(run["gate_status"] for run in runs.values()),
        "thresholds": config["receiver_state_gate"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    config_path = _repo_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    config_sha = _sha256(config_path)
    root = _repo_path(config["output_root"])
    if root.exists() and not args.resume:
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    simulator = GpsSdrSimRunner(str(_repo_path(config["simulator"]["executable"])))

    rf_manifests: dict[str, Path] = {}
    component_manifests: dict[str, Path] = {}
    for run in config["runs"]:
        name = str(run["name"])
        print(f"[component] {name}", flush=True)
        component_path, component_manifest_path, component_manifest = _authentic_component(
            config,
            config_sha,
            root,
            run,
            simulator,
            resume=args.resume,
        )
        component_manifests[name] = component_manifest_path
        print(f"[normal-rf] {name}", flush=True)
        rf_manifests[name] = _normal_rf(
            config,
            config_sha,
            root,
            run,
            component_path,
            component_manifest_path,
            component_manifest,
            resume=args.resume,
        )
    if args.generate_only:
        print(root / "rf")
        return 0

    features_by_run: dict[str, list[dict[str, str]]] = {}
    states: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Any] = {}
    run_by_name = {str(run["name"]): run for run in config["runs"]}
    for run in config["runs"]:
        name = str(run["name"])
        receiver_manifest, feature_path, state, rows = _receiver_and_features(
            config,
            root,
            run,
            rf_manifests[name],
            resume=args.resume,
        )
        features_by_run[name] = rows
        states[name] = state
        artifacts[name] = {
            "component_manifest": str(component_manifests[name]),
            "component_manifest_sha256": _sha256(component_manifests[name]),
            "rf_manifest": str(rf_manifests[name]),
            "rf_manifest_sha256": _sha256(rf_manifests[name]),
            "receiver_manifest": str(receiver_manifest),
            "receiver_manifest_sha256": _sha256(receiver_manifest),
            "feature_csv": str(feature_path),
            "feature_csv_sha256": _sha256(feature_path),
            "feature_rows": len(rows),
        }

    real = calibration._load_real_clean(config)
    static_names = [
        name for name, run in run_by_name.items() if run["domain"] == "static"
    ]
    dynamic_names = [
        name for name, run in run_by_name.items() if run["domain"] == "dynamic"
    ]
    static_rows = [row for name in static_names for row in features_by_run[name]]
    dynamic_rows = [row for name in dynamic_names for row in features_by_run[name]]
    combined_rows = static_rows + dynamic_rows
    combined_real = real["cleanStatic"] + real["cleanDynamic"]
    comparisons = {
        "cleanStatic": _comparison(static_rows, real["cleanStatic"], config),
        "cleanDynamic": _comparison(dynamic_rows, real["cleanDynamic"], config),
        "cleanCombined": _comparison(combined_rows, combined_real, config),
    }
    run_diagnostics: dict[str, Any] = {}
    for name, rows in features_by_run.items():
        target = "cleanStatic" if run_by_name[name]["domain"] == "static" else "cleanDynamic"
        run_diagnostics[name] = {
            "target": target,
            **_comparison(rows, real[target], config),
        }
        print(
            f"[score] {name} target={target} "
            f"auc={run_diagnostics[name]['domain_classifier']['pooled_separability_auc']:.6f} "
            f"status={run_diagnostics[name]['gate_status']}",
            flush=True,
        )

    state_result = receiver_state_gate(states, config)
    domain_status = worst_gate_status(
        result["gate_status"] for result in comparisons.values()
    )
    overall_status = worst_gate_status((domain_status, state_result["gate_status"]))

    aggregate_rows: list[dict[str, Any]] = []
    per_feature_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for name, result in comparisons.items():
        aggregate_rows.append({
            "comparison": name,
            "gate_status": result["gate_status"],
            "domain_auc": result["domain_classifier"]["pooled_separability_auc"],
            "mean_fold_domain_auc": result["domain_classifier"]["mean_fold_separability_auc"],
            "median_ks": result["distribution"]["median_ks_statistic"],
            "median_robust_shift": result["distribution"]["median_robust_median_shift"],
            "simulation_rows": result["distribution"]["simulation_input_rows"],
            "real_rows": result["distribution"]["real_input_rows"],
        })
        per_feature_rows.extend({
            "scope": "aggregate",
            "name": name,
            **row,
        } for row in result["per_feature"])
        fold_rows.extend({
            "scope": "aggregate",
            "name": name,
            **row,
        } for row in result["domain_classifier"]["folds"])
    run_rows: list[dict[str, Any]] = []
    for name, result in run_diagnostics.items():
        state = states[name]
        run_rows.append({
            "run": name,
            "domain": run_by_name[name]["domain"],
            "target": result["target"],
            "gate_status": result["gate_status"],
            "domain_auc": result["domain_classifier"]["pooled_separability_auc"],
            "median_ks": result["distribution"]["median_ks_statistic"],
            "median_robust_shift": result["distribution"]["median_robust_median_shift"],
            "feature_rows": len(features_by_run[name]),
            "lock_fraction": state["carrier_lock_above_0_5_fraction"],
            "cn0_median_db_hz": state["cn0_db_hz_median"],
            "receiver_state_gate": state_result["runs"][name]["gate_status"],
        })
        per_feature_rows.extend({
            "scope": "run",
            "name": name,
            **row,
        } for row in result["per_feature"])
        fold_rows.extend({
            "scope": "run",
            "name": name,
            **row,
        } for row in result["domain_classifier"]["folds"])

    _write_csv(root / "aggregate_scores.csv", aggregate_rows)
    _write_csv(root / "run_scores.csv", run_rows)
    _write_csv(root / "per_feature_metrics.csv", per_feature_rows)
    _write_csv(root / "domain_classifier_folds.csv", fold_rows)
    summary = {
        "schema": "gnss-doppler-lab.simulation-v4-normal-independent-validation",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "runner_script_sha256": _sha256(Path(__file__)),
        "campaign": config["campaign"],
        "rf_profile": config["rf_profile"],
        "effective_receiver": {
            "executable": calibration._executable_path(config["gnss_sdr"]["executable"]),
            "tracking_tap_count": int(config["gnss_sdr"]["tracking_tap_count"]),
            "feature_tap_count": int(config["features"]["tap_count"]),
            "tracking_tap_spacing_chips": float(
                config["gnss_sdr"]["tracking_tap_spacing_chips"]
            ),
        },
        "data_boundary": {
            **config["data_boundary"],
            "forbidden_scenarios_accessed": False,
            "spoofing_data_generated": False,
            "detector_trained": False,
        },
        "runs": run_by_name,
        "artifacts": artifacts,
        "receiver_states": states,
        "receiver_state_gate": state_result,
        "comparisons": comparisons,
        "run_diagnostics": run_diagnostics,
        "domain_gate_status": domain_status,
        "overall_gate_status": overall_status,
        "interpretation": (
            "qualified for a larger independent normal campaign"
            if overall_status in {"pass", "conditional"}
            else "not qualified for scale generation or detector training"
        ),
        "elapsed_seconds": round(time.time() - started, 3),
        "limitations": [
            "This independent pilot uses two static and three dynamic 30-second runs.",
            "TEXBAT cleanStatic and cleanDynamic remain development calibration references.",
            "Per-run diagnostics are reported but aggregate matched-domain gates are primary.",
            "No TEXBAT spoofing scenario was accessed and no detector was trained.",
        ],
    }
    summary_path = root / "summary.json"
    _atomic_json(summary_path, summary)
    print(json.dumps({
        "summary": str(summary_path),
        "overall_gate_status": overall_status,
        "domain_gate_status": domain_status,
        "receiver_state_gate_status": state_result["gate_status"],
        "comparisons": {
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
