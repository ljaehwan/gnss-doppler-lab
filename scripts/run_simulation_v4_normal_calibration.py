#!/usr/bin/env python3
"""Generate and rank normal-only simulation-v4 domain-calibration candidates."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import shutil
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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
)
from gnss_doppler_lab.rf_impairments import ImpairmentConfig  # noqa: E402
from gnss_doppler_lab.simulation_v4 import (  # noqa: E402
    SimulationScenario,
    compose_paired_iq,
)
from gnss_doppler_lab.tracking_feature_windows import (  # noqa: E402
    export_receiver_run_tracking_feature_csv,
)

DEFAULT_CONFIG = Path("configs/experiments/simulation_v4_normal_calibration_v1.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _executable_path(value: str | Path) -> str:
    """Resolve PATH names and repository-relative receiver executables."""
    path = Path(value)
    if path.is_absolute():
        return str(path.resolve())
    if path.parent != Path("."):
        return str(_repo_path(path).resolve())
    located = shutil.which(str(value))
    return str(Path(located).resolve()) if located else str(value)


def _receiver_run_id(candidate_name: str, timestamp: str) -> str:
    """Keep GNSS-SDR config paths below its file-source path-length limit."""
    return f"simv4cal-{candidate_name}_{timestamp}"


def _receiver_iq_alias(root: Path, candidate_name: str, storage_path: Path) -> Path:
    alias = root / "receiver_iq" / f"{candidate_name}.bin"
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.exists():
        if not alias.samefile(storage_path):
            raise ValueError(f"receiver IQ alias collision: {alias}")
    else:
        alias.hardlink_to(storage_path)
    return alias.resolve()


def _variant_suffix(value: str) -> str:
    if not value:
        return ""
    if any(not (character.isalnum() or character in "-_") for character in value):
        raise ValueError("receiver variant must contain only letters, digits, hyphen, or underscore")
    return f"_{value}"


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0])
    extras = sorted({key for row in rows for key in row} - set(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + extras, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def equivalent_sample_snr_db(target_composite_cn0_db_hz: float, sample_rate_hz: int) -> float:
    """Convert the configured composite C/N0 reference to per-sample SNR."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if not math.isfinite(target_composite_cn0_db_hz):
        raise ValueError("target_composite_cn0_db_hz must be finite")
    return float(target_composite_cn0_db_hz - 10.0 * math.log10(sample_rate_hz))


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("version") != 1:
        raise ValueError("unsupported normal calibration config version")
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list")
    names = [str(candidate.get("name", "")) for candidate in candidates]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("candidate names must be non-empty and unique")
    for candidate in candidates:
        rate = int(candidate["rf_sample_rate_hz"])
        if rate < 1_000_000:
            raise ValueError("candidate sample rate must be at least 1 MHz")
        cutoff = candidate.get("frontend_cutoff_hz")
        if cutoff is not None and not 0 < float(cutoff) < rate / 2:
            raise ValueError(f"invalid frontend cutoff for {candidate['name']}")
        sample_snr = equivalent_sample_snr_db(float(candidate["target_composite_cn0_db_hz"]), rate)
        if not -30 <= sample_snr <= 100:
            raise ValueError(f"realized sample SNR is out of range for {candidate['name']}")
    component_sources = config.get("authentic_components", {})
    candidate_rates = {str(int(row["rf_sample_rate_hz"])) for row in candidates}
    if component_sources and not candidate_rates.issubset(component_sources):
        raise ValueError("authentic_components must cover every candidate sample rate")
    for sample_rate, source in component_sources.items():
        if int(sample_rate) < 1_000_000 or set(source) != {"path", "sha256"}:
            raise ValueError("invalid authentic_components entry")
    allowed = set(config["data_boundary"]["allowed_texbat_recordings"])
    if set(config["real_clean"]) != allowed:
        raise ValueError("real_clean inputs must exactly match the allowed clean recordings")
    serialized = json.dumps(config["real_clean"]).lower()
    for scenario in config["data_boundary"]["forbidden_texbat_recordings"]:
        if f"/{str(scenario).lower()}/" in serialized:
            raise ValueError(f"forbidden TEXBAT scenario configured: {scenario}")
    weights = config["ranking_weights"]
    if not math.isclose(sum(float(value) for value in weights.values()), 1.0, abs_tol=1e-12):
        raise ValueError("ranking weights must sum to one")


def _campaign_datetime(config: dict[str, Any]) -> datetime:
    value = datetime.fromisoformat(config["campaign"]["utc"].replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("campaign UTC must be timezone-aware")
    return value.astimezone(timezone.utc)


def _clean_rf_config(config: dict[str, Any], sample_rate_hz: int, output_root: Path) -> RFGenerationConfig:
    campaign = config["campaign"]
    position = campaign["position"]
    return RFGenerationConfig(
        version=1,
        scenario=Scenario(
            f"{campaign['name']}-source-{sample_rate_hz}",
            "GPS",
            "L1CA",
            _campaign_datetime(config),
            int(campaign["duration_seconds"]),
            StaticPosition(
                float(position["latitude_deg"]),
                float(position["longitude_deg"]),
                float(position["altitude_m"]),
            ),
        ),
        input=InputConfig(_repo_path(config["input"]["rinex_nav"])),
        output=OutputConfig(output_root, sample_rate_hz, "s8_iq"),
        simulator=SimulatorConfig(str(_repo_path(config["simulator"]["executable"]))),
        impairments=ImpairmentConfig(),
    )


def _candidate_impairment(config: dict[str, Any], candidate: dict[str, Any]) -> ImpairmentConfig:
    receiver = config["receiver"]
    sample_rate = int(candidate["rf_sample_rate_hz"])
    reference_sample_rate = int(receiver["reference_sample_rate_hz"])
    phase_noise = float(receiver["phase_noise_std_rad_per_sqrt_sample"]) * math.sqrt(
        reference_sample_rate / sample_rate
    )
    return ImpairmentConfig(
        enabled=True,
        profile="explicit",
        seed=int(receiver["seed"]),
        sample_snr_db=equivalent_sample_snr_db(
            float(candidate["target_composite_cn0_db_hz"]),
            sample_rate,
        ),
        carrier_offset_hz=float(receiver["carrier_offset_hz"]),
        frequency_drift_hz_per_s=float(receiver["frequency_drift_hz_per_s"]),
        phase_noise_std_rad_per_sqrt_sample=phase_noise,
        frontend_cutoff_hz=(
            None if candidate.get("frontend_cutoff_hz") is None
            else float(candidate["frontend_cutoff_hz"])
        ),
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


def _component(
    config: dict[str, Any],
    config_sha256: str,
    root: Path,
    sample_rate_hz: int,
    runner: GpsSdrSimRunner,
    *,
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    external = config.get("authentic_components", {}).get(str(sample_rate_hz))
    if external is not None:
        iq_path = _repo_path(external["path"])
        observed_sha = _sha256(iq_path)
        if observed_sha != external["sha256"]:
            raise ValueError(f"external authentic component integrity failure: {sample_rate_hz}")
        return iq_path, {
            "sample_rate_hz": sample_rate_hz,
            "iq_path": str(iq_path),
            "iq_sha256": observed_sha,
            "bytes": iq_path.stat().st_size,
            "reused_external_component": True,
        }
    directory = root / "components" / f"fs{sample_rate_hz}"
    iq_path = directory / "authentic_gps_l1ca_s8_iq.bin"
    log_path = directory / "gps-sdr-sim.log"
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != config_sha256 or manifest.get("sample_rate_hz") != sample_rate_hz:
            raise ValueError(f"component provenance mismatch: {manifest_path}")
        if not iq_path.is_file() or _sha256(iq_path) != manifest["iq_sha256"]:
            raise ValueError(f"component IQ integrity failure: {iq_path}")
        return iq_path, manifest
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"partial component directory requires manual review: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    rf_config = _clean_rf_config(config, sample_rate_hz, directory)
    result = runner.run(rf_config, iq_path, log_path)
    expected = runner.expected_output_bytes(rf_config)
    if iq_path.stat().st_size != expected:
        raise RuntimeError(f"component byte contract failed for {sample_rate_hz}")
    manifest = {
        "config_sha256": config_sha256,
        "sample_rate_hz": sample_rate_hz,
        "iq_path": str(iq_path),
        "iq_sha256": _sha256(iq_path),
        "bytes": iq_path.stat().st_size,
        "complex_samples": iq_path.stat().st_size // 2,
        "actual_duration_seconds": (iq_path.stat().st_size // 2) / sample_rate_hz,
        "simulator": {
            "identity": runner.identity,
            "executable": runner.executable,
            "provenance": runner.provenance,
            "cli_contract": runner.cli_contract,
            "command": result["command"],
        },
    }
    _atomic_json(manifest_path, manifest)
    return iq_path, manifest


def _candidate_rf(
    config: dict[str, Any],
    config_sha256: str,
    root: Path,
    candidate: dict[str, Any],
    component_path: Path,
    component_manifest: dict[str, Any],
    *,
    resume: bool,
) -> Path:
    name = str(candidate["name"])
    timestamp = _campaign_datetime(config).strftime("%Y%m%dT%H%M%SZ")
    storage_id = f"{config['campaign']['name']}-{name}_{timestamp}"
    run_id = _receiver_run_id(name, timestamp)
    run_dir = root / "rf" / storage_id
    iq_path = run_dir / "gps_l1ca_s8_iq.bin"
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("calibration", {}).get("config_sha256") != config_sha256:
            raise ValueError(f"candidate config provenance mismatch: {manifest_path}")
        if manifest.get("calibration", {}).get("candidate") != candidate:
            raise ValueError(f"candidate definition changed: {name}")
        if not iq_path.is_file() or _sha256(iq_path) != manifest["iq"]["sha256"]:
            raise ValueError(f"candidate IQ integrity failure: {name}")
        alias = _receiver_iq_alias(root, name, iq_path)
        manifest["run_id"] = run_id
        manifest["iq"]["path"] = str(alias)
        manifest["iq"]["canonical_storage_path"] = str(iq_path)
        return manifest_path
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"partial RF directory requires manual review: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    scenario = SimulationScenario(name, "steady_normal")
    impairment = _candidate_impairment(config, candidate)
    composition = compose_paired_iq(
        component_path,
        component_path,
        {name: iq_path},
        (scenario,),
        sample_rate_hz=int(candidate["rf_sample_rate_hz"]),
        receiver=impairment,
        normal_target_rms=float(config["normal_target_rms"]),
    )
    report = composition["scenarios"][name]
    alias = _receiver_iq_alias(root, name, iq_path)
    manifest = {
        "schema_version": 4,
        "run_id": run_id,
        "scenario": {
            "name": name,
            "campaign": config["campaign"]["name"],
            "utc": config["campaign"]["utc"],
            "duration_seconds": int(config["campaign"]["duration_seconds"]),
            "position": config["campaign"]["position"],
            "class": "normal",
            "event": "steady",
            "is_spoofing": False,
        },
        "iq": {
            "path": str(alias),
            "canonical_storage_path": str(iq_path),
            "sha256": report["sha256"],
            "actual_bytes": report["bytes"],
            "complex_samples": report["complex_samples"],
            "actual_duration_seconds": report["actual_duration_seconds"],
            "rf_sample_rate_hz": int(candidate["rf_sample_rate_hz"]),
            "sample_format": "s8_iq",
            "channels": 2,
        },
        "calibration": {
            "config_sha256": config_sha256,
            "candidate": candidate,
            "runner_script_sha256": _sha256(Path(__file__)),
            "realized_sample_snr_db": impairment.sample_snr_db,
            "target_composite_cn0_db_hz": float(candidate["target_composite_cn0_db_hz"]),
            "receiver_impairment": impairment.manifest(),
            "authentic_component_sha256": component_manifest["iq_sha256"],
            "composition_reference": composition["reference"],
            "composition_processing": composition["processing"],
            "measurements": report,
            "scope": "normal-only simulation-to-real calibration; no spoofing scenario input",
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _receiver_state_summary(receiver_dir: Path) -> dict[str, Any]:
    manifest = json.loads((receiver_dir / "manifest.json").read_text(encoding="utf-8"))
    sample_rate = float(manifest["source"]["sample_rate_hz"])
    raw_dir = receiver_dir / str(manifest.get("tracking", {}).get("raw_directory", "raw"))
    locks: list[np.ndarray] = []
    cn0s: list[np.ndarray] = []
    prns: set[str] = set()
    for path in sorted(raw_dir.glob("epl_tracking_ch_*.mat")):
        with h5py.File(path, "r") as handle:
            raw_prn = np.asarray(handle["PRN"]).reshape(-1)
            lock = np.asarray(handle["carrier_lock_test"]).reshape(-1).astype(np.float64)
            cn0 = np.asarray(handle["CN0_SNV_dB_Hz"]).reshape(-1).astype(np.float64)
        selected = (
            np.isfinite(raw_prn) & (raw_prn >= 1) & (raw_prn <= 32)
            & np.isfinite(lock) & np.isfinite(cn0)
        )
        if selected.any():
            locks.append(lock[selected])
            cn0s.append(cn0[selected])
            prns.update(f"G{int(value):02d}" for value in raw_prn[selected])
    if not locks:
        raise ValueError(f"no valid receiver-state epochs: {receiver_dir}")
    all_locks = np.concatenate(locks)
    all_cn0 = np.concatenate(cn0s)
    return {
        "sample_rate_hz": int(sample_rate),
        "epoch_count": int(all_locks.size),
        "tracked_prn_count": len(prns),
        "tracked_prns": sorted(prns),
        "carrier_lock_median": float(np.median(all_locks)),
        "carrier_lock_above_0_5_fraction": float(np.mean(all_locks > 0.5)),
        "cn0_db_hz_median": float(np.median(all_cn0)),
        "cn0_db_hz_iqr": float(np.subtract(*np.quantile(all_cn0, [0.75, 0.25]))),
    }


def _range_distance(value: float, bounds: list[float]) -> float:
    low, high = map(float, bounds)
    if not low < high:
        raise ValueError("receiver-state target bounds must increase")
    if value < low:
        return (low - value) / (high - low)
    if value > high:
        return (value - high) / (high - low)
    return 0.0


def fidelity_loss(
    combined_result: dict[str, Any],
    receiver_state: dict[str, Any],
    config: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Return a predeclared ranking heuristic; gate status remains authoritative."""
    auc = float(combined_result["domain_classifier"]["pooled_separability_auc"])
    median_ks = float(combined_result["distribution"]["median_ks_statistic"])
    median_shift = float(combined_result["distribution"]["median_robust_median_shift"])
    conditional_shift = float(config["gate"]["conditional"]["median_robust_shift_max"])
    target = config["receiver_state_target"]
    terms = {
        "domain_separation": max(0.0, (auc - 0.5) / 0.5),
        "median_ks": median_ks,
        "median_robust_shift": median_shift / conditional_shift,
        "lock_range_distance": _range_distance(
            float(receiver_state["carrier_lock_above_0_5_fraction"]),
            target["carrier_lock_above_0_5_fraction"],
        ),
        "cn0_range_distance": _range_distance(
            float(receiver_state["cn0_db_hz_median"]),
            target["cn0_db_hz_median"],
        ),
    }
    weights = config["ranking_weights"]
    return float(sum(float(weights[name]) * value for name, value in terms.items())), terms


def _load_real_clean(config: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for name, source in config["real_clean"].items():
        path = _repo_path(source["feature_csv"])
        if _sha256(path) != source["feature_sha256"]:
            raise ValueError(f"real-clean feature integrity failure: {name}")
        rows = _read_csv(path)
        for row in rows:
            row["domain_source"] = name
        result[name] = rows
    return result


def _score_candidate(
    candidate: dict[str, Any],
    feature_rows: list[dict[str, str]],
    real_by_source: dict[str, list[dict[str, str]]],
    receiver_state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    comparisons = dict(real_by_source)
    comparisons["cleanCombined"] = [
        row for name in sorted(real_by_source) for row in real_by_source[name]
    ]
    results: dict[str, Any] = {}
    for name, real_rows in comparisons.items():
        metrics, distribution = compare_feature_distributions(
            feature_rows,
            real_rows,
            config["features"]["columns"],
        )
        classifier = domain_classifier_audit(
            feature_rows,
            real_rows,
            config["features"]["columns"],
            simulation_source_column="run_id",
            real_source_column="domain_source",
            max_rows_per_group=int(config["classifier"]["max_windows_per_source_prn_group"]),
            n_splits=int(config["classifier"]["n_splits"]),
            random_state=int(config["classifier"]["random_state"]),
        )
        status, reasons = assign_gate_status(distribution, classifier, config["gate"])
        results[name] = {
            "gate_status": status,
            "stop_reasons": reasons,
            "distribution": distribution,
            "domain_classifier": classifier,
            "per_feature": metrics,
        }
    overall = worst_gate_status(result["gate_status"] for result in results.values())
    loss, terms = fidelity_loss(results["cleanCombined"], receiver_state, config)
    return {
        "candidate": candidate,
        "feature_row_count": len(feature_rows),
        "receiver_state": receiver_state,
        "comparisons": results,
        "overall_gate_status": overall,
        "fidelity_loss": loss,
        "fidelity_loss_terms": terms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--receiver-variant", default="")
    parser.add_argument("--receiver-executable")
    parser.add_argument("--receiver-tap-count", type=int)
    parser.add_argument("--feature-tap-count", type=int)
    args = parser.parse_args(argv)
    variant_suffix = _variant_suffix(args.receiver_variant)

    has_receiver_override = any(
        value is not None
        for value in (args.receiver_executable, args.receiver_tap_count, args.feature_tap_count)
    )
    if has_receiver_override and not args.receiver_variant:
        parser.error("receiver overrides require --receiver-variant")

    started = time.time()
    config_path = _repo_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    receiver_executable = _executable_path(
        args.receiver_executable
        if args.receiver_executable is not None
        else config["gnss_sdr"]["executable"]
    )
    receiver_tap_count = int(
        args.receiver_tap_count
        if args.receiver_tap_count is not None
        else config["gnss_sdr"]["tracking_tap_count"]
    )
    feature_tap_count = int(
        args.feature_tap_count
        if args.feature_tap_count is not None
        else config["gnss_sdr"]["tracking_tap_count"]
    )
    if receiver_tap_count not in {3, 5, 9}:
        parser.error("receiver tap count must be 3, 5, or 9")
    if feature_tap_count != 3:
        parser.error("current-paper calibration features require feature tap count 3")
    config_sha = _sha256(config_path)
    root = _repo_path(config["output_root"])
    if root.exists() and not args.resume:
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    runner = GpsSdrSimRunner(str(_repo_path(config["simulator"]["executable"])))

    manifests: dict[str, Path] = {}
    components: dict[int, dict[str, Any]] = {}
    candidates_by_rate: dict[int, list[dict[str, Any]]] = {}
    for candidate in config["candidates"]:
        candidates_by_rate.setdefault(int(candidate["rf_sample_rate_hz"]), []).append(candidate)
    for sample_rate in sorted(candidates_by_rate):
        print(f"[component] fs={sample_rate}", flush=True)
        component_path, component_manifest = _component(
            config,
            config_sha,
            root,
            sample_rate,
            runner,
            resume=args.resume,
        )
        components[sample_rate] = component_manifest
        for candidate in candidates_by_rate[sample_rate]:
            print(f"[candidate] {candidate['name']}", flush=True)
            manifests[candidate["name"]] = _candidate_rf(
                config,
                config_sha,
                root,
                candidate,
                component_path,
                component_manifest,
                resume=args.resume,
            )
    if not config["keep_authentic_components"]:
        for manifest in components.values():
            Path(manifest["iq_path"]).unlink()
    if args.generate_only:
        print(root / "rf")
        return 0

    receiver_root = root / f"receiver{variant_suffix}"
    feature_root = root / f"features{variant_suffix}"
    receiver_root.mkdir(exist_ok=True)
    feature_root.mkdir(exist_ok=True)
    real_by_source = _load_real_clean(config)
    scored: dict[str, Any] = {}
    score_rows: list[dict[str, Any]] = []
    per_feature_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for candidate in config["candidates"]:
        name = candidate["name"]
        rf_manifest = manifests[name]
        rf_document = json.loads(rf_manifest.read_text(encoding="utf-8"))
        receiver_dir = receiver_root / rf_document["run_id"]
        receiver_manifest = receiver_dir / "manifest.json"
        feature_path = feature_root / f"{name}_tracking_features.csv"
        if receiver_manifest.is_file() and feature_path.is_file():
            if not args.resume:
                raise FileExistsError(receiver_manifest)
            receiver_document = json.loads(receiver_manifest.read_text(encoding="utf-8"))
            if receiver_document.get("source", {}).get("rf_manifest_sha256") != _sha256(rf_manifest):
                raise ValueError(f"receiver provenance mismatch: {name}")
            tracking = receiver_document.get("tracking", {})
            if int(tracking.get("tap_count", 3)) != receiver_tap_count:
                raise ValueError(f"receiver tap-count mismatch: {name}")
            command = receiver_document.get("receiver", {}).get("command", [])
            if command and str(Path(command[0]).resolve()) != str(Path(receiver_executable).resolve()):
                raise ValueError(f"receiver executable mismatch: {name}")
        else:
            if receiver_dir.exists():
                raise FileExistsError(f"partial receiver directory requires manual review: {receiver_dir}")
            print(f"[receiver] {name}", flush=True)
            receiver_manifest = run_receiver(
                rf_manifest,
                receiver_root,
                executable=receiver_executable,
                channel_count=int(config["gnss_sdr"]["channel_count"]),
                timeout_seconds=int(config["gnss_sdr"]["timeout_seconds"]),
                tracking_tap_count=receiver_tap_count,
                tracking_tap_spacing_chips=float(config["gnss_sdr"]["tracking_tap_spacing_chips"]),
            )
            export_receiver_run_tracking_feature_csv(
                receiver_manifest.parent,
                output_path=feature_path,
                tap_count=feature_tap_count,
                window_s=float(config["features"]["window_s"]),
                stride_s=float(config["features"]["stride_s"]),
                min_epochs=int(config["features"]["min_epochs"]),
                label="normal",
            )
        rows = _read_csv(feature_path)
        state = _receiver_state_summary(receiver_manifest.parent)
        result = _score_candidate(candidate, rows, real_by_source, state, config)
        result["rf_manifest"] = str(rf_manifest)
        result["rf_manifest_sha256"] = _sha256(rf_manifest)
        result["receiver_manifest"] = str(receiver_manifest)
        result["receiver_manifest_sha256"] = _sha256(receiver_manifest)
        result["feature_csv"] = str(feature_path)
        result["feature_csv_sha256"] = _sha256(feature_path)
        scored[name] = result
        combined = result["comparisons"]["cleanCombined"]
        score_rows.append({
            "candidate": name,
            "overall_gate_status": result["overall_gate_status"],
            "fidelity_loss": result["fidelity_loss"],
            "domain_auc": combined["domain_classifier"]["pooled_separability_auc"],
            "mean_fold_domain_auc": combined["domain_classifier"]["mean_fold_separability_auc"],
            "median_ks": combined["distribution"]["median_ks_statistic"],
            "median_robust_shift": combined["distribution"]["median_robust_median_shift"],
            "lock_fraction": state["carrier_lock_above_0_5_fraction"],
            "cn0_median_db_hz": state["cn0_db_hz_median"],
            "tracked_prn_count": state["tracked_prn_count"],
            "feature_rows": len(rows),
        })
        for comparison, comparison_result in result["comparisons"].items():
            per_feature_rows.extend({
                "candidate": name,
                "comparison": comparison,
                **row,
            } for row in comparison_result["per_feature"])
            fold_rows.extend({
                "candidate": name,
                "comparison": comparison,
                **row,
            } for row in comparison_result["domain_classifier"]["folds"])
        print(
            f"[score] {name} status={result['overall_gate_status']} "
            f"auc={score_rows[-1]['domain_auc']:.6f} loss={result['fidelity_loss']:.6f}",
            flush=True,
        )

    status_rank = {"pass": 0, "conditional": 1, "stop": 2}
    ranking = sorted(
        scored,
        key=lambda name: (status_rank[scored[name]["overall_gate_status"]], scored[name]["fidelity_loss"]),
    )
    best_name = ranking[0]
    _write_csv(root / f"candidate_scores{variant_suffix}.csv", score_rows)
    _write_csv(root / f"per_feature_metrics{variant_suffix}.csv", per_feature_rows)
    _write_csv(root / f"domain_classifier_folds{variant_suffix}.csv", fold_rows)
    summary = {
        "schema": "gnss-doppler-lab.simulation-v4-normal-calibration",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "runner_script_sha256": _sha256(Path(__file__)),
        "receiver_variant": args.receiver_variant or "default",
        "effective_receiver": {
            "executable": receiver_executable,
            "tracking_tap_count": receiver_tap_count,
            "feature_tap_count": feature_tap_count,
            "tracking_tap_spacing_chips": float(config["gnss_sdr"]["tracking_tap_spacing_chips"]),
        },
        "data_boundary": {
            **config["data_boundary"],
            "forbidden_scenarios_accessed": False,
            "spoofing_data_generated": False,
        },
        "components": components,
        "candidates": scored,
        "ranking": ranking,
        "selected_candidate": best_name,
        "selected_candidate_gate_status": scored[best_name]["overall_gate_status"],
        "selection_interpretation": (
            "qualified for the next independent-seed normal calibration stage"
            if scored[best_name]["overall_gate_status"] in {"pass", "conditional"}
            else "best available candidate only; not qualified for detector training or scale generation"
        ),
        "ranking_contract": {
            "gate_status_is_primary": True,
            "fidelity_loss_is_secondary": True,
            "weights": config["ranking_weights"],
            "receiver_state_target": config["receiver_state_target"],
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "limitations": [
            "The sweep uses one location, epoch, receiver seed, and 30-second normal recording per candidate.",
            "TEXBAT cleanStatic and cleanDynamic are development calibration references, not final validation recordings.",
            "No TEXBAT spoofing scenario was accessed and no spoof detector was trained.",
        ],
    }
    summary_path = root / f"summary{variant_suffix}.json"
    _atomic_json(summary_path, summary)
    print(json.dumps({
        "summary": str(summary_path),
        "selected_candidate": best_name,
        "selected_candidate_gate_status": scored[best_name]["overall_gate_status"],
        "ranking": ranking,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
