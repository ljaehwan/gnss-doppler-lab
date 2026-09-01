#!/usr/bin/env python3
"""Run the static code/carrier-decoupling CGC development pilot."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for search_root in (ROOT / "scripts", ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_cgc_rf_challenge_pilot as challenge  # noqa: E402
import run_cgc_rf_geometry_aperture_validation as geometry  # noqa: E402
import run_simulation_v4_normal_independent_validation as normal  # noqa: E402
from gnss_doppler_lab.code_carrier_sim import (  # noqa: E402
    CodeCarrierGpsSdrSimRunner,
    DecoupledSimulationRequest,
    sha256,
    summarize_truth_triplet,
)
from gnss_doppler_lab.peak_mixture_law import parse_gps_sdr_sim_los_table  # noqa: E402
from gnss_doppler_lab.simulation_v4 import (  # noqa: E402
    SimulationScenario,
    SpoofEvent,
    build_carryoff_rows,
    compose_paired_iq,
)
from gnss_doppler_lab.static_reference_geometry import partial_f_score  # noqa: E402
import run_simulation_v4_paired_train_generation as source  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/experiments/cgc_code_carrier_decoupling_pilot_v1.json"
PROTOCOL = ROOT / "docs/results/cgc_code_carrier_decoupling_pilot_protocol_v1.md"


def path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def write_json(destination: Path, document: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(destination: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def read_csv(source_path: Path) -> list[dict[str, str]]:
    with source_path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify(record: dict[str, Any], label: str) -> Path:
    candidate = path(record["path"])
    if not candidate.is_file():
        raise FileNotFoundError(f"missing {label}: {candidate}")
    expected = record.get("sha256")
    if expected is not None and sha256(candidate) != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    expected_bytes = record.get("bytes")
    if expected_bytes is not None and candidate.stat().st_size != int(expected_bytes):
        raise ValueError(f"{label} byte count mismatch")
    return candidate


def load_config(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "gnss-doppler-lab.cgc-code-carrier-decoupling-pilot-config":
        raise ValueError("unsupported pilot schema")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported pilot schema version")
    if config["experiment"]["role"] != "development-only static receiver-RF mechanism pilot":
        raise ValueError("pilot claim role drifted")
    if config["analysis"]["minimum_prns"] != 8:
        raise ValueError("minimum PRN support drifted")
    if float(config["analysis"]["partial_f_p_alarm_threshold"]) != 0.06028418845288192:
        raise ValueError("frozen Partial-F threshold drifted")
    for key in ("normal_profile", "controlled_template", "rinex_nav", "authentic_component", "authentic_los_log", "normal_rf_manifest"):
        verify(config["inputs"][key], key)
    for key in ("decoupled_simulator", "simulator_patch", "receiver"):
        verify(config["tools"][key], key)
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    return config


def event(config: dict[str, Any]) -> SpoofEvent:
    item = config["carryoff"]
    return SpoofEvent(
        start_seconds=float(item["start_seconds"]),
        transition_seconds=float(item["transition_seconds"]),
        target_offset_enu_m=tuple(float(value) for value in item["target_offset_enu_m"]),
        initial_advantage_db=float(item["initial_advantage_db"]),
        final_advantage_db=float(item["final_advantage_db"]),
        power_ramp_seconds=float(item["power_ramp_seconds"]),
    )


def ensure_trajectories(root: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    directory = root / "trajectories"; directory.mkdir(parents=True, exist_ok=True)
    authentic_path, false_path = directory / "authentic.csv", directory / "false_code.csv"
    run = config["static_geometry"]; position = run["position"]
    rows = tuple(
        (index / 10.0, float(position["latitude_deg"]), float(position["longitude_deg"]), float(position["altitude_m"]))
        for index in range(int(run["duration_seconds"]) * 10)
    )
    false_rows = build_carryoff_rows(rows, event(config))

    def payload(values):
        return "".join(f"{time_s:.1f},{lat:.9f},{lon:.9f},{height:.4f}\n" for time_s, lat, lon, height in values)

    for destination, text in ((authentic_path, payload(rows)), (false_path, payload(false_rows))):
        if destination.is_file():
            if destination.read_text(encoding="ascii") != text:
                raise ValueError(f"trajectory resume mismatch: {destination}")
        else:
            destination.write_text(text, encoding="ascii")
    write_json(directory / "manifest.json", {
        "schema": "gnss-doppler-lab.cgc-code-carrier-trajectories",
        "schema_version": 1,
        "authentic": {"path": str(authentic_path.resolve()), "sha256": sha256(authentic_path), "rows": len(rows)},
        "false_code": {"path": str(false_path.resolve()), "sha256": sha256(false_path), "rows": len(false_rows)},
        "carryoff": asdict(event(config)),
    })
    return authentic_path, false_path


def simulator(config: dict[str, Any]) -> CodeCarrierGpsSdrSimRunner:
    return CodeCarrierGpsSdrSimRunner(
        path(config["tools"]["decoupled_simulator"]["path"]),
        path(config["tools"]["simulator_patch"]["path"]),
    )


def request(config: dict[str, Any], code: Path, carrier: Path | None, mode: str, sample_rate: int) -> DecoupledSimulationRequest:
    run = config["static_geometry"]
    return DecoupledSimulationRequest(
        nav=path(config["inputs"]["rinex_nav"]["path"]),
        code_motion=code,
        carrier_motion=carrier,
        utc=datetime.fromisoformat(run["utc"].replace("Z", "+00:00")),
        duration_seconds=int(run["duration_seconds"]),
        sample_rate_hz=sample_rate,
        mode=mode,
    )


def run_truth(root: Path, config: dict[str, Any], authentic: Path, false: Path) -> Path:
    directory = root / "truth_preflight"; directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        return summary_path
    runner = simulator(config); rate = int(config["analysis"]["truth_sample_rate_hz"])
    specs = {
        "authentic": request(config, authentic, None, "coupled", rate),
        "carrier_coupled": request(config, false, None, "coupled", rate),
        "doppler_locked": request(config, false, authentic, "doppler_locked", rate),
    }
    records: dict[str, Any] = {}
    for name, item in specs.items():
        records[name] = runner.run(item, directory / f"{name}.bin", directory / f"{name}.csv", directory / f"{name}.log")
    metrics = summarize_truth_triplet(
        directory / "authentic.csv", directory / "carrier_coupled.csv", directory / "doppler_locked.csv",
        hold_start_seconds=float(config["carryoff"]["hold_start_seconds"]),
    )
    gates = {
        "identical_code_range": metrics["locked_vs_coupled_code_range_max_abs_m"] <= 1e-6,
        "identical_code_rate": metrics["locked_vs_coupled_code_rate_max_abs_mps"] <= 1e-6,
        "locked_carrier_range_matches_authentic": metrics["locked_vs_authentic_carrier_range_max_abs_m"] <= 1e-6,
        "locked_carrier_rate_matches_authentic": metrics["locked_vs_authentic_carrier_rate_max_abs_mps"] <= 1e-6,
        "nonzero_code_carryoff": metrics["locked_code_vs_carrier_hold_max_abs_m"] >= 20.0,
    }
    document = {"schema": "gnss-doppler-lab.cgc-code-carrier-truth-preflight", "schema_version": 1, "metrics": metrics, "gates": gates, "passed": all(gates.values()), "runs": records}
    write_json(summary_path, document)
    for name in specs:
        (directory / f"{name}.bin").unlink()
    if not document["passed"]:
        raise RuntimeError("code/carrier truth preflight failed")
    return summary_path


def ensure_component(root: Path, config: dict[str, Any], authentic: Path, false: Path, mode: str) -> tuple[Path, Path, dict[str, Any]]:
    directory = root / "components" / mode; directory.mkdir(parents=True, exist_ok=True)
    iq, truth, log, manifest_path = directory / "counterfeit.bin", directory / "truth.csv", directory / "simulator.log", directory / "manifest.json"
    if manifest_path.is_file():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not iq.is_file() or sha256(iq) != document["iq"]["sha256"]:
            raise ValueError(f"{mode} component resume integrity failure")
        return iq, manifest_path, document
    carrier = authentic if mode == "doppler-locked" else None
    sim_mode = "doppler_locked" if mode == "doppler-locked" else "coupled"
    record = simulator(config).run(
        request(config, false, carrier, sim_mode, int(config["analysis"]["rf_sample_rate_hz"])),
        iq, truth, log,
    )
    document = {"schema": "gnss-doppler-lab.cgc-code-carrier-counterfeit-component", "schema_version": 1, "condition": mode, **record}
    write_json(manifest_path, document)
    return iq, manifest_path, document


def ensure_rf(root: Path, config: dict[str, Any], mode: str, component: Path, component_manifest: Path) -> tuple[Path, Path, dict[str, Any]]:
    directory = root / "conditions" / mode / "rf"; directory.mkdir(parents=True, exist_ok=True)
    iq, manifest_path = directory / "gps_l1ca_s8_iq.bin", directory / "manifest.json"
    if manifest_path.is_file():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not iq.is_file() or sha256(iq) != document["iq"]["sha256"]:
            raise ValueError(f"{mode} composed RF resume integrity failure")
        return iq, manifest_path, document
    profile = json.loads(path(config["inputs"]["normal_profile"]["path"]).read_text(encoding="utf-8"))
    run = config["static_geometry"]
    receiver = normal._run_impairment(profile, run)
    scenario = SimulationScenario(mode, "carryoff_spoof", spoofing=event(config))
    normal_manifest_path = path(config["inputs"]["normal_rf_manifest"]["path"])
    normal_manifest = json.loads(normal_manifest_path.read_text(encoding="utf-8"))
    composition = compose_paired_iq(
        path(config["inputs"]["authentic_component"]["path"]), component,
        {mode: iq}, (scenario,), sample_rate_hz=int(config["analysis"]["rf_sample_rate_hz"]),
        receiver=receiver, normal_target_rms=float(profile["normal_target_rms"]),
        reference_override=normal_manifest["validation"]["composition_reference"],
    )
    report = composition["scenarios"][mode]
    normal_iq = Path(normal_manifest["iq"]["canonical_storage_path"])
    prefix = source.compare_prefix(normal_iq, iq, int(event(config).start_seconds * int(config["analysis"]["rf_sample_rate_hz"])))
    if prefix.get("byte_identical") is not True:
        raise RuntimeError(f"{mode} pre-onset RF is not paired to the pinned normal")
    document = {
        "schema_version": 4,
        "run_id": f"cgc-cc-pilot-{mode}",
        "scenario": {"name": mode, "class": "spoofing", "event": "carryoff", "is_spoofing": True, "domain": "static", **run},
        "iq": {"path": iq.name, "sha256": report["sha256"], "actual_bytes": report["bytes"], "complex_samples": report["complex_samples"], "actual_duration_seconds": report["actual_duration_seconds"], "rf_sample_rate_hz": int(config["analysis"]["rf_sample_rate_hz"]), "sample_format": "s8_iq", "channels": 2},
        "simulation_v4": {"truth": {"class": "spoofing", "is_spoofing": True, "condition": mode, "carryoff": asdict(event(config))}, "receiver": {"requested": receiver.manifest(), "reference": composition["reference"], "processing": composition["processing"]}, "measurements": report, "paired_prefix_check": prefix},
        "generation": {"component_manifest": str(component_manifest.resolve()), "component_manifest_sha256": sha256(component_manifest), "config_sha256": sha256(DEFAULT_CONFIG)},
        "claim_boundary": config["claim_boundary"],
    }
    write_json(manifest_path, document)
    return iq, manifest_path, document


def ensure_receiver(root: Path, config: dict[str, Any], mode: str, rf_manifest: Path) -> Path:
    receiver_root = root / "conditions" / mode / "receiver"
    expected = receiver_root / f"cgc-cc-pilot-{mode}" / "manifest.json"
    receiver = config["tools"]["receiver"]
    return challenge._ensure_receiver(
        rf_manifest, receiver_root,
        {"executable": str(path(receiver["path"])), "channel_count": int(receiver["channel_count"]), "timeout_seconds": int(receiver["timeout_seconds"]), "tracking_tap_spacing_chips": float(receiver["tracking_tap_spacing_chips"])},
        resume=expected.is_file(),
    )


def score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        statistic, p_value = partial_f_score(float(row["clock_centered_geometry_residual"]), int(row["prn_count"]))
        result.append({**row, "partial_f": statistic, "partial_f_p_value": p_value, "partial_f_score": float(-np.log10(max(p_value, np.finfo(float).tiny)))})
    return result


def persistence(rows: list[dict[str, Any]], threshold: float) -> tuple[float | None, list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: float(row["bin_start_s"]))
    output = []
    first = None
    raw = [float(row["partial_f_p_value"]) <= threshold for row in ordered]
    for index, row in enumerate(ordered):
        active = index >= 4 and sum(raw[index - 4:index + 1]) >= 3
        if active and first is None:
            first = float(row["bin_start_s"])
        output.append({**row, "raw_spoof_alarm": raw[index], "persistent_spoof_alarm": active})
    return first, output


def condition_metrics(rows: list[dict[str, Any]], multipath: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    hold_start = float(config["carryoff"]["hold_start_seconds"])
    attack_start = float(config["carryoff"]["start_seconds"])
    threshold = float(config["analysis"]["partial_f_p_alarm_threshold"])
    primary = [row for row in rows if float(row["bin_start_s"]) >= hold_start]
    control = [row for row in multipath if float(row["bin_start_s"]) >= hold_start]
    if not primary or not control:
        raise ValueError("empty primary or multipath comparison interval")
    first, annotated = persistence(rows, threshold)
    first_after_onset = next(
        (float(row["bin_start_s"]) for row in annotated if float(row["bin_start_s"]) >= attack_start and row["persistent_spoof_alarm"]),
        None,
    )
    pre_attack = [row for row in annotated if float(row["bin_start_s"]) < attack_start]
    labels = np.r_[np.ones(len(primary), dtype=int), np.zeros(len(control), dtype=int)]
    scores = np.asarray([row["partial_f_score"] for row in primary + control], dtype=float)
    return {
        "primary_bin_count": len(primary),
        "minimum_primary_prns": min(int(row["prn_count"]) for row in primary),
        "median_clock_centered_residual": float(np.median([row["clock_centered_geometry_residual"] for row in primary])),
        "median_partial_f_p_value": float(np.median([row["partial_f_p_value"] for row in primary])),
        "raw_alarm_rate": float(np.mean([float(row["partial_f_p_value"]) <= threshold for row in primary])),
        "first_persistent_alarm_s": first,
        "first_persistent_alarm_at_or_after_onset_s": first_after_onset,
        "detection_latency_from_carryoff_start_s": None if first_after_onset is None else first_after_onset - attack_start,
        "pre_attack_raw_alarm_count": sum(bool(row["raw_spoof_alarm"]) for row in pre_attack),
        "pre_attack_persistent_alarm_count": sum(bool(row["persistent_spoof_alarm"]) for row in pre_attack),
        "auc_vs_existing_seoul_multipath": float(roc_auc_score(labels, scores)),
        "annotated_rows": annotated,
    }


def run_rf(root: Path, config: dict[str, Any], authentic: Path, false: Path) -> Path:
    controlled = json.loads(path(config["inputs"]["controlled_template"]["path"]).read_text(encoding="utf-8"))
    estimator = challenge._estimator(controlled)
    los = parse_gps_sdr_sim_los_table(path(config["inputs"]["authentic_los_log"]["path"]).read_text(encoding="utf-8"))
    multipath_raw = read_csv(path(config["inputs"]["multipath_geometry_scores"]["path"]))
    multipath = score_rows([{key: (float(value) if key not in {"scenario"} else value) for key, value in row.items()} for row in multipath_raw])
    outcomes: dict[str, Any] = {}
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    for mode in ("carrier-coupled", "doppler-locked"):
        component, component_manifest, _ = ensure_component(root, config, authentic, false, mode)
        _, rf_manifest, _ = ensure_rf(root, config, mode, component, component_manifest)
        receiver_manifest = ensure_receiver(root, config, mode, rf_manifest)
        delays, geometry_rows = geometry.analyze_stream(mode, receiver_manifest, estimator, los, config, 9)
        scored = score_rows(geometry_rows); first, annotated = persistence(scored, float(config["analysis"]["partial_f_p_alarm_threshold"]))
        del first
        write_csv(root / "conditions" / mode / "delay_estimates.csv", delays)
        write_csv(root / "conditions" / mode / "geometry_scores.csv", annotated)
        metrics = condition_metrics(scored, multipath, config)
        metrics.pop("annotated_rows")
        outcomes[mode] = {"metrics": metrics, "receiver_manifest": {"path": str(receiver_manifest.resolve()), "sha256": sha256(receiver_manifest)}, "rf_manifest": {"path": str(rf_manifest.resolve()), "sha256": sha256(rf_manifest)}, "tracked_geometry_bins": len(scored)}
        rows_by_mode[mode] = scored
    coupled = {int(row["bin_index"]): row for row in rows_by_mode["carrier-coupled"]}
    locked = {int(row["bin_index"]): row for row in rows_by_mode["doppler-locked"]}
    common = sorted(set(coupled) & set(locked))
    comparison = {
        "common_bins": len(common),
        "partial_f_score_correlation": float(np.corrcoef([coupled[i]["partial_f_score"] for i in common], [locked[i]["partial_f_score"] for i in common])[0, 1]),
        "median_absolute_partial_f_score_difference": float(np.median([abs(coupled[i]["partial_f_score"] - locked[i]["partial_f_score"]) for i in common])),
        "median_absolute_residual_difference": float(np.median([abs(coupled[i]["clock_centered_geometry_residual"] - locked[i]["clock_centered_geometry_residual"]) for i in common])),
    }
    observability_path = root / "doppler_observability" / "summary.json"
    summary = {
        "schema": "gnss-doppler-lab.cgc-code-carrier-decoupling-pilot-result", "schema_version": 1,
        "config": {"path": str(DEFAULT_CONFIG.resolve()), "sha256": sha256(DEFAULT_CONFIG)},
        "protocol": {"path": str(PROTOCOL.resolve()), "sha256": sha256(PROTOCOL)},
        "conditions": outcomes, "coupled_vs_locked": comparison,
        "doppler_observability": None if not observability_path.is_file() else {
            "path": str(observability_path.resolve()),
            "sha256": sha256(observability_path),
        },
        "claim_boundary": config["claim_boundary"],
    }
    summary_path = root / "summary.json"; write_json(summary_path, summary)
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("truth", "rf", "all"), default="all")
    args = parser.parse_args()
    config_path = args.config.resolve(); config = load_config(config_path)
    root = Path(config["output_root"]); root.mkdir(parents=True, exist_ok=True)
    authentic, false = ensure_trajectories(root, config)
    truth_summary = run_truth(root, config, authentic, false)
    print(f"truth preflight: {truth_summary}", flush=True)
    if args.phase == "truth":
        return 0
    summary = run_rf(root, config, authentic, false)
    print(summary.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
