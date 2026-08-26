#!/usr/bin/env python3
"""Run one train-only receiver/RF challenge: coherent spoof vs PRN multipath."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
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

import run_simulation_v4_normal_independent_validation as normal  # noqa: E402
from gnss_doppler_lab.clock_centered_geometry import (  # noqa: E402
    fit_clock_centered_geometry,
)
from gnss_doppler_lab.correlator_geometry import (  # noqa: E402
    TemplateDelayEstimator,
    build_complex_template_bank,
    complex_profile_features,
)
from gnss_doppler_lab.gnss_sdr import run_receiver  # noqa: E402
from gnss_doppler_lab.peak_mixture_law import parse_gps_sdr_sim_los_table  # noqa: E402
from gnss_doppler_lab.rf_config import (  # noqa: E402
    InputConfig,
    OutputConfig,
    RFGenerationConfig,
    Scenario,
    SimulatorConfig,
    StaticPosition,
)
from gnss_doppler_lab.rf_impairments import ImpairmentConfig  # noqa: E402
from gnss_doppler_lab.satellite_multipath import (  # noqa: E402
    PrnMultipathGpsSdrSimRunner,
    independent_echoes,
)
from gnss_doppler_lab.simulation_v4 import (  # noqa: E402
    SimulationScenario,
    compose_paired_iq,
)
from gnss_doppler_lab.tracking_peaks import (  # noqa: E402
    available_tracking_prns,
    load_receiver_tracking_peak_series_segments,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/cgc_rf_challenge_pilot_v1.json"


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_pair(config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    split_path = _repo_path(config["split_config"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    matches = [
        pair for pair in split["pairs"]
        if pair["paired_group_id"] == config["pair_id"]
    ]
    if len(matches) != 1 or matches[0]["split"] != "train":
        raise ValueError("RF challenge requires exactly one configured train pair")
    return matches[0], split_path


def _rf_config(
    pair: dict[str, Any],
    normal_profile: dict[str, Any],
    output_root: Path,
    executable: Path,
) -> RFGenerationConfig:
    position = pair["position"]
    return RFGenerationConfig(
        version=1,
        scenario=Scenario(
            name=f"{pair['paired_group_id']}-independent-multipath",
            constellation="GPS",
            signal="L1CA",
            utc=datetime.fromisoformat(pair["utc"].replace("Z", "+00:00")),
            duration_seconds=int(pair["duration_seconds"]),
            position=StaticPosition(
                float(position["latitude_deg"]),
                float(position["longitude_deg"]),
                float(position["altitude_m"]),
            ),
        ),
        input=InputConfig(_repo_path(normal_profile["input"]["rinex_nav"])),
        output=OutputConfig(
            output_root,
            int(normal_profile["rf_profile"]["rf_sample_rate_hz"]),
            "s8_iq",
        ),
        simulator=SimulatorConfig(str(executable)),
        impairments=ImpairmentConfig(),
    )


def _ensure_component(
    root: Path,
    config: dict[str, Any],
    pair: dict[str, Any],
    normal_profile: dict[str, Any],
    *,
    resume: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    component_dir = root / "component"
    iq_path = component_dir / "multipath_gps_l1ca_s8_iq.bin"
    log_path = component_dir / "gps-sdr-sim.log"
    manifest_path = component_dir / "manifest.json"
    multipath = config["multipath"]
    simulator = _repo_path(multipath["simulator_executable"])
    echoes = independent_echoes(
        multipath["prns"],
        seed=int(multipath["seed"]),
        delay_chips_range=tuple(float(x) for x in multipath["delay_chips_range"]),
        amplitude_range=tuple(float(x) for x in multipath["amplitude_range"]),
    )
    rf_config = _rf_config(pair, normal_profile, component_dir, simulator)
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not iq_path.is_file() or _sha256(iq_path) != manifest["iq"]["sha256"]:
            raise ValueError("multipath component integrity failure")
        simulator_record = manifest.get("simulator", {})
        patch_path = _repo_path(multipath["simulator_patch"])
        executable_sha256 = _sha256(simulator)
        patch_sha256 = _sha256(patch_path)
        if simulator_record.get("executable_sha256") == executable_sha256:
            if simulator_record.get("patch_sha256") != patch_sha256:
                raise ValueError(
                    "multipath simulator patch provenance mismatch"
                )
        else:
            equivalence = manifest.get("reproducibility_equivalence", {})
            full = equivalence.get("full_component", {})
            if (
                equivalence.get("artifact_executable_sha256")
                != simulator_record.get("executable_sha256")
                or equivalence.get("corrected_executable_sha256")
                != executable_sha256
                or equivalence.get("corrected_patch_sha256") != patch_sha256
                or full.get("artifact_sha256") != manifest["iq"]["sha256"]
                or full.get("corrected_regeneration_sha256")
                != manifest["iq"]["sha256"]
                or int(full.get("artifact_bytes", -1))
                != int(manifest["iq"]["bytes"])
                or int(full.get("corrected_regeneration_bytes", -1))
                != int(manifest["iq"]["bytes"])
            ):
                raise ValueError(
                    "multipath simulator executable provenance mismatch"
                )
        if (
            simulator_record.get("upstream_commit")
            != multipath["simulator_upstream_commit"]
        ):
            raise ValueError(
                "multipath simulator upstream provenance mismatch"
            )
        return iq_path, log_path, manifest
    if component_dir.exists() and any(component_dir.iterdir()):
        raise FileExistsError(f"partial component directory: {component_dir}")
    print("[1/4] generating 29.9 s PRN-specific multipath RF component", flush=True)
    runner = PrnMultipathGpsSdrSimRunner(str(simulator), echoes)
    result = runner.run(rf_config, iq_path, log_path)
    expected = runner.expected_output_bytes(rf_config)
    if result["actual_bytes"] != expected:
        raise ValueError("patched simulator violated the pinned byte contract")
    manifest = {
        "schema": "gnss-doppler-lab.cgc-prn-multipath-component",
        "schema_version": 1,
        "pair": pair,
        "simulator": {
            "executable": str(simulator),
            "executable_sha256": _sha256(simulator),
            "upstream_commit": multipath["simulator_upstream_commit"],
            "patch": str(_repo_path(multipath["simulator_patch"])),
            "patch_sha256": _sha256(_repo_path(multipath["simulator_patch"])),
            "cli_contract": runner.cli_contract,
            "command": result["command"],
        },
        "multipath": result["multipath"],
        "iq": {
            "path": str(iq_path.resolve()),
            "sha256": _sha256(iq_path),
            "bytes": iq_path.stat().st_size,
            "rf_sample_rate_hz": rf_config.output.rf_sample_rate_hz,
        },
        "log": {"path": str(log_path.resolve()), "sha256": _sha256(log_path)},
        "scope": "offline train-only RF component",
    }
    _write_json(manifest_path, manifest)
    return iq_path, log_path, manifest


def multipath_run_id(pair: dict[str, Any]) -> str:
    """Return a collision-free receiver run ID while preserving pair-001."""
    pair_number = str(pair["paired_group_id"]).rsplit("-", 1)[-1]
    utc = datetime.fromisoformat(str(pair["utc"]).replace("Z", "+00:00"))
    return f"cgc-rf-mp-p{pair_number}_{utc.strftime('%Y%m%dT%H%M%SZ')}"


def _ensure_rf(
    root: Path,
    config: dict[str, Any],
    pair: dict[str, Any],
    normal_profile: dict[str, Any],
    component: Path,
    component_manifest: dict[str, Any],
    *,
    resume: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    rf_dir = root / "rf"
    iq_path = rf_dir / "gps_l1ca_s8_iq.bin"
    manifest_path = rf_dir / "manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not iq_path.is_file() or _sha256(iq_path) != manifest["iq"]["sha256"]:
            raise ValueError("multipath receiver RF integrity failure")
        return iq_path, manifest_path, manifest
    if rf_dir.exists() and any(rf_dir.iterdir()):
        raise FileExistsError(f"partial RF directory: {rf_dir}")
    print("[2/4] applying the frozen pair-001 frontend, gain, and AWGN", flush=True)
    source_normal_manifest_path = _repo_path(config["source_normal_rf_manifest"])
    source_normal = json.loads(
        source_normal_manifest_path.read_text(encoding="utf-8")
    )
    reference = source_normal["simulation_v4"]["receiver"]["reference"]
    receiver = normal._run_impairment(normal_profile, pair)
    scenario = SimulationScenario("independent_multipath", "steady_normal")
    composition = compose_paired_iq(
        component,
        component,
        {scenario.name: iq_path},
        (scenario,),
        sample_rate_hz=int(normal_profile["rf_profile"]["rf_sample_rate_hz"]),
        receiver=receiver,
        normal_target_rms=float(normal_profile["normal_target_rms"]),
        reference_override=reference,
    )
    report = composition["scenarios"][scenario.name]
    manifest = {
        "schema_version": 4,
        "run_id": multipath_run_id(pair),
        "scenario": {
            "name": f"{pair['paired_group_id']}-independent-multipath",
            "class": "multipath",
            "event": "steady_prn_specific_multipath",
            "is_spoofing": False,
            "split": "train",
            "paired_group_id": pair["paired_group_id"],
            "utc": pair["utc"],
            "duration_seconds": pair["duration_seconds"],
            "position": pair["position"],
        },
        "iq": {
            "path": iq_path.name,
            "sha256": report["sha256"],
            "actual_bytes": report["bytes"],
            "complex_samples": report["complex_samples"],
            "actual_duration_seconds": report["actual_duration_seconds"],
            "rf_sample_rate_hz": int(normal_profile["rf_profile"]["rf_sample_rate_hz"]),
            "sample_format": "s8_iq",
            "channels": 2,
        },
        "simulation_v4": {
            "truth": {
                "class": "multipath",
                "is_spoofing": False,
                "echoes": component_manifest["multipath"]["echoes"],
            },
            "receiver": {
                "requested": receiver.manifest(),
                "reference": composition["reference"],
                "reference_source_manifest": str(source_normal_manifest_path),
                "reference_source_manifest_sha256": _sha256(source_normal_manifest_path),
                "processing": composition["processing"],
            },
            "measurements": report,
            "source_component": component_manifest["iq"],
            "scope": "offline train-only RF challenge; no transmission",
        },
    }
    _write_json(manifest_path, manifest)
    return iq_path, manifest_path, manifest


def _ensure_receiver(
    source_manifest: Path,
    receiver_root: Path,
    receiver_config: dict[str, Any],
    *,
    resume: bool,
) -> Path:
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    expected = receiver_root / source["run_id"] / "manifest.json"
    executable = _repo_path(receiver_config["executable"])
    if expected.is_file():
        if not resume:
            raise FileExistsError(expected)
        document = json.loads(expected.read_text(encoding="utf-8"))
        if document["source"]["rf_manifest_sha256"] != _sha256(source_manifest):
            raise ValueError("receiver source provenance mismatch")
        if document["receiver"]["executable_sha256"] != _sha256(executable):
            raise ValueError("receiver executable provenance mismatch")
        return expected
    return run_receiver(
        source_manifest,
        receiver_root,
        executable=executable,
        channel_count=int(receiver_config["channel_count"]),
        timeout_seconds=int(receiver_config["timeout_seconds"]),
        tracking_tap_count=9,
        tracking_tap_spacing_chips=float(
            receiver_config["tracking_tap_spacing_chips"]
        ),
    )


def _axis(low: float, high: float, step: float) -> np.ndarray:
    count = int(round((high - low) / step)) + 1
    return np.linspace(low, high, count, dtype=np.float64)


def _estimator(controlled: dict[str, Any]) -> TemplateDelayEstimator:
    correlator = controlled["correlator"]
    template = controlled["template_estimator"]
    phase_axis = np.linspace(
        -np.pi,
        np.pi,
        int(template["phase_count"]),
        endpoint=False,
        dtype=np.float64,
    )
    bank = build_complex_template_bank(
        correlator["tap_offsets_chips"],
        prompt_index=int(correlator["prompt_index"]),
        delays_chips=_axis(
            float(template["delay_min_chips"]),
            float(template["delay_max_chips"]),
            float(template["delay_step_chips"]),
        ),
        centers_chips=_axis(
            float(template["center_min_chips"]),
            float(template["center_max_chips"]),
            float(template["center_step_chips"]),
        ),
        amplitude_ratios=_axis(
            float(template["amplitude_min"]),
            float(template["amplitude_max"]),
            float(template["amplitude_step"]),
        ),
        phases_rad=phase_axis,
    )
    return TemplateDelayEstimator(bank)


def _scenario_geometry(
    name: str,
    receiver_manifest: Path,
    estimator: TemplateDelayEstimator,
    los: dict[str, tuple[float, float, float]],
    *,
    bin_seconds: float,
    minimum_prns: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = receiver_manifest.parent
    delay_rows: list[dict[str, Any]] = []
    by_bin: dict[int, list[tuple[str, float, float]]] = {}
    for prn in available_tracking_prns(run_dir):
        if prn not in los:
            continue
        segments = load_receiver_tracking_peak_series_segments(
            run_dir, prn, tap_count=9, require_complex_taps=True
        )
        times = np.concatenate([segment.time_s for segment in segments])
        profiles = np.concatenate([segment.complex_taps for segment in segments])
        features = complex_profile_features(profiles, prompt_index=4)
        estimates, distances, _ = estimator.estimate(features)
        bins = np.floor(times / bin_seconds).astype(np.int64)
        for bin_index in np.unique(bins):
            mask = bins == bin_index
            delay = float(np.median(estimates[mask]))
            distance = float(np.median(distances[mask]))
            start = float(bin_index * bin_seconds)
            delay_rows.append({
                "scenario": name,
                "bin_index": int(bin_index),
                "bin_start_s": start,
                "prn": prn,
                "epoch_count": int(np.count_nonzero(mask)),
                "estimated_delay_chips": delay,
                "median_template_distance": distance,
            })
            by_bin.setdefault(int(bin_index), []).append((prn, delay, distance))
    geometry_rows: list[dict[str, Any]] = []
    for bin_index, rows in sorted(by_bin.items()):
        if len(rows) < minimum_prns:
            continue
        prns = [row[0] for row in rows]
        delays = np.asarray([row[1] for row in rows], dtype=np.float64)
        los_matrix = np.asarray([los[prn] for prn in prns], dtype=np.float64)
        fit = fit_clock_centered_geometry(los_matrix, delays)
        geometry_rows.append({
            "scenario": name,
            "bin_index": bin_index,
            "bin_start_s": float(bin_index * bin_seconds),
            "prn_count": len(prns),
            "complex_geometry_residual": fit.normalized_residual,
            "complex_geometry_coherence": fit.coherence,
            "clock_centered_geometry_residual": fit.clock_centered_normalized_residual,
            "directional_geometry_coherence": fit.directional_coherence,
            "clock_only_bias_chips": fit.clock_only_bias_chips,
            "fit_rank": fit.rank,
            "estimated_displacement_e_chips": float(fit.theta[0]),
            "estimated_displacement_n_chips": float(fit.theta[1]),
            "estimated_displacement_u_chips": float(fit.theta[2]),
            "estimated_clock_bias_chips": float(fit.theta[3]),
        })
    return delay_rows, geometry_rows


def _analyze(
    root: Path,
    config: dict[str, Any],
    multipath_receiver: Path,
    spoof_receiver: Path,
    component_log: Path,
) -> dict[str, Any]:
    controlled_path = _repo_path(config["controlled_cgc_config"])
    controlled = json.loads(controlled_path.read_text(encoding="utf-8"))
    estimator = _estimator(controlled)
    source_los_path = _repo_path(config["source_los_log"])
    source_los = parse_gps_sdr_sim_los_table(
        source_los_path.read_text(encoding="utf-8")
    )
    component_los = parse_gps_sdr_sim_los_table(
        component_log.read_text(encoding="utf-8")
    )
    if set(source_los) != set(component_los) or any(
        not np.allclose(source_los[prn], component_los[prn], atol=1e-12, rtol=0)
        for prn in source_los
    ):
        raise ValueError("multipath component LOS differs from frozen pair-001 geometry")
    analysis = config["analysis"]
    all_delays: list[dict[str, Any]] = []
    all_geometry: list[dict[str, Any]] = []
    for name, manifest in (
        ("independent_multipath", multipath_receiver),
        ("carryoff_spoof", spoof_receiver),
    ):
        delays, geometry = _scenario_geometry(
            name,
            manifest,
            estimator,
            source_los,
            bin_seconds=float(analysis["bin_seconds"]),
            minimum_prns=int(analysis["minimum_prns"]),
        )
        all_delays.extend(delays)
        all_geometry.extend(geometry)
    delay_path = root / "analysis" / "delay_estimates.csv"
    geometry_path = root / "analysis" / "geometry_scores.csv"
    _write_csv(delay_path, all_delays)
    _write_csv(geometry_path, all_geometry)
    start = float(analysis["comparison_start_seconds"])
    comparison = [
        row for row in all_geometry if float(row["bin_start_s"]) >= start
    ]
    labels = np.asarray(
        [1 if row["scenario"] == "carryoff_spoof" else 0 for row in comparison]
    )
    legacy_scores = -np.asarray([
        float(row["complex_geometry_residual"]) for row in comparison
    ])
    directional_scores = -np.asarray([
        float(row["clock_centered_geometry_residual"]) for row in comparison
    ])
    aucs = {
        "legacy_zero_referenced": float(roc_auc_score(labels, legacy_scores)),
        "clock_centered_directional": float(
            roc_auc_score(labels, directional_scores)
        ),
    }

    def scenario_medians(field: str) -> dict[str, float]:
        return {
            name: float(np.median([
                float(row[field])
                for row in comparison if row["scenario"] == name
            ]))
            for name in ("independent_multipath", "carryoff_spoof")
        }

    legacy_medians = scenario_medians("complex_geometry_residual")
    directional_medians = scenario_medians(
        "clock_centered_geometry_residual"
    )
    return {
        "comparison_start_seconds": start,
        "comparison_bin_count": len(comparison),
        "exploratory_geometry_auc": aucs,
        "median_geometry_residual": {
            "legacy_zero_referenced": legacy_medians,
            "clock_centered_directional": directional_medians,
        },
        "multipath_minus_spoof_residual_separation": {
            "legacy_zero_referenced": (
                legacy_medians["independent_multipath"]
                - legacy_medians["carryoff_spoof"]
            ),
            "clock_centered_directional": (
                directional_medians["independent_multipath"]
                - directional_medians["carryoff_spoof"]
            ),
        },
        "primary_candidate": "clock_centered_directional",
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
        "candidate_origin": (
            "post-hoc train-only physical correction after the legacy "
            "zero-referenced residual failed on this RF pilot"
        ),
        "legacy_failure_observed_first": True,
        "score_law": {
            "legacy": "SSE_full / sum(weight * delay^2)",
            "clock_centered_directional": (
                "SSE_full / sum(weight * (delay - weighted_mean(delay))^2)"
            ),
            "detection_score": "negative residual",
        },
        "next_gate": (
            "freeze this candidate, replicate on unused train pairs 002--006, "
            "then preregister the still-locked test protocol"
        ),
        "interpretation": "exploratory single-geometry receiver/RF pilot only",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pair, split_path = _load_pair(config)
    normal_profile_path = _repo_path(config["normal_profile"])
    normal_profile = json.loads(normal_profile_path.read_text(encoding="utf-8"))
    root = _repo_path(config["output_root"])
    if root.exists() and not args.resume:
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)

    component, component_log, component_manifest = _ensure_component(
        root, config, pair, normal_profile, resume=args.resume
    )
    _, multipath_rf_manifest, _ = _ensure_rf(
        root,
        config,
        pair,
        normal_profile,
        component,
        component_manifest,
        resume=args.resume,
    )
    print("[3/4] receiving multipath and carry-off spoof RF with complex 9 taps", flush=True)
    receiver_root = root / "receiver"
    multipath_receiver = _ensure_receiver(
        multipath_rf_manifest, receiver_root, config["gnss_sdr"], resume=args.resume
    )
    spoof_receiver = _ensure_receiver(
        _repo_path(config["source_spoof_rf_manifest"]),
        receiver_root,
        config["gnss_sdr"],
        resume=args.resume,
    )
    print("[4/4] estimating per-PRN delays and common-geometry residuals", flush=True)
    result = _analyze(
        root,
        config,
        multipath_receiver,
        spoof_receiver,
        component_log,
    )
    summary = {
        "schema": "gnss-doppler-lab.cgc-rf-challenge-pilot-result",
        "schema_version": 1,
        "role": config["role"],
        "claim_boundary": config["claim_boundary"],
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "pair": pair,
        "split_config": str(split_path),
        "split_config_sha256": _sha256(split_path),
        "normal_profile": str(normal_profile_path),
        "normal_profile_sha256": _sha256(normal_profile_path),
        "component_manifest": component_manifest,
        "multipath_receiver_manifest": {
            "path": str(multipath_receiver),
            "sha256": _sha256(multipath_receiver),
        },
        "spoof_receiver_manifest": {
            "path": str(spoof_receiver),
            "sha256": _sha256(spoof_receiver),
        },
        "multipath_simulator": {
            "executable": str(
                _repo_path(config["multipath"]["simulator_executable"])
            ),
            "executable_sha256": _sha256(
                _repo_path(config["multipath"]["simulator_executable"])
            ),
            "upstream_commit": config["multipath"][
                "simulator_upstream_commit"
            ],
            "patch_sha256": _sha256(
                _repo_path(config["multipath"]["simulator_patch"])
            ),
        },
        "analysis_implementation": {
            "legacy_frozen_module_sha256": _sha256(
                REPO_ROOT / "src/gnss_doppler_lab/correlator_geometry.py"
            ),
            "clock_centered_module_sha256": _sha256(
                REPO_ROOT / "src/gnss_doppler_lab/clock_centered_geometry.py"
            ),
        },
        "gnss_sdr": {
            "executable": str(_repo_path(config["gnss_sdr"]["executable"])),
            "executable_sha256": _sha256(
                _repo_path(config["gnss_sdr"]["executable"])
            ),
            "patch_sha256": _sha256(
                _repo_path(config["gnss_sdr"]["patch"])
            ),
        },
        "analysis": result,
        "data_boundary": config["data_boundary"],
    }
    summary_path = root / "summary.json"
    _write_json(summary_path, summary)
    print(json.dumps(summary["analysis"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
