#!/usr/bin/env python3
"""Run the single-release frozen CGC direction check on TEXBAT DS1--DS3."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
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
from run_gcmr_texbat_external import preflight_ds4_alternate  # noqa: E402
from gnss_doppler_lab.clock_centered_geometry import fit_clock_centered_geometry  # noqa: E402
from gnss_doppler_lab.correlator_geometry import complex_profile_features  # noqa: E402
from gnss_doppler_lab.gcmr_geometry import (  # noqa: E402
    ephemeris_health_selection,
    parse_gnss_sdr_gps_ephemeris_xml,
    satellite_observation,
)

DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/cgc_texbat_external_v1.json"
PROTOCOL_PATH = REPO_ROOT / "docs/results/cgc_texbat_external_protocol_v1.md"
RELEASE_TOKEN = "RELEASE-CGC-TEXBAT-EXTERNAL-V1"
RELEASE_INPUTS = (
    "configs/experiments/cgc_texbat_external_v1.json",
    "docs/results/cgc_texbat_external_protocol_v1.md",
    "scripts/run_cgc_texbat_external.py",
)
EXPECTED_SCENARIOS = ["ds1", "ds2", "ds3"]
EXPECTED_ARRAYS = {"complex_iq", "sample_count", "time_s", "prn", "channel", "segment_index", "cn0_db_hz"}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _verify_record(record: dict[str, str], label: str) -> Path:
    path = _repo_path(record["path"])
    observed = _sha256(path)
    if observed != record["sha256"]:
        raise ValueError(f"{label} hash mismatch: {observed}")
    return path


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "gnss-doppler-lab.cgc-texbat-external-config" or config.get("schema_version") != 1:
        raise ValueError("unsupported TEXBAT external config")
    experiment = config.get("experiment", {})
    if experiment.get("name") != "cgc-texbat-external-v1" or experiment.get("candidate_commit") != "60280575737ff70618c5619ff7522c516bbdc67a":
        raise ValueError("experiment identity drifted")
    if experiment.get("runner_path") != "scripts/run_cgc_texbat_external.py":
        raise ValueError("runner path drifted")
    frozen = config.get("frozen_candidate", {})
    for key in ("clock_centered_module", "correlator_geometry_module", "template_runner", "controlled_template_config", "geometry_module", "preflight_module"):
        _verify_record(frozen[key], key)
    if frozen.get("residual") != "SSE_full / sum(weight * (delay - weighted_mean(delay))^2)" or frozen.get("detection_score") != "negative residual":
        raise ValueError("frozen score law drifted")
    if frozen.get("absolute_threshold_applied") is not False or frozen.get("threshold_or_calibration_fitting") is not False:
        raise ValueError("threshold use is forbidden")
    _verify_record(config["prior_simulation_evidence"], "prior simulation evidence")

    analysis = config.get("analysis", {})
    expected_analysis = {
        "bin_seconds": 1.0, "minimum_prns": 8, "stable_pre_seconds": [30.0, 90.0],
        "excluded_transition_seconds": [90.0, 110.0], "stable_post_start_seconds": 110.0,
        "epoch_chunk_size": 50000, "cn0_weighting": False,
    }
    if any(analysis.get(key) != value for key, value in expected_analysis.items()):
        raise ValueError("analysis contract drifted")
    evaluation = config.get("evaluation", {})
    expected_evaluation = {
        "required_scenario_count": 3, "required_positive_change_scenario_count": 3,
        "minimum_stable_pre_bins_per_scenario": 50, "minimum_stable_post_bins_per_scenario": 300,
        "minimum_prns_per_geometry_bin": 8, "threshold_fitting": False,
        "post_release_tuning_or_retest": False,
    }
    if any(evaluation.get(key) != value for key, value in expected_evaluation.items()):
        raise ValueError("evaluation contract drifted")
    support = config.get("input_support_observed_before_score", {})
    if support.get("score_accessed") is not False or support.get("healthy_tracked_prns_each") != 11:
        raise ValueError("input-only support declaration drifted")

    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or [row.get("name") for row in scenarios] != EXPECTED_SCENARIOS:
        raise ValueError("TEXBAT scenario roster drifted")
    sources = []
    for row in scenarios:
        paths = {key: _verify_record(row[key], f"{row['name']} {key}") for key in ("complex_epoch_npz", "export_manifest", "ephemeris", "nmea")}
        manifest = json.loads(paths["export_manifest"].read_text(encoding="utf-8"))
        if manifest.get("scenario_id") != row["name"] or manifest.get("feature_schema", {}).get("tensor") != "complex_iq":
            raise ValueError(f"export manifest schema drifted: {row['name']}")
        if manifest.get("output", {}).get("sha256") != row["complex_epoch_npz"]["sha256"] or manifest.get("output", {}).get("shape", [None, None])[1:] != [9, 2]:
            raise ValueError(f"export manifest output drifted: {row['name']}")
        sources.append({**row, "paths": paths, "manifest": manifest})
    output_root = _repo_path(config["output_root"])
    if output_root != REPO_ROOT / "artifacts/cgc_texbat_external_v1":
        raise ValueError("output root drifted")
    controlled = json.loads(_verify_record(frozen["controlled_template_config"], "controlled template config").read_text(encoding="utf-8"))
    return {"sources": sources, "controlled": controlled, "output_root": output_root}


def region(bin_start_s: float) -> str:
    value = float(bin_start_s)
    if 30.0 <= value < 90.0:
        return "stable_pre"
    if value >= 110.0:
        return "stable_post"
    return "excluded"


def summarize_scenario(name: str, geometry: list[dict[str, Any]]) -> dict[str, Any]:
    pre = [row for row in geometry if row["region"] == "stable_pre"]
    post = [row for row in geometry if row["region"] == "stable_post"]
    if not pre or not post:
        raise ValueError(f"empty stable region: {name}")
    pre_values = np.asarray([row["clock_centered_geometry_residual"] for row in pre], dtype=np.float64)
    post_values = np.asarray([row["clock_centered_geometry_residual"] for row in post], dtype=np.float64)
    labels = np.r_[np.zeros(len(pre_values), dtype=np.int64), np.ones(len(post_values), dtype=np.int64)]
    residuals = np.r_[pre_values, post_values]
    pre_median = float(np.median(pre_values))
    post_median = float(np.median(post_values))
    return {
        "scenario": name,
        "stable_pre_bin_count": len(pre),
        "stable_post_bin_count": len(post),
        "minimum_prn_count": min(int(row["prn_count"]) for row in pre + post),
        "stable_pre_median_residual": pre_median,
        "stable_post_median_residual": post_median,
        "pre_minus_post_residual_change": pre_median - post_median,
        "positive_preregistered_direction": pre_median > post_median,
        "secondary_serial_bin_auc": float(roc_auc_score(labels, -residuals)),
    }


def _estimate_epochs(estimator: Any, complex_iq: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    count = len(complex_iq)
    estimates = np.empty(count, dtype=np.float64)
    distances = np.empty(count, dtype=np.float64)
    for start in range(0, count, chunk_size):
        end = min(count, start + chunk_size)
        profiles = complex_iq[start:end, :, 0].astype(np.float64) + 1j * complex_iq[start:end, :, 1].astype(np.float64)
        features = complex_profile_features(profiles, prompt_index=4)
        estimates[start:end], distances[start:end], _ = estimator.estimate(features)
    return estimates, distances


def evaluate_source(source: dict[str, Any], analysis: dict[str, Any], estimator: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    name = source["name"]
    with np.load(source["paths"]["complex_epoch_npz"], allow_pickle=False) as loaded:
        if set(loaded.files) != EXPECTED_ARRAYS:
            raise ValueError(f"NPZ arrays drifted: {name}")
        arrays = {key: loaded[key] for key in loaded.files}
    complex_iq = arrays["complex_iq"]
    count = len(complex_iq)
    if complex_iq.shape != (count, 9, 2) or any(arrays[key].shape != (count,) for key in EXPECTED_ARRAYS - {"complex_iq"}):
        raise ValueError(f"NPZ shape mismatch: {name}")
    if count != int(source["manifest"]["output"]["row_count"]):
        raise ValueError(f"NPZ row count mismatch: {name}")
    times = np.asarray(arrays["time_s"], dtype=np.float64)
    prns = np.asarray(arrays["prn"], dtype=np.int64)
    if not np.isfinite(times).all() or np.any(prns < 1) or np.any(prns > 32):
        raise ValueError(f"invalid time or PRN array: {name}")

    ephemerides = parse_gnss_sdr_gps_ephemeris_xml(source["paths"]["ephemeris"])
    tracked = set(int(value) for value in np.unique(prns))
    receiver_root = source["paths"]["nmea"].parent
    preflight = preflight_ds4_alternate(
        receiver_root, ephemerides, configured_tow0_s=float(source["tow0_s"]),
        tracked_prns=tracked, min_prns=int(analysis["minimum_prns"]),
    )
    healthy, health = ephemeris_health_selection(ephemerides, tracked_prns=tracked, min_prns=int(analysis["minimum_prns"]))
    receiver_ecef = preflight["receiver_position_contract"]["ecef"]
    estimates, distances = _estimate_epochs(estimator, complex_iq, int(analysis["epoch_chunk_size"]))

    bins = np.floor(times / float(analysis["bin_seconds"])).astype(np.int64)
    valid = np.isin(prns, np.asarray(sorted(healthy), dtype=np.int64))
    valid_bins = bins[valid]
    valid_prns = prns[valid]
    valid_estimates = estimates[valid]
    valid_distances = distances[valid]
    order = np.lexsort((valid_prns, valid_bins))
    valid_bins, valid_prns = valid_bins[order], valid_prns[order]
    valid_estimates, valid_distances = valid_estimates[order], valid_distances[order]
    keys = valid_bins * 64 + valid_prns
    boundaries = np.r_[0, np.flatnonzero(np.diff(keys)) + 1, len(keys)]
    delay_rows = []
    by_bin: dict[int, list[tuple[int, float]]] = {}
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        bin_index = int(valid_bins[start])
        prn = int(valid_prns[start])
        delay = float(np.median(valid_estimates[start:end]))
        delay_rows.append({
            "scenario": name, "bin_index": bin_index, "bin_start_s": float(bin_index),
            "prn": f"G{prn:02d}", "epoch_count": int(end - start),
            "estimated_delay_chips": delay,
            "median_template_distance": float(np.median(valid_distances[start:end])),
        })
        by_bin.setdefault(bin_index, []).append((prn, delay))

    geometry_rows = []
    for bin_index, entries in sorted(by_bin.items()):
        if len(entries) < int(analysis["minimum_prns"]):
            continue
        tow = (float(source["tow0_s"]) + float(bin_index) + 0.5) % 604800.0
        los = np.asarray([satellite_observation(receiver_ecef, healthy[prn], tow).los_ecef for prn, _ in entries], dtype=np.float64)
        delays = np.asarray([delay for _, delay in entries], dtype=np.float64)
        fit = fit_clock_centered_geometry(los, delays)
        geometry_rows.append({
            "scenario": name, "bin_index": bin_index, "bin_start_s": float(bin_index),
            "region": region(float(bin_index)), "prn_count": len(entries), "los_tow_s": tow,
            "clock_centered_geometry_residual": fit.clock_centered_normalized_residual,
            "directional_geometry_coherence": fit.directional_coherence,
            "legacy_zero_referenced_residual": fit.normalized_residual,
            "fit_rank": fit.rank, "clock_only_bias_chips": fit.clock_only_bias_chips,
        })
    metadata = {
        "scenario": name, "input_row_count": count,
        "healthy_tracked_prns": health["healthy_tracked_prns"],
        "preflight": preflight,
    }
    return delay_rows, geometry_rows, metadata


def evaluate_summaries(summaries: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    gates = {
        "required_scenario_count": len(summaries) == int(evaluation["required_scenario_count"]),
        "required_positive_change_scenario_count": sum(bool(row["positive_preregistered_direction"]) for row in summaries) >= int(evaluation["required_positive_change_scenario_count"]),
        "minimum_stable_pre_bins_per_scenario": min(int(row["stable_pre_bin_count"]) for row in summaries) >= int(evaluation["minimum_stable_pre_bins_per_scenario"]),
        "minimum_stable_post_bins_per_scenario": min(int(row["stable_post_bin_count"]) for row in summaries) >= int(evaluation["minimum_stable_post_bins_per_scenario"]),
        "minimum_prns_per_geometry_bin": min(int(row["minimum_prn_count"]) for row in summaries) >= int(evaluation["minimum_prns_per_geometry_bin"]),
    }
    passed = all(gates.values())
    changes = [float(row["pre_minus_post_residual_change"]) for row in summaries]
    return {
        "status": "DIRECTIONALLY_CONSISTENT" if passed else "NOT_DIRECTIONALLY_CONSISTENT",
        "all_preregistered_gates_passed": passed,
        "gates": gates,
        "scenario_changes": changes,
        "median_scenario_change": float(np.median(changes)),
    }


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0])
    extras = sorted({key for row in rows for key in row} - set(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields + extras, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _committed_release() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        _git("ls-files", "--error-unmatch", relative)
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=REPO_ROOT).returncode:
            raise ValueError(f"release input is not committed and clean: {relative}")
    candidate = "60280575737ff70618c5619ff7522c516bbdc67a"
    if subprocess.run(["git", "merge-base", "--is-ancestor", candidate, "HEAD"], cwd=REPO_ROOT).returncode:
        raise ValueError("frozen candidate commit is not an ancestor of HEAD")
    return {
        "head_commit": _git("rev-parse", "HEAD"), "candidate_commit": candidate,
        "input_commits": {relative: _git("log", "-1", "--format=%H", "--", relative) for relative in RELEASE_INPUTS},
        "runner_sha256": _sha256(Path(__file__).resolve()),
    }


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    context = validate_config(config)
    root = context["output_root"]
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    state_path = root / "release_state.json"
    state = {
        "schema": "gnss-doppler-lab.cgc-texbat-external-release-state", "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "phase": "released_before_outcome_access",
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": _sha256(PROTOCOL_PATH)},
        "commits": _committed_release(), "metrics_emitted": False,
    }
    _write_json(state_path, state)
    estimator = pilot._estimator(context["controlled"])
    all_delays, all_geometry, summaries, metadata = [], [], [], []
    for source in context["sources"]:
        print(f"[texbat] {source['name']}", flush=True)
        delays, geometry, scenario_metadata = evaluate_source(source, config["analysis"], estimator)
        all_delays.extend(delays)
        all_geometry.extend(geometry)
        summaries.append(summarize_scenario(source["name"], geometry))
        metadata.append(scenario_metadata)
    primary = evaluate_summaries(summaries, config["evaluation"])
    state["phase"] = "metrics_emitted"
    state["metrics_emitted"] = True
    state["metrics_emitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(state_path, state)
    delay_path, geometry_path, scenario_path = root / "delay_estimates.csv", root / "geometry_scores.csv", root / "scenario_summary.csv"
    _write_csv(delay_path, all_delays)
    _write_csv(geometry_path, all_geometry)
    _write_csv(scenario_path, summaries)
    result = {
        "schema": "gnss-doppler-lab.cgc-texbat-external-result", "schema_version": 1,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": _sha256(PROTOCOL_PATH)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve()), "commit": state["commits"]["input_commits"]["scripts/run_cgc_texbat_external.py"]},
        "release_state": {"path": str(state_path.resolve()), "sha256": _sha256(state_path)},
        "scenario_summaries": summaries, "primary_evaluation": primary, "scenario_metadata": metadata,
        "artifacts": {
            "delay_estimates": {"path": str(delay_path.resolve()), "sha256": _sha256(delay_path), "row_count": len(all_delays)},
            "geometry_scores": {"path": str(geometry_path.resolve()), "sha256": _sha256(geometry_path), "row_count": len(all_geometry)},
            "scenario_summary": {"path": str(scenario_path.resolve()), "sha256": _sha256(scenario_path), "row_count": len(summaries)},
        },
        "absolute_threshold_applied": False, "threshold_fitted": False,
        "post_release_tuning_or_retest": False, "claim_boundary": config["claim_boundary"],
    }
    summary_path = root / "summary.json"
    _write_json(summary_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--release-token", choices=[RELEASE_TOKEN])
    args = parser.parse_args(argv)
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    if args.validate_only:
        print("TEXBAT config and pinned inputs verified; no CGC outcome accessed")
        return 0
    run(DEFAULT_CONFIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
