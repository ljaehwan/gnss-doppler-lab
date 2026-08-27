#!/usr/bin/env python3
"""Run the frozen normal-calibrated sequential CGC test on TEXBAT DS8."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_RUNNER_PATH = REPO_ROOT / "scripts/run_cgc_real_detection.py"
SPEC = importlib.util.spec_from_file_location("run_cgc_real_detection", REAL_RUNNER_PATH)
REAL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REAL)

DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/cgc_sequential_ds8_v1.json"
PROTOCOL_PATH = REPO_ROOT / "docs/results/cgc_sequential_ds8_protocol_v1.md"
RELEASE_TOKEN = "RELEASE-CGC-SEQUENTIAL-DS8-V1"
RELEASE_INPUTS = (
    "configs/experiments/cgc_sequential_ds8_v1.json",
    "docs/results/cgc_sequential_ds8_protocol_v1.md",
    "scripts/run_cgc_sequential_ds8.py",
)


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


def page_cusum(
    residuals: np.ndarray,
    bins: np.ndarray,
    *,
    location: float,
    scale: float,
    clip: float,
    reference: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal lower-shift Page CUSUM with gap resets."""
    values = np.asarray(residuals, dtype=np.float64)
    indices = np.asarray(bins, dtype=np.int64)
    if values.ndim != 1 or indices.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("residuals and bins must be matching finite vectors")
    if not np.isfinite(location) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid normal location or scale")
    if clip <= 0 or reference < 0:
        raise ValueError("invalid Page-CUSUM constants")
    evidence = np.clip((location - values) / scale, -clip, clip)
    statistic = np.zeros(len(values), dtype=np.float64)
    previous_bin: int | None = None
    running = 0.0
    for index, (bin_index, increment) in enumerate(zip(indices, evidence)):
        if previous_bin is None or int(bin_index) != previous_bin + 1:
            running = 0.0
        running = max(0.0, running + float(increment) - reference)
        statistic[index] = running
        previous_bin = int(bin_index)
    return evidence, statistic


def robust_normal_parameters(residuals: np.ndarray) -> tuple[float, float]:
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("normal residuals must be a nonempty finite vector")
    location = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - location)))
    if scale <= 0:
        raise ValueError("normal MAD scale is zero")
    return location, scale


def read_normal_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            rows.append({
                "scenario": source["scenario"],
                "region": source["region"],
                "bin_index": int(source["bin_index"]),
                "bin_start_s": float(source["bin_start_s"]),
                "bin_end_s": float(source["bin_end_s"]),
                "clock_centered_geometry_residual": float(
                    source["clock_centered_geometry_residual"]
                ),
            })
    if not rows:
        raise ValueError("normal score CSV is empty")
    return rows


def select_rows(rows: list[dict[str, Any]], selector: dict[str, str]) -> list[dict[str, Any]]:
    return sorted(
        (
            row for row in rows
            if row["scenario"] == selector["scenario"] and row["region"] == selector["region"]
        ),
        key=lambda row: int(row["bin_index"]),
    )


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "gnss-doppler-lab.cgc-sequential-ds8-config" or config.get("schema_version") != 1:
        raise ValueError("unsupported sequential DS8 config")
    experiment = config.get("experiment", {})
    if experiment.get("name") != "cgc-sequential-ds8-v1":
        raise ValueError("experiment identity drifted")
    if experiment.get("official_onset_seconds") != 110.0:
        raise ValueError("DS8 onset drifted")
    if experiment.get("primary_attack_cgc_outcome_accessed_before_freeze") is not False:
        raise ValueError("DS8 CGC outcome must be unseen before freeze")
    frozen = config.get("frozen_candidate", {})
    if frozen.get("residual") != "SSE_LOS_plus_clock / SSE_clock_only":
        raise ValueError("physical residual drifted")
    if frozen.get("spoof_score") != "negative clock-centered residual":
        raise ValueError("spoof score drifted")
    frozen_paths = {
        key: verify_record(frozen[key], key)
        for key in (
            "real_detection_runner", "clock_centered_module",
            "correlator_geometry_module", "geometry_module", "template_config",
        )
    }
    analysis = config.get("analysis", {})
    expected_analysis = {
        "bin_seconds": 1.0,
        "minimum_prns": 8,
        "epoch_chunk_size": 50000,
        "monitor_start_seconds": 30.0,
        "official_onset_seconds": 110.0,
        "stable_post_start_seconds": 120.0,
        "normal_location": "median",
        "normal_scale": "1.4826_times_MAD",
        "standardized_evidence": "clip((normal_median - residual) / normal_scale, -3, 3)",
        "page_cusum_reference_value": 0.5,
        "page_cusum_decision_threshold": 5.0,
        "page_cusum_recursion": "S_t=max(0,S_(t-1)+evidence_t-0.5)",
        "reset_on_nonconsecutive_bin": True,
        "score_available_at": "bin_end",
    }
    if any(analysis.get(key) != value for key, value in expected_analysis.items()):
        raise ValueError("analysis contract drifted")
    expected_gates = {
        "minimum_baseline_bins": 80,
        "minimum_locked_normal_bins": 25,
        "maximum_baseline_alarm_count": 0,
        "maximum_locked_normal_alarm_count": 0,
        "minimum_primary_pre_bins": 70,
        "minimum_primary_post_bins": 300,
        "maximum_primary_pre_alarm_count": 0,
        "primary_attack_must_have_post_onset_detection": True,
        "maximum_primary_detection_delay_seconds": 60.0,
    }
    gates = config.get("primary_gates", {})
    if any(gates.get(key) != value for key, value in expected_gates.items()):
        raise ValueError("primary gates drifted")

    normal_path = verify_record(config["normal_scores"], "normal score artifact")
    source = config.get("primary_source", {})
    if (source.get("name"), source.get("role"), source.get("tow0_s")) != ("ds8", "primary_attack", 477900.0):
        raise ValueError("primary source contract drifted")
    source_paths = {
        key: verify_record(source[key], f"DS8 {key}")
        for key in ("complex_epoch_npz", "export_manifest", "ephemeris", "nmea")
    }
    manifest = json.loads(source_paths["export_manifest"].read_text(encoding="utf-8"))
    if manifest.get("feature_schema", {}).get("tensor") != "complex_iq":
        raise ValueError("DS8 is not complex-IQ input")
    output = manifest.get("output", {})
    if output.get("shape", [None, None])[1:] != [9, 2] or output.get("sha256") != source["complex_epoch_npz"]["sha256"]:
        raise ValueError("DS8 nine-tap manifest drifted")
    output_root = repo_path(config["output_root"])
    if output_root != REPO_ROOT / "artifacts/cgc_sequential_ds8_v1":
        raise ValueError("output root drifted")
    template = json.loads(frozen_paths["template_config"].read_text(encoding="utf-8"))
    return {
        "normal_path": normal_path,
        "source": {**source, "paths": source_paths, "manifest": manifest},
        "template": template,
        "output_root": output_root,
    }


def apply_trace(
    rows: list[dict[str, Any]],
    *,
    location: float,
    scale: float,
    analysis: dict[str, Any],
) -> None:
    ordered = sorted(rows, key=lambda row: int(row["bin_index"]))
    evidence, statistic = page_cusum(
        np.asarray([row["clock_centered_geometry_residual"] for row in ordered]),
        np.asarray([row["bin_index"] for row in ordered]),
        location=location,
        scale=scale,
        clip=3.0,
        reference=float(analysis["page_cusum_reference_value"]),
    )
    threshold = float(analysis["page_cusum_decision_threshold"])
    for row, increment, value in zip(ordered, evidence, statistic):
        row["sequential_evidence"] = float(increment)
        row["page_cusum"] = float(value)
        row["sequential_alarm_threshold"] = threshold
        row["sequential_spoof_alarm"] = bool(value >= threshold)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"bin_count": 0, "alarm_count": 0, "alarm_rate": None, "median_residual": None}
    alarms = np.asarray([row["sequential_spoof_alarm"] for row in rows], dtype=bool)
    return {
        "bin_count": len(rows),
        "alarm_count": int(alarms.sum()),
        "alarm_rate": float(alarms.mean()),
        "median_residual": float(np.median([
            row["clock_centered_geometry_residual"] for row in rows
        ])),
        "maximum_page_cusum": float(max(row["page_cusum"] for row in rows)),
        "minimum_prn_count": (
            min(int(row["prn_count"]) for row in rows) if "prn_count" in rows[0] else None
        ),
    }


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
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


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def committed_release() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        git("ls-files", "--error-unmatch", relative)
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=REPO_ROOT).returncode:
            raise ValueError(f"release input is not committed and clean: {relative}")
    return {
        "head_commit": git("rev-parse", "HEAD"),
        "input_commits": {
            relative: git("log", "-1", "--format=%H", "--", relative)
            for relative in RELEASE_INPUTS
        },
        "runner_sha256": sha256(Path(__file__).resolve()),
    }


def build_evaluation(
    baseline: list[dict[str, Any]],
    locked: list[dict[str, Any]],
    primary: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    analysis = config["analysis"]
    gates_config = config["primary_gates"]
    monitor_start = float(analysis["monitor_start_seconds"])
    onset = float(analysis["official_onset_seconds"])
    stable_post_start = float(analysis["stable_post_start_seconds"])
    monitored = [row for row in primary if float(row["bin_end_s"]) > monitor_start]
    pre = [row for row in monitored if float(row["bin_end_s"]) < onset]
    post = [row for row in monitored if float(row["bin_end_s"]) >= onset]
    stable_post = [row for row in monitored if float(row["bin_end_s"]) >= stable_post_start]
    post_alarms = [row for row in post if row["sequential_spoof_alarm"]]
    first = float(post_alarms[0]["bin_end_s"]) if post_alarms else None
    delay = first - onset if first is not None else None
    auc = None
    if pre and stable_post:
        labels = np.r_[np.zeros(len(pre), dtype=int), np.ones(len(stable_post), dtype=int)]
        scores = np.asarray([
            -float(row["clock_centered_geometry_residual"])
            for row in pre + stable_post
        ])
        auc = float(roc_auc_score(labels, scores))
    baseline_summary = summarize(baseline)
    locked_summary = summarize(locked)
    pre_summary = summarize(pre)
    post_summary = summarize(post)
    gates = {
        "minimum_baseline_bins": len(baseline) >= int(gates_config["minimum_baseline_bins"]),
        "minimum_locked_normal_bins": len(locked) >= int(gates_config["minimum_locked_normal_bins"]),
        "maximum_baseline_alarm_count": baseline_summary["alarm_count"] <= int(gates_config["maximum_baseline_alarm_count"]),
        "maximum_locked_normal_alarm_count": locked_summary["alarm_count"] <= int(gates_config["maximum_locked_normal_alarm_count"]),
        "minimum_primary_pre_bins": len(pre) >= int(gates_config["minimum_primary_pre_bins"]),
        "minimum_primary_post_bins": len(post) >= int(gates_config["minimum_primary_post_bins"]),
        "maximum_primary_pre_alarm_count": pre_summary["alarm_count"] <= int(gates_config["maximum_primary_pre_alarm_count"]),
        "primary_attack_must_have_post_onset_detection": first is not None,
        "maximum_primary_detection_delay_seconds": delay is not None and delay <= float(gates_config["maximum_primary_detection_delay_seconds"]),
    }
    passed = all(gates.values())
    return {
        "status": "SEQUENTIAL_DS8_SUPPORTED" if passed else "SEQUENTIAL_DS8_NOT_SUPPORTED",
        "all_primary_gates_passed": passed,
        "primary_gates": gates,
        "baseline": baseline_summary,
        "locked_normal": locked_summary,
        "primary_pre_onset": pre_summary,
        "primary_post_onset": post_summary,
        "primary_stable_post": summarize(stable_post),
        "first_post_onset_alarm_bin_end_s": first,
        "detection_delay_from_official_onset_s": delay,
        "descriptive_pre_vs_stable_post_auc": auc,
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
        "schema": "gnss-doppler-lab.cgc-sequential-ds8-release-state",
        "schema_version": 1,
        "phase": "released_before_ds8_cgc_outcome_access",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)},
        "commits": committed_release(),
        "metrics_emitted": False,
    }
    write_json(state_path, state)

    normal_rows = read_normal_rows(context["normal_path"])
    baseline = select_rows(normal_rows, config["normal_scores"]["baseline_selector"])
    locked = select_rows(normal_rows, config["normal_scores"]["locked_normal_selector"])
    location, scale = robust_normal_parameters(np.asarray([
        row["clock_centered_geometry_residual"] for row in baseline
    ]))
    apply_trace(baseline, location=location, scale=scale, analysis=config["analysis"])
    apply_trace(locked, location=location, scale=scale, analysis=config["analysis"])

    estimator = REAL.pilot._estimator(context["template"])
    delays, geometry, metadata = REAL.evaluate_source(
        context["source"], config["analysis"], estimator
    )
    monitor_start = float(config["analysis"]["monitor_start_seconds"])
    primary = sorted(
        (row for row in geometry if float(row["bin_end_s"]) > monitor_start),
        key=lambda row: int(row["bin_index"]),
    )
    apply_trace(primary, location=location, scale=scale, analysis=config["analysis"])
    evaluation = build_evaluation(baseline, locked, primary, config)

    state["phase"] = "metrics_emitted"
    state["metrics_emitted"] = True
    state["metrics_emitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, state)
    delay_path = root / "ds8_delay_estimates.csv"
    score_path = root / "ds8_sequential_scores.csv"
    write_csv(delay_path, delays)
    write_csv(score_path, primary)
    summary = {
        "schema": "gnss-doppler-lab.cgc-sequential-ds8-result",
        "schema_version": 1,
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "release_state": {"path": str(state_path.resolve()), "sha256": sha256(state_path)},
        "normal_parameters": {"median": location, "mad_scale": scale},
        "primary_evaluation": evaluation,
        "primary_source_metadata": metadata,
        "artifacts": {
            "delay_estimates": {"path": str(delay_path.resolve()), "sha256": sha256(delay_path), "row_count": len(delays)},
            "sequential_scores": {"path": str(score_path.resolve()), "sha256": sha256(score_path), "row_count": len(primary)},
        },
        "claim_boundary": config["claim_boundary"],
        "post_release_tuning_or_retest": False,
    }
    summary_path = root / "summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), "primary": evaluation}, indent=2, sort_keys=True))
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
        print("sequential DS8 config and pinned inputs verified; no DS8 CGC outcome accessed")
        return 0
    run(DEFAULT_CONFIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
