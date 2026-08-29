#!/usr/bin/env python3
"""Calibrate and evaluate a causal run-normalized OAKBAT b-tail gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import eval_btail_support_gate as gate

CALIBRATION_SCHEMA = "gnss-doppler-lab.oakbat-run-normalized-btail-calibration.v1"
EVALUATION_SCHEMA = "gnss-doppler-lab.oakbat-run-normalized-btail-evaluation.v1"
CHECKPOINT_SHA256 = "f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b"
CLEAN_SCENARIOS = ("cleanStatic", "cleanDynamic")
DEFAULT_ATTACK_SCENARIOS = ("os2", "os3", "os4")
SCORE_TIME_AVAILABILITY_OFFSET_S = 1.0
MAD_TO_SIGMA = 1.4826


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_path(score_root: Path, scenario: str, prefix: str) -> Path:
    return score_root / scenario / f"{prefix}_{scenario}_prn_local_scores.csv"


def score_summary_path(score_root: Path, scenario: str, prefix: str) -> Path:
    return score_root / scenario / f"{prefix}_{scenario}_prn_local_onset_summary.json"


def validate_checkpoint(score_root: Path, scenario: str, prefix: str) -> dict[str, object]:
    path = score_summary_path(score_root, scenario, prefix)
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid score summary for {scenario}: {exc}") from exc
    actual = document.get("checkpoint_provenance", {}).get("checkpoint_sha256")
    if actual != CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint mismatch for {scenario}: {actual}")
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def load_scores(score_root: Path, scenario: str, prefix: str) -> tuple[pd.DataFrame, Path]:
    path = score_path(score_root, scenario, prefix)
    if not path.is_file():
        raise FileNotFoundError(path)
    return gate._validated_prn_scores(pd.read_csv(path)), path


def normalize_run_scores(
    prn_scores: pd.DataFrame,
    *,
    warmup_end_s: float,
    minimum_baseline_rows: int,
    scale_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one robust baseline per run/PRN on the trusted startup interval."""
    if warmup_end_s <= 0 or minimum_baseline_rows < 2 or scale_floor <= 0:
        raise ValueError("invalid run-normalization contract")
    frame = gate._validated_prn_scores(prn_scores)
    normalized_parts: list[pd.DataFrame] = []
    baseline_rows: list[dict[str, object]] = []
    for (run_id, prn), group in frame.groupby(["run_id", "prn"], sort=True):
        baseline = group[group["window_start_s"] < warmup_end_s]
        eligible = len(baseline) >= minimum_baseline_rows
        row: dict[str, object] = {
            "run_id": str(run_id),
            "prn": str(prn),
            "baseline_rows": int(len(baseline)),
            "eligible": bool(eligible),
            "warmup_end_s": float(warmup_end_s),
        }
        if not eligible:
            row.update({"center": None, "raw_mad_scale": None, "applied_scale": None})
            baseline_rows.append(row)
            continue
        center = float(baseline["prn_node_rmse"].median())
        raw_scale = MAD_TO_SIGMA * float(
            np.median(np.abs(baseline["prn_node_rmse"].to_numpy(float) - center))
        )
        applied_scale = max(float(scale_floor), raw_scale)
        target = group[group["window_start_s"] >= warmup_end_s].copy()
        if target.empty:
            row.update({
                "center": center,
                "raw_mad_scale": raw_scale,
                "applied_scale": applied_scale,
            })
            baseline_rows.append(row)
            continue
        target["prn_node_rmse_raw"] = target["prn_node_rmse"]
        target["prn_node_rmse"] = (
            target["prn_node_rmse_raw"] - center
        ) / applied_scale
        target["run_normalization_center"] = center
        target["run_normalization_scale"] = applied_scale
        normalized_parts.append(target)
        row.update({
            "center": center,
            "raw_mad_scale": raw_scale,
            "applied_scale": applied_scale,
        })
        baseline_rows.append(row)
    if not normalized_parts:
        raise ValueError("run normalization produced no eligible scores")
    normalized = pd.concat(normalized_parts, ignore_index=True)
    if (normalized["window_start_s"] < warmup_end_s).any():
        raise AssertionError("warm-up rows leaked into detector output")
    numeric = normalized.select_dtypes(include=[np.number]).to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("run normalization produced non-finite values")
    return normalized, pd.DataFrame(baseline_rows)


def supported_events(
    normalized_scores: pd.DataFrame,
    node_thresholds: dict[str, float],
    *,
    min_event_prns: int,
    ewma_previous_weight: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if min_event_prns < 1:
        raise ValueError("min_event_prns must be positive")
    events = gate.build_event_scores(
        normalized_scores, node_thresholds, alpha=ewma_previous_weight
    )
    eligible = events["tracked_prn_count"] >= min_event_prns
    support = {
        "all_event_windows": int(len(events)),
        "eligible_event_windows": int(eligible.sum()),
        "ineligible_event_windows": int((~eligible).sum()),
        "minimum_observed_prns": int(events["tracked_prn_count"].min()),
        "minimum_required_prns": int(min_event_prns),
    }
    return events.loc[eligible].reset_index(drop=True), support


def write_frame(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "rows": int(len(frame)),
    }


def calibrate(
    *,
    score_root: Path,
    out_dir: Path,
    score_prefix: str,
    warmup_end_s: float,
    minimum_baseline_rows: int,
    scale_floor: float,
    min_event_prns: int,
    event_quantile: float,
    ewma_previous_weight: float,
) -> Path:
    if not 0 < event_quantile < 1:
        raise ValueError("event_quantile must be between zero and one")
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized: dict[str, pd.DataFrame] = {}
    events: dict[str, pd.DataFrame] = {}
    baselines: dict[str, pd.DataFrame] = {}
    source_contract: dict[str, object] = {}
    artifacts: dict[str, object] = {}

    for scenario in CLEAN_SCENARIOS:
        validate_checkpoint(score_root, scenario, score_prefix)
        scores, path = load_scores(score_root, scenario, score_prefix)
        normalized[scenario], baselines[scenario] = normalize_run_scores(
            scores,
            warmup_end_s=warmup_end_s,
            minimum_baseline_rows=minimum_baseline_rows,
            scale_floor=scale_floor,
        )
        source_contract[scenario] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
        }

    combined_nodes = pd.concat(
        [normalized[scenario] for scenario in CLEAN_SCENARIOS], ignore_index=True
    )
    node_thresholds = {
        name: float(combined_nodes["prn_node_rmse"].quantile(quantile))
        for name, quantile in gate.NODE_QUANTILES.items()
    }

    support: dict[str, object] = {}
    for scenario in CLEAN_SCENARIOS:
        events[scenario], support[scenario] = supported_events(
            normalized[scenario],
            node_thresholds,
            min_event_prns=min_event_prns,
            ewma_previous_weight=ewma_previous_weight,
        )
        artifacts[f"{scenario}_normalized_scores"] = write_frame(
            normalized[scenario], out_dir / f"{scenario}_normalized_prn_scores.csv"
        )
        artifacts[f"{scenario}_baselines"] = write_frame(
            baselines[scenario], out_dir / f"{scenario}_run_baselines.csv"
        )
        artifacts[f"{scenario}_events"] = write_frame(
            events[scenario], out_dir / f"{scenario}_event_scores.csv"
        )

    combined_events = pd.concat(
        [events[scenario] for scenario in CLEAN_SCENARIOS], ignore_index=True
    )
    event_threshold = float(
        combined_events[gate.FINAL_SCORE].quantile(event_quantile)
    )
    clean_metrics = {
        scenario: gate.evaluate_clean_scenario(events[scenario], event_threshold)
        for scenario in CLEAN_SCENARIOS
    }
    calibration = {
        "schema": CALIBRATION_SCHEMA,
        "status": "exploratory_post_hoc_protocol",
        "scope": "calibration reads OAKBAT cleanStatic and cleanDynamic only",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "score_prefix": score_prefix,
        "normalizer": {
            "kind": "per-run-per-prn trusted-startup median/MAD",
            "warmup_end_s": float(warmup_end_s),
            "minimum_baseline_rows": int(minimum_baseline_rows),
            "mad_to_sigma": MAD_TO_SIGMA,
            "scale_floor": float(scale_floor),
            "output_direction": "(rmse - startup_median) / max(1.4826*MAD, scale_floor)",
            "causality": "no scores are emitted before warmup_end_s",
        },
        "gate": {
            "node_quantiles": gate.NODE_QUANTILES,
            "node_thresholds": node_thresholds,
            "event_quantile": float(event_quantile),
            "event_threshold": event_threshold,
            "event_score": gate.FINAL_SCORE,
            "ewma_previous_weight": float(ewma_previous_weight),
            "min_event_prns": int(min_event_prns),
        },
        "clean_score_sources": source_contract,
        "clean_support": support,
        "clean_metrics": clean_metrics,
        "artifacts": artifacts,
    }
    path = out_dir / "calibration.json"
    path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")
    print(json.dumps(calibration, indent=2, sort_keys=True))
    return path


def _load_calibration(path: Path) -> dict[str, object]:
    try:
        calibration = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid calibration JSON: {exc}") from exc
    if calibration.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("wrong run-normalized calibration schema")
    if set(calibration.get("clean_score_sources", {})) != set(CLEAN_SCENARIOS):
        raise ValueError("calibration must be sourced from exactly the two clean controls")
    if calibration.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("calibration checkpoint mismatch")
    return calibration


def _add_availability_metrics(metrics: dict[str, object]) -> None:
    detection = metrics.get("first_detection_s")
    delay = metrics.get("first_delay_s")
    metrics["score_availability_offset_s"] = SCORE_TIME_AVAILABILITY_OFFSET_S
    metrics["first_operational_availability_s"] = (
        None if detection is None else float(detection) + SCORE_TIME_AVAILABILITY_OFFSET_S
    )
    metrics["first_operational_delay_s"] = (
        None if delay is None else float(delay) + SCORE_TIME_AVAILABILITY_OFFSET_S
    )




def wilson_interval(successes: int, trials: int) -> list[float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial count")
    z = 1.959963984540054
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(
            probability * (1.0 - probability) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return [float(max(0.0, center - half_width)), float(min(1.0, center + half_width))]


def add_binomial_intervals(metrics: dict[str, object]) -> None:
    metrics["pre_false_positive_wilson95"] = wilson_interval(
        int(metrics["pre_false_flags"]), int(metrics["pre_windows"])
    )
    metrics["post_detection_wilson95"] = wilson_interval(
        int(metrics["post_detection_flags"]), int(metrics["post_windows"])
    )

def physical_contrast(
    normalized_scores: pd.DataFrame,
    events: pd.DataFrame,
    *,
    onset_s: float,
    onset_buffer_s: float,
    q80_threshold: float,
) -> dict[str, object]:
    """Summarize whether the anomaly is sustained across many PRNs."""
    pre_events = events[
        (events["window_start_s"] >= normalized_scores["window_start_s"].min())
        & (events["window_start_s"] < onset_s - onset_buffer_s)
    ]
    post_events = events[events["window_start_s"] >= onset_s + onset_buffer_s]
    per_prn: list[dict[str, object]] = []
    for prn, group in normalized_scores.groupby("prn", sort=True):
        pre = group[
            (group["window_start_s"] >= normalized_scores["window_start_s"].min())
            & (group["window_start_s"] < onset_s - onset_buffer_s)
        ]["prn_node_rmse"]
        post = group[group["window_start_s"] >= onset_s + onset_buffer_s][
            "prn_node_rmse"
        ]
        if pre.empty or post.empty:
            continue
        pre_median = float(pre.median())
        post_median = float(post.median())
        per_prn.append({
            "prn": str(prn),
            "pre_median_z": pre_median,
            "post_median_z": post_median,
            "median_shift_z": post_median - pre_median,
            "post_q80_exceedance_rate": float((post > q80_threshold).mean()),
            "sustained_post_median_above_q80": bool(post_median > q80_threshold),
        })

    def event_phase(frame: pd.DataFrame) -> dict[str, float | int]:
        if frame.empty:
            raise ValueError("physical contrast requires non-empty pre/post event phases")
        return {
            "windows": int(len(frame)),
            "median_tracked_prns": float(frame["tracked_prn_count"].median()),
            "median_q80_exceeding_prns": float(frame["k_q80"].median()),
            "median_gate_score": float(frame[gate.FINAL_SCORE].median()),
        }

    return {
        "interpretation": (
            "sustained_post_median_above_q80 counts PRNs whose median normalized "
            "post-onset score exceeds the clean-derived q80 node threshold"
        ),
        "q80_node_threshold": float(q80_threshold),
        "pre_event_phase": event_phase(pre_events),
        "post_event_phase": event_phase(post_events),
        "comparable_prns": int(len(per_prn)),
        "sustained_post_q80_prns": int(
            sum(row["sustained_post_median_above_q80"] for row in per_prn)
        ),
        "per_prn": per_prn,
    }

def evaluate(
    *,
    score_root: Path,
    out_dir: Path,
    calibration_json: Path,
    scenarios: tuple[str, ...],
    onsets: dict[str, float],
    onset_buffer_s: float,
) -> Path:
    calibration = _load_calibration(calibration_json)
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("attack scenarios must be non-empty and unique")
    if set(scenarios) & set(CLEAN_SCENARIOS):
        raise ValueError("evaluate stage forbids clean calibration scenarios")
    if set(scenarios) != set(onsets):
        raise ValueError("onsets must match evaluation scenarios exactly")

    normalizer = calibration["normalizer"]
    gate_contract = calibration["gate"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    sources: dict[str, object] = {}
    artifacts: dict[str, object] = {}

    for scenario in scenarios:
        score_summary = validate_checkpoint(score_root, scenario, calibration["score_prefix"])
        scores, path = load_scores(score_root, scenario, calibration["score_prefix"])
        normalized, baselines = normalize_run_scores(
            scores,
            warmup_end_s=float(normalizer["warmup_end_s"]),
            minimum_baseline_rows=int(normalizer["minimum_baseline_rows"]),
            scale_floor=float(normalizer["scale_floor"]),
        )
        events, support = supported_events(
            normalized,
            {key: float(value) for key, value in gate_contract["node_thresholds"].items()},
            min_event_prns=int(gate_contract["min_event_prns"]),
            ewma_previous_weight=float(gate_contract["ewma_previous_weight"]),
        )
        metrics = gate.evaluate_scenario(
            events,
            float(gate_contract["event_threshold"]),
            float(onsets[scenario]),
            onset_buffer_s=float(onset_buffer_s),
        )
        _add_availability_metrics(metrics)
        add_binomial_intervals(metrics)
        contrast = physical_contrast(normalized, events, onset_s=float(onsets[scenario]), onset_buffer_s=float(onset_buffer_s), q80_threshold=float(gate_contract["node_thresholds"]["q80"]))
        results[scenario] = {"support": support, "metrics": metrics, "physical_contrast": contrast}
        sources[scenario] = {
            "score_csv": {"path": str(path.resolve()), "sha256": sha256(path)},
            "score_summary": score_summary,
        }
        artifacts[f"{scenario}_normalized_scores"] = write_frame(
            normalized, out_dir / f"{scenario}_normalized_prn_scores.csv"
        )
        artifacts[f"{scenario}_baselines"] = write_frame(
            baselines, out_dir / f"{scenario}_run_baselines.csv"
        )
        artifacts[f"{scenario}_events"] = write_frame(
            events, out_dir / f"{scenario}_event_scores.csv"
        )

    aggregate_pre_flags = sum(int(results[s]["metrics"]["pre_false_flags"]) for s in scenarios)
    aggregate_pre_windows = sum(int(results[s]["metrics"]["pre_windows"]) for s in scenarios)
    aggregate_pre = {
        "false_flags": aggregate_pre_flags,
        "windows": aggregate_pre_windows,
        "false_positive_rate": aggregate_pre_flags / aggregate_pre_windows,
        "wilson95": wilson_interval(aggregate_pre_flags, aggregate_pre_windows),
    }

    summary = {
        "schema": EVALUATION_SCHEMA,
        "status": "exploratory_post_hoc_external_dataset_audit",
        "claim_boundary": (
            "model weights and clean-derived gate are frozen before attack evaluation; "
            "each run still requires the declared trusted startup baseline"
        ),
        "calibration": {
            "path": str(calibration_json.resolve()),
            "sha256": sha256(calibration_json),
        },
        "onset_buffer_s": float(onset_buffer_s),
        "score_time_availability_offset_s": SCORE_TIME_AVAILABILITY_OFFSET_S,
        "attack_file_preattack_negative_control": aggregate_pre,
        "sources": sources,
        "scenarios": results,
        "artifacts": artifacts,
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("--score-root", required=True)
    calibration.add_argument("--out-dir", required=True)
    calibration.add_argument("--score-prefix", default="oakbat")
    calibration.add_argument("--warmup-end-s", type=float, default=60.0)
    calibration.add_argument("--minimum-baseline-rows", type=int, default=40)
    calibration.add_argument("--scale-floor", type=float, default=0.001)
    calibration.add_argument("--min-event-prns", type=int, default=8)
    calibration.add_argument("--event-quantile", type=float, default=0.99)
    calibration.add_argument("--ewma-previous-weight", type=float, choices=[0.75], default=0.75)

    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--score-root", required=True)
    evaluation.add_argument("--out-dir", required=True)
    evaluation.add_argument("--calibration-json", required=True)
    evaluation.add_argument(
        "--scenarios", nargs="+", default=list(DEFAULT_ATTACK_SCENARIOS)
    )
    evaluation.add_argument(
        "--onsets-json", default='{"os2":120.0,"os3":120.0,"os4":120.0}'
    )
    evaluation.add_argument("--onset-buffer-s", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "calibrate":
        calibrate(
            score_root=Path(args.score_root),
            out_dir=Path(args.out_dir),
            score_prefix=args.score_prefix,
            warmup_end_s=args.warmup_end_s,
            minimum_baseline_rows=args.minimum_baseline_rows,
            scale_floor=args.scale_floor,
            min_event_prns=args.min_event_prns,
            event_quantile=args.event_quantile,
            ewma_previous_weight=args.ewma_previous_weight,
        )
    else:
        evaluate(
            score_root=Path(args.score_root),
            out_dir=Path(args.out_dir),
            calibration_json=Path(args.calibration_json),
            scenarios=tuple(args.scenarios),
            onsets={key: float(value) for key, value in json.loads(args.onsets_json).items()},
            onset_buffer_s=args.onset_buffer_s,
        )


if __name__ == "__main__":
    main()
