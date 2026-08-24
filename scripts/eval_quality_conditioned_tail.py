#!/usr/bin/env python3
"""Run frozen global-vs-quality-conditioned tail campaigns from a manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnss_doppler_lab.quality_conditioned_tail import (  # noqa: E402
    GLOBAL_SCORE,
    QUALITY_SCORE,
    build_event_scores,
    calibrate_tail_detectors,
    evaluate_attack,
    evaluate_clean,
)

MANIFEST_SCHEMA = "gnss-doppler-lab.quality-conditioned-tail-campaign.v1"
SUMMARY_SCHEMA = "gnss-doppler-lab.quality-conditioned-tail-results.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_scores(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("at least one calibration score path is required")
    frames = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def _validate_manifest(document: dict[str, object]) -> None:
    if document.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    campaigns = document.get("campaigns")
    if not isinstance(campaigns, dict) or not campaigns:
        raise ValueError("manifest campaigns must be a nonempty object")
    for name, campaign in campaigns.items():
        if not isinstance(campaign, dict):
            raise ValueError(f"campaign {name} must be an object")
        if not isinstance(campaign.get("calibration_scores"), list) or not campaign["calibration_scores"]:
            raise ValueError(f"campaign {name} requires calibration_scores")
        for field in ("clean_evaluations", "attack_evaluations"):
            if not isinstance(campaign.get(field, {}), dict):
                raise ValueError(f"campaign {name} {field} must be an object")


def _detector_specs(calibration) -> dict[str, tuple[str, float]]:
    return {
        "global_btail": (GLOBAL_SCORE, calibration.global_event_threshold),
        "quality_conditioned_btail": (QUALITY_SCORE, calibration.quality_event_threshold),
    }


def _comparison_row(
    campaign: str,
    scenario: str,
    role: str,
    detector: str,
    metrics: dict[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {
        "campaign": campaign,
        "scenario": scenario,
        "role": role,
        "detector": detector,
        "threshold": metrics["threshold"],
    }
    if role == "clean":
        row.update({
            "pre_or_clean_fpr": metrics["false_positive_rate"],
            "post_tpr": None,
            "first_delay_s": None,
            "first_available_delay_s": None,
            "three_consecutive_delay_s": None,
        })
    else:
        row.update({
            "pre_or_clean_fpr": metrics["pre_false_positive_rate"],
            "post_tpr": metrics["post_detection_rate"],
            "first_delay_s": metrics["first_detection_delay_s"],
            "first_available_delay_s": metrics["first_detection_available_delay_s"],
            "three_consecutive_delay_s": metrics["first_three_consecutive_delay_s"],
        })
    return row


def run_campaign(
    name: str,
    campaign: dict[str, object],
    out_dir: Path,
    defaults: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    campaign_out = out_dir / name
    campaign_out.mkdir(parents=True, exist_ok=True)
    calibration_paths = [_resolve_path(str(value)) for value in campaign["calibration_scores"]]

    # Calibration is completed and frozen before any clean-control or attack
    # score CSV is opened below.
    calibration_scores = _read_scores(calibration_paths)
    age_cutoffs_s = tuple(campaign.get("age_cutoffs_s", defaults["age_cutoffs_s"]))
    max_gap_s = float(campaign.get("max_gap_s", defaults["max_gap_s"]))
    min_bin_rows = int(campaign.get("min_bin_rows", defaults["min_bin_rows"]))
    event_quantile = float(campaign.get("event_quantile", defaults["event_quantile"]))
    guard_s = float(campaign.get("guard_s", defaults["guard_s"]))
    availability_offset_s = float(
        campaign.get("availability_offset_s", defaults["availability_offset_s"])
    )
    calibration, calibration_events = calibrate_tail_detectors(
        calibration_scores,
        age_cutoffs_s=age_cutoffs_s,
        max_gap_s=max_gap_s,
        min_bin_rows=min_bin_rows,
        event_quantile=event_quantile,
    )
    calibration_events.to_csv(campaign_out / "calibration_event_scores.csv", index=False)
    calibration_record = asdict(calibration)
    calibration_record.update({
        "event_quantile": event_quantile,
        "normal_only": True,
        "attack_score_csv_opened_before_freeze": False,
        "input_scores": [_file_record(path) for path in calibration_paths],
    })
    (campaign_out / "calibration.json").write_text(
        json.dumps(calibration_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary: dict[str, object] = {
        "dataset": campaign.get("dataset", name),
        "protocol": campaign.get("protocol", ""),
        "caveat": campaign.get("caveat", ""),
        "calibration": calibration_record,
        "clean_evaluations": {},
        "attack_evaluations": {},
    }
    comparison: list[dict[str, object]] = []
    detector_specs = _detector_specs(calibration)

    for scenario, relative_path in campaign.get("clean_evaluations", {}).items():
        path = _resolve_path(str(relative_path))
        scores = _read_scores([path])
        events = build_event_scores(
            scores,
            global_node_thresholds=calibration.global_node_thresholds,
            quality_node_thresholds=calibration.quality_node_thresholds,
            age_cutoffs_s=calibration.age_cutoffs_s,
            max_gap_s=calibration.max_gap_s,
        )
        events.to_csv(campaign_out / f"{scenario}_event_scores.csv", index=False)
        scenario_result = {"input": _file_record(path), "detectors": {}}
        for detector, (score_column, threshold) in detector_specs.items():
            metrics = evaluate_clean(events, score_column, threshold)
            scenario_result["detectors"][detector] = metrics
            comparison.append(_comparison_row(name, scenario, "clean", detector, metrics))
        summary["clean_evaluations"][scenario] = scenario_result

    for scenario, specification in campaign.get("attack_evaluations", {}).items():
        if not isinstance(specification, dict) or "score_csv" not in specification or "onset_s" not in specification:
            raise ValueError(f"attack {name}/{scenario} requires score_csv and onset_s")
        path = _resolve_path(str(specification["score_csv"]))
        onset_s = float(specification["onset_s"])
        scores = _read_scores([path])
        events = build_event_scores(
            scores,
            global_node_thresholds=calibration.global_node_thresholds,
            quality_node_thresholds=calibration.quality_node_thresholds,
            age_cutoffs_s=calibration.age_cutoffs_s,
            max_gap_s=calibration.max_gap_s,
        )
        events.to_csv(campaign_out / f"{scenario}_event_scores.csv", index=False)
        scenario_result = {
            "input": _file_record(path),
            "onset_s": onset_s,
            "detectors": {},
        }
        for detector, (score_column, threshold) in detector_specs.items():
            metrics = evaluate_attack(
                events,
                score_column,
                threshold,
                onset_s,
                guard_s=guard_s,
                availability_offset_s=availability_offset_s,
            )
            scenario_result["detectors"][detector] = metrics
            comparison.append(_comparison_row(name, scenario, "attack", detector, metrics))
        summary["attack_evaluations"][scenario] = scenario_result

    (campaign_out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary, comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    manifest_path = _resolve_path(args.manifest)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(document)
    defaults: dict[str, object] = {
        "age_cutoffs_s": document.get("age_cutoffs_s", [5.0, 20.0]),
        "max_gap_s": document.get("max_gap_s", 0.75),
        "min_bin_rows": document.get("min_bin_rows", 100),
        "event_quantile": document.get("event_quantile", 0.99),
        "guard_s": document.get("guard_s", 10.0),
        "availability_offset_s": document.get("availability_offset_s", 1.0),
    }
    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: dict[str, object] = {}
    comparison: list[dict[str, object]] = []
    for name, campaign in document["campaigns"].items():
        summary, rows = run_campaign(name, campaign, out_dir, defaults)
        all_summaries[name] = summary
        comparison.extend(rows)

    comparison_frame = pd.DataFrame(comparison)
    comparison_frame.to_csv(out_dir / "comparison.csv", index=False)
    aggregate = {
        "schema": SUMMARY_SCHEMA,
        "manifest": _file_record(manifest_path),
        "calibration_policy": "normal score CSVs only; frozen before evaluation score CSVs are opened",
        "quality_state": "causal observed-score continuity age; resets after timestamp gaps",
        "campaigns": all_summaries,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(comparison_frame.to_string(index=False))


if __name__ == "__main__":
    main()
