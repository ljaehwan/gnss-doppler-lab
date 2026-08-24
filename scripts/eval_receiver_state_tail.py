#!/usr/bin/env python3
"""Evaluate clean-only receiver-state conditional conformal calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnss_doppler_lab.receiver_state_tail import (  # noqa: E402
    GLOBAL_SCORE,
    STATE_SCORE,
    build_conformal_event_scores,
    calibrate_receiver_state_detectors,
    chronological_clean_split,
    evaluate_attack,
    evaluate_clean,
    reference_inventory,
)

MANIFEST_SCHEMA = "gnss-doppler-lab.receiver-state-tail-campaign.v1"
SUMMARY_SCHEMA = "gnss-doppler-lab.receiver-state-tail-results.v1"


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
        raise ValueError("at least one score path is required")
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def _validate_manifest(document: dict[str, object]) -> None:
    if document.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    clean_scores = document.get("clean_scores")
    if not isinstance(clean_scores, list) or not clean_scores:
        raise ValueError("manifest clean_scores must be a nonempty array")
    attacks = document.get("attacks")
    if not isinstance(attacks, dict) or not attacks:
        raise ValueError("manifest attacks must be a nonempty object")
    for name, attack in attacks.items():
        if not isinstance(attack, dict) or "score_csv" not in attack or "onset_s" not in attack:
            raise ValueError(f"attack {name} requires score_csv and onset_s")


def _pool_usage(nodes: pd.DataFrame) -> dict[str, object]:
    levels = nodes["calibration_pool_level"].value_counts().sort_index()
    origins = nodes.groupby("receiver_origin")["calibration_pool_level"].value_counts()
    return {
        "rows": int(len(nodes)),
        "by_pool_level": {str(key): int(value) for key, value in levels.items()},
        "by_origin_and_pool_level": {
            f"{origin}|{level}": int(value)
            for (origin, level), value in origins.items()
        },
        "reacquisition_rows": int(nodes["reacquisition_flag"].eq(1).sum()),
    }


def _comparison_row(
    scenario: str,
    role: str,
    detector: str,
    metrics: dict[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario": scenario,
        "role": role,
        "detector": detector,
        "threshold": metrics["threshold"],
    }
    if role == "clean":
        row.update({
            "pre_or_clean_fpr": metrics["false_positive_rate"],
            "flag_count": metrics["false_positive_flags"],
            "post_tpr": None,
            "first_delay_s": None,
            "three_consecutive_delay_s": None,
        })
    else:
        row.update({
            "pre_or_clean_fpr": metrics["pre_false_positive_rate"],
            "flag_count": metrics["pre_false_flags"],
            "post_tpr": metrics["post_detection_rate"],
            "first_delay_s": metrics["first_detection_delay_s"],
            "three_consecutive_delay_s": metrics["first_three_consecutive_delay_s"],
        })
    return row


def _paired_flags(
    events: pd.DataFrame,
    global_threshold: float,
    state_threshold: float,
    mask: pd.Series | None = None,
) -> dict[str, int]:
    selected = pd.Series(True, index=events.index) if mask is None else mask.astype(bool)
    global_flags = events[GLOBAL_SCORE].gt(global_threshold) & selected
    state_flags = events[STATE_SCORE].gt(state_threshold) & selected
    return {
        "windows": int(selected.sum()),
        "both_flag": int((global_flags & state_flags).sum()),
        "global_only_flag": int((global_flags & ~state_flags & selected).sum()),
        "state_only_flag": int((~global_flags & state_flags & selected).sum()),
        "neither_flag": int((~global_flags & ~state_flags & selected).sum()),
    }


def _delay_not_worse(state_delay: object, global_delay: object) -> bool:
    if global_delay is None:
        return state_delay is None
    if state_delay is None:
        return False
    return float(state_delay) <= float(global_delay)


def run(document: dict[str, object], out_dir: Path, manifest_path: Path) -> dict[str, object]:
    _validate_manifest(document)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_paths = [_resolve_path(str(value)) for value in document["clean_scores"]]

    # Only clean score files are opened until both reference pools and event
    # thresholds are fitted. Attack paths are first opened in the loop below.
    clean_scores = _read_scores(clean_paths)
    split_document = document.get("clean_split", {})
    if not isinstance(split_document, dict):
        raise ValueError("clean_split must be an object")
    reference_fraction = float(split_document.get("reference_fraction", 0.60))
    event_fraction = float(split_document.get("event_calibration_fraction", 0.20))
    score_column = str(document.get("score_column", "prn_node_rmse"))
    split = chronological_clean_split(
        clean_scores,
        reference_fraction=reference_fraction,
        event_calibration_fraction=event_fraction,
        score_column=score_column,
    )
    age_cutoffs_s = tuple(document.get("age_cutoffs_s", [10.0, 30.0]))
    min_pool_rows = int(document.get("min_pool_rows", 100))
    event_quantile = float(document.get("event_quantile", 0.99))
    calibration, calibration_events, calibration_nodes = calibrate_receiver_state_detectors(
        split.reference,
        split.event_calibration,
        score_column=score_column,
        age_cutoffs_s=age_cutoffs_s,
        min_pool_rows=min_pool_rows,
        event_quantile=event_quantile,
    )
    calibration_events.to_csv(out_dir / "event_calibration_scores.csv", index=False)
    calibration_nodes.to_csv(out_dir / "event_calibration_nodes.csv", index=False)
    split.reference.to_csv(out_dir / "clean_reference_scores.csv", index=False)
    split.held_clean.to_csv(out_dir / "held_clean_scores.csv", index=False)

    calibration_record: dict[str, object] = {
        "normal_only": True,
        "attack_score_csv_opened_before_freeze": False,
        "score_column": score_column,
        "reference_fraction_per_clean_run": reference_fraction,
        "event_calibration_fraction_per_clean_run": event_fraction,
        "held_clean_fraction_per_clean_run": 1.0 - reference_fraction - event_fraction,
        "event_quantile": calibration.event_quantile,
        "global_event_threshold": calibration.global_event_threshold,
        "state_event_threshold": calibration.state_event_threshold,
        "reference_rows": calibration.reference_rows,
        "reference_events": calibration.reference_events,
        "event_calibration_rows": calibration.event_calibration_rows,
        "event_calibration_events": calibration.event_calibration_events,
        "reference_inventory": reference_inventory(calibration.reference),
        "event_calibration_pool_usage": _pool_usage(calibration_nodes),
        "clean_split_inventory": list(split.inventory),
        "clean_inputs": [_file_record(path) for path in clean_paths],
        "implementation_files": [
            _file_record(ROOT / "src" / "gnss_doppler_lab" / "receiver_state_tail.py"),
            _file_record(ROOT / "scripts" / "eval_receiver_state_tail.py"),
        ],
        "excluded_conditioning_fields": {
            "epoch_count": "receiver integration/update-rate regimes are not portable lock-quality states",
        },
    }
    source_model = document.get("source_model")
    if source_model is not None:
        model_path = _resolve_path(str(source_model))
        calibration_record["source_model"] = _file_record(model_path)
    (out_dir / "calibration.json").write_text(
        json.dumps(calibration_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    detector_specs = {
        "matched_global_conformal": (GLOBAL_SCORE, calibration.global_event_threshold),
        "receiver_state_conformal": (STATE_SCORE, calibration.state_event_threshold),
    }
    comparison: list[dict[str, object]] = []
    held_events, held_nodes = build_conformal_event_scores(
        split.held_clean, reference=calibration.reference
    )
    held_events.to_csv(out_dir / "held_clean_event_scores.csv", index=False)
    held_nodes.to_csv(out_dir / "held_clean_nodes.csv", index=False)
    held_result: dict[str, object] = {
        "pool_usage": _pool_usage(held_nodes),
        "paired_flags": _paired_flags(
            held_events,
            calibration.global_event_threshold,
            calibration.state_event_threshold,
        ),
        "detectors": {},
    }
    for detector, (score_name, threshold) in detector_specs.items():
        metrics = evaluate_clean(held_events, score_name, threshold)
        held_result["detectors"][detector] = metrics
        comparison.append(_comparison_row("held_clean", "clean", detector, metrics))

    guard_s = float(document.get("guard_s", 10.0))
    availability_offset_s = float(document.get("availability_offset_s", 1.0))
    attacks: dict[str, object] = {}
    for name, specification in document["attacks"].items():
        path = _resolve_path(str(specification["score_csv"]))
        onset_s = float(specification["onset_s"])
        attack_scores = _read_scores([path])
        events, nodes = build_conformal_event_scores(
            attack_scores, reference=calibration.reference
        )
        events.to_csv(out_dir / f"{name}_event_scores.csv", index=False)
        nodes.to_csv(out_dir / f"{name}_nodes.csv", index=False)
        result: dict[str, object] = {
            "input": _file_record(path),
            "onset_s": onset_s,
            "pool_usage": _pool_usage(nodes),
            "paired_pre_flags": _paired_flags(
                events,
                calibration.global_event_threshold,
                calibration.state_event_threshold,
                events["window_start_s"].lt(onset_s - guard_s),
            ),
            "paired_post_flags": _paired_flags(
                events,
                calibration.global_event_threshold,
                calibration.state_event_threshold,
                events["window_start_s"].ge(onset_s + guard_s),
            ),
            "detectors": {},
        }
        for detector, (score_name, threshold) in detector_specs.items():
            metrics = evaluate_attack(
                events,
                score_name,
                threshold,
                onset_s,
                guard_s=guard_s,
                availability_offset_s=availability_offset_s,
            )
            result["detectors"][detector] = metrics
            comparison.append(_comparison_row(name, "attack", detector, metrics))
        attacks[str(name)] = result

    global_clean = held_result["detectors"]["matched_global_conformal"]
    state_clean = held_result["detectors"]["receiver_state_conformal"]
    decision_checks = {
        "fewer_held_clean_false_flags": (
            state_clean["false_positive_flags"] < global_clean["false_positive_flags"]
        ),
        "no_attack_pre_false_flag_increase": all(
            result["detectors"]["receiver_state_conformal"]["pre_false_flags"]
            <= result["detectors"]["matched_global_conformal"]["pre_false_flags"]
            for result in attacks.values()
        ),
        "no_attack_post_detection_loss": all(
            result["detectors"]["receiver_state_conformal"]["post_detection_flags"]
            >= result["detectors"]["matched_global_conformal"]["post_detection_flags"]
            for result in attacks.values()
        ),
        "no_first_detection_delay_increase": all(
            _delay_not_worse(
                result["detectors"]["receiver_state_conformal"]["first_detection_delay_s"],
                result["detectors"]["matched_global_conformal"]["first_detection_delay_s"],
            )
            for result in attacks.values()
        ),
    }
    decision = {
        "rule": "fewer held-clean false flags and no per-attack increase in pre-onset flags, post-detection loss, or first-detection delay",
        "checks": decision_checks,
        "passed": bool(all(decision_checks.values())),
    }

    comparison_frame = pd.DataFrame(comparison)
    comparison_frame.to_csv(out_dir / "comparison.csv", index=False)
    summary: dict[str, object] = {
        "schema": SUMMARY_SCHEMA,
        "manifest": _file_record(manifest_path),
        "protocol": document.get("protocol", ""),
        "hypothesis": document.get("hypothesis", ""),
        "calibration": calibration_record,
        "held_clean": held_result,
        "attacks": attacks,
        "decision": decision,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(comparison_frame.to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    manifest_path = _resolve_path(args.manifest)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    run(document, _resolve_path(args.out_dir), manifest_path)


if __name__ == "__main__":
    main()
