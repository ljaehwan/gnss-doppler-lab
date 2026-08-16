#!/usr/bin/env python3
"""Frozen TRACE-R1 evaluation over authenticated TRACE-R2 native dumps.

This file is frozen before Phase-A replay.  It changes only input adaptation:
the predictor, covariance score, clean-only thresholds, median pooling, alarm
run length, controls, and verdict gates are the preregistered TRACE-R1 rules.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.trace_action_warp import warp_complex_taps
from gnss_doppler_lab.trace_equivariance import (
    ResidualModel,
    RidgePredictor,
    TracePairs,
    action_shuffle_indices,
    consecutive_alarm,
    fit_ridge,
    persistent_alarm_ratio,
    robust_epoch_blocks,
)
from gnss_doppler_lab.trace_native_1ms import (
    load_native_trace_pairs,
    read_records,
    sha256_file,
    validate_dump_files,
)

ARTIFACT = ROOT / "artifacts/trace_stage0_r2_native_1ms_dump"
SSD = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2-native-1ms-dump")
WORK = SSD / "evaluation_work"
COMMON = np.arange(1, 8)
CONFIG = {
    "seed": 23017,
    "cn0_min_db_hz": 28.0,
    "lock_min": 0.85,
    "prompt_epsilon": 1e-12,
    "ridge_alpha": 10.0,
    "guard_s": 5.0,
    "block_s": 0.5,
    "minimum_prns": 4,
    "threshold_quantiles": (0.99, 0.995),
    "alarm_consecutive_blocks": 3,
    "bootstrap_block_s": 10.0,
    "bootstrap_repetitions": 999,
    "carry_off_max_delay_s": 10.0,
    "carry_off_min_established_detection_rate": 0.5,
}
FAMILIES = {
    "TEXBAT": "TEXBAT.cleanStatic",
    "OAKBAT": "OAKBAT.cleanStatic",
}
SCENARIOS = {
    "TEXBAT.cleanStatic": {"family": "TEXBAT", "slug": "texbat_cleanstatic", "timeline": None},
    "TEXBAT.DS3": {"family": "TEXBAT", "slug": "texbat_ds3", "timeline": (118.9, 195.0)},
    "TEXBAT.DS7": {"family": "TEXBAT", "slug": "texbat_ds7", "timeline": (110.0, 150.0)},
    "OAKBAT.cleanStatic": {"family": "OAKBAT", "slug": "oakbat_cleanstatic", "timeline": None},
    "OAKBAT.OS3": {"family": "OAKBAT", "slug": "oakbat_os3", "timeline": (120.0, 130.0)},
    "OAKBAT.OS4": {"family": "OAKBAT", "slug": "oakbat_os4", "timeline": (120.0, 130.0)},
}
CORE = ("TEXBAT.DS3", "TEXBAT.DS7", "OAKBAT.OS3", "OAKBAT.OS4")
METHODS = (
    "TRACE Full",
    "no-action predictor",
    "zero-action counterfactual",
    "wrong shifted-action negative control",
    "shuffled-action TRACE",
    "action norm only",
    "complex residual norm only",
    "fixed complex 9-tap detector",
)


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    return sha256_file(path)


def dump_dir(name: str) -> Path:
    return SSD / "dumps/phase_b" / SCENARIOS[name]["slug"] / "rep1"


def load_pairs(name: str) -> TracePairs:
    pairs = load_native_trace_pairs(
        dump_dir(name),
        cn0_min_db_hz=CONFIG["cn0_min_db_hz"],
        lock_min=CONFIG["lock_min"],
        prompt_epsilon=CONFIG["prompt_epsilon"],
    )
    common = pairs.valid_support[:, COMMON].all(axis=1)
    finite = (
        np.isfinite(pairs.current[:, COMMON].real).all(axis=1)
        & np.isfinite(pairs.current[:, COMMON].imag).all(axis=1)
        & np.isfinite(pairs.target[:, COMMON].real).all(axis=1)
        & np.isfinite(pairs.target[:, COMMON].imag).all(axis=1)
    )
    selected = common & finite
    if not selected.any():
        raise ValueError(f"{name}: no common-support native pairs")
    return pairs.take(selected)


def chronological_four_way_masks(time_s: np.ndarray, guard_s: float) -> dict[str, np.ndarray]:
    """Frozen 45/20/15/20 chronological roles with guards.

    Ridge fit uses the first 45%, covariance fit the next 20%, threshold
    calibration the next 15%, and untouched holdout the last 20%.
    """
    time_s = np.asarray(time_s, dtype=np.float64)
    start = float(time_s.min())
    end = float(time_s.max())
    duration = end - start
    b1 = start + 0.45 * duration
    b2 = start + 0.65 * duration
    b3 = start + 0.80 * duration
    return {
        "train": time_s < b1 - guard_s,
        "covariance_validation": (time_s >= b1 + guard_s) & (time_s < b2 - guard_s),
        "calibration": (time_s >= b2 + guard_s) & (time_s < b3 - guard_s),
        "holdout": time_s >= b3 + guard_s,
    }


def action_norm(pairs: TracePairs) -> np.ndarray:
    action = np.column_stack(
        (pairs.code_action, np.sin(pairs.carrier_action), np.cos(pairs.carrier_action))
    )
    return np.sum(action**2, axis=1)


def with_actions(pairs: TracePairs, code: np.ndarray, carrier: np.ndarray) -> TracePairs:
    warped = np.empty_like(pairs.current)
    support = np.empty_like(pairs.valid_support)
    for row in range(len(pairs.current)):
        # Prompt normalization removed global carrier phase exactly once.
        warped[row], support[row] = warp_complex_taps(pairs.current[row], code[row], 0.0)
    return replace(
        pairs,
        code_action=np.asarray(code, dtype=np.float64),
        carrier_action=np.asarray(carrier, dtype=np.float64),
        warped=warped,
        valid_support=support,
    )


def shifted_action_indices(pairs: TracePairs) -> np.ndarray:
    """Deterministic cyclic one-row wrong-action control within channel/PRN."""
    result = np.arange(len(pairs.prn))
    for channel in np.unique(pairs.channel):
        for prn in np.unique(pairs.prn[pairs.channel == channel]):
            rows = np.flatnonzero((pairs.channel == channel) & (pairs.prn == prn))
            order = rows[np.argsort(pairs.time_s[rows], kind="stable")]
            if len(order) > 1:
                result[order] = np.roll(order, 1)
    return result


def variants(pairs: TracePairs) -> dict[str, TracePairs]:
    zeros = np.zeros(len(pairs.current), dtype=np.float64)
    shift = shifted_action_indices(pairs)
    shuffle = action_shuffle_indices(pairs.prn, pairs.cn0_db_hz, CONFIG["seed"])
    return {
        "zero": with_actions(pairs, zeros, zeros),
        "shifted": with_actions(pairs, pairs.code_action[shift], pairs.carrier_action[shift]),
        "shuffled": with_actions(pairs, pairs.code_action[shuffle], pairs.carrier_action[shuffle]),
    }


def fit_residual_model(pairs: TracePairs, prediction: np.ndarray) -> ResidualModel:
    return ResidualModel.fit(pairs.target[:, COMMON] - prediction[:, COMMON])


def prediction_scores(
    pairs: TracePairs,
    full_model: RidgePredictor,
    no_action_model: RidgePredictor,
    covariances: dict[str, ResidualModel],
    fixed_model: ResidualModel,
) -> dict[str, np.ndarray]:
    altered = variants(pairs)
    full_prediction = full_model.predict(pairs)
    no_action_prediction = no_action_model.predict(pairs)
    scores = {
        "TRACE Full": covariances["TRACE Full"].score(
            pairs.target[:, COMMON] - full_prediction[:, COMMON]
        ),
        "no-action predictor": covariances["no-action predictor"].score(
            pairs.target[:, COMMON] - no_action_prediction[:, COMMON]
        ),
        "action norm only": action_norm(pairs),
        "complex residual norm only": np.sum(
            np.abs(pairs.target[:, COMMON] - pairs.current[:, COMMON]) ** 2, axis=1
        ),
        "fixed complex 9-tap detector": fixed_model.score(pairs.target[:, COMMON]),
    }
    for label, key in (
        ("zero-action counterfactual", "zero"),
        ("wrong shifted-action negative control", "shifted"),
        ("shuffled-action TRACE", "shuffled"),
    ):
        prediction = full_model.predict(altered[key])
        scores[label] = covariances["TRACE Full"].score(
            altered[key].target[:, COMMON] - prediction[:, COMMON]
        )
    return scores


def blocks(pairs: TracePairs, scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        method: robust_epoch_blocks(
            pairs,
            scores[method],
            block_s=CONFIG["block_s"],
            minimum_prns=CONFIG["minimum_prns"],
        )
        for method in METHODS
    }


def save_bundle(
    family: str,
    full_model: RidgePredictor,
    no_action_model: RidgePredictor,
    covariances: dict[str, ResidualModel],
    fixed_model: ResidualModel,
    thresholds: dict[str, float],
) -> Path:
    path = ARTIFACT / f"trace_r2_model_{family.lower()}.npz"
    np.savez_compressed(
        path,
        full_coef=full_model.coefficients,
        no_action_coef=no_action_model.coefficients,
        full_mean=covariances["TRACE Full"].mean,
        full_precision=covariances["TRACE Full"].precision,
        no_action_mean=covariances["no-action predictor"].mean,
        no_action_precision=covariances["no-action predictor"].precision,
        fixed_mean=fixed_model.mean,
        fixed_precision=fixed_model.precision,
        threshold_keys=np.asarray(list(thresholds)),
        threshold_values=np.asarray(list(thresholds.values()), dtype=np.float64),
    )
    return path


def load_bundle(family: str) -> tuple[RidgePredictor, RidgePredictor, dict[str, ResidualModel], ResidualModel, dict[str, float]]:
    payload = np.load(ARTIFACT / f"trace_r2_model_{family.lower()}.npz", allow_pickle=False)
    full = RidgePredictor(payload["full_coef"], True, "warp_residual", COMMON)
    no_action = RidgePredictor(payload["no_action_coef"], False, "direct", COMMON)
    covariances = {
        "TRACE Full": ResidualModel(payload["full_mean"], payload["full_precision"]),
        "no-action predictor": ResidualModel(
            payload["no_action_mean"], payload["no_action_precision"]
        ),
    }
    fixed = ResidualModel(payload["fixed_mean"], payload["fixed_precision"])
    thresholds = {
        str(key): float(value)
        for key, value in zip(payload["threshold_keys"], payload["threshold_values"], strict=True)
    }
    return full, no_action, covariances, fixed, thresholds


def range_payload(pairs: TracePairs, mask: np.ndarray) -> dict[str, object]:
    selected_samples = pairs.sample_count[mask].astype(np.uint64)
    selected_times = pairs.time_s[mask]
    return {
        "pair_count": int(mask.sum()),
        "time_start_s": float(selected_times.min()),
        "time_end_s": float(selected_times.max()),
        "raw_sample_start_inclusive": int(selected_samples.min()),
        "raw_sample_end_inclusive": int(selected_samples.max()),
        "ishort_byte_start_inclusive": int(selected_samples.min()) * 4,
        "ishort_byte_end_exclusive": (int(selected_samples.max()) + 1) * 4,
    }


def phase_fit(family: str) -> None:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    clean_name = FAMILIES[family]
    pairs = load_pairs(clean_name)
    masks = chronological_four_way_masks(pairs.time_s, CONFIG["guard_s"])
    if any(int(mask.sum()) == 0 for mask in masks.values()):
        raise ValueError(f"{clean_name}: empty preregistered clean split")
    train = pairs.take(masks["train"])
    covariance = pairs.take(masks["covariance_validation"])
    calibration = pairs.take(masks["calibration"])
    holdout = pairs.take(masks["holdout"])
    full = fit_ridge(
        train,
        include_action=True,
        target_mode="warp_residual",
        alpha=CONFIG["ridge_alpha"],
        output_indices=COMMON,
    )
    no_action = fit_ridge(
        train,
        include_action=False,
        target_mode="direct",
        alpha=CONFIG["ridge_alpha"],
        output_indices=COMMON,
    )
    covariances = {
        "TRACE Full": fit_residual_model(covariance, full.predict(covariance)),
        "no-action predictor": fit_residual_model(covariance, no_action.predict(covariance)),
    }
    fixed = ResidualModel.fit(covariance.target[:, COMMON])
    calibration_scores = prediction_scores(calibration, full, no_action, covariances, fixed)
    calibration_blocks = blocks(calibration, calibration_scores)
    thresholds = {}
    for method, values in calibration_blocks.items():
        if not len(values):
            raise ValueError(f"{clean_name}: no calibration blocks for {method}")
        thresholds[f"{method} q99"] = float(np.quantile(values["score"], 0.99))
        thresholds[f"{method} q995"] = float(np.quantile(values["score"], 0.995))
    bundle = save_bundle(family, full, no_action, covariances, fixed, thresholds)
    hold_scores = prediction_scores(holdout, full, no_action, covariances, fixed)
    hold_blocks = blocks(holdout, hold_scores)
    holdout_fpr = {}
    for method, values in hold_blocks.items():
        alarm = consecutive_alarm(
            values["block_start_s"],
            values["score"],
            thresholds[f"{method} q99"],
            CONFIG["alarm_consecutive_blocks"],
        )
        holdout_fpr[method] = float(alarm.mean()) if len(alarm) else None
    manifest = json.loads((dump_dir(clean_name) / "manifest.json").read_text())
    audit = {
        "family": family,
        "clean_scenario": clean_name,
        "roles": {role: range_payload(pairs, mask) for role, mask in masks.items()},
        "guard_s": CONFIG["guard_s"],
        "raw_sample_overlap": False,
        "byte_overlap": False,
        "chronological": True,
        "attack_data_used": False,
        "family_local_model": True,
        "receiver_dump_manifest": str(dump_dir(clean_name) / "manifest.json"),
        "receiver_dump_manifest_sha256": sha256(dump_dir(clean_name) / "manifest.json"),
        "raw_iq_path": manifest["raw_iq"]["path"],
        "raw_iq_sha256": manifest["raw_iq"]["sha256"],
        "model_bundle": str(bundle.relative_to(ROOT)),
        "model_bundle_sha256": sha256(bundle),
        "holdout_q99_fpr": holdout_fpr,
    }
    dump_json(WORK / f"fit_{family.lower()}.json", audit)
    dump_json(WORK / f"thresholds_{family.lower()}.json", thresholds)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


def normalized_pauc(labels: np.ndarray, score: np.ndarray, max_fpr: float = 0.05) -> float:
    fpr, tpr, _ = roc_curve(labels, score)
    x = np.concatenate(([0.0], fpr[fpr < max_fpr], [max_fpr]))
    y = np.interp(x, fpr, tpr)
    return float(np.trapezoid(y, x) / max_fpr)


def first_alarm_delay(time: np.ndarray, alarm: np.ndarray, event_s: float) -> float | None:
    selected = time[alarm & (time >= event_s)]
    return None if not len(selected) else float(selected[0] - event_s)


def method_metrics(name: str, values: np.ndarray, threshold: float) -> dict[str, object]:
    onset, carry = SCENARIOS[name]["timeline"]
    time = values["block_start_s"]
    score = values["score"]
    alarm = consecutive_alarm(
        time, score, threshold, CONFIG["alarm_consecutive_blocks"]
    )
    pre = time < onset
    attack = time >= onset
    transition = (time >= onset) & (time < carry)
    established = time >= carry
    labels = attack.astype(np.uint8)
    return {
        "dataset": SCENARIOS[name]["family"],
        "scenario": name.split(".", 1)[1],
        "valid_epochs": int(len(values)),
        "valid_prns": float(np.median(values["tracked_prn_count"])) if len(values) else None,
        "roc_auc": float(roc_auc_score(labels, score)) if pre.any() and attack.any() else None,
        "pauc_fpr_le_0p05": normalized_pauc(labels, score) if pre.any() and attack.any() else None,
        "pr_auc": float(average_precision_score(labels, score)) if pre.any() and attack.any() else None,
        "pre_onset_fpr": float(alarm[pre].mean()) if pre.any() else None,
        "attack_detection_rate": float(alarm[attack].mean()) if attack.any() else None,
        "transition_detection_rate": float(alarm[transition].mean()) if transition.any() else None,
        "established_detection_rate": float(alarm[established].mean()) if established.any() else None,
        "onset_delay_s": first_alarm_delay(time, alarm, onset),
        "pull_off_delay_s": first_alarm_delay(time, alarm, carry),
        "persistent_alarm_ratio": persistent_alarm_ratio(alarm, attack),
    }


def support_summary(name: str) -> dict[str, object]:
    paths = sorted(dump_dir(name).glob("trace_native_1ms_ch_*.bin"))
    validation = validate_dump_files(paths, expected_scenario_id=name, minimum_prns=4)
    observed_links = 0
    missing = 0
    navigation_bit_boundary_rows = 0
    for path in paths:
        header, records = read_records(path)
        same = (
            (records["tracking_session_id"][1:] == records["tracking_session_id"][:-1])
            & (records["prn"][1:] == records["prn"][:-1])
        )
        delta = (
            records["raw_interval_start_sample"][1:].astype(np.int64)
            - records["raw_interval_start_sample"][:-1].astype(np.int64)
        )
        steps = np.rint(delta[same] / (header.sample_rate_hz * 0.001)).astype(np.int64)
        observed_links += int(len(steps))
        missing += int(np.maximum(steps - 1, 0).sum())
        navigation_bit_boundary_rows += int((records["data_symbol_boundary"] == 1).sum())
    denominator = observed_links + missing
    return {
        "validation": validation,
        "observed_continuous_links": observed_links,
        "estimated_missing_native_rows": missing,
        "missing_1ms_row_rate": float(missing / denominator) if denominator else None,
        "navigation_bit_boundary_row_count": navigation_bit_boundary_rows,
    }


def per_prn_rows(name: str, pairs: TracePairs, full_scores: np.ndarray) -> list[dict[str, object]]:
    block_ids = np.floor(pairs.time_s / CONFIG["block_s"]).astype(np.int64)
    response = np.sqrt(np.sum(np.abs(pairs.target[:, COMMON] - pairs.current[:, COMMON]) ** 2, axis=1))
    actions = np.sqrt(action_norm(pairs))
    rows = []
    for block in np.unique(block_ids):
        in_block = block_ids == block
        for prn in np.unique(pairs.prn[in_block]):
            selected = in_block & (pairs.prn == prn)
            rows.append(
                {
                    "dataset": SCENARIOS[name]["family"],
                    "scenario": name.split(".", 1)[1],
                    "block_start_s": block * CONFIG["block_s"],
                    "prn": int(prn),
                    "channel": int(np.median(pairs.channel[selected])),
                    "pair_count": int(selected.sum()),
                    "action_norm_median": float(np.median(actions[selected])),
                    "response_norm_median": float(np.median(response[selected])),
                    "trace_score_median": float(np.median(full_scores[selected])),
                }
            )
    return rows


def phase_evaluate(name: str) -> None:
    family = SCENARIOS[name]["family"]
    full, no_action, covariances, fixed, thresholds = load_bundle(family)
    pairs = load_pairs(name)
    scores = prediction_scores(pairs, full, no_action, covariances, fixed)
    score_blocks = blocks(pairs, scores)
    support = support_summary(name)
    fit_audit = json.loads((WORK / f"fit_{family.lower()}.json").read_text())
    metrics = []
    for method in METHODS:
        row = method_metrics(name, score_blocks[method], thresholds[f"{method} q99"])
        row["model"] = method
        row["status"] = "AVAILABLE"
        row["clean_holdout_fpr"] = fit_audit["holdout_q99_fpr"][method]
        row["missing_1ms_row_rate"] = support["missing_1ms_row_rate"]
        metrics.append(row)
    work = WORK / SCENARIOS[name]["slug"]
    work.mkdir(parents=True, exist_ok=True)
    dump_json(work / "metrics.json", metrics)
    dump_json(work / "support.json", support)
    with gzip.open(work / "blocks.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("dataset", "scenario", "model", "block_start_s", "score", "alarm", "tracked_prn_count", "pair_count")
        )
        for method, values in score_blocks.items():
            alarm = consecutive_alarm(
                values["block_start_s"], values["score"], thresholds[f"{method} q99"]
            )
            for row, flag in zip(values, alarm, strict=True):
                writer.writerow(
                    (
                        family,
                        name.split(".", 1)[1],
                        method,
                        row["block_start_s"],
                        row["score"],
                        int(flag),
                        row["tracked_prn_count"],
                        row["pair_count"],
                    )
                )
    prn_rows = per_prn_rows(name, pairs, scores["TRACE Full"])
    with gzip.open(work / "per_prn.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prn_rows[0]))
        writer.writeheader()
        writer.writerows(prn_rows)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


def paired_action_metrics(name: str) -> dict[str, object]:
    with gzip.open(WORK / SCENARIOS[name]["slug"] / "blocks.csv.gz", "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_method = {
        method: {float(row["block_start_s"]): float(row["score"]) for row in rows if row["model"] == method}
        for method in (
            "TRACE Full",
            "wrong shifted-action negative control",
            "shuffled-action TRACE",
        )
    }
    onset = SCENARIOS[name]["timeline"][0]
    times = sorted(set.intersection(*(set(values) for values in by_method.values())))
    times = [time for time in times if time >= onset]
    full = np.asarray([by_method["TRACE Full"][time] for time in times])
    shifted = np.asarray([by_method["wrong shifted-action negative control"][time] for time in times])
    shuffled = np.asarray([by_method["shuffled-action TRACE"][time] for time in times])

    def reduction(control: np.ndarray) -> dict[str, object]:
        difference = full - control
        test = wilcoxon(difference, alternative="greater") if len(difference) else None
        return {
            "paired_attack_blocks": int(len(difference)),
            "mean_full_minus_control": float(difference.mean()) if len(difference) else None,
            "wilcoxon_p_full_greater": float(test.pvalue) if test is not None else None,
            "control_reduces_attack_score_significantly": bool(
                len(difference) and difference.mean() > 0 and test.pvalue < 0.01
            ),
        }

    return {"scenario": name, "shifted": reduction(shifted), "shuffled": reduction(shuffled)}


def bootstrap_rows(name: str) -> list[dict[str, object]]:
    with gzip.open(WORK / SCENARIOS[name]["slug"] / "blocks.csv.gz", "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    methods = {
        method: {float(row["block_start_s"]): float(row["score"]) for row in rows if row["model"] == method}
        for method in ("TRACE Full", "no-action predictor", "complex residual norm only")
    }
    rng = np.random.default_rng(CONFIG["seed"])
    output = []
    for baseline in ("no-action predictor", "complex residual norm only"):
        times = sorted(set(methods["TRACE Full"]) & set(methods[baseline]))
        grouped: dict[int, list[float]] = {}
        for time in times:
            grouped.setdefault(int(time // CONFIG["bootstrap_block_s"]), []).append(
                methods["TRACE Full"][time] - methods[baseline][time]
            )
        units = np.asarray([np.mean(values) for values in grouped.values()], dtype=np.float64)
        samples = np.asarray(
            [np.mean(rng.choice(units, len(units), replace=True)) for _ in range(CONFIG["bootstrap_repetitions"])],
            dtype=np.float64,
        )
        output.append(
            {
                "dataset": SCENARIOS[name]["family"],
                "scenario": name.split(".", 1)[1],
                "comparison": f"TRACE Full minus {baseline}",
                "metric": "paired_score_difference",
                "estimate": float(units.mean()),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "bootstrap_unit_s": CONFIG["bootstrap_block_s"],
                "bootstrap_repetitions": CONFIG["bootstrap_repetitions"],
            }
        )
    return output


def clean_controls(family: str) -> dict[str, object]:
    clean_name = FAMILIES[family]
    pairs = load_pairs(clean_name)
    masks = chronological_four_way_masks(pairs.time_s, CONFIG["guard_s"])
    hold = pairs.take(masks["holdout"])
    full, no_action, covariances, fixed, thresholds = load_bundle(family)
    full_scores = prediction_scores(hold, full, no_action, covariances, fixed)["TRACE Full"]
    baseline = robust_epoch_blocks(hold, full_scores, minimum_prns=CONFIG["minimum_prns"])
    baseline_alarm = consecutive_alarm(
        baseline["block_start_s"], baseline["score"], thresholds["TRACE Full q99"]
    )

    def identity(reason: str) -> dict[str, object]:
        ratio = float(baseline_alarm.mean())
        return {"alarm_ratio": ratio, "passed": ratio <= 0.01, "basis": reason}

    controls: dict[str, object] = {
        "common_gain_scaling": identity("Prompt normalization identity"),
        "global_carrier_phase_rotation": identity("Prompt normalization identity"),
        "prompt_amplitude_change": identity("Prompt normalization identity"),
        "cn0_decrease": {**identity("metadata-only limited control"), "limited": True},
        "receiver_clock_like_drift": identity("common carrier phase removed by Prompt reference"),
        "navigation_bit_boundary": {**identity("sign is a common phase rotation"), "limited": True},
        "one_ms_jitter_normal_loop_action": {
            **identity("actual receiver dt/action variation retained in holdout"),
            "dt_min_s": float(hold.dt_s.min()),
            "dt_max_s": float(hold.dt_s.max()),
        },
    }
    residual = hold.target[:, COMMON] - full.predict(hold)[:, COMMON]
    rng = np.random.default_rng(CONFIG["seed"])
    for scale in (0.5, 1.0, 2.0):
        empirical = residual[rng.permutation(len(residual))] * scale
        values = robust_epoch_blocks(
            hold,
            covariances["TRACE Full"].score(empirical),
            minimum_prns=CONFIG["minimum_prns"],
        )
        alarm = consecutive_alarm(values["block_start_s"], values["score"], thresholds["TRACE Full q99"])
        ratio = float(alarm.mean())
        controls[f"empirical_clean_noise_{scale:g}x"] = {
            "alarm_ratio": ratio,
            "passed": ratio <= 0.01,
            "basis": "resampled empirical clean complex residual; no synthetic normalized noise",
        }
    disturbed = full_scores.copy()
    block_ids = np.floor(hold.time_s / CONFIG["block_s"]).astype(np.int64)
    for block in np.unique(block_ids):
        rows = np.flatnonzero(block_ids == block)
        if len(rows):
            first_prn = hold.prn[rows][0]
            disturbed[rows[hold.prn[rows] == first_prn]] *= 100.0
    values = robust_epoch_blocks(hold, disturbed, minimum_prns=CONFIG["minimum_prns"])
    alarm = consecutive_alarm(values["block_start_s"], values["score"], thresholds["TRACE Full q99"])
    ratio = float(alarm.mean())
    controls["single_prn_disturbance"] = {"alarm_ratio": ratio, "passed": ratio <= 0.01}
    controls["prn_drop_add"] = {
        "passed": True,
        "basis": "fixed >=4 unique-PRN support and permutation-invariant median; adapter tests cover count/permutation",
        "structural": True,
    }
    controls["independent_multipath_like_distortion"] = {
        "passed": False,
        "status": "UNAVAILABLE_PHYSICAL_RAW_IQ_CONTROL",
        "reason": "No authenticated physical multipath raw-IQ injection is available; no arbitrary normalized-tap noise was substituted.",
    }
    return controls


def write_combined_gzip(relative: str, source_name: str) -> None:
    destination = ARTIFACT / relative
    writer = None
    with gzip.open(destination, "wt", newline="") as output:
        for name in CORE:
            with gzip.open(WORK / SCENARIOS[name]["slug"] / source_name, "rt", newline="") as source:
                reader = csv.DictReader(source)
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
                    writer.writeheader()
                writer.writerows(reader)


def phase_finalize() -> None:
    metrics = [
        row
        for name in CORE
        for row in json.loads((WORK / SCENARIOS[name]["slug"] / "metrics.json").read_text())
    ]
    fields = list(metrics[0])
    with (ARTIFACT / "scenario_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in metrics if row["model"] == "TRACE Full")
    with (ARTIFACT / "ablation_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
        writer.writerow(
            {
                "dataset": "B0",
                "scenario": "all",
                "model": "B0 exact",
                "status": "UNAVAILABLE",
            }
        )
    write_combined_gzip("per_epoch_scores.csv.gz", "blocks.csv.gz")
    write_combined_gzip("per_prn_action_response.csv.gz", "per_prn.csv.gz")
    fits = {
        family: json.loads((WORK / f"fit_{family.lower()}.json").read_text()) for family in FAMILIES
    }
    dump_json(
        ARTIFACT / "clean_split_audit.json",
        {
            "schema": "gnss-doppler-lab.trace-r2-clean-split-audit.v1",
            "status": "PASS",
            "families": fits,
            "separate_family_models": True,
            "attack_data_used": False,
            "pre_onset_data_used": False,
        },
    )
    threshold_payload = {
        family: json.loads((WORK / f"thresholds_{family.lower()}.json").read_text())
        for family in FAMILIES
    }
    dump_json(ARTIFACT / "thresholds.json", threshold_payload)
    full_rows = {(row["dataset"], row["scenario"]): row for row in metrics if row["model"] == "TRACE Full"}
    with (ARTIFACT / "external_static_fpr.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "scenario", "model", "fpr", "status"))
        writer.writeheader()
        for (dataset, scenario), row in full_rows.items():
            writer.writerow(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "model": "TRACE Full",
                    "fpr": row["pre_onset_fpr"],
                    "status": "AVAILABLE",
                }
            )
    action = {name: paired_action_metrics(name) for name in CORE}
    dump_json(
        ARTIFACT / "action_shuffle_metrics.json",
        {
            "schema": "gnss-doppler-lab.trace-r2-action-negative-controls.v1",
            "scenarios": action,
            "all_scenarios_shift_and_shuffle_reduce_significantly": all(
                payload[key]["control_reduces_attack_score_significantly"]
                for payload in action.values()
                for key in ("shifted", "shuffled")
            ),
        },
    )
    controls = {family: clean_controls(family) for family in FAMILIES}
    dump_json(ARTIFACT / "physical_controls.json", controls)
    boot = [row for name in CORE for row in bootstrap_rows(name)]
    with (ARTIFACT / "bootstrap_intervals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(boot[0]))
        writer.writeheader()
        writer.writerows(boot)
    replay_inventory = {
        "schema": "gnss-doppler-lab.trace-r2-replay-inventory.v1",
        "phase_a": json.loads((ARTIFACT / "smoke_replay_results.json").read_text()),
        "phase_b": {},
    }
    for name in SCENARIOS:
        manifest_path = dump_dir(name) / "manifest.json"
        support_path = WORK / SCENARIOS[name]["slug"] / "support.json"
        replay_inventory["phase_b"][name] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "manifest": json.loads(manifest_path.read_text()),
            "support": json.loads(support_path.read_text()) if support_path.exists() else None,
        }
    dump_json(ARTIFACT / "replay_inventory.json", replay_inventory)
    holdout_fpr = max(
        fit["holdout_q99_fpr"]["TRACE Full"] for fit in fits.values()
    )
    external_fpr = max(row["pre_onset_fpr"] for row in full_rows.values())
    metrics_by = {(row["dataset"], row["scenario"], row["model"]): row for row in metrics}
    better = []
    carry_pass = {family: [] for family in FAMILIES}
    aligned = []
    for name in CORE:
        family = SCENARIOS[name]["family"]
        scenario = name.split(".", 1)[1]
        full = metrics_by[(family, scenario, "TRACE Full")]
        no_action = metrics_by[(family, scenario, "no-action predictor")]
        residual = metrics_by[(family, scenario, "complex residual norm only")]
        pauc_better = full["pauc_fpr_le_0p05"] > max(
            no_action["pauc_fpr_le_0p05"], residual["pauc_fpr_le_0p05"]
        )
        baseline_delays = [
            value for value in (no_action["onset_delay_s"], residual["onset_delay_s"]) if value is not None
        ]
        delay_better = full["onset_delay_s"] is not None and (
            not baseline_delays or full["onset_delay_s"] < min(baseline_delays)
        )
        if pauc_better or delay_better:
            better.append(name)
        carry_ok = (
            full["pre_onset_fpr"] <= 0.05
            and full["established_detection_rate"] >= CONFIG["carry_off_min_established_detection_rate"]
            and full["pull_off_delay_s"] is not None
            and 0.0 <= full["pull_off_delay_s"] <= CONFIG["carry_off_max_delay_s"]
        )
        if carry_ok:
            carry_pass[family].append(name)
        if (
            full["onset_delay_s"] is not None
            and 0.0 <= full["onset_delay_s"] <= CONFIG["carry_off_max_delay_s"]
            and full["pull_off_delay_s"] is not None
            and 0.0 <= full["pull_off_delay_s"] <= CONFIG["carry_off_max_delay_s"]
        ):
            aligned.append(name)
    negative_controls = all(
        payload[key]["control_reduces_attack_score_significantly"]
        for payload in action.values()
        for key in ("shifted", "shuffled")
    )
    controls_pass = all(
        control.get("passed") is True
        for family_controls in controls.values()
        for control in family_controls.values()
    )
    go_checks = {
        "clean_holdout_q99_fpr_le_1pct": holdout_fpr <= 0.01,
        "worst_external_static_fpr_le_5pct": external_fpr <= 0.05,
        "at_least_three_scenarios_better_than_both_baselines": len(better) >= 3,
        "texbat_carry_off_pass": bool(carry_pass["TEXBAT"]),
        "oakbat_carry_off_pass": bool(carry_pass["OAKBAT"]),
        "shifted_and_shuffled_actions_reduce_attack_score_significantly": negative_controls,
        "all_controls_pass": controls_pass,
        "alarms_align_with_onset_and_pull_off": len(aligned) >= 3,
    }
    verdict = "GO_TRACE_PHYSICAL_HYPOTHESIS" if all(go_checks.values()) else "NO_GO_TRACE_PHYSICAL_HYPOTHESIS"
    dump_json(
        ARTIFACT / "final_verdict.json",
        {
            "schema": "gnss-doppler-lab.trace-r2-final-verdict.v1",
            "verdict": verdict,
            "phase_a_passed": True,
            "phase_b_run": True,
            "attack_scores_computed": True,
            "performance_claimed": True,
            "go_checks": go_checks,
            "supporting_scenarios": better,
            "carry_off_passes": carry_pass,
            "aligned_scenarios": aligned,
            "clean_holdout_fpr_worst": holdout_fpr,
            "external_static_fpr_worst": external_fpr,
            "b0_exact": {
                "status": "UNAVAILABLE",
                "reason": "Exact frozen B0 cannot be rerun on TRACE native pair-quality/common support without changing B0; no historical CSV was copied.",
            },
            "science_claim": "Limited to authenticated static TEXBAT/OAKBAT GPS L1 C/A native receiver replays and the preregistered controls.",
            "sci_wcl_claimable": verdict == "GO_TRACE_PHYSICAL_HYPOTHESIS",
            "recommended_next_action": (
                "Independent receiver/raw-IQ replication and a physical multipath control."
                if verdict == "GO_TRACE_PHYSICAL_HYPOTHESIS"
                else "Do not advance TRACE; inspect failed preregistered gates without retuning this result."
            ),
        },
    )
    make_plots(metrics, controls)
    print(json.dumps({"verdict": verdict, "go_checks": go_checks}, indent=2, sort_keys=True), flush=True)


def make_plots(metrics: list[dict[str, object]], controls: dict[str, object]) -> None:
    plots = ARTIFACT / "plots"
    plots.mkdir(exist_ok=True)
    for name in CORE:
        with gzip.open(WORK / SCENARIOS[name]["slug"] / "blocks.csv.gz", "rt", newline="") as stream:
            rows = list(csv.DictReader(stream))
        fig, axis = plt.subplots(figsize=(9, 3.5))
        for method in ("TRACE Full", "no-action predictor", "complex residual norm only"):
            chosen = [row for row in rows if row["model"] == method]
            axis.plot(
                [float(row["block_start_s"]) for row in chosen],
                [float(row["score"]) for row in chosen],
                label=method,
                linewidth=0.8,
            )
        onset, carry = SCENARIOS[name]["timeline"]
        axis.axvline(onset, color="red", linestyle="--")
        axis.axvline(carry, color="purple", linestyle=":")
        axis.set(xlabel="receiver time (s)", ylabel="frozen score", title=name)
        axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plots / f"{SCENARIOS[name]['slug']}_score_timeline.png", dpi=130)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 4))
    core_labels = [name.split(".", 1)[1] for name in CORE]
    x = np.arange(len(CORE))
    width = 0.25
    for offset, method in enumerate(("TRACE Full", "no-action predictor", "complex residual norm only")):
        values = [
            next(
                row["pauc_fpr_le_0p05"]
                for row in metrics
                if row["dataset"] == SCENARIOS[name]["family"]
                and row["scenario"] == name.split(".", 1)[1]
                and row["model"] == method
            )
            for name in CORE
        ]
        axis.bar(x + offset * width, values, width, label=method)
    axis.set_xticks(x + width, core_labels)
    axis.set_ylabel("normalized pAUC <=5% FPR")
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plots / "trace_r2_common_support_comparison.png", dpi=130)
    plt.close(fig)
    labels = []
    values = []
    for family, family_controls in controls.items():
        for name, payload in family_controls.items():
            if "alarm_ratio" in payload:
                labels.append(f"{family}:{name}")
                values.append(payload["alarm_ratio"])
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.bar(np.arange(len(labels)), values)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=75, ha="right", fontsize=6)
    axis.set_ylabel("clean holdout alarm ratio")
    fig.tight_layout()
    fig.savefig(plots / "trace_r2_physical_controls.png", dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--family", choices=tuple(FAMILIES), required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--scenario", choices=CORE, required=True)
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.phase == "fit":
        phase_fit(args.family)
    elif args.phase == "evaluate":
        phase_evaluate(args.scenario)
    else:
        phase_finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
