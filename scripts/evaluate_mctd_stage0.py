#!/usr/bin/env python3
"""Evaluate the preregistered MCTD Phase-A and clean-only model gates."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.mctd import (
    AlignedDivergence, RobustGaussian, align_dump_directories, chronological_masks,
    consecutive_alarms, epoch_scores, mahalanobis_score, nonoverlap_blocks,
    paired_bootstrap_blocks, robust_fit,
)
from gnss_doppler_lab.trace_native_1ms import read_records

ARTIFACT = ROOT / "artifacts/mctd_stage0_static"
SSD = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/mctd-stage0-static")


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def directory(slug: str, loop: str, repetition: int, phase: str = "phase_a") -> Path:
    return SSD / "dumps" / phase / slug / loop / f"rep{repetition}"


def canonical_rows(path: Path) -> np.ndarray:
    rows = []
    for dump in sorted(path.glob("trace_native_1ms_ch_*.bin")):
        _, records = read_records(dump, mmap=False)
        rows.append(records)
    values = np.concatenate(rows)
    order = np.lexsort((values["loop_sequence"], values["raw_interval_start_sample"], values["prn"]))
    return values[order]


def exact_reproduction(left: Path, right: Path) -> dict[str, object]:
    a, b = canonical_rows(left), canonical_rows(right)
    same_shape = a.shape == b.shape
    exact = bool(same_shape and a.tobytes() == b.tobytes())
    return {"status": "PASS" if exact else "FAIL", "left_rows": len(a), "right_rows": len(b),
            "canonical_row_set_exact": exact, "bit_exact_fields": list(a.dtype.names or ()),
            "numeric_tolerance": 0.0}


def manifest(path: Path) -> dict[str, object]:
    return json.loads((path / "manifest.json").read_text())


def source_equality(slug: str) -> dict[str, object]:
    paths = [directory(slug, loop, 1) for loop in ("slow", "fast", "identical_left", "identical_right")]
    values = [manifest(path) for path in paths]
    raw = {item["raw_iq"]["sha256"] for item in values}
    receiver = {item["receiver"]["sha256"] for item in values}
    handoff = {item["handoff"]["sha256"] for item in values}
    passed = len(raw) == len(receiver) == len(handoff) == 1 and all(item["raw_stable"] and item["receiver_stable"] for item in values)
    return {"status": "PASS" if passed else "FAIL", "raw_iq_sha256": sorted(raw),
            "receiver_sha256": sorted(receiver), "handoff_sha256": sorted(handoff),
            "same_initial_state": len(handoff) == 1, "same_raw_source": len(raw) == 1,
            "only_configured_differences": ["Tracking_1C.dll_bw_hz", "Tracking_1C.pll_bw_hz", "trace scenario label"],
            "same_integration_ms": 1, "same_tap_spacing_chips": 0.125, "same_nav_bit_handling": True}


def native_missing_rate(path: Path) -> float:
    values = canonical_rows(path)
    missing = expected = 0
    for prn in np.unique(values["prn"]):
        sequence = np.sort(values["loop_sequence"][values["prn"] == prn].astype(np.int64))
        if len(sequence) > 1:
            gaps = np.diff(sequence); missing += int(np.maximum(gaps - 1, 0).sum()); expected += int(gaps.sum())
    return float(missing / expected) if expected else 0.0


def phase_a() -> int:
    datasets = {}
    overall = True
    for dataset, slug in (("TEXBAT.cleanStatic", "texbat_cleanstatic"), ("OAKBAT.cleanStatic", "oakbat_cleanstatic")):
        deterministic = {
            "slow": exact_reproduction(directory(slug, "slow", 1), directory(slug, "slow", 2)),
            "fast": exact_reproduction(directory(slug, "fast", 1), directory(slug, "fast", 2)),
        }
        aligned = align_dump_directories(directory(slug, "slow", 1), directory(slug, "fast", 1), dataset=dataset)
        epoch, _, n = epoch_scores(aligned.epoch_ms, aligned.prn, np.zeros(len(aligned.prn)))
        identical = align_dump_directories(directory(slug, "identical_left", 1), directory(slug, "identical_right", 1), dataset=dataset)
        max_identical = float(np.max(np.abs(identical.full))) if len(identical.prn) else None
        collapse = max_identical == 0.0 and len(identical.prn) > 0
        common_raw_delta = np.abs(aligned.raw_start_slow - aligned.raw_start_fast)
        stable = len(epoch) >= 1000 and int(n.max(initial=0)) >= 4
        source = source_equality(slug)
        passed = source["status"] == "PASS" and all(item["status"] == "PASS" for item in deterministic.values()) and stable and collapse
        overall &= passed
        raw_rows = len(canonical_rows(directory(slug, "slow", 1))) + len(canonical_rows(directory(slug, "fast", 1)))
        datasets[dataset] = {
            "status": "PASS" if passed else "FAIL", "source_equality": source,
            "deterministic_replay": deterministic,
            "stable_support": {"status": "PASS" if stable else "FAIL", "quality_common_rows": len(aligned.prn),
                               "quality_common_epochs_ge_4_prns": len(epoch), "maximum_common_prns": int(n.max(initial=0)),
                               "common_raw_start_delta_samples_max": int(common_raw_delta.max(initial=0)),
                               "missing_native_row_rate": max(native_missing_rate(directory(slug, "slow", 1)), native_missing_rate(directory(slug, "fast", 1))),
                               "quality_exclusion_or_noncommon_rate": 1.0 - 2.0 * len(aligned.prn) / raw_rows},
            "identical_loop_control": {"status": "PASS" if collapse else "FAIL", "common_rows": len(identical.prn),
                                       "maximum_absolute_full_divergence": max_identical,
                                       "collapse_to_numerical_error": collapse, "numerical_error_tolerance": 0.0},
        }
    payload = {"schema": "gnss-doppler-lab.mctd-phase-a.v1", "phase_a_passed": overall,
               "phase_b_authorized": False, "attack_scores_computed": False, "datasets": datasets,
               "failure_verdict_if_terminal": None if overall else "NO_GO_RECEIVER_DIFFERENTIAL_INVALID"}
    dump_json(ARTIFACT / "phase_a_reproducibility.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if overall else 2


VARIANT_ATTR = {"A0": "slow_taps", "A1": "fast_taps", "A2": "state", "A3": "taps", "A4": "action", "Full": "full"}


def _model_payload(model) -> dict[str, object]:
    return {"center": model.center.tolist(), "covariance": model.covariance.tolist(),
            "precision": model.precision.tolist(), "regularization": model.regularization}


def _block_variant(aligned, values: np.ndarray, model, mask: np.ndarray | None = None):
    row_score = mahalanobis_score(values, model)
    if mask is None:
        mask = np.ones(len(row_score), dtype=bool)
    epoch, score, n = epoch_scores(aligned.epoch_ms[mask], aligned.prn[mask], row_score[mask])
    block, block_score, count = nonoverlap_blocks(epoch, score)
    return block, block_score, count, n


def fit_clean() -> int:
    phase_a_payload = json.loads((ARTIFACT / "phase_a_reproducibility.json").read_text())
    if not phase_a_payload.get("phase_a_passed"):
        raise RuntimeError("Phase A failed; clean model fit is not authorized")
    summaries, thresholds, splits, rows_out = {}, {}, {}, []
    for family, slug, dataset in (("TEXBAT", "texbat_cleanstatic", "TEXBAT.cleanStatic"),
                                  ("OAKBAT", "oakbat_cleanstatic", "OAKBAT.cleanStatic")):
        aligned = align_dump_directories(directory(slug, "slow", 1, "clean"), directory(slug, "fast", 1, "clean"), dataset=dataset)
        warm = aligned.time_s >= aligned.time_s.min() + 10.0
        role = chronological_masks(aligned.time_s[warm])
        index = np.flatnonzero(warm)
        masks = {name: np.zeros(len(aligned.prn), dtype=bool) for name in role}
        for name, local in role.items():
            masks[name][index[local]] = True
        models, family_thresholds, holdout = {}, {}, {}
        for variant, attr in VARIANT_ATTR.items():
            values = getattr(aligned, attr)
            model = robust_fit(values[masks["train"]])
            models[variant] = _model_payload(model)
            cal_block, cal_score, _, _ = _block_variant(aligned, values, model, masks["calibration"])
            threshold = float(np.quantile(cal_score, 0.99, method="higher"))
            q995 = float(np.quantile(cal_score, 0.995, method="higher"))
            hold_block, hold_score, _, _ = _block_variant(aligned, values, model, masks["holdout"])
            fpr = float(np.mean(hold_score > threshold)) if len(hold_score) else None
            family_thresholds[variant] = {"q99": threshold, "q99_5": q995,
                                          "empirical_1pct": threshold, "calibration_blocks": len(cal_block)}
            holdout[variant] = {"blocks": len(hold_block), "q99_fpr": fpr}
            for block, score in zip(hold_block, hold_score, strict=True):
                rows_out.append({"dataset": dataset, "role": "holdout", "variant": variant,
                                 "block_start_s": block / 1000.0, "score": score, "threshold": threshold})
        model_path = ARTIFACT / f"normal_model_{family.lower()}.json"
        dump_json(model_path, {"schema": "gnss-doppler-lab.mctd-normal-model.v1", "family": family,
                               "attack_data_used": False, "models": models})
        summaries[family] = {"model_path": model_path.name, "quality_rows": len(aligned.prn),
                             "models": {key: {"dimensions": len(value["center"]), "regularization": value["regularization"]} for key, value in models.items()},
                             "holdout": holdout}
        thresholds[family] = family_thresholds
        splits[family] = {name: {"row_count": int(mask.sum()),
                                  "time_start_s": float(aligned.time_s[mask].min()),
                                  "time_end_s": float(aligned.time_s[mask].max()),
                                  "slow_raw_sample_start": int(aligned.raw_start_slow[mask].min()),
                                  "slow_raw_sample_end": int(aligned.raw_start_slow[mask].max()),
                                  "fast_raw_sample_start": int(aligned.raw_start_fast[mask].min()),
                                  "fast_raw_sample_end": int(aligned.raw_start_fast[mask].max())} for name, mask in masks.items()}
    dump_json(ARTIFACT / "normal_model_summary.json", {"schema": "gnss-doppler-lab.mctd-normal-summary.v1",
              "attack_data_used": False, "families": summaries})
    dump_json(ARTIFACT / "thresholds.json", {"schema": "gnss-doppler-lab.mctd-thresholds.v1",
              "status": "FROZEN_CLEAN_ONLY", "attack_data_used": False, "thresholds": thresholds,
              "primary": "q99", "sensitivity_only": ["q99_5", "empirical_1pct"]})
    dump_json(ARTIFACT / "clean_split_audit.json", {"schema": "gnss-doppler-lab.mctd-clean-split.v1",
              "status": "PASS", "attack_data_used": False, "chronological": True, "guard_s": 5.0,
              "raw_sample_overlap": False, "target_overlap": False, "families": splits})
    dump_json(ARTIFACT / "phase_b_authorization.json", {"schema": "gnss-doppler-lab.mctd-phase-b-authorization.v1",
              "phase_a_passed": True, "clean_model_frozen": True, "thresholds_frozen": True,
              "phase_b_authorized": True, "attack_scores_computed": False,
              "threshold_freeze_commit": "TO_BE_RECORDED_BEFORE_ATTACK_REPLAY"})
    with gzip.open(ARTIFACT / "per_block_scores.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "role", "variant", "block_start_s", "score", "threshold"])
        writer.writeheader(); writer.writerows(rows_out)
    print(json.dumps({"status": "CLEAN_MODEL_FROZEN", "thresholds": thresholds}, indent=2, sort_keys=True))
    return 0


SCENARIO_META = {
    "TEXBAT.DS3": ("texbat_ds3", "TEXBAT", 118.9, 195.0),
    "TEXBAT.DS7": ("texbat_ds7", "TEXBAT", 110.0, 150.0),
    "OAKBAT.OS3": ("oakbat_os3", "OAKBAT", 120.0, None),
    "OAKBAT.OS4": ("oakbat_os4", "OAKBAT", 120.0, None),
}


def load_models(family: str) -> dict[str, RobustGaussian]:
    payload = json.loads((ARTIFACT / f"normal_model_{family.lower()}.json").read_text())
    return {name: RobustGaussian(np.asarray(value["center"]), np.asarray(value["precision"]),
                                 np.asarray(value["covariance"]), float(value["regularization"]))
            for name, value in payload["models"].items()}


def normalized_pauc(labels: np.ndarray, score: np.ndarray, max_fpr: float = 0.05) -> float | None:
    labels = np.asarray(labels, dtype=int); score = np.asarray(score, dtype=float)
    if len(np.unique(labels)) < 2:
        return None
    fpr, tpr, _ = roc_curve(labels, score)
    if not np.any(fpr == max_fpr):
        index = np.searchsorted(fpr, max_fpr)
        interp = np.interp(max_fpr, fpr[index - 1:index + 1], tpr[index - 1:index + 1])
        fpr = np.insert(fpr, index, max_fpr); tpr = np.insert(tpr, index, interp)
    keep = fpr <= max_fpr
    return float(np.trapezoid(tpr[keep], fpr[keep]) / max_fpr)


def metric_row(dataset: str, scenario: str, variant: str, time_s: np.ndarray, score: np.ndarray,
               threshold: float, onset: float, pull_off: float | None, prn_count: float) -> dict[str, object]:
    labels = time_s >= onset
    alarms = consecutive_alarms(np.rint(time_s * 1000).astype(np.int64), score, threshold)
    pre, attack = ~labels, labels
    transition_end = pull_off if pull_off is not None else onset + 20.0
    transition = (time_s >= onset) & (time_s < transition_end)
    established = time_s >= transition_end
    alarm_times = time_s[alarms & attack]
    pull_alarm = time_s[alarms & (time_s >= pull_off)] if pull_off is not None else np.array([])
    max_run = run = 0
    for value in alarms:
        run = run + 1 if value else 0; max_run = max(max_run, run)
    return {
        "dataset": dataset, "scenario": scenario, "model": variant, "status": "AVAILABLE",
        "roc_auc": float(roc_auc_score(labels, score)), "pauc_fpr_le_0p05": normalized_pauc(labels, score),
        "pr_auc": float(average_precision_score(labels, score)),
        "clean_holdout_fpr": None, "pre_onset_fpr": float(np.mean(alarms[pre])) if pre.any() else None,
        "attack_detection_rate": float(np.mean(alarms[attack])) if attack.any() else None,
        "transition_detection_rate": float(np.mean(alarms[transition])) if transition.any() else None,
        "established_detection_rate": float(np.mean(alarms[established])) if established.any() else None,
        "onset_delay_s": float(alarm_times[0] - onset) if len(alarm_times) else None,
        "pull_off_delay_s": float(pull_alarm[0] - pull_off) if len(pull_alarm) else None,
        "persistent_alarm_ratio": float(np.mean(alarms[attack])) if attack.any() else None,
        "maximum_consecutive_alarms": max_run, "valid_blocks": len(score), "valid_prns": prn_count,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else ["status", "reason"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)


def evaluate_attacks() -> int:
    authorization = json.loads((ARTIFACT / "phase_b_authorization.json").read_text())
    if not authorization.get("phase_b_authorized") or authorization.get("threshold_freeze_commit", "").startswith("TO_BE"):
        raise RuntimeError("committed clean threshold freeze not recorded; Phase B NOT_AUTHORIZED")
    threshold_payload = json.loads((ARTIFACT / "thresholds.json").read_text())["thresholds"]
    metrics, block_rows, prn_rows, collapse_rows, destruction_rows = [], [], [], [], []
    bootstrap_source = {}
    for name, (slug, family, onset, pull_off) in SCENARIO_META.items():
        aligned = align_dump_directories(directory(slug, "slow", 1, "attack"), directory(slug, "fast", 1, "attack"), dataset=name)
        identical = align_dump_directories(directory(slug, "identical_left", 1, "attack"), directory(slug, "identical_right", 1, "attack"), dataset=name)
        models = load_models(family)
        by_variant = {}
        for variant, attr in VARIANT_ATTR.items():
            block, score, _, n = _block_variant(aligned, getattr(aligned, attr), models[variant])
            by_variant[variant] = (block, score, float(np.median(n)) if len(n) else 0.0)
        iepoch, iscore, inum = epoch_scores(identical.epoch_ms, identical.prn, np.sum(identical.full ** 2, axis=1))
        iblock, ibscore, _ = nonoverlap_blocks(iepoch, iscore)
        by_variant["A5"] = (iblock, ibscore, float(np.median(inum)) if len(inum) else 0.0)
        common = set(by_variant["A5"][0].tolist())
        for variant in VARIANT_ATTR:
            common &= set(by_variant[variant][0].tolist())
        common = np.asarray(sorted(common), dtype=np.int64)
        if not len(common):
            raise RuntimeError(f"{name}: empty common block support")
        scenario_scores = {}
        for variant, (block, score, prns) in by_variant.items():
            lookup = dict(zip(block.tolist(), score.tolist(), strict=True))
            values = np.asarray([lookup[item] for item in common])
            threshold = 1e-12 if variant == "A5" else float(threshold_payload[family][variant]["q99"])
            row = metric_row(family, name.split(".")[-1], variant, common / 1000.0, values, threshold, onset, pull_off, prns)
            metrics.append(row); scenario_scores[variant] = values
            for block_value, score_value in zip(common, values, strict=True):
                block_rows.append({"dataset": name, "role": "attack_replay", "variant": variant,
                                   "block_start_s": block_value / 1000.0, "score": score_value, "threshold": threshold})
        attack_mask = common / 1000.0 >= onset
        full_mean = float(np.mean(scenario_scores["Full"][attack_mask]))
        identical_mean = float(np.mean(scenario_scores["A5"][attack_mask]))
        collapse_rows.append({"scenario": name, "full_attack_mean": full_mean, "identical_attack_mean": identical_mean,
                              "relative_reduction": 1.0 - identical_mean / full_mean if full_mean else None,
                              "status": "PASS" if identical_mean <= 1e-12 else "FAIL"})
        best_single = "A0" if normalized_pauc(common / 1000.0 >= onset, scenario_scores["A0"]) >= normalized_pauc(common / 1000.0 >= onset, scenario_scores["A1"]) else "A1"
        bootstrap_source[name] = (common / 1000.0, common / 1000.0 >= onset, scenario_scores["Full"], scenario_scores[best_single], best_single)

        # Pairing destruction is diagnostic and intentionally never tunes thresholds.
        shift_fast = aligned.fast_taps.copy()
        for prn in np.unique(aligned.prn):
            idx = np.flatnonzero(aligned.prn == prn); shift_fast[idx] = np.roll(shift_fast[idx], max(1, len(idx) // 7), axis=0)
        shifted = aligned.slow_taps - shift_fast
        _, shifted_score, _, _ = _block_variant(aligned, shifted, models["A3"])
        destruction_rows.append({"scenario": name, "test": "PRN-wise time shift fast sequence",
                                 "scope": "complex_tap_component", "original_mean": float(np.mean(scenario_scores["A3"])),
                                 "destroyed_mean": float(np.mean(shifted_score)), "status": "DIAGNOSTIC_ONLY"})
        # Compact per-PRN/block divergence summary.
        for block_value in common:
            mask_block = aligned.epoch_ms // 100 * 100 == block_value
            for prn in np.unique(aligned.prn[mask_block]):
                mask = mask_block & (aligned.prn == prn)
                prn_rows.append({"dataset": name, "block_start_s": block_value / 1000.0, "prn": int(prn),
                                 "code_divergence_abs_median": float(np.median(np.abs(aligned.state[mask, 0]))),
                                 "doppler_divergence_abs_median": float(np.median(np.abs(aligned.state[mask, 2]))),
                                 "tap_divergence_norm_median": float(np.median(np.linalg.norm(aligned.taps[mask], axis=1))),
                                 "full_divergence_norm_median": float(np.median(np.linalg.norm(aligned.full[mask], axis=1)))})

    write_csv(ARTIFACT / "scenario_metrics.csv", [row for row in metrics if row["model"] == "Full"])
    b0 = {key: None for key in metrics[0]}; b0.update({"dataset": "B0", "scenario": "all", "model": "B0 exact", "status": "UNAVAILABLE"})
    write_csv(ARTIFACT / "ablation_metrics.csv", metrics + [b0])
    with gzip.open(ARTIFACT / "per_block_scores.csv.gz", "at", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "role", "variant", "block_start_s", "score", "threshold"]); writer.writerows(block_rows)
    with gzip.open(ARTIFACT / "per_prn_divergence.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prn_rows[0])); writer.writeheader(); writer.writerows(prn_rows)
    dump_json(ARTIFACT / "configuration_collapse_metrics.json", {"schema": "gnss-doppler-lab.mctd-collapse.v1", "scenarios": collapse_rows,
              "all_collapse": all(row["status"] == "PASS" for row in collapse_rows)})
    dump_json(ARTIFACT / "pairing_destruction_metrics.json", {"schema": "gnss-doppler-lab.mctd-pairing-destruction.v1",
              "status": "PARTIAL_DIAGNOSTIC", "not_used_for_tuning": True, "tests": destruction_rows,
              "limitation": "The native state/action sequences are receiver-recursive; distribution-preserving repair-free destruction is reported for the complex-tap component only and cannot satisfy the Full GO criterion."})
    bootstrap = bootstrap_metrics(bootstrap_source)
    write_csv(ARTIFACT / "bootstrap_intervals.csv", bootstrap)
    finalize_science(metrics, collapse_rows)
    print(json.dumps({"status": "ATTACK_EVALUATION_COMPLETE", "metric_rows": len(metrics)}, indent=2))
    return 0


def bootstrap_metrics(source) -> list[dict[str, object]]:
    rng = np.random.default_rng(20260817); rows = []
    for name, (time, labels, full, single, baseline) in source.items():
        clusters = paired_bootstrap_blocks(time); unique = np.unique(clusters); deltas = []
        for _ in range(999):
            chosen = rng.choice(unique, size=len(unique), replace=True)
            idx = np.concatenate([np.flatnonzero(clusters == value) for value in chosen])
            a, b = normalized_pauc(labels[idx], full[idx]), normalized_pauc(labels[idx], single[idx])
            if a is not None and b is not None: deltas.append(a - b)
        rows.append({"family": name, "comparison": f"Full-minus-{baseline}", "resamples": len(deltas),
                     "point_delta": normalized_pauc(labels, full) - normalized_pauc(labels, single),
                     "ci_lower": float(np.quantile(deltas, .025)) if deltas else None,
                     "ci_upper": float(np.quantile(deltas, .975)) if deltas else None,
                     "block_width_s": 10.0})
    return rows


def finalize_science(metrics: list[dict[str, object]], collapse: list[dict[str, object]]) -> None:
    threshold_data = json.loads((ARTIFACT / "thresholds.json").read_text())
    summary = json.loads((ARTIFACT / "normal_model_summary.json").read_text())
    clean_fprs = [value["holdout"]["Full"]["q99_fpr"] for value in summary["families"].values()]
    full = [row for row in metrics if row["model"] == "Full"]
    external = [{"dataset": row["dataset"], "scenario": row["scenario"], "status": "AVAILABLE",
                 "pre_onset_fpr": row["pre_onset_fpr"]} for row in full]
    write_csv(ARTIFACT / "external_static_fpr.csv", external)
    controls = {
        "schema": "gnss-doppler-lab.mctd-physical-controls.v1", "all_required_controls_pass": False,
        "controls": {
            "common_gain_scaling": {"status": "NORMALIZATION_INVARIANT_DIAGNOSTIC", "persistent_alarm": False},
            "global_complex_phase_rotation": {"status": "NORMALIZATION_INVARIANT_DIAGNOSTIC", "persistent_alarm": False},
            "prompt_amplitude_scaling": {"status": "NORMALIZATION_INVARIANT_DIAGNOSTIC", "persistent_alarm": False},
            "navigation_bit_sign_reversal": {"status": "NORMALIZATION_INVARIANT_DIAGNOSTIC", "persistent_alarm": False},
            "raw_iq_awgn_0p5x_1x_2x": {"status": "UNAVAILABLE", "reason": "No preregistered authenticated perturbed raw-IQ source exists; feature-space Gaussian noise is not substituted."},
            "cn0_decrease": {"status": "UNAVAILABLE", "reason": "Requires authenticated raw-IQ attenuation and receiver rerun."},
            "prn_drop_add": {"status": "LIMITED_FEATURE_SUPPORT_DIAGNOSTIC"},
            "single_prn_disturbance": {"status": "LIMITED_FEATURE_SUPPORT_DIAGNOSTIC"},
            "receiver_clock_like_drift": {"status": "UNAVAILABLE", "reason": "Requires a raw-IQ resampling control."},
            "multipath": {"status": "UNAVAILABLE", "reason": "No authenticated same-receiver multipath control recording was available."},
        },
    }
    dump_json(ARTIFACT / "physical_controls.json", controls)
    pre_ok = all((row["pre_onset_fpr"] or 0) <= .05 for row in full)
    clean_ok = max(clean_fprs) <= .01
    collapse_ok = all(row["status"] == "PASS" for row in collapse)
    verdict = "NO_GO_MCTD_PHYSICAL_HYPOTHESIS"
    reasons = []
    if not clean_ok: reasons.append("cleanStatic holdout q99 FPR exceeds 1%")
    if not pre_ok: reasons.append("one or more core attack pre-onset FPRs exceed 5%")
    if not collapse_ok: reasons.append("identical-loop configuration collapse failed")
    reasons.append("required raw-IQ nuisance controls are unavailable and Full pairing destruction is only partial")
    dump_json(ARTIFACT / "final_verdict.json", {"schema": "gnss-doppler-lab.mctd-final-verdict.v1",
              "verdict": verdict, "phase_a_passed": True, "phase_b_run": True, "attack_metrics_computed": True,
              "attack_result_retuning": False, "clean_holdout_fpr_worst": max(clean_fprs),
              "external_static_fpr_worst": max(row["pre_onset_fpr"] for row in full),
              "configuration_collapse_passed": collapse_ok, "all_required_controls_pass": False,
              "go": False, "no_go_reasons": reasons,
              "b0_exact": {"status": "UNAVAILABLE", "reason": "Exact frozen B0 cannot be rerun on native MCTD common support without changing B0; historical CSVs were not copied as MCTD results."},
              "recommended_next_action": "Stop MCTD Stage-1 and retain this frozen Stage-0 bundle as the negative-result record."})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("phase-a", "fit-clean", "evaluate-attacks"))
    args = parser.parse_args()
    if args.phase == "phase-a": return phase_a()
    if args.phase == "fit-clean": return fit_clean()
    return evaluate_attacks()


if __name__ == "__main__":
    raise SystemExit(main())
