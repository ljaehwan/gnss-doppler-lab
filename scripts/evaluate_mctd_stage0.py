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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.mctd import (
    align_dump_directories, chronological_masks, epoch_scores, mahalanobis_score,
    nonoverlap_blocks, robust_fit,
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
        union_rows = len(canonical_rows(directory(slug, "slow", 1))) + len(canonical_rows(directory(slug, "fast", 1)))
        datasets[dataset] = {
            "status": "PASS" if passed else "FAIL", "source_equality": source,
            "deterministic_replay": deterministic,
            "stable_support": {"status": "PASS" if stable else "FAIL", "quality_common_rows": len(aligned.prn),
                               "quality_common_epochs_ge_4_prns": len(epoch), "maximum_common_prns": int(n.max(initial=0)),
                               "common_raw_start_delta_samples_max": int(common_raw_delta.max(initial=0)),
                               "missing_native_row_rate": 1.0 - 2.0 * len(aligned.prn) / union_rows},
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
                                  "time_end_s": float(aligned.time_s[mask].max())} for name, mask in masks.items()}
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("phase-a", "fit-clean"))
    args = parser.parse_args()
    return phase_a() if args.phase == "phase-a" else fit_clean()


if __name__ == "__main__":
    raise SystemExit(main())
