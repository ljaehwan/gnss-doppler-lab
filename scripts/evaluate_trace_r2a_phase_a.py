#!/usr/bin/env python3
"""Evaluate preregistered TRACE-R2a source, causal, and semantic gates."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.trace_equivariance import ResidualModel, fit_ridge, robust_epoch_blocks, consecutive_alarm
from gnss_doppler_lab.trace_native_1ms import load_native_trace_pairs, validate_dump_files
from gnss_doppler_lab.trace_reproducibility import (
    METADATA_FIELDS,
    PHYSICAL_FIELDS,
    canonical_join,
    canonical_semantic_hash,
    common_epoch_count,
    exact_equal,
    field_statistics,
    load_replay,
)

ARTIFACT = ROOT / "artifacts/trace_stage0_r2a_reproducibility_repair"
SSD = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2a-reproducibility-repair")
SCENARIOS = {
    "TEXBAT.cleanStatic.rep3": ("TEXBAT.cleanStatic", "texbat_cleanstatic", 3, 25_000_000),
    "TEXBAT.cleanStatic.rep4": ("TEXBAT.cleanStatic", "texbat_cleanstatic", 4, 25_000_000),
    "TEXBAT.DS3.smoke": ("TEXBAT.DS3", "texbat_ds3", 3, 25_000_000),
    "OAKBAT.OS3.smoke": ("OAKBAT.OS3", "oakbat_os3", 3, 5_000_000),
}
COMMON = np.arange(1, 8)


def dump_json(name: str, value: object) -> None:
    path = ARTIFACT / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def replay_dir(key: str) -> Path:
    _, slug, repetition, _ = SCENARIOS[key]
    return SSD / "dumps/phase_a" / slug / f"rep{repetition}"


def source_gate(first_manifest: dict, second_manifest: dict, build: dict, preregistration: dict) -> dict[str, object]:
    frozen = preregistration["source_hashes"]
    expected_config = frozen["receiver_configs"]["phase_a"]["TEXBAT.cleanStatic"]["sha256"]
    expected_handoff = frozen["handoffs"]["TEXBAT.cleanStatic"]
    checks = {
        "raw_iq_sha256_identical": first_manifest["raw_iq"]["sha256"] == second_manifest["raw_iq"]["sha256"],
        "receiver_executable_sha256_identical": first_manifest["receiver_executable"]["sha256"] == second_manifest["receiver_executable"]["sha256"] == build["receiver_executable"]["sha256"],
        "receiver_config_sha256_identical_and_frozen": first_manifest["receiver_config_sha256"] == second_manifest["receiver_config_sha256"] == expected_config,
        "receiver_patch_sha256_identical": first_manifest["receiver_patch_sha256"] == second_manifest["receiver_patch_sha256"] == build["receiver_repair_diff"]["sha256"],
        "raw_sample_range_identical": first_manifest["raw_sample_range"] == second_manifest["raw_sample_range"],
        "frozen_handoff_sha256_identical_and_frozen": first_manifest["frozen_handoff_sha256"] == second_manifest["frozen_handoff_sha256"] == expected_handoff,
        "raw_and_receiver_stable_during_both_runs": all(first_manifest[key] and second_manifest[key] for key in ("raw_iq_stable_during_run", "receiver_stable_during_run")),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def score_reproduction(first_dir: Path, second_dir: Path) -> dict[str, object]:
    first = load_native_trace_pairs(first_dir, cn0_min_db_hz=28.0, lock_min=0.85)
    second = load_native_trace_pairs(second_dir, cn0_min_db_hz=28.0, lock_min=0.85)
    first_frame = pd.DataFrame({"prn": first.prn, "sample": first.sample_count, "i": np.arange(len(first.prn))})
    second_frame = pd.DataFrame({"prn": second.prn, "sample": second.sample_count, "j": np.arange(len(second.prn))})
    if first_frame.duplicated(["prn", "sample"]).any() or second_frame.duplicated(["prn", "sample"]).any():
        raise ValueError("duplicate score-alignment key")
    joined = first_frame.merge(second_frame, on=["prn", "sample"], validate="one_to_one")
    i = joined["i"].to_numpy(dtype=np.int64)
    j = joined["j"].to_numpy(dtype=np.int64)
    tmin, tmax = float(first.time_s.min()), float(first.time_s.max())
    span = tmax - tmin
    train_mask = first.time_s < tmin + 0.60 * span
    covariance_mask = (first.time_s >= tmin + 0.60 * span) & (first.time_s < tmin + 0.80 * span)
    evaluation_mask = first.time_s >= tmin + 0.80 * span
    model = fit_ridge(first.take(train_mask), include_action=True, target_mode="warp_residual", alpha=10.0, output_indices=COMMON)
    covariance_pairs = first.take(covariance_mask)
    residual_model = ResidualModel.fit(covariance_pairs.target[:, COMMON] - model.predict(covariance_pairs)[:, COMMON])
    score1 = residual_model.score(first.target[:, COMMON] - model.predict(first)[:, COMMON])
    score2 = residual_model.score(second.target[:, COMMON] - model.predict(second)[:, COMMON])
    differences = np.abs(score1[i] - score2[j])
    blocks1 = robust_epoch_blocks(first, score1, block_s=0.5, minimum_prns=4)
    blocks2 = robust_epoch_blocks(second, score2, block_s=0.5, minimum_prns=4)
    threshold_values = blocks1[blocks1["block_start_s"] >= tmin + 0.80 * span]["score"]
    threshold = float(np.quantile(threshold_values, 0.99))
    alarm1 = consecutive_alarm(blocks1["block_start_s"], blocks1["score"], threshold, 3)
    alarm2 = consecutive_alarm(blocks2["block_start_s"], blocks2["score"], threshold, 3)
    block_frame1 = pd.DataFrame({name: blocks1[name] for name in blocks1.dtype.names}).rename(columns={"score": "score_rep3"})
    block_frame2 = pd.DataFrame({name: blocks2[name] for name in blocks2.dtype.names}).rename(columns={"score": "score_rep4"})
    block_join = block_frame1.merge(
        block_frame2,
        on=["block_start_s", "tracked_prn_count", "pair_count"],
        how="inner",
        validate="one_to_one",
    )
    block_keys_equal = len(block_join) == len(blocks1) == len(blocks2)
    block_score_error = float(np.max(np.abs(block_join["score_rep3"] - block_join["score_rep4"]))) if len(block_join) else None
    return {
        "rep3_pair_count": int(len(first.prn)),
        "rep4_pair_count": int(len(second.prn)),
        "common_pair_count": int(len(joined)),
        "common_pair_ratio_rep3": float(len(joined) / len(first.prn)),
        "common_pair_ratio_rep4": float(len(joined) / len(second.prn)),
        "maximum_common_pair_trace_score_absolute_error": float(differences.max()) if len(differences) else None,
        "trace_score_tolerance": 1e-12,
        "trace_score_within_tolerance": bool(len(differences) and differences.max() <= 1e-12),
        "block_keys_identical": bool(block_keys_equal),
        "maximum_block_trace_score_absolute_error": block_score_error,
        "common_block_count": int(len(block_join)),
        "q99_threshold_from_rep3": threshold,
        "threshold_crossing_and_alarm_identical": bool(block_keys_equal and np.array_equal(alarm1, alarm2)),
        "train_pair_count": int(train_mask.sum()),
        "covariance_pair_count": int(covariance_mask.sum()),
        "evaluation_pair_count": int(evaluation_mask.sum()),
        "rep3_blocks": blocks1,
        "rep4_blocks": blocks2,
        "common_blocks": block_join,
    }


def main() -> int:
    manifests = {key: json.loads((replay_dir(key) / "manifest.json").read_text()) for key in SCENARIOS}
    validations = {
        key: validate_dump_files(sorted(replay_dir(key).glob("trace_native_1ms_ch_*.bin")), expected_scenario_id=value[0], minimum_prns=4)
        for key, value in SCENARIOS.items()
    }
    rep3 = load_replay(replay_dir("TEXBAT.cleanStatic.rep3"), "TEXBAT.cleanStatic")
    rep4 = load_replay(replay_dir("TEXBAT.cleanStatic.rep4"), "TEXBAT.cleanStatic")
    joined = canonical_join(rep3, rep4)
    i = joined["rep1_row_index"].to_numpy(dtype=np.int64)
    j = joined["rep2_row_index"].to_numpy(dtype=np.int64)
    statistics, exact_rows = field_statistics(rep3.records[i], rep4.records[j])
    metadata_equal = np.ones(len(joined), dtype=bool)
    for field in METADATA_FIELDS:
        metadata_equal &= exact_equal(rep3.records[field][i], rep4.records[field][j])
    score = score_reproduction(replay_dir("TEXBAT.cleanStatic.rep3"), replay_dir("TEXBAT.cleanStatic.rep4"))
    blocks3 = score.pop("rep3_blocks")
    blocks4 = score.pop("rep4_blocks")
    common_blocks = score.pop("common_blocks")
    build = json.loads((ARTIFACT / "receiver_build_manifest.json").read_text())
    preregistration = json.loads((ARTIFACT / "preregistration.json").read_text())
    source = source_gate(manifests["TEXBAT.cleanStatic.rep3"], manifests["TEXBAT.cleanStatic.rep4"], build, preregistration)
    causal_counts = {
        "sequence_mismatch": sum(value["causal_sequence_mismatch_count"] for value in validations.values()),
        "action_value_mapping_mismatch": sum(value["causal_value_mismatch_count"] for value in validations.values()),
        "consume_span_mismatch": sum(value["consume_span_mismatch_count"] for value in validations.values()),
        "finite_failure": sum(value["finite_failure_count"] for value in validations.values()),
        "zero_tap": sum(value["zero_tap_row_count"] for value in validations.values()),
        "zero_action": sum(value["zero_action_row_count"] for value in validations.values()),
        "repeated_tap": sum(value["repeated_tap_row_count"] for value in validations.values()),
        "repeated_action": sum(value["repeated_action_row_count"] for value in validations.values()),
    }
    causal = {"status": "PASS" if all(value == 0 for value in causal_counts.values()) else "FAIL", "counts": causal_counts}
    common_epochs = common_epoch_count(joined, 25_000_000, minimum_prns=4)
    semantic_checks = {
        "common_epoch_count_at_least_1000": common_epochs >= 1000,
        "all_common_physical_rows_bit_exact": bool(len(exact_rows) and exact_rows.all()),
        "canonical_semantic_hash_identical": canonical_semantic_hash(rep3) == canonical_semantic_hash(rep4),
        "trace_score_within_1e_12": score["trace_score_within_tolerance"],
        "threshold_crossing_and_alarm_identical": score["threshold_crossing_and_alarm_identical"],
    }
    semantic = {
        "status": "PASS" if all(semantic_checks.values()) else "FAIL",
        "checks": semantic_checks,
        "common_canonical_row_count": int(len(joined)),
        "common_ratio_rep3": float(len(joined) / len(rep3.records)),
        "common_ratio_rep4": float(len(joined) / len(rep4.records)),
        "common_at_least_4_prn_rounded_ms_epoch_count": common_epochs,
        "exact_physical_bit_match_ratio": float(exact_rows.mean()) if len(exact_rows) else None,
        "metadata_exact_match_ratio": float(metadata_equal.mean()) if len(metadata_equal) else None,
        "metadata_only_difference_proven": bool((~metadata_equal).any() and exact_rows[~metadata_equal].all()),
        "field_statistics": statistics,
        "score_reproduction": score,
    }
    support = {
        key: {
            "status": value["status"],
            "maximum_valid_prns_same_rounded_ms_epoch": value["maximum_valid_prns_same_rounded_ms_epoch"],
            "native_cadence_fraction": value["native_cadence_fraction"],
            "causal_pair_count": value["causal_pair_count"],
        }
        for key, value in validations.items()
    }
    support_pass = all(value["status"] == "PASS" and value["maximum_valid_prns_same_rounded_ms_epoch"] >= 4 for value in validations.values())
    raw_binding = json.loads((ARTIFACT / "raw_source_binding.json").read_text())
    raw_timeline = {}
    for key, manifest in manifests.items():
        scenario_name = SCENARIOS[key][0]
        bound = raw_binding["datasets"][scenario_name]
        replay = rep3 if key == "TEXBAT.cleanStatic.rep3" else rep4 if key == "TEXBAT.cleanStatic.rep4" else load_replay(replay_dir(key), scenario_name)
        observed_start = int(replay.records["raw_interval_start_sample"].min())
        observed_end = int(replay.records["raw_interval_end_sample"].max())
        declared = manifest["raw_sample_range"]
        checks = {
            "fresh_bound_sha_matches_replay": bound["fresh_sha256"] == manifest["raw_iq"]["sha256"],
            "source_stable_during_preregistration_hash": bound["stable_during_hash"],
            "source_stable_during_replay": manifest["raw_iq_stable_during_run"],
            "raw_interval_is_within_manifest_range": declared["start_inclusive"] <= observed_start and observed_end <= declared["end_exclusive"],
        }
        raw_timeline[key] = {
            "raw_sha256": manifest["raw_iq"]["sha256"],
            "raw_sample_range": manifest["raw_sample_range"],
            "observed_dump_raw_interval": {"start_inclusive": observed_start, "end_exclusive": observed_end},
            "checks": checks,
            "status": "PASS" if all(checks.values()) else "FAIL",
        }
    raw_timeline_pass = raw_binding["status"] == "PASS" and all(row["status"] == "PASS" for row in raw_timeline.values())
    phase_a_pass = source["status"] == causal["status"] == semantic["status"] == "PASS" and support_pass and raw_timeline_pass
    payload = {
        "schema": "gnss-doppler-lab.trace-r2a-phase-a-semantic-reproduction.v1",
        "phase_a_status": "PASS" if phase_a_pass else "FAIL",
        "phase_b_authorized": phase_a_pass,
        "source_gate": source,
        "causal_gate": causal,
        "semantic_reproduction_gate": semantic,
        "scenario_support": support,
        "raw_source_timeline_binding": {"status": "PASS" if raw_timeline_pass else "FAIL", "scenarios": raw_timeline},
        "failure_verdict_if_any": None if phase_a_pass else "INCONCLUSIVE_RECEIVER_REPRODUCIBILITY",
    }
    dump_json("rep3_rep4_reproduction_metrics.json", payload)
    dump_json("action_mapping_validation.json", {"schema": "gnss-doppler-lab.trace-r2a-action-mapping.v1", "status": causal["status"], "scenario_validations": validations, "combined_counts": causal_counts})
    dump_json("replay_inventory.json", {"schema": "gnss-doppler-lab.trace-r2a-replay-inventory.v1", "phase_a": {key: {"manifest_path": str(replay_dir(key) / "manifest.json"), "manifest": value, "validation": validations[key]} for key, value in manifests.items()}, "phase_a_decision": {"status": payload["phase_a_status"], "phase_b_authorized": phase_a_pass}})
    plots = ARTIFACT / "plots"
    plots.mkdir(exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(common_blocks["score_rep3"], common_blocks["score_rep4"], s=12)
    low = min(common_blocks["score_rep3"].min(), common_blocks["score_rep4"].min())
    high = max(common_blocks["score_rep3"].max(), common_blocks["score_rep4"].max())
    axis.plot([low, high], [low, high], color="black", linewidth=0.8)
    axis.set(xlabel="rep3 frozen TRACE block score", ylabel="rep4 frozen TRACE block score", title="rep3/rep4 semantic reproducibility")
    fig.tight_layout()
    fig.savefig(plots / "rep3_rep4_semantic_reproducibility.png", dpi=140)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 3.5))
    axis.plot(blocks3["block_start_s"], blocks3["score"], label="rep3")
    axis.plot(blocks4["block_start_s"], blocks4["score"], label="rep4", linestyle="--")
    axis.legend()
    axis.set(xlabel="receiver time (s)", ylabel="frozen TRACE block score", title="Replay-by-replay TRACE score")
    fig.tight_layout()
    fig.savefig(plots / "replay_by_replay_trace_score_comparison.png", dpi=140)
    plt.close(fig)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if phase_a_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
