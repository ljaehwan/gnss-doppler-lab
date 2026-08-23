#!/usr/bin/env python3
"""Run the bounded Jammertest 2025 CRPA Stage-0B validation."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gnss_doppler_lab.jammertest_crpa_stage0b import (
    BLOCK_SIZES,
    EXPECTED_BYTES,
    EXPECTED_SHA256,
    PAIR_INDICES,
    SEED,
    FeatureSet,
    block_bootstrap_mean_difference,
    channel_quality,
    circular_shift_batch,
    compute_features,
    expected_reader_view,
    load_label_rows,
    mismatch_batch,
    mismatch_source_map,
    observe_npy_schema,
    phase_randomize_batch,
    public_reader_view,
    sha256_file,
    write_csv,
    write_json,
    write_manifest,
)
from gnss_doppler_lab.jammertest_crpa_split_audit import assess_blocked_split_feasibility


CONTROL_NAMES = ("actual", "mismatched", "circular_shift", "fourier_phase_randomized")


def concatenate(parts: list[FeatureSet]) -> FeatureSet:
    return FeatureSet(**{
        name: np.concatenate([getattr(part, name) for part in parts], axis=0)
        for name in FeatureSet.__dataclass_fields__
    })


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def feature_summary(features: dict[str, FeatureSet], rows: list[dict]) -> dict:
    groups: dict[tuple, list[int]] = {}
    for position, row in enumerate(rows):
        key = (row["area"], row["class_name"], row["transmit_power_dbm"])
        groups.setdefault(key, []).append(position)
    result: dict[str, object] = {"normalization": "per-snapshot, per-channel centered RMS; covariance diagonal normalized to one"}
    for control, values in features.items():
        control_rows = []
        for (area, class_name, power), positions in sorted(groups.items()):
            select = np.asarray(positions, dtype=np.int64)
            control_rows.append(
                {
                    "area": area,
                    "class": class_name,
                    "transmit_power_dbm": power,
                    "count": len(positions),
                    "mean_coherence": summarize(values.mean_coherence[select]),
                    "lambda1_trace": summarize(values.lambda1_fraction[select]),
                    "effective_rank": summarize(values.effective_rank[select]),
                    "condition_number": summarize(values.condition_number[select]),
                }
            )
        result[control] = {
            "overall": {
                "count": len(values.mean_coherence),
                "mean_coherence": summarize(values.mean_coherence),
                "lambda1_trace": summarize(values.lambda1_fraction),
                "effective_rank": summarize(values.effective_rank),
                "condition_number": summarize(values.condition_number),
            },
            "by_area_class_power": control_rows,
        }
    return result


def lag_rows(features: dict[str, FeatureSet], rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[int]] = {}
    for position, row in enumerate(rows):
        key = (row["area"], row["class_name"], row["transmit_power_dbm"])
        groups.setdefault(key, []).append(position)
    output = []
    for control, values in features.items():
        for (area, class_name, power), positions in sorted(groups.items()):
            select = np.asarray(positions, dtype=np.int64)
            for pair_no, (left, right) in enumerate(PAIR_INDICES):
                lags = values.peak_lags[select, pair_no]
                coherence = values.peak_lag_coherence[select, pair_no]
                counts = Counter(int(value) for value in lags)
                mode_lag, mode_count = min(counts.items(), key=lambda item: (-item[1], item[0]))
                output.append(
                    {
                        "control": control,
                        "area": area,
                        "class": class_name,
                        "transmit_power_dbm": power,
                        "channel_pair": f"{left}-{right}",
                        "count": len(select),
                        "mode_peak_lag": mode_lag,
                        "mode_fraction": mode_count / len(select),
                        "zero_lag_fraction": float(np.mean(lags == 0)),
                        "mean_peak_coherence": float(np.mean(coherence)),
                        "median_peak_coherence": float(np.median(coherence)),
                        "q05_peak_coherence": float(np.quantile(coherence, 0.05)),
                        "q95_peak_coherence": float(np.quantile(coherence, 0.95)),
                    }
                )
    return output


def build_figures(output: Path, features: dict[str, FeatureSet]) -> None:
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    plt.boxplot(
        [features[name].mean_coherence for name in CONTROL_NAMES],
        tick_labels=["actual", "mismatch", "shift", "phase-rand"],
        showfliers=False,
    )
    plt.ylabel("mean |pairwise coherence|")
    plt.title("Normalized spatial coherence destruction controls")
    plt.tight_layout()
    plt.savefig(figures / "coherence_destruction_controls.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.8))
    for name in CONTROL_NAMES:
        values = np.sort(features[name].lambda1_fraction)
        plt.plot(values, np.linspace(0, 1, len(values), endpoint=False), label=name)
    plt.xlabel("largest eigenvalue / trace")
    plt.ylabel("empirical CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "lambda1_fraction_ecdf.png", dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-npy", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    raw_path = args.raw_npy.resolve()
    split_root = args.split_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)

    actual_size = raw_path.stat().st_size
    actual_sha = sha256_file(raw_path)
    if actual_size != EXPECTED_BYTES or actual_sha != EXPECTED_SHA256:
        raise SystemExit("bounded CRPA object integrity mismatch")
    write_json(
        output / "download_integrity.json",
        {
            "official_repository": "https://github.com/FelixOtt94/FraunhoferIIS_Jammertest2025",
            "tag": "dataset",
            "commit": "3b778a12147ded5c86c3edfc586b5de6ae6a67d7",
            "object_path": "all_crpa_files.npy",
            "download_url": args.source_url,
            "expected_size_bytes": EXPECTED_BYTES,
            "actual_size_bytes": actual_size,
            "expected_sha256": EXPECTED_SHA256,
            "actual_sha256": actual_sha,
            "downloaded_lfs_object_count": 1,
            "integrity_passed": True,
        },
    )

    array, schema = observe_npy_schema(raw_path)
    sample_checks = []
    for index in (0, 1, 17, 10_000, 20_000, 30_000, 42_672):
        official = public_reader_view(array[index])
        expected = expected_reader_view(array[index])
        sample_checks.append({"sample_index": index, "exact_equal": bool(np.array_equal(official, expected))})
    schema["official_reader_conversion"] = {
        "observed_output_shape": [8, 1_024],
        "axis_order": ["ch0_I", "ch0_Q", "ch1_I", "ch1_Q", "ch2_I", "ch2_Q", "ch3_I", "ch3_Q"],
        "sample_checks": sample_checks,
        "all_exact": all(item["exact_equal"] for item in sample_checks),
    }
    write_json(output / "npy_schema_observed.json", schema)

    if not schema["schema_valid"]:
        write_json(output / "classifier_not_run.json", {"reason": "STOP_SCHEMA_INVALID"})
        write_json(
            output / "final_verdict.json",
            {
                "verdict": "STOP_SCHEMA_INVALID",
                "task": "spoof_meacon_vs_non_deceptive_terrestrial_jammer",
                "spatial_gate_passed": False,
                "classification_run": False,
                "ready_for_wcl_declared": False,
            },
        )
        raise SystemExit("STOP_SCHEMA_INVALID")

    rows = load_label_rows(split_root)
    sample_indices = np.asarray([row["sample_index"] for row in rows], dtype=np.int64)
    if sample_indices.min() < 0 or sample_indices.max() >= array.shape[0]:
        raise SystemExit("label sample index outside NPY bounds")
    class_counts = Counter((row["area"], row["class_name"]) for row in rows)
    positive = sum("Spoof" in row["class_name"] or "Meac" in row["class_name"] for row in rows)
    write_json(
        output / "label_binding_audit.json",
        {
            "split_root_sha256": {
                path.name: sha256_file(path)
                for path in sorted(split_root.glob("*_crpa_*.txt"))
            },
            "row_count": len(rows),
            "unique_sample_index_count": len(set(sample_indices.tolist())),
            "minimum_sample_index": int(sample_indices.min()),
            "maximum_sample_index": int(sample_indices.max()),
            "npy_snapshot_count": int(array.shape[0]),
            "unlabelled_or_unreleased_split_indices": int(array.shape[0] - len(rows)),
            "all_indices_in_bounds": True,
            "spoof_or_meacon_count": positive,
            "non_deceptive_jammer_count": len(rows) - positive,
            "clean_count": 0,
            "counts_by_area_class": [
                {"area": area, "class": class_name, "count": count}
                for (area, class_name), count in sorted(class_counts.items())
            ],
            "public_split_is_not_used_as_scientific_random_split": True,
        },
    )

    quality = channel_quality(array, batch_size=args.batch_size)
    quality["all_values_finite"] = all(item["finite_ratio"] == 1.0 for item in quality["channels"])
    write_json(output / "channel_quality.json", quality)

    # One sequentially sorted copy of the released labeled snapshots limits
    # repeated remote-SSD reads.  All controls below mutate only batch copies.
    data = np.asarray(array[sample_indices]).copy()
    del array
    mapping = mismatch_source_map(rows)
    part_sets: dict[str, list[FeatureSet]] = {name: [] for name in CONTROL_NAMES}
    shift_rng = np.random.default_rng(SEED + 1)
    phase_rng = np.random.default_rng(SEED + 2)
    for start in range(0, len(rows), args.batch_size):
        stop = min(start + args.batch_size, len(rows))
        positions = np.arange(start, stop)
        actual = data[start:stop]
        variants = {
            "actual": actual,
            "mismatched": mismatch_batch(data, mapping, positions),
            "circular_shift": circular_shift_batch(actual, shift_rng),
            "fourier_phase_randomized": phase_randomize_batch(actual, phase_rng),
        }
        for name, batch in variants.items():
            part_sets[name].append(compute_features(batch))
    features = {name: concatenate(parts) for name, parts in part_sets.items()}

    spatial_summary = feature_summary(features, rows)
    write_json(output / "spatial_metrics_summary.json", spatial_summary)
    write_csv(
        output / "lag_coherence_results.csv",
        [
            "control", "area", "class", "transmit_power_dbm", "channel_pair", "count",
            "mode_peak_lag", "mode_fraction", "zero_lag_fraction", "mean_peak_coherence",
            "median_peak_coherence", "q05_peak_coherence", "q95_peak_coherence",
        ],
        lag_rows(features, rows),
    )

    comparisons = []
    for control in CONTROL_NAMES[1:]:
        for block_size in BLOCK_SIZES:
            item = block_bootstrap_mean_difference(
                features["actual"].mean_coherence,
                features[control].mean_coherence,
                sample_indices,
                block_size,
                seed=SEED + CONTROL_NAMES.index(control) * 10_000,
            )
            item["control"] = control
            comparisons.append(item)

    global_subset = np.asarray(data[: min(512, len(data))], dtype=np.complex128)
    global_actual = compute_features(global_subset)
    global_changed = compute_features(global_subset * (3.7 * np.exp(1j * 1.123)))
    global_invariance = {
        "gain": 3.7,
        "phase_radians": 1.123,
        "max_abs_spatial_feature_difference": float(np.max(np.abs(global_actual.spatial - global_changed.spatial))),
        "max_abs_mean_coherence_difference": float(np.max(np.abs(global_actual.mean_coherence - global_changed.mean_coherence))),
        "invariant_within_1e-12": bool(np.max(np.abs(global_actual.spatial - global_changed.spatial)) <= 1e-12),
    }
    control_passes = {
        control: all(
            item["actual_significantly_higher"]
            for item in comparisons
            if item["control"] == control
        )
        for control in CONTROL_NAMES[1:]
    }
    all_decisions = [item["actual_significantly_higher"] for item in comparisons]
    coherence_destruction_gate_passed = all(all_decisions)
    spatial_gate_passed = (
        coherence_destruction_gate_passed
        and global_invariance["invariant_within_1e-12"]
    )
    block_decision_flips = any(all_decisions) and not all(all_decisions)
    write_json(
        output / "destruction_control_results.json",
        {
            "seed": SEED,
            "primary_metric": "per-snapshot mean magnitude of six zero-lag normalized pairwise coherences",
            "comparisons": comparisons,
            "control_all_block_sizes_pass": control_passes,
            "global_gain_phase_invariance": global_invariance,
            "coherence_destruction_gate_passed": coherence_destruction_gate_passed,
            "spatial_gate_passed": spatial_gate_passed,
            "block_decision_flips": block_decision_flips,
            "mismatched_contract": "each channel is a different deterministic permutation within identical Area/class/power stratum",
            "circular_shift_contract": "independent per-channel circular shifts, fixed seed",
            "phase_randomization_contract": "independent Fourier-bin phases per channel, fixed seed; channel power spectrum preserved",
        },
    )

    split_audits = [assess_blocked_split_feasibility(rows, size) for size in BLOCK_SIZES]
    split_feasible = all(item["balanced_block_disjoint_split_feasible"] for item in split_audits)
    write_json(
        output / "blocked_split_manifest.json",
        {
            "block_sizes": list(BLOCK_SIZES),
            "frozen_area": 1,
            "frozen_power_dbm": [15, 25, 30, 35, 40],
            "positive": "Spoof + Meac (including mixed labels containing either token)",
            "negative": "all labels without Spoof or Meac",
            "audits": split_audits,
            "all_block_sizes_feasible": split_feasible,
            "recording_safe_provenance_proven": False,
        },
    )

    classification_run = False
    if not coherence_destruction_gate_passed:
        classifier_reason = "actual coherence did not consistently exceed destruction controls; model execution forbidden"
    elif not spatial_gate_passed:
        classifier_reason = "global gain/phase invariance failed for the full condition-bearing feature vector; model execution forbidden"
    elif not split_feasible:
        classifier_reason = "exact power-balanced block-disjoint train/test with guard is impossible for every frozen block size"
    else:
        # The released labels are currently known not to satisfy this branch.
        # Fail closed rather than silently substituting a random snapshot split.
        classifier_reason = "classification implementation intentionally fail-closed because provenance-safe split proof is required"
    write_json(
        output / "classifier_not_run.json",
        {
            "classification_run": False,
            "reason": classifier_reason,
            "power_only_baseline": "NOT_RUN",
            "single_channel_amplitude_spectrum_baseline": "NOT_RUN",
            "normalized_spatial_logistic_regression": "NOT_RUN",
            "snapshot_random_split_used": False,
            "deep_learning_used": False,
            "hyperparameter_search_used": False,
        },
    )

    if block_decision_flips:
        verdict = "INCONCLUSIVE_SPATIAL_SIGNAL_PROVENANCE_BLOCKED"
        verdict_reason = "spatial coherence conclusion changes with contiguous block size"
    elif not coherence_destruction_gate_passed:
        verdict = "STOP_NO_USABLE_SPATIAL_COHERENCE"
        verdict_reason = "actual tuples do not consistently exceed all destruction controls"
    elif not spatial_gate_passed:
        verdict = "INCONCLUSIVE_SPATIAL_SIGNAL_PROVENANCE_BLOCKED"
        verdict_reason = "coherence survives B/C/D comparison, but the full condition-bearing feature vector fails frozen global-invariance tolerance"
    elif not split_feasible:
        verdict = "INCONCLUSIVE_SPATIAL_SIGNAL_PROVENANCE_BLOCKED"
        verdict_reason = "spatial coherence exists, but exact power-balanced blocked classification is not identifiable from released labels"
    else:
        verdict = "INCONCLUSIVE_SPATIAL_SIGNAL_PROVENANCE_BLOCKED"
        verdict_reason = "recording, transmitter position, and calibration provenance remain unavailable"

    write_json(
        output / "block_sensitivity.json",
        {
            "block_sizes": list(BLOCK_SIZES),
            "destruction_comparison_decisions": comparisons,
            "decision_flips": block_decision_flips,
            "classification_split_feasibility": [
                {"block_size": item["block_size"], "feasible": item["balanced_block_disjoint_split_feasible"]}
                for item in split_audits
            ],
        },
    )
    (output / "confound_analysis.md").write_text(
        "# Confound analysis\n\n"
        "This is **not** clean-versus-spoofing detection. The bounded object has no clean CRPA class. "
        "The only intended task is spoof/meacon versus non-deceptive terrestrial jammer.\n\n"
        "The official release does not bind snapshots to recording ID, timestamp/day, transmitter identity/position, "
        "receiver orientation, antenna ordering, geometry, cable/channel calibration, or VGA. Sample-index blocks are "
        "therefore only a leakage-reduction proxy, never proof of recording independence.\n\n"
        "Within Area 1, the nominal common transmit-power values are severely class-imbalanced. At least one negative "
        "power cell occupies only one block, so no block-disjoint train/test assignment can contain every exact power "
        "cell on both sides for all frozen block sizes. A snapshot-random substitute would violate the contract.\n\n"
        "Even when actual tuples exceed channel-mismatched and phase-destroyed controls, this demonstrates simultaneous "
        "multi-channel structure, not spoofing-specific directionality. Transmitter location, waveform family, multipath, "
        "and unknown array calibration remain plausible causes.\n",
        encoding="utf-8",
    )

    build_figures(output, features)
    write_json(
        output / "access_audit.json",
        {
            "mode": "ONE_BOUNDED_CRPA_OBJECT",
            "downloaded_payload_bytes": EXPECTED_BYTES,
            "downloaded_object_count": 1,
            "downloaded_object_sha256": EXPECTED_SHA256,
            "other_lfs_payload_bytes": 0,
            "innosense_hdf5_bytes": 0,
            "texbat_payload_bytes": 0,
            "oakbat_payload_bytes": 0,
            "tuni_payload_bytes": 0,
            "unique_raw_object_bytes_opened": EXPECTED_BYTES,
            "raw_objects_opened": 1,
            "logical_integrity_read_bytes": EXPECTED_BYTES,
            "logical_channel_quality_read_bytes": EXPECTED_BYTES - schema["data_offset_bytes"],
            "logical_labelled_snapshot_read_bytes": int(len(rows) * 4 * 1_024 * 8),
            "raw_source_modified": False,
            "models_trained": classification_run,
            "scores_computed": classification_run,
        },
    )
    write_json(
        output / "final_verdict.json",
        {
            "verdict": verdict,
            "reason": verdict_reason,
            "task": "spoof_meacon_vs_non_deceptive_terrestrial_jammer",
            "schema_gate_passed": True,
            "coherence_destruction_gate_passed": coherence_destruction_gate_passed,
            "spatial_gate_passed": spatial_gate_passed,
            "classification_split_feasible": split_feasible,
            "classification_run": classification_run,
            "clean_class_present": False,
            "recording_provenance_present": False,
            "antenna_calibration_present": False,
            "transmitter_position_provenance_present": False,
            "ready_for_wcl_declared": False,
            "general_spoofing_detector_success_declared": False,
        },
    )
    (output / "README.md").write_text(
        "# Jammertest 2025 CRPA Stage-0B bounded-object validation\n\n"
        f"Final verdict: `{verdict}`. The observed NPY is `(42673, 4, 1024)` complex64 and was opened read-only.\n\n"
        "This work evaluates simultaneous-channel structure and, only if provenance-safe splitting is possible, the "
        "restricted spoof/meacon-versus-non-deceptive-jammer question. It does not claim normal-versus-spoof detection, "
        "READY_FOR_WCL, or a deployable detector. No clean CRPA class exists.\n\n"
        f"Spatial gate passed: `{spatial_gate_passed}`. Classification run: `{classification_run}`. "
        f"Reason: {classifier_reason}.\n",
        encoding="utf-8",
    )
    write_manifest(output)
    print(json.dumps({
        "status": "BUILT",
        "verdict": verdict,
        "schema": schema["shape"],
        "spatial_gate_passed": spatial_gate_passed,
        "classification_run": classification_run,
        "output": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
