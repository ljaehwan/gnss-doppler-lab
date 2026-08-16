#!/usr/bin/env python3
"""Produce the mandatory post-hoc TRACE-R2 rep1/rep2 root-cause audit."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.trace_reproducibility import (
    KEY_FIELDS,
    METADATA_FIELDS,
    PHYSICAL_FIELDS,
    assignment_rows,
    canonical_join,
    canonical_semantic_hash,
    common_epoch_count,
    complex_correlation,
    exact_equal,
    field_statistics,
    load_replay,
)
from gnss_doppler_lab.trace_native_1ms import ACTION_VALUE_FIELDS, TAPS


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prn_summary(replay) -> dict[str, object]:
    result = {}
    for prn in np.unique(replay.records["prn"]):
        rows = replay.records[replay.records["prn"] == prn]
        result[str(int(prn))] = {
            "channels": sorted(map(int, np.unique(rows["channel"]))),
            "record_count": int(len(rows)),
            "first_raw_interval_start_sample": int(rows["raw_interval_start_sample"].min()),
            "last_raw_interval_end_sample": int(rows["raw_interval_end_sample"].max()),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep1", type=Path, required=True)
    parser.add_argument("--rep2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="TEXBAT.cleanStatic")
    parser.add_argument("--sample-rate-hz", type=float, default=25_000_000.0)
    args = parser.parse_args()
    output = args.output
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    rep1 = load_replay(args.rep1, args.dataset)
    rep2 = load_replay(args.rep2, args.dataset)
    joined = canonical_join(rep1, rep2)
    ia = joined["rep1_row_index"].to_numpy(dtype=np.int64)
    ib = joined["rep2_row_index"].to_numpy(dtype=np.int64)
    left = rep1.records[ia]
    right = rep2.records[ib]
    statistics, exact_rows = field_statistics(left, right)
    common_count = len(joined)
    rep1_prns = sorted(map(int, np.unique(rep1.records["prn"])))
    rep2_prns = sorted(map(int, np.unique(rep2.records["prn"])))

    divergent = np.flatnonzero(~exact_rows)
    first_index = (
        int(divergent[np.argmin(joined.iloc[divergent]["raw_interval_start_sample"].to_numpy())])
        if len(divergent)
        else None
    )
    if first_index is None:
        first = None
    else:
        same_prn = np.flatnonzero(joined["prn"].to_numpy() == joined.iloc[first_index]["prn"])
        within_prn = int(np.flatnonzero(same_prn == first_index)[0])
        context_positions = same_prn[max(0, within_prn - 1) : within_prn + 2]
        first = {
            "canonical_key": {
                field: (str(joined.iloc[first_index][field]) if field == "dataset" else int(joined.iloc[first_index][field]))
                for field in KEY_FIELDS
            },
            "rep1_file": str(rep1.files[ia[first_index]]),
            "rep2_file": str(rep2.files[ib[first_index]]),
            "metadata_before_after": {
                field: {"rep1": int(left[field][first_index]), "rep2": int(right[field][first_index])}
                for field in METADATA_FIELDS
            },
            "physical_differences": [
                {
                    "field": field,
                    "rep1": float(left[field][first_index]),
                    "rep2": float(right[field][first_index]),
                    "absolute_error": float(abs(float(left[field][first_index]) - float(right[field][first_index]))),
                }
                for field in PHYSICAL_FIELDS
                if not bool(exact_equal(left[field][first_index : first_index + 1], right[field][first_index : first_index + 1])[0])
            ],
            "receiver_state_before_at_after": [
                {
                    "canonical_key": {
                        field: (str(joined.iloc[position][field]) if field == "dataset" else int(joined.iloc[position][field]))
                        for field in KEY_FIELDS
                    },
                    "rep1": {
                        "channel": int(left["channel"][position]),
                        "tracking_session_id": int(left["tracking_session_id"][position]),
                        "loop_sequence": int(left["loop_sequence"][position]),
                        "receiver_state": int(left["receiver_state"][position]),
                        "carrier_doppler_hz": float(left["action_used_carrier_doppler_hz"][position]),
                        "code_nco_rate_chips_s": float(left["action_used_code_nco_rate_chips_s"][position]),
                    },
                    "rep2": {
                        "channel": int(right["channel"][position]),
                        "tracking_session_id": int(right["tracking_session_id"][position]),
                        "loop_sequence": int(right["loop_sequence"][position]),
                        "receiver_state": int(right["receiver_state"][position]),
                        "carrier_doppler_hz": float(right["action_used_carrier_doppler_hz"][position]),
                        "code_nco_rate_chips_s": float(right["action_used_code_nco_rate_chips_s"][position]),
                    },
                }
                for position in context_positions
            ],
        }

    summary1 = prn_summary(rep1)
    summary2 = prn_summary(rep2)
    starts = {}
    for prn in sorted(set(rep1_prns) | set(rep2_prns)):
        a = summary1.get(str(prn))
        b = summary2.get(str(prn))
        starts[str(prn)] = {
            "rep1_first_raw_interval_start_sample": None if a is None else a["first_raw_interval_start_sample"],
            "rep2_first_raw_interval_start_sample": None if b is None else b["first_raw_interval_start_sample"],
            "start_delta_samples_rep2_minus_rep1": None if a is None or b is None else b["first_raw_interval_start_sample"] - a["first_raw_interval_start_sample"],
        }

    rep1_semantic_hash = canonical_semantic_hash(rep1)
    rep2_semantic_hash = canonical_semantic_hash(rep2)
    audit = {
        "schema": "gnss-doppler-lab.trace-r2a-root-cause-audit.v1",
        "analysis_designation": "POST_HOC_ROOT_CAUSE_AUDIT",
        "dataset": args.dataset,
        "canonical_key": list(KEY_FIELDS),
        "classification": ["Case B", "Case C"],
        "classification_text": "Mixed Case B/C: concurrent acquisition/channel scheduling changed PRN assignment and tracking start samples; the changed handoff then produced true physical tap/action/state differences at identical canonical keys.",
        "case_a_representational_only": False,
        "case_b_acquisition_start_difference": True,
        "case_c_true_physical_value_nondeterminism": True,
        "rep1_prn_set": rep1_prns,
        "rep2_prn_set": rep2_prns,
        "rep1_only_prns": sorted(set(rep1_prns) - set(rep2_prns)),
        "rep2_only_prns": sorted(set(rep2_prns) - set(rep1_prns)),
        "rep1_prn_summary": summary1,
        "rep2_prn_summary": summary2,
        "acquisition_or_lock_start_deltas": starts,
        "rep1_row_count": int(len(rep1.records)),
        "rep2_row_count": int(len(rep2.records)),
        "common_canonical_row_count": int(common_count),
        "common_ratio_rep1": float(common_count / len(rep1.records)),
        "common_ratio_rep2": float(common_count / len(rep2.records)),
        "exact_physical_bit_match_row_count": int(exact_rows.sum()),
        "exact_physical_bit_match_ratio": float(exact_rows.mean()) if len(exact_rows) else None,
        "common_at_least_4_prn_epoch_count": common_epoch_count(joined, args.sample_rate_hz),
        "complex_tap_correlation": complex_correlation(left, right),
        "action_correlations": {
            row["field"]: row["correlation"] for row in statistics if row["field"].startswith("action_")
        },
        "field_error_statistics": statistics,
        "rep1_canonicalized_semantic_sha256": rep1_semantic_hash,
        "rep2_canonicalized_semantic_sha256": rep2_semantic_hash,
        "canonicalized_semantic_hash_match": rep1_semantic_hash == rep2_semantic_hash,
        "first_physical_divergence_position": None if first is None else first["canonical_key"],
        "interpretation": "Whole-file hash differences are not just channel/file metadata. Identical PRN/raw intervals frequently carry different correlator taps and loop state after acquisition starts moved by integer-millisecond epochs.",
    }
    dump_json(output / "rep1_rep2_root_cause_audit.json", audit)
    dump_json(
        output / "first_divergence_analysis.json",
        {
            "schema": "gnss-doppler-lab.trace-r2a-first-divergence.v1",
            "analysis_designation": "POST_HOC_ROOT_CAUSE_AUDIT",
            "first_divergence": first,
        },
    )

    assignments = assignment_rows(rep1, "rep1") + assignment_rows(rep2, "rep2")
    with (output / "channel_prn_assignment_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(assignments[0]))
        writer.writeheader()
        writer.writerows(assignments)

    with gzip.open(output / "canonical_record_comparison.csv.gz", "wt", newline="") as stream:
        fields = [
            *KEY_FIELDS,
            "rep1_channel",
            "rep2_channel",
            "metadata_equal",
            "physical_exact_bit_match",
            "maximum_tap_component_absolute_error",
            "maximum_action_absolute_error",
            "maximum_other_physical_absolute_error",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        tap_fields = [value for tap in TAPS for value in (f"{tap}_i", f"{tap}_q")]
        action_fields = [field for field in PHYSICAL_FIELDS if field.startswith("action_")]
        other_fields = [field for field in PHYSICAL_FIELDS if field not in set(tap_fields + action_fields)]
        for position, row in joined.reset_index(drop=True).iterrows():
            item = {field: row[field] for field in KEY_FIELDS}
            item.update(
                {
                    "rep1_channel": int(left["channel"][position]),
                    "rep2_channel": int(right["channel"][position]),
                    "metadata_equal": int(all(left[field][position] == right[field][position] for field in METADATA_FIELDS)),
                    "physical_exact_bit_match": int(exact_rows[position]),
                    "maximum_tap_component_absolute_error": max(abs(float(left[field][position]) - float(right[field][position])) for field in tap_fields),
                    "maximum_action_absolute_error": max(abs(float(left[field][position]) - float(right[field][position])) for field in action_fields),
                    "maximum_other_physical_absolute_error": max(abs(float(left[field][position]) - float(right[field][position])) for field in other_fields),
                }
            )
            writer.writerow(item)

    fig, axis = plt.subplots(figsize=(10, 4))
    prns = sorted(map(int, starts))
    x = np.arange(len(prns))
    axis.scatter(x - 0.1, [starts[str(p)]["rep1_first_raw_interval_start_sample"] or np.nan for p in prns], label="rep1", s=25)
    axis.scatter(x + 0.1, [starts[str(p)]["rep2_first_raw_interval_start_sample"] or np.nan for p in prns], label="rep2", s=25)
    axis.set(xticks=x, xticklabels=prns, xlabel="PRN", ylabel="first raw interval start sample", title="R2 acquisition/tracking start and PRN support")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots / "rep1_rep2_acquisition_start_and_prn_assignment.png", dpi=140)
    plt.close(fig)

    tap_rows = [row for row in statistics if row["field"].split("_")[0] in set(TAPS)]
    fig, axis = plt.subplots(figsize=(11, 4))
    axis.bar(np.arange(len(tap_rows)), [row["median_absolute_error"] for row in tap_rows])
    axis.set(xticks=np.arange(len(tap_rows)), xticklabels=[row["field"] for row in tap_rows], xlabel="tap component", ylabel="median absolute error", title="Common raw-sample tap error")
    axis.tick_params(axis="x", rotation=70)
    fig.tight_layout()
    fig.savefig(plots / "common_raw_sample_tap_action_error.png", dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 3.5))
    window = slice(max(0, (first_index or 0) - 50), min(len(exact_rows), (first_index or 0) + 200))
    axis.step(np.arange(len(exact_rows))[window], exact_rows[window].astype(int), where="post")
    axis.set(xlabel="canonical common-row position", ylabel="all physical fields exact", ylim=(-0.1, 1.1), title="First physical divergence")
    fig.tight_layout()
    fig.savefig(plots / "first_divergence.png", dpi=140)
    plt.close(fig)
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
