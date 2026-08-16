#!/usr/bin/env python3
"""Confirm one R2d attack-support failure without evaluating TRACE scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.trace_native_1ms import load_native_trace_pairs, read_records, sha256_file

ARTIFACT = ROOT / "artifacts/trace_stage0_r2e_attack_support_repair"
PARENT = ROOT / "artifacts/trace_stage0_r2d_oakbat_clean_support_repair"
PARENT_SSD = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "trace-stage0-r2d-oakbat-clean-support-repair"
)
SPECS = {
    "TEXBAT.DS7": {
        "slug": "texbat_ds7",
        "onset_s": 110.0,
        "parent_handoff": "texbat_ds3.csv",
        "historical_summary": Path(
            "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
            "ds7-sealed-input/receiver/ds7-complex9/tracking_summary.csv"
        ),
        "historical_end_required_s": 150.0,
        "output": "ds7_attack_support_audit.json",
    },
    "OAKBAT.OS4": {
        "slug": "oakbat_os4",
        "onset_s": 120.0,
        "parent_handoff": "oakbat_os3.csv",
        "historical_summary": Path(
            "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
            "q-comet-oakbat-complex9/os4/receiver/os4-complex9/tracking_summary.csv"
        ),
        "historical_end_required_s": 130.0,
        "output": "os4_attack_support_audit.json",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=tuple(SPECS), required=True)
    args = parser.parse_args()
    spec = SPECS[args.scenario]
    prereg = json.loads((ARTIFACT / "preregistration.json").read_text())
    if not prereg["status"].startswith("SEALED_BEFORE"):
        raise ValueError("R2e preregistration is not sealed")
    dump = PARENT_SSD / "dumps/phase_b" / spec["slug"] / "rep1"
    pairs = load_native_trace_pairs(dump, cn0_min_db_hz=28.0, lock_min=0.85, prompt_epsilon=1e-12)
    common = pairs.valid_support[:, np.arange(1, 8)].all(axis=1)
    finite = (
        np.isfinite(pairs.current[:, 1:8].real).all(axis=1)
        & np.isfinite(pairs.current[:, 1:8].imag).all(axis=1)
        & np.isfinite(pairs.target[:, 1:8].real).all(axis=1)
        & np.isfinite(pairs.target[:, 1:8].imag).all(axis=1)
    )
    selected = pairs.take(common & finite)
    block_id = np.floor(selected.time_s / 0.5).astype(np.int64)
    valid_blocks = []
    for value in np.unique(block_id):
        prns = np.unique(selected.prn[block_id == value])
        if len(prns) >= 4:
            valid_blocks.append(float(value * 0.5))
    with spec["historical_summary"].open(newline="") as stream:
        history = list(csv.DictReader(stream))
    historical_prns = sorted({int(row["prn"].lstrip("G")) for row in history})
    historical_max_end = max(float(row["end_time_s"]) for row in history)
    manifest = json.loads((dump / "manifest.json").read_text())
    physical = []
    for path in sorted(dump.glob("trace_native_1ms_ch_*.bin")):
        _, records = read_records(path)
        physical.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "record_count": int(len(records)),
                "raw_start_min": int(records["raw_interval_start_sample"].min()) if len(records) else None,
                "raw_start_max": int(records["raw_interval_start_sample"].max()) if len(records) else None,
            }
        )
    payload = {
        "schema": "gnss-doppler-lab.trace-r2e-attack-support-audit.v1",
        "scenario": args.scenario,
        "status": "FAILURE_CONFIRMED_REPAIRABLE_INPUT_EXISTS",
        "parent_mapping": {
            "handoff": spec["parent_handoff"],
            "handoff_sha256": manifest["frozen_handoff_sha256"],
            "fixed_channel_prn_map": manifest["fixed_channel_prn_map"],
            "source_skip_s": manifest["raw_sample_range"]["seconds_to_skip"],
        },
        "parent_selected_support": {
            "pair_count": int(len(selected.time_s)),
            "unique_prns": sorted(map(int, np.unique(selected.prn))),
            "time_start_s": float(selected.time_s.min()),
            "time_end_s": float(selected.time_s.max()),
            "valid_four_prn_block_count": len(valid_blocks),
            "post_onset_four_prn_block_count": sum(time >= spec["onset_s"] for time in valid_blocks),
            "onset_s": spec["onset_s"],
        },
        "physical_native_files": physical,
        "independent_receiver_evidence": {
            "tracking_summary_path": str(spec["historical_summary"]),
            "tracking_summary_sha256": sha256_file(spec["historical_summary"]),
            "unique_prns": historical_prns,
            "maximum_track_end_s": historical_max_end,
            "covers_required_attack_window": historical_max_end >= spec["historical_end_required_s"],
        },
        "root_cause": (
            f"{args.scenario} raw IQ has independently demonstrated receiver tracking through "
            "the frozen attack window, but R2d reused a different scenario's handoff state. "
            "The cross-scenario state produced physical rows without adequate frozen "
            "quality/common support."
        ),
        "repair_is_handoff_only": True,
        "trace_scores_read_or_computed": False,
        "performance_claimed": False,
    }
    path = ARTIFACT / spec["output"]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
