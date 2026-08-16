#!/usr/bin/env python3
"""Derive preregistered deterministic Phase-A handoffs from preserved R2 rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.trace_native_1ms import read_records, sha256_file

SCENARIOS = {
    "TEXBAT.cleanStatic": {
        "slug": "texbat_cleanstatic",
        "fs": 25_000_000,
        "raw_offset": 0,
        "onset_s": None,
        "raw_sha256": "dd295ab46616bfe9634d1c37479520e720ebc54bcb64adf0a247315a541fb9b9",
        "base_config": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/receiver/cleanStatic-complex9/receiver.conf"),
        "source": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2-native-1ms-dump/dumps/phase_a/texbat_cleanstatic/rep1"),
    },
    "TEXBAT.DS3": {
        "slug": "texbat_ds3",
        "fs": 25_000_000,
        "raw_offset": 2_250_000_000,
        "onset_s": 118.9,
        "raw_sha256": "e37e11b060bc2c675d4a60024f8b4a53e95e7cd1d304bea80cd903856075a30d",
        "base_config": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/receiver/ds3-complex9/receiver.conf"),
        "source": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2-native-1ms-dump/dumps/phase_a/texbat_ds3/rep1"),
    },
    "OAKBAT.OS3": {
        "slug": "oakbat_os3",
        "fs": 5_000_000,
        "raw_offset": 450_000_000,
        "onset_s": 120.0,
        "raw_sha256": "2a3c3c5cf1accaa287fe14181e43070903500e0250c69e3c335f91c89c0cdc6c",
        "base_config": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/q-comet-oakbat-complex9/os3/receiver/os3-complex9/receiver.conf"),
        "source": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2-native-1ms-dump/dumps/phase_a/oakbat_os3/rep1"),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "gnss-doppler-lab.trace-r2a-frozen-handoff-manifest.v1",
        "procedure": "For each preserved pre-onset R2 channel, fix its first acquired PRN and grid Doppler; defer tracking to 30 s into the replay at the same 1 ms code-epoch residue and reset residual code/carrier phase by an explicit convention. The 30 s boundary is fixed from the failed rep3 handoff-start audit, before any physical TRACE values were evaluated.",
        "handoff_kind": "deterministic_tracking_start_from_preserved_pre_onset_acquisition_result",
        "preserved_schema_limitation": "The R2 native tracking schema did not serialize the acquisition test statistic or original acquisition-engine sample stamp. They are therefore marked unavailable rather than inferred or fabricated. The repair freezes the observable acquisition result (channel, PRN, acquisition-grid Doppler, and code-epoch residue) and an explicit deterministic first-tracking-sample/phase convention.",
        "guard_s_from_replay_start": 30.0,
        "scenarios": {},
    }
    for name, spec in SCENARIOS.items():
        target_path = args.output / f"{spec['slug']}.csv"
        rows = []
        source_files = []
        for path in sorted(spec["source"].glob("trace_native_1ms_ch_*.bin")):
            _, records = read_records(path)
            if not len(records):
                continue
            first = records[0]
            absolute_start = int(first["raw_interval_start_sample"])
            source_relative_start = absolute_start - int(spec["raw_offset"])
            code_epoch_residue = source_relative_start % int(round(int(spec["fs"]) * 0.001))
            target = int(30.0 * int(spec["fs"])) + code_epoch_residue
            pre_onset = spec["onset_s"] is None or absolute_start / int(spec["fs"]) < float(spec["onset_s"])
            if not pre_onset:
                raise ValueError(f"{name} channel {int(first['channel'])}: source handoff is not pre-onset")
            rows.append(
                {
                    "channel": int(first["channel"]),
                    "prn": int(first["prn"]),
                    "first_raw_interval_start_sample": target,
                    "carrier_doppler_hz": float(first["action_used_carrier_doppler_hz"]),
                    "source_first_absolute_raw_sample": absolute_start,
                    "code_epoch_residue_samples": code_epoch_residue,
                    "code_phase_convention": "residual_code_phase_chips_and_samples_zero_at_first_interval",
                    "carrier_phase_convention": "residual_carrier_phase_zero_and_accumulator_minus_doppler_step_times_first_sample",
                    "acquisition_sample_stamp": "UNAVAILABLE_IN_PRESERVED_R2_RELEASE_SCHEMA",
                    "acquisition_metric": "UNAVAILABLE_IN_PRESERVED_R2_RELEASE_SCHEMA",
                }
            )
            source_files.append({"path": str(path), "sha256": sha256_file(path), "source_first_absolute_raw_sample": absolute_start})
        rows.sort(key=lambda row: row["channel"])
        if [row["channel"] for row in rows] != list(range(11)):
            raise ValueError(f"{name}: expected one preserved handoff for channels 0..10")
        if len({row["prn"] for row in rows}) != len(rows):
            raise ValueError(f"{name}: first-session PRNs are not unique")
        if args.validate_only:
            with target_path.open(newline="") as stream:
                existing = list(csv.DictReader(stream))
            normalized = [
                {
                    "channel": int(row["channel"]),
                    "prn": int(row["prn"]),
                    "first_raw_interval_start_sample": int(row["first_raw_interval_start_sample"]),
                    "carrier_doppler_hz": float(row["carrier_doppler_hz"]),
                    "source_first_absolute_raw_sample": int(row["source_first_absolute_raw_sample"]),
                    "code_epoch_residue_samples": int(row["code_epoch_residue_samples"]),
                    "code_phase_convention": row["code_phase_convention"],
                    "carrier_phase_convention": row["carrier_phase_convention"],
                    "acquisition_sample_stamp": row["acquisition_sample_stamp"],
                    "acquisition_metric": row["acquisition_metric"],
                }
                for row in existing
            ]
            if normalized != rows:
                raise ValueError(f"{name}: frozen handoff differs from deterministic derivation")
        else:
            with target_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        manifest["scenarios"][name] = {
            "handoff_path": str(target_path),
            "handoff_sha256": sha256_file(target_path),
            "fixed_channel_prn_map": {str(row["channel"]): row["prn"] for row in rows},
            "all_source_rows_pre_onset": True,
            "raw_iq_sha256": spec["raw_sha256"],
            "source_receiver_config_sha256": sha256_file(spec["base_config"]),
            "receiver_repair_binding": "Completed in preregistration.json with receiver executable and patch SHA-256 before replay.",
            "phase_conventions": {
                "code": rows[0]["code_phase_convention"],
                "carrier": rows[0]["carrier_phase_convention"],
            },
            "unavailable_preserved_acquisition_fields": ["acquisition_sample_stamp", "acquisition_metric"],
            "source_r2_files": source_files,
        }
    manifest_path = args.output / "manifest.json"
    if args.validate_only:
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise ValueError("frozen handoff manifest differs from deterministic derivation")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
