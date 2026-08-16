#!/usr/bin/env python3
"""Freeze exact-sample TRACE-R2b tracking-state handoffs from preserved R2 rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.trace_native_1ms import ACTION_VALUE_FIELDS, read_records, sha256_file

R2 = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2-native-1ms-dump/dumps/phase_a")
SCENARIOS = {
    "TEXBAT.cleanStatic": {"slug": "texbat_cleanstatic", "source": R2 / "texbat_cleanstatic/rep1", "fs": 25_000_000, "skip_s": 0.0, "guard_s": 30.0, "onset": None},
    "TEXBAT.DS3": {"slug": "texbat_ds3", "source": R2 / "texbat_ds3/rep1", "fs": 25_000_000, "skip_s": 90.0, "guard_s": 5.0, "onset": 2_500_000_000},
    "OAKBAT.OS3": {"slug": "oakbat_os3", "source": R2 / "oakbat_os3/rep1", "fs": 5_000_000, "skip_s": 90.0, "guard_s": 5.0, "onset": 600_000_000},
}
CSV_FIELDS = ["channel", "source_channel", "prn", "first_raw_interval_start_sample", "source_raw_interval_start_sample", *ACTION_VALUE_FIELDS, "interval_length_samples"]


def scalar(value):
    return value.item() if hasattr(value, "item") else value


def derive(spec: dict) -> tuple[list[dict], list[dict]]:
    raw_offset = int(round(spec["skip_s"] * spec["fs"]))
    guard = raw_offset + int(round(spec["guard_s"] * spec["fs"]))
    rows, sources = [], []
    for path in sorted(spec["source"].glob("trace_native_1ms_ch_*.bin")):
        _, records = read_records(path)
        eligible = records[records["raw_interval_start_sample"] >= guard]
        if not len(eligible):
            sources.append({"path": str(path), "sha256": sha256_file(path), "status": "EXCLUDED_NO_TARGET_ALIGNED_ROW"})
            continue
        row = eligible[0]
        absolute = int(row["raw_interval_start_sample"])
        if spec["onset"] is not None and absolute >= spec["onset"]:
            raise ValueError(f"{path}: selected row is not pre-onset")
        output = {
            "channel": len(rows),
            "source_channel": int(row["channel"]),
            "prn": int(row["prn"]),
            "first_raw_interval_start_sample": absolute - raw_offset,
            "source_raw_interval_start_sample": absolute,
        }
        for field in ACTION_VALUE_FIELDS:
            output[field] = scalar(row[f"action_used_{field}"])
        output["interval_length_samples"] = int(row["action_used_interval_length_samples"])
        rows.append(output)
        sources.append({"path": str(path), "sha256": sha256_file(path), "status": "SELECTED", "selected_row_index": int((records["raw_interval_start_sample"] < guard).sum()), "selected_absolute_raw_sample": absolute})
    rows.sort(key=lambda value: value["channel"])
    if len(rows) < 4 or [row["channel"] for row in rows] != list(range(len(rows))):
        raise ValueError("fewer than four target-aligned rows or non-contiguous output channels")
    return rows, sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "gnss-doppler-lab.trace-r2b-handoff-manifest.v1", "selection_rule": "first preserved R2 rep1 row at or after the preregistered guard; no quality or attack outcome used", "state_kind": "exact_target_aligned_action_used", "scenarios": {}}
    for name, spec in SCENARIOS.items():
        rows, sources = derive(spec)
        path = args.output / f"{spec['slug']}.csv"
        if path.exists():
            with path.open(newline="") as stream:
                if list(csv.DictReader(stream)) != [{key: str(value) for key, value in row.items()} for row in rows]:
                    raise ValueError(f"{path}: existing freeze differs")
        else:
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        manifest["scenarios"][name] = {"handoff_path": str(path), "handoff_sha256": sha256_file(path), "guard_absolute_sample": int(round((spec["skip_s"] + spec["guard_s"]) * spec["fs"])), "raw_offset_samples": int(round(spec["skip_s"] * spec["fs"])), "channel_count": len(rows), "fixed_channel_prn_map": {str(row["channel"]): row["prn"] for row in rows}, "source_channel_map": {str(row["channel"]): row["source_channel"] for row in rows}, "source_files": sources, "all_selected_rows_pre_onset": True}
    manifest_path = args.output / "manifest.json"
    payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if manifest_path.exists() and manifest_path.read_text() != payload:
        raise ValueError("existing handoff manifest differs")
    manifest_path.write_text(payload)
    print(json.dumps({"status": "PASS", "scenarios": {k: v["handoff_sha256"] for k, v in manifest["scenarios"].items()}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
