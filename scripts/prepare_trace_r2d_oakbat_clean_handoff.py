#!/usr/bin/env python3
"""Freeze the preregistered cleanStatic-specific target-aligned handoff."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.trace_native_1ms import ACTION_VALUE_FIELDS, read_records, sha256_file

ARTIFACT = ROOT / "artifacts/trace_stage0_r2d_oakbat_clean_support_repair"
PARENT = ROOT / "artifacts/trace_stage0_r2c_terminal_drain_repair"
SOURCE = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "trace-stage0-r2d-oakbat-clean-support-repair/support_acquisition/"
    "oakbat_cleanstatic/rep1"
)
FIELDS = [
    "channel",
    "source_channel",
    "prn",
    "first_raw_interval_start_sample",
    "source_raw_interval_start_sample",
    *ACTION_VALUE_FIELDS,
    "interval_length_samples",
]
GUARD_SAMPLE = 30 * 5_000_000


def scalar(value):
    return value.item() if hasattr(value, "item") else value


def main() -> int:
    plan = json.loads((ARTIFACT / "repair_plan_preregistered.json").read_text())
    if plan["support_acquisition"]["selection_guard_s"] != 30.0:
        raise ValueError("selection guard differs from preregistration")
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    if manifest["status"] != "PASS" or not manifest["normal_only"]:
        raise ValueError("normal-only support acquisition did not pass")
    handoffs = ARTIFACT / "handoffs"
    if not handoffs.exists():
        shutil.copytree(PARENT / "handoffs", handoffs)
    rows, sources = [], []
    for path in sorted(SOURCE.glob("trace_native_1ms_ch_*.bin")):
        _, records = read_records(path)
        eligible = records[records["raw_interval_start_sample"] >= GUARD_SAMPLE]
        if not len(eligible):
            sources.append({"path": str(path), "sha256": sha256_file(path), "status": "EXCLUDED_NO_GUARD_ROW"})
            continue
        source = eligible[0]
        output = {
            "channel": len(rows),
            "source_channel": int(source["channel"]),
            "prn": int(source["prn"]),
            "first_raw_interval_start_sample": int(source["raw_interval_start_sample"]),
            "source_raw_interval_start_sample": int(source["raw_interval_start_sample"]),
        }
        for field in ACTION_VALUE_FIELDS:
            output[field] = scalar(source[f"action_used_{field}"])
        output["interval_length_samples"] = int(source["action_used_interval_length_samples"])
        rows.append(output)
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "status": "SELECTED",
                "source_channel": int(source["channel"]),
                "prn": int(source["prn"]),
                "selected_raw_sample": int(source["raw_interval_start_sample"]),
            }
        )
    if len(rows) < 4 or [row["channel"] for row in rows] != list(range(len(rows))):
        raise ValueError("fewer than four target-aligned rows or non-contiguous output channels")
    output = handoffs / "oakbat_cleanstatic.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    inherited = json.loads((handoffs / "manifest.json").read_text())
    inherited["schema"] = "gnss-doppler-lab.trace-r2d-handoff-manifest.v1"
    inherited["scenarios"]["OAKBAT.cleanStatic"] = {
        "handoff_path": str(output.relative_to(ROOT)),
        "handoff_sha256": sha256_file(output),
        "selection_rule": plan["support_acquisition"]["selection_rule"],
        "guard_absolute_sample": GUARD_SAMPLE,
        "raw_offset_samples": 0,
        "channel_count": len(rows),
        "fixed_channel_prn_map": {str(row["channel"]): row["prn"] for row in rows},
        "source_channel_map": {str(row["channel"]): row["source_channel"] for row in rows},
        "source_files": sources,
        "normal_only": True,
        "attack_data_used": False,
    }
    (handoffs / "manifest.json").write_text(json.dumps(inherited, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (ARTIFACT / "handoff_manifest.json").write_text(json.dumps(inherited, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "channel_count": len(rows), "handoff_sha256": sha256_file(output), "prns": [row["prn"] for row in rows]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
