#!/usr/bin/env python3
"""Acquire bounded scenario-specific pre-onset state for DS7 or OS4."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gnss_doppler_lab.trace_native_1ms import read_records, sha256_file
import run_trace_stage0_r2e as r2e

ARTIFACT = r2e.driver.ARTIFACT
SSD = r2e.driver.SSD_ROOT
RECEIVER = r2e.driver.RECEIVER
SPECS = {
    "TEXBAT.DS7": {"slug": "texbat_ds7", "onset_s": 110.0},
    "OAKBAT.OS4": {"slug": "oakbat_os4", "onset_s": 120.0},
}


def stat(path: Path) -> dict[str, int | str]:
    value = path.stat()
    return {
        "path": str(path),
        "size_bytes": value.st_size,
        "device": value.st_dev,
        "inode": value.st_ino,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=tuple(SPECS), required=True)
    args = parser.parse_args()
    prereg = json.loads((ARTIFACT / "preregistration.json").read_text())
    if prereg["status"] != "SEALED_BEFORE_REPAIRED_SUPPORT_ACQUISITION_OR_METRIC_EVALUATION":
        raise ValueError("support acquisition lacks sealed R2e preregistration")
    repair = prereg["scenario_repairs"][args.scenario]
    spec = r2e.driver.PHASE_B_SCENARIOS[args.scenario]
    skip_s = float(repair["source_skip_s"])
    duration_s = float(repair["support_duration_s"])
    if skip_s + duration_s >= float(repair["onset_s"]):
        raise ValueError("bounded acquisition is not strictly pre-onset")
    out = SSD / "support_acquisition" / SPECS[args.scenario]["slug"] / "rep1"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    fs = int(spec["fs"])
    raw_offset = int(round(skip_s * fs))
    values = {
        "SignalSource.filename": str(spec["raw"]),
        "SignalSource.seconds_to_skip": str(skip_s),
        "SignalSource.samples": str(int(round(duration_s * fs * 2))),
        "Tracking_1C.dump": "false",
        "Tracking_1C.dump_mat": "false",
        "Tracking_1C.tap_count": "9",
        "Tracking_1C.tap_spacing_chips": "0.125",
        "Tracking_1C.extend_correlation_symbols": "1",
        "Tracking_1C.trace_dump": "true",
        "Tracking_1C.trace_dump_filename": "trace_native_1ms_ch_",
        "Tracking_1C.trace_scenario_id": f"{args.scenario}.support",
        "Tracking_1C.trace_raw_sample_offset": str(raw_offset),
        "Tracking_1C.trace_handoff_filename": "",
        "Observables.dump": "false",
        "SignalSource.enable_terminal_drain": "true",
    }
    config = out / "receiver.conf"
    config.write_text(r2e.driver.set_config_values(spec["base_config"].read_text(), values))
    before_raw, before_exe = stat(spec["raw"]), stat(RECEIVER)
    command = [str(RECEIVER), f"--config_file={config}", "--keyboard=false"]
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(command, cwd=out, check=False)
    ended = datetime.now(timezone.utc).isoformat()
    dumps = []
    selectable = 0
    threshold = int(round(float(repair["selection_time_s"]) * fs))
    onset_sample = int(round(float(repair["onset_s"]) * fs))
    for path in sorted(out.glob("trace_native_1ms_ch_*.bin")):
        _, records = read_records(path)
        eligible = records[
            (records["raw_interval_start_sample"] >= threshold)
            & (records["raw_interval_start_sample"] < onset_sample)
        ]
        selectable += bool(len(eligible))
        dumps.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "byte_size": path.stat().st_size,
                "record_count": int(len(records)),
                "eligible_pre_onset_row_count": int(len(eligible)),
                "prns": sorted({int(value) for value in records["prn"]}),
            }
        )
    passed = completed.returncode == 0 and selectable >= 4
    manifest = {
        "schema": "gnss-doppler-lab.trace-r2e-attack-support-acquisition.v1",
        "status": "PASS" if passed else "FAIL",
        "preregistration": "preregistration.json",
        "dataset": args.scenario,
        "pre_onset_only": True,
        "trace_scores_used": False,
        "started_at": started,
        "ended_at": ended,
        "command": command,
        "exit_code": completed.returncode,
        "receiver": {**before_exe, "sha256": sha256_file(RECEIVER), "stable": before_exe == stat(RECEIVER)},
        "receiver_config": {"path": str(config), "sha256": sha256_file(config)},
        "raw_iq": {**before_raw, "expected_sha256": spec["sha256"], "stable": before_raw == stat(spec["raw"])},
        "source_skip_s": skip_s,
        "bounded_duration_s": duration_s,
        "selection_time_s": repair["selection_time_s"],
        "onset_s": repair["onset_s"],
        "selectable_physical_channel_count": int(selectable),
        "dump_files": dumps,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
