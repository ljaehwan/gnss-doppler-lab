#!/usr/bin/env python3
"""Run the preregistered normal-only cleanStatic support acquisition."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gnss_doppler_lab.trace_native_1ms import read_records, sha256_file
import run_trace_stage0_r2d as r2d

ARTIFACT = r2d.driver.ARTIFACT
SSD = r2d.driver.SSD_ROOT
RECEIVER = r2d.driver.RECEIVER
RAW = r2d.driver.PHASE_B_SCENARIOS["OAKBAT.cleanStatic"]["raw"]
EXPECTED_RAW_SHA = r2d.driver.PHASE_B_SCENARIOS["OAKBAT.cleanStatic"]["sha256"]
BASE_CONFIG = r2d.driver.PHASE_B_SCENARIOS["OAKBAT.cleanStatic"]["base_config"]
OUT = SSD / "support_acquisition/oakbat_cleanstatic/rep1"


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
    plan = json.loads((ARTIFACT / "repair_plan_preregistered.json").read_text())
    if plan["status"] != "SEALED_BEFORE_SUPPORT_ACQUISITION":
        raise ValueError("support acquisition lacks sealed preregistration")
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    values = {
        "SignalSource.filename": str(RAW),
        "SignalSource.seconds_to_skip": "0.0",
        "SignalSource.samples": str(45 * 5_000_000 * 2),
        "Tracking_1C.dump": "false",
        "Tracking_1C.dump_mat": "false",
        "Tracking_1C.tap_count": "9",
        "Tracking_1C.tap_spacing_chips": "0.125",
        "Tracking_1C.extend_correlation_symbols": "1",
        "Tracking_1C.trace_dump": "true",
        "Tracking_1C.trace_dump_filename": "trace_native_1ms_ch_",
        "Tracking_1C.trace_scenario_id": "OAKBAT.cleanStatic.support",
        "Tracking_1C.trace_raw_sample_offset": "0",
        "Tracking_1C.trace_handoff_filename": "",
        "Observables.dump": "false",
        "SignalSource.enable_terminal_drain": "true",
    }
    config = OUT / "receiver.conf"
    config.write_text(r2d.driver.set_config_values(BASE_CONFIG.read_text(), values))
    before_raw, before_exe = stat(RAW), stat(RECEIVER)
    command = [str(RECEIVER), f"--config_file={config}", "--keyboard=false"]
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(command, cwd=OUT, check=False)
    ended = datetime.now(timezone.utc).isoformat()
    dumps = []
    physical = 0
    for path in sorted(OUT.glob("trace_native_1ms_ch_*.bin")):
        _, records = read_records(path)
        physical += bool(len(records))
        dumps.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "byte_size": path.stat().st_size,
                "record_count": int(len(records)),
                "prns": sorted({int(value) for value in records["prn"]}),
            }
        )
    manifest = {
        "schema": "gnss-doppler-lab.trace-r2d-clean-support-acquisition.v1",
        "status": "PASS" if completed.returncode == 0 and physical >= 4 else "FAIL",
        "preregistration": "repair_plan_preregistered.json",
        "dataset": "OAKBAT.cleanStatic",
        "normal_only": True,
        "attack_data_used": False,
        "started_at": started,
        "ended_at": ended,
        "command": command,
        "exit_code": completed.returncode,
        "receiver": {**before_exe, "sha256": sha256_file(RECEIVER), "stable": before_exe == stat(RECEIVER)},
        "receiver_config": {"path": str(config), "sha256": sha256_file(config)},
        "raw_iq": {**before_raw, "expected_sha256": EXPECTED_RAW_SHA, "stable": before_raw == stat(RAW)},
        "bounded_duration_s": 45.0,
        "selection_guard_s": 30.0,
        "physical_dump_file_count": int(physical),
        "dump_files": dumps,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0 if manifest["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
