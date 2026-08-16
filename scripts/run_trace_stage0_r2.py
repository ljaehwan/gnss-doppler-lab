#!/usr/bin/env python3
"""TRACE-R2 receiver replay and provenance driver.

Long invocations of this script are submitted by the local research runner.
Large native receiver dumps stay under the SSD root; Git artifacts contain
only manifests, hashes, validation results, and explicit unavailable outputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.trace_native_1ms import sha256_file

ARTIFACT = ROOT / "artifacts/trace_stage0_r2_native_1ms_dump"
SSD_ROOT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2-native-1ms-dump")
RECEIVER = SSD_ROOT / "receiver-build/src/main/gnss-sdr"

SCENARIOS = {
    "TEXBAT.cleanStatic": {
        "raw": Path("/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/cleanStatic.bin"),
        "sha256": "dd295ab46616bfe9634d1c37479520e720ebc54bcb64adf0a247315a541fb9b9",
        "fs": 25_000_000,
        "base_config": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/receiver/cleanStatic-complex9/receiver.conf"),
        "smoke_skip_s": 0.0,
        "smoke_duration_s": 45.0,
    },
    "TEXBAT.DS3": {
        "raw": Path("/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds3.bin"),
        "sha256": "e37e11b060bc2c675d4a60024f8b4a53e95e7cd1d304bea80cd903856075a30d",
        "fs": 25_000_000,
        "base_config": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/receiver/ds3-complex9/receiver.conf"),
        "smoke_skip_s": 90.0,
        "smoke_duration_s": 45.0,
    },
    "TEXBAT.DS7": {
        "raw": Path("/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds7.bin"),
        "sha256": "d5fb1430d476f68930f3bb0290b80b649f08012eb6b6981d112493528813400e",
        "fs": 25_000_000,
        "base_config": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/ds7-sealed-input/receiver/ds7-complex9/receiver.conf"),
    },
    "OAKBAT.cleanStatic": {
        "raw": Path("/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/cleanStatic_gps.bin"),
        "sha256": "8e3428abb1b94211118c1ec9f505322ef89fdb176f0cf33961eca3cf3da80dfe",
        "fs": 5_000_000,
        "base_config": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/q-comet-oakbat-complex9/cleanstatic/receiver/cleanstatic-complex9/receiver.conf"),
    },
    "OAKBAT.OS3": {
        "raw": Path("/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/os3.bin"),
        "sha256": "2a3c3c5cf1accaa287fe14181e43070903500e0250c69e3c335f91c89c0cdc6c",
        "fs": 5_000_000,
        "base_config": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/q-comet-oakbat-complex9/os3/receiver/os3-complex9/receiver.conf"),
        "smoke_skip_s": 90.0,
        "smoke_duration_s": 45.0,
    },
    "OAKBAT.OS4": {
        "raw": Path("/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/os4.bin"),
        "sha256": "803f3c76bcc618efbc6b394eb536fe61ed8c3e34b1822c0088b4475621bfa8e4",
        "fs": 5_000_000,
        "base_config": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/q-comet-oakbat-complex9/os4/receiver/os4-complex9/receiver.conf"),
    },
}
PHASE_A = ("TEXBAT.cleanStatic", "TEXBAT.DS3", "OAKBAT.OS3")
PHASE_B = tuple(SCENARIOS)


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def stat_payload(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def audit_raw(names: tuple[str, ...]) -> int:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT / "raw_source_binding.json"
    prior = json.loads(path.read_text()) if path.exists() else {"datasets": {}}
    datasets = dict(prior.get("datasets", {}))
    failed = False
    for name in names:
        scenario = SCENARIOS[name]
        raw = scenario["raw"]
        before = stat_payload(raw)
        actual = sha256_file(raw)
        after = stat_payload(raw)
        matches = actual == scenario["sha256"] and before == after
        failed |= not matches
        datasets[name] = {
            "raw_iq_path": str(raw),
            "sample_format": "interleaved_int16_iq",
            "sample_rate_hz": scenario["fs"],
            "expected_sha256_from_authenticated_R1": scenario["sha256"],
            "fresh_sha256": actual,
            "stat_before": before,
            "stat_after": after,
            "stable_during_hash": before == after,
            "status": "PASS" if matches else "FAIL",
        }
        print(f"{name}: {datasets[name]['status']} {actual}", flush=True)
    dump_json(
        path,
        {
            "schema": "gnss-doppler-lab.trace-r2-raw-source-binding.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "datasets": datasets,
            "status": "PASS" if all(item.get("status") == "PASS" for item in datasets.values()) else "FAIL",
        },
    )
    return 2 if failed else 0


def set_config_values(text: str, values: dict[str, str]) -> str:
    lines = text.splitlines()
    found: set[str] = set()
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0]
            if key in values:
                output.append(f"{key}={values[key]}")
                found.add(key)
                continue
        output.append(line)
    for key, value in values.items():
        if key not in found:
            output.append(f"{key}={value}")
    return "\n".join(output) + "\n"


def run_receiver(name: str, mode: str, repetition: int, skip_s: float | None, duration_s: float | None) -> int:
    scenario = SCENARIOS[name]
    if mode == "smoke":
        skip_s = scenario.get("smoke_skip_s") if skip_s is None else skip_s
        duration_s = scenario.get("smoke_duration_s") if duration_s is None else duration_s
    else:
        skip_s = 0.0 if skip_s is None else skip_s
        duration_s = None if duration_s is None else duration_s
    if skip_s is None or (mode == "smoke" and duration_s is None):
        raise ValueError(f"no {mode} slice configured for {name}")
    fs = int(scenario["fs"])
    raw_offset = int(round(skip_s * fs))
    source_items = 0 if duration_s is None else int(round(duration_s * fs * 2))
    slug = name.lower().replace(".", "_")
    out = SSD_ROOT / "dumps" / ("phase_a" if mode == "smoke" else "phase_b") / slug / f"rep{repetition}"
    out.mkdir(parents=True, exist_ok=True)
    base_config = scenario["base_config"]
    config = out / "receiver.conf"
    config_text = set_config_values(
        base_config.read_text(),
        {
            "SignalSource.filename": str(scenario["raw"]),
            "SignalSource.seconds_to_skip": str(skip_s),
            "SignalSource.samples": str(source_items),
            "Tracking_1C.dump": "false",
            "Tracking_1C.dump_mat": "false",
            "Tracking_1C.tap_count": "9",
            "Tracking_1C.tap_spacing_chips": "0.125",
            "Tracking_1C.extend_correlation_symbols": "1",
            "Tracking_1C.trace_dump": "true",
            "Tracking_1C.trace_dump_filename": str(out / "trace_native_1ms_ch_"),
            "Tracking_1C.trace_scenario_id": name,
            "Tracking_1C.trace_raw_sample_offset": str(raw_offset),
            "Observables.dump": "false",
        },
    )
    config.write_text(config_text)
    raw_before = stat_payload(scenario["raw"])
    exe_before = stat_payload(RECEIVER)
    command = [str(RECEIVER), f"--config_file={config}", "--keyboard=false"]
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(command, cwd=out, check=False)
    ended = datetime.now(timezone.utc).isoformat()
    raw_after = stat_payload(scenario["raw"])
    exe_after = stat_payload(RECEIVER)
    dumps = []
    for dump in sorted(out.glob("trace_native_1ms_ch_*.bin")):
        dumps.append({"path": str(dump), "size_bytes": dump.stat().st_size, "sha256": sha256_file(dump)})
    manifest = {
        "schema": "gnss-doppler-lab.trace-r2-receiver-replay.v1",
        "scenario_id": name,
        "mode": mode,
        "repetition": repetition,
        "started_at": started,
        "ended_at": ended,
        "command": command,
        "exit_code": completed.returncode,
        "receiver_executable": {**exe_before, "sha256": sha256_file(RECEIVER)},
        "receiver_executable_stat_after": exe_after,
        "receiver_stable_during_run": exe_before == exe_after,
        "receiver_config_path": str(config),
        "receiver_config_sha256": sha256_file(config),
        "raw_iq": {**raw_before, "sha256": scenario["sha256"]},
        "raw_iq_stat_after": raw_after,
        "raw_iq_stable_during_run": raw_before == raw_after,
        "raw_sample_range": {
            "start_inclusive": raw_offset,
            "end_exclusive": None if duration_s is None else raw_offset + int(round(duration_s * fs)),
            "seconds_to_skip": skip_s,
            "duration_s": duration_s,
        },
        "dump_files": dumps,
    }
    dump_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-raw")
    audit.add_argument("--phase", choices=("a", "b", "all"), default="a")
    replay = subparsers.add_parser("run-receiver")
    replay.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
    replay.add_argument("--mode", choices=("smoke", "full"), required=True)
    replay.add_argument("--repetition", type=int, default=1)
    replay.add_argument("--skip-s", type=float)
    replay.add_argument("--duration-s", type=float)
    args = parser.parse_args()
    if args.command == "audit-raw":
        names = PHASE_A if args.phase == "a" else PHASE_B
        return audit_raw(names)
    return run_receiver(args.scenario, args.mode, args.repetition, args.skip_s, args.duration_s)


if __name__ == "__main__":
    raise SystemExit(main())
