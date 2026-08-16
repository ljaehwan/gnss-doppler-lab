#!/usr/bin/env python3
"""Frozen TRACE-R2a receiver replay and source-binding driver."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.trace_native_1ms import read_records, sha256_file

ARTIFACT = ROOT / "artifacts/trace_stage0_r2a_reproducibility_repair"
SSD_ROOT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2a-reproducibility-repair")
RECEIVER = SSD_ROOT / "receiver-build/src/main/gnss-sdr"
HANDOFF_ROOT = ARTIFACT / "handoffs"
RELEASE = "r2a"

SCENARIOS = {
    "TEXBAT.cleanStatic": {
        "slug": "texbat_cleanstatic",
        "raw": Path("/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/cleanStatic.bin"),
        "sha256": "dd295ab46616bfe9634d1c37479520e720ebc54bcb64adf0a247315a541fb9b9",
        "fs": 25_000_000,
        "base_config": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/receiver/cleanStatic-complex9/receiver.conf"),
        "smoke_skip_s": 0.0,
        "smoke_duration_s": 45.0,
        "handoff": "texbat_cleanstatic.csv",
    },
    "TEXBAT.DS3": {
        "slug": "texbat_ds3",
        "raw": Path("/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds3.bin"),
        "sha256": "e37e11b060bc2c675d4a60024f8b4a53e95e7cd1d304bea80cd903856075a30d",
        "fs": 25_000_000,
        "base_config": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/receiver/ds3-complex9/receiver.conf"),
        "smoke_skip_s": 90.0,
        "smoke_duration_s": 45.0,
        "handoff": "texbat_ds3.csv",
    },
    "OAKBAT.OS3": {
        "slug": "oakbat_os3",
        "raw": Path("/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/os3.bin"),
        "sha256": "2a3c3c5cf1accaa287fe14181e43070903500e0250c69e3c335f91c89c0cdc6c",
        "fs": 5_000_000,
        "base_config": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/q-comet-oakbat-complex9/os3/receiver/os3-complex9/receiver.conf"),
        "smoke_skip_s": 90.0,
        "smoke_duration_s": 45.0,
        "handoff": "oakbat_os3.csv",
    },
}

PHASE_B_SCENARIOS = {
    "TEXBAT.cleanStatic": {
        **SCENARIOS["TEXBAT.cleanStatic"],
        "phase_b_skip_s": 0.0,
        "phase_b_handoff": "texbat_cleanstatic.csv",
    },
    "TEXBAT.DS3": {
        **SCENARIOS["TEXBAT.DS3"],
        "phase_b_skip_s": 90.0,
        "phase_b_handoff": "texbat_ds3.csv",
    },
    "TEXBAT.DS7": {
        "slug": "texbat_ds7",
        "raw": Path("/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds7.bin"),
        "sha256": "d5fb1430d476f68930f3bb0290b80b649f08012eb6b6981d112493528813400e",
        "fs": 25_000_000,
        "base_config": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/ds7-sealed-input/receiver/ds7-complex9/receiver.conf"),
        "phase_b_skip_s": 90.0,
        "phase_b_handoff": "texbat_ds3.csv",
    },
    "OAKBAT.cleanStatic": {
        "slug": "oakbat_cleanstatic",
        "raw": Path("/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/cleanStatic_gps.bin"),
        "sha256": "8e3428abb1b94211118c1ec9f505322ef89fdb176f0cf33961eca3cf3da80dfe",
        "fs": 5_000_000,
        "base_config": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/q-comet-oakbat-complex9/cleanstatic/receiver/cleanstatic-complex9/receiver.conf"),
        "phase_b_skip_s": 90.0,
        "phase_b_handoff": "oakbat_os3.csv",
    },
    "OAKBAT.OS3": {
        **SCENARIOS["OAKBAT.OS3"],
        "phase_b_skip_s": 90.0,
        "phase_b_handoff": "oakbat_os3.csv",
    },
    "OAKBAT.OS4": {
        "slug": "oakbat_os4",
        "raw": Path("/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/os4.bin"),
        "sha256": "803f3c76bcc618efbc6b394eb536fe61ed8c3e34b1822c0088b4475621bfa8e4",
        "fs": 5_000_000,
        "base_config": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/q-comet-oakbat-complex9/os4/receiver/os4-complex9/receiver.conf"),
        "phase_b_skip_s": 90.0,
        "phase_b_handoff": "oakbat_os3.csv",
    },
}


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


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


def set_config_values(text: str, values: dict[str, str]) -> str:
    lines = text.splitlines()
    found = set()
    output = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0]
            if key in values:
                output.append(f"{key}={values[key]}")
                found.add(key)
                continue
        output.append(line)
    output.extend(f"{key}={value}" for key, value in values.items() if key not in found)
    return "\n".join(output) + "\n"


def handoff_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def frozen_config_text(name: str) -> str:
    spec = SCENARIOS[name]
    handoff = HANDOFF_ROOT / spec["handoff"]
    rows = handoff_rows(handoff)
    fixed = {f"Channel{row['channel']}.satellite": row["prn"] for row in rows}
    skip_s = float(spec["smoke_skip_s"])
    duration_s = float(spec["smoke_duration_s"])
    raw_offset = int(round(skip_s * int(spec["fs"])))
    source_items = int(round(duration_s * int(spec["fs"]) * 2))
    values = {
        "SignalSource.filename": str(spec["raw"]),
        "SignalSource.seconds_to_skip": str(skip_s),
        "SignalSource.samples": str(source_items),
        "Channels_1C.count": str(len(rows)),
        "Channels.in_acquisition": str(len(rows)),
        "Tracking_1C.dump": "false",
        "Tracking_1C.dump_mat": "false",
        "Tracking_1C.tap_count": "9",
        "Tracking_1C.tap_spacing_chips": "0.125",
        "Tracking_1C.extend_correlation_symbols": "1",
        "Tracking_1C.trace_dump": "true",
        "Tracking_1C.trace_dump_filename": "trace_native_1ms_ch_",
        "Tracking_1C.trace_scenario_id": name,
        "Tracking_1C.trace_raw_sample_offset": str(raw_offset),
        "Tracking_1C.trace_handoff_filename": str(handoff.resolve()),
        "Observables.dump": "false",
        **fixed,
    }
    return set_config_values(spec["base_config"].read_text(), values)


def frozen_phase_b_config_text(name: str) -> str:
    spec = PHASE_B_SCENARIOS[name]
    handoff = HANDOFF_ROOT / spec["phase_b_handoff"]
    rows = handoff_rows(handoff)
    fixed = {f"Channel{row['channel']}.satellite": row["prn"] for row in rows}
    skip_s = float(spec["phase_b_skip_s"])
    raw_offset = int(round(skip_s * int(spec["fs"])))
    values = {
        "SignalSource.filename": str(spec["raw"]),
        "SignalSource.seconds_to_skip": str(skip_s),
        "SignalSource.samples": "0",
        "Channels_1C.count": str(len(rows)),
        "Channels.in_acquisition": str(len(rows)),
        "Tracking_1C.dump": "false",
        "Tracking_1C.dump_mat": "false",
        "Tracking_1C.tap_count": "9",
        "Tracking_1C.tap_spacing_chips": "0.125",
        "Tracking_1C.extend_correlation_symbols": "1",
        "Tracking_1C.trace_dump": "true",
        "Tracking_1C.trace_dump_filename": "trace_native_1ms_ch_",
        "Tracking_1C.trace_scenario_id": name,
        "Tracking_1C.trace_raw_sample_offset": str(raw_offset),
        "Tracking_1C.trace_handoff_filename": str(handoff.resolve()),
        "Observables.dump": "false",
        **fixed,
    }
    return set_config_values(spec["base_config"].read_text(), values)


def audit_raw() -> int:
    datasets = {}
    for name, spec in PHASE_B_SCENARIOS.items():
        before = stat_payload(spec["raw"])
        actual = sha256_file(spec["raw"])
        after = stat_payload(spec["raw"])
        passed = actual == spec["sha256"] and before == after
        datasets[name] = {
            "raw_iq": before,
            "expected_sha256": spec["sha256"],
            "fresh_sha256": actual,
            "stable_during_hash": before == after,
            "sample_rate_hz": spec["fs"],
            "status": "PASS" if passed else "FAIL",
        }
        print(f"{name}: {datasets[name]['status']} {actual}", flush=True)
    status = "PASS" if all(item["status"] == "PASS" for item in datasets.values()) else "FAIL"
    dump_json(
        ARTIFACT / "raw_source_binding.json",
        {"schema": f"gnss-doppler-lab.trace-{RELEASE}-raw-source-binding.v1", "status": status, "datasets": datasets},
    )
    return 0 if status == "PASS" else 2


def run_receiver(name: str, repetition: int, attempt: str | None = None) -> int:
    spec = SCENARIOS[name]
    repetition_directory = f"rep{repetition}" if attempt is None else f"rep{repetition}-{attempt}"
    out = SSD_ROOT / "dumps/phase_a" / spec["slug"] / repetition_directory
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing {RELEASE} replay directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    handoff = HANDOFF_ROOT / spec["handoff"]
    build_manifest_path = ARTIFACT / "receiver_build_manifest.json"
    if not build_manifest_path.exists():
        raise FileNotFoundError(f"missing preregistered receiver build manifest: {build_manifest_path}")
    build_manifest = json.loads(build_manifest_path.read_text())
    rows = handoff_rows(handoff)
    skip_s = float(spec["smoke_skip_s"])
    duration_s = float(spec["smoke_duration_s"])
    raw_offset = int(round(skip_s * int(spec["fs"])))
    source_items = int(round(duration_s * int(spec["fs"]) * 2))
    config = out / "receiver.conf"
    frozen_config = ARTIFACT / "frozen_configs" / f"{spec['slug']}.conf"
    if not frozen_config.exists():
        raise FileNotFoundError(f"missing preregistered frozen config: {frozen_config}")
    expected_text = frozen_config_text(name)
    if frozen_config.read_text() != expected_text:
        raise ValueError(f"frozen config no longer matches deterministic source template: {frozen_config}")
    config.write_bytes(frozen_config.read_bytes())
    raw_before = stat_payload(spec["raw"])
    exe_before = stat_payload(RECEIVER)
    command = [str(RECEIVER), f"--config_file={config}", "--keyboard=false"]
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(command, cwd=out, check=False)
    ended = datetime.now(timezone.utc).isoformat()
    raw_after = stat_payload(spec["raw"])
    exe_after = stat_payload(RECEIVER)
    dumps = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(out.glob("trace_native_1ms_ch_*.bin"))
    ]
    record_counts = []
    for item in dumps:
        _, records = read_records(Path(item["path"]))
        record_counts.append(len(records))
    replay_validation = {
        "expected_dump_file_count": len(rows),
        "observed_dump_file_count": len(dumps),
        "physical_dump_file_count": sum(count > 0 for count in record_counts),
        "minimum_physical_dump_files_required": 4,
        "all_dump_files_have_physical_records": len(record_counts) == len(rows) and all(count > 0 for count in record_counts),
        "record_counts": record_counts,
    }
    replay_validation["status"] = "PASS" if len(dumps) == len(rows) and replay_validation["physical_dump_file_count"] >= 4 else "FAIL"
    manifest = {
        "schema": f"gnss-doppler-lab.trace-{RELEASE}-receiver-replay.v1",
        "scenario_id": name,
        "repetition": repetition,
        "attempt": attempt,
        "started_at": started,
        "ended_at": ended,
        "command": command,
        "exit_code": completed.returncode,
        "receiver_executable": {**exe_before, "sha256": sha256_file(RECEIVER)},
        "receiver_patch_sha256": build_manifest["receiver_repair_diff"]["sha256"],
        "receiver_stable_during_run": exe_before == exe_after,
        "receiver_config_path": str(config),
        "receiver_config_sha256": sha256_file(config),
        "frozen_handoff_path": str(handoff.resolve()),
        "frozen_handoff_sha256": sha256_file(handoff),
        "fixed_channel_prn_map": {row["channel"]: int(row["prn"]) for row in rows},
        "raw_iq": {**raw_before, "sha256": spec["sha256"]},
        "raw_iq_stable_during_run": raw_before == raw_after,
        "raw_sample_range": {
            "start_inclusive": raw_offset,
            "end_exclusive": raw_offset + int(round(duration_s * int(spec["fs"]))),
            "seconds_to_skip": skip_s,
            "duration_s": duration_s,
        },
        "dump_files": dumps,
        "replay_validation": replay_validation,
    }
    dump_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return completed.returncode if completed.returncode else (0 if replay_validation["status"] == "PASS" else 2)


def run_phase_b_receiver(name: str) -> int:
    spec = PHASE_B_SCENARIOS[name]
    out = SSD_ROOT / "dumps/phase_b" / spec["slug"] / "rep1"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing {RELEASE} replay directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    handoff = HANDOFF_ROOT / spec["phase_b_handoff"]
    rows = handoff_rows(handoff)
    build_manifest = json.loads((ARTIFACT / "receiver_build_manifest.json").read_text())
    skip_s = float(spec["phase_b_skip_s"])
    raw_offset = int(round(skip_s * int(spec["fs"])))
    frozen_config = ARTIFACT / "frozen_configs/phase_b" / f"{spec['slug']}.conf"
    if frozen_config.read_text() != frozen_phase_b_config_text(name):
        raise ValueError(f"frozen Phase-B config no longer matches deterministic source template: {frozen_config}")
    config = out / "receiver.conf"
    config.write_bytes(frozen_config.read_bytes())
    raw_before = stat_payload(spec["raw"])
    exe_before = stat_payload(RECEIVER)
    command = [str(RECEIVER), f"--config_file={config}", "--keyboard=false"]
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(command, cwd=out, check=False)
    ended = datetime.now(timezone.utc).isoformat()
    raw_after = stat_payload(spec["raw"])
    exe_after = stat_payload(RECEIVER)
    dumps = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(out.glob("trace_native_1ms_ch_*.bin"))
    ]
    record_counts = []
    for item in dumps:
        _, records = read_records(Path(item["path"]))
        record_counts.append(len(records))
    replay_validation = {
        "expected_dump_file_count": 11,
        "observed_dump_file_count": len(dumps),
        "all_dump_files_have_physical_records": len(record_counts) == 11 and all(count > 0 for count in record_counts),
        "record_counts": record_counts,
    }
    replay_validation["status"] = "PASS" if replay_validation["all_dump_files_have_physical_records"] else "FAIL"
    manifest = {
        "schema": "gnss-doppler-lab.trace-r2a-receiver-replay.v1",
        "scenario_id": name,
        "phase": "B",
        "repetition": 1,
        "started_at": started,
        "ended_at": ended,
        "command": command,
        "exit_code": completed.returncode,
        "receiver_executable": {**exe_before, "sha256": sha256_file(RECEIVER)},
        "receiver_patch_sha256": build_manifest["receiver_repair_diff"]["sha256"],
        "receiver_stable_during_run": exe_before == exe_after,
        "receiver_config_path": str(config),
        "receiver_config_sha256": sha256_file(config),
        "frozen_handoff_path": str(handoff.resolve()),
        "frozen_handoff_sha256": sha256_file(handoff),
        "fixed_channel_prn_map": {row["channel"]: int(row["prn"]) for row in rows},
        "raw_iq": {**raw_before, "sha256": spec["sha256"]},
        "raw_iq_stable_during_run": raw_before == raw_after,
        "raw_sample_range": {"start_inclusive": raw_offset, "end_exclusive": None, "seconds_to_skip": skip_s, "duration_s": None},
        "dump_files": dumps,
        "replay_validation": replay_validation,
    }
    dump_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return completed.returncode if completed.returncode else (0 if replay_validation["status"] == "PASS" else 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit-raw")
    replay = sub.add_parser("run-receiver")
    replay.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
    replay.add_argument("--repetition", type=int, required=True)
    replay.add_argument("--attempt")
    phase_b = sub.add_parser("run-phase-b-family")
    phase_b.add_argument("--family", choices=("TEXBAT", "OAKBAT"), required=True)
    args = parser.parse_args()
    if args.command == "audit-raw":
        return audit_raw()
    if args.command == "run-receiver":
        return run_receiver(args.scenario, args.repetition, args.attempt)
    phase_a_path = ARTIFACT / "rep3_rep4_reproduction_metrics.json"
    if not phase_a_path.exists() or not json.loads(phase_a_path.read_text()).get("phase_b_authorized", False):
        raise RuntimeError("Phase B NOT_AUTHORIZED: preregistered Phase A PASS record is absent")
    names = [name for name in PHASE_B_SCENARIOS if name.startswith(args.family + ".")]
    for name in names:
        code = run_phase_b_receiver(name)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
