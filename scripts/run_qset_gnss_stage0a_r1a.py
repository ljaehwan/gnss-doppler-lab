#!/usr/bin/env python3
"""Run the preregistered Q-SET R1a Galileo C-1 terminal-drain repair."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_r1():
    path = REPO_ROOT / "scripts/run_qset_gnss_stage0a_r1.py"
    spec = importlib.util.spec_from_file_location("qset_r1_frozen", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen R1 runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R1 = _load_r1()
ARTIFACT_REL = Path("artifacts/qset_gnss_stage0a_r1a_galileo_c1_terminal_drain_repair")
ARTIFACT = REPO_ROOT / ARTIFACT_REL
R1_ARTIFACT = REPO_ROOT / "artifacts/qset_gnss_stage0a_r1_galileo_c1_receiver_preflight"
R1_OUTPUT_ROOT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/qset-gnss-stage0a-r1-galileo-c1-receiver-preflight")
OUTPUT_ROOT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/qset-gnss-stage0a-r1a-galileo-c1-terminal-drain-repair")
BASE_R1_FINAL_SHA = "cfa1dedfe330d74047e5ffcac2b9f94fd33f23a7"
PREREGISTRATION_SHA = "a4d734fa4bbbf8ac0656b3eebf66e945d575adec"
R1_FREEZE_SHA = "de44b6d448b9395a3c1a7199637b0df32cf51077"
BRANCH = "research/qset-gnss-stage0a-r1a-galileo-c1-terminal-drain-repair"
EXPECTED_R1_HASHES = {
    "final_verdict.json": "b2fd3d30dae14ca83b5f7007f68cbd70824255763eead89c922c99ff24d4e41a",
    "receiver_run_manifest.json": "9b02caddbd493f6c24360dcbbe02fc95c7eb29fd6188f224d2233001f104d80a",
    "execution_interrupt_audit.json": "9307abe804f75669685a3f59de1f1fdbe7a18911724095b84ff92dd70e65e52b",
    "artifact_manifest_sha256.json": "bb3ce40a9a6b369564672f278f27564f4eded5201ba7c765ed9aa3c1e3656566",
}
EXPECTED_CONFIG_SHA256 = "6dda45478b1b2521605e4544a51c84e513872b7bd433bb308d888dccd5fe4061"
EXPECTED_RECEIVER_SHA256 = "9e523c156c288cae401628ba308bc886bcb7c7989e35747b7482b1d50a0c6131"
TRACK_RECORD_BYTES = 96
EOF_MARGIN_OUTPUT_SAMPLES = 4_000_000
POST_EOF_QUIET_S = 5.0
STOP_GRACE_S = 60.0
MAX_RUNTIME_S = 3600.0
POLL_S = 0.5
EXPECTED_CHANNELS = 12
VERDICTS = {
    "READY_FOR_TUNI2025_C3_CLEAN_DOWNLOAD",
    "BLOCKED_C1_FORMAT_UNRESOLVED",
    "BLOCKED_GALILEO_E1_RECEIVER_NOT_AVAILABLE",
    "BLOCKED_INSUFFICIENT_GALILEO_PRN_SUPPORT",
    "BLOCKED_RECEIVER_TIME_MAPPING",
    "INCONCLUSIVE_RECEIVER_FEASIBILITY",
}


class RepairError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RepairError(message)


def git(*args: str, cwd: Path = REPO_ROOT, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True)
    if check and result.returncode:
        raise RepairError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    return R1.sha256_file(path, chunk)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    R1.write_json(path, payload)


def read_json(path: Path) -> dict[str, Any]:
    return R1.read_json(path)


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def output_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append({"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    aggregate = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"files": rows, "file_count": len(rows), "aggregate_sha256": aggregate}


def audit_preserved_r1_large_output() -> dict[str, Any]:
    manifest = read_json(R1_ARTIFACT / "receiver_run_manifest.json")
    checked_files = 0
    checked_bytes = 0
    for segment in manifest["segments"]:
        root = R1_OUTPUT_ROOT / segment["segment_id"]
        decoder = root / "c1_4msps_gr_complex.bin"
        require(decoder.is_file(), f"missing preserved R1 decoder: {decoder}")
        require(decoder.stat().st_size == int(segment["decoder"]["output_size_bytes"]), "preserved R1 decoder size drift")
        require(sha256_file(decoder) == segment["decoder"]["output_sha256"], "preserved R1 decoder hash drift")
        checked_files += 1
        checked_bytes += decoder.stat().st_size
        for row in segment["receiver"]["output_set"]["files"]:
            path = root / "receiver" / row["path"]
            require(path.is_file() and path.stat().st_size == int(row["size_bytes"]), f"preserved R1 output size drift: {path}")
            require(sha256_file(path) == row["sha256"], f"preserved R1 output hash drift: {path}")
            checked_files += 1
            checked_bytes += path.stat().st_size
    return {"status": "PASS", "files_checked": checked_files, "bytes_checked": checked_bytes}


def _signal_group(process: subprocess.Popen[Any], sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def supervise_receiver(
    command: list[str],
    cwd: Path,
    log_path: Path,
    output_samples: int,
    *,
    max_runtime_s: float = MAX_RUNTIME_S,
    quiet_s: float = POST_EOF_QUIET_S,
    stop_grace_s: float = STOP_GRACE_S,
    poll_s: float = POLL_S,
) -> dict[str, Any]:
    """Stop GNSS-SDR only after its acquisition stamp proves bounded-source EOF."""
    threshold = output_samples - EOF_MARGIN_OUTPUT_SAMPLES
    started = time.monotonic()
    max_stamp: int | None = None
    last_stamp_event: float | None = None
    controlled_signal_sent = False
    forced_signal: str | None = None
    stop_deadline: float | None = None
    read_offset = 0
    pending = ""
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            now = time.monotonic()
            if log_path.exists():
                with log_path.open("rb") as reader:
                    reader.seek(read_offset)
                    block = reader.read()
                    read_offset += len(block)
                if block:
                    pending += block.decode("utf-8", errors="replace")
                    lines = pending.splitlines(keepends=True)
                    if lines and not lines[-1].endswith(("\n", "\r")):
                        pending = lines.pop()
                    else:
                        pending = ""
                    for line in lines:
                        matches = re.findall(r"sample[ _]stamp[:, ]+(\d+)", line, flags=re.IGNORECASE)
                        if matches:
                            max_stamp = max([max_stamp or 0, *(int(value) for value in matches)])
                            last_stamp_event = now
            eof_confirmed = max_stamp is not None and max_stamp >= threshold
            if not controlled_signal_sent and eof_confirmed and last_stamp_event is not None and now - last_stamp_event >= quiet_s:
                _signal_group(process, signal.SIGINT)
                controlled_signal_sent = True
                stop_deadline = now + stop_grace_s
            if not controlled_signal_sent and now - started >= max_runtime_s:
                forced_signal = "SIGINT_MAX_RUNTIME_WITHOUT_EOF"
                _signal_group(process, signal.SIGINT)
                controlled_signal_sent = True
                stop_deadline = now + stop_grace_s
            if controlled_signal_sent and stop_deadline is not None and now >= stop_deadline:
                forced_signal = forced_signal or "SIGTERM_AFTER_STOP_GRACE"
                _signal_group(process, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    forced_signal = "SIGKILL_AFTER_TERM_GRACE"
                    _signal_group(process, signal.SIGKILL)
                break
            time.sleep(poll_s)
        exit_code = process.wait()
    elapsed = time.monotonic() - started
    text = log_path.read_text(encoding="utf-8", errors="replace")
    markers = {
        "received_sigint": "GNSS-SDR received 2 OS signal" in text,
        "received_stop": "Received action STOP" in text,
        "flowgraph_stopped": "Flowgraph stopped" in text,
    }
    eof_confirmed = max_stamp is not None and max_stamp >= threshold
    terminal_drain = bool(eof_confirmed and forced_signal is None and controlled_signal_sent and exit_code == 0 and all(markers.values()))
    return {
        "status": "PASS" if terminal_drain else "FAIL_CLOSED",
        "exit_code": exit_code,
        "elapsed_s": elapsed,
        "output_samples": output_samples,
        "eof_threshold_sample": threshold,
        "max_acquisition_sample_stamp": max_stamp,
        "eof_evidence_pass": eof_confirmed,
        "quiet_period_s": quiet_s,
        "controlled_signal": "SIGINT" if controlled_signal_sent else None,
        "forced_signal": forced_signal,
        "markers": markers,
        "terminal_drain": terminal_drain,
    }


def tracking_dump_inventory(receiver_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for channel in range(EXPECTED_CHANNELS):
        path = receiver_dir / f"veml_tracking_ch_{channel}.dat"
        require(path.is_file(), f"missing tracking dump: {path}")
        size = path.stat().st_size
        complete_bytes = size - size % TRACK_RECORD_BYTES
        with path.open("rb") as stream:
            stream.seek(complete_bytes)
            tail = stream.read()
        require(len(tail) < TRACK_RECORD_BYTES, "tracking tail bound")
        rows.append({
            "channel": channel,
            "path": path.name,
            "size_bytes": size,
            "sha256": sha256_file(path),
            "complete_record_count": complete_bytes // TRACK_RECORD_BYTES,
            "complete_bytes": complete_bytes,
            "trailing_fragment_size_bytes": len(tail),
            "trailing_fragment_sha256": hashlib.sha256(tail).hexdigest(),
            "status": "EMPTY_UNAVAILABLE_CHANNEL" if size == 0 else ("COMPLETE_RECORDS_ONLY" if not tail else "COMPLETE_RECORDS_WITH_PRESERVED_TRAILING_FRAGMENT"),
        })
    return rows


def read_complete_tracking_records(path: Path) -> np.ndarray:
    require(path.is_file(), f"missing tracking dump: {path}")
    count = path.stat().st_size // TRACK_RECORD_BYTES
    return np.fromfile(path, dtype=R1.TRACK_RECORD_DTYPE, count=count)


def run_receiver(segment_dir: Path, decoder_path: Path, decoder_info: dict[str, Any]) -> dict[str, Any]:
    receiver_dir = segment_dir / "receiver"
    require(not receiver_dir.exists(), f"refusing to overwrite receiver output: {receiver_dir}")
    receiver_dir.mkdir(parents=True)
    template_path = REPO_ROOT / R1.CONFIG_TEMPLATE_REL
    require(sha256_file(template_path) == EXPECTED_CONFIG_SHA256, "frozen receiver template drift")
    require(sha256_file(R1.SYSTEM_RECEIVER) == EXPECTED_RECEIVER_SHA256, "frozen receiver binary drift")
    config_path = receiver_dir / "receiver.conf"
    tracking_prefix = receiver_dir / "veml_tracking_ch_"
    config_path.write_text(R1.render_config(template_path.read_text(encoding="utf-8"), decoder_path, tracking_prefix, int(decoder_info["output_samples"])), encoding="utf-8")
    log_path = receiver_dir / "receiver.log"
    command = [str(R1.SYSTEM_RECEIVER), f"--config_file={config_path}", "--keyboard=false", "--logtostderr=true", "--logbufsecs=0"]
    supervision = supervise_receiver(command, receiver_dir, log_path, int(decoder_info["output_samples"]))
    dumps = tracking_dump_inventory(receiver_dir)
    manifest = output_manifest(receiver_dir)
    return {
        "command": command,
        "config_sha256": sha256_file(config_path),
        "receiver_sha256": sha256_file(R1.SYSTEM_RECEIVER),
        "exit_code": supervision["exit_code"],
        "terminal_eof": supervision["terminal_drain"],
        "terminal_drain": supervision,
        "tracking_dump_count": len(dumps),
        "nonempty_tracking_dump_count": sum(row["size_bytes"] > 0 for row in dumps),
        "tracking_dump_inventory": dumps,
        "output_set": manifest,
    }


def analyze_outputs(run_manifest: dict[str, Any], output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    observations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    segment_summary: dict[str, dict[str, Any]] = {}
    any_nonfinite = False
    time_mapping_ok = True
    telemetry_union: set[int] = set()
    for segment in run_manifest["segments"]:
        segment_id = segment["segment_id"]
        receiver_dir = output_root / segment_id / "receiver"
        start_sample = int(segment["source_start_sample"])
        output_samples = int(segment["decoder"]["output_samples"])
        per_segment_prns: set[int] = set()
        telemetry = R1.parse_telemetry((receiver_dir / "receiver.log").read_text(encoding="utf-8", errors="replace"))
        telemetry_union.update(telemetry)
        for channel in range(EXPECTED_CHANNELS):
            dump = receiver_dir / f"veml_tracking_ch_{channel}.dat"
            records = read_complete_tracking_records(dump)
            if not len(records):
                continue
            for field in R1.FLOAT_TRACK_FIELDS:
                if not np.isfinite(records[field]).all():
                    any_nonfinite = True
            for prn in sorted(set(int(value) for value in records["prn"] if 1 <= int(value) <= 36)):
                selected = records[records["prn"] == prn]
                per_segment_prns.add(prn)
                for begin, end in R1.continuous_runs(selected["sample"]):
                    run = selected[begin:end]
                    if not len(run):
                        continue
                    first_output = int(run["sample"][0])
                    last_output = int(run["sample"][-1])
                    duration = max(0.0, (last_output - first_output) / R1.OUTPUT_SAMPLE_RATE + 0.004)
                    effective_first = start_sample + first_output * R1.DECIMATION / R1.INTERPOLATION - R1.RESAMPLER_GROUP_DELAY_INPUT_SAMPLES
                    effective_last = start_sample + last_output * R1.DECIMATION / R1.INTERPOLATION - R1.RESAMPLER_GROUP_DELAY_INPUT_SAMPLES
                    mapping_valid = 0 <= first_output <= output_samples + 80_000 and 0 <= last_output <= output_samples + 80_000
                    mapping_valid = mapping_valid and effective_first >= start_sample - 1_000 and effective_last <= start_sample + int(segment["source_sample_count"]) + 1_000
                    time_mapping_ok = time_mapping_ok and mapping_valid
                    observations[prn].append({
                        "segment_id": segment_id,
                        "channel_file": dump.name,
                        "records": int(len(run)),
                        "duration_s": duration,
                        "first_output_sample": first_output,
                        "last_output_sample": last_output,
                        "effective_first_raw_sample": effective_first,
                        "effective_last_raw_sample": effective_last,
                        "mapping_valid": mapping_valid,
                        "doppler_median_hz": float(np.median(run["doppler_hz"])),
                        "doppler_min_hz": float(np.min(run["doppler_hz"])),
                        "doppler_max_hz": float(np.max(run["doppler_hz"])),
                        "cn0_median_db_hz": float(np.median(run["cn0_db_hz"])),
                        "cn0_min_db_hz": float(np.min(run["cn0_db_hz"])),
                        "cn0_max_db_hz": float(np.max(run["cn0_db_hz"])),
                    })
        segment_summary[segment_id] = {
            "observed_prns": sorted(per_segment_prns),
            "telemetry_sync_prns": telemetry,
            "observed_prn_count": len(per_segment_prns),
        }
    rows = []
    for prn in sorted(observations):
        runs = observations[prn]
        longest = max(row["duration_s"] for row in runs)
        rows.append({
            "prn": prn,
            "acquired": True,
            "segments_observed": ";".join(sorted({row["segment_id"] for row in runs})),
            "run_count": len(runs),
            "longest_continuous_tracking_s": longest,
            "tracking_ge_10s": longest >= 10.0,
            "all_finite": not any_nonfinite,
            "telemetry_sync": prn in telemetry_union,
        })
    segment_ids = [item["id"] for item in R1.RECEIVER_WINDOWS]
    panel_a, panel_b = (set(segment_summary[item]["observed_prns"]) for item in segment_ids)
    common = sorted(panel_a & panel_b)
    union = sorted(panel_a | panel_b)
    physical_checks = []
    for prn in common:
        best = {}
        for segment_id in segment_ids:
            candidates = [row for row in observations[prn] if row["segment_id"] == segment_id]
            best[segment_id] = max(candidates, key=lambda row: row["duration_s"])
        physical_checks.append({
            "prn": prn,
            "doppler_delta_hz": abs(best[segment_ids[1]]["doppler_median_hz"] - best[segment_ids[0]]["doppler_median_hz"]),
            "cn0_delta_db": abs(best[segment_ids[1]]["cn0_median_db_hz"] - best[segment_ids[0]]["cn0_median_db_hz"]),
            "doppler_in_search_range": all(abs(best[item]["doppler_median_hz"]) <= 15_000 for item in segment_ids),
            "cn0_physical": all(15.0 <= best[item]["cn0_median_db_hz"] <= 65.0 for item in segment_ids),
        })
    physical_consistency = (
        len(common) >= 4 and len(panel_a) >= 4 and len(panel_b) >= 4
        and all(row["doppler_delta_hz"] <= 500.0 and row["cn0_delta_db"] <= 15.0 and row["doppler_in_search_range"] and row["cn0_physical"] for row in physical_checks)
    )
    return {
        "per_prn": rows,
        "runs": {str(prn): entries for prn, entries in sorted(observations.items())},
        "segments": segment_summary,
        "availability_mask": {item: {str(prn): prn in set(segment_summary[item]["observed_prns"]) for prn in union} for item in segment_ids},
        "dynamic_panel": union,
        "common_panel": common,
        "acquisition_prn_count": len(rows),
        "tracking_ge_10s_prn_count": sum(bool(row["tracking_ge_10s"]) for row in rows),
        "all_tracking_finite": not any_nonfinite,
        "time_mapping_pass": time_mapping_ok,
        "physical_consistency_pass": physical_consistency,
        "physical_consistency_details": physical_checks,
        "telemetry_sync_prns": sorted(telemetry_union),
    }


def prepare_freeze() -> None:
    require(git("rev-parse", "HEAD") == PREREGISTRATION_SHA, "prepare-freeze must start at pushed preregistration SHA")
    require(git("diff", "--name-only", "--", str(ARTIFACT_REL / "repair_preregistration.json"), str(ARTIFACT_REL / "r1_preservation_audit.json")) == "", "preregistered contract drift")
    require(git("rev-parse", f"origin/{BRANCH}") == PREREGISTRATION_SHA, "preregistration SHA is not remote branch tip")
    require(git("rev-parse", "main") == git("rev-parse", "origin/main") == "461eb4dc7bb794e719295daf028f6811658ba37f", "main drift")
    for name, expected in EXPECTED_R1_HASHES.items():
        require(sha256_file(R1_ARTIFACT / name) == expected, f"R1 compact evidence drift: {name}")
    require(sha256_file(REPO_ROOT / R1.CONFIG_TEMPLATE_REL) == EXPECTED_CONFIG_SHA256, "frozen config template drift")
    require(sha256_file(R1.SYSTEM_RECEIVER) == EXPECTED_RECEIVER_SHA256, "receiver binary drift")
    preserved_large = audit_preserved_r1_large_output()
    code_paths = (
        Path("scripts/run_qset_gnss_stage0a_r1a.py"),
        Path("scripts/verify_qset_gnss_stage0a_r1a.py"),
        Path("tests/test_qset_gnss_stage0a_r1a.py"),
        Path("scripts/run_qset_gnss_stage0a_r1.py"),
        Path("scripts/verify_qset_gnss_stage0a_r1.py"),
        Path("tests/test_qset_gnss_stage0a_r1.py"),
        R1.CONFIG_TEMPLATE_REL,
    )
    bindings = {str(path): sha256_file(REPO_ROOT / path) for path in code_paths}
    source = {
        "schema": "gnss-doppler-lab.qset-r1a-source-binding.v1",
        "status": "PASS_PRE_EXECUTION",
        "base_r1_final_sha": BASE_R1_FINAL_SHA,
        "r1_freeze_sha": R1_FREEZE_SHA,
        "preregistration_sha": PREREGISTRATION_SHA,
        "c1": {"path": str(R1.RAW), "size_bytes": R1.RAW_SIZE, "md5": R1.RAW_MD5},
        "readme": {"path": str(R1.README_PDF), "md5": R1.README_MD5},
        "receiver": {"path": str(R1.SYSTEM_RECEIVER), "sha256": EXPECTED_RECEIVER_SHA256, "version": R1.version_output(R1.SYSTEM_RECEIVER)},
        "receiver_configuration_sha256": EXPECTED_CONFIG_SHA256,
        "code_bindings": bindings,
        "preserved_r1_large_output": preserved_large,
        "main_local_sha": git("rev-parse", "main"),
        "main_remote_sha": git("rev-parse", "origin/main"),
    }
    freeze = {
        "schema": "gnss-doppler-lab.qset-r1a-execution-repair-freeze.v1",
        "status": "FROZEN_PRE_EXECUTION",
        "code_bindings": bindings,
        "command": "/usr/bin/python3 scripts/run_qset_gnss_stage0a_r1a.py execute --freeze-sha <pushed-repair-freeze-sha>",
        "worker_count": 1,
        "sequential_order": [item["id"] for item in R1.RECEIVER_WINDOWS],
        "output_root": str(OUTPUT_ROOT),
        "terminal_drain": {"eof_margin_output_samples": EOF_MARGIN_OUTPUT_SAMPLES, "post_eof_quiet_s": POST_EOF_QUIET_S, "signal": "SIGINT", "stop_grace_s": STOP_GRACE_S, "required_exit_code": 0},
        "complete_record_adapter": {"record_size_bytes": TRACK_RECORD_BYTES, "trailing_fragment_max_bytes": TRACK_RECORD_BYTES - 1, "missing_dump": "FAIL_CLOSED"},
        "scientific_contract_changed": False,
        "environment": {"python": platform.python_version(), "os": platform.platform(), "numpy": np.__version__},
    }
    write_json(ARTIFACT / "preregistration_commit.json", {"schema": "gnss-doppler-lab.qset-r1a-preregistration-commit.v1", "status": "PASS", "commit_sha": PREREGISTRATION_SHA, "remote_branch": BRANCH})
    write_json(ARTIFACT / "source_binding.json", source)
    write_json(ARTIFACT / "execution_repair_freeze.json", freeze)
    placeholders = {
        "freeze_commit.json": {"status": "PENDING_REPAIR_FREEZE_COMMIT"},
        "format_identification.json": {"status": "PENDING_FROZEN_EXECUTION"},
        "decoder_validation.json": {"status": "PENDING_FROZEN_EXECUTION"},
        "receiver_run_manifest.json": {"status": "PENDING_FROZEN_EXECUTION"},
        "tracking_dump_inventory.json": {"status": "PENDING_FROZEN_EXECUTION"},
        "support_audit.json": {"status": "PENDING_FROZEN_EXECUTION"},
        "time_mapping_validation.json": {"status": "PENDING_FROZEN_EXECUTION"},
        "deterministic_reproduction.json": {"status": "PENDING_FROZEN_EXECUTION"},
        "access_audit.json": {"status": "PENDING_FROZEN_EXECUTION"},
        "final_verdict.json": {"status": "PENDING_FROZEN_EXECUTION"},
    }
    for name, payload in placeholders.items():
        write_json(ARTIFACT / name, payload)
    csv_write(ARTIFACT / "acquisition_inventory.csv", [], ["prn", "acquired", "segments_observed", "run_count", "longest_continuous_tracking_s", "tracking_ge_10s", "all_finite", "telemetry_sync"])
    csv_write(ARTIFACT / "per_prn_tracking_support.csv", [], ["prn", "acquired", "segments_observed", "run_count", "longest_continuous_tracking_s", "tracking_ge_10s", "all_finite", "telemetry_sync"])
    (ARTIFACT / "test_output.txt").write_text("PENDING FINAL TESTS\n", encoding="utf-8")
    (ARTIFACT / "verifier_output.txt").write_text("PENDING FINAL VERIFIER\n", encoding="utf-8")
    (ARTIFACT / "README.md").write_text("# Q-SET-GNSS Stage-0A R1a Galileo C-1 terminal-drain repair\n\nExecution repair is frozen and pending. R1 evidence is preserved. No C-3 or attack access is authorized.\n", encoding="utf-8")


def _forbidden_zero() -> dict[str, int]:
    return {key: 0 for key in ("stats", "hashes", "opens", "mmaps", "bytes_read", "downloads")}


def write_inconclusive(freeze_sha: str, audit: dict[str, Any], reason: str, run_manifest: dict[str, Any] | None = None) -> None:
    audit["status"] = "FAIL_CLOSED"
    write_json(ARTIFACT / "access_audit.json", audit)
    if run_manifest is not None:
        write_json(ARTIFACT / "receiver_run_manifest.json", run_manifest)
    write_json(ARTIFACT / "final_verdict.json", {
        "schema": "gnss-doppler-lab.qset-r1a-final-verdict.v1",
        "status": "PASS",
        "verdict": "INCONCLUSIVE_RECEIVER_FEASIBILITY",
        "reason": reason,
        "base_r1_final_sha": BASE_R1_FINAL_SHA,
        "freeze_sha": freeze_sha,
        "c3_clean_download_authorized": False,
        "attack_download_or_access_authorized": False,
        "next_state": "NOT_AUTHORIZED",
    })


def execute(freeze_sha: str) -> None:
    require(git("rev-parse", "HEAD") == freeze_sha, "execution checkout is not repair-freeze SHA")
    require(git("status", "--porcelain") == "", "execution checkout is not clean")
    require(git("rev-parse", f"origin/{BRANCH}") == freeze_sha, "repair-freeze SHA is not remote branch tip")
    freeze = read_json(ARTIFACT / "execution_repair_freeze.json")
    for relative, expected in freeze["code_bindings"].items():
        require(sha256_file(REPO_ROOT / relative) == expected, f"frozen executable drift: {relative}")
    require(not OUTPUT_ROOT.exists(), f"large output root already exists: {OUTPUT_ROOT}")
    require(os.statvfs(OUTPUT_ROOT.parent).f_bavail * os.statvfs(OUTPUT_ROOT.parent).f_frsize >= R1.MIN_FREE_BYTES, "insufficient disk")
    OUTPUT_ROOT.mkdir(parents=True)
    audit = {
        "schema": "gnss-doppler-lab.qset-r1a-access-audit.v1",
        "status": "RUNNING",
        "c1": {"identity_hash_passes": 0, "identity_hash_bytes": 0, "format_window_bytes": 0, "receiver_decode_bytes": 0, "total_payload_bytes_read": 0},
        "c3": _forbidden_zero(),
        "attack": _forbidden_zero(),
        "other_tuni2025_raw": _forbidden_zero(),
    }
    write_json(ARTIFACT / "access_audit.json", audit)
    print("[1/5] verifying C-1 identity and frozen bounded format", flush=True)
    require(R1.RAW.is_file() and R1.RAW.stat().st_size == R1.RAW_SIZE, "C-1 raw size mismatch")
    require(R1.md5_file(R1.RAW) == R1.RAW_MD5, "C-1 raw MD5 mismatch")
    audit["c1"]["identity_hash_passes"] = 1
    audit["c1"]["identity_hash_bytes"] = R1.RAW_SIZE
    audit["c1"]["total_payload_bytes_read"] += R1.RAW_SIZE
    fmt = R1.identify_format(R1.RAW)
    audit["c1"]["format_window_bytes"] = len(R1.FORMAT_WINDOWS) * R1.FORMAT_WINDOW_BYTES
    audit["c1"]["total_payload_bytes_read"] += audit["c1"]["format_window_bytes"]
    write_json(ARTIFACT / "format_identification.json", fmt)
    if fmt["status"] != "PASS":
        write_inconclusive(freeze_sha, audit, "frozen C-1 format reproduction failed")
        return
    sample_bytes = bytes.fromhex("ff8501c80315febf")
    decoder_unit_ok = R1.decoder_round_trip(sample_bytes)
    require(decoder_unit_ok, "inherited decoder unit validation failed")
    print("[2/5] decoding and running two unchanged receiver windows", flush=True)
    segments = []
    all_dumps = []
    for window in R1.RECEIVER_WINDOWS:
        segment_id = str(window["id"])
        segment_dir = OUTPUT_ROOT / segment_id
        require(not segment_dir.exists(), f"segment output exists: {segment_dir}")
        segment_dir.mkdir()
        start_sample = int(float(window["start_s"]) * R1.RAW_SAMPLE_RATE)
        sample_count = int(float(window["duration_s"]) * R1.RAW_SAMPLE_RATE)
        decoder_path = segment_dir / "c1_4msps_gr_complex.bin"
        print(f"  decode {segment_id}", flush=True)
        decoder = R1.stream_decode_resample(R1.RAW, decoder_path, start_sample, sample_count)
        write_json(segment_dir / "decoder_sidecar.json", decoder)
        audit["c1"]["receiver_decode_bytes"] += decoder["source_bytes_read"]
        audit["c1"]["total_payload_bytes_read"] += decoder["source_bytes_read"]
        print(f"  receiver {segment_id}", flush=True)
        receiver = run_receiver(segment_dir, decoder_path, decoder)
        item = {"segment_id": segment_id, "source_start_sample": start_sample, "source_sample_count": sample_count, "decoder": decoder, "receiver": receiver}
        segments.append(item)
        all_dumps.extend({"segment_id": segment_id, **row} for row in receiver["tracking_dump_inventory"])
        run_manifest = {
            "schema": "gnss-doppler-lab.qset-r1a-receiver-run-manifest.v1",
            "status": "RUNNING",
            "freeze_sha": freeze_sha,
            "output_root": str(OUTPUT_ROOT),
            "worker_count": 1,
            "segments": segments,
        }
        write_json(ARTIFACT / "receiver_run_manifest.json", run_manifest)
        write_json(ARTIFACT / "tracking_dump_inventory.json", {"schema": "gnss-doppler-lab.qset-r1a-tracking-dump-inventory.v1", "status": "RUNNING", "record_size_bytes": TRACK_RECORD_BYTES, "rows": all_dumps})
        write_json(ARTIFACT / "access_audit.json", audit)
        if not receiver["terminal_drain"]["terminal_drain"]:
            run_manifest["status"] = "FAIL_CLOSED_TERMINAL_DRAIN"
            write_inconclusive(freeze_sha, audit, f"terminal drain failed for {segment_id}", run_manifest)
            return
    run_manifest = {
        "schema": "gnss-doppler-lab.qset-r1a-receiver-run-manifest.v1",
        "status": "PASS",
        "freeze_sha": freeze_sha,
        "output_root": str(OUTPUT_ROOT),
        "worker_count": 1,
        "segments": segments,
    }
    write_json(ARTIFACT / "receiver_run_manifest.json", run_manifest)
    write_json(ARTIFACT / "tracking_dump_inventory.json", {"schema": "gnss-doppler-lab.qset-r1a-tracking-dump-inventory.v1", "status": "PASS", "record_size_bytes": TRACK_RECORD_BYTES, "rows": all_dumps})
    decoder_validation = {
        "schema": "gnss-doppler-lab.qset-r1a-decoder-validation.v1",
        "status": "PASS",
        "input_dtype": ">i2",
        "interleaving": "I,Q",
        "normalization_scale_factor": 1.0,
        "unit_round_trip_pass": decoder_unit_ok,
        "source_windows": [{"segment_id": item["segment_id"], **item["decoder"]} for item in segments],
    }
    write_json(ARTIFACT / "decoder_validation.json", decoder_validation)
    print("[3/5] analyzing complete native dump records", flush=True)
    analysis_one = analyze_outputs(run_manifest)
    analysis_two = analyze_outputs(run_manifest)
    canonical_one = json.dumps(analysis_one, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    canonical_two = json.dumps(analysis_two, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    deterministic = {
        "schema": "gnss-doppler-lab.qset-r1a-deterministic.v1",
        "status": "PASS" if canonical_one == canonical_two else "FAIL",
        "analysis_runs": 2,
        "byte_identical": canonical_one == canonical_two,
        "analysis_sha256": hashlib.sha256(canonical_one).hexdigest(),
    }
    write_json(ARTIFACT / "deterministic_reproduction.json", deterministic)
    fields = ["prn", "acquired", "segments_observed", "run_count", "longest_continuous_tracking_s", "tracking_ge_10s", "all_finite", "telemetry_sync"]
    csv_write(ARTIFACT / "acquisition_inventory.csv", analysis_one["per_prn"], fields)
    csv_write(ARTIFACT / "per_prn_tracking_support.csv", analysis_one["per_prn"], fields)
    mapping = {
        "schema": "gnss-doppler-lab.qset-r1a-time-mapping.v1",
        "status": "PASS" if analysis_one["time_mapping_pass"] else "FAIL",
        "mapping": "effective_raw_sample = window_start_raw_sample + receiver_output_sample*25/2 - 205.25",
        "raw_sample_rate_hz": R1.RAW_SAMPLE_RATE,
        "receiver_sample_rate_hz": R1.OUTPUT_SAMPLE_RATE,
        "all_tracking_records_within_source_window": analysis_one["time_mapping_pass"],
    }
    write_json(ARTIFACT / "time_mapping_validation.json", mapping)
    write_json(ARTIFACT / "support_audit.json", {"schema": "gnss-doppler-lab.qset-r1a-support-audit.v1", "status": "PASS", **analysis_one})
    audit["status"] = "PASS"
    write_json(ARTIFACT / "access_audit.json", audit)
    print("[4/5] applying unchanged R1 receiver-feasibility gates", flush=True)
    technical = run_manifest["status"] == "PASS" and deterministic["status"] == "PASS"
    if not technical:
        verdict, reason = "INCONCLUSIVE_RECEIVER_FEASIBILITY", "receiver execution or deterministic analysis incomplete"
    elif not analysis_one["time_mapping_pass"]:
        verdict, reason = "BLOCKED_RECEIVER_TIME_MAPPING", "receiver/raw sample mapping gate failed"
    elif analysis_one["acquisition_prn_count"] < 4 or analysis_one["tracking_ge_10s_prn_count"] < 4 or not analysis_one["all_tracking_finite"] or not analysis_one["physical_consistency_pass"]:
        verdict, reason = "BLOCKED_INSUFFICIENT_GALILEO_PRN_SUPPORT", "one or more unchanged acquisition/tracking/finite/two-window physical support gates failed"
    else:
        verdict, reason = "READY_FOR_TUNI2025_C3_CLEAN_DOWNLOAD", "all unchanged C-1 format and Galileo E1 receiver feasibility gates passed after engineering terminal-drain repair"
    require(verdict in VERDICTS, "verdict vocabulary")
    final = {
        "schema": "gnss-doppler-lab.qset-r1a-final-verdict.v1",
        "status": "PASS",
        "verdict": verdict,
        "reason": reason,
        "base_r1_final_sha": BASE_R1_FINAL_SHA,
        "freeze_sha": freeze_sha,
        "actual_format": fmt["selected_format"],
        "acquisition_prn_count": analysis_one["acquisition_prn_count"],
        "tracking_ge_10s_prn_count": analysis_one["tracking_ge_10s_prn_count"],
        "telemetry_sync_prns": analysis_one["telemetry_sync_prns"],
        "dynamic_panel": analysis_one["dynamic_panel"],
        "common_panel": analysis_one["common_panel"],
        "gates": {
            "identity_format": True,
            "receiver_terminal_drain": True,
            "acquisition": analysis_one["acquisition_prn_count"] >= 4,
            "tracking": analysis_one["tracking_ge_10s_prn_count"] >= 4,
            "finite": analysis_one["all_tracking_finite"],
            "physical_consistency": analysis_one["physical_consistency_pass"],
            "time_mapping": analysis_one["time_mapping_pass"],
            "dynamic_availability_mask": bool(analysis_one["dynamic_panel"]),
        },
        "qset_training_performed": False,
        "threshold_calibrated": False,
        "attack_scoring_performed": False,
        "detection_performance_claimed": False,
        "c3_clean_download_authorized": verdict == "READY_FOR_TUNI2025_C3_CLEAN_DOWNLOAD",
        "attack_download_or_access_authorized": False,
        "next_state": "C3_CLEAN_DOWNLOAD_ONLY" if verdict == "READY_FOR_TUNI2025_C3_CLEAN_DOWNLOAD" else "NOT_AUTHORIZED",
    }
    write_json(ARTIFACT / "final_verdict.json", final)
    write_json(ARTIFACT / "freeze_commit.json", {"schema": "gnss-doppler-lab.qset-r1a-freeze-commit.v1", "status": "PASS", "commit_sha": freeze_sha, "remote_branch": BRANCH})
    print("[5/5] writing compact R1a attestation", flush=True)
    (ARTIFACT / "README.md").write_text(f"""# Q-SET-GNSS Stage-0A R1a Galileo C-1 terminal-drain repair

Final verdict: `{verdict}`.

R1 remains preserved as historical inconclusive evidence. R1a changed only process supervision after bounded-source EOF and complete-record parsing of native dumps. The receiver configuration, decoder, windows, and success gates were unchanged.

- Actual format: `{fmt['selected_format']}`
- Acquired Galileo E1 PRNs: {analysis_one['acquisition_prn_count']}
- PRNs continuously tracked for at least 10 s: {analysis_one['tracking_ge_10s_prn_count']}
- Dynamic observed panel: {analysis_one['dynamic_panel']}
- Common two-window panel: {analysis_one['common_panel']}
- Telemetry-sync auxiliary PRNs: {analysis_one['telemetry_sync_prns']}
- New R1a C-1 payload bytes read: {audit['c1']['total_payload_bytes_read']}
- C-3/attack payload bytes read: 0/0

This step performed no Q-SET training, threshold calibration, attack scoring, or detection claim. Even a passing verdict authorizes only a future C-3 clean download; attack data remains unauthorized.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-freeze")
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--freeze-sha", required=True)
    args = parser.parse_args()
    if args.command == "prepare-freeze":
        prepare_freeze()
    else:
        execute(args.freeze_sha)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RepairError, R1.PreflightError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
