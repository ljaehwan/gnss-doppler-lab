#!/usr/bin/env python3
"""Execute the preregistered Q-SET Stage-0A R1 Galileo C-1 preflight."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_REL = Path("artifacts/qset_gnss_stage0a_r1_galileo_c1_receiver_preflight")
ARTIFACT = REPO_ROOT / ARTIFACT_REL
BASE_SHA = "3e6e17ed705bad33124cff234f36621dd782b384"
BRANCH = "research/qset-gnss-stage0a-r1-galileo-c1-receiver-preflight"
RAW = Path("/home/ubuntu/ssd_data/gnss-datasets/tuni2025/galileo/C-1/clearsky_signal_C-1.bin")
README_PDF = Path("/home/ubuntu/ssd_data/gnss-datasets/tuni2025/galileo/C-1/Readme.pdf")
ZENODO_JSON = Path("/home/ubuntu/ssd_data/gnss-datasets/tuni2025/galileo/C-1/zenodo_record_15470143.json")
SYSTEM_RECEIVER = Path("/usr/bin/gnss-sdr")
METHOD_A_RECEIVER = Path("/home/ubuntu/projects/gnss-doppler-lab/.tools/gnss-sdr-method-a-9tap")
EXAMPLE_CONFIG = Path("/usr/share/gnss-sdr/conf/gnss-sdr_Galileo_E1_ishort.conf")
CONFIG_TEMPLATE_REL = Path("configs/qset_galileo_e1_c1_preflight.conf.template")
OUTPUT_ROOT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/qset-gnss-stage0a-r1-galileo-c1-receiver-preflight")

RAW_SIZE = 29_999_832_000
RAW_MD5 = "4ff0e86938792bf3150c30d5f1481917"
README_MD5 = "317e2f82dc89cfbe36272630e3c4f5e3"
RAW_SAMPLE_RATE = 50_000_000
BYTES_PER_COMPLEX = 4
RAW_COMPLEX_SAMPLES = 7_499_958_000
RAW_DURATION_S = 149.99916
OUTPUT_SAMPLE_RATE = 4_000_000
INTERPOLATION = 2
DECIMATION = 25
RESAMPLER_FRACTIONAL_BW = 0.4
RESAMPLER_TAP_COUNT = 822
RESAMPLER_TAP_SHA256 = "f50d0eb8ee4e707ccb94387e4ec327e42cb2bd1e42a544afbc687f74ea31826f"
RESAMPLER_GROUP_DELAY_INPUT_SAMPLES = 205.25
RESAMPLER_EOF_LOSS_OUTPUT_SAMPLES = 34
FORMAT_WINDOW_BYTES = 4_194_304
FORMAT_WINDOWS = (
    {"id": "head", "offset_bytes": 0, "length_bytes": FORMAT_WINDOW_BYTES},
    {"id": "middle", "offset_bytes": 14_997_818_848, "length_bytes": FORMAT_WINDOW_BYTES},
    {"id": "tail", "offset_bytes": 29_995_637_696, "length_bytes": FORMAT_WINDOW_BYTES},
)
RECEIVER_WINDOWS = (
    {"id": "segment_030_060", "start_s": 30.0, "duration_s": 30.0},
    {"id": "segment_100_130", "start_s": 100.0, "duration_s": 30.0},
)
MIN_FREE_BYTES = 10_000_000_000
TRACK_RECORD_DTYPE = np.dtype([
    ("VE", "<f4"), ("E", "<f4"), ("P", "<f4"), ("L", "<f4"), ("VL", "<f4"),
    ("prompt_i", "<f4"), ("prompt_q", "<f4"), ("sample", "<u8"),
    ("acc_carrier_phase", "<f4"), ("doppler_hz", "<f4"), ("doppler_rate", "<f4"),
    ("code_freq", "<f4"), ("code_rate", "<f4"), ("carr_error", "<f4"),
    ("carr_nco", "<f4"), ("code_error", "<f4"), ("code_nco", "<f4"),
    ("cn0_db_hz", "<f4"), ("carrier_lock", "<f4"), ("aux1", "<f4"),
    ("aux2", "<f8"), ("prn", "<u4"),
])
FLOAT_TRACK_FIELDS = tuple(name for name in TRACK_RECORD_DTYPE.names or () if name not in {"sample", "prn"})


class PreflightError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str, cwd: Path = REPO_ROOT, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True)
    if check and result.returncode:
        raise PreflightError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def version_output(path: Path) -> str:
    result = subprocess.run([str(path), "--version"], text=True, capture_output=True, timeout=30)
    return (result.stdout + result.stderr).strip()


def file_identity(path: Path, algorithm: str = "sha256") -> dict[str, Any]:
    require(path.is_file(), f"missing file: {path}")
    digest = sha256_file(path) if algorithm == "sha256" else md5_file(path)
    return {"path": str(path), "size_bytes": path.stat().st_size, algorithm: digest}


def decoder_from_bytes(data: bytes) -> np.ndarray:
    require(len(data) % 4 == 0, "decoder input is not whole complex samples")
    words = np.frombuffer(data, dtype=">i2").astype("<i2", copy=False).reshape(-1, 2)
    output = words[:, 0].astype(np.float32) + 1j * words[:, 1].astype(np.float32)
    return output.astype("<c8", copy=False)


def decoder_round_trip(data: bytes) -> bool:
    decoded = decoder_from_bytes(data)
    words = np.empty((decoded.size, 2), dtype=">i2")
    words[:, 0] = decoded.real.astype(np.int16)
    words[:, 1] = decoded.imag.astype(np.int16)
    return words.tobytes() == data


def safe_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def distribution(values: np.ndarray, integer: bool) -> dict[str, Any]:
    values = np.asarray(values)
    finite = np.isfinite(values)
    finite_values = values[finite].astype(np.float64, copy=False)
    quantiles: dict[str, float | None] = {}
    for q in (0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0):
        quantiles[f"q{q:g}"] = safe_float(np.quantile(finite_values, q)) if finite_values.size else None
    saturation = np.mean((values == -32768) | (values == 32767)) if integer else 0.0
    return {
        "count": int(values.size), "finite_fraction": float(np.mean(finite)),
        "nonfinite_count": int(values.size - finite_values.size),
        "mean": safe_float(np.mean(finite_values)) if finite_values.size else None,
        "std": safe_float(np.std(finite_values)) if finite_values.size else None,
        "zero_fraction": float(np.mean(values == 0)), "saturation_fraction": float(saturation),
        "quantiles": quantiles,
    }


def identify_format(path: Path, windows: Iterable[dict[str, int]] = FORMAT_WINDOWS, expected_size: int | None = RAW_SIZE) -> dict[str, Any]:
    if expected_size is not None:
        require(path.stat().st_size == expected_size, "raw size mismatch before format read")
    rows: list[dict[str, Any]] = []
    with path.open("rb", buffering=0) as stream:
        for window in windows:
            offset, length = int(window["offset_bytes"]), int(window["length_bytes"])
            require(offset % 4 == 0 and length % 8 == 0, "unaligned format window")
            require(offset >= 0 and offset + length <= path.stat().st_size, "format window out of bounds")
            stream.seek(offset)
            data = stream.read(length)
            require(len(data) == length, "short bounded format read")
            for dtype_name in ("<f4", ">f4", "<i2", ">i2"):
                values = np.frombuffer(data, dtype=np.dtype(dtype_name)).reshape(-1, 2)
                rows.append({
                    "window_id": str(window["id"]), "offset_bytes": offset, "length_bytes": length,
                    "dtype": dtype_name, "complex_samples": int(values.shape[0]),
                    "i": distribution(values[:, 0], dtype_name.endswith("i2")),
                    "q": distribution(values[:, 1], dtype_name.endswith("i2")),
                })
    be_rows = [row for row in rows if row["dtype"] == ">i2"]
    be_stds = [float(row[axis]["std"]) for row in be_rows for axis in ("i", "q")]
    be_ok = (
        len(be_rows) == 3 and all(1.0 <= value <= 100.0 for value in be_stds)
        and max(be_stds) / min(be_stds) <= 1.25
        and all(float(row[axis]["saturation_fraction"]) == 0.0 for row in be_rows for axis in ("i", "q"))
        and all(abs(float(row[axis]["quantiles"]["q0"])) < 2000 and abs(float(row[axis]["quantiles"]["q1"])) < 2000 for row in be_rows for axis in ("i", "q"))
    )
    float_rejections = {}
    for dtype_name in ("<f4", ">f4"):
        typed = [row for row in rows if row["dtype"] == dtype_name]
        nonfinite = any(float(row[axis]["finite_fraction"]) < 0.999999 for row in typed for axis in ("i", "q"))
        extreme = any(abs(float(row[axis]["quantiles"]["q0.999"] or 0.0)) > 1e6 for row in typed for axis in ("i", "q"))
        float_rejections[dtype_name] = {"rejected": bool(nonfinite or extreme), "nonfinite": nonfinite, "extreme_magnitude": extreme}
    little_i2 = [row for row in rows if row["dtype"] == "<i2"]
    little_stds = [float(row[axis]["std"]) for row in little_i2 for axis in ("i", "q")]
    little_rejected = min(little_stds) > 500.0
    status = "PASS" if be_ok and all(value["rejected"] for value in float_rejections.values()) and little_rejected else "FAIL"
    return {
        "schema": "gnss-doppler-lab.qset-r1-format-identification.v1", "status": status,
        "raw_size_bytes": path.stat().st_size, "bytes_per_complex_sample": BYTES_PER_COMPLEX,
        "complex_sample_count": path.stat().st_size // BYTES_PER_COMPLEX,
        "expected_duration_s": (path.stat().st_size // BYTES_PER_COMPLEX) / RAW_SAMPLE_RATE,
        "header_bytes": 0, "header_conclusion_basis": "exact official size and stationary head/middle/tail statistics; no guessed header",
        "selected_format": ">i2 interleaved I,Q" if status == "PASS" else None,
        "float32_rejection": float_rejections, "little_endian_i2_rejected": little_rejected,
        "windows": rows,
    }


def resampler_identity() -> dict[str, Any]:
    from gnuradio import filter as gr_filter
    block = gr_filter.rational_resampler_ccc(interpolation=INTERPOLATION, decimation=DECIMATION, fractional_bw=RESAMPLER_FRACTIONAL_BW)
    taps = block.taps()
    encoded = b"".join(struct.pack("<ff", float(value.real), float(value.imag)) for value in taps)
    return {"tap_count": len(taps), "tap_sha256": hashlib.sha256(encoded).hexdigest(), "declared_sample_delay": int(block.sample_delay(0))}


def stream_decode_resample(raw: Path, destination: Path, start_sample: int, sample_count: int) -> dict[str, Any]:
    from gnuradio import blocks, filter as gr_filter, gr
    identity = resampler_identity()
    require(identity["tap_count"] == RESAMPLER_TAP_COUNT, "resampler tap count drift")
    require(identity["tap_sha256"] == RESAMPLER_TAP_SHA256, "resampler taps drift")
    require(not destination.exists(), f"refusing to overwrite decoder output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    top = gr.top_block("qset-r1-be-i2-resampler", catch_exceptions=False)
    source = blocks.file_source(gr.sizeof_short, str(raw), False, start_sample * 2, sample_count * 2)
    endian = blocks.endian_swap(gr.sizeof_short)
    iq = blocks.interleaved_short_to_complex(False, False, 1.0)
    resampler = gr_filter.rational_resampler_ccc(interpolation=INTERPOLATION, decimation=DECIMATION, fractional_bw=RESAMPLER_FRACTIONAL_BW)
    sink = blocks.file_sink(gr.sizeof_gr_complex, str(destination), False)
    top.connect(source, endian, iq, resampler, sink)
    started = time.time(); top.run(); elapsed = time.time() - started
    output_samples = destination.stat().st_size // 8
    expected = sample_count * INTERPOLATION // DECIMATION - RESAMPLER_EOF_LOSS_OUTPUT_SAMPLES
    require(destination.stat().st_size % 8 == 0, "partial gr_complex output")
    require(abs(output_samples - expected) <= 1, f"resampler sample count mismatch: {output_samples} vs {expected}")
    return {
        "source_offset_samples": start_sample, "source_sample_count": sample_count,
        "source_offset_bytes": start_sample * BYTES_PER_COMPLEX, "source_bytes_read": sample_count * BYTES_PER_COMPLEX,
        "decoder": "big-endian signed int16 interleaved I,Q -> native float32 complex; scale_factor=1.0; no I/Q swap",
        "resampler": {"interpolation": INTERPOLATION, "decimation": DECIMATION, "fractional_bw": RESAMPLER_FRACTIONAL_BW, "eof_loss_output_samples": RESAMPLER_EOF_LOSS_OUTPUT_SAMPLES, **identity},
        "output_sample_rate_hz": OUTPUT_SAMPLE_RATE, "output_samples": output_samples,
        "output_size_bytes": destination.stat().st_size, "output_sha256": sha256_file(destination),
        "elapsed_s": elapsed,
    }


def render_config(template: str, input_path: Path, tracking_prefix: Path, sample_count: int) -> str:
    values = {"@@INPUT@@": str(input_path), "@@TRACK_PREFIX@@": str(tracking_prefix), "@@SAMPLES@@": str(sample_count)}
    for token, value in values.items():
        require(token in template, f"missing config token {token}")
        template = template.replace(token, value)
    require("@@" not in template, "unrendered config token")
    return template


def output_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append({"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    aggregate = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"files": rows, "file_count": len(rows), "aggregate_sha256": aggregate}


def run_receiver(segment_dir: Path, decoder_path: Path, decoder_info: dict[str, Any]) -> dict[str, Any]:
    receiver_dir = segment_dir / "receiver"
    require(not receiver_dir.exists(), f"refusing to overwrite receiver output: {receiver_dir}")
    receiver_dir.mkdir(parents=True)
    template = (REPO_ROOT / CONFIG_TEMPLATE_REL).read_text(encoding="utf-8")
    config_path = receiver_dir / "receiver.conf"
    tracking_prefix = receiver_dir / "veml_tracking_ch_"
    config_text = render_config(template, decoder_path, tracking_prefix, int(decoder_info["output_samples"]))
    config_path.write_text(config_text, encoding="utf-8")
    log_path = receiver_dir / "receiver.log"
    command = [str(SYSTEM_RECEIVER), f"--config_file={config_path}", "--keyboard=false", "--logtostderr=true", "--logbufsecs=0"]
    started = time.time()
    with log_path.open("wb") as log:
        result = subprocess.run(command, cwd=receiver_dir, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    elapsed = time.time() - started
    manifest = output_manifest(receiver_dir)
    dumps = [row for row in manifest["files"] if row["path"].endswith(".dat") and "veml_tracking_ch_" in row["path"]]
    return {
        "command": command, "exit_code": result.returncode, "elapsed_s": elapsed,
        "terminal_eof": result.returncode == 0, "config_sha256": sha256_file(config_path),
        "receiver_sha256": sha256_file(SYSTEM_RECEIVER), "tracking_dump_count": len(dumps),
        "nonempty_tracking_dump_count": sum(int(row["size_bytes"]) > 0 for row in dumps),
        "output_set": manifest,
    }


def read_tracking_dump(path: Path) -> np.ndarray:
    size = path.stat().st_size
    require(size % TRACK_RECORD_DTYPE.itemsize == 0, f"tracking dump record-size mismatch: {path}")
    return np.fromfile(path, dtype=TRACK_RECORD_DTYPE)


def continuous_runs(samples: np.ndarray) -> list[tuple[int, int]]:
    if not len(samples):
        return []
    breaks = np.where((np.diff(samples.astype(np.int64)) <= 0) | (np.diff(samples.astype(np.int64)) > 80_000))[0] + 1
    edges = np.r_[0, breaks, len(samples)]
    return [(int(edges[index]), int(edges[index + 1])) for index in range(len(edges) - 1)]


def parse_telemetry(log_text: str) -> list[int]:
    prns = set()
    for pattern in (r"Preamble detection for Galileo satellite Galileo E(\d+)", r"New Galileo E1 I/NAV message.*?E(\d+)"):
        prns.update(int(match) for match in re.findall(pattern, log_text))
    return sorted(prns)


def analyze_outputs(run_manifest: dict[str, Any]) -> dict[str, Any]:
    observations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    segment_summary: dict[str, dict[str, Any]] = {}
    any_nonfinite = False
    time_mapping_ok = True
    telemetry_union: set[int] = set()
    for segment in run_manifest["segments"]:
        segment_id = segment["segment_id"]
        segment_dir = OUTPUT_ROOT / segment_id
        start_sample = int(segment["source_start_sample"])
        output_samples = int(segment["decoder"]["output_samples"])
        per_segment_prns: set[int] = set()
        telemetry = parse_telemetry((segment_dir / "receiver" / "receiver.log").read_text(encoding="utf-8", errors="replace"))
        telemetry_union.update(telemetry)
        for dump in sorted((segment_dir / "receiver").glob("veml_tracking_ch_*.dat")):
            records = read_tracking_dump(dump)
            if not len(records):
                continue
            for field in FLOAT_TRACK_FIELDS:
                if not np.isfinite(records[field]).all():
                    any_nonfinite = True
            for prn in sorted(set(int(value) for value in records["prn"] if 1 <= int(value) <= 36)):
                selected = records[records["prn"] == prn]
                per_segment_prns.add(prn)
                for begin, end in continuous_runs(selected["sample"]):
                    run = selected[begin:end]
                    if not len(run):
                        continue
                    duration = max(0.0, (int(run["sample"][-1]) - int(run["sample"][0])) / OUTPUT_SAMPLE_RATE + 0.004)
                    first_output = int(run["sample"][0]); last_output = int(run["sample"][-1])
                    effective_first = start_sample + first_output * DECIMATION / INTERPOLATION - RESAMPLER_GROUP_DELAY_INPUT_SAMPLES
                    effective_last = start_sample + last_output * DECIMATION / INTERPOLATION - RESAMPLER_GROUP_DELAY_INPUT_SAMPLES
                    mapping_valid = 0 <= first_output <= output_samples + 80_000 and 0 <= last_output <= output_samples + 80_000
                    mapping_valid = mapping_valid and effective_first >= start_sample - 1_000 and effective_last <= start_sample + int(segment["source_sample_count"]) + 1_000
                    time_mapping_ok = time_mapping_ok and mapping_valid
                    observations[prn].append({
                        "segment_id": segment_id, "channel_file": dump.name, "records": int(len(run)),
                        "duration_s": duration, "first_output_sample": first_output, "last_output_sample": last_output,
                        "effective_first_raw_sample": effective_first, "effective_last_raw_sample": effective_last,
                        "mapping_valid": mapping_valid, "doppler_median_hz": float(np.median(run["doppler_hz"])),
                        "doppler_min_hz": float(np.min(run["doppler_hz"])), "doppler_max_hz": float(np.max(run["doppler_hz"])),
                        "cn0_median_db_hz": float(np.median(run["cn0_db_hz"])),
                        "cn0_min_db_hz": float(np.min(run["cn0_db_hz"])), "cn0_max_db_hz": float(np.max(run["cn0_db_hz"])),
                    })
        segment_summary[segment_id] = {"observed_prns": sorted(per_segment_prns), "telemetry_sync_prns": telemetry, "observed_prn_count": len(per_segment_prns)}
    rows = []
    for prn in sorted(observations):
        runs = observations[prn]
        rows.append({
            "prn": prn, "acquired": True, "segments_observed": ";".join(sorted({row["segment_id"] for row in runs})),
            "run_count": len(runs), "longest_continuous_tracking_s": max(row["duration_s"] for row in runs),
            "tracking_ge_10s": max(row["duration_s"] for row in runs) >= 10.0,
            "all_finite": not any_nonfinite,
            "telemetry_sync": prn in telemetry_union,
        })
    segment_ids = [item["id"] for item in RECEIVER_WINDOWS]
    panel_a, panel_b = (set(segment_summary[item]["observed_prns"]) for item in segment_ids)
    common = sorted(panel_a & panel_b); union = sorted(panel_a | panel_b)
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
    tracked_ge_10 = sum(bool(row["tracking_ge_10s"]) for row in rows)
    return {
        "per_prn": rows, "runs": {str(prn): entries for prn, entries in sorted(observations.items())},
        "segments": segment_summary, "availability_mask": {item: {str(prn): prn in set(segment_summary[item]["observed_prns"]) for prn in union} for item in segment_ids},
        "dynamic_panel": union, "common_panel": common, "acquisition_prn_count": len(rows),
        "tracking_ge_10s_prn_count": tracked_ge_10, "all_tracking_finite": not any_nonfinite,
        "time_mapping_pass": time_mapping_ok, "physical_consistency_pass": physical_consistency,
        "physical_consistency_details": physical_checks, "telemetry_sync_prns": sorted(telemetry_union),
    }


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def prepare_freeze() -> None:
    require(git("rev-parse", "HEAD") == BASE_SHA, "prepare-freeze must start at exact base")
    require(RAW.is_file() and RAW.stat().st_size == RAW_SIZE, "C-1 raw size mismatch")
    require(md5_file(README_PDF) == README_MD5, "C-1 README MD5 mismatch")
    metadata = read_json(ZENODO_JSON)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    code_bindings = {str(path): sha256_file(REPO_ROOT / path) for path in (
        Path("scripts/run_qset_gnss_stage0a_r1.py"), Path("scripts/verify_qset_gnss_stage0a_r1.py"),
        Path("tests/test_qset_gnss_stage0a_r1.py"), CONFIG_TEMPLATE_REL,
    )}
    prereg = {
        "schema": "gnss-doppler-lab.qset-r1-preregistration.v1", "status": "FROZEN_PRE_EXECUTION",
        "objective": "C-1 clean-only format and Galileo E1 receiver feasibility; no Q-SET training, threshold, attack scoring, or detection claim",
        "base_sha": BASE_SHA, "branch": BRANCH, "allowed_inputs": [str(RAW), str(README_PDF), str(ZENODO_JSON)],
        "receiver_windows": list(RECEIVER_WINDOWS), "format_windows": list(FORMAT_WINDOWS),
        "receiver_configuration_order": ["single frozen primary configuration; no fallback or post-result tuning"],
        "success_gate": {
            "format": ">i2 and official identity PASS", "acquisition_distinct_prns_min": 4,
            "tracking_distinct_prns_ge_10s_min": 4, "all_tracking_finite": True,
            "two_window_physical_consistency": "each panel >=4, common panel >=4, median Doppler delta <=500 Hz, median C/N0 delta <=15 dB, Doppler within +/-15 kHz, C/N0 15..65 dB-Hz",
            "receiver_raw_time_mapping": "output counter maps through exact 25/2 ratio and fixed 205.25-input-sample FIR group delay",
            "availability_mask": "union of actually observed Galileo E1 PRNs; no fixed ten-slot panel",
            "telemetry_sync": "auxiliary only; not a success gate",
        },
        "verdicts": ["READY_FOR_TUNI2025_C3_CLEAN_DOWNLOAD", "BLOCKED_C1_FORMAT_UNRESOLVED", "BLOCKED_GALILEO_E1_RECEIVER_NOT_AVAILABLE", "BLOCKED_INSUFFICIENT_GALILEO_PRN_SUPPORT", "BLOCKED_RECEIVER_TIME_MAPPING", "INCONCLUSIVE_RECEIVER_FEASIBILITY"],
        "post_result_change_forbidden": True, "c3_or_attack_authorized": False,
    }
    source = {
        "schema": "gnss-doppler-lab.qset-r1-source-binding.v1", "status": "PASS_PRE_EXECUTION",
        "base_sha": BASE_SHA, "base_tree": git("show", "-s", "--format=%T", BASE_SHA),
        "main_local_sha": git("rev-parse", "main"), "main_remote_sha": git("rev-parse", "origin/main"),
        "c1": {"raw_path": str(RAW), "size_bytes": RAW_SIZE, "md5": RAW_MD5, "identity_hash_completed_pre_freeze": True,
               "readme": {**file_identity(README_PDF, "md5")}, "zenodo_record_id": metadata["id"], "concept_record_id": metadata["conceptrecid"], "doi": metadata["doi"]},
        "scientific_scope": {"constellation_signal": "Galileo E1", "scenario": "authentic static clear sky", "spoofers": "none", "multipath": "none", "nominal_sampling_rate_hz": RAW_SAMPLE_RATE},
        "code_bindings": code_bindings,
    }
    provenance = {
        "schema": "gnss-doppler-lab.qset-r1-format-discovery-provenance.v1", "status": "HISTORICAL_PRE_R1_CLEAN_ONLY_OBSERVATION",
        "not_preregistered_by_r1": True, "authorization": "user-approved clean-only inspection before R1",
        "historical_observation": {"official_readme_claim": "interleaved 32-bit float", "float_interpretations": "non-finite/extreme and nonphysical", "candidate": ">i2 interleaved I,Q", "candidate_samples": RAW_COMPLEX_SAMPLES, "candidate_duration_s": RAW_DURATION_S, "representative_std": 18, "representative_range": [-100, 87], "saturation_fraction": 0},
        "r1_action": "independently reproduce with frozen equal-size head/middle/tail windows and fail closed if >i2 is not reproduced",
    }
    bounded = {
        "schema": "gnss-doppler-lab.qset-r1-bounded-window.v1", "status": "FROZEN_PRE_EXECUTION",
        "identity_hash_exception": {"purpose": "official full-file MD5", "passes_total_planned_including_pre_freeze": 2},
        "format_windows": list(FORMAT_WINDOWS), "receiver_windows": list(RECEIVER_WINDOWS),
        "receiver_source_bytes_total": sum(int(item["duration_s"] * RAW_SAMPLE_RATE * BYTES_PER_COMPLEX) for item in RECEIVER_WINDOWS),
        "full_150s_conversion_forbidden": True, "outside_window_decode_forbidden": True,
    }
    receiver_inventory = {
        "schema": "gnss-doppler-lab.qset-r1-receiver-inventory.v1", "status": "PASS_PRE_EXECUTION",
        "selected": {**file_identity(SYSTEM_RECEIVER), "version": version_output(SYSTEM_RECEIVER)},
        "method_a_inventory_only": {**file_identity(METHOD_A_RECEIVER), "resolved_path": str(METHOD_A_RECEIVER.resolve()), "version": version_output(METHOD_A_RECEIVER)},
        "galileo_e1_components": {"acquisition": "Galileo_E1_PCPS_Ambiguous_Acquisition", "tracking": "Galileo_E1_DLL_PLL_VEML_Tracking", "telemetry": "Galileo_E1B_Telemetry_Decoder", "support_evidence": "installed Galileo E1 example configs, receiver source factory entries, and a synthetic zero-stream launch with no component-construction error before the bounded audit stop"},
    }
    config_freeze = {
        "schema": "gnss-doppler-lab.qset-r1-receiver-config-freeze.v1", "status": "FROZEN_PRE_EXECUTION",
        "template_path": str(CONFIG_TEMPLATE_REL), "template_sha256": code_bindings[str(CONFIG_TEMPLATE_REL)],
        "reference_example": {**file_identity(EXAMPLE_CONFIG)},
        "selected_path": "offline fixed anti-alias 50 MSps -> 4 MSps rational 2/25; native gr_complex",
        "resampler": {"interpolation": INTERPOLATION, "decimation": DECIMATION, "fractional_bw": RESAMPLER_FRACTIONAL_BW, "tap_count": RESAMPLER_TAP_COUNT, "tap_sha256": RESAMPLER_TAP_SHA256, "group_delay_input_samples": RESAMPLER_GROUP_DELAY_INPUT_SAMPLES, "deterministic_eof_loss_output_samples": RESAMPLER_EOF_LOSS_OUTPUT_SAMPLES},
        "channels_capacity": 12, "channels_in_acquisition": 4, "dynamic_panel_contract": "observed PRNs only; channel capacity is not a fixed panel",
        "fallback": "none; any resource or receiver failure is inconclusive/blocked without tuning",
        "large_output_root": str(OUTPUT_ROOT), "disk_output_upper_bound_bytes": 4_000_000_000, "minimum_free_bytes": MIN_FREE_BYTES,
    }
    execution = {
        "schema": "gnss-doppler-lab.qset-r1-execution-freeze.v1", "status": "FROZEN_PRE_EXECUTION",
        "code_bindings": code_bindings, "command": "/usr/bin/python3 scripts/run_qset_gnss_stage0a_r1.py execute --freeze-sha <pushed-freeze-sha>",
        "environment": {"python": platform.python_version(), "os": platform.platform(), "numpy": np.__version__},
        "worker_count": 1, "sequential_order": [item["id"] for item in RECEIVER_WINDOWS], "result_dependent_changes_forbidden": True,
    }
    placeholders = {
        "format_identification.json": {"schema": "gnss-doppler-lab.qset-r1-pending.v1", "status": "PENDING_FROZEN_EXECUTION"},
        "decoder_validation.json": {"schema": "gnss-doppler-lab.qset-r1-pending.v1", "status": "PENDING_FROZEN_EXECUTION"},
        "receiver_run_manifest.json": {"schema": "gnss-doppler-lab.qset-r1-pending.v1", "status": "PENDING_FROZEN_EXECUTION"},
        "time_mapping_validation.json": {"schema": "gnss-doppler-lab.qset-r1-pending.v1", "status": "PENDING_FROZEN_EXECUTION"},
        "support_audit.json": {"schema": "gnss-doppler-lab.qset-r1-pending.v1", "status": "PENDING_FROZEN_EXECUTION"},
        "access_audit.json": {"schema": "gnss-doppler-lab.qset-r1-pending.v1", "status": "PENDING_FROZEN_EXECUTION"},
        "deterministic_reproduction.json": {"schema": "gnss-doppler-lab.qset-r1-pending.v1", "status": "PENDING_FROZEN_EXECUTION"},
        "final_verdict.json": {"schema": "gnss-doppler-lab.qset-r1-pending.v1", "status": "PENDING_FROZEN_EXECUTION"},
    }
    write_json(ARTIFACT / "preregistration.json", prereg); write_json(ARTIFACT / "source_binding.json", source)
    write_json(ARTIFACT / "format_discovery_provenance.json", provenance); write_json(ARTIFACT / "bounded_window_contract.json", bounded)
    write_json(ARTIFACT / "receiver_binary_inventory.json", receiver_inventory); write_json(ARTIFACT / "receiver_configuration_freeze.json", config_freeze)
    write_json(ARTIFACT / "execution_freeze.json", execution)
    for name, payload in placeholders.items(): write_json(ARTIFACT / name, payload)
    (ARTIFACT / "acquisition_inventory.csv").write_text("prn,acquired,segments_observed,telemetry_sync\n", encoding="utf-8")
    (ARTIFACT / "per_prn_tracking_support.csv").write_text("prn,longest_continuous_tracking_s,tracking_ge_10s,all_finite\n", encoding="utf-8")
    (ARTIFACT / "test_output.txt").write_text("PENDING FINAL TESTS\n", encoding="utf-8")
    (ARTIFACT / "verifier_output.txt").write_text("PENDING FINAL VERIFIER\n", encoding="utf-8")
    (ARTIFACT / "README.md").write_text("# Q-SET-GNSS Stage-0A R1 Galileo C-1 receiver preflight\n\nExecution is frozen and pending. This clean-only step performs no model training, threshold calibration, attack scoring, or detection claim.\n", encoding="utf-8")


def execute(freeze_sha: str) -> None:
    require(git("rev-parse", "HEAD") == freeze_sha, "execution checkout is not freeze SHA")
    require(git("status", "--porcelain") == "", "execution checkout is not clean")
    require(git("rev-parse", f"origin/{BRANCH}") == freeze_sha, "freeze SHA not equal to remote branch")
    freeze = read_json(ARTIFACT / "execution_freeze.json")
    for relative, expected in freeze["code_bindings"].items(): require(sha256_file(REPO_ROOT / relative) == expected, f"frozen executable drift: {relative}")
    require(not OUTPUT_ROOT.exists(), f"large output root already exists: {OUTPUT_ROOT}")
    usage = shutil.disk_usage(OUTPUT_ROOT.parent)
    require(usage.free >= MIN_FREE_BYTES, "insufficient disk before execution")
    OUTPUT_ROOT.mkdir(parents=True)
    audit = {
        "schema": "gnss-doppler-lab.qset-r1-access-audit.v1", "status": "RUNNING",
        "c1": {"identity_hash_passes": 1, "identity_hash_bytes": RAW_SIZE, "format_window_bytes": 0, "receiver_decode_bytes": 0, "total_payload_bytes_read_including_pre_freeze_hash": RAW_SIZE},
        "c3": {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0, "downloads": 0},
        "attack": {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0, "downloads": 0},
        "other_tuni2025_raw": {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0, "downloads": 0},
    }
    write_json(ARTIFACT / "access_audit.json", audit)
    print("[1/5] verifying full C-1 identity MD5", flush=True)
    require(RAW.stat().st_size == RAW_SIZE, "raw size mismatch")
    require(md5_file(RAW) == RAW_MD5, "raw MD5 mismatch")
    audit["c1"]["identity_hash_passes"] = 2; audit["c1"]["identity_hash_bytes"] = RAW_SIZE * 2
    audit["c1"]["total_payload_bytes_read_including_pre_freeze_hash"] += RAW_SIZE
    print("[2/5] running bounded format identification", flush=True)
    format_result = identify_format(RAW)
    audit["c1"]["format_window_bytes"] = len(FORMAT_WINDOWS) * FORMAT_WINDOW_BYTES
    audit["c1"]["total_payload_bytes_read_including_pre_freeze_hash"] += len(FORMAT_WINDOWS) * FORMAT_WINDOW_BYTES
    write_json(ARTIFACT / "format_identification.json", format_result)
    if format_result["status"] != "PASS":
        finalize_failure(freeze_sha, audit, "BLOCKED_C1_FORMAT_UNRESOLVED", "format identification failed"); return
    sample_bytes = struct.pack(">hhhh", -123, 456, 789, -321)
    decoder_unit_ok = np.array_equal(decoder_from_bytes(sample_bytes), np.array([-123 + 456j, 789 - 321j], dtype=np.complex64)) and decoder_round_trip(sample_bytes)
    require(decoder_unit_ok, "decoder unit validation failed")
    print("[3/5] decoding/resampling and running two frozen receiver windows", flush=True)
    segments = []
    for window in RECEIVER_WINDOWS:
        segment_id = str(window["id"]); segment_dir = OUTPUT_ROOT / segment_id
        require(not segment_dir.exists(), f"segment output exists: {segment_dir}")
        segment_dir.mkdir()
        start_sample = int(float(window["start_s"]) * RAW_SAMPLE_RATE); sample_count = int(float(window["duration_s"]) * RAW_SAMPLE_RATE)
        decoder_path = segment_dir / "c1_4msps_gr_complex.bin"
        print(f"  decoding {segment_id}", flush=True)
        decoder_info = stream_decode_resample(RAW, decoder_path, start_sample, sample_count)
        write_json(segment_dir / "decoder_sidecar.json", decoder_info)
        audit["c1"]["receiver_decode_bytes"] += decoder_info["source_bytes_read"]
        audit["c1"]["total_payload_bytes_read_including_pre_freeze_hash"] += decoder_info["source_bytes_read"]
        print(f"  receiver {segment_id}", flush=True)
        receiver_info = run_receiver(segment_dir, decoder_path, decoder_info)
        segments.append({"segment_id": segment_id, "source_start_sample": start_sample, "source_sample_count": sample_count, "decoder": decoder_info, "receiver": receiver_info})
        write_json(ARTIFACT / "access_audit.json", audit)
    run_manifest = {
        "schema": "gnss-doppler-lab.qset-r1-receiver-run-manifest.v1", "status": "PASS" if all(item["receiver"]["exit_code"] == 0 for item in segments) else "FAIL",
        "freeze_sha": freeze_sha, "output_root": str(OUTPUT_ROOT), "worker_count": 1, "segments": segments,
    }
    write_json(ARTIFACT / "receiver_run_manifest.json", run_manifest)
    decoder_validation = {
        "schema": "gnss-doppler-lab.qset-r1-decoder-validation.v1", "status": "PASS",
        "input_dtype": ">i2", "interleaving": "I,Q", "byteswap": "big endian to native little endian", "normalization_scale_factor": 1.0,
        "unit_vector_pass": decoder_unit_ok, "streaming_round_trip_pass": decoder_round_trip(sample_bytes),
        "source_windows": [{"segment_id": item["segment_id"], **item["decoder"]} for item in segments],
    }
    write_json(ARTIFACT / "decoder_validation.json", decoder_validation)
    print("[4/5] analyzing native tracking dumps", flush=True)
    analysis_one = analyze_outputs(run_manifest); analysis_two = analyze_outputs(run_manifest)
    canonical_one = json.dumps(analysis_one, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    canonical_two = json.dumps(analysis_two, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    deterministic = {"schema": "gnss-doppler-lab.qset-r1-deterministic.v1", "status": "PASS" if canonical_one == canonical_two else "FAIL", "analysis_runs": 2, "byte_identical": canonical_one == canonical_two, "analysis_sha256": hashlib.sha256(canonical_one).hexdigest()}
    write_json(ARTIFACT / "deterministic_reproduction.json", deterministic)
    fields = ["prn", "acquired", "segments_observed", "run_count", "longest_continuous_tracking_s", "tracking_ge_10s", "all_finite", "telemetry_sync"]
    csv_write(ARTIFACT / "acquisition_inventory.csv", analysis_one["per_prn"], fields)
    csv_write(ARTIFACT / "per_prn_tracking_support.csv", analysis_one["per_prn"], fields)
    time_mapping = {
        "schema": "gnss-doppler-lab.qset-r1-time-mapping.v1", "status": "PASS" if analysis_one["time_mapping_pass"] else "FAIL",
        "mapping": "effective_raw_sample = window_start_raw_sample + receiver_output_sample*25/2 - 205.25",
        "raw_sample_rate_hz": RAW_SAMPLE_RATE, "receiver_sample_rate_hz": OUTPUT_SAMPLE_RATE,
        "group_delay_input_samples": RESAMPLER_GROUP_DELAY_INPUT_SAMPLES,
        "all_tracking_records_within_source_window": analysis_one["time_mapping_pass"],
    }
    write_json(ARTIFACT / "time_mapping_validation.json", time_mapping)
    write_json(ARTIFACT / "support_audit.json", {"schema": "gnss-doppler-lab.qset-r1-support-audit.v1", "status": "PASS", **analysis_one})
    audit["status"] = "PASS"; write_json(ARTIFACT / "access_audit.json", audit)
    receiver_logs = [
        (OUTPUT_ROOT / item["segment_id"] / "receiver" / "receiver.log").read_text(encoding="utf-8", errors="replace").lower()
        for item in segments
    ]
    receiver_unavailable = any(
        any(marker in text for marker in ("implementation not found", "unknown implementation", "not available in gnss-sdr", "unrecognized block"))
        for text in receiver_logs
    )
    technical = run_manifest["status"] == "PASS" and deterministic["status"] == "PASS"
    if receiver_unavailable:
        verdict, reason = "BLOCKED_GALILEO_E1_RECEIVER_NOT_AVAILABLE", "frozen Galileo E1 receiver component was unavailable"
    elif not technical:
        verdict, reason = "INCONCLUSIVE_RECEIVER_FEASIBILITY", "receiver execution or deterministic analysis incomplete"
    elif not analysis_one["time_mapping_pass"]:
        verdict, reason = "BLOCKED_RECEIVER_TIME_MAPPING", "receiver/raw sample mapping gate failed"
    elif analysis_one["acquisition_prn_count"] < 4 or analysis_one["tracking_ge_10s_prn_count"] < 4 or not analysis_one["all_tracking_finite"] or not analysis_one["physical_consistency_pass"]:
        verdict, reason = "BLOCKED_INSUFFICIENT_GALILEO_PRN_SUPPORT", "one or more frozen acquisition/tracking/finite/two-window physical support gates failed"
    else:
        verdict, reason = "READY_FOR_TUNI2025_C3_CLEAN_DOWNLOAD", "all C-1 format and Galileo E1 receiver feasibility gates passed"
    print("[5/5] writing frozen verdict", flush=True)
    final = {
        "schema": "gnss-doppler-lab.qset-r1-final-verdict.v1", "status": "PASS", "verdict": verdict, "reason": reason,
        "base_sha": BASE_SHA, "freeze_sha": freeze_sha, "acquisition_prn_count": analysis_one["acquisition_prn_count"],
        "tracking_ge_10s_prn_count": analysis_one["tracking_ge_10s_prn_count"], "telemetry_sync_prns": analysis_one["telemetry_sync_prns"],
        "actual_format": format_result["selected_format"], "dynamic_panel": analysis_one["dynamic_panel"], "common_panel": analysis_one["common_panel"],
        "gates": {"identity_format": True, "acquisition": analysis_one["acquisition_prn_count"] >= 4, "tracking": analysis_one["tracking_ge_10s_prn_count"] >= 4, "finite": analysis_one["all_tracking_finite"], "physical_consistency": analysis_one["physical_consistency_pass"], "time_mapping": analysis_one["time_mapping_pass"], "dynamic_availability_mask": bool(analysis_one["dynamic_panel"])},
        "qset_training_performed": False, "threshold_calibrated": False, "attack_scoring_performed": False,
        "detection_performance_claimed": False, "c3_clean_download_authorized": verdict == "READY_FOR_TUNI2025_C3_CLEAN_DOWNLOAD",
        "attack_download_or_access_authorized": False, "next_state": "C3_CLEAN_DOWNLOAD_ONLY" if verdict == "READY_FOR_TUNI2025_C3_CLEAN_DOWNLOAD" else "NOT_AUTHORIZED",
    }
    write_json(ARTIFACT / "final_verdict.json", final)
    write_json(ARTIFACT / "freeze_commit.json", {"schema": "gnss-doppler-lab.qset-r1-freeze-commit.v1", "status": "PASS", "commit_sha": freeze_sha, "remote_branch": BRANCH})
    readme = f"""# Q-SET-GNSS Stage-0A R1 Galileo C-1 receiver preflight

Final verdict: `{verdict}`.

This was a clean-only format and receiver feasibility preflight. It did not train a Q-SET model, calibrate a threshold, score an attack, or make a detection-performance claim. Only C-3 clean download can be authorized by a passing verdict; attack data remains unauthorized.

- Actual format: `{format_result['selected_format']}`
- Acquired Galileo E1 PRNs: {analysis_one['acquisition_prn_count']}
- PRNs continuously tracked for at least 10 s: {analysis_one['tracking_ge_10s_prn_count']}
- Dynamic observed panel: {analysis_one['dynamic_panel']}
- Telemetry-sync auxiliary PRNs: {analysis_one['telemetry_sync_prns']}
- C-1 payload bytes read (including two full identity hashes): {audit['c1']['total_payload_bytes_read_including_pre_freeze_hash']}
- C-3/attack payload bytes read: 0/0
"""
    (ARTIFACT / "README.md").write_text(readme, encoding="utf-8")


def finalize_failure(freeze_sha: str, audit: dict[str, Any], verdict: str, reason: str) -> None:
    audit["status"] = "FAIL_CLOSED"; write_json(ARTIFACT / "access_audit.json", audit)
    write_json(ARTIFACT / "final_verdict.json", {"schema": "gnss-doppler-lab.qset-r1-final-verdict.v1", "status": "PASS", "verdict": verdict, "reason": reason, "base_sha": BASE_SHA, "freeze_sha": freeze_sha, "c3_clean_download_authorized": False, "attack_download_or_access_authorized": False, "next_state": "NOT_AUTHORIZED"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-freeze")
    execute_parser = sub.add_parser("execute"); execute_parser.add_argument("--freeze-sha", required=True)
    args = parser.parse_args()
    if args.command == "prepare-freeze": prepare_freeze()
    elif args.command == "execute": execute(args.freeze_sha)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreflightError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
