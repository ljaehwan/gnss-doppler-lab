"""GNSS-SDR configuration and receiver-output normalization."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable

import h5py
import numpy as np


TRACKING_FIELDS = (
    "carrier_doppler_hz",
    "carrier_doppler_rate_hz",
    "CN0_SNV_dB_Hz",
    "Prompt_I",
    "Prompt_Q",
    "carrier_lock_test",
    "carr_error_hz",
    "code_error_chips",
)


def render_receiver_config(
    iq_path: str | Path,
    output_dir: str | Path,
    *,
    sample_rate_hz: int,
    channel_count: int = 11,
    tracking_tap_count: int = 3,
    tracking_tap_spacing_chips: float = 0.125,
) -> str:
    """Build a GPS L1 C/A file-receiver configuration for s8 interleaved IQ."""
    if sample_rate_hz < 1_000_000:
        raise ValueError("sample_rate_hz must be at least 1000000")
    if channel_count < 1:
        raise ValueError("channel_count must be positive")
    if tracking_tap_count not in {3, 5, 9}:
        raise ValueError("tracking_tap_count must be one of 3, 5, or 9")
    if tracking_tap_spacing_chips <= 0:
        raise ValueError("tracking_tap_spacing_chips must be positive")
    iq = Path(iq_path).resolve()
    output = Path(output_dir).resolve()
    tracking_prefix = output / "raw" / "epl_tracking_ch_"
    observables = output / "raw" / "observables.dat"
    return f"""[GNSS-SDR]
GNSS-SDR.internal_fs_sps={sample_rate_hz}

SignalSource.implementation=File_Signal_Source
SignalSource.filename={iq}
SignalSource.item_type=ibyte
SignalSource.sampling_frequency={sample_rate_hz}
SignalSource.samples=0
SignalSource.repeat=false
SignalSource.dump=false
SignalSource.enable_throttle_control=false

SignalConditioner.implementation=Signal_Conditioner
DataTypeAdapter.implementation=Ibyte_To_Complex
InputFilter.implementation=Pass_Through
InputFilter.input_item_type=gr_complex
InputFilter.output_item_type=gr_complex
Resampler.implementation=Pass_Through
Resampler.item_type=gr_complex

Channels_1C.count={channel_count}
Channels.in_acquisition={channel_count}
Channel.signal=1C

Acquisition_1C.implementation=GPS_L1_CA_PCPS_Acquisition
Acquisition_1C.item_type=gr_complex
Acquisition_1C.coherent_integration_time_ms=1
Acquisition_1C.threshold=2.5
Acquisition_1C.doppler_max=6000
Acquisition_1C.doppler_step=100
Acquisition_1C.dump=false

Tracking_1C.implementation=GPS_L1_CA_DLL_PLL_Tracking
Tracking_1C.item_type=gr_complex
Tracking_1C.pll_bw_hz=20.0
Tracking_1C.dll_bw_hz=1.5
Tracking_1C.order=3
Tracking_1C.dump=true
Tracking_1C.dump_filename={tracking_prefix}
Tracking_1C.tap_count={tracking_tap_count}
Tracking_1C.tap_spacing_chips={tracking_tap_spacing_chips}

TelemetryDecoder_1C.implementation=GPS_L1_CA_Telemetry_Decoder
TelemetryDecoder_1C.dump=false

Observables.implementation=Hybrid_Observables
Observables.dump=true
Observables.dump_filename={observables}

PVT.implementation=RTKLIB_PVT
PVT.positioning_mode=Single
PVT.output_rate_ms=100
PVT.display_rate_ms=500
PVT.flag_rtcm_server=false
PVT.flag_rtcm_tty_port=false
PVT.dump=false
"""


def _ordered_gps_prns_from_matches(matches: Iterable[re.Match[str]]) -> list[str]:
    prns: list[int] = []
    seen: set[int] = set()
    for match in matches:
        prn = int(match.group(1))
        if prn not in seen:
            seen.add(prn)
            prns.append(prn)
    return [f"G{prn:02d}" for prn in prns]


def parse_acquired_prns(log_text: str) -> list[str]:
    """Extract unique GNSS-SDR tracking-start GPS PRNs from console output.

    GNSS-SDR writes to stdout from multiple threads, so the satellite part of a
    tracking-start message can be interrupted by another tracking-start message,
    a receiver-time message, and/or a newline. Treat the result as an audit set,
    not the sole evidence that a PRN was validly tracked.
    """
    start = "Tracking of GPS L1 C/A signal started"
    prn_pattern = re.compile(r"\bGPS\s+PRN\s+(\d+)\b")
    lines = log_text.splitlines()
    found: list[re.Match[str]] = []

    for index, line in enumerate(lines):
        markers = [m.start() for m in re.finditer(re.escape(start), line)]
        for offset, marker in enumerate(markers):
            end = markers[offset + 1] if offset + 1 < len(markers) else len(line)
            candidates = [line[marker + len(start):end]]
            if not prn_pattern.search(candidates[0]):
                for continuation in lines[index + 1:index + 3]:
                    stripped = continuation.lstrip()
                    if prn_pattern.match(stripped):
                        candidates.append(stripped)
                        break
                    if stripped.startswith("Current receiver time:"):
                        candidates.append(stripped)
                        continue
                    break
            match = next((prn_pattern.search(candidate) for candidate in candidates
                          if prn_pattern.search(candidate)), None)
            if match:
                found.append(match)

    return _ordered_gps_prns_from_matches(found)


def parse_receiver_reported_prns(log_text: str) -> list[str]:
    """Extract PRNs with receiver evidence from tracking, bit-sync, or NAV logs."""
    evidence_markers = (
        "Tracking of GPS L1 C/A signal started",
        "GPS L1 C/A tracking bit synchronization locked",
        "New GPS NAV message received",
    )
    prn_pattern = re.compile(r"\bGPS\s+PRN\s+(\d{1,2})\b")
    lines = log_text.splitlines()
    found: list[re.Match[str]] = []
    for index, line in enumerate(lines):
        if any(marker in line for marker in evidence_markers):
            for match in prn_pattern.finditer(line):
                prn = int(match.group(1))
                if 1 <= prn <= 32:
                    found.append(match)
            if not prn_pattern.search(line) and index + 1 < len(lines):
                stripped = lines[index + 1].lstrip()
                match = prn_pattern.match(stripped)
                if match and 1 <= int(match.group(1)) <= 32:
                    found.append(match)
    return _ordered_gps_prns_from_matches(found)

def _channel_number(path: Path) -> int:
    match = re.search(r"_ch_(\d+)\.mat$", path.name)
    if not match:
        raise ValueError(f"Cannot determine channel from {path.name}")
    return int(match.group(1))


def _vector(handle: h5py.File, name: str) -> np.ndarray:
    if name not in handle:
        raise ValueError(f"Tracking MAT is missing dataset: {name}")
    return np.asarray(handle[name]).reshape(-1)


def export_tracking_csv(
    mat_paths: Iterable[str | Path],
    output_path: str | Path,
    summary_path: str | Path,
    *,
    sample_rate_hz: int,
) -> dict[str, object]:
    """Normalize GNSS-SDR tracking MAT files into long-form and summary CSVs."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    paths = sorted((Path(path) for path in mat_paths), key=_channel_number)
    for path in paths:
        channel = _channel_number(path)
        with h5py.File(path, "r") as handle:
            prn_values = _vector(handle, "PRN")
            sample_counts = _vector(handle, "PRN_start_sample_count")
            fields = {name: _vector(handle, name) for name in TRACKING_FIELDS}
        # GNSS-SDR's MAT converter represents an empty zero-byte channel dump
        # as two sentinel values [1, 0] in every dataset.  It is not a two-epoch
        # observation of PRN 1 and must not enter tracking evidence/features.
        sentinel = np.array([1, 0])
        vectors = (prn_values, sample_counts, *fields.values())
        if all(values.shape == (2,) and np.array_equal(values, sentinel) for values in vectors):
            continue
        lengths = {len(prn_values), len(sample_counts), *(len(values) for values in fields.values())}
        if len(lengths) != 1 or not prn_values.size:
            raise ValueError(f"Tracking datasets have inconsistent or empty lengths: {path}")
        prn = f"G{int(prn_values[0]):02d}"
        for index in range(len(prn_values)):
            row: dict[str, object] = {
                "time_s": float(sample_counts[index] / sample_rate_hz),
                "sample_count": int(sample_counts[index]),
                "channel": channel,
                "prn": prn,
            }
            row.update({name: float(values[index]) for name, values in fields.items()})
            rows.append(row)
        summaries.append(
            {
                "channel": channel,
                "prn": prn,
                "epoch_count": len(prn_values),
                "start_time_s": float(sample_counts[0] / sample_rate_hz),
                "end_time_s": float(sample_counts[-1] / sample_rate_hz),
                "median_doppler_hz": float(np.median(fields["carrier_doppler_hz"])),
                "doppler_min_hz": float(np.min(fields["carrier_doppler_hz"])),
                "doppler_max_hz": float(np.max(fields["carrier_doppler_hz"])),
                "median_cn0_db_hz": float(np.median(fields["CN0_SNV_dB_Hz"])),
            }
        )
    if not rows:
        raise ValueError("No tracking MAT files were provided")
    output = Path(output_path)
    summary = Path(summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    prns = sorted({str(row["prn"]) for row in summaries})
    return {"row_count": len(rows), "prns": prns, "channel_count": len(summaries)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_receiver(
    rf_manifest_path: str | Path,
    output_root: str | Path,
    *,
    executable: str | Path = "gnss-sdr",
    channel_count: int = 11,
    timeout_seconds: int = 300,
    tracking_tap_count: int = 3,
    tracking_tap_spacing_chips: float = 0.125,
) -> Path:
    """Process one RF run with GNSS-SDR and publish normalized receiver artifacts."""
    source_manifest_path = Path(rf_manifest_path).resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_run = source_manifest_path.parent
    source_run_id = str(source_manifest["run_id"])
    iq_info = source_manifest["iq"]
    iq_path = (source_run / iq_info.get("path", "gps_l1ca_s8_iq.bin")).resolve()
    if _sha256(iq_path) != iq_info["sha256"]:
        raise ValueError("IQ SHA-256 does not match the source RF manifest")
    sample_rate_hz = int(iq_info["rf_sample_rate_hz"])

    root = Path(output_root).resolve()
    run_dir = (root / source_run_id).resolve()
    if run_dir.parent != root:
        raise ValueError("receiver run directory must remain directly under output root")
    run_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = run_dir / "raw"
    raw_dir.mkdir()
    config_path = run_dir / "receiver.conf"
    log_path = run_dir / "receiver.log"

    resolved_executable = shutil.which(str(executable)) or str(Path(executable).resolve())
    config_path.write_text(
        render_receiver_config(
            iq_path,
            run_dir,
            sample_rate_hz=sample_rate_hz,
            channel_count=channel_count,
            tracking_tap_count=tracking_tap_count,
            tracking_tap_spacing_chips=tracking_tap_spacing_chips,
        ),
        encoding="utf-8",
    )
    version_result = subprocess.run(
        [resolved_executable, "--version"], capture_output=True, text=True, check=False, timeout=30
    )
    version = (version_result.stdout or version_result.stderr).strip().splitlines()[0]
    command = [resolved_executable, f"--config_file={config_path}", "--keyboard=false"]
    result = subprocess.run(
        command,
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    log_text = result.stdout + result.stderr
    log_path.write_text(log_text, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"GNSS-SDR failed with exit code {result.returncode}; see {log_path}")

    mat_paths = sorted(raw_dir.glob("epl_tracking_ch_*.mat"), key=_channel_number)
    report = export_tracking_csv(
        mat_paths,
        run_dir / "tracking.csv",
        run_dir / "tracking_summary.csv",
        sample_rate_hz=sample_rate_hz,
    )
    tracked_prns = parse_acquired_prns(log_text)
    receiver_reported_prns = parse_receiver_reported_prns(log_text)
    unreported_tracking_prns = sorted(set(report["prns"]) - set(receiver_reported_prns))
    if unreported_tracking_prns:
        raise ValueError(
            "GNSS-SDR MAT contains PRNs absent from receiver tracking/NAV evidence: "
            f"{unreported_tracking_prns}"
        )
    executable_path = Path(resolved_executable)
    manifest = {
        "schema_version": 1,
        "receiver_run_id": source_run_id,
        "source_rf_run_id": source_run_id,
        "source": {
            "rf_manifest": str(source_manifest_path),
            "rf_manifest_sha256": _sha256(source_manifest_path),
            "iq": str(iq_path),
            "iq_sha256": iq_info["sha256"],
            "sample_rate_hz": sample_rate_hz,
        },
        "receiver": {
            "name": "GNSS-SDR",
            "version": version,
            "executable": resolved_executable,
            "executable_sha256": _sha256(executable_path) if executable_path.is_file() else None,
            "config": config_path.name,
            "config_sha256": _sha256(config_path),
            "command": command,
            "exit_code": result.returncode,
        },
        "acquisition": {
            "channel_count": channel_count,
            "tracked_prns": tracked_prns,
            "tracked_prn_count": len(tracked_prns),
            "receiver_reported_prns": receiver_reported_prns,
            "receiver_reported_prn_count": len(receiver_reported_prns),
        },
        "tracking": {
            **report,
            "csv": "tracking.csv",
            "summary_csv": "tracking_summary.csv",
            "raw_directory": "raw",
            "tap_count": tracking_tap_count,
            "tap_spacing_chips": tracking_tap_spacing_chips,
        },
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
