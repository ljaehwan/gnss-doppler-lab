#!/usr/bin/env python3
"""Clean-only TUNI GPS C-5 nine-tap receiver compatibility preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IQ = Path(
    "/home/ubuntu/ssd_data/gnss-datasets/tuni2025/gps/C-5/C-5.bin"
)
DEFAULT_OUTPUT = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "tuni-gps-clean-complex9-preflight-be-50msps-v1"
)
DEFAULT_EXECUTABLE = ROOT / ".tools" / "gnss-sdr-gps-be-complex9-v1"
EXPECTED_BYTES = 29_999_832_000
EXPECTED_MD5 = "a03dedd79ac4208f6d60b4c916484dba"
INPUT_RATE_HZ = 50_000_000
INTERNAL_RATE_HZ = 5_000_000


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ishort_source_item_count(duration_s: float) -> int:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    return round(duration_s * INPUT_RATE_HZ * 2)


def render_config(
    *, iq_path: Path, output_dir: Path, duration_s: float,
    channel_count: int = 31,
) -> str:
    if not 0.5 <= duration_s <= 15.0:
        raise ValueError("clean preflight duration must be in [0.5, 15] seconds")
    if not 8 <= channel_count <= 31:
        raise ValueError("channel_count must be in [8, 31]")
    raw = output_dir / "raw"
    return f"""[GNSS-SDR]
GNSS-SDR.internal_fs_sps={INTERNAL_RATE_HZ}

SignalSource.implementation=File_Signal_Source
SignalSource.filename={iq_path.resolve()}
SignalSource.item_type=ishort
SignalSource.sampling_frequency={INPUT_RATE_HZ}
SignalSource.samples={ishort_source_item_count(duration_s)}
SignalSource.repeat=false
SignalSource.dump=false
SignalSource.enable_throttle_control=false

SignalConditioner.implementation=Signal_Conditioner
DataTypeAdapter.implementation=Ishort_To_Complex
DataTypeAdapter.swap_endian=true
InputFilter.implementation=Pass_Through
InputFilter.input_item_type=gr_complex
InputFilter.output_item_type=gr_complex
Resampler.implementation=Direct_Resampler
Resampler.sample_freq_in={INPUT_RATE_HZ}
Resampler.sample_freq_out={INTERNAL_RATE_HZ}
Resampler.item_type=gr_complex

Channels_1C.count={channel_count}
Channels.in_acquisition={channel_count}
Channel.signal=1C

Acquisition_1C.implementation=GPS_L1_CA_PCPS_Acquisition
Acquisition_1C.item_type=gr_complex
Acquisition_1C.coherent_integration_time_ms=1
Acquisition_1C.pfa=0.01
Acquisition_1C.max_dwells=5
Acquisition_1C.doppler_max=6000
Acquisition_1C.doppler_step=125
Acquisition_1C.blocking=true
Acquisition_1C.dump=false

Tracking_1C.implementation=GPS_L1_CA_DLL_PLL_Tracking
Tracking_1C.item_type=gr_complex
Tracking_1C.pll_bw_hz=20.0
Tracking_1C.dll_bw_hz=1.5
Tracking_1C.order=3
Tracking_1C.early_late_space_chips=0.125
Tracking_1C.early_late_space_narrow_chips=0.125
Tracking_1C.tap_count=9
Tracking_1C.tap_spacing_chips=0.125
Tracking_1C.dump=true
Tracking_1C.dump_filename={raw / 'epl_tracking_ch_'}

TelemetryDecoder_1C.implementation=GPS_L1_CA_Telemetry_Decoder
TelemetryDecoder_1C.dump=false

Observables.implementation=Hybrid_Observables
Observables.dump=true
Observables.dump_filename={raw / 'observables.dat'}

PVT.implementation=RTKLIB_PVT
PVT.positioning_mode=Single
PVT.output_rate_ms=100
PVT.display_rate_ms=1000
PVT.flag_rtcm_server=false
PVT.flag_rtcm_tty_port=false
PVT.dump=false
"""


def tracking_support(paths: list[Path]) -> tuple[list[int], int]:
    prns: set[int] = set()
    epochs = 0
    for path in paths:
        with h5py.File(path, "r") as handle:
            if "PRN" not in handle:
                continue
            values = np.asarray(handle["PRN"]).reshape(-1)
        if values.shape == (2,) and np.array_equal(values, np.asarray([1, 0])):
            continue
        valid = [int(value) for value in values if 1 <= int(value) <= 32]
        prns.update(valid)
        epochs += len(valid)
    return sorted(prns), epochs


def run(args: argparse.Namespace) -> dict[str, Any]:
    iq = args.iq.resolve()
    if iq != DEFAULT_IQ.resolve():
        raise ValueError("clean preflight is restricted to TUNI GPS C-5")
    if not iq.is_file() or iq.stat().st_size != EXPECTED_BYTES:
        raise ValueError("TUNI GPS C-5 byte count mismatch")
    observed_md5 = file_hash(iq, "md5")
    if observed_md5 != EXPECTED_MD5:
        raise ValueError("TUNI GPS C-5 MD5 mismatch")
    executable_match = shutil.which(str(args.executable))
    executable = (
        Path(executable_match).resolve()
        if executable_match else args.executable.resolve()
    )
    if not executable.is_file():
        raise FileNotFoundError(executable)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    (output / "raw").mkdir(parents=True)
    config_path = output / "receiver.conf"
    config_path.write_text(
        render_config(
            iq_path=iq, output_dir=output, duration_s=args.duration_s,
            channel_count=args.channel_count,
        ),
        encoding="utf-8",
    )
    command = [
        str(executable), f"--config_file={config_path}", "--keyboard=false"
    ]
    completed = subprocess.run(
        command, cwd=output, capture_output=True, text=True,
        timeout=args.timeout_s, check=False,
    )
    (output / "receiver.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    mats = sorted((output / "raw").glob("epl_tracking_ch_*.mat"))
    prns, epochs = tracking_support(mats)
    manifest: dict[str, Any] = {
        "schema": "gnss-doppler-lab.tuni-gps-clean-preflight.v1",
        "scope": "C-5 clean-only compatibility; no attack payload accessed",
        "source": {
            "scenario": "C-5",
            "path": str(iq),
            "bytes": iq.stat().st_size,
            "md5": observed_md5,
            "sample_format": "big-endian interleaved int16 I/Q (ishort)",
            "byte_order": "big-endian",
            "sample_rate_basis": "Zenodo C-5 record metadata",
            "input_sample_rate_hz": INPUT_RATE_HZ,
            "duration_s": args.duration_s,
            "source_int16_items": ishort_source_item_count(args.duration_s),
        },
        "receiver": {
            "executable": str(executable),
            "executable_sha256": file_hash(executable, "sha256"),
            "config_sha256": file_hash(config_path, "sha256"),
            "return_code": completed.returncode,
        },
        "tracking": {
            "tap_count": 9,
            "tap_spacing_chips": 0.125,
            "mat_file_count": len(mats),
            "valid_prns": prns,
            "valid_prn_count": len(prns),
            "valid_epoch_count": epochs,
        },
        "compatible": completed.returncode == 0 and bool(prns) and epochs > 0,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError(f"GNSS-SDR exited {completed.returncode}")
    if not manifest["compatible"]:
        raise RuntimeError("TUNI GPS C-5 produced no valid tracking support")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iq", type=Path, default=DEFAULT_IQ)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--channel-count", type=int, default=31)
    parser.add_argument("--timeout-s", type=int, default=1800)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
