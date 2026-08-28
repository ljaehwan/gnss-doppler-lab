#!/usr/bin/env python3
"""Clean-only TUNI Galileo receiver compatibility preflight.

This runner deliberately accepts only the C-1 and C-3 clear-sky recordings.
It must not be generalized to an attack recording before the Galileo model
and evaluation contract are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/ubuntu/unraid_hdd/tuni2025/galileo")
CLEAN_ALLOWLIST = {
    "C-1": "C-1/clearsky_signal_C-1.bin",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ishort_source_item_count(duration_s: float, complex_sample_rate_hz: int) -> int:
    if duration_s <= 0 or complex_sample_rate_hz <= 0:
        raise ValueError("duration and complex sample rate must be positive")
    # GNSS-SDR counts 16-bit source items; each complex sample contains I and Q.
    return round(duration_s * complex_sample_rate_hz * 2)



def render_config(
    *, iq_path: Path, output_dir: Path, input_samples: int, channel_count: int,
    tracking_tap_count: int = 5,
    input_rate_hz: int = 50_000_000, internal_rate_hz: int = 12_500_000,
) -> str:
    if input_samples <= 0:
        raise ValueError("input_samples must be positive")
    if channel_count <= 0:
        raise ValueError("channel_count must be positive")
    if tracking_tap_count not in {5, 9}:
        raise ValueError("tracking_tap_count must be 5 or 9")
    raw = output_dir / "raw"
    return f"""[GNSS-SDR]
GNSS-SDR.internal_fs_sps={internal_rate_hz}

SignalSource.implementation=File_Signal_Source
SignalSource.filename={iq_path}
SignalSource.item_type=ishort
SignalSource.sampling_frequency={input_rate_hz}
SignalSource.samples={input_samples}
SignalSource.repeat=false
SignalSource.dump=false
SignalSource.enable_throttle_control=false

SignalConditioner.implementation=Signal_Conditioner
DataTypeAdapter.implementation=Ishort_To_Complex
InputFilter.implementation=Pass_Through
InputFilter.input_item_type=gr_complex
InputFilter.output_item_type=gr_complex
Resampler.implementation=Direct_Resampler
Resampler.sample_freq_in={input_rate_hz}
Resampler.sample_freq_out={internal_rate_hz}
Resampler.item_type=gr_complex

Channels_1C.count=0
Channels_1B.count={channel_count}
Channels.in_acquisition={channel_count}
Channel.signal=1B

Acquisition_1B.implementation=Galileo_E1_PCPS_Ambiguous_Acquisition
Acquisition_1B.item_type=gr_complex
Acquisition_1B.coherent_integration_time_ms=4
Acquisition_1B.acquire_pilot=true
Acquisition_1B.pfa=0.000001
Acquisition_1B.doppler_max=6000
Acquisition_1B.doppler_step=125
Acquisition_1B.bit_transition_flag=true
Acquisition_1B.dump=false

Tracking_1B.implementation=Galileo_E1_DLL_PLL_VEML_Tracking
Tracking_1B.item_type=gr_complex
Tracking_1B.track_pilot=true
Tracking_1B.pll_bw_hz=15.0
Tracking_1B.dll_bw_hz=1.0
Tracking_1B.order=3
Tracking_1B.early_late_space_chips=0.125
Tracking_1B.very_early_late_space_chips=0.25
Tracking_1B.early_late_space_narrow_chips=0.125
Tracking_1B.very_early_late_space_narrow_chips=0.25
Tracking_1B.tap_count={tracking_tap_count}
Tracking_1B.tap_spacing_chips=0.125
Tracking_1B.dump=true
Tracking_1B.dump_filename={raw / 'epl_tracking_ch_'}

TelemetryDecoder_1B.implementation=Galileo_E1B_Telemetry_Decoder
TelemetryDecoder_1B.dump=false

Observables.implementation=Hybrid_Observables
Observables.dump=false

PVT.implementation=RTKLIB_PVT
PVT.positioning_mode=Single
PVT.output_rate_ms=100
PVT.display_rate_ms=1000
PVT.flag_rtcm_server=false
PVT.flag_rtcm_tty_port=false
PVT.dump=false
"""


def valid_prns(mat_paths: list[Path]) -> tuple[list[int], int]:
    seen: set[int] = set()
    epochs = 0
    for path in mat_paths:
        with h5py.File(path, "r") as handle:
            if "PRN" not in handle:
                continue
            values = np.asarray(handle["PRN"]).reshape(-1)
            # Empty GNSS-SDR dumps can contain the [1, 0] sentinel.
            if values.shape == (2,) and np.array_equal(values, np.array([1, 0])):
                continue
            current = [int(value) for value in values if 1 <= int(value) <= 36]
            seen.update(current)
            epochs += len(current)
    return sorted(seen), epochs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(CLEAN_ALLOWLIST), default="C-1")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "artifacts" / "tuni_galileo_clean_preflight_v1",
    )
    parser.add_argument(
        "--executable", type=Path,
        default=ROOT / ".tools" / "gnss-sdr-src" / "build" / "src" / "main" / "gnss-sdr",
    )
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--tap-count", type=int, choices=[5, 9], default=5)
    parser.add_argument("--channel-count", type=int, default=12)
    parser.add_argument("--timeout-s", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (0.5 <= args.duration_s <= 10.0):
        raise ValueError("clean preflight duration must be between 0.5 and 10 seconds")
    data_root = args.data_root.resolve()
    iq_path = (data_root / CLEAN_ALLOWLIST[args.scenario]).resolve()
    if iq_path.parent.parent.parent != data_root.parent:
        raise ValueError("clean IQ path escaped the TUNI Galileo data root")
    if not iq_path.is_file():
        raise FileNotFoundError(iq_path)
    executable_match = shutil.which(str(args.executable))
    executable = Path(executable_match).resolve() if executable_match else args.executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)

    output_dir = (args.output_root.resolve() / args.scenario.lower()).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    (output_dir / "raw").mkdir(parents=True)
    input_source_items = ishort_source_item_count(args.duration_s, 50_000_000)
    config_path = output_dir / "receiver.conf"
    config_path.write_text(
        render_config(
            iq_path=iq_path,
            output_dir=output_dir,
            input_samples=input_source_items,
            channel_count=args.channel_count,
            tracking_tap_count=args.tap_count,
        ),
        encoding="utf-8",
    )
    command = [str(executable), f"--config_file={config_path}", "--keyboard=false"]
    completed = subprocess.run(
        command,
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=args.timeout_s,
        check=False,
    )
    log_path = output_dir / "receiver.log"
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    mat_paths = sorted((output_dir / "raw").glob("epl_tracking_ch_*.mat"))
    prns, epoch_count = valid_prns(mat_paths)
    summary: dict[str, Any] = {
        "schema": "gnss-doppler-lab.tuni-galileo-clean-preflight.v1",
        "scope": "clean-only receiver compatibility; no attack payload accessed",
        "scenario": args.scenario,
        "input": {
            "path": str(iq_path),
            "bytes": iq_path.stat().st_size,
            "sample_format": "interleaved int16 I/Q (GNU Radio ishort; 32 bits per complex sample)",
            "input_rate_hz": 50_000_000,
            "source_int16_items_processed": input_source_items,
            "complex_samples_processed": input_source_items // 2,
            "duration_s": args.duration_s,
        },
        "receiver": {
            "executable": str(executable),
            "executable_sha256": sha256(executable),
            "config_sha256": sha256(config_path),
            "return_code": completed.returncode,
        },
        "tracking": {
            "tap_count": args.tap_count,
            "tap_spacing_chips": 0.125,
            "mat_file_count": len(mat_paths),
            "valid_prns": prns,
            "valid_prn_count": len(prns),
            "valid_epoch_count": epoch_count,
        },
        "compatible": completed.returncode == 0 and bool(prns) and epoch_count > 0,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if completed.returncode != 0:
        print(f"GNSS-SDR failed; see {log_path}", file=sys.stderr)
        return completed.returncode or 1
    return 0 if summary["compatible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

