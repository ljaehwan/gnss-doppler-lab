#!/usr/bin/env python3
"""Run the frozen complex-nine-tap receiver on GNSS-OpenIF Scenario 1."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IQ = Path(
    "/home/ubuntu/ssd_data/gnss-datasets/gnss-openif/raw/S1_suburban_HK.bin"
)
DEFAULT_OUTPUT = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "gnss-openif-s1-real-multipath-v1/receiver"
)
DEFAULT_EXECUTABLE = ROOT / ".tools" / "gnss-sdr-method-a-9tap"

INPUT_RATE_HZ = 58_000_000
# GNSS-OpenIF documents a 4.58 MHz IF magnitude.  The signed complex samples
# place the GPS L1 band at -4.58 MHz under GNSS-SDR's I+jQ convention, so the
# translating filter must use the observed negative center frequency.
IF_HZ = -4_580_000
DECIMATION = 10
INTERNAL_RATE_HZ = INPUT_RATE_HZ // DECIMATION
EXPECTED_BYTES = 10_200_547_328


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ibyte_source_item_count(duration_s: float, input_rate_hz: int = INPUT_RATE_HZ) -> int:
    """Return scalar int8 source items for interleaved complex I/Q."""
    if duration_s <= 0 or input_rate_hz <= 0:
        raise ValueError("duration and input rate must be positive")
    return round(duration_s * input_rate_hz * 2)


def render_config(
    *, iq_path: Path, output_dir: Path, duration_s: float, start_offset_s: float = 0.0,
    channel_count: int = 31, tracking_tap_count: int = 9,
) -> str:
    """Render the fixed IF-translation and 10:1 decimation receiver config."""
    if channel_count < 8 or channel_count > 31:
        raise ValueError("channel_count must be in [8, 31]")
    if start_offset_s < 0:
        raise ValueError("start_offset_s must be nonnegative")
    if tracking_tap_count != 9:
        raise ValueError("GNSS-OpenIF CGC validation requires exactly nine taps")
    samples = 0 if duration_s == 0 else ibyte_source_item_count(duration_s)
    raw = output_dir / "raw"
    # Remez bands are normalized to the 29 MHz input Nyquist frequency.
    # The 2.20 MHz passband contains the GPS L1 C/A main lobe; the stopband
    # begins at the 2.90 MHz output Nyquist limit before 10:1 decimation.
    return f"""[GNSS-SDR]
GNSS-SDR.internal_fs_sps={INTERNAL_RATE_HZ}

SignalSource.implementation=File_Signal_Source
SignalSource.filename={iq_path.resolve()}
SignalSource.item_type=ibyte
SignalSource.sampling_frequency={INPUT_RATE_HZ}
SignalSource.seconds_to_skip={start_offset_s}
SignalSource.samples={samples}
SignalSource.repeat=false
SignalSource.dump=false
SignalSource.enable_throttle_control=false

SignalConditioner.implementation=Signal_Conditioner
DataTypeAdapter.implementation=Ibyte_To_Complex
InputFilter.implementation=Freq_Xlating_Fir_Filter
InputFilter.item_type=gr_complex
InputFilter.input_item_type=gr_complex
InputFilter.output_item_type=gr_complex
InputFilter.taps_item_type=float
InputFilter.number_of_taps=33
InputFilter.number_of_bands=2
InputFilter.band1_begin=0.0
InputFilter.band1_end=0.07586206896551724
InputFilter.band2_begin=0.10
InputFilter.band2_end=1.0
InputFilter.ampl1_begin=1.0
InputFilter.ampl1_end=1.0
InputFilter.ampl2_begin=0.0
InputFilter.ampl2_end=0.0
InputFilter.band1_error=1.0
InputFilter.band2_error=1.0
InputFilter.filter_type=bandpass
InputFilter.grid_density=32
InputFilter.sampling_frequency={INPUT_RATE_HZ}
InputFilter.IF={IF_HZ}
InputFilter.decimation_factor={DECIMATION}
InputFilter.dump=false

Resampler.implementation=Pass_Through
Resampler.item_type=gr_complex

Channels_1C.count={channel_count}
Channels.in_acquisition={channel_count}
Channel.signal=1C
Channel0.signal=1C
Channel0.satellite=22
Channel1.signal=1C
Channel1.satellite=5

Acquisition_1C.implementation=GPS_L1_CA_PCPS_Acquisition
Acquisition_1C.item_type=gr_complex
Acquisition_1C.coherent_integration_time_ms=1
Acquisition_1C.pfa=0.01
Acquisition_1C.max_dwells=5
Acquisition_1C.doppler_max=10000
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
PVT.rinex_version=3
"""


def _channel_number(path: Path) -> int:
    match = re.search(r"_ch_(\d+)\.mat$", path.name)
    return int(match.group(1)) if match else 10**9


def tracking_support(mat_paths: list[Path]) -> tuple[list[int], int, dict[int, int]]:
    seen: set[int] = set()
    epochs = 0
    by_prn: dict[int, int] = {}
    sentinel = np.asarray([1, 0])
    for path in mat_paths:
        with h5py.File(path, "r") as handle:
            if "PRN" not in handle:
                continue
            values = np.asarray(handle["PRN"]).reshape(-1)
        if values.shape == (2,) and np.array_equal(values, sentinel):
            continue
        for raw in values:
            prn = int(raw)
            if 1 <= prn <= 32:
                seen.add(prn)
                by_prn[prn] = by_prn.get(prn, 0) + 1
                epochs += 1
    return sorted(seen), epochs, dict(sorted(by_prn.items()))


def run(args: argparse.Namespace) -> dict[str, Any]:
    iq_path = args.iq.resolve()
    if not iq_path.is_file():
        raise FileNotFoundError(iq_path)
    if args.duration_s == 0 and iq_path.stat().st_size != EXPECTED_BYTES:
        raise ValueError(
            f"full S1 byte count mismatch: {iq_path.stat().st_size} != {EXPECTED_BYTES}"
        )
    available_duration_s = iq_path.stat().st_size / (2.0 * INPUT_RATE_HZ)
    if args.start_offset_s >= available_duration_s:
        raise ValueError("start offset is outside the S1 recording")
    if (args.duration_s > 0 and
            args.start_offset_s + args.duration_s > available_duration_s + 1e-9):
        raise ValueError("requested window extends beyond the S1 recording")

    executable_match = shutil.which(str(args.executable))
    executable = Path(executable_match).resolve() if executable_match else args.executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    suffix = "full" if args.duration_s == 0 else f"preflight-{args.duration_s:g}s"
    run_id = args.run_id or f"s1-complex9-{suffix}"
    output_dir = (args.output_root.resolve() / run_id).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    (output_dir / "raw").mkdir(parents=True)
    config_path = output_dir / "receiver.conf"
    config_path.write_text(
        render_config(
            iq_path=iq_path,
            output_dir=output_dir,
            duration_s=args.duration_s,
            start_offset_s=args.start_offset_s,
            channel_count=args.channel_count,
        ),
        encoding="utf-8",
    )
    version_result = subprocess.run(
        [str(executable), "--version"], capture_output=True, text=True,
        check=False, timeout=30,
    )
    version_text = version_result.stdout or version_result.stderr
    version = version_text.strip().splitlines()[0] if version_text.strip() else "unknown"
    command = [str(executable), f"--config_file={config_path}", "--keyboard=false"]
    log_path = output_dir / "receiver.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=output_dir, stdout=log, stderr=subprocess.STDOUT,
            timeout=args.timeout_s, check=False, text=True,
        )
    mats = sorted((output_dir / "raw").glob("epl_tracking_ch_*.mat"), key=_channel_number)
    prns, epochs, by_prn = tracking_support(mats)
    manifest: dict[str, Any] = {
        "schema": "gnss-doppler-lab.gnss-openif-s1-receiver.v1",
        "schema_version": 1,
        "receiver_run_id": run_id,
        "source_rf_run_id": "GNSS-OpenIF-S1",
        "source": {
            "dataset": "GNSS-OpenIF",
            "scenario_id": "S1",
            "iq": str(iq_path),
            "iq_bytes": iq_path.stat().st_size,
            "iq_sha256": sha256(iq_path) if args.hash_source else None,
            "sample_format": "signed int8 interleaved complex I/Q",
            "input_sample_rate_hz": INPUT_RATE_HZ,
            "official_intermediate_frequency_magnitude_hz": abs(IF_HZ),
            "intermediate_frequency_hz": IF_HZ,
            "decimation_factor": DECIMATION,
            # Tracking sample counters are on the post-filter stream.
            "sample_rate_hz": INTERNAL_RATE_HZ,
            "requested_duration_s": args.duration_s,
            "start_offset_s": args.start_offset_s,
            "source_i8_items": (
                0 if args.duration_s == 0 else ibyte_source_item_count(args.duration_s)
            ),
        },
        "receiver": {
            "name": "GNSS-SDR Method-A complex-nine-tap",
            "version": version,
            "executable": str(executable),
            "executable_sha256": sha256(executable),
            "config": config_path.name,
            "config_sha256": sha256(config_path),
            "command": command,
            "exit_code": completed.returncode,
        },
        "acquisition": {
            "channel_count": args.channel_count,
            "fixed_priority_prns": [22, 5],
            "tracked_prns": [f"G{prn:02d}" for prn in prns],
            "prn_epoch_counts": {f"G{prn:02d}": count for prn, count in by_prn.items()},
        },
        "tracking": {
            "raw_directory": "raw",
            "mat_file_count": len(mats),
            "valid_epoch_count": epochs,
            "tap_count": 9,
            "tap_spacing_chips": 0.125,
            "complex_taps_required": True,
        },
        "navigation": {
            "gps_ephemeris_xml_present": (output_dir / "gps_ephemeris.xml").is_file(),
            "nmea_present": (output_dir / "nmea_pvt.nmea").is_file(),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"GNSS-SDR exited {completed.returncode}; see {log_path}")
    if not prns:
        raise RuntimeError(f"GNSS-SDR produced no valid tracking epochs; see {log_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iq", type=Path, default=DEFAULT_IQ)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 processes the full file")
    parser.add_argument("--start-offset-s", type=float, default=0.0)
    parser.add_argument("--channel-count", type=int, default=31)
    parser.add_argument("--run-id")
    parser.add_argument("--timeout-s", type=int, default=7200)
    parser.add_argument("--hash-source", action="store_true")
    args = parser.parse_args()
    if args.duration_s < 0:
        parser.error("--duration-s must be nonnegative")
    if args.start_offset_s < 0:
        parser.error("--start-offset-s must be nonnegative")
    return args


if __name__ == "__main__":
    run(parse_args())
