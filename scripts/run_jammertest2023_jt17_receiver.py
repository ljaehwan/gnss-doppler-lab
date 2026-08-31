#!/usr/bin/env python3
"""Run the frozen complex-nine-tap receiver on JammerTest JT23-17.1.6."""
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
DEFAULT_CONFIG = ROOT / "configs/experiments/jammertest2023_jt17_cgc_v1.json"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "gnss-doppler-lab.jammertest2023-jt17-cgc-config":
        raise ValueError("unsupported JammerTest config schema")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported JammerTest config version")
    dataset, receiver = config["dataset"], config["receiver"]
    if int(dataset["input_sample_rate_hz"]) != 30_690_000:
        raise ValueError("JammerTest input sample rate drifted")
    if int(dataset["iq_bytes"]) != 72_110_000_000:
        raise ValueError("JammerTest full-file byte contract drifted")
    if int(receiver["decimation_factor"]) != 5 or int(receiver["internal_sample_rate_hz"]) != 6_138_000:
        raise ValueError("JammerTest decimation contract drifted")
    if int(receiver["tap_count"]) != 9 or float(receiver["tap_spacing_chips"]) != 0.125:
        raise ValueError("complex-nine-tap contract drifted")
    if float(receiver["duration_seconds"]) != 560.0 or int(receiver["channel_count"]) != 31:
        raise ValueError("receiver duration/channel contract drifted")


def ibyte_source_item_count(duration_s: float, sample_rate_hz: int) -> int:
    """Return scalar int8 items for interleaved complex I/Q."""
    if duration_s <= 0.0 or sample_rate_hz <= 0:
        raise ValueError("duration and sample rate must be positive")
    return round(duration_s * sample_rate_hz * 2)


def render_config(iq_path: Path, output_dir: Path, config: dict[str, Any]) -> str:
    dataset, receiver = config["dataset"], config["receiver"]
    input_rate = int(dataset["input_sample_rate_hz"])
    internal_rate = int(receiver["internal_sample_rate_hz"])
    duration = float(receiver["duration_seconds"])
    raw = output_dir / "raw"
    return f"""[GNSS-SDR]
GNSS-SDR.internal_fs_sps={internal_rate}

SignalSource.implementation=File_Signal_Source
SignalSource.filename={iq_path.resolve()}
SignalSource.item_type=ibyte
SignalSource.sampling_frequency={input_rate}
SignalSource.samples={ibyte_source_item_count(duration, input_rate)}
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
InputFilter.band1_end=0.136852394917
InputFilter.band2_begin=0.19550342131
InputFilter.band2_end=1.0
InputFilter.ampl1_begin=1.0
InputFilter.ampl1_end=1.0
InputFilter.ampl2_begin=0.0
InputFilter.ampl2_end=0.0
InputFilter.band1_error=1.0
InputFilter.band2_error=1.0
InputFilter.filter_type=bandpass
InputFilter.grid_density=32
InputFilter.sampling_frequency={input_rate}
InputFilter.IF=0
InputFilter.decimation_factor={int(receiver['decimation_factor'])}
InputFilter.dump=false

Resampler.implementation=Pass_Through
Resampler.item_type=gr_complex

Channels_1C.count={int(receiver['channel_count'])}
Channels.in_acquisition={int(receiver['channel_count'])}
Channel.signal=1C

Acquisition_1C.implementation=GPS_L1_CA_PCPS_Acquisition
Acquisition_1C.item_type=gr_complex
Acquisition_1C.coherent_integration_time_ms=1
Acquisition_1C.pfa={float(receiver['acquisition_pfa'])}
Acquisition_1C.max_dwells={int(receiver['acquisition_max_dwells'])}
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


def tracking_support(paths: list[Path]) -> tuple[list[int], int, dict[int, int]]:
    seen: set[int] = set()
    epochs = 0
    by_prn: dict[int, int] = {}
    sentinel = np.asarray([1, 0])
    for path in paths:
        with h5py.File(path, "r") as handle:
            values = np.asarray(handle["PRN"]).reshape(-1) if "PRN" in handle else np.asarray([])
        if values.shape == (2,) and np.array_equal(values, sentinel):
            continue
        for raw in values:
            prn = int(raw)
            if 1 <= prn <= 32:
                seen.add(prn)
                by_prn[prn] = by_prn.get(prn, 0) + 1
                epochs += 1
    return sorted(seen), epochs, dict(sorted(by_prn.items()))


def run(config_path: Path, *, hash_source: bool) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    dataset, receiver = config["dataset"], config["receiver"]
    iq_path = resolve(dataset["iq_path"])
    if not iq_path.is_file() or iq_path.stat().st_size != int(dataset["iq_bytes"]):
        raise ValueError("JammerTest full IQ file is missing or has the wrong byte count")
    observed_source_hash = sha256(iq_path) if hash_source else None
    if observed_source_hash is not None and observed_source_hash != dataset["iq_sha256"]:
        raise ValueError("JammerTest IQ SHA-256 mismatch")
    executable = resolve(receiver["executable"])
    executable_match = shutil.which(str(executable))
    executable = Path(executable_match).resolve() if executable_match else executable
    if not executable.is_file() or sha256(executable) != receiver["executable_sha256"]:
        raise ValueError("frozen GNSS-SDR executable identity mismatch")
    output_dir = resolve(receiver["output_root"]) / receiver["run_id"]
    if output_dir.exists():
        raise FileExistsError(output_dir)
    (output_dir / "raw").mkdir(parents=True)
    receiver_config = output_dir / "receiver.conf"
    receiver_config.write_text(render_config(iq_path, output_dir, config), encoding="utf-8")
    version_result = subprocess.run(
        [str(executable), "--version"], capture_output=True, text=True, check=False, timeout=30
    )
    version_text = version_result.stdout or version_result.stderr
    command = [str(executable), f"--config_file={receiver_config}", "--keyboard=false"]
    log_path = output_dir / "receiver.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=output_dir, stdout=log, stderr=subprocess.STDOUT,
            timeout=int(receiver["timeout_seconds"]), check=False, text=True,
        )
    mats = sorted((output_dir / "raw").glob("epl_tracking_ch_*.mat"), key=_channel_number)
    prns, epochs, by_prn = tracking_support(mats)
    manifest = {
        "schema": "gnss-doppler-lab.jammertest2023-jt17-receiver.v1",
        "schema_version": 1,
        "receiver_run_id": receiver["run_id"],
        "source": {
            "dataset": dataset["name"], "scenario_id": dataset["scenario"],
            "iq": str(iq_path), "iq_bytes": iq_path.stat().st_size,
            "iq_sha256": observed_source_hash,
            "sample_format": dataset["sample_format"],
            "input_sample_rate_hz": int(dataset["input_sample_rate_hz"]),
            "sample_rate_hz": int(receiver["internal_sample_rate_hz"]),
            "requested_duration_s": float(receiver["duration_seconds"]),
            "start_offset_s": 0.0,
            "source_i8_items": ibyte_source_item_count(
                float(receiver["duration_seconds"]), int(dataset["input_sample_rate_hz"])
            ),
        },
        "receiver": {
            "name": "GNSS-SDR Method-A complex-nine-tap",
            "version": version_text.strip().splitlines()[0] if version_text.strip() else "unknown",
            "executable": str(executable), "executable_sha256": sha256(executable),
            "config": receiver_config.name, "config_sha256": sha256(receiver_config),
            "command": command, "exit_code": completed.returncode,
        },
        "acquisition": {
            "channel_count": int(receiver["channel_count"]),
            "tracked_prns": [f"G{prn:02d}" for prn in prns],
            "prn_epoch_counts": {f"G{prn:02d}": count for prn, count in by_prn.items()},
        },
        "tracking": {
            "raw_directory": "raw", "mat_file_count": len(mats),
            "valid_epoch_count": epochs, "tap_count": 9,
            "tap_spacing_chips": 0.125, "complex_taps_required": True,
        },
        "navigation": {
            "gps_ephemeris_xml_present": (output_dir / "gps_ephemeris.xml").is_file(),
            "nmea_present": (output_dir / "nmea_pvt.nmea").is_file(),
            "observables_present": (output_dir / "raw" / "observables.mat").is_file(),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"GNSS-SDR exited {completed.returncode}; see {log_path}")
    if not prns:
        raise RuntimeError(f"GNSS-SDR produced no valid tracking epochs; see {log_path}")
    print(json.dumps({
        "manifest": str(output_dir / "manifest.json"), "tracked_prns": manifest["acquisition"]["tracked_prns"],
        "valid_epoch_count": epochs, "exit_code": completed.returncode,
    }, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hash-source", action="store_true")
    args = parser.parse_args()
    run(args.config, hash_source=args.hash_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
