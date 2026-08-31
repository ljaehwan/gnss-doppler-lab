#!/usr/bin/env python3
"""Run the frozen score-free support gate on FGI-SpoofRepo targeted DFMC."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for search_root in (ROOT / "scripts", ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

from gnss_doppler_lab.clean_geometry_support import audit_clean_geometry_support  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/experiments/fgi_spoofrepo_tgd_support_preflight_v1.json"
PROTOCOL = ROOT / "docs/results/fgi_spoofrepo_tgd_support_preflight_protocol_v1.md"
RELEASE_TOKEN = "RELEASE-FGI-SPOOFREPO-TGD-SUPPORT-PREFLIGHT-V1"
RELEASE_INPUTS = (
    "configs/experiments/fgi_spoofrepo_tgd_support_preflight_v1.json",
    "docs/results/fgi_spoofrepo_tgd_support_preflight_protocol_v1.md",
    "scripts/preflight_fgi_spoofrepo_tgd_support.py",
    "src/gnss_doppler_lab/clean_geometry_support.py",
)


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def byte_source_item_count(duration_s: float, input_rate_hz: int = 26_000_000) -> int:
    if duration_s <= 0.0 or input_rate_hz <= 0:
        raise ValueError("duration and input sample rate must be positive")
    return round(duration_s * input_rate_hz)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "gnss-doppler-lab.fgi-spoofrepo-tgd-support-preflight-config":
        raise ValueError("unsupported FGI support-preflight schema")
    if config.get("schema_version") != 1:
        raise ValueError("unsupported FGI support-preflight version")
    experiment = config["experiment"]
    forbidden = (
        "detector_score_access", "delay_template_access", "threshold_refitting",
        "post_preflight_rule_change",
    )
    if any(experiment.get(key) is not False for key in forbidden):
        raise ValueError("support preflight must forbid detector access and rule changes")
    dataset = config["dataset"]
    expected_dataset = {
        "scenario": "TG_DFMC",
        "expected_bytes": 9_961_930_752,
        "sha256": "10aad73665db7c5e530d9ef1d3b2fdb57bab0a7b9b19177a0128867fbad2606b",
        "sample_format": "signed int8 real IF",
        "input_sample_rate_hz": 26_000_000,
        "intermediate_frequency_hz": 6_390_000,
        "documented_attack_free_until_s": 130.0,
        "documented_attack_onset_s": 138.0,
    }
    for key, value in expected_dataset.items():
        if dataset.get(key) != value:
            raise ValueError(f"FGI dataset contract drifted: {key}")
    receiver = config["receiver"]
    expected_receiver = {
        "duration_seconds": 240.0,
        "channel_count": 31,
        "decimation_factor": 4,
        "internal_sample_rate_hz": 6_500_000,
        "tap_count": 9,
        "tap_spacing_chips": 0.125,
    }
    for key, value in expected_receiver.items():
        if receiver.get(key) != value:
            raise ValueError(f"FGI receiver contract drifted: {key}")
    if config["analysis_intervals_seconds"] != {
        "clean": [40.0, 120.0],
        "post_onset_support": [160.0, 230.0],
    }:
        raise ValueError("FGI analysis intervals drifted")
    expected_gate = {
        "bin_seconds": 1.0,
        "minimum_epochs_per_prn_bin": 200,
        "minimum_primary_prns": 8,
        "secondary_boundary_prns": 7,
        "minimum_primary_bins_each_interval": 60,
        "require_complex_nine_tap": True,
    }
    if config["support_gate"] != expected_gate:
        raise ValueError("FGI support gate drifted")


def render_config(*, iq_path: Path, output_dir: Path, config: dict[str, Any]) -> str:
    dataset, receiver = config["dataset"], config["receiver"]
    input_rate = int(dataset["input_sample_rate_hz"])
    internal_rate = int(receiver["internal_sample_rate_hz"])
    duration = float(receiver["duration_seconds"])
    raw = output_dir / "raw"
    return f"""[GNSS-SDR]
GNSS-SDR.internal_fs_sps={internal_rate}

SignalSource.implementation=File_Signal_Source
SignalSource.filename={iq_path.resolve()}
SignalSource.item_type=byte
SignalSource.sampling_frequency={input_rate}
SignalSource.samples={byte_source_item_count(duration, input_rate)}
SignalSource.repeat=false
SignalSource.dump=false
SignalSource.enable_throttle_control=false

SignalConditioner.implementation=Signal_Conditioner
DataTypeAdapter.implementation=Byte_To_Short
InputFilter.implementation=Freq_Xlating_Fir_Filter
InputFilter.input_item_type=short
InputFilter.output_item_type=gr_complex
InputFilter.taps_item_type=float
InputFilter.filter_type=lowpass
InputFilter.bw=2100000
InputFilter.tw=500000
InputFilter.sampling_frequency={input_rate}
InputFilter.IF={int(dataset['intermediate_frequency_hz'])}
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


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def committed_release() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        _git("ls-files", "--error-unmatch", relative)
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT)
        if dirty.returncode:
            raise ValueError(f"release input is not committed and clean: {relative}")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "inputs": {relative: {"sha256": file_hash(ROOT / relative)} for relative in RELEASE_INPUTS},
    }


def _channel_number(path: Path) -> int:
    match = re.search(r"_ch_(\d+)\.mat$", path.name)
    return int(match.group(1)) if match else 10**9


def _tracking_summary(paths: list[Path]) -> tuple[list[int], int]:
    import h5py
    import numpy as np

    prns: set[int] = set()
    epochs = 0
    sentinel = np.asarray([1, 0])
    for path in paths:
        with h5py.File(path, "r") as handle:
            values = np.asarray(handle["PRN"]).reshape(-1) if "PRN" in handle else np.asarray([])
        if values.shape == (2,) and np.array_equal(values, sentinel):
            continue
        valid = [int(value) for value in values if 1 <= int(value) <= 32]
        prns.update(valid)
        epochs += len(valid)
    return sorted(prns), epochs


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    release = committed_release()
    dataset, receiver = config["dataset"], config["receiver"]
    iq_path = _resolve(dataset["path"])
    if not iq_path.is_file() or iq_path.stat().st_size != int(dataset["expected_bytes"]):
        raise ValueError("FGI TGD_L1_E1.dat byte count mismatch")
    observed_source_hash = file_hash(iq_path)
    if observed_source_hash != dataset["sha256"]:
        raise ValueError("FGI TGD_L1_E1.dat SHA-256 mismatch")
    executable = _resolve(receiver["executable"])
    executable_match = shutil.which(str(executable))
    executable = Path(executable_match).resolve() if executable_match else executable
    if not executable.is_file() or file_hash(executable) != receiver["executable_sha256"]:
        raise ValueError("pinned complex-nine-tap receiver mismatch")
    output = _resolve(config["output_root"])
    if output.exists():
        raise FileExistsError(output)
    raw = output / "receiver" / "raw"
    raw.mkdir(parents=True)
    receiver_dir = output / "receiver"
    receiver_config = receiver_dir / "receiver.conf"
    receiver_config.write_text(
        render_config(iq_path=iq_path, output_dir=receiver_dir, config=config), encoding="utf-8"
    )
    command = [str(executable), f"--config_file={receiver_config}", "--keyboard=false"]
    log_path = receiver_dir / "receiver.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=receiver_dir, stdout=log, stderr=subprocess.STDOUT,
            timeout=7200, check=False, text=True,
        )
    mats = sorted(raw.glob("epl_tracking_ch_*.mat"), key=_channel_number)
    prns, epochs = _tracking_summary(mats)
    manifest = {
        "schema": "gnss-doppler-lab.fgi-spoofrepo-tgd-receiver.v1",
        "schema_version": 1,
        "receiver_run_id": "fgi-spoofrepo-tgd-complex9-prefix-240s",
        "source": {
            "dataset": dataset["name"],
            "scenario_id": dataset["scenario"],
            "iq": str(iq_path),
            "iq_bytes": iq_path.stat().st_size,
            "iq_sha256": observed_source_hash,
            "sample_format": dataset["sample_format"],
            "input_sample_rate_hz": dataset["input_sample_rate_hz"],
            "intermediate_frequency_hz": dataset["intermediate_frequency_hz"],
            "decimation_factor": receiver["decimation_factor"],
            "sample_rate_hz": receiver["internal_sample_rate_hz"],
            "start_offset_s": 0.0,
            "requested_duration_s": receiver["duration_seconds"]
        },
        "receiver": {
            "name": "GNSS-SDR Method-A complex-nine-tap",
            "executable": str(executable),
            "executable_sha256": file_hash(executable),
            "config": receiver_config.name,
            "config_sha256": file_hash(receiver_config),
            "command": command,
            "exit_code": completed.returncode
        },
        "acquisition": {
            "channel_count": receiver["channel_count"],
            "tracked_prns": [f"G{prn:02d}" for prn in prns],
            "valid_epoch_count": epochs
        },
        "tracking": {
            "raw_directory": "raw",
            "mat_file_count": len(mats),
            "tap_count": 9,
            "tap_spacing_chips": 0.125,
            "complex_taps_required": True
        }
    }
    manifest_path = receiver_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"GNSS-SDR exited {completed.returncode}; see {log_path}")
    if not prns:
        raise RuntimeError(f"GNSS-SDR produced no valid tracking epochs; see {log_path}")
    gate = config["support_gate"]
    audits: dict[str, Any] = {}
    for name, interval in config["analysis_intervals_seconds"].items():
        audits[name] = audit_clean_geometry_support(
            receiver_dir, start_s=float(interval[0]), end_s=float(interval[1]),
            bin_seconds=float(gate["bin_seconds"]),
            minimum_epochs=int(gate["minimum_epochs_per_prn_bin"]),
            minimum_primary_prns=int(gate["minimum_primary_prns"]),
            secondary_boundary_prns=int(gate["secondary_boundary_prns"]),
            minimum_primary_bins=int(gate["minimum_primary_bins_each_interval"]),
            require_complex_nine_tap=True,
        )
    eligible = all(audit["support_eligible"] for audit in audits.values())
    result = {
        "schema": "gnss-doppler-lab.fgi-spoofrepo-tgd-support-preflight-result",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release": release,
        "config": {"path": str(config_path), "sha256": file_hash(config_path)},
        "protocol": {"path": str(PROTOCOL), "sha256": file_hash(PROTOCOL)},
        "score_accessed": False,
        "delay_template_accessed": False,
        "detector_loaded": False,
        "receiver_manifest": {"path": str(manifest_path), "sha256": file_hash(manifest_path)},
        "support_audits": audits,
        "decision": "SUPPORT_ELIGIBLE" if eligible else "INSUFFICIENT_SUPPORT",
        "claim_boundary": config["claim_boundary"]
    }
    summary = output / "summary.json"
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(summary),
        "decision": result["decision"],
        "clean_maximum_eligible_prns": audits["clean"]["maximum_eligible_prns"],
        "post_onset_maximum_eligible_prns": audits["post_onset_support"]["maximum_eligible_prns"],
        "score_accessed": False
    }, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release-token", required=True)
    args = parser.parse_args()
    if args.release_token != RELEASE_TOKEN:
        raise ValueError("release token mismatch")
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

