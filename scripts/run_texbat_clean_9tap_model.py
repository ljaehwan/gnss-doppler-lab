#!/usr/bin/env python3
"""Build a normal-only Method-A 9-tap model from TEXBAT clean reference data.

This script intentionally requires real 9-tap tracking MAT datasets. Stock GNSS-SDR
0.0.19 writes only real E/P/L plus zero VE/VL placeholders, so use a patched
Method-A receiver executable that honors Tracking_1C.tap_count=9.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SRC_ROOT = Path.cwd() / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gnss_doppler_lab.gnss_sdr import export_tracking_csv, parse_acquired_prns, parse_receiver_reported_prns
from gnss_doppler_lab.tracking_feature_windows import export_receiver_run_tap_feature_csv
from gnss_doppler_lab.normal_multi_prn_dataset import export_tap_multi_prn_dataset

SAMPLE_RATE = 25_000_000
CHANNELS = 11


def sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def channel_number(path: Path) -> int:
    import re
    m = re.search(r"_ch_(\d+)\.mat$", path.name)
    return int(m.group(1)) if m else 9999


def receiver_config(iq: Path, run_dir: Path, *, tap_count: int, tap_spacing_chips: float, samples: int = 0) -> str:
    tracking_prefix = run_dir / "raw" / "epl_tracking_ch_"
    observables = run_dir / "raw" / "observables.dat"
    return f"""[GNSS-SDR]
GNSS-SDR.internal_fs_sps={SAMPLE_RATE}

SignalSource.implementation=File_Signal_Source
SignalSource.filename={iq.resolve()}
SignalSource.item_type=ishort
SignalSource.sampling_frequency={SAMPLE_RATE}
SignalSource.samples={samples}
SignalSource.repeat=false
SignalSource.dump=false
SignalSource.enable_throttle_control=false

SignalConditioner.implementation=Signal_Conditioner
DataTypeAdapter.implementation=Ishort_To_Complex
InputFilter.implementation=Pass_Through
InputFilter.input_item_type=gr_complex
InputFilter.output_item_type=gr_complex
Resampler.implementation=Pass_Through
Resampler.item_type=gr_complex

Channels_1C.count={CHANNELS}
Channels.in_acquisition={CHANNELS}
Channel.signal=1C

Acquisition_1C.implementation=GPS_L1_CA_PCPS_Acquisition
Acquisition_1C.item_type=gr_complex
Acquisition_1C.coherent_integration_time_ms=1
Acquisition_1C.threshold=2.5
Acquisition_1C.doppler_max=10000
Acquisition_1C.doppler_step=100
Acquisition_1C.dump=false

Tracking_1C.implementation=GPS_L1_CA_DLL_PLL_Tracking
Tracking_1C.item_type=gr_complex
Tracking_1C.pll_bw_hz=20.0
Tracking_1C.dll_bw_hz=1.5
Tracking_1C.order=3
Tracking_1C.dump=true
Tracking_1C.dump_filename={tracking_prefix.resolve()}
Tracking_1C.tap_count={tap_count}
Tracking_1C.tap_spacing_chips={tap_spacing_chips}

TelemetryDecoder_1C.implementation=GPS_L1_CA_Telemetry_Decoder
TelemetryDecoder_1C.dump=false

Observables.implementation=Hybrid_Observables
Observables.dump=true
Observables.dump_filename={observables.resolve()}

PVT.implementation=RTKLIB_PVT
PVT.positioning_mode=Single
PVT.output_rate_ms=100
PVT.display_rate_ms=500
PVT.flag_rtcm_server=false
PVT.flag_rtcm_tty_port=false
PVT.dump=false
"""


def run_receiver(raw: Path, out: Path, scenario: str, *, exe: str, force: bool, tap_count: int, tap_spacing_chips: float, samples: int) -> Path:
    run_id = f"texbat-{scenario}-method-a-9tap-clean-reference"
    receiver_dir = out / "receiver" / run_id
    manifest = receiver_dir / "manifest.json"
    if manifest.exists() and not force:
        return manifest
    if receiver_dir.exists():
        shutil.rmtree(receiver_dir)
    raw_dir = receiver_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    config_path = receiver_dir / "receiver.conf"
    log_path = receiver_dir / "receiver.log"
    config_path.write_text(receiver_config(raw, receiver_dir, tap_count=tap_count, tap_spacing_chips=tap_spacing_chips, samples=samples), encoding="utf-8")
    version_result = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
    version = (version_result.stdout or version_result.stderr).strip().splitlines()[0] if (version_result.stdout or version_result.stderr) else "unknown"
    cmd = [exe, f"--config_file={config_path.resolve()}", "--keyboard=false"]
    result = subprocess.run(cmd, cwd=receiver_dir, capture_output=True, text=True, timeout=3600)
    log_text = result.stdout + result.stderr
    log_path.write_text(log_text, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"GNSS-SDR failed rc={result.returncode}; see {log_path}")
    mats = sorted(raw_dir.glob("epl_tracking_ch_*.mat"), key=channel_number)
    if not mats:
        raise RuntimeError("GNSS-SDR produced no tracking MAT files")
    report = export_tracking_csv(mats, receiver_dir / "tracking.csv", receiver_dir / "tracking_summary.csv", sample_rate_hz=SAMPLE_RATE)
    doc = {
        "schema_version": 1,
        "receiver_run_id": run_id,
        "source_rf_run_id": run_id,
        "source": {"dataset": "TEXBAT", "scenario_id": scenario, "iq": str(raw), "iq_sha256": sha256(raw), "sample_rate_hz": SAMPLE_RATE, "sample_format": "ishort_complex_iq"},
        "receiver": {"name": "GNSS-SDR Method-A", "version": version, "executable": exe, "config": config_path.name, "command": cmd, "exit_code": result.returncode},
        "acquisition": {"channel_count": CHANNELS, "tracked_prns": parse_acquired_prns(log_text), "receiver_reported_prns": parse_receiver_reported_prns(log_text)},
        "tracking": {**report, "csv": "tracking.csv", "summary_csv": "tracking_summary.csv", "raw_directory": "raw", "tap_count": tap_count, "tap_spacing_chips": tap_spacing_chips},
    }
    manifest.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["cleanStatic", "cleanDynamic"], default="cleanStatic")
    ap.add_argument("--exe", default=os.environ.get("GNSS_SDR_METHOD_A_EXE", shutil.which("gnss-sdr") or "gnss-sdr"))
    ap.add_argument("--out", default="artifacts/texbat_cleanStatic_method_a_9tap_model")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tap-spacing-chips", type=float, default=0.125)
    ap.add_argument("--samples", type=int, default=0, help="GNSS-SDR SignalSource.samples; 0 means full file")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--feature-mode", choices=["all", "normalized_dmcpd"], default="all")
    args = ap.parse_args()
    raw = Path(f"data/external/texbat/raw/{args.scenario}.bin")
    if not raw.exists():
        raise SystemExit(f"missing TEXBAT raw file: {raw}")
    out = Path(args.out)
    if args.scenario == "cleanDynamic" and "cleanStatic" in str(out):
        out = Path(str(out).replace("cleanStatic", "cleanDynamic"))
    out.mkdir(parents=True, exist_ok=True)
    manifest = run_receiver(raw, out, args.scenario, exe=args.exe, force=args.force, tap_count=9, tap_spacing_chips=args.tap_spacing_chips, samples=args.samples)
    receiver_dir = manifest.parent
    feature_csv = out / "tap9_tracking_features.csv"
    if args.force or not feature_csv.exists():
        export_receiver_run_tap_feature_csv(receiver_dir, output_path=feature_csv, tap_count=9, window_s=1.0, stride_s=0.5, min_epochs=4, label=f"texbat_{args.scenario}_normal_9tap")
    suffix = "" if args.feature_mode == "all" else f"_{args.feature_mode}"
    multi_dir = out / f"multi_prn_method_a_9tap{suffix}"
    node_csv, graph_csv, dataset_manifest = export_tap_multi_prn_dataset(feature_csv, output_dir=multi_dir, stride_s=0.5, min_prns_per_graph=2, feature_mode=args.feature_mode)
    model_dir = out / f"conditional_integrated_gru_9tap{suffix}"
    cmd = [sys.executable, "scripts/train_conditional_integrated_gru.py", "--node-csv", str(node_csv), "--graph-csv", str(graph_csv), "--output-dir", str(model_dir), "--epochs", str(args.epochs), "--seq-len", str(args.seq_len), "--batch-size", str(args.batch_size)]
    subprocess.run(cmd, check=True)
    summary = {"receiver_manifest": str(manifest), "tap_feature_csv": str(feature_csv), "feature_mode": args.feature_mode, "dataset_manifest": str(dataset_manifest), "model_summary": str(model_dir / "training_summary.json")}
    (out / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
