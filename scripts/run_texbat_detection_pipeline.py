#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[0]
# This script is copied/run from repo root via python /tmp path; force repo cwd modules.
REPO_ROOT = Path.cwd()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gnss_doppler_lab.gnss_sdr import export_tracking_csv, parse_acquired_prns, parse_receiver_reported_prns  # noqa:E402
from gnss_doppler_lab.tracking_feature_dataset import export_tracking_feature_dataset  # noqa:E402
from gnss_doppler_lab.normal_multi_prn_dataset import export_normal_multi_prn_dataset, NODE_FEATURE_COLUMNS  # noqa:E402

import importlib.util
spec = importlib.util.spec_from_file_location("train_conditional_integrated_gru", str(REPO_ROOT / "scripts/train_conditional_integrated_gru.py"))
train_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train_mod
spec.loader.exec_module(train_mod)

SCENARIOS = {
    "cleanStatic": {"slug": "clean-static-reference", "title": "TEXBAT cleanStatic reference", "label": "texbat_clean_static"},
    "cleanDynamic": {"slug": "clean-dynamic-reference", "title": "TEXBAT cleanDynamic reference", "label": "texbat_clean_dynamic"},
    "ds1": {"slug": "static-switch", "title": "TEXBAT ds1 static switch", "label": "texbat_ds1_spoofing"},
    "ds2": {"slug": "static-overpowered-time-push", "title": "TEXBAT ds2 static overpowered time push", "label": "texbat_ds2_spoofing"},
    "ds4": {"slug": "static-matched-power-position-push", "title": "TEXBAT ds4 static matched-power position push", "label": "texbat_ds4_spoofing"},
}
SCENARIO_ID = os.environ.get("TEXBAT_SCENARIO", "ds4")
if SCENARIO_ID not in SCENARIOS:
    raise SystemExit(f"Unsupported TEXBAT_SCENARIO={SCENARIO_ID}; choose {sorted(SCENARIOS)}")
SCENARIO = SCENARIOS[SCENARIO_ID]
RUN_ID = f"texbat-{SCENARIO_ID}-{SCENARIO['slug']}"
RAW = Path(f"data/external/texbat/raw/{SCENARIO_ID}.bin")
OUT = Path(f"artifacts/texbat_{SCENARIO_ID}_detection_normal_v3_large_50")
RECEIVER_DIR = OUT / "receiver" / RUN_ID
FEATURE_CSV = OUT / "tracking_features.csv"
MODEL_DIR = Path(os.environ.get("TEXBAT_MODEL_DIR", "artifacts/conditional_integrated_gru_normal_v3_large_50"))
CKPT_PATH = MODEL_DIR / "conditional_integrated_gru_predictor.pt"
TRAIN_SUMMARY = MODEL_DIR / "training_summary.json"
NORMAL_SCORES = MODEL_DIR / "validation_conditional_scores.csv"
SAMPLE_RATE = 25_000_000
CHANNELS = 11

def sha256(path: Path, block=8*1024*1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()

def channel_number(path: Path) -> int:
    import re
    m = re.search(r"_ch_(\d+)\.mat$", path.name)
    return int(m.group(1)) if m else 9999

def receiver_config(iq: Path, run_dir: Path) -> str:
    tracking_prefix = run_dir / "raw" / "epl_tracking_ch_"
    observables = run_dir / "raw" / "observables.dat"
    return f"""[GNSS-SDR]
GNSS-SDR.internal_fs_sps={SAMPLE_RATE}

SignalSource.implementation=File_Signal_Source
SignalSource.filename={iq.resolve()}
SignalSource.item_type=ishort
SignalSource.sampling_frequency={SAMPLE_RATE}
SignalSource.samples=0
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

def run_receiver_if_needed(force: bool = False) -> Path:
    manifest = RECEIVER_DIR / "manifest.json"
    if manifest.exists() and not force:
        return manifest
    if RECEIVER_DIR.exists():
        shutil.rmtree(RECEIVER_DIR)
    raw_dir = RECEIVER_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    config_path = RECEIVER_DIR / "receiver.conf"
    log_path = RECEIVER_DIR / "receiver.log"
    config_path.write_text(receiver_config(RAW, RECEIVER_DIR), encoding="utf-8")
    exe = shutil.which("gnss-sdr") or "gnss-sdr"
    version_result = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
    version = (version_result.stdout or version_result.stderr).strip().splitlines()[0]
    cmd = [exe, f"--config_file={config_path.resolve()}", "--keyboard=false"]
    result = subprocess.run(cmd, cwd=RECEIVER_DIR, capture_output=True, text=True, timeout=1800)
    log_text = result.stdout + result.stderr
    log_path.write_text(log_text, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"GNSS-SDR failed rc={result.returncode}; see {log_path}")
    mats = sorted(raw_dir.glob("epl_tracking_ch_*.mat"), key=channel_number)
    if not mats:
        raise RuntimeError("GNSS-SDR produced no tracking MAT files")
    report = export_tracking_csv(mats, RECEIVER_DIR / "tracking.csv", RECEIVER_DIR / "tracking_summary.csv", sample_rate_hz=SAMPLE_RATE)
    tracked = parse_acquired_prns(log_text)
    reported = parse_receiver_reported_prns(log_text)
    raw_sha = sha256(RAW)
    doc = {
        "schema_version": 1,
        "receiver_run_id": RUN_ID,
        "source_rf_run_id": RUN_ID,
        "source": {"dataset": "TEXBAT", "scenario_id": SCENARIO_ID, "iq": str(RAW), "iq_sha256": raw_sha, "sample_rate_hz": SAMPLE_RATE, "sample_format": "ishort_complex_iq"},
        "receiver": {"name": "GNSS-SDR", "version": version, "executable": exe, "config": config_path.name, "command": cmd, "exit_code": result.returncode},
        "acquisition": {"channel_count": CHANNELS, "tracked_prns": tracked, "tracked_prn_count": len(tracked), "receiver_reported_prns": reported, "receiver_reported_prn_count": len(reported)},
        "tracking": {**report, "csv": "tracking.csv", "summary_csv": "tracking_summary.csv", "raw_directory": "raw"},
    }
    manifest.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def sanitize_tracking_features(path: Path) -> dict:
    """Drop rows with non-finite model feature values before multi-PRN export.

    TEXBAT real RF tracking can contain rare windows where a derived statistic such
    as cn0_std is NaN because the receiver did not provide finite C/N0 samples for
    that window. The normal multi-PRN builder intentionally rejects non-finite
    values; for external corpus processing we drop only those bad windows and keep
    the original tracking_features.csv plus a cleaned audit manifest.
    """
    df = pd.read_csv(path)
    before = int(len(df))
    mask = np.ones(before, dtype=bool)
    bad_counts = {}
    for column in NODE_FEATURE_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        bad = int((~finite).sum())
        if bad:
            bad_counts[column] = bad
        mask &= finite
    dropped = int((~mask).sum())
    if dropped:
        backup = path.with_name(path.stem + ".raw_nonfinite.csv")
        if not backup.exists():
            path.replace(backup)
        else:
            path.unlink()
        cleaned = df.loc[mask].copy()
        cleaned.to_csv(path, index=False)
        manifest = {
            "schema": "gnss-doppler-lab.tracking-features-sanitization",
            "input_backup_csv": str(backup),
            "output_csv": str(path),
            "rows_before": before,
            "rows_after": int(len(cleaned)),
            "rows_dropped": dropped,
            "bad_counts_by_feature": bad_counts,
        }
        path.with_suffix(".sanitization.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        return manifest
    return {"rows_before": before, "rows_after": before, "rows_dropped": 0, "bad_counts_by_feature": {}}

def build_texbat_features(force: bool = False):
    rec_manifest = run_receiver_if_needed(force=force)
    if force or not FEATURE_CSV.exists():
        export_tracking_feature_dataset([RECEIVER_DIR], output_path=FEATURE_CSV, manifest_output_path=FEATURE_CSV.with_suffix(".manifest.json"), window_s=1.0, stride_s=0.5, min_epochs=4, label=SCENARIO["label"])
    sanitize_tracking_features(FEATURE_CSV)
    multi_dir = OUT / "multi_prn_morphology_dynamics_v3"
    node_csv = multi_dir / "normal_prn_node_windows.csv"
    graph_csv = multi_dir / "normal_receiver_graph_windows.csv"
    if force or not (node_csv.exists() and graph_csv.exists()):
        node_csv, graph_csv, _ = export_normal_multi_prn_dataset(FEATURE_CSV, output_dir=multi_dir, stride_s=0.5, min_prns_per_graph=2)
    return Path(node_csv), Path(graph_csv)

def numeric_feature_columns(df: pd.DataFrame, meta: set[str]) -> list[str]:
    return [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]

def score_texbat(node_csv: Path, graph_csv: Path):
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    cfg = train_mod.TrainConfig(**ckpt["config"])
    node_cols = ckpt["node_feature_columns"]
    graph_cols = ckpt["graph_feature_columns"]
    std = ckpt["standardizer"]
    node_mean = np.asarray(std["node_mean"], dtype=np.float32); node_std = np.asarray(std["node_std"], dtype=np.float32)
    graph_mean = np.asarray(std["graph_mean"], dtype=np.float32); graph_std = np.asarray(std["graph_std"], dtype=np.float32)
    node_df = pd.read_csv(node_csv)
    graph_df = pd.read_csv(graph_csv)
    groups = []
    meta = []
    for run_id, g_run in graph_df.groupby("run_id", sort=True):
        n_run = node_df[node_df["run_id"] == run_id]
        node_sets = []
        graph_rows = []
        bins = []
        starts = []
        ends = []
        prn_counts = []
        for _, grow in g_run.sort_values("window_bin_s").iterrows():
            bin_s = grow["window_bin_s"]
            nodes = n_run[n_run["window_bin_s"] == bin_s].sort_values("prn")
            if nodes.empty:
                continue
            nx = train_mod.standardize(nodes[node_cols].to_numpy(np.float32), node_mean, node_std)
            gx = train_mod.standardize(grow[graph_cols].to_numpy(np.float32)[None, :], graph_mean, graph_std)[0]
            node_sets.append(nx); graph_rows.append(gx)
            bins.append(float(bin_s)); starts.append(float(grow["window_start_s_min"])); ends.append(float(grow["window_end_s_max"])); prn_counts.append(int(grow["tracked_prn_count"]))
        if len(node_sets) >= cfg.seq_len + 1:
            node_obj = np.empty(len(node_sets), dtype=object); node_obj[:] = node_sets
            groups.append((node_obj, np.stack(graph_rows).astype(np.float32)))
            meta.append({"run_id": run_id, "bins": bins, "starts": starts, "ends": ends, "prn_counts": prn_counts})
    if not groups:
        raise RuntimeError("no scoreable TEXBAT sequences")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_mod.ConditionalIntegratedGRU(len(node_cols), len(graph_cols), len(node_cols)*3, cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    train_summary = json.loads(TRAIN_SUMMARY.read_text())
    gate_center = float(train_summary["gate_center_node_rmse_q90"])
    normal_scores = pd.read_csv(NORMAL_SCORES)
    thresholds = {"q90": float(normal_scores["joint_score"].quantile(0.90)), "q95": float(normal_scores["joint_score"].quantile(0.95)), "q99": float(normal_scores["joint_score"].quantile(0.99))}
    rows = []
    mse = torch.nn.MSELoss(reduction="none")
    with torch.no_grad():
        for gi, (nodes, graphs) in enumerate(groups):
            m = meta[gi]
            for start in range(0, len(nodes) - cfg.seq_len):
                # target is start+seq_len
                ds = train_mod.PrnSetSequenceDataset([(nodes, graphs)], cfg.seq_len, cfg.max_prns)
                node_seq, mask_seq, graph_seq, target_nodes, target_graph = ds[start]
                node_seq=node_seq[None].to(device); mask_seq=mask_seq[None].to(device); graph_seq=graph_seq[None].to(device)
                target_nodes=target_nodes[None].to(device); target_graph=target_graph[None].to(device)
                pred_node, pred_graph = model(node_seq, mask_seq, graph_seq)
                node_rmse = torch.sqrt(((pred_node-target_nodes)**2).mean(dim=1))[0].item()
                graph_rmse = torch.sqrt(((pred_graph-target_graph)**2).mean(dim=1))[0].item()
                gate = torch.sigmoid((torch.tensor(node_rmse)-gate_center)*8.0).item()
                joint = node_rmse + gate*graph_rmse
                ti = start + cfg.seq_len
                row = {"run_id": m["run_id"], "target_window_index": ti, "window_bin_s": m["bins"][ti], "window_start_s": m["starts"][ti], "window_end_s": m["ends"][ti], "tracked_prn_count": m["prn_counts"][ti], "node_rmse": node_rmse, "graph_rmse": graph_rmse, "relation_gate": gate, "joint_score": joint}
                for name, th in thresholds.items():
                    row[f"above_{name}"] = bool(joint > th)
                rows.append(row)
    scores = pd.DataFrame(rows)
    scores_path = OUT / f"texbat_{SCENARIO_ID}_conditional_scores.csv"
    scores.to_csv(scores_path, index=False)
    summary = {"schema": f"gnss-doppler-lab.texbat-{SCENARIO_ID}-detection-summary", "scenario": SCENARIO["title"], "model": str(CKPT_PATH), "node_csv": str(node_csv), "graph_csv": str(graph_csv), "score_csv": str(scores_path), "gate_center_node_rmse_q90": gate_center, "normal_joint_thresholds": thresholds, "rows": len(scores), "time_range_s": [float(scores.window_start_s.min()), float(scores.window_end_s.max())], "score_summary": scores[["node_rmse","graph_rmse","relation_gate","joint_score"]].describe(percentiles=[0.5,0.9,0.95,0.99]).to_dict()}
    for name, th in thresholds.items():
        mask = scores[f"above_{name}"]
        summary[f"detection_{name}"] = {"threshold": th, "flagged_windows": int(mask.sum()), "flagged_fraction": float(mask.mean()), "first_flag_window_start_s": float(scores.loc[mask, "window_start_s"].min()) if mask.any() else None}
    (OUT / f"texbat_{SCENARIO_ID}_detection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary

def main():
    global SCENARIO_ID, SCENARIO, RUN_ID, RAW, OUT, RECEIVER_DIR, FEATURE_CSV
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default=SCENARIO_ID)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    SCENARIO_ID = args.scenario
    SCENARIO = SCENARIOS[SCENARIO_ID]
    RUN_ID = f"texbat-{SCENARIO_ID}-{SCENARIO['slug']}"
    RAW = Path(f"data/external/texbat/raw/{SCENARIO_ID}.bin")
    OUT = Path(f"artifacts/texbat_{SCENARIO_ID}_detection_normal_v3_large_50")
    RECEIVER_DIR = OUT / "receiver" / RUN_ID
    FEATURE_CSV = OUT / "tracking_features.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        raise SystemExit(f"missing raw TEXBAT file: {RAW}")
    node_csv, graph_csv = build_texbat_features(force=args.force)
    summary = score_texbat(node_csv, graph_csv)
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
