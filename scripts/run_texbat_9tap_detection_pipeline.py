#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SRC_ROOT = Path.cwd() / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gnss_doppler_lab.gnss_sdr import export_tracking_csv, parse_acquired_prns, parse_receiver_reported_prns
from gnss_doppler_lab.tracking_feature_windows import export_receiver_run_tap_feature_csv
from gnss_doppler_lab.normal_multi_prn_dataset import export_tap_multi_prn_dataset

import importlib.util
spec = importlib.util.spec_from_file_location("train_conditional_integrated_gru", str(Path.cwd() / "scripts/train_conditional_integrated_gru.py"))
train_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train_mod
spec.loader.exec_module(train_mod)

SAMPLE_RATE = 25_000_000
CHANNELS = 11
SCENARIOS = {
    "ds1": "TEXBAT ds1 static switch",
    "ds2": "TEXBAT ds2 static overpowered time push",
    "ds3": "TEXBAT ds3 static matched-power time push",
    "ds4": "TEXBAT ds4 static matched-power position push",
    "ds5": "TEXBAT ds5 dynamic overpowered time push",
    "ds6": "TEXBAT ds6 dynamic matched-power position push",
}
ONSET_S = 100.0


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


def receiver_config(iq: Path, run_dir: Path, *, tap_count: int, tap_spacing_chips: float, samples: int) -> str:
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


def run_receiver(scenario: str, raw: Path, out: Path, *, exe: str, force: bool, samples: int) -> Path:
    run_id = f"texbat-{scenario}-method-a-9tap-external-validation"
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
    config_path.write_text(receiver_config(raw, receiver_dir, tap_count=9, tap_spacing_chips=0.125, samples=samples), encoding="utf-8")
    version_result = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
    version = (version_result.stdout or version_result.stderr).strip().splitlines()[0] if (version_result.stdout or version_result.stderr) else "unknown"
    cmd = [exe, f"--config_file={config_path.resolve()}", "--keyboard=false"]
    result = subprocess.run(cmd, cwd=receiver_dir, capture_output=True, text=True, timeout=3600)
    log_text = result.stdout + result.stderr
    log_path.write_text(log_text, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"GNSS-SDR failed for {scenario} rc={result.returncode}; see {log_path}")
    mats = sorted(raw_dir.glob("epl_tracking_ch_*.mat"), key=channel_number)
    if not mats:
        import re
        fallback_patterns = [
            ("epl_tracking_c*.mat", r"epl_tracking_c(\d+)\.mat$"),
            ("e*.mat", r"e(\d+)\.mat$"),
        ]
        for pattern, regex in fallback_patterns:
            for src in sorted(raw_dir.glob(pattern), key=channel_number):
                m = re.search(regex, src.name)
                if m:
                    dst = raw_dir / f"epl_tracking_ch_{m.group(1)}.mat"
                    if not dst.exists():
                        dst.symlink_to(src.name)
        mats = sorted(raw_dir.glob("epl_tracking_ch_*.mat"), key=channel_number)
    if not mats:
        raise RuntimeError(f"GNSS-SDR produced no tracking MAT files for {scenario}")
    report = export_tracking_csv(mats, receiver_dir / "tracking.csv", receiver_dir / "tracking_summary.csv", sample_rate_hz=SAMPLE_RATE)
    doc = {
        "schema_version": 1,
        "receiver_run_id": run_id,
        "source_rf_run_id": run_id,
        "source": {"dataset": "TEXBAT", "scenario_id": scenario, "iq": str(raw), "iq_sha256": sha256(raw), "sample_rate_hz": SAMPLE_RATE, "sample_format": "ishort_complex_iq"},
        "receiver": {"name": "GNSS-SDR Method-A", "version": version, "executable": exe, "config": config_path.name, "command": cmd, "exit_code": result.returncode},
        "acquisition": {"channel_count": CHANNELS, "tracked_prns": parse_acquired_prns(log_text), "receiver_reported_prns": parse_receiver_reported_prns(log_text)},
        "tracking": {**report, "csv": "tracking.csv", "summary_csv": "tracking_summary.csv", "raw_directory": "raw", "tap_count": 9, "tap_spacing_chips": 0.125},
    }
    manifest.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_features(scenario: str, out: Path, manifest: Path, *, force: bool, feature_mode: str = "all"):
    feature_csv = out / "tap9_tracking_features_w1.0_s0.5.csv"
    if force or not feature_csv.exists():
        export_receiver_run_tap_feature_csv(manifest.parent, output_path=feature_csv, tap_count=9, window_s=1.0, stride_s=0.5, min_epochs=4, label=f"texbat_{scenario}_spoofing_9tap")
        df = pd.read_csv(feature_csv)
        numeric = df.select_dtypes(include=[np.number]).columns
        mask = np.isfinite(df[numeric].to_numpy(float)).all(axis=1)
        if not bool(mask.all()):
            backup = feature_csv.with_name(feature_csv.stem + ".raw_nonfinite.csv")
            if not backup.exists():
                feature_csv.rename(backup)
            df.loc[mask].to_csv(feature_csv, index=False)
            feature_csv.with_suffix(".sanitization.json").write_text(json.dumps({"rows_before": int(len(df)), "rows_after": int(mask.sum()), "rows_dropped": int((~mask).sum())}, indent=2) + "\n")
    suffix = "" if feature_mode == "all" else f"_{feature_mode}"
    multi_dir = out / f"multi_prn_method_a_9tap_w1.0_s0.5{suffix}"
    node_csv = multi_dir / "normal_prn_node_windows.csv"
    graph_csv = multi_dir / "normal_receiver_graph_windows.csv"
    if force or not (node_csv.exists() and graph_csv.exists()):
        node_csv, graph_csv, _ = export_tap_multi_prn_dataset(feature_csv, output_dir=multi_dir, stride_s=0.5, min_prns_per_graph=2, feature_mode=feature_mode)
    return Path(node_csv), Path(graph_csv), feature_csv


def score(node_csv: Path, graph_csv: Path, *, model_dir: Path, out: Path, scenario: str):
    ckpt_path = model_dir / "conditional_integrated_gru_predictor.pt"
    train_summary_path = model_dir / "training_summary.json"
    normal_scores_path = model_dir / "validation_conditional_scores.csv"
    ckpt = torch.load(ckpt_path, map_location="cpu")
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
        node_sets=[]; graph_rows=[]; bins=[]; starts=[]; ends=[]; counts=[]
        for _, grow in g_run.sort_values("window_bin_s").iterrows():
            bin_s = grow["window_bin_s"]
            nodes = n_run[n_run["window_bin_s"] == bin_s].sort_values("prn")
            if nodes.empty:
                continue
            nx = train_mod.standardize(nodes[node_cols].to_numpy(np.float32), node_mean, node_std)
            gx = train_mod.standardize(grow[graph_cols].to_numpy(np.float32)[None,:], graph_mean, graph_std)[0]
            node_sets.append(nx); graph_rows.append(gx); bins.append(float(bin_s)); starts.append(float(grow["window_start_s_min"])); ends.append(float(grow["window_end_s_max"])); counts.append(int(grow["tracked_prn_count"]))
        if len(node_sets) >= cfg.seq_len + 1:
            obj = np.empty(len(node_sets), dtype=object); obj[:] = node_sets
            groups.append((obj, np.stack(graph_rows).astype(np.float32)))
            meta.append({"run_id": run_id, "bins": bins, "starts": starts, "ends": ends, "counts": counts})
    if not groups:
        raise RuntimeError(f"no scoreable sequences for {scenario}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_mod.ConditionalIntegratedGRU(len(node_cols), len(graph_cols), len(node_cols)*3, cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    train_summary = json.loads(train_summary_path.read_text())
    gate_center = float(train_summary["gate_center_node_rmse_q90"])
    normal_scores = pd.read_csv(normal_scores_path)
    pfa_quantiles = {"pfa10_q90":0.90,"pfa5_q95":0.95,"pfa1_q99":0.99,"pfa0_5_q995":0.995,"pfa0_1_q999":0.999}
    thresholds = {q: float(normal_scores["joint_score"].quantile(v)) for q,v in pfa_quantiles.items()}
    rows=[]
    with torch.no_grad():
        for gi,(nodes,graphs) in enumerate(groups):
            m=meta[gi]
            ds = train_mod.PrnSetSequenceDataset([(nodes, graphs)], cfg.seq_len, cfg.max_prns)
            for start in range(0, len(nodes)-cfg.seq_len):
                node_seq, mask_seq, graph_seq, target_nodes, target_graph = ds[start]
                node_seq=node_seq[None].to(device); mask_seq=mask_seq[None].to(device); graph_seq=graph_seq[None].to(device)
                target_nodes=target_nodes[None].to(device); target_graph=target_graph[None].to(device)
                pred_node,pred_graph = model(node_seq, mask_seq, graph_seq)
                node_rmse = torch.sqrt(((pred_node-target_nodes)**2).mean(dim=1))[0].item()
                graph_rmse = torch.sqrt(((pred_graph-target_graph)**2).mean(dim=1))[0].item()
                gate = torch.sigmoid((torch.tensor(node_rmse)-gate_center)*8.0).item()
                joint = node_rmse + gate*graph_rmse
                ti=start+cfg.seq_len
                row={"run_id":m["run_id"],"target_window_index":ti,"window_bin_s":m["bins"][ti],"window_start_s":m["starts"][ti],"window_end_s":m["ends"][ti],"tracked_prn_count":m["counts"][ti],"node_rmse":node_rmse,"graph_rmse":graph_rmse,"relation_gate":gate,"joint_score":joint}
                for name, th in thresholds.items(): row[f"above_{name}"] = bool(joint > th)
                rows.append(row)
    scores = pd.DataFrame(rows)
    scores_path = out / f"texbat_{scenario}_9tap_conditional_scores.csv"
    scores.to_csv(scores_path, index=False)
    summary = {
        "schema": f"gnss-doppler-lab.texbat-{scenario}-9tap-onset-aware-summary",
        "scenario": SCENARIOS[scenario],
        "model": str(ckpt_path),
        "normal_score_source": str(normal_scores_path),
        "node_csv": str(node_csv),
        "graph_csv": str(graph_csv),
        "score_csv": str(scores_path),
        "onset_s": ONSET_S,
        "rows": int(len(scores)),
        "time_range_s": [float(scores.window_start_s.min()), float(scores.window_end_s.max())],
        "normal_joint_thresholds": thresholds,
        "score_summary": scores[["node_rmse","graph_rmse","relation_gate","joint_score"]].describe(percentiles=[.5,.9,.95,.99]).to_dict(),
    }
    initial_clean = scores[scores["window_start_s"] < 90.0]
    initial_clean_thresholds = {name: float(initial_clean["joint_score"].quantile(q)) for name, q in pfa_quantiles.items()} if len(initial_clean) else {}
    summary["initial_clean_t_lt_90_joint_thresholds"] = initial_clean_thresholds
    all_threshold_sets = [("normal", thresholds), ("initial_clean_t_lt_90", initial_clean_thresholds)]
    for threshold_source, threshold_map in all_threshold_sets:
        for name, th in threshold_map.items():
            col = f"above_{threshold_source}_{name}"
            scores[col] = scores["joint_score"] > th
            flags = scores[col]
            pre = scores["window_start_s"] < ONSET_S
            post = scores["window_start_s"] >= ONSET_S
            pre90 = scores["window_start_s"] < 90.0
            post110 = scores["window_start_s"] >= 110.0
            post_flags = flags & post
            post110_flags = flags & post110
            first_post = float(scores.loc[post_flags, "window_start_s"].min()) if post_flags.any() else None
            first_post110 = float(scores.loc[post110_flags, "window_start_s"].min()) if post110_flags.any() else None
            summary[f"onset_metrics_{threshold_source}_{name}"] = {
                "threshold": th,
                "total_flagged_windows": int(flags.sum()),
                "total_flagged_fraction": float(flags.mean()),
                "pre_attack_windows_t_lt_100": int(pre.sum()),
                "pre_attack_false_flags_t_lt_100": int((flags & pre).sum()),
                "pre_attack_false_positive_rate_t_lt_100": float((flags & pre).sum() / max(1, pre.sum())),
                "post_attack_windows_t_ge_100": int(post.sum()),
                "post_attack_flags_t_ge_100": int(post_flags.sum()),
                "post_attack_window_detection_rate_t_ge_100": float(post_flags.sum() / max(1, post.sum())),
                "event_detected_t_ge_100": bool(post_flags.any()),
                "first_post_attack_detection_s": first_post,
                "detection_delay_s_from_100": None if first_post is None else first_post - ONSET_S,
                "buffered_pre_windows_t_lt_90": int(pre90.sum()),
                "buffered_pre_false_flags_t_lt_90": int((flags & pre90).sum()),
                "buffered_post_windows_t_ge_110": int(post110.sum()),
                "buffered_post_flags_t_ge_110": int(post110_flags.sum()),
                "buffered_post_detection_rate_t_ge_110": float(post110_flags.sum() / max(1, post110.sum())),
                "buffered_event_detected_t_ge_110": bool(post110_flags.any()),
                "buffered_first_post_detection_s": first_post110,
                "buffered_detection_delay_s_from_100": None if first_post110 is None else first_post110 - ONSET_S,
            }
    # Re-write score CSV after adding threshold-source columns.
    scores.to_csv(scores_path, index=False)
    summary_path = out / f"texbat_{scenario}_9tap_onset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    make_plot(scores, thresholds, out / f"texbat_{scenario}_9tap_score_vs_time.png", scenario)
    return summary


def make_plot(scores: pd.DataFrame, thresholds: dict, path: Path, scenario: str):
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(scores.window_start_s, scores.joint_score, lw=1.2, label="joint_score")
    for name, th in thresholds.items():
        axes[0].axhline(th, ls="--", lw=.9, label=name)
    axes[0].axvline(ONSET_S, color="red", lw=1.0, label="~100s onset")
    axes[0].set_ylabel("joint")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    for ax, col in zip(axes[1:], ["node_rmse", "graph_rmse", "relation_gate"]):
        ax.plot(scores.window_start_s, scores[col], lw=1.0)
        ax.axvline(ONSET_S, color="red", lw=1.0)
        ax.set_ylabel(col)
    axes[-1].set_xlabel("TEXBAT time (s)")
    fig.suptitle(f"{scenario} 9-tap Method-A normal-model score")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIOS), default=["ds1","ds2","ds4"])
    ap.add_argument("--model-dir", default="artifacts/texbat_clean_normal_9tap_features/combined/conditional_integrated_gru_9tap_w1.0_s0.5")
    ap.add_argument("--out-root", default="artifacts/texbat_9tap_external_validation")
    ap.add_argument("--exe", default="/home/ubuntu/projects/gnss-doppler-lab/.tools/gnss-sdr-method-a-9tap")
    ap.add_argument("--force-receiver", action="store_true")
    ap.add_argument("--force-features", action="store_true")
    ap.add_argument("--feature-mode", choices=["all", "normalized_dmcpd"], default="all")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--skip-score", action="store_true", help="Stop after receiver/features/dataset export; useful for non-neural q70 morphology evaluation.")
    args = ap.parse_args()
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for scenario in args.scenarios:
        raw = Path(f"data/external/texbat/raw/{scenario}.bin")
        if not raw.exists():
            raise SystemExit(f"missing raw TEXBAT file: {raw}")
        out = out_root / scenario
        out.mkdir(parents=True, exist_ok=True)
        manifest = run_receiver(scenario, raw, out, exe=args.exe, force=args.force_receiver, samples=args.samples)
        node_csv, graph_csv, feature_csv = build_features(scenario, out, manifest, force=args.force_features, feature_mode=args.feature_mode)
        if args.skip_score:
            summaries[scenario] = {"node_csv": str(node_csv), "graph_csv": str(graph_csv), "feature_csv": str(feature_csv), "score_skipped": True}
        else:
            summaries[scenario] = score(node_csv, graph_csv, model_dir=Path(args.model_dir), out=out, scenario=scenario)
        print(json.dumps({scenario: summaries[scenario]}, indent=2, sort_keys=True), flush=True)
    combined = {"schema":"gnss-doppler-lab.texbat-9tap-external-validation-combined", "model_dir":args.model_dir, "feature_mode": args.feature_mode, "scenarios": summaries}
    (out_root / "combined_9tap_onset_summary.json").write_text(json.dumps(combined, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(combined, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
