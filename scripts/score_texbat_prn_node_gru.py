#!/usr/bin/env python3
"""Score TEXBAT node-window CSVs with a PRN-local GRU model.

This intentionally does not read graph CSVs or aggregate relation features. The
event-level score is derived from independent per-PRN anomaly scores at each time
bin (max/mean/top3), so PRN relation can be evaluated later as a separate filter.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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

spec = importlib.util.spec_from_file_location("train_prn_node_gru", str(Path.cwd() / "scripts/train_prn_node_gru.py"))
train_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train_mod
spec.loader.exec_module(train_mod)

ONSET_S = 100.0


def topk_mean(values: pd.Series, k: int = 3) -> float:
    arr = np.sort(values.to_numpy(float))[::-1]
    if arr.size == 0:
        return 0.0
    return float(arr[:min(k, arr.size)].mean())


def score_node_csv(node_csv: Path, model_dir: Path, out_dir: Path, scenario: str) -> dict:
    ckpt_path = model_dir / "prn_local_gru_predictor.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = train_mod.TrainConfig(**ckpt["config"])
    feature_cols = ckpt["node_feature_columns"]
    std = ckpt["standardizer"]
    mean = np.asarray(std["node_mean"], dtype=np.float32)
    stdev = np.asarray(std["node_std"], dtype=np.float32)
    df = pd.read_csv(node_csv)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"node CSV missing model feature columns: {missing[:5]}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_mod.PrnLocalGRU(len(feature_cols), cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    rows = []
    with torch.no_grad():
        for (run_id, prn), g in df.groupby(["run_id", "prn"], sort=True):
            g = g.sort_values("window_bin_s").reset_index(drop=True)
            x = train_mod.standardize(g[feature_cols].to_numpy(np.float32), mean, stdev)
            if len(x) < cfg.seq_len + 1:
                continue
            seqs = []
            targets = []
            meta_idx = []
            for start in range(len(x) - cfg.seq_len):
                seqs.append(x[start:start+cfg.seq_len])
                targets.append(x[start+cfg.seq_len])
                meta_idx.append(start+cfg.seq_len)
            for offset in range(0, len(seqs), 1024):
                seq = torch.from_numpy(np.stack(seqs[offset:offset+1024]).astype(np.float32)).to(device)
                target = torch.from_numpy(np.stack(targets[offset:offset+1024]).astype(np.float32)).to(device)
                pred = model(seq)
                rmse = torch.sqrt(((pred-target)**2).mean(dim=1)).cpu().numpy()
                mae = torch.mean(torch.abs(pred-target), dim=1).cpu().numpy()
                for j, (r, a) in enumerate(zip(rmse, mae)):
                    src = g.iloc[meta_idx[offset+j]]
                    rows.append({
                        "run_id": run_id,
                        "prn": prn,
                        "target_window_index": int(meta_idx[offset+j]),
                        "window_bin_s": float(src["window_bin_s"]),
                        "window_start_s": float(src["window_start_s"]),
                        "window_end_s": float(src["window_end_s"]),
                        "window_mid_s": float(src["window_mid_s"]),
                        "prn_node_rmse": float(r),
                        "prn_node_mae": float(a),
                    })
    out_dir.mkdir(parents=True, exist_ok=True)
    prn_scores = pd.DataFrame(rows)
    if prn_scores.empty:
        raise RuntimeError(f"no PRN-local scores produced for {scenario}")
    prn_scores_path = out_dir / f"texbat_{scenario}_prn_local_scores.csv"
    prn_scores.to_csv(prn_scores_path, index=False)

    event = prn_scores.groupby("window_bin_s", as_index=False).agg(
        window_start_s=("window_start_s", "min"),
        window_end_s=("window_end_s", "max"),
        tracked_prn_count=("prn", "nunique"),
        prn_node_rmse_max=("prn_node_rmse", "max"),
        prn_node_rmse_mean=("prn_node_rmse", "mean"),
        prn_node_rmse_median=("prn_node_rmse", "median"),
    )
    event["prn_node_rmse_top3_mean"] = prn_scores.groupby("window_bin_s")["prn_node_rmse"].apply(lambda s: topk_mean(s, 3)).to_numpy()

    normal_scores = pd.read_csv(model_dir / "validation_prn_node_scores.csv")
    qmap = {"pfa10_q90": .90, "pfa5_q95": .95, "pfa1_q99": .99, "pfa0_5_q995": .995, "pfa0_1_q999": .999}
    # Per-PRN validation thresholds applied to event max/top3 scores. This is intentionally high-recall biased.
    thresholds = {name: float(normal_scores["prn_node_rmse"].quantile(q)) for name, q in qmap.items()}

    for name, th in thresholds.items():
        event[f"above_max_{name}"] = event["prn_node_rmse_max"] > th
        event[f"above_top3_{name}"] = event["prn_node_rmse_top3_mean"] > th
    event_scores_path = out_dir / f"texbat_{scenario}_prn_local_event_scores.csv"
    event.to_csv(event_scores_path, index=False)

    summary = {
        "schema": "gnss-doppler-lab.texbat-prn-local-node-only-summary",
        "scenario": scenario,
        "model": str(ckpt_path),
        "node_csv": str(node_csv),
        "prn_score_csv": str(prn_scores_path),
        "event_score_csv": str(event_scores_path),
        "onset_s": ONSET_S,
        "rows": {"prn_scores": int(len(prn_scores)), "event_windows": int(len(event))},
        "score_summary": event[["prn_node_rmse_max", "prn_node_rmse_top3_mean", "prn_node_rmse_mean"]].describe(percentiles=[.5,.9,.95,.99]).to_dict(),
        "normal_prn_thresholds": thresholds,
    }
    for agg in ["max", "top3"]:
        for name, th in thresholds.items():
            col = f"above_{agg}_{name}"
            flags = event[col]
            pre = event["window_start_s"] < ONSET_S
            post = event["window_start_s"] >= ONSET_S
            pre90 = event["window_start_s"] < 90.0
            post110 = event["window_start_s"] >= 110.0
            post_flags = flags & post
            post110_flags = flags & post110
            first_post = float(event.loc[post_flags, "window_start_s"].min()) if post_flags.any() else None
            first_post110 = float(event.loc[post110_flags, "window_start_s"].min()) if post110_flags.any() else None
            summary[f"onset_metrics_{agg}_{name}"] = {
                "threshold": th,
                "total_flagged_windows": int(flags.sum()),
                "total_flagged_fraction": float(flags.mean()),
                "pre_attack_windows_t_lt_100": int(pre.sum()),
                "pre_attack_false_flags_t_lt_100": int((flags & pre).sum()),
                "pre_attack_false_positive_rate_t_lt_100": float((flags & pre).sum()/max(1, pre.sum())),
                "post_attack_windows_t_ge_100": int(post.sum()),
                "post_attack_flags_t_ge_100": int(post_flags.sum()),
                "post_attack_window_detection_rate_t_ge_100": float(post_flags.sum()/max(1, post.sum())),
                "event_detected_t_ge_100": bool(post_flags.any()),
                "first_post_attack_detection_s": first_post,
                "detection_delay_s_from_100": None if first_post is None else first_post - ONSET_S,
                "buffered_pre_windows_t_lt_90": int(pre90.sum()),
                "buffered_pre_false_flags_t_lt_90": int((flags & pre90).sum()),
                "buffered_post_windows_t_ge_110": int(post110.sum()),
                "buffered_post_flags_t_ge_110": int(post110_flags.sum()),
                "buffered_post_detection_rate_t_ge_110": float(post110_flags.sum()/max(1, post110.sum())),
                "buffered_event_detected_t_ge_110": bool(post110_flags.any()),
                "buffered_first_post_detection_s": first_post110,
                "buffered_detection_delay_s_from_100": None if first_post110 is None else first_post110 - ONSET_S,
            }

    summary_path = out_dir / f"texbat_{scenario}_prn_local_onset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_plot(event, thresholds, out_dir / f"texbat_{scenario}_prn_local_score_vs_time.png", scenario)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def make_plot(event: pd.DataFrame, thresholds: dict[str, float], path: Path, scenario: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(event.window_start_s, event.prn_node_rmse_max, lw=1.2, label="max per-PRN RMSE")
    for name, th in thresholds.items():
        axes[0].axhline(th, ls="--", lw=.8, label=name)
    axes[0].axvline(ONSET_S, color="red", lw=1.0)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[0].set_ylabel("max")
    axes[1].plot(event.window_start_s, event.prn_node_rmse_top3_mean, lw=1.0)
    axes[1].axvline(ONSET_S, color="red", lw=1.0)
    axes[1].set_ylabel("top3 mean")
    axes[2].plot(event.window_start_s, event.tracked_prn_count, lw=1.0)
    axes[2].axvline(ONSET_S, color="red", lw=1.0)
    axes[2].set_ylabel("PRN count")
    axes[2].set_xlabel("TEXBAT time (s)")
    fig.suptitle(f"{scenario} PRN-local node-only Doppler/tap anomaly score")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-csv", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scenario", default="ds4")
    args = ap.parse_args()
    score_node_csv(Path(args.node_csv), Path(args.model_dir), Path(args.out_dir), args.scenario)

if __name__ == "__main__":
    main()
