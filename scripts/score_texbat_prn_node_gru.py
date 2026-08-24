#!/usr/bin/env python3
"""Score TEXBAT node-window CSVs with a PRN-local GRU model.

This intentionally does not read graph CSVs or aggregate relation features. The
event-level score is derived from independent per-PRN anomaly scores at each time
bin (max/mean/top3), so PRN relation can be evaluated later as a separate filter.
"""
from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gnss_doppler_lab import receiver_quality_contract as quality_contract

spec = importlib.util.spec_from_file_location("train_prn_node_gru", str(ROOT / "scripts/train_prn_node_gru.py"))
train_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train_mod
spec.loader.exec_module(train_mod)

ONSET_S = 100.0
TIMING_CONTRACT = {
    "score_time_field": "window_start_s",
    "window_duration_s": 1.0,
    "window_availability_offset_s": 1.0,
    "interpretation": "scores timestamped at frozen window start become available one full window later",
}


def load_checkpoint(path: Path) -> dict[str, object]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"unable to safely load checkpoint {path}: {exc}") from exc
    required = {"config", "node_feature_columns", "standardizer", "model_state_dict"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(f"checkpoint missing required structure: {sorted(required)}")
    features = checkpoint["node_feature_columns"]
    standardizer = checkpoint["standardizer"]
    if (not isinstance(checkpoint["config"], dict) or not isinstance(features, (list, tuple))
            or not features or not all(isinstance(x, str) and x for x in features)
            or not isinstance(standardizer, dict)
            or not {"node_mean", "node_std"}.issubset(standardizer)
            or not isinstance(checkpoint["model_state_dict"], dict)):
        raise ValueError("checkpoint contains invalid config, feature, standardizer, or state structure")
    try:
        mean = np.asarray(standardizer["node_mean"], dtype=np.float32)
        stdev = np.asarray(standardizer["node_std"], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint standardizer must be numeric") from exc
    if (mean.shape != (len(features),) or stdev.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(stdev).all() or np.any(stdev <= 0)):
        raise ValueError("checkpoint standardizer dimensions and finite positive scales must match features")
    return checkpoint


def validate_node_inputs(frame: pd.DataFrame, feature_columns: list[str]) -> None:
    required = [
        "run_id", "prn", "window_bin_s", "window_start_s", "window_end_s",
        "window_mid_s", *feature_columns,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"node CSV missing required columns: {missing[:5]}")
    try:
        numeric = frame[
            ["window_bin_s", "window_start_s", "window_end_s", "window_mid_s", *feature_columns]
        ].apply(pd.to_numeric, errors="raise").to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("node CSV timing and feature inputs must be numeric") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("node CSV timing and feature inputs must be finite")
    quality_contract.validate_quality_node_frame(frame, feature_columns)


def aggregate_event_scores(prn_scores: pd.DataFrame) -> pd.DataFrame:
    keys = ["run_id", "window_bin_s"]
    event = prn_scores.groupby(keys, as_index=False, sort=True).agg(
        window_start_s=("window_start_s", "min"),
        window_end_s=("window_end_s", "max"),
        tracked_prn_count=("prn", "nunique"),
        prn_node_rmse_max=("prn_node_rmse", "max"),
        prn_node_rmse_mean=("prn_node_rmse", "mean"),
        prn_node_rmse_median=("prn_node_rmse", "median"),
    )
    top3 = (prn_scores.groupby(keys, sort=True)["prn_node_rmse"]
            .apply(lambda values: topk_mean(values, 3)).rename("prn_node_rmse_top3_mean").reset_index())
    return event.merge(top3, on=keys, how="left", validate="one_to_one")


def topk_mean(values: pd.Series, k: int = 3) -> float:
    arr = np.sort(values.to_numpy(float))[::-1]
    if arr.size == 0:
        return 0.0
    return float(arr[:min(k, arr.size)].mean())


def score_output_paths(out_dir: Path, scenario: str, output_prefix: str = "texbat") -> tuple[Path, Path, Path]:
    return (out_dir / f"{output_prefix}_{scenario}_prn_local_scores.csv",
            out_dir / f"{output_prefix}_{scenario}_prn_local_event_scores.csv",
            out_dir / f"{output_prefix}_{scenario}_prn_local_onset_summary.json")


def checkpoint_provenance(checkpoint: Path, feature_columns: list[str]) -> dict[str, object]:
    return {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "node_feature_columns": list(feature_columns)}


def threshold_flag_metrics(event: pd.DataFrame, flags: pd.Series, onset_s: float | None,
                           *, legacy_aliases: bool = False) -> dict[str, object]:
    """Summarize threshold flags without embedding a particular onset in key names."""
    result: dict[str, object] = {
        "threshold_exceedance_flags": int(flags.sum()),
        "threshold_exceedance_rate": float(flags.mean()),
    }
    if onset_s is None:
        return {
            "windows": int(len(flags)),
            "false_positive_flags": int(flags.sum()),
            "false_positive_exceedance_rate": float(flags.mean()),
            "any_false_positive": bool(flags.any()),
        }
    times = event["window_start_s"]
    pre = times < onset_s
    post = times >= onset_s
    buffered_pre = times < onset_s - 10.0
    buffered_post = times >= onset_s + 10.0
    post_flags = flags & post
    buffered_post_flags = flags & buffered_post
    first = float(times[post_flags].min()) if post_flags.any() else None
    first_buffered = float(times[buffered_post_flags].min()) if buffered_post_flags.any() else None
    result.update({
        "evaluation": ({"kind": "clean_negative_control"} if onset_s is None else {"kind": "onset", "onset_s": float(onset_s)}),
        "pre_attack_windows": int(pre.sum()),
        "pre_attack_false_flags": int((flags & pre).sum()),
        "pre_attack_false_positive_rate": float((flags & pre).sum() / max(1, pre.sum())),
        "post_attack_windows": int(post.sum()),
        "post_attack_flags": int(post_flags.sum()),
        "post_attack_window_detection_rate": float(post_flags.sum() / max(1, post.sum())),
        "event_detected": bool(post_flags.any()),
        "first_post_attack_detection_s": first,
        "detection_delay_s": None if first is None else first - onset_s,
        "onset_buffer_s": 10.0,
        "buffered_pre_windows": int(buffered_pre.sum()),
        "buffered_pre_false_flags": int((flags & buffered_pre).sum()),
        "buffered_post_windows": int(buffered_post.sum()),
        "buffered_post_flags": int(buffered_post_flags.sum()),
        "buffered_post_detection_rate": float(buffered_post_flags.sum() / max(1, buffered_post.sum())),
        "buffered_event_detected": bool(buffered_post_flags.any()),
        "buffered_first_post_detection_s": first_buffered,
        "buffered_detection_delay_s": None if first_buffered is None else first_buffered - onset_s,
    })
    if legacy_aliases and onset_s == 100.0:
        result.update({
            "total_flagged_windows": result["threshold_exceedance_flags"],
            "total_flagged_fraction": result["threshold_exceedance_rate"],
            "pre_attack_windows_t_lt_100": result["pre_attack_windows"],
            "pre_attack_false_flags_t_lt_100": result["pre_attack_false_flags"],
            "pre_attack_false_positive_rate_t_lt_100": result["pre_attack_false_positive_rate"],
            "post_attack_windows_t_ge_100": result["post_attack_windows"],
            "post_attack_flags_t_ge_100": result["post_attack_flags"],
            "post_attack_window_detection_rate_t_ge_100": result["post_attack_window_detection_rate"],
            "event_detected_t_ge_100": result["event_detected"],
            "detection_delay_s_from_100": result["detection_delay_s"],
            "buffered_pre_windows_t_lt_90": result["buffered_pre_windows"],
            "buffered_pre_false_flags_t_lt_90": result["buffered_pre_false_flags"],
            "buffered_post_windows_t_ge_110": result["buffered_post_windows"],
            "buffered_post_flags_t_ge_110": result["buffered_post_flags"],
            "buffered_post_detection_rate_t_ge_110": result["buffered_post_detection_rate"],
            "buffered_event_detected_t_ge_110": result["buffered_event_detected"],
            "buffered_detection_delay_s_from_100": result["buffered_detection_delay_s"],
        })
    return result


def load_validation_prn_scores(path: Path) -> pd.DataFrame:
    """Load finite, nonempty PRN validation RMSE values for calibration."""
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"validation prn_node_rmse CSV is unreadable: {path}: {exc}") from exc
    if "prn_node_rmse" not in frame.columns:
        raise ValueError(f"validation CSV missing required prn_node_rmse column: {path}")
    if frame.empty:
        raise ValueError(f"validation prn_node_rmse values are empty: {path}")
    try:
        values = pd.to_numeric(frame["prn_node_rmse"], errors="raise").to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"validation prn_node_rmse values must be numeric and finite: {path}") from exc
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"validation prn_node_rmse values must be nonempty and finite: {path}")
    frame = frame.copy()
    frame["prn_node_rmse"] = values
    return frame


def score_node_csv(node_csv: Path, model_dir: Path, out_dir: Path, scenario: str,
                   onset_s: float | None = ONSET_S, output_prefix: str = "texbat",
                   dataset_prefix: str = "TEXBAT", stride_s: float = 0.5) -> dict:
    ckpt_path = model_dir / "prn_local_gru_predictor.pt"
    ckpt = load_checkpoint(ckpt_path)
    cfg = train_mod.TrainConfig(**ckpt["config"])
    trained_stride = ckpt.get("expected_stride_s")
    if trained_stride is not None and not np.isclose(
        float(trained_stride), stride_s, rtol=0.0, atol=1e-12
    ):
        raise ValueError("requested stride_s does not match checkpoint expected_stride_s")
    feature_cols = ckpt["node_feature_columns"]
    std = ckpt["standardizer"]
    mean = np.asarray(std["node_mean"], dtype=np.float32)
    stdev = np.asarray(std["node_std"], dtype=np.float32)
    df = pd.read_csv(node_csv)
    validate_node_inputs(df, feature_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_mod.PrnLocalGRU(len(feature_cols), cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    rows = []
    blocks = quality_contract.segment_safe_blocks(
        df, feature_cols, expected_stride_s=stride_s
    )
    with torch.no_grad():
        for block in blocks:
            g = block.frame
            x = train_mod.standardize(
                g[feature_cols].to_numpy(np.float32), mean, stdev
            )
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
                    target_position = meta_idx[offset+j]
                    src = g.iloc[target_position]
                    row = {
                        "run_id": block.run_id,
                        "prn": block.prn,
                        "window_bin_s": float(src["window_bin_s"]),
                        "window_start_s": float(src["window_start_s"]),
                        "window_end_s": float(src["window_end_s"]),
                        "window_mid_s": float(src["window_mid_s"]),
                    }
                    row.update(quality_contract.score_quality_metadata(
                        block, target_position, cfg.seq_len, expected_stride_s=stride_s
                    ))
                    row.update({
                        "prn_node_rmse": float(r),
                        "prn_node_mae": float(a),
                    })
                    rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    prn_scores = pd.DataFrame(rows)
    if prn_scores.empty:
        raise RuntimeError(f"no PRN-local scores produced for {scenario}")
    prn_scores_path, event_scores_path, summary_path = score_output_paths(out_dir, scenario, output_prefix)
    prn_scores.to_csv(prn_scores_path, index=False)

    event = aggregate_event_scores(prn_scores)

    normal_scores = load_validation_prn_scores(model_dir / "validation_prn_node_scores.csv")
    qmap = {"pfa10_q90": .90, "pfa5_q95": .95, "pfa1_q99": .99, "pfa0_5_q995": .995, "pfa0_1_q999": .999}
    # Per-PRN validation thresholds applied to event max/top3 scores. This is intentionally high-recall biased.
    thresholds = {name: float(normal_scores["prn_node_rmse"].quantile(q)) for name, q in qmap.items()}

    for name, th in thresholds.items():
        event[f"above_max_{name}"] = event["prn_node_rmse_max"] > th
        event[f"above_top3_{name}"] = event["prn_node_rmse_top3_mean"] > th
    event.to_csv(event_scores_path, index=False)

    summary = {
        "schema": ("gnss-doppler-lab.texbat-prn-local-node-only-summary" if dataset_prefix == "TEXBAT" else "gnss-doppler-lab.prn-local-node-only-summary"),
        "dataset": dataset_prefix,
        "scenario": scenario,
        "model": str(ckpt_path),
        "node_csv": str(node_csv),
        "prn_score_csv": str(prn_scores_path),
        "event_score_csv": str(event_scores_path),
        "evaluation": ({"kind": "clean_negative_control"} if onset_s is None else {"kind": "onset", "onset_s": float(onset_s)}),
        "checkpoint_provenance": checkpoint_provenance(ckpt_path, feature_cols),
        "timing_contract": TIMING_CONTRACT,
        "receiver_quality_score_contract": quality_contract.score_contract_document(
            expected_stride_s=stride_s, history_length=cfg.seq_len
        ),
        "rows": {"prn_scores": int(len(prn_scores)), "event_windows": int(len(event))},
        "score_summary": event[["prn_node_rmse_max", "prn_node_rmse_top3_mean", "prn_node_rmse_mean"]].describe(percentiles=[.5,.9,.95,.99]).to_dict(),
        "normal_prn_thresholds": thresholds,
    }
    if dataset_prefix == "TEXBAT" and onset_s == ONSET_S:
        summary["onset_s"] = ONSET_S
    for agg in ["max", "top3"]:
        for name, th in thresholds.items():
            flags = event[f"above_{agg}_{name}"]
            metrics = threshold_flag_metrics(
                event, flags, onset_s,
                legacy_aliases=(dataset_prefix == "TEXBAT" and onset_s == ONSET_S),
            )
            metrics["threshold"] = th
            summary[("onset_metrics" if onset_s is not None else "clean_metrics") + f"_{agg}_{name}"] = metrics

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_plot(event, thresholds, out_dir / f"{output_prefix}_{scenario}_prn_local_score_vs_time.png", scenario, onset_s, dataset_prefix)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def make_plot(event: pd.DataFrame, thresholds: dict[str, float], path: Path, scenario: str, onset_s: float | None = ONSET_S, dataset_prefix: str = "TEXBAT") -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(event.window_start_s, event.prn_node_rmse_max, lw=1.2, label="max per-PRN RMSE")
    for name, th in thresholds.items():
        axes[0].axhline(th, ls="--", lw=.8, label=name)
    if onset_s is not None:
        axes[0].axvline(onset_s, color="red", lw=1.0)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[0].set_ylabel("max")
    axes[1].plot(event.window_start_s, event.prn_node_rmse_top3_mean, lw=1.0)
    if onset_s is not None:
        axes[1].axvline(onset_s, color="red", lw=1.0)
    axes[1].set_ylabel("top3 mean")
    axes[2].plot(event.window_start_s, event.tracked_prn_count, lw=1.0)
    if onset_s is not None:
        axes[2].axvline(onset_s, color="red", lw=1.0)
    axes[2].set_ylabel("PRN count")
    axes[2].set_xlabel(f"{dataset_prefix} time (s)")
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
    ap.add_argument("--onset-s", type=float, default=ONSET_S)
    ap.add_argument("--output-prefix", default="texbat")
    ap.add_argument("--dataset-prefix", default="TEXBAT")
    ap.add_argument("--stride-s", type=float, default=0.5)
    ap.add_argument("--clean-only", action="store_true", help="Score a clean negative control without onset metrics.")
    args = ap.parse_args()
    score_node_csv(Path(args.node_csv), Path(args.model_dir), Path(args.out_dir), args.scenario, None if args.clean_only else args.onset_s, args.output_prefix, args.dataset_prefix, args.stride_s)

if __name__ == "__main__":
    main()
