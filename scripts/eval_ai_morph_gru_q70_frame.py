#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ONSET = 100.0
BUFFER = 10.0
LOW_NODE_Q = 0.65
AGG_Q = 0.70
ROLL_WINDOW = 3
ROLL_MIN_PERIODS = 3
SCORE_PERSISTENCE_WINDOW = 1
QUORUM_TAU = 0.50
SOFT_ALPHA = 2.0
OFFSET_BETA = 3.0
QUANTILES = [("q90", .90), ("q95", .95), ("q99", .99), ("q99_5", .995), ("q99_9", .999)]


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, int)
    scores = np.asarray(scores, float)
    n1 = int((labels == 1).sum())
    n0 = int((labels == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    r1 = ranks[labels == 1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n0))


def event_scores(prn_scores: pd.DataFrame, low_thr: float, *, aggregation_quantile: float, roll_window: int) -> pd.DataFrame:
    df = prn_scores.copy()
    df["event_bin_s"] = (df["window_mid_s"].astype(float) * 2).round() / 2.0
    df["low_high"] = df["prn_node_rmse"] > low_thr
    g = df.groupby("event_bin_s", sort=True)
    ev = g["prn_node_rmse"].agg(
        ai_rmse_mean="mean",
        ai_rmse_q=lambda s: s.quantile(aggregation_quantile),
        ai_rmse_top3_mean=lambda s: np.sort(s.to_numpy(float))[::-1][: min(3, len(s))].mean(),
        ai_rmse_max="max",
        tracked_prn_count="size",
    ).reset_index().rename(columns={"event_bin_s": "window_mid_s"})
    ev["low_high_fraction"] = g["low_high"].mean().to_numpy(float)
    roll_min_periods = min(roll_window, ROLL_MIN_PERIODS)
    roll_col = f"low_high_fraction_roll{roll_window}"
    ev[roll_col] = ev["low_high_fraction"].rolling(roll_window, min_periods=roll_min_periods).mean().fillna(ev["low_high_fraction"])
    for base in ["ai_rmse_mean", "ai_rmse_q", "ai_rmse_top3_mean"]:
        ev[f"{base}_tau50_gate"] = np.where(
            ev[roll_col] >= QUORUM_TAU,
            ev[base] * (1.0 + SOFT_ALPHA * ev[roll_col]) + OFFSET_BETA * ev[roll_col],
            ev[base],
        )
    return ev


def add_score_persistence(ev: pd.DataFrame, score_cols: list[str], window: int) -> list[str]:
    """Add causal rolling-max event scores over the current tracked PRN context.

    This is not a PRN-ID threshold: calibration is applied to the same persisted
    clean-reference event score. A short causal persistence window makes isolated
    PRN-local surprises less brittle while preserving onset-aware first-delay
    semantics.
    """
    if window <= 1:
        return score_cols
    out_cols: list[str] = []
    for col in score_cols:
        pcol = f"{col}_persist{window}"
        ev[pcol] = ev[col].rolling(window, min_periods=1).max()
        out_cols.append(pcol)
    return score_cols + out_cols


def eval_scenario(ev: pd.DataFrame, cal: pd.DataFrame, score_col: str) -> dict[str, object]:
    pre = ev.window_mid_s < (ONSET - BUFFER)
    post = ev.window_mid_s >= (ONSET + BUFFER)
    mask = pre | post
    out: dict[str, object] = {
        "score_col": score_col,
        "calibration": "cleanStatic_plus_cleanDynamic_event_cutoffs",
        "pre_windows_t_lt_90": int(pre.sum()),
        "post_windows_t_ge_110": int(post.sum()),
        "auc_pre_vs_post_buffered": auc_rank(post.to_numpy()[mask.to_numpy()].astype(int), ev.loc[mask, score_col].to_numpy()),
        "pre_score_median": float(ev.loc[pre, score_col].median()),
        "post_score_median": float(ev.loc[post, score_col].median()),
        "pre_score_q95": float(ev.loc[pre, score_col].quantile(.95)),
        "post_score_q95": float(ev.loc[post, score_col].quantile(.95)),
    }
    for name, q in QUANTILES:
        th = float(cal[score_col].quantile(q))
        flags = ev[score_col] > th
        first = ev.loc[flags & post, "window_mid_s"].min() if (flags & post).any() else np.nan
        out[f"{name}_threshold"] = th
        out[f"{name}_pre_fp_rate"] = float((flags & pre).sum() / max(1, int(pre.sum())))
        out[f"{name}_post_det_rate"] = float((flags & post).sum() / max(1, int(post.sum())))
        out[f"{name}_first_delay_s"] = None if np.isnan(first) else float(first - ONSET)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-root", default="artifacts/ai_morph_gru_cleanStatic_q70_frame/scored")
    ap.add_argument("--out-dir", default="artifacts/ai_morph_gru_cleanStatic_q70_frame/q70_event_calibration")
    ap.add_argument("--scenario", default="ds3",
                    help="Scenario to evaluate, comma-separated scenarios, or all (all scored ds* directories).")
    ap.add_argument("--low-node-quantile", type=float, default=LOW_NODE_Q,
                    help="CleanStatic PRN-node RMSE quantile used to form the morphology quorum fraction.")
    ap.add_argument("--aggregation-quantile", type=float, default=AGG_Q,
                    help="Per-event quantile over the currently tracked PRN set; 0.70 keeps the q70 baseline framing.")
    ap.add_argument("--roll-window", type=int, default=ROLL_WINDOW,
                    help="Number of adjacent 0.5 s event bins used to stabilize the quorum fraction.")
    ap.add_argument("--score-persistence-window", type=int, default=SCORE_PERSISTENCE_WINDOW,
                    help="Optional causal rolling-max window over event scores before clean-reference quantile calibration; 1 disables it.")
    args = ap.parse_args()
    score_root = ROOT / args.score_root
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_static_prn = pd.read_csv(score_root / "cleanStatic" / "texbat_cleanStatic_prn_local_scores.csv")
    clean_dynamic_prn = pd.read_csv(score_root / "cleanDynamic" / "texbat_cleanDynamic_prn_local_scores.csv")
    low_thr = float(clean_static_prn["prn_node_rmse"].quantile(args.low_node_quantile))
    clean_static_ev = event_scores(clean_static_prn, low_thr, aggregation_quantile=args.aggregation_quantile, roll_window=args.roll_window)
    clean_dynamic_ev = event_scores(clean_dynamic_prn, low_thr, aggregation_quantile=args.aggregation_quantile, roll_window=args.roll_window)
    cal = pd.concat([clean_static_ev, clean_dynamic_ev], ignore_index=True)
    base_score_cols = ["ai_rmse_mean_tau50_gate", "ai_rmse_q_tau50_gate", "ai_rmse_top3_mean_tau50_gate"]
    score_cols = add_score_persistence(cal, base_score_cols, args.score_persistence_window)

    if args.scenario == "all":
        scenarios = sorted(p.name for p in score_root.iterdir() if p.is_dir() and p.name.startswith("ds"))
    else:
        scenarios = [s.strip() for s in args.scenario.split(",") if s.strip()]
    if not scenarios:
        raise RuntimeError("no scenarios requested")

    all_metrics: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for scenario in scenarios:
        scenario_prn = pd.read_csv(score_root / scenario / f"texbat_{scenario}_prn_local_scores.csv")
        scenario_ev = event_scores(scenario_prn, low_thr, aggregation_quantile=args.aggregation_quantile, roll_window=args.roll_window)
        add_score_persistence(scenario_ev, base_score_cols, args.score_persistence_window)
        metrics = [eval_scenario(scenario_ev, cal, c) | {"scenario": scenario} for c in score_cols]
        scenario_ev.to_csv(out_dir / f"{scenario}_ai_morph_gru_q70_event_scores.csv", index=False)
        pd.DataFrame(metrics).to_csv(out_dir / f"{scenario}_ai_morph_gru_q70_metrics.csv", index=False)
        all_metrics.extend(metrics)
        summary = {
            "schema": "gnss-doppler-lab.ai-morph-gru-q70-event-calibration.v1",
            "purpose": "AI normal-only PRN-local morphology GRU scored with q70 morphology-quorum framing; no PRN ID thresholds; cleanStatic+cleanDynamic event quantile calibration.",
            "scenario": scenario,
            "low_node_quantile": args.low_node_quantile,
            "low_node_threshold_cleanStatic_prn_rmse": low_thr,
            "aggregation_quantile": args.aggregation_quantile,
            "quorum_tau": QUORUM_TAU,
            "roll_window": args.roll_window,
            "score_persistence_window": args.score_persistence_window,
            "calibration_event_windows": int(len(cal)),
            "metrics": metrics,
            "events_csv": str((out_dir / f"{scenario}_ai_morph_gru_q70_event_scores.csv").relative_to(ROOT)),
            "metrics_csv": str((out_dir / f"{scenario}_ai_morph_gru_q70_metrics.csv").relative_to(ROOT)),
        }
        summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        (out_dir / f"summary_{scenario}.json").write_text(summary_text)
        summaries.append(summary)

    pd.DataFrame(all_metrics).to_csv(out_dir / "all_scenarios_ai_morph_gru_q70_metrics.csv", index=False)
    combined = {
        "schema": "gnss-doppler-lab.ai-morph-gru-q70-event-calibration.multi-scenario.v1",
        "scenarios": scenarios,
        "scenario_count": len(scenarios),
        "combined_metrics_csv": str((out_dir / "all_scenarios_ai_morph_gru_q70_metrics.csv").relative_to(ROOT)),
        "summaries": summaries,
    }
    combined_text = json.dumps(combined, indent=2, sort_keys=True) + "\n"
    (out_dir / "summary.json").write_text(combined_text)
    print(combined_text, end="")


if __name__ == "__main__":
    main()
