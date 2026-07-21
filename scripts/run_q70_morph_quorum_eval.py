#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_STATIC_PATH = ROOT / "artifacts/texbat_clean_normal_9tap_features/cleanStatic_normalized_dmcpd/tap9_tracking_features_w1.0_s0.5_dmcpd.csv"
CLEAN_COMBINED_PATH = ROOT / "artifacts/texbat_clean_normal_9tap_features/combined/tap9_tracking_features_w1.0_s0.5_combined.csv"
DEFAULT_FEATURE_ROOT = ROOT / "artifacts/texbat_9tap_external_validation_cleanStatic_normalized_dmcpd"
DEFAULT_OUT = ROOT / "artifacts/sci_q70_morph_quorum"
META_COLS = set(
    "run_id source_fingerprint label window_bin_s window_start_s window_end_s window_mid_s "
    "prn channel segment_index window_index epoch_count tap_count tap_layout sample_rate_hz".split()
)
QUANTILES = [("q90", 0.90), ("q95", 0.95), ("q99", 0.99), ("q99_5", 0.995), ("q99_9", 0.999)]
ONSET = 100.0
BUFFER = 10.0
LOW_NODE_Q = 0.65
AGG_Q = 0.70
SOFT_ALPHA = 2.0
OFFSET_BETA = 3.0
ROLL_WINDOW = 3
ROLL_MIN_PERIODS = 3
QUORUM_TAU = 0.50


def morphology_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in META_COLS
        and pd.api.types.is_numeric_dtype(df[c])
        and c.startswith("tap_")
        and "_rel_prompt_" in c
    ]


def robust_fit(df: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.Series]:
    x = df[cols].replace([np.inf, -np.inf], np.nan).astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    scale = 1.4826 * mad
    scale = scale.where(scale > 1e-9, x.std().replace(0, np.nan)).fillna(1.0)
    return med, scale


def node_scores(df: pd.DataFrame, cols: list[str], med: pd.Series, scale: pd.Series) -> pd.DataFrame:
    use_cols = [c for c in cols if c in df.columns]
    x = df[use_cols].replace([np.inf, -np.inf], np.nan).astype(float).fillna(med[use_cols])
    z = ((x - med[use_cols]) / scale[use_cols]).clip(-20, 20)
    out = df[["window_mid_s", "window_start_s", "window_end_s", "prn"]].copy()
    out["node_score"] = np.sqrt((z.to_numpy() ** 2).mean(axis=1))
    return out


def event_scores(ns: pd.DataFrame, low_thr: float) -> pd.DataFrame:
    ns = ns.copy()
    ns["event_bin_s"] = (ns["window_mid_s"] * 2).round() / 2.0
    ns["low_high"] = ns["node_score"] > low_thr
    g = ns.groupby("event_bin_s")
    ev = g["node_score"].agg(
        score_mean="mean",
        score_q70=lambda s: s.quantile(AGG_Q),
        score_max="max",
        prn_count="size",
    ).reset_index().rename(columns={"event_bin_s": "window_mid_s"})
    ev["low_high_fraction"] = g["low_high"].mean().to_numpy(dtype=float)
    ev["low_high_fraction_roll3"] = ev["low_high_fraction"].rolling(
        ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS
    ).mean().fillna(ev["low_high_fraction"])
    for base in ["score_mean", "score_q70"]:
        ev[f"{base}_tau50_gate"] = np.where(
            ev["low_high_fraction_roll3"] >= QUORUM_TAU,
            ev[base] * (1.0 + SOFT_ALPHA * ev["low_high_fraction_roll3"])
            + OFFSET_BETA * ev["low_high_fraction_roll3"],
            ev[base],
        )
    return ev.sort_values("window_mid_s").reset_index(drop=True)


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


def eval_scenario(ev: pd.DataFrame, calibration_ev: pd.DataFrame, score_col: str) -> dict[str, object]:
    pre = ev.window_mid_s < (ONSET - BUFFER)
    post = ev.window_mid_s >= (ONSET + BUFFER)
    mask = np.logical_or(pre.to_numpy(), post.to_numpy())
    out: dict[str, object] = {
        "score_col": score_col,
        "calibration": "cleanStatic_plus_cleanDynamic_event_cutoffs",
        "pre_windows_t_lt_90": int(pre.sum()),
        "post_windows_t_ge_110": int(post.sum()),
        "auc_pre_vs_post_buffered": auc_rank(post.to_numpy()[mask].astype(int), ev.loc[mask, score_col].to_numpy()),
        "pre_score_median": float(ev.loc[pre, score_col].median()),
        "post_score_median": float(ev.loc[post, score_col].median()),
        "pre_score_q95": float(ev.loc[pre, score_col].quantile(0.95)),
        "post_score_q95": float(ev.loc[post, score_col].quantile(0.95)),
    }
    for name, q in QUANTILES:
        threshold = float(calibration_ev[score_col].quantile(q))
        flags = ev[score_col] > threshold
        pre_flags = np.logical_and(flags.to_numpy(), pre.to_numpy())
        post_flags_bool = np.logical_and(flags.to_numpy(), post.to_numpy())
        post_flags = pd.Series(post_flags_bool, index=ev.index)
        first = ev.loc[post_flags, "window_mid_s"].min() if post_flags.any() else np.nan
        out[f"{name}_threshold"] = threshold
        out[f"{name}_pre_fp_rate"] = float(pre_flags.sum() / max(1, int(pre.sum())))
        out[f"{name}_post_det_rate"] = float(post_flags_bool.sum() / max(1, int(post.sum())))
        out[f"{name}_first_delay_s"] = None if np.isnan(first) else float(first - ONSET)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["ds4"], help="TEXBAT scenario ids, e.g. ds1 ds2 ds4 ds6")
    ap.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    feature_root = Path(args.feature_root)
    if not feature_root.is_absolute():
        feature_root = ROOT / feature_root
    clean_static = pd.read_csv(CLEAN_STATIC_PATH)
    clean_combined = pd.read_csv(CLEAN_COMBINED_PATH)
    cols = morphology_cols(clean_static)
    if not cols:
        raise RuntimeError("no prompt-relative tap morphology columns found")
    med, scale = robust_fit(clean_static, cols)
    clean_static_ns = node_scores(clean_static, cols, med, scale)
    clean_combined_ns = node_scores(clean_combined, cols, med, scale)
    low_thr = float(clean_static_ns["node_score"].quantile(LOW_NODE_Q))
    calibration_ev = event_scores(clean_combined_ns, low_thr)

    all_metrics = []
    summaries = {}
    for scenario in args.scenarios:
        src = feature_root / scenario / "tap9_tracking_features_w1.0_s0.5_dmcpd.csv"
        if not src.exists():
            raise FileNotFoundError(f"missing feature csv for {scenario}: {src}")
        df = pd.read_csv(src)
        ns = node_scores(df, cols, med, scale)
        ev = event_scores(ns, low_thr)
        scenario_out = out_root / scenario
        scenario_out.mkdir(parents=True, exist_ok=True)
        ev_path = scenario_out / f"{scenario}_q70_event_scores.csv"
        ev[["window_mid_s", "score_mean", "score_q70", "score_mean_tau50_gate", "score_q70_tau50_gate", "prn_count", "low_high_fraction", "low_high_fraction_roll3"]].to_csv(ev_path, index=False)
        metrics = [eval_scenario(ev, calibration_ev, c) | {"scenario": scenario} for c in ["score_mean_tau50_gate", "score_q70_tau50_gate"]]
        pd.DataFrame(metrics).to_csv(scenario_out / f"{scenario}_q70_morph_quorum_metrics.csv", index=False)
        all_metrics.extend(metrics)
        summaries[scenario] = {"events_csv": str(ev_path.relative_to(ROOT)), "metrics": metrics}

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_root / "q70_morph_quorum_metrics.csv", index=False)
    summary = {
        "schema": "gnss-doppler-lab.q70-morphology-quorum-eval.v1",
        "scenarios": args.scenarios,
        "feature_root": str(feature_root),
        "feature_count": len(cols),
        "features": cols,
        "low_node_quantile": LOW_NODE_Q,
        "low_node_threshold": low_thr,
        "aggregation_quantile": AGG_Q,
        "soft_alpha": SOFT_ALPHA,
        "offset_beta": OFFSET_BETA,
        "roll_window": ROLL_WINDOW,
        "roll_min_periods": ROLL_MIN_PERIODS,
        "quorum_tau": QUORUM_TAU,
        "metrics_csv": str((out_root / "q70_morph_quorum_metrics.csv").relative_to(ROOT)),
        "scenario_summaries": summaries,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
