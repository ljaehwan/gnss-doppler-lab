from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ubuntu/projects/gnss-doppler-lab")
OUT = ROOT / "artifacts" / "sci_ds4_temporal_persistence"
OUT.mkdir(parents=True, exist_ok=True)
BASE = ROOT / "artifacts/texbat_9tap_external_validation_cleanStatic_normalized_dmcpd"
PATHS = {
    "cleanStatic": ROOT / "artifacts/texbat_clean_normal_9tap_features/cleanStatic_normalized_dmcpd/tap9_tracking_features_w1.0_s0.5_dmcpd.csv",
    "ds4": BASE / "ds4/tap9_tracking_features_w1.0_s0.5_dmcpd.csv",
}
META_COLS = {
    "run_id", "source_fingerprint", "label", "window_bin_s", "window_start_s", "window_end_s",
    "window_mid_s", "prn", "channel", "segment_index", "window_index", "epoch_count",
    "tap_count", "tap_layout", "sample_rate_hz",
}
QUANTILES = [("q90", 0.90), ("q95", 0.95), ("q99", 0.99), ("q99_5", 0.995), ("q99_9", 0.999)]
ONSET = 100.0
BUFFER = 10.0


def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def robust_fit(df: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.Series]:
    x = df[cols].replace([np.inf, -np.inf], np.nan).astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    scale = 1.4826 * mad
    std = x.std().replace(0, np.nan)
    scale = scale.where(scale > 1e-9, std).fillna(1.0)
    return med, scale


def node_scores(df: pd.DataFrame, cols: list[str], med: pd.Series, scale: pd.Series) -> pd.DataFrame:
    use = [c for c in cols if c in df.columns]
    x = df[use].replace([np.inf, -np.inf], np.nan).astype(float).fillna(med[use])
    z = ((x - med[use]) / scale[use]).clip(-20, 20)
    out = df[["window_mid_s", "window_start_s", "window_end_s", "prn"]].copy()
    out["node_score"] = np.sqrt((z.to_numpy() ** 2).mean(axis=1))
    return out


def event_scores(ns: pd.DataFrame) -> pd.DataFrame:
    ns = ns.copy()
    ns["event_bin_s"] = (ns["window_mid_s"] * 2).round() / 2.0
    grouped = ns.groupby("event_bin_s")["node_score"]
    ev = pd.DataFrame(
        {
            "window_mid_s": grouped.mean().index,
            "score_mean": grouped.mean().values,
            "score_max": grouped.max().values,
            "prn_count": grouped.size().values,
        }
    ).sort_values("window_mid_s").reset_index(drop=True)
    return ev


def add_rolling(ev: pd.DataFrame, value_col: str = "score_mean") -> pd.DataFrame:
    ev = ev.copy()
    # Causal rolling median over 5 half-second bins: no future attack samples leak into thresholding/evaluation.
    ev["score_roll5_median"] = ev[value_col].rolling(window=5, min_periods=3).median().fillna(ev[value_col])
    ev["score_roll5_mean"] = ev[value_col].rolling(window=5, min_periods=3).mean().fillna(ev[value_col])
    return ev


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


def eval_ds4(ev: pd.DataFrame, thresholds: dict[str, float], score_col: str) -> dict[str, object]:
    pre = ev.window_mid_s < (ONSET - BUFFER)
    post = ev.window_mid_s >= (ONSET + BUFFER)
    mask = pre | post
    out: dict[str, object] = {
        "score_col": score_col,
        "pre_windows_t_lt_90": int(pre.sum()),
        "post_windows_t_ge_110": int(post.sum()),
        "auc_pre_vs_post_buffered": auc_rank(post[mask].astype(int).to_numpy(), ev.loc[mask, score_col].to_numpy()),
        "pre_score_median": float(ev.loc[pre, score_col].median()),
        "post_score_median": float(ev.loc[post, score_col].median()),
        "pre_score_q95": float(ev.loc[pre, score_col].quantile(0.95)),
        "post_score_q95": float(ev.loc[post, score_col].quantile(0.95)),
    }
    for name, threshold in thresholds.items():
        flags = ev[score_col] > threshold
        post_flags = flags & post
        first = ev.loc[post_flags, "window_mid_s"].min() if post_flags.any() else np.nan
        out[f"{name}_threshold"] = float(threshold)
        out[f"{name}_pre_fp_rate"] = float((flags & pre).sum() / max(1, int(pre.sum())))
        out[f"{name}_post_det_rate"] = float(post_flags.sum() / max(1, int(post.sum())))
        out[f"{name}_first_delay_s"] = None if np.isnan(first) else float(first - ONSET)
    return out


def main() -> None:
    clean = pd.read_csv(PATHS["cleanStatic"])
    ds4 = pd.read_csv(PATHS["ds4"])
    all_cols = numeric_feature_cols(clean)
    # PRN-local morphology: prompt-relative tap ratios only. No Doppler/CN0/absolute raw power and no attack data in fitting.
    cols = [c for c in all_cols if c.startswith("tap_") and "_rel_prompt_" in c]
    if not cols:
        raise RuntimeError("no tap prompt-relative morphology columns found")
    med, scale = robust_fit(clean, cols)
    clean_ev = add_rolling(event_scores(node_scores(clean, cols, med, scale)))
    ds4_ev = add_rolling(event_scores(node_scores(ds4, cols, med, scale)))

    rows = []
    for score_col in ["score_mean", "score_roll5_median", "score_roll5_mean"]:
        thresholds = {name: float(clean_ev[score_col].quantile(q)) for name, q in QUANTILES}
        rows.append(eval_ds4(ds4_ev, thresholds, score_col))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "ds4_temporal_persistence_metrics.csv", index=False)
    ds4_ev.to_csv(OUT / "ds4_event_scores.csv", index=False)

    best = metrics.sort_values(
        ["q95_post_det_rate", "q99_post_det_rate", "auc_pre_vs_post_buffered", "q95_pre_fp_rate"],
        ascending=[False, False, False, True],
    ).iloc[0].to_dict()
    summary = {
        "hypothesis": "A causal 5-bin temporal persistence filter on PRN-local prompt-relative tap morphology suppresses isolated normal/pre-attack spikes and exposes ds4's sustained post-onset morphology shift.",
        "protocol": {
            "fit_scaler": "cleanStatic robust median/MAD on tap_*_rel_prompt_* columns only",
            "threshold_source": "cleanStatic event-score quantiles after the same causal filter",
            "attack_use": "ds4 is used only for held-out evaluation; onset=100 s, pre<90 s, post>=110 s",
            "aggregation": "per-PRN robust-z node_score, event score = cross-PRN mean, optional causal roll5 filter",
        },
        "feature_count": len(cols),
        "features": cols,
        "metrics_csv": str((OUT / "ds4_temporal_persistence_metrics.csv").relative_to(ROOT)),
        "events_csv": str((OUT / "ds4_event_scores.csv").relative_to(ROOT)),
        "metrics": json.loads(metrics.replace({np.nan: None}).to_json(orient="records")),
        "best_by_q95_detection": best,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# SCI ds4 temporal-persistence morphology experiment\n\n",
        "## Hypothesis\n",
        summary["hypothesis"] + "\n\n",
        "## Protocol\n",
        "- Scaler: robust median/MAD fitted on cleanStatic only.\n",
        "- Features: PRN-local `tap_*_rel_prompt_*` morphology only; no Doppler/CN0/raw power.\n",
        "- Thresholds: q90/q95/q99/q99.5/q99.9 from cleanStatic after the identical causal score filter.\n",
        "- Evaluation: ds4 only, spoof onset assumed 100 s; pre-FP uses t<90 s and post detection uses t>=110 s.\n\n",
        "## ds4 metrics\n",
    ]
    for row in summary["metrics"]:
        lines.append(
            f"- {row['score_col']}: AUC={row['auc_pre_vs_post_buffered']:.3f}, "
            f"q90 det/FP={row['q90_post_det_rate']:.3f}/{row['q90_pre_fp_rate']:.3f}, "
            f"q95 det/FP={row['q95_post_det_rate']:.3f}/{row['q95_pre_fp_rate']:.3f}, "
            f"q99 det/FP={row['q99_post_det_rate']:.3f}/{row['q99_pre_fp_rate']:.3f}, "
            f"q99.5 det/FP={row['q99_5_post_det_rate']:.3f}/{row['q99_5_pre_fp_rate']:.3f}, "
            f"q99.9 det/FP={row['q99_9_post_det_rate']:.3f}/{row['q99_9_pre_fp_rate']:.3f}\n"
        )
    lines.extend([
        "\n## Interpretation\n",
        "The rolling filter is causal and therefore paper-defensible as temporal evidence accumulation, not attack-data tuning. It primarily tests whether ds4 is a sustained PRN-local morphology shift rather than an isolated per-window anomaly.\n",
    ])
    (OUT / "README.md").write_text("".join(lines))
    print(json.dumps(summary, indent=2)[:8000])


if __name__ == "__main__":
    main()
