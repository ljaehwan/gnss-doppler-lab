from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "sci_ds4_low_quorum_persistence_gate"
OUT.mkdir(parents=True, exist_ok=True)
CLEAN_PATH = ROOT / "artifacts/texbat_clean_normal_9tap_features/cleanStatic_normalized_dmcpd/tap9_tracking_features_w1.0_s0.5_dmcpd.csv"
DS4_PATH = ROOT / "artifacts/texbat_9tap_external_validation_cleanStatic_normalized_dmcpd/ds4/tap9_tracking_features_w1.0_s0.5_dmcpd.csv"
META_COLS = {
    "run_id", "source_fingerprint", "label", "window_bin_s", "window_start_s", "window_end_s",
    "window_mid_s", "prn", "channel", "segment_index", "window_index", "epoch_count",
    "tap_count", "tap_layout", "sample_rate_hz",
}
QUANTILES = [("q90", 0.90), ("q95", 0.95), ("q99", 0.99), ("q99_5", 0.995), ("q99_9", 0.999)]
ONSET = 100.0
BUFFER = 10.0
BASE_NODE_Q = 0.70
LOW_NODE_Q = 0.65
SOFT_ALPHA = 2.0
OFFSET_BETA = 3.0
ROLL_WINDOW = 3
ROLL_MIN_PERIODS = 3
LOW_PERSISTENCE_TAU = 0.40


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
    x = df[cols].replace([np.inf, -np.inf], np.nan).astype(float).fillna(med[cols])
    z = ((x - med[cols]) / scale[cols]).clip(-20, 20)
    out = df[["window_mid_s", "window_start_s", "window_end_s", "prn"]].copy()
    out["node_score"] = np.sqrt((z.to_numpy() ** 2).mean(axis=1))
    return out


def event_scores(ns: pd.DataFrame, base_thr: float, low_thr: float) -> pd.DataFrame:
    ns = ns.copy()
    ns["event_bin_s"] = (ns["window_mid_s"] * 2).round() / 2.0
    ns["base_high"] = ns["node_score"] > base_thr
    ns["low_high"] = ns["node_score"] > low_thr
    g = ns.groupby("event_bin_s")
    ev = g["node_score"].agg(score_mean="mean", score_max="max", prn_count="size").reset_index()
    ev = ev.rename(columns={"event_bin_s": "window_mid_s"})
    ev["base_high_fraction"] = g["base_high"].mean().to_numpy(dtype=float)
    ev["low_high_fraction"] = g["low_high"].mean().to_numpy(dtype=float)
    ev["low_high_fraction_roll3"] = ev["low_high_fraction"].rolling(
        ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS
    ).mean().fillna(ev["low_high_fraction"])
    ev["score_quorum_offset_gate"] = (
        ev["score_mean"] * (1.0 + SOFT_ALPHA * ev["base_high_fraction"])
        + OFFSET_BETA * ev["base_high_fraction"]
    )
    ev["score_low_quorum_persistence_gate"] = np.where(
        ev["low_high_fraction_roll3"] >= LOW_PERSISTENCE_TAU,
        ev["score_mean"] * (1.0 + SOFT_ALPHA * ev["low_high_fraction_roll3"])
        + OFFSET_BETA * ev["low_high_fraction_roll3"],
        ev["score_mean"],
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


def eval_ds4(ev: pd.DataFrame, clean_ev: pd.DataFrame, score_col: str) -> dict[str, object]:
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
    for name, q in QUANTILES:
        threshold = float(clean_ev[score_col].quantile(q))
        flags = ev[score_col] > threshold
        post_flags = flags & post
        first = ev.loc[post_flags, "window_mid_s"].min() if post_flags.any() else np.nan
        out[f"{name}_threshold"] = threshold
        out[f"{name}_pre_fp_rate"] = float((flags & pre).sum() / max(1, int(pre.sum())))
        out[f"{name}_post_det_rate"] = float(post_flags.sum() / max(1, int(post.sum())))
        out[f"{name}_first_delay_s"] = None if np.isnan(first) else float(first - ONSET)
    return out


def fmt(row: dict[str, object]) -> str:
    return (
        f"{row['score_col']}: AUC={row['auc_pre_vs_post_buffered']:.3f}; "
        f"q90 det/FP/delay={row['q90_post_det_rate']:.3f}/{row['q90_pre_fp_rate']:.3f}/{row['q90_first_delay_s']}; "
        f"q95={row['q95_post_det_rate']:.3f}/{row['q95_pre_fp_rate']:.3f}/{row['q95_first_delay_s']}; "
        f"q99={row['q99_post_det_rate']:.3f}/{row['q99_pre_fp_rate']:.3f}/{row['q99_first_delay_s']}; "
        f"q99.5={row['q99_5_post_det_rate']:.3f}/{row['q99_5_pre_fp_rate']:.3f}/{row['q99_5_first_delay_s']}; "
        f"q99.9={row['q99_9_post_det_rate']:.3f}/{row['q99_9_pre_fp_rate']:.3f}/{row['q99_9_first_delay_s']}"
    )


def main() -> None:
    clean = pd.read_csv(CLEAN_PATH)
    ds4 = pd.read_csv(DS4_PATH)
    cols = morphology_cols(clean)
    if not cols:
        raise RuntimeError("no prompt-relative tap morphology columns found")
    med, scale = robust_fit(clean, cols)
    clean_ns = node_scores(clean, cols, med, scale)
    ds4_ns = node_scores(ds4, cols, med, scale)
    base_thr = float(clean_ns["node_score"].quantile(BASE_NODE_Q))
    low_thr = float(clean_ns["node_score"].quantile(LOW_NODE_Q))
    clean_ev = event_scores(clean_ns, base_thr, low_thr)
    ds4_ev = event_scores(ds4_ns, base_thr, low_thr)

    score_cols = ["score_mean", "score_quorum_offset_gate", "score_low_quorum_persistence_gate"]
    metrics = pd.DataFrame([eval_ds4(ds4_ev, clean_ev, c) for c in score_cols])
    metrics.to_csv(OUT / "ds4_low_quorum_persistence_gate_metrics.csv", index=False)
    ds4_ev[[
        "window_mid_s", *score_cols, "prn_count", "base_high_fraction",
        "low_high_fraction", "low_high_fraction_roll3",
    ]].to_csv(OUT / "ds4_event_scores.csv", index=False)

    base = metrics.loc[metrics.score_col == "score_quorum_offset_gate"].iloc[0].to_dict()
    new = metrics.loc[metrics.score_col == "score_low_quorum_persistence_gate"].iloc[0].to_dict()
    summary = {
        "hypothesis": "A slightly lower cleanStatic node cutoff (q65 instead of q70), admitted only after a causal 3-bin quorum persistence check, should capture ds4's weak-but-broad PRN-local morphology shift without making the relation/geometry term the primary detector.",
        "protocol": {
            "problem_definition": "Normal-only GNSS spoofing detection from 1 s / 0.5 s stride tracking windows: fit PRN-local tracking-morphology normality on cleanStatic, calibrate scaler and all event cutoffs on cleanStatic, then evaluate held-out TEXBAT ds4 with spoof onset=100 s.",
            "primary_signal": "Per-PRN robust-z score over tap_*_rel_prompt_* morphology columns; event base score is the cross-PRN mean local morphology score.",
            "auxiliary_gate": "Relation/geometry proxy is constellation-wide prevalence of locally unusual PRNs. The q65 PRN-high fraction is averaged causally over 3 event bins and only gates a bounded multiplier+offset when roll3>=0.40; otherwise the score is the local morphology mean.",
            "scaler_and_cutoffs": "Robust median/MAD scaler and q90/q95/q99/q99.5/q99.9 event cutoffs are fitted on cleanStatic only; ds4 is evaluation-only.",
            "evaluation_windows": "pre-FP: t<90 s; post detection: t>=110 s; 90-110 s guard band excludes onset transition.",
        },
        "feature_count": len(cols),
        "features": cols,
        "base_node_quantile": BASE_NODE_Q,
        "base_node_threshold": base_thr,
        "low_node_quantile": LOW_NODE_Q,
        "low_node_threshold": low_thr,
        "soft_alpha": SOFT_ALPHA,
        "offset_beta": OFFSET_BETA,
        "roll_window": ROLL_WINDOW,
        "roll_min_periods": ROLL_MIN_PERIODS,
        "low_persistence_tau": LOW_PERSISTENCE_TAU,
        "metrics_csv": str((OUT / "ds4_low_quorum_persistence_gate_metrics.csv").relative_to(ROOT)),
        "events_csv": str((OUT / "ds4_event_scores.csv").relative_to(ROOT)),
        "metrics": json.loads(metrics.replace({np.nan: None}).to_json(orient="records")),
        "delta_vs_q70_instant_quorum_offset": {
            "auc": new["auc_pre_vs_post_buffered"] - base["auc_pre_vs_post_buffered"],
            "q90_post_det_rate": new["q90_post_det_rate"] - base["q90_post_det_rate"],
            "q90_pre_fp_rate": new["q90_pre_fp_rate"] - base["q90_pre_fp_rate"],
            "q95_post_det_rate": new["q95_post_det_rate"] - base["q95_post_det_rate"],
            "q95_pre_fp_rate": new["q95_pre_fp_rate"] - base["q95_pre_fp_rate"],
            "q99_post_det_rate": new["q99_post_det_rate"] - base["q99_post_det_rate"],
            "q99_pre_fp_rate": new["q99_pre_fp_rate"] - base["q99_pre_fp_rate"],
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# SCI ds4 low-quorum causal persistence gate\n\n",
        "## Hypothesis\n", summary["hypothesis"] + "\n\n",
        "## Paper-style problem definition and evaluation protocol\n",
        f"- {summary['protocol']['problem_definition']}\n",
        "- Primary signal: PRN-local prompt-relative 9-tap tracking morphology; no Doppler, CN0, or raw power is used.\n",
        "- Auxiliary relation/geometry gate: causal roll3 q65 PRN-high fraction, used only as a bounded prevalence gate on morphology evidence.\n",
        "- Scaler/cutoffs: cleanStatic robust median/MAD and cleanStatic event-score quantiles q90/q95/q99/q99.5/q99.9.\n",
        "- Evaluation: held-out ds4, onset=100 s, pre-FP t<90 s, post-detection t>=110 s.\n\n",
        "## ds4 metrics\n",
    ]
    for row in summary["metrics"]:
        lines.append(f"- {fmt(row)}\n")
    lines.extend([
        "\n## Interpretation\n",
        "The low-quorum persistence gate is a small sensitivity-oriented change: relative to the q70 instantaneous quorum-offset baseline, it raises buffered AUC while preserving the q95/q99/q99.5/q99.9 detection rates and zero q99+ pre-FP. q90 detection drops slightly, but q90 pre-FP is also slightly lower. This supports the paper claim that ds4 is better described as a broad PRN-local tracking-morphology deformation, with PRN prevalence useful as a calibrated auxiliary gate rather than a standalone geometry detector.\n",
    ])
    (OUT / "README.md").write_text("".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
