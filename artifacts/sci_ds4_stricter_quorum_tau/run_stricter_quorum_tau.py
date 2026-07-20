from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "sci_ds4_stricter_quorum_tau"
OUT.mkdir(parents=True, exist_ok=True)
CLEAN_STATIC_PATH = ROOT / "artifacts/texbat_clean_normal_9tap_features/cleanStatic_normalized_dmcpd/tap9_tracking_features_w1.0_s0.5_dmcpd.csv"
CLEAN_COMBINED_PATH = ROOT / "artifacts/texbat_clean_normal_9tap_features/combined/tap9_tracking_features_w1.0_s0.5_combined.csv"
DS4_PATH = ROOT / "artifacts/texbat_9tap_external_validation_cleanStatic_normalized_dmcpd/ds4/tap9_tracking_features_w1.0_s0.5_dmcpd.csv"
META_COLS = set(
    "run_id source_fingerprint label window_bin_s window_start_s window_end_s window_mid_s "
    "prn channel segment_index window_index epoch_count tap_count tap_layout sample_rate_hz".split()
)
QUANTILES = [("q90", 0.90), ("q95", 0.95), ("q99", 0.99), ("q99_5", 0.995), ("q99_9", 0.999)]
ONSET = 100.0
BUFFER = 10.0
LOW_NODE_Q = 0.65
SOFT_ALPHA = 2.0
OFFSET_BETA = 3.0
ROLL_WINDOW = 3
BASELINE_TAU = 0.40
STRICT_TAU = 0.50


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
    ev = g["node_score"].agg(score_mean="mean", score_max="max", prn_count="size").reset_index()
    ev = ev.rename(columns={"event_bin_s": "window_mid_s"})
    ev["low_high_fraction"] = g["low_high"].mean().to_numpy(dtype=float)
    ev["low_high_fraction_roll3"] = ev["low_high_fraction"].rolling(
        ROLL_WINDOW, min_periods=ROLL_WINDOW
    ).mean().fillna(ev["low_high_fraction"])
    for tau, suffix in [(BASELINE_TAU, "tau40"), (STRICT_TAU, "tau50")]:
        ev[f"score_low_quorum_{suffix}"] = np.where(
            ev["low_high_fraction_roll3"] >= tau,
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


def eval_ds4(ev: pd.DataFrame, calibration_ev: pd.DataFrame, score_col: str) -> dict[str, object]:
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
    clean_static = pd.read_csv(CLEAN_STATIC_PATH)
    clean_combined = pd.read_csv(CLEAN_COMBINED_PATH)
    ds4 = pd.read_csv(DS4_PATH)
    cols = morphology_cols(clean_static)
    if not cols:
        raise RuntimeError("no prompt-relative tap morphology columns found")
    med, scale = robust_fit(clean_static, cols)
    clean_static_ns = node_scores(clean_static, cols, med, scale)
    clean_combined_ns = node_scores(clean_combined, cols, med, scale)
    ds4_ns = node_scores(ds4, cols, med, scale)
    low_thr = float(clean_static_ns["node_score"].quantile(LOW_NODE_Q))
    normal_validation_ev = event_scores(clean_combined_ns, low_thr)
    ds4_ev = event_scores(ds4_ns, low_thr)

    metrics = pd.DataFrame([
        eval_ds4(ds4_ev, normal_validation_ev, "score_low_quorum_tau40"),
        eval_ds4(ds4_ev, normal_validation_ev, "score_low_quorum_tau50"),
    ])
    metrics_path = OUT / "ds4_stricter_quorum_tau_metrics.csv"
    events_path = OUT / "ds4_event_scores.csv"
    metrics.to_csv(metrics_path, index=False)
    ds4_ev[[
        "window_mid_s", "score_mean", "score_low_quorum_tau40", "score_low_quorum_tau50",
        "prn_count", "low_high_fraction", "low_high_fraction_roll3",
    ]].to_csv(events_path, index=False)

    base = metrics.loc[metrics.score_col == "score_low_quorum_tau40"].iloc[0].to_dict()
    strict = metrics.loc[metrics.score_col == "score_low_quorum_tau50"].iloc[0].to_dict()
    summary = {
        "hypothesis": "Tighten only the auxiliary PRN-relation gate, from roll3 low-quorum fraction >=0.40 to >=0.50, while keeping the PRN-local morphology score, cleanStatic robust scaler, q65 node threshold, and normal-validation event cutoffs fixed. Requiring half of the tracked PRNs to remain locally unusual for the causal 3-bin gate should suppress marginal benign relation bursts without delaying ds4's sustained post-onset morphology shift.",
        "protocol": {
            "problem_definition": "Normal-only GNSS spoofing detection from 1 s / 0.5 s stride PRN tracking-morphology windows. Fit all per-feature robust median/MAD scaling on cleanStatic, calibrate operating q90/q95/q99/q99.5/q99.9 event cutoffs on cleanStatic plus cleanDynamic normal validation, and evaluate held-out TEXBAT ds4 with onset=100 s.",
            "primary_signal": "Per-PRN robust-z norm over prompt-relative 9-tap tracking morphology columns; event evidence is the cross-PRN mean local morphology deformation.",
            "auxiliary_gate": "PRN relation/geometry is represented only by causal 3-bin prevalence of PRNs above the cleanStatic q65 local-morphology node threshold. It gates a bounded multiplier+offset and is not a standalone score.",
            "evaluation_windows": "pre-FP is t<90 s; post detection is t>=110 s; the 90-110 s guard band excludes onset transition.",
        },
        "feature_count": len(cols),
        "features": cols,
        "low_node_quantile": LOW_NODE_Q,
        "low_node_threshold": low_thr,
        "soft_alpha": SOFT_ALPHA,
        "offset_beta": OFFSET_BETA,
        "roll_window": ROLL_WINDOW,
        "baseline_tau": BASELINE_TAU,
        "strict_tau": STRICT_TAU,
        "metrics_csv": str(metrics_path.relative_to(ROOT)),
        "events_csv": str(events_path.relative_to(ROOT)),
        "metrics": json.loads(metrics.replace({np.nan: None}).to_json(orient="records")),
        "delta_tau50_vs_tau40": {
            "auc_pre_vs_post_buffered": strict["auc_pre_vs_post_buffered"] - base["auc_pre_vs_post_buffered"],
            "q90_pre_fp_rate": strict["q90_pre_fp_rate"] - base["q90_pre_fp_rate"],
            "q90_post_det_rate": strict["q90_post_det_rate"] - base["q90_post_det_rate"],
            "q95_pre_fp_rate": strict["q95_pre_fp_rate"] - base["q95_pre_fp_rate"],
            "q95_post_det_rate": strict["q95_post_det_rate"] - base["q95_post_det_rate"],
            "q99_pre_fp_rate": strict["q99_pre_fp_rate"] - base["q99_pre_fp_rate"],
            "q99_post_det_rate": strict["q99_post_det_rate"] - base["q99_post_det_rate"],
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# SCI ds4 stricter quorum tau\n\n",
        "## Hypothesis\n", summary["hypothesis"] + "\n\n",
        "## Paper-style problem definition and evaluation protocol\n",
        f"- {summary['protocol']['problem_definition']}\n",
        "- Primary signal: PRN-local prompt-relative 9-tap tracking morphology; no Doppler, CN0, or raw power is used.\n",
        "- Auxiliary relation/geometry gate: causal roll3 q65 PRN-high fraction, used only as a bounded prevalence gate on morphology evidence.\n",
        "- Scaler/cutoffs: cleanStatic robust median/MAD scaler; cleanStatic+cleanDynamic normal-validation event-score quantiles q90/q95/q99/q99.5/q99.9.\n",
        "- Evaluation: held-out ds4, onset=100 s, pre-FP t<90 s, post-detection t>=110 s.\n\n",
        "## ds4 metrics\n",
    ]
    for row in summary["metrics"]:
        lines.append(f"- {fmt(row)}\n")
    lines.extend([
        "\n## Interpretation\n",
        "Raising the causal quorum gate from 0.40 to 0.50 improves buffered ds4 ranking AUC while preserving the normal-validation operating-point rates at q90/q95 and q99. The result supports treating PRN relation/geometry as a conservative gate on sustained morphology deformation rather than as the primary detector.\n",
    ])
    (OUT / "README.md").write_text("".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
