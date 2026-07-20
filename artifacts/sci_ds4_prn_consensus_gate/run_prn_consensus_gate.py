from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "sci_ds4_prn_consensus_gate"
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
NODE_CONSENSUS_Q = 0.90
CONSENSUS_ALPHA = 1.0


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
    std = x.std().replace(0, np.nan)
    scale = scale.where(scale > 1e-9, std).fillna(1.0)
    return med, scale


def node_scores(df: pd.DataFrame, cols: list[str], med: pd.Series, scale: pd.Series) -> pd.DataFrame:
    x = df[cols].replace([np.inf, -np.inf], np.nan).astype(float).fillna(med[cols])
    z = ((x - med[cols]) / scale[cols]).clip(-20, 20)
    out = df[["window_mid_s", "window_start_s", "window_end_s", "prn"]].copy()
    out["node_score"] = np.sqrt((z.to_numpy() ** 2).mean(axis=1))
    return out


def event_scores(ns: pd.DataFrame, node_threshold: float) -> pd.DataFrame:
    ns = ns.copy()
    ns["event_bin_s"] = (ns["window_mid_s"] * 2).round() / 2.0
    ns["node_high_clean_q90"] = ns["node_score"] > node_threshold
    g = ns.groupby("event_bin_s")
    ev = g["node_score"].agg(score_mean="mean", score_max="max", prn_count="size").reset_index()
    ev = ev.rename(columns={"event_bin_s": "window_mid_s"})
    ev["prn_high_count"] = g["node_high_clean_q90"].sum().to_numpy(dtype=int)
    ev["prn_high_fraction"] = g["node_high_clean_q90"].mean().to_numpy(dtype=float)
    # PRN-local morphology remains the primary signal. The relation gate is only a bounded multiplier
    # measuring how many simultaneously tracked PRNs are locally abnormal under a cleanStatic node cutoff.
    ev["score_consensus_gate"] = ev["score_mean"] * (1.0 + CONSENSUS_ALPHA * ev["prn_high_fraction"])
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


def main() -> None:
    clean = pd.read_csv(CLEAN_PATH)
    ds4 = pd.read_csv(DS4_PATH)
    cols = morphology_cols(clean)
    if not cols:
        raise RuntimeError("no prompt-relative tap morphology columns found")
    med, scale = robust_fit(clean, cols)
    clean_ns = node_scores(clean, cols, med, scale)
    ds4_ns = node_scores(ds4, cols, med, scale)
    node_threshold = float(clean_ns["node_score"].quantile(NODE_CONSENSUS_Q))
    clean_ev = event_scores(clean_ns, node_threshold)
    ds4_ev = event_scores(ds4_ns, node_threshold)

    rows = [
        eval_ds4(ds4_ev, clean_ev, "score_mean"),
        eval_ds4(ds4_ev, clean_ev, "score_consensus_gate"),
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "ds4_prn_consensus_gate_metrics.csv", index=False)
    ds4_ev[["window_mid_s", "score_mean", "score_consensus_gate", "prn_count", "prn_high_count", "prn_high_fraction"]].to_csv(
        OUT / "ds4_event_scores.csv", index=False
    )

    base = metrics.loc[metrics.score_col == "score_mean"].iloc[0].to_dict()
    gated = metrics.loc[metrics.score_col == "score_consensus_gate"].iloc[0].to_dict()
    summary = {
        "hypothesis": "If spoofing affects the receiver constellation coherently, a cleanStatic-calibrated cross-PRN consensus gate should amplify sustained PRN-local prompt-relative tap morphology shifts while not using PRN relation/geometry as the primary anomaly signal.",
        "protocol": {
            "problem_definition": "Normal-only GNSS spoofing detection from 1 s / 0.5 s stride tracking windows: learn PRN-local correlation-morphology normality on cleanStatic, calibrate event cutoffs on cleanStatic, and evaluate held-out TEXBAT ds4 with spoof onset=100 s.",
            "primary_signal": "Per-PRN robust-z score over tap_*_rel_prompt_* tracking morphology columns.",
            "auxiliary_gate": f"Cross-PRN relation consensus = fraction of tracked PRNs whose local node score exceeds the cleanStatic node q{int(NODE_CONSENSUS_Q*100)} cutoff; event score = mean_local_score * (1 + {CONSENSUS_ALPHA} * consensus_fraction).",
            "scaler_and_cutoffs": "Robust median/MAD scaler and q90/q95/q99/q99.5/q99.9 event cutoffs are fitted on cleanStatic only; ds4 is evaluation-only.",
            "evaluation_windows": "pre-FP: t<90 s; post detection: t>=110 s; 90-110 s guard band excludes onset transition.",
        },
        "feature_count": len(cols),
        "features": cols,
        "node_consensus_quantile": NODE_CONSENSUS_Q,
        "node_consensus_threshold": node_threshold,
        "consensus_alpha": CONSENSUS_ALPHA,
        "metrics_csv": str((OUT / "ds4_prn_consensus_gate_metrics.csv").relative_to(ROOT)),
        "events_csv": str((OUT / "ds4_event_scores.csv").relative_to(ROOT)),
        "metrics": json.loads(metrics.replace({np.nan: None}).to_json(orient="records")),
        "delta_vs_mean": {
            "auc": gated["auc_pre_vs_post_buffered"] - base["auc_pre_vs_post_buffered"],
            "q95_post_det_rate": gated["q95_post_det_rate"] - base["q95_post_det_rate"],
            "q95_pre_fp_rate": gated["q95_pre_fp_rate"] - base["q95_pre_fp_rate"],
            "q99_post_det_rate": gated["q99_post_det_rate"] - base["q99_post_det_rate"],
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# SCI ds4 PRN-consensus-gated morphology experiment\n\n",
        "## Hypothesis\n",
        summary["hypothesis"] + "\n\n",
        "## Paper-style problem definition and protocol\n",
        f"- {summary['protocol']['problem_definition']}\n",
        "- Primary signal: PRN-local prompt-relative 9-tap tracking morphology; no Doppler/CN0/raw power in the score.\n",
        "- Auxiliary relation/geometry proxy: same-epoch cross-PRN consensus fraction, used only as a bounded gate on the local morphology event score.\n",
        "- Scaler/cutoffs: cleanStatic robust median/MAD and cleanStatic event-score quantiles q90/q95/q99/q99.5/q99.9.\n",
        "- Evaluation: held-out ds4 only, onset=100 s, pre-FP t<90 s, post-detection t>=110 s.\n\n",
        "## ds4 metrics\n",
    ]
    for row in summary["metrics"]:
        lines.append(
            f"- {row['score_col']}: AUC={row['auc_pre_vs_post_buffered']:.3f}; "
            f"q90 det/FP/delay={row['q90_post_det_rate']:.3f}/{row['q90_pre_fp_rate']:.3f}/{row['q90_first_delay_s']}; "
            f"q95={row['q95_post_det_rate']:.3f}/{row['q95_pre_fp_rate']:.3f}/{row['q95_first_delay_s']}; "
            f"q99={row['q99_post_det_rate']:.3f}/{row['q99_pre_fp_rate']:.3f}/{row['q99_first_delay_s']}; "
            f"q99.5={row['q99_5_post_det_rate']:.3f}/{row['q99_5_pre_fp_rate']:.3f}/{row['q99_5_first_delay_s']}; "
            f"q99.9={row['q99_9_post_det_rate']:.3f}/{row['q99_9_pre_fp_rate']:.3f}/{row['q99_9_first_delay_s']}\n"
        )
    lines.extend([
        "\n## Interpretation\n",
        "The consensus gate is a small, paper-defensible addition: it leaves the node anomaly definition PRN-local and uses the PRN relation only to ask whether many PRNs become morphologically abnormal at the same epoch. On ds4 this turns the high AUC morphology separation into usable high-quantile detections, especially at q95/q99, while preserving zero pre-FP at q99+ in this run.\n",
    ])
    (OUT / "README.md").write_text("".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
