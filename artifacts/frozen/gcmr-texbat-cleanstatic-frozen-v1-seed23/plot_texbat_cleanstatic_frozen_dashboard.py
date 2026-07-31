#!/usr/bin/env python3
"""Plot the frozen one-model TEXBAT cleanStatic evaluation without recalibration."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, PercentFormatter

ARTIFACT = Path(__file__).resolve().parent
SUMMARY = ARTIFACT / "summary.json"
OUTPUT = ARTIFACT / "texbat_cleanstatic_frozen_dashboard.png"
SCENARIOS = ["DS1", "DS2", "DS3", "DS4"]
COLORS = {"DS1": "#3b82f6", "DS2": "#8b5cf6", "DS3": "#ec4899", "DS4": "#f59e0b"}
THRESHOLD_DISPLAY = "134914.8"


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"window_start_s", "window_end_s", "score_available_s", "combined_score", "threshold", "alarm"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Missing required frozen score columns in {path}")
    return {
        "start": np.array([float(r["window_start_s"]) for r in rows]),
        "end": np.array([float(r["window_end_s"]) for r in rows]),
        "time": np.array([float(r["score_available_s"]) for r in rows]),
        "score": np.array([float(r["combined_score"]) for r in rows]),
        "threshold": np.array([float(r["threshold"]) for r in rows]),
        "alarm": np.array([r["alarm"].strip().lower() == "true" for r in rows]),
    }


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    threshold = float(summary["threshold"])
    if not np.isclose(threshold, 134914.8453732542, rtol=0, atol=1e-8):
        raise RuntimeError(f"Unexpected frozen threshold: {threshold}")
    data = {name: load_csv(ARTIFACT / f"{name}_scores.csv") for name in ["cleanStatic", *SCENARIOS]}
    if any(not np.allclose(d["threshold"], threshold, rtol=0, atol=1e-8) for d in data.values()):
        raise RuntimeError("CSV threshold differs from frozen summary threshold")
    total_events = sum(len(d["score"]) for d in data.values())
    total_alarms = sum(int(d["alarm"].sum()) for d in data.values())
    if total_alarms != 0:
        raise RuntimeError(f"Frozen artifact no longer has zero alarms: {total_alarms}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.titleweight": "bold", "axes.edgecolor": "#475569",
        "axes.labelcolor": "#1e293b", "xtick.color": "#334155", "ytick.color": "#334155",
    })
    # Reserve dedicated, non-overlapping bands for header, plots, and two footer lines.
    fig = plt.figure(figsize=(18, 15.5), facecolor="#f8fafc")
    gs = fig.add_gridspec(2, 2, height_ratios=[3.3, 5.6], hspace=0.44, wspace=0.22,
                          left=0.06, right=0.98, top=0.76, bottom=0.17)

    # Honest, prominent result header.  Each line has its own fixed vertical band.
    ax_head = fig.add_axes([0.06, 0.795, 0.92, 0.18]); ax_head.axis("off")
    ax_head.text(0.0, 0.96, "TEXBAT cleanStatic — frozen one-model external evaluation",
                 fontsize=22, fontweight="bold", color="#0f172a", va="top")
    ax_head.text(0.0, 0.72,
                 "Single cleanStatic model frozen before DS access • DS1–DS4 external inference only",
                 fontsize=11.8, color="#334155", va="top")
    ax_head.text(0.0, 0.43,
                 f"CALIBRATION FAILURE / NO DETECTOR SUCCESS   •   threshold = {THRESHOLD_DISPLAY}   •   ZERO ALARMS ({total_alarms}/{total_events})",
                 fontsize=15.2, fontweight="bold", color="white", va="center",
                 bbox=dict(boxstyle="round,pad=0.55", facecolor="#b91c1c", edgecolor="#7f1d1d"))
    ax_head.text(0.0, 0.06,
                 "Trained, calibrated, frozen, and reloaded before any DS access. No DS adaptation, recalibration, or result modification.",
                 fontsize=11.3, color="#334155", va="bottom")

    # Alarm-rate panel.
    ax = fig.add_subplot(gs[0, 0], facecolor="white")
    names = ["Clean sealed\nFPR", *SCENARIOS]
    x = np.arange(len(names), dtype=float)
    clean = summary["sealed_held"]["stable_post"]
    clean_rate = float(clean["alarm_rate"])
    ax.bar(x[0], clean_rate, width=0.52, color="#0f766e", label="sealed clean FPR", zorder=3)
    pre_rates, post_rates = [], []
    for name in SCENARIOS:
        r = summary["results"][name]
        pre_rates.append(float(r["stable_pre"]["alarm_rate"]))
        post_rates.append(float(r["stable_post"]["alarm_rate"]))
    ax.bar(x[1:] - 0.18, pre_rates, width=0.34, color="#64748b", label="DS stable pre [30,90)", zorder=3)
    ax.bar(x[1:] + 0.18, post_rates, width=0.34, color="#dc2626", label="DS stable post ≥110 s", zorder=3)
    ax.scatter(x, np.zeros_like(x), color="#0f172a", marker="o", s=30, zorder=5)
    ax.text(x[0], 0.035, f"0/{clean['event_count']}", ha="center", va="bottom", fontweight="bold")
    for i, name in enumerate(SCENARIOS, 1):
        r = summary["results"][name]
        ax.text(x[i]-0.18, 0.035, f"0/{r['stable_pre']['event_count']}", ha="center", va="bottom", fontsize=8.5)
        ax.text(x[i]+0.18, 0.10, f"0/{r['stable_post']['event_count']}", ha="center", va="bottom", fontsize=8.5)
    ax.set_title("Sealed clean FPR and DS stable alarm rates", loc="left", fontsize=14)
    ax.set_ylabel("Alarm rate")
    ax.set_ylim(0, 1.03); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks(x, names); ax.grid(axis="y", color="#e2e8f0", linewidth=0.8, zorder=0)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.text(0.01, 0.82, "All rates are 0% — including every post-onset window.", transform=ax.transAxes,
            color="#991b1b", fontweight="bold", fontsize=11)

    # Score summaries versus frozen threshold.
    ax = fig.add_subplot(gs[0, 1], facecolor="white")
    labels = ["Clean\nsealed"]
    med = [float(clean["score_median"])]; q99 = [float(clean["score_q99"])]
    colors = ["#0f766e"]
    for name in SCENARIOS:
        r = summary["results"][name]
        for phase in ("stable_pre", "stable_post"):
            labels.append(f"{name}\n{'pre' if phase == 'stable_pre' else 'post'}")
            med.append(float(r[phase]["score_median"])); q99.append(float(r[phase]["score_q99"]))
            colors.append("#64748b" if phase == "stable_pre" else COLORS[name])
    xx = np.arange(len(labels))
    ax.vlines(xx, med, q99, colors=colors, linewidth=2.2, alpha=0.75, zorder=2)
    ax.scatter(xx, med, c=colors, s=48, marker="o", edgecolor="white", linewidth=0.6, label="median", zorder=4)
    ax.scatter(xx, q99, c=colors, s=58, marker="D", edgecolor="#0f172a", linewidth=0.5, label="q99", zorder=4)
    ax.axhline(threshold, color="#b91c1c", linewidth=2.4, linestyle="--", label=f"frozen threshold {THRESHOLD_DISPLAY}")
    ax.text(len(labels)-0.05, threshold/1.55, f"threshold {THRESHOLD_DISPLAY}", ha="right", va="top",
            color="#991b1b", fontweight="bold")
    ax.set_yscale("log"); ax.set_ylim(0.35, threshold*2.2)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.set_xticks(xx, labels); ax.set_ylabel("Combined score (log scale)")
    ax.set_title("Score median / q99 versus frozen threshold", loc="left", fontsize=14)
    ax.grid(axis="y", which="both", color="#e2e8f0", linewidth=0.7)
    ax.legend(frameon=False, loc="lower right", ncol=3, fontsize=8.5)

    # DS timelines around nominal onset (100 s), all using frozen score and threshold.
    sub = gs[1, :].subgridspec(2, 2, hspace=0.34, wspace=0.16)
    for i, name in enumerate(SCENARIOS):
        ax = fig.add_subplot(sub[i // 2, i % 2], facecolor="white")
        d = data[name]; rel = d["time"] - 100.0
        use = (rel >= -40.0) & (rel <= 40.0)
        score = np.maximum(d["score"][use], 1e-3)
        ax.axvspan(-10, 10, color="#fef3c7", alpha=0.75, label="transition 90–110 s")
        ax.axvspan(10, 40, color="#fee2e2", alpha=0.40, label="stable post ≥110 s")
        ax.plot(rel[use], score, color=COLORS[name], linewidth=1.45, label="combined score")
        ax.axvline(0, color="#0f172a", linestyle="-", linewidth=1.2, label="primary onset 100 s")
        ax.axhline(threshold, color="#b91c1c", linestyle="--", linewidth=1.5, label="frozen threshold")
        ax.set_yscale("log"); ax.set_ylim(1e-2, threshold*2.0); ax.set_xlim(-40, 40)
        ax.set_title(f"{name}: 0 alarms", loc="left", color=COLORS[name], fontsize=12.5)
        ax.set_xlabel("Time relative to primary onset (s)")
        ax.set_ylabel("Combined score (log)")
        ax.grid(which="both", color="#e2e8f0", linewidth=0.6)
        if i == 0:
            ax.legend(frameon=False, fontsize=8.2, loc="lower left", ncol=2)
    # Two footer lines occupy separate rows in the dedicated bottom margin.
    fig.text(0.06, 0.095,
             "Frozen contract: threshold = cleanStatic event-calibration q99 only; sealed held clean FPR = 0/119; primary onset = 100 s; stable pre = start≥30,end≤90; stable post = start≥110.",
             fontsize=9.5, color="#475569", va="center")
    fig.text(0.06, 0.050,
             "Interpretation: the extreme threshold suppresses both false and true alarms; zero FPR is not detection success.",
             fontsize=9.5, color="#991b1b", fontweight="bold", va="center")
    fig.savefig(OUTPUT, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight", metadata={"Title": "Frozen TEXBAT cleanStatic evaluation dashboard"})
    print(json.dumps({"output": str(OUTPUT), "threshold": threshold, "total_events": total_events, "total_alarms": total_alarms}, indent=2))


if __name__ == "__main__":
    main()
