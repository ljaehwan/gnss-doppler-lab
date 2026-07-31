import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

BASE = Path(__file__).resolve().parent
SUMMARY = BASE / "summary.json"
OUTPUT = BASE / "gcmr_oakbat_dashboard.png"
MANIFEST = BASE / "gcmr_oakbat_dashboard.sha256.txt"
SCENARIOS = ["os1", "os2", "os3", "os4"]
THRESHOLD_TOL = 1e-12


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    times = np.array([float(row["score_available_s"]) for row in rows])
    scores = np.array([float(row["combined_score"]) for row in rows])
    thresholds = np.array([float(row["threshold"]) for row in rows])
    alarms = np.array([row["alarm"].strip().lower() == "true" for row in rows])
    return times, scores, thresholds, alarms


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def main():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    frozen_threshold = float(summary["threshold"])
    data, metrics = {}, {}
    for scenario in SCENARIOS:
        times, scores, thresholds, alarms = load_csv(BASE / f"{scenario}_scores.csv")
        if not np.allclose(thresholds, frozen_threshold, rtol=0, atol=THRESHOLD_TOL):
            raise ValueError(f"{scenario}: CSV threshold differs from frozen summary threshold")
        if not np.array_equal(alarms, scores > thresholds):
            raise ValueError(f"{scenario}: frozen alarms inconsistent with score > threshold")
        data[scenario] = (times, alarms)
        pre = (times >= 60) & (times < 110)
        post = times >= 130
        metrics[scenario] = {
            "pre_n": int(pre.sum()), "pre_a": int(alarms[pre].sum()), "pre": float(alarms[pre].mean()),
            "post_n": int(post.sum()), "post_a": int(alarms[post].sum()), "post": float(alarms[post].mean()),
        }

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.2, "axes.titlesize": 10.8,
        "axes.titleweight": "bold", "axes.labelsize": 9.4, "axes.spines.top": False,
        "axes.spines.right": False, "axes.labelcolor": "#334155", "xtick.color": "#334155",
        "ytick.color": "#334155", "figure.facecolor": "#F8FAFC", "savefig.facecolor": "#F8FAFC",
        "axes.facecolor": "white", "grid.color": "#E2E8F0", "grid.linewidth": 0.8,
    })
    fig = plt.figure(figsize=(17.5, 10.5))
    grid = fig.add_gridspec(2, 3, left=0.055, right=0.975, bottom=0.105, top=0.835, wspace=0.28, hspace=0.34)
    timeline_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
                     fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    bar_ax = fig.add_subplot(grid[0, 2])
    ablation_ax = fig.add_subplot(grid[1, 2])
    colors = ["#0072B2", "#D55E00", "#009E73", "#7C3F98"]

    for ax, scenario, color, panel in zip(timeline_axes, SCENARIOS, colors, ["a", "b", "c", "d"]):
        times, alarms = data[scenario]
        selected = times >= 30
        selected_times = times[selected]
        selected_alarms = alarms[selected].astype(float)
        rolling = np.convolve(selected_alarms, np.ones(20) / 20, mode="valid")
        rolling_times = selected_times[19:]
        ax.axvspan(0, 30, color="#94A3B8", alpha=0.18, linewidth=0)
        ax.axvspan(110, 130, color="#F59E0B", alpha=0.26, linewidth=0)
        ax.axvspan(130, float(times.max()), color="#ECFDF5", alpha=0.75, linewidth=0)
        ax.plot(rolling_times, rolling, color=color, linewidth=1.9, solid_capstyle="round")
        ax.axvline(120, color="#111827", linestyle="--", linewidth=1.1)
        ax.set_xlim(0, float(times.max()) + 2)
        ax.set_ylim(-0.03, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y")
        ax.set_title(f"{panel})  {scenario.upper()} | 10-s rolling alarm rate", loc="left", pad=8)
        ax.text(0.985, 0.92, f"Stable post: {metrics[scenario]['post']:.0%}", transform=ax.transAxes,
                ha="right", va="top", color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D1D5DB", alpha=0.92))
    for ax in timeline_axes[2:]:
        ax.set_xlabel("Score availability time (s)")
    for ax in [timeline_axes[0], timeline_axes[2]]:
        ax.set_ylabel("Alarm rate")

    x = np.arange(4)
    pre_values = np.array([metrics[s]["pre"] for s in SCENARIOS])
    post_values = np.array([metrics[s]["post"] for s in SCENARIOS])
    width = 0.30
    pre_bars = bar_ax.bar(x - width / 2, pre_values, width, color="#94A3B8", label="Stable pre FPR  60 to <110 s")
    post_bars = bar_ax.bar(x + width / 2, post_values, width, color="#0072B2", label="Stable post TPR  >=130 s")
    bar_ax.set(xticks=x, xticklabels=[s.upper() for s in SCENARIOS], ylim=(0, 1.15), yticks=[0, .25, .5, .75, 1])
    bar_ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    bar_ax.grid(axis="y")
    bar_ax.set_title("e)  Stable-phase operating points\nOS2-4 strong (100% TPR) | OS1 weak (17.9% TPR)", loc="left", pad=8)
    bar_ax.legend(loc="upper left", frameon=False, fontsize=8.1)
    for bars, values in [(pre_bars, pre_values), (post_bars, post_values)]:
        for rectangle, value in zip(bars, values):
            bar_ax.text(rectangle.get_x() + rectangle.get_width() / 2, value + 0.025, f"{value:.0%}",
                        ha="center", va="bottom", fontsize=8.4, fontweight="bold")

    original = summary["results"]["sealed_held"]
    permutation = summary["inference_diagnostics"]["geometry_channels_permutation"]
    zero = summary["inference_diagnostics"]["geometry_channels_zero"]
    ablation_values = np.array([original["alarm_rate"], permutation["alarm_rate"], zero["alarm_rate"]])
    ablation_labels = ["Original", "Geometry\npermutation", "Geometry\nzero"]
    ablation_bars = ablation_ax.bar(np.arange(3), ablation_values, width=0.62, color=["#0072B2", "#F59E0B", "#D55E00"])
    ablation_ax.set(xticks=np.arange(3), xticklabels=ablation_labels, ylim=(0, 1.15), yticks=[0, .25, .5, .75, 1])
    ablation_ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ablation_ax.grid(axis="y")
    ablation_ax.set_title("f)  Held-normal geometry inference ablation\nDeterministic inference only - sealed held normal (n = 119)", loc="left", pad=8)
    for rectangle, value in zip(ablation_bars, ablation_values):
        ablation_ax.text(rectangle.get_x() + rectangle.get_width() / 2, value + 0.025, f"{value:.1%}",
                         ha="center", va="bottom", fontweight="bold")
    ablation_ax.text(0.02, 0.91, f"Frozen threshold = {frozen_threshold:.3f}", transform=ablation_ax.transAxes,
                     va="top", fontsize=8.2, color="#475569")

    fig.suptitle("OAKBAT GCMR - Frozen v5 Evaluation Dashboard", x=0.055, y=0.968, ha="left",
                 fontsize=18.5, fontweight="bold", color="#111827")
    fig.text(0.055, 0.908, "Clean-only normality model | frozen event alarms | score available at window end",
             fontsize=10.8, color="#475569")
    fig.text(0.055, 0.871, "Phases:", fontsize=9.2, fontweight="bold", color="#334155")
    fig.text(0.105, 0.871, "Acquisition <30 s excluded  |  Stable pre 60 to <110 s  |  Guard / transition 110 to <130 s  |  Onset 120 s  |  Stable post >=130 s",
             fontsize=9.2, color="#64748B")
    fig.text(0.055, 0.035, f"Frozen artifact: {BASE.name}  |  Threshold source: {summary['threshold_source']}  |  No recalibration",
             fontsize=8.2, color="#64748B")
    fig.savefig(OUTPUT, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    sources = [SUMMARY, BASE / "cleanStatic_scores.csv"] + [BASE / f"{s}_scores.csv" for s in SCENARIOS]
    lines = [
        "OAKBAT GCMR frozen v5 dashboard SHA256 manifest",
        f"frozen_threshold={frozen_threshold:.17g}",
        f"threshold_source={summary['threshold_source']}",
        "stable_pre_definition=60<=score_available_s<110",
        "stable_post_definition=score_available_s>=130",
        "",
        f"{digest(OUTPUT)}  {OUTPUT.name}",
        f"{digest(Path(__file__))}  {Path(__file__).name}",
        "",
        "Source integrity (read-only):",
    ]
    lines.extend(f"{digest(path)}  {path.name}" for path in sources)
    lines.extend(["", "Recomputed stable phase counts:"])
    for scenario in SCENARIOS:
        value = metrics[scenario]
        lines.append(f"{scenario.upper()}: pre={value['pre_a']}/{value['pre_n']} ({value['pre']:.6f}), post={value['post_a']}/{value['post_n']} ({value['post']:.6f})")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {MANIFEST}")
    print(f"SHA256 {digest(OUTPUT)}")


if __name__ == "__main__":
    main()
