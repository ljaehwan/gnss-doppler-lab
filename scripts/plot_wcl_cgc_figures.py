#!/usr/bin/env python3
"""Build the two compact, publication-facing figures for the CGC WCL draft.

The script reads only frozen/archived experiment artifacts.  Figure 1 combines
the nine-tap measurement principle with one representative held-out receiver-RF
epoch.  Figure 2 aggregates the preregistered mechanism, the nested-aperture
audit (including the labeled post-hoc seven-tap result), TEXBAT, and
real-multipath results used in the paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnss_doppler_lab.clock_centered_geometry import (  # noqa: E402
    fit_clock_centered_geometry,
)
from gnss_doppler_lab.peak_mixture_law import (  # noqa: E402
    parse_gps_sdr_sim_los_table,
)


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7B3294"
GRAY = "#666666"
LIGHT_GRAY = "#D8D8D8"
CHIP_LENGTH_M = 299_792_458.0 / 1_023_000.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.2,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "figure.dpi": 160,
            "savefig.dpi": 350,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_label(ax: mpl.axes.Axes, label: str) -> None:
    text_method = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_method(
        -0.14,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _load_representative_epoch() -> dict[str, object]:
    pair_id = "fv2-static-01"
    bin_index = 21
    log_path = (
        ROOT
        / "artifacts/simulation_v5_fresh_test_generation_v2/pairs"
        / pair_id
        / "components/authentic-gps-sdr-sim.log"
    )
    delay_path = ROOT / "artifacts/cgc_rf_fresh_test_v2/analysis/delay_estimates.csv"
    config_path = ROOT / "configs/experiments/cgc_rf_fresh_test_v2.json"
    los = parse_gps_sdr_sim_los_table(log_path.read_text(encoding="utf-8"))
    delays = pd.read_csv(delay_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pair = next(item for item in config["pairs"] if item["paired_group_id"] == pair_id)

    scenarios: dict[str, dict[str, object]] = {}
    for scenario in ("carryoff_spoof", "independent_multipath"):
        rows = delays[
            (delays["pair_id"] == pair_id)
            & (delays["scenario"] == scenario)
            & (delays["bin_index"] == bin_index)
        ].copy()
        rows = rows[rows["prn"].isin(los)].sort_values("prn")
        prns = rows["prn"].tolist()
        los_matrix = np.asarray([los[prn] for prn in prns], dtype=float)
        observed = rows["estimated_delay_chips"].to_numpy(dtype=float)
        fit = fit_clock_centered_geometry(los_matrix, observed)
        scenarios[scenario] = {
            "prns": prns,
            "los": los_matrix,
            "observed": observed,
            "fit": fit,
        }

    target_m = np.asarray(pair["spoofing"]["target_offset_enu_m"], dtype=float)
    return {
        "pair_id": pair_id,
        "bin_index": bin_index,
        "target_chips": target_m / CHIP_LENGTH_M,
        "scenarios": scenarios,
        "sources": [log_path, delay_path, config_path],
    }


def _plot_nine_tap_principle(ax: mpl.axes.Axes) -> None:
    taps = np.linspace(-0.5, 0.5, 9)
    dense = np.linspace(-0.58, 0.58, 500)

    def correlation(x: np.ndarray, shift: float) -> np.ndarray:
        return np.clip(1.0 - np.abs(x - shift) / 0.52, 0.0, None)

    clean = correlation(dense, 0.0)
    shifted = correlation(dense, 0.22)
    ax.plot(dense, clean, color=GRAY, linestyle="--", label="Prompt-centered")
    ax.plot(dense, shifted, color=BLUE, label="Delayed replica")
    ax.scatter(taps, correlation(taps, 0.22), color=BLUE, s=16, zorder=3)
    for tap in taps:
        ax.vlines(tap, 0, correlation(np.asarray([tap]), 0.22)[0], color=LIGHT_GRAY, lw=0.45)
    ax.axvline(0.0, color="#999999", lw=0.65, linestyle=":")
    ax.annotate(
        r"signed $\widehat{\delta}_i$",
        xy=(0.22, 1.0),
        xytext=(-0.08, 0.76),
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 0.8},
        color=BLUE,
    )
    ax.set_xlim(-0.58, 0.58)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(taps[::2])
    ax.set_xlabel("Code offset (chips)")
    ax.set_ylabel("Normalized correlation")
    ax.set_title("(a) Nine-tap delay sensor", loc="left", pad=7, fontsize=8.2)
    ax.legend(frameon=False, loc="upper left", fontsize=6.2)
    ax.spines[["top", "right"]].set_visible(False)


def _plot_los_geometry(ax: mpl.axes.Axes, epoch: dict[str, object]) -> None:
    spoof = epoch["scenarios"]["carryoff_spoof"]
    los = spoof["los"]
    observed = spoof["observed"]
    fit = spoof["fit"]
    centered = observed - np.mean(observed)
    maximum = max(0.05, float(np.max(np.abs(centered))))
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)

    theta = np.linspace(0, 2 * np.pi, 240)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta), color="#B0B0B0", lw=0.55)
    for az in np.deg2rad([0, 45, 90, 135]):
        elevation = np.linspace(0, np.pi / 2, 100)
        horizontal = np.cos(elevation)
        ax.plot(
            horizontal * np.sin(az),
            horizontal * np.cos(az),
            np.sin(elevation),
            color="#E0E0E0",
            lw=0.4,
        )
    scatter = ax.scatter(
        los[:, 0],
        los[:, 1],
        los[:, 2],
        c=centered,
        cmap="coolwarm",
        norm=norm,
        s=24,
        edgecolor="black",
        linewidth=0.35,
        depthshade=False,
    )
    for prn, vector in zip(spoof["prns"], los):
        ax.text(vector[0], vector[1], vector[2] + 0.045, prn, fontsize=5.8, ha="center")

    true_vector = np.asarray(epoch["target_chips"], dtype=float)
    estimated_vector = np.asarray(fit.theta[:3], dtype=float)
    true_unit = true_vector / np.linalg.norm(true_vector)
    estimated_unit = estimated_vector / np.linalg.norm(estimated_vector)
    ax.quiver(0, 0, 0, *true_unit, length=0.78, color=ORANGE, linewidth=1.7, arrow_length_ratio=0.11)
    ax.quiver(0, 0, 0, *estimated_unit, length=0.78, color=GREEN, linewidth=1.7, arrow_length_ratio=0.11)
    true_label = true_unit * 0.86 + np.asarray([0.0, -0.08, 0.03])
    estimated_label = estimated_unit * 0.92 + np.asarray([0.0, 0.08, 0.08])
    ax.text(*true_label, r"$\mathbf{d}$", color=ORANGE, fontsize=8.5, fontweight="bold")
    ax.text(
        *estimated_label, r"$\widehat{\mathbf{d}}$", color=GREEN, fontsize=8.5, fontweight="bold"
    )
    ax.set_xlabel("East", labelpad=-8)
    ax.set_ylabel("North", labelpad=-8)
    ax.set_zlabel("")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(0, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1, 1, 0.72))
    ax.view_init(elev=27, azim=-58)
    ax.set_title("(b) Held-out LOS geometry", loc="left", pad=2, fontsize=8.2)
    ax.grid(False)
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    colorbar = ax.figure.colorbar(scatter, ax=ax, shrink=0.56, pad=0.02, aspect=15)
    colorbar.set_label("Delay (chips)", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6)


def _plot_observed_vs_fitted(ax: mpl.axes.Axes, epoch: dict[str, object]) -> None:
    styles = {
        "carryoff_spoof": (BLUE, "o", "Carry-off spoof"),
        "independent_multipath": (ORANGE, "s", "Independent multipath"),
    }
    all_values: list[float] = []
    for name, (color, marker, label) in styles.items():
        item = epoch["scenarios"][name]
        observed = np.asarray(item["observed"], dtype=float) - np.mean(item["observed"])
        predicted = np.asarray(item["fit"].predicted_delays_chips, dtype=float) - np.mean(item["fit"].predicted_delays_chips)
        all_values.extend(observed.tolist())
        all_values.extend(predicted.tolist())
        ax.scatter(
            predicted,
            observed,
            s=23,
            marker=marker,
            color=color if name == "carryoff_spoof" else "white",
            edgecolor=color,
            linewidth=0.8,
            label=f"{'Spoof' if name == 'carryoff_spoof' else 'Multipath'}: $R^2_{{dir}}$={item['fit'].directional_coherence:.2f}",
            zorder=3,
        )
    limit = max(abs(min(all_values)), abs(max(all_values))) * 1.12
    ax.plot([-limit, limit], [-limit, limit], color="#777777", linestyle="--", lw=0.8)
    ax.axhline(0, color="#DDDDDD", lw=0.5)
    ax.axvline(0, color="#DDDDDD", lw=0.5)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Geometry-fitted delay (chips)")
    ax.set_ylabel("Observed delay (chips)")
    ax.set_title("(c) Common-displacement fit", loc="left", pad=7, fontsize=8.2)
    ax.legend(frameon=False, loc="lower right", fontsize=6.2)
    ax.spines[["top", "right"]].set_visible(False)


def build_principle_figure(output_dir: Path) -> tuple[Path, Path, list[Path]]:
    epoch = _load_representative_epoch()
    figure = plt.figure(figsize=(7.16, 2.42), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(0.94, 1.08, 1.0))
    ax0 = figure.add_subplot(grid[0, 0])
    ax1 = figure.add_subplot(grid[0, 1], projection="3d")
    ax2 = figure.add_subplot(grid[0, 2])
    _plot_nine_tap_principle(ax0)
    _plot_los_geometry(ax1, epoch)
    _plot_observed_vs_fitted(ax2, epoch)
    pdf = output_dir / "wcl_cgc_principle.pdf"
    png = output_dir / "wcl_cgc_principle.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, bbox_inches="tight")
    plt.close(figure)
    return pdf, png, list(epoch["sources"])


def _plot_auc_ablation(ax: mpl.axes.Axes, train: dict[str, object]) -> None:
    metrics = train["metrics"]
    labels = ["Width", "Magnitude\n+ LOS", "Complex\n+ LOS", "Oracle\ndelay"]
    values = [
        metrics["single_prn_width_auc"],
        metrics["magnitude_geometry_auc"]["estimate"],
        metrics["complex_geometry_auc"]["estimate"],
        metrics["oracle_geometry_auc"]["estimate"],
    ]
    intervals = [
        (0.5, 0.5),
        tuple(metrics["magnitude_geometry_auc"]["ci95"]),
        tuple(metrics["complex_geometry_auc"]["ci95"]),
        tuple(metrics["oracle_geometry_auc"]["ci95"]),
    ]
    lower = np.asarray(values) - np.asarray([item[0] for item in intervals])
    upper = np.asarray([item[1] for item in intervals]) - np.asarray(values)
    x = np.arange(len(values))
    colors = [GRAY, ORANGE, BLUE, GREEN]
    ax.bar(x, values, color=colors, width=0.64, edgecolor="black", linewidth=0.45)
    ax.errorbar(x, values, yerr=np.vstack([lower, upper]), fmt="none", ecolor="black", capsize=2.2, lw=0.7)
    ax.axhline(0.5, color="#777777", linestyle="--", lw=0.7)
    for index, value in enumerate(values):
        ax.text(index, value + 0.028, f"{value:.3f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.44, 1.07)
    ax.set_ylabel("AUC")
    ax.set_title("Matched-profile mechanism audit")
    ax.spines[["top", "right"]].set_visible(False)
    _panel_label(ax, "(a)")


def _plot_aperture(ax: mpl.axes.Axes, aperture: pd.DataFrame) -> None:
    subset = aperture[aperture["distance_m"] == 240.0]
    taps = np.asarray([3, 5, 7, 9])
    series = [
        ("median_absolute_direction_cosine", BLUE, "Direction cosine"),
        ("recovery", ORANGE, "Recovered / true displacement"),
    ]
    subset = subset.copy()
    subset["recovery"] = subset["median_estimated_displacement_norm_chips"] / subset["distance_chips"]
    for column, color, label in series:
        medians = []
        lower = []
        upper = []
        for tap in taps:
            values = subset[subset["aperture_taps"] == tap][column].to_numpy(dtype=float)
            q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
            medians.append(median)
            lower.append(median - q25)
            upper.append(q75 - median)
        ax.errorbar(
            taps,
            medians,
            yerr=np.vstack([lower, upper]),
            color=color,
            marker="o" if column.startswith("median") else "s",
            capsize=2.3,
            label=label,
        )
        for x, y in zip(taps, medians):
            ax.text(x, y + (0.035 if column.startswith("median") else -0.075), f"{y:.2f}", ha="center", color=color, fontsize=6.6)
    ax.set_xticks(taps, ["3", "5", "7*", "9"])
    ax.set_xlim(2.2, 9.6)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Correlator taps")
    ax.set_ylabel("Median across 10 conditions")
    ax.set_title("Finite-aperture behavior at 240 m")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    _panel_label(ax, "(b)")


def _plot_texbat_pre_post(ax: mpl.axes.Axes, texbat: dict[str, object]) -> None:
    scenarios = texbat["scenarios"]
    x = np.asarray([0, 1])
    for index, item in enumerate(scenarios):
        values = [item["stable_pre_median_residual"], item["stable_post_median_residual"]]
        color = [BLUE, ORANGE, GREEN][index]
        ax.plot(x, values, color=color, marker="o", label=item["scenario"].upper())
        ax.text(1.035, values[1], f"{item['secondary_serial_bin_auc']:.2f}", color=color, va="center", fontsize=6.6)
    ax.set_xticks(x, ["Pre-onset", "Stable post"])
    ax.set_xlim(-0.16, 1.32)
    ax.set_ylim(0.42, 0.76)
    ax.set_ylabel("Clock-centered residual")
    ax.set_title("Real TEXBAT spoof recordings")
    ax.text(1.03, 0.745, "AUC", fontsize=6.3, color=GRAY)
    ax.annotate("more spoof-like", xy=(0.03, 0.44), xytext=(0.03, 0.49), arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=6.5)
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    _panel_label(ax, "(c)")


def _plot_openif_false_alarm(ax: mpl.axes.Axes, openif: dict[str, object]) -> None:
    result = openif["s1_result"]
    values = [
        100.0 * result["legacy_unadjusted_persistent_spoof_alarm_rate"],
        100.0 * result["persistent_spoof_alarm_rate"],
    ]
    x = np.arange(2)
    ax.bar(x, values, color=[GRAY, BLUE], width=0.62, edgecolor="black", linewidth=0.45)
    ax.axhline(5.0, color=ORANGE, linestyle="--", lw=0.9, label="Frozen 5% gate")
    for index, value in enumerate(values):
        ax.text(index, value + 0.55, f"{value:.2f}%", ha="center", fontsize=7)
    ax.set_xticks(x, ["Unadjusted", "Partial-F\nsupport corrected"])
    ax.set_ylim(0, 15.2)
    ax.set_ylabel("Persistent false alarms (%)")
    ax.set_title("Real GNSS-OpenIF S1 multipath")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    _panel_label(ax, "(d)")


def build_evidence_figure(output_dir: Path) -> tuple[Path, Path, list[Path]]:
    train_path = ROOT / "docs/results/correlator_geometry_identifiability_train_v1_summary.json"
    aperture_path = ROOT / "artifacts/cgc_rf_ga_v1/condition_aperture_summary.csv"
    seven_tap_path = ROOT / "docs/results/cgc_rf_7tap_aperture_audit_v1_condition_summary.csv"
    texbat_path = ROOT / "docs/results/cgc_texbat_external_v1_summary.json"
    openif_path = ROOT / "docs/results/gnss_openif_s1_real_multipath_v1_summary.json"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    aperture = pd.read_csv(aperture_path)
    seven_tap = pd.read_csv(seven_tap_path)
    aperture = pd.concat([aperture, seven_tap], ignore_index=True)
    texbat = json.loads(texbat_path.read_text(encoding="utf-8"))
    openif = json.loads(openif_path.read_text(encoding="utf-8"))

    figure, axes = plt.subplots(2, 2, figsize=(7.16, 4.35), constrained_layout=True)
    _plot_auc_ablation(axes[0, 0], train)
    _plot_aperture(axes[0, 1], aperture)
    _plot_texbat_pre_post(axes[1, 0], texbat)
    _plot_openif_false_alarm(axes[1, 1], openif)
    pdf = output_dir / "wcl_cgc_evidence.pdf"
    png = output_dir / "wcl_cgc_evidence.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, bbox_inches="tight")
    plt.close(figure)
    return pdf, png, [train_path, aperture_path, seven_tap_path, texbat_path, openif_path]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/results/figures/wcl",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()

    principle_pdf, principle_png, principle_sources = build_principle_figure(output_dir)
    evidence_pdf, evidence_png, evidence_sources = build_evidence_figure(output_dir)
    metadata = {
        "schema": "gnss-doppler-lab.wcl-cgc-figure-manifest",
        "schema_version": 1,
        "figures": {
            "wcl_cgc_principle": {
                "pdf": principle_pdf.name,
                "png": principle_png.name,
                "representative_pair": "fv2-static-01",
                "representative_bin_index": 21,
                "sources": {str(path.relative_to(ROOT)): _sha256(path) for path in principle_sources},
            },
            "wcl_cgc_evidence": {
                "pdf": evidence_pdf.name,
                "png": evidence_png.name,
                "sources": {str(path.relative_to(ROOT)): _sha256(path) for path in evidence_sources},
            },
        },
    }
    manifest_path = output_dir / "wcl_cgc_figure_manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "files": [path.name for path in (principle_pdf, principle_png, evidence_pdf, evidence_png, manifest_path)]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
