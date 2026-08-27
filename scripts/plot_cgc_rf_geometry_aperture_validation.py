#!/usr/bin/env python3
"""Render the frozen five-geometry CGC state and aperture result."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "artifacts/cgc_rf_ga_v1/condition_aperture_summary.csv"
DEFAULT_OUTPUT = REPO_ROOT / "docs/results/figures/cgc_rf_geometry_aperture_validation_v1.png"
GEOMETRIES = ("denver-static", "seoul-static", "tokyo-straight", "london-circle", "sydney-sweep")
COLORS = {
    "denver-static": "#3366cc",
    "seoul-static": "#109618",
    "tokyo-straight": "#dc3912",
    "london-circle": "#990099",
    "sydney-sweep": "#ff9900",
}


def load_rows(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float | int | str] = dict(raw)
            for key in (
                "distance_m", "final_advantage_db", "serial_bin_auc",
                "median_absolute_direction_cosine", "distance_chips",
                "median_estimated_displacement_norm_chips", "template_delay_edge_fraction",
            ):
                row[key] = float(raw[key])
            row["aperture_taps"] = int(raw["aperture_taps"])
            rows.append(row)
    return rows


def render(input_path: Path, output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows(input_path)
    if len(rows) != 120:
        raise ValueError("expected the complete 120-row condition-aperture grid")
    distances = (40.0, 60.0, 100.0, 240.0)
    taps = (3, 5, 9)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

    ax = axes[0]
    primary = [row for row in rows if row["aperture_taps"] == 9]
    for geometry in GEOMETRIES:
        for power, style in ((-6.0, "--"), (3.0, "-")):
            selected = {
                row["distance_m"]: row for row in primary
                if row["geometry_id"] == geometry and row["final_advantage_db"] == power
            }
            ax.plot(
                distances, [selected[distance]["serial_bin_auc"] for distance in distances],
                color=COLORS[geometry], linestyle=style, marker="o", linewidth=1.4,
                alpha=0.9, label=f"{geometry}, {power:+g} dB",
            )
    ax.axhline(0.8, color="black", linestyle=":", linewidth=1.2, label="AUC gate")
    ax.set_title("(a) Geometry-dependent state boundary")
    ax.set_xlabel("Pull-off distance (m)")
    ax.set_ylabel("Serial-bin AUC (9 taps)")
    ax.set_ylim(0.35, 1.03)
    ax.set_xticks(distances)
    ax.grid(alpha=0.25)

    at240 = [row for row in rows if row["distance_m"] == 240.0]
    truth = float(at240[0]["distance_chips"])
    recovered_fraction, absolute_error, edge_fraction = [], [], []
    auc_median, direction_median = [], []
    for tap_count in taps:
        selected = [row for row in at240 if row["aperture_taps"] == tap_count]
        norms = [float(row["median_estimated_displacement_norm_chips"]) for row in selected]
        errors = [abs(norm / truth - 1.0) for norm in norms]
        recovered_fraction.append(statistics.median(norms) / truth)
        absolute_error.append(statistics.median(errors))
        edge_fraction.append(statistics.median(float(row["template_delay_edge_fraction"]) for row in selected))
        auc_median.append(statistics.median(float(row["serial_bin_auc"]) for row in selected))
        direction_median.append(statistics.median(float(row["median_absolute_direction_cosine"]) for row in selected))

    ax = axes[1]
    ax.plot(taps, recovered_fraction, marker="o", linewidth=2.2, label="Recovered / true distance")
    ax.plot(taps, absolute_error, marker="s", linewidth=2.2, label="Absolute relative error")
    ax.plot(taps, edge_fraction, marker="^", linewidth=2.2, label="Template-edge fraction")
    ax.set_title("(b) 240 m aperture mechanism")
    ax.set_xlabel("Central complex correlator taps")
    ax.set_ylabel("Median over 10 geometry-power groups")
    ax.set_xticks(taps)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    ax = axes[2]
    for tap_count, x in zip(taps, range(len(taps))):
        selected = [row for row in at240 if row["aperture_taps"] == tap_count]
        values = [float(row["serial_bin_auc"]) for row in selected]
        offsets = [x + (index - 4.5) * 0.025 for index in range(10)]
        ax.scatter(offsets, values, color="#777777", alpha=0.75, s=24)
    ax.plot(range(3), auc_median, marker="o", linewidth=2.2, color="#3366cc", label="Median AUC")
    ax.plot(range(3), direction_median, marker="s", linewidth=2.2, color="#109618", label="Median direction cosine")
    ax.axhline(0.8, color="black", linestyle=":", linewidth=1.2, label="AUC gate")
    ax.set_title("(c) Detection and direction at 240 m")
    ax.set_xlabel("Central complex correlator taps")
    ax.set_ylabel("Metric value")
    ax.set_xticks(range(3), [str(value) for value in taps])
    ax.set_ylim(0.35, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, fontsize=7.5)
    fig.suptitle("Five-geometry CGC receiver-RF validation and same-stream aperture intervention", fontsize=13)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(render(args.input.resolve(), args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
