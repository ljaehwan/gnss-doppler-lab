#!/usr/bin/env python3
"""Render the development locked-phase causal sweep as a compact vector plot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "artifacts/cgc_locked_phase_sweep_dev_v1/summary.json"
DEFAULT_OUTPUT = ROOT / "artifacts/cgc_locked_phase_sweep_dev_v1/locked_phase_sweep.svg"


def render(summary_path: Path, output: Path) -> None:
    document = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = sorted(document["rows"], key=lambda row: float(row["phase_offset_deg"]))
    phase = np.asarray([float(row["phase_offset_deg"]) for row in rows])
    quadrature = np.asarray([float(row["median_quadrature_fraction"]) for row in rows])
    agreement = np.asarray([float(row["median_truth_direction_r2"]) for row in rows])
    p_value = np.asarray([float(row["median_partial_f_p_value"]) for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45), constrained_layout=True)
    style = {"color": "#0072B2", "marker": "o", "linewidth": 1.5, "markersize": 4.5}
    axes[0].plot(phase, quadrature, **style)
    axes[0].set_ylabel("Quadrature fraction")
    axes[0].set_ylim(0.0, 0.34)
    axes[1].plot(phase, agreement, **style)
    axes[1].set_ylabel(r"Delay-geometry agreement $R^2$")
    axes[1].set_ylim(0.0, 1.04)
    axes[2].semilogy(phase, p_value, **style)
    axes[2].axhline(0.06028418845288192, color="#D55E00", linestyle="--", linewidth=1.0, label="alarm threshold")
    axes[2].set_ylabel(r"Median Partial-$F$ $p$-value")
    axes[2].set_ylim(1e-5, 0.4)
    axes[2].legend(frameon=False, fontsize=7, loc="lower right")
    for label, axis in zip(("(a)", "(b)", "(c)"), axes):
        axis.set_xlabel("Global spoof phase offset (deg)")
        axis.set_xticks(phase)
        axis.grid(True, color="0.88", linewidth=0.6)
        axis.text(0.02, 0.97, label, transform=axis.transAxes, va="top", ha="left", fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg")
    output.write_text("\n".join(line.rstrip() for line in output.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.summary.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
