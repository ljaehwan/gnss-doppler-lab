#!/usr/bin/env python3
"""Plot the exploratory CGC receiver transfer curve."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPO_ROOT / "artifacts/cgc_rf_transfer_sweep_v1/summary.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/cgc_rf_transfer_sweep_v1/transfer_curve.png"


def load_curves(path: Path) -> dict[float, list[dict]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "gnss-doppler-lab.cgc-rf-transfer-sweep-result":
        raise ValueError("unexpected transfer summary schema")
    curves: dict[float, list[dict]] = {}
    for row in document["condition_summaries"]:
        curves.setdefault(float(row["final_advantage_db"]), []).append(row)
    for rows in curves.values():
        rows.sort(key=lambda row: float(row["distance_m"]))
    return curves


def plot(summary: Path, output: Path) -> Path:
    curves = load_curves(summary)
    styles = {-6.0: ("#2563eb", "-6 dB"), 3.0: ("#dc2626", "+3 dB")}
    figure, axes = plt.subplots(3, 1, figsize=(8.2, 9.2), sharex=True, constrained_layout=True)
    for power, rows in sorted(curves.items()):
        color, label = styles[power]
        distance = np.asarray([row["distance_m"] for row in rows], dtype=float)
        auc = np.asarray([row["serial_bin_auc"] for row in rows], dtype=float)
        truth = np.asarray([row["distance_chips"] for row in rows], dtype=float)
        recovered = np.asarray([row["median_estimated_displacement_norm_chips"] for row in rows], dtype=float)
        edge = np.asarray([row["template_delay_edge_fraction"] for row in rows], dtype=float)
        axes[0].plot(distance, auc, marker="o", linewidth=2.0, color=color, label=label)
        axes[1].plot(distance, recovered, marker="o", linewidth=2.0, color=color, label=label)
        axes[2].plot(distance, 100.0 * edge, marker="o", linewidth=2.0, color=color, label=label)
        if power == -6.0:
            axes[1].plot(distance, truth, linestyle="--", linewidth=1.7, color="#111827", label="physical truth")
    axes[0].axhline(0.8, color="#4b5563", linestyle="--", linewidth=1.2, label="descriptive AUC 0.8")
    axes[0].set_ylabel("Serial-bin AUC")
    axes[0].set_ylim(0.0, 1.04)
    axes[0].set_title("Fixed-receiver CGC transfer curve (20 m/s carry-off)")
    axes[0].legend(ncol=3, frameon=False, loc="lower right")
    axes[1].set_ylabel("Displacement norm (chip)")
    axes[1].legend(ncol=3, frameon=False, loc="upper left")
    axes[2].set_ylabel("Delay estimates at edge (%)")
    axes[2].set_xlabel("Final carry-off distance (m)")
    axes[2].set_ylim(0.0, 36.0)
    axes[2].legend(frameon=False, loc="upper left")
    for axis in axes:
        axis.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.8)
        axis.set_xticks([20, 40, 60, 80, 100, 120, 160, 200, 240])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(plot(args.summary.resolve(), args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
