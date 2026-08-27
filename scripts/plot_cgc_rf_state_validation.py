#!/usr/bin/env python3
"""Plot the held-out multi-geometry CGC receiver-state validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPO_ROOT / "artifacts/cgc_rf_state_validation_v1/summary.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/cgc_rf_state_validation_v1/state_validation_curves.png"


def load_groups(path: Path) -> dict[tuple[str, float], list[dict]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "gnss-doppler-lab.cgc-rf-state-validation-result":
        raise ValueError("unexpected state-validation summary schema")
    groups: dict[tuple[str, float], list[dict]] = {}
    for row in document["condition_summaries"]:
        key = (str(row["geometry_id"]), float(row["final_advantage_db"]))
        groups.setdefault(key, []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: float(row["distance_m"]))
    return groups


def plot(summary: Path, output: Path) -> Path:
    groups = load_groups(summary)
    colors = {"straight": "#2563eb", "sweep": "#dc2626"}
    lines = {-6.0: "-", 3.0: "--"}
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), sharex=True, constrained_layout=True)
    auc_axis, direction_axis, error_axis, edge_axis = axes.flat
    for (geometry, power), rows in sorted(groups.items()):
        distance = np.asarray([row["distance_m"] for row in rows], dtype=float)
        truth = np.asarray([row["distance_chips"] for row in rows], dtype=float)
        recovered = np.asarray([row["median_estimated_displacement_norm_chips"] for row in rows], dtype=float)
        label = f"{geometry}, {power:+g} dB"
        style = dict(color=colors[geometry], linestyle=lines[power], marker="o", linewidth=2.0, label=label)
        auc_axis.plot(distance, [row["serial_bin_auc"] for row in rows], **style)
        direction_axis.plot(distance, [row["median_absolute_direction_cosine"] for row in rows], **style)
        error_axis.plot(distance, 100.0 * (recovered - truth) / truth, **style)
        edge_axis.plot(distance, 100.0 * np.asarray([row["template_delay_edge_fraction"] for row in rows]), **style)

    auc_axis.axhline(0.8, color="#4b5563", linestyle=":", linewidth=1.3)
    direction_axis.axhline(0.85, color="#4b5563", linestyle=":", linewidth=1.3)
    error_axis.axhline(15.0, color="#4b5563", linestyle=":", linewidth=1.3)
    error_axis.axhline(-15.0, color="#4b5563", linestyle=":", linewidth=1.3)
    edge_axis.axhline(10.0, color="#4b5563", linestyle=":", linewidth=1.3)
    auc_axis.set_ylabel("Serial-bin AUC")
    direction_axis.set_ylabel("Absolute direction cosine")
    error_axis.set_ylabel("Displacement norm error (%)")
    edge_axis.set_ylabel("Delay estimates at edge (%)")
    error_axis.set_xlabel("Final carry-off distance (m)")
    edge_axis.set_xlabel("Final carry-off distance (m)")
    auc_axis.set_ylim(0.35, 1.03)
    direction_axis.set_ylim(0.35, 1.03)
    error_axis.set_ylim(-40.0, 115.0)
    edge_axis.set_ylim(0.0, 58.0)
    auc_axis.legend(ncol=2, frameon=False, fontsize=9, loc="lower right")
    figure.suptitle("Held-out receiver/LOS geometry validation (20 m/s carry-off)")
    for axis in axes.flat:
        axis.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.8)
        axis.set_xticks([40, 60, 80, 100, 160, 240])
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
