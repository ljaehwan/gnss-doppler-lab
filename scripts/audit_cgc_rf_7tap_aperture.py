#!/usr/bin/env python3
"""Post-hoc seven-tap audit on the frozen CGC geometry/aperture campaign."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_cgc_rf_geometry_aperture_validation as ga  # noqa: E402
import run_cgc_rf_transfer_sweep as transfer  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/cgc_rf_geometry_aperture_validation_v1.json"
DEFAULT_FROZEN = REPO_ROOT / "artifacts/cgc_rf_ga_v1"
DEFAULT_OUTPUT = REPO_ROOT / "docs/results/cgc_rf_7tap_aperture_audit_v1_summary.json"
SEVEN_TAP_INDICES = (1, 2, 3, 4, 5, 6, 7)
SEVEN_TAP_OFFSETS = tuple(ga.FULL_TAPS[index] for index in SEVEN_TAP_INDICES)
METRICS = (
    "serial_bin_auc",
    "median_absolute_direction_cosine",
    "median_estimated_displacement_norm_chips",
    "template_delay_edge_fraction",
)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def seven_tap_estimator(controlled: dict[str, Any]) -> Any:
    modified = deepcopy(controlled)
    modified["correlator"]["tap_offsets_chips"] = list(SEVEN_TAP_OFFSETS)
    modified["correlator"]["prompt_index"] = 3
    return ga.pilot._estimator(modified)


def verified_manifest(record: dict[str, Any], label: str) -> Path:
    path = Path(record["path"]).resolve()
    if not path.is_file() or ga.sha256(path) != record["sha256"]:
        raise ValueError(f"frozen receiver manifest mismatch: {label}")
    return path


def numeric(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def summarize(rows: list[dict[str, Any]], taps: int, distance_m: float) -> dict[str, Any]:
    selected = [
        row for row in rows
        if int(float(row["aperture_taps"])) == taps
        and float(row["distance_m"]) == distance_m
    ]
    if len(selected) != len(ga.GEOMETRY_IDS) * len(ga.POWERS_DB):
        raise ValueError(f"incomplete {taps}-tap grid at {distance_m:g} m")
    ratios = [
        numeric(row, "median_estimated_displacement_norm_chips")
        / numeric(row, "distance_chips")
        for row in selected
    ]
    return {
        "condition_count": len(selected),
        "median_serial_bin_auc": statistics.median(numeric(row, "serial_bin_auc") for row in selected),
        "median_absolute_direction_cosine": statistics.median(
            numeric(row, "median_absolute_direction_cosine") for row in selected
        ),
        "median_recovered_to_true_displacement": statistics.median(ratios),
        "median_absolute_relative_displacement_error": statistics.median(
            abs(ratio - 1.0) for ratio in ratios
        ),
        "median_template_delay_edge_fraction": statistics.median(
            numeric(row, "template_delay_edge_fraction") for row in selected
        ),
        "direction_iqr": [
            float(value) for value in np.percentile(
                [numeric(row, "median_absolute_direction_cosine") for row in selected],
                [25, 75],
            )
        ],
        "recovered_to_true_iqr": [float(value) for value in np.percentile(ratios, [25, 75])],
    }


def paired_7_vs_9(rows: list[dict[str, Any]], distance_m: float) -> dict[str, Any]:
    selected = [row for row in rows if float(row["distance_m"]) == distance_m]
    keyed = {
        (str(row["condition_id"]), int(float(row["aperture_taps"]))): row
        for row in selected
        if int(float(row["aperture_taps"])) in {7, 9}
    }
    conditions = sorted({condition for condition, taps in keyed if taps == 7})
    if len(conditions) != len(ga.GEOMETRY_IDS) * len(ga.POWERS_DB):
        raise ValueError(f"incomplete paired 7-vs-9 grid at {distance_m:g} m")
    deltas: dict[str, list[float]] = {
        "auc_7_minus_9": [],
        "direction_7_minus_9": [],
        "absolute_error_7_minus_9": [],
    }
    wins = {"auc": 0, "direction": 0, "absolute_error": 0}
    for condition in conditions:
        row7, row9 = keyed[(condition, 7)], keyed[(condition, 9)]
        auc_delta = numeric(row7, "serial_bin_auc") - numeric(row9, "serial_bin_auc")
        direction_delta = (
            numeric(row7, "median_absolute_direction_cosine")
            - numeric(row9, "median_absolute_direction_cosine")
        )
        error7 = abs(
            numeric(row7, "median_estimated_displacement_norm_chips")
            / numeric(row7, "distance_chips") - 1.0
        )
        error9 = abs(
            numeric(row9, "median_estimated_displacement_norm_chips")
            / numeric(row9, "distance_chips") - 1.0
        )
        error_delta = error7 - error9
        deltas["auc_7_minus_9"].append(auc_delta)
        deltas["direction_7_minus_9"].append(direction_delta)
        deltas["absolute_error_7_minus_9"].append(error_delta)
        wins["auc"] += int(auc_delta > 0)
        wins["direction"] += int(direction_delta > 0)
        wins["absolute_error"] += int(error_delta < 0)
    return {
        "condition_count": len(conditions),
        "seven_tap_win_count": wins,
        "median_deltas": {key: statistics.median(values) for key, values in deltas.items()},
        "ties_are_not_wins": True,
    }


def run(config_path: Path, frozen_root: Path, output_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    context = ga.validate_config(config)
    if context["output_root"].resolve() != frozen_root.resolve():
        raise ValueError("config output root does not match the frozen campaign")
    original_summary = frozen_root / "summary.json"
    original_table = frozen_root / "condition_aperture_summary.csv"
    if not original_summary.is_file() or not original_table.is_file():
        raise FileNotFoundError("frozen 3/5/9-tap summary is incomplete")

    ga.APERTURE_INDICES[7] = SEVEN_TAP_INDICES
    estimator = seven_tap_estimator(context["controlled"])
    seven_rows: list[dict[str, Any]] = []
    receiver_hashes: list[dict[str, str]] = []
    specs = ga.condition_specs()
    for geometry_id in ga.GEOMETRY_IDS:
        base = context["contexts"][geometry_id]
        multipath_manifest = (
            frozen_root / "geometries" / geometry_id / "common_multipath"
            / "receiver" / f"ga-mp-{geometry_id}" / "manifest.json"
        )
        if not multipath_manifest.is_file():
            raise FileNotFoundError(multipath_manifest)
        _, multipath_geometry = ga.analyze_stream(
            "independent_multipath", multipath_manifest, estimator,
            base["los"], config, 7,
        )
        receiver_hashes.append({"path": str(multipath_manifest), "sha256": ga.sha256(multipath_manifest)})
        for spec in (row for row in specs if row["geometry_id"] == geometry_id):
            result_path = (
                frozen_root / "geometries" / geometry_id / "conditions"
                / spec["condition_id"] / "condition_result.json"
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            receiver_manifest = verified_manifest(result["receiver_manifest"], spec["condition_id"])
            delays, geometry = ga.analyze_stream(
                spec["condition_id"], receiver_manifest, estimator,
                base["los"], config, 7,
            )
            summary = transfer.summarize_condition(
                spec, delays, geometry, multipath_geometry, config
            )
            seven_rows.append({**summary, "aperture_taps": 7})
            receiver_hashes.append({"path": str(receiver_manifest), "sha256": ga.sha256(receiver_manifest)})

    if len(seven_rows) != len(specs):
        raise ValueError("seven-tap audit did not reproduce the complete condition grid")
    seven_csv = output_path.with_name(output_path.stem.replace("_summary", "_condition_summary") + ".csv")
    write_csv(seven_csv, seven_rows)
    existing = read_csv(original_table)
    combined = [*existing, *seven_rows]
    aggregates = {
        str(taps): {
            str(int(distance)): summarize(combined, taps, distance)
            for distance in (100.0, 240.0)
        }
        for taps in (3, 5, 7, 9)
    }
    document = {
        "schema": "gnss-doppler-lab.cgc-rf-7tap-aperture-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "post-hoc descriptive aperture audit; the preregistered nine-tap primary is unchanged",
        "subset": {
            "policy": "central subset of the same raw nine-tap receiver dumps",
            "indices": list(SEVEN_TAP_INDICES),
            "offsets_chips": list(SEVEN_TAP_OFFSETS),
            "tracking_rerun": False,
        },
        "inputs": {
            "config": {"path": str(config_path), "sha256": ga.sha256(config_path)},
            "frozen_summary": {"path": str(original_summary), "sha256": ga.sha256(original_summary)},
            "frozen_condition_table": {"path": str(original_table), "sha256": ga.sha256(original_table)},
            "receiver_manifest_count": len(receiver_hashes),
            "receiver_manifest_set_sha256": hashlib.sha256(
                json.dumps(receiver_hashes, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
        "seven_tap_condition_count": len(seven_rows),
        "aggregates": aggregates,
        "paired_7_vs_9": {
            str(int(distance)): paired_7_vs_9(combined, distance)
            for distance in (100.0, 240.0)
        },
        "artifact": {
            "path": str(seven_csv),
            "sha256": ga.sha256(seven_csv),
            "row_count": len(seven_rows),
        },
    }
    write_json(output_path, document)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frozen-root", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.frozen_root.resolve(), args.output.resolve())
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
