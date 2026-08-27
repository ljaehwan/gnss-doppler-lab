#!/usr/bin/env python3
"""Map the profile-level CGC observability boundary and select RF anchors."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from gnss_doppler_lab.clock_centered_geometry import fit_clock_centered_geometry
from gnss_doppler_lab.correlator_geometry import (
    TemplateDelayEstimator,
    build_complex_template_bank,
    complex_profile_features,
    random_derangement,
    two_path_complex_profile,
)
from gnss_doppler_lab.peak_mixture_law import parse_gps_sdr_sim_los_table


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/cgc_observability_boundary_v1.json"
EXPECTED_TAPS = np.asarray([-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5])


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def verify_record(record: dict[str, str], label: str) -> Path:
    path = repo_path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    observed = sha256(path)
    if observed != record["sha256"]:
        raise ValueError(f"{label} hash mismatch: {observed}")
    return path


def inclusive_axis(low: float, high: float, step: float) -> np.ndarray:
    values = (float(low), float(high), float(step))
    if not all(math.isfinite(value) for value in values) or step <= 0 or high < low:
        raise ValueError("invalid inclusive axis")
    count = (high - low) / step
    if not math.isclose(count, round(count), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("axis range is not divisible by step")
    return np.linspace(low, high, int(round(count)) + 1)


def dominant_secondary(
    geometry_delay_chips: np.ndarray, counterfeit_advantage_db: float
) -> tuple[np.ndarray, float, str]:
    """Express the weaker path relative to the prompt-owning stronger path."""
    delay = np.asarray(geometry_delay_chips, dtype=np.float64)
    advantage = float(counterfeit_advantage_db)
    if not np.isfinite(delay).all() or not math.isfinite(advantage) or advantage == 0:
        raise ValueError("finite nonzero power advantage is required")
    counterfeit_ratio = 10.0 ** (advantage / 20.0)
    if counterfeit_ratio < 1.0:
        return delay.copy(), float(counterfeit_ratio), "authentic_prompt"
    return -delay, float(1.0 / counterfeit_ratio), "counterfeit_prompt"


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "gnss-doppler-lab.cgc-observability-boundary-config" or config.get("schema_version") != 1:
        raise ValueError("unsupported observability-boundary config")
    if config.get("experiment", {}).get("name") != "cgc-observability-boundary-v1":
        raise ValueError("experiment identity drifted")
    frozen = config.get("frozen_candidate", {})
    if frozen.get("residual") != "SSE_LOS_plus_clock / SSE_clock_only" or frozen.get("score") != "negative clock-centered residual":
        raise ValueError("frozen score law drifted")
    template_path = verify_record(frozen["template_config"], "template config")
    verify_record(frozen["correlator_geometry_module"], "correlator geometry module")
    verify_record(frozen["clock_centered_module"], "clock-centered module")
    template = json.loads(template_path.read_text(encoding="utf-8"))
    taps = np.asarray(template.get("correlator", {}).get("tap_offsets_chips", []), dtype=np.float64)
    if not np.array_equal(taps, EXPECTED_TAPS) or template["correlator"].get("prompt_index") != 4:
        raise ValueError("frozen nine-tap layout drifted")
    grid = config.get("grid", {})
    separations = [float(value) for value in grid.get("spatial_separation_m", [])]
    powers = [float(value) for value in grid.get("counterfeit_advantage_db", [])]
    if separations != [5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0]:
        raise ValueError("spatial grid drifted")
    if powers != [-15.0, -9.0, -6.0, -3.0, 3.0, 6.0, 9.0, 15.0] or 0.0 in powers:
        raise ValueError("power grid drifted")
    chip_length = float(grid.get("gps_l1_ca_chip_length_m", 0.0))
    if not math.isclose(chip_length, 299792458.0 / 1.023e6, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("chip length drifted")
    if max(separations) / chip_length > 0.5:
        raise ValueError("spatial grid exceeds delay-template support")
    generator = config.get("generator", {})
    if generator.get("events_per_geometry_per_cell") != 100 or generator.get("seed") != 2026082703:
        raise ValueError("generator size or seed drifted")
    if generator.get("multipath_control") != "exact PRN derangement of the complete complex profile and estimated-delay multiset within every event":
        raise ValueError("matched-control law drifted")
    evaluation = config.get("evaluation", {})
    expected_rule = {
        "minimum_pooled_complex_auc": 0.8,
        "minimum_each_geometry_complex_auc": 0.7,
        "minimum_positive_separation_geometry_count": 5,
        "minimum_complex_delay_sign_accuracy": 0.75,
    }
    if evaluation.get("observable_cell_rule") != expected_rule:
        raise ValueError("observable-cell rule drifted")
    if evaluation.get("minimum_prns") != 8 or evaluation.get("bootstrap_repetitions") != 300:
        raise ValueError("evaluation support drifted")
    boundary = config.get("data_boundary", {})
    if boundary != {
        "allowed_geometry_partition": "train",
        "geometry_source_policy": "reuse and hash-verify only the six LOS tables pinned by the template config",
        "new_rf_generated": False,
        "receiver_tracking_run": False,
        "texbat_accessed": False,
    }:
        raise ValueError("data boundary drifted")
    output_root = repo_path(config["output_root"])
    if output_root != REPO_ROOT / "artifacts/cgc_observability_boundary_v1":
        raise ValueError("output root drifted")
    return {
        "template_path": template_path,
        "template": template,
        "separations": separations,
        "powers": powers,
        "chip_length": chip_length,
        "output_root": output_root,
    }


def build_estimator(template: dict[str, Any]) -> TemplateDelayEstimator:
    specification = template["template_estimator"]
    taps = np.asarray(template["correlator"]["tap_offsets_chips"], dtype=np.float64)
    bank = build_complex_template_bank(
        taps,
        prompt_index=int(template["correlator"]["prompt_index"]),
        delays_chips=inclusive_axis(
            specification["delay_min_chips"], specification["delay_max_chips"],
            specification["delay_step_chips"],
        ),
        centers_chips=inclusive_axis(
            specification["center_min_chips"], specification["center_max_chips"],
            specification["center_step_chips"],
        ),
        amplitude_ratios=inclusive_axis(
            specification["amplitude_min"], specification["amplitude_max"],
            specification["amplitude_step"],
        ),
        phases_rad=np.linspace(
            -np.pi, np.pi, int(specification["phase_count"]), endpoint=False,
        ),
    )
    return TemplateDelayEstimator(bank)


def load_los(template: dict[str, Any], minimum_prns: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    result: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    allowed = template["data_boundary"]["allowed_pair_ids"]
    if template["data_boundary"]["allowed_partition"] != "train" or len(allowed) != 6:
        raise ValueError("template LOS boundary is not the six train geometries")
    for pair_id in allowed:
        record = template["los_sources"][pair_id]
        path = verify_record(record, f"LOS {pair_id}")
        table = parse_gps_sdr_sim_los_table(path.read_text(encoding="utf-8"))
        prns = sorted(table)
        los = np.asarray([table[prn] for prn in prns], dtype=np.float64)
        rank = int(np.linalg.matrix_rank(np.column_stack((-los, np.ones(len(los))))))
        if len(los) < minimum_prns or rank != 4:
            raise ValueError(f"insufficient LOS support: {pair_id}")
        result[pair_id] = los
        provenance[pair_id] = {
            "path": str(path), "sha256": sha256(path), "prns": prns,
            "prn_count": len(prns), "design_rank": rank,
        }
    return result, provenance


def nuisance_design(
    los_count: int, events: int, generator: dict[str, Any], seed: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    vertical = rng.uniform(
        -float(generator["vertical_direction_abs_max"]),
        float(generator["vertical_direction_abs_max"]), events,
    )
    azimuth = rng.uniform(-np.pi, np.pi, events)
    horizontal = np.sqrt(1.0 - vertical**2)
    directions = np.column_stack((horizontal * np.cos(azimuth), horizontal * np.sin(azimuth), vertical))
    centers = np.clip(
        rng.normal(0.0, float(generator["authentic_center_std_chips"]), (events, los_count)),
        -float(generator["authentic_center_abs_max_chips"]),
        float(generator["authentic_center_abs_max_chips"]),
    )
    phases = rng.uniform(
        float(generator["relative_phase_min_rad"]),
        float(generator["relative_phase_max_rad"]), (events, los_count),
    )
    noise_std = rng.uniform(
        float(generator["complex_noise_std_min"]),
        float(generator["complex_noise_std_max"]), (events, los_count),
    )
    noise = noise_std[:, :, None] * (
        rng.normal(size=(events, los_count, len(EXPECTED_TAPS)))
        + 1j * rng.normal(size=(events, los_count, len(EXPECTED_TAPS)))
    )
    permutations = np.stack([random_derangement(los_count, rng) for _ in range(events)])
    return {
        "directions": directions,
        "centers": centers,
        "phases": phases,
        "noise": noise,
        "permutations": permutations,
    }


def paired_bootstrap_auc(
    spoof_scores: np.ndarray, multipath_scores: np.ndarray, repetitions: int, seed: int
) -> tuple[float, float]:
    if spoof_scores.shape != multipath_scores.shape or spoof_scores.ndim != 1:
        raise ValueError("paired bootstrap scores must align")
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    labels = np.r_[np.ones(len(spoof_scores), dtype=int), np.zeros(len(multipath_scores), dtype=int)]
    for index in range(repetitions):
        sample = rng.integers(0, len(spoof_scores), len(spoof_scores))
        values[index] = roc_auc_score(labels, np.r_[spoof_scores[sample], multipath_scores[sample]])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def evaluate_cell(
    separation_m: float,
    advantage_db: float,
    *,
    chip_length_m: float,
    los_sets: dict[str, np.ndarray],
    designs: dict[str, dict[str, np.ndarray]],
    estimator: TemplateDelayEstimator,
    config: dict[str, Any],
    cell_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    separation_chips = float(separation_m / chip_length_m)
    event_rows: list[dict[str, Any]] = []
    spoof_scores: list[float] = []
    multipath_scores: list[float] = []
    oracle_spoof_scores: list[float] = []
    oracle_multipath_scores: list[float] = []
    per_geometry_scores: dict[str, tuple[list[float], list[float]]] = {}
    sign_correct = 0
    sign_eligible = 0
    positive_separations = 0
    geometry_aucs: dict[str, float] = {}
    tracking_owner: str | None = None
    secondary_ratio: float | None = None

    for pair_id, los in los_sets.items():
        design = designs[pair_id]
        displacement = separation_chips * design["directions"]
        geometry_delay = -np.einsum("pd,ed->ep", los, displacement)
        secondary_delay, ratio, owner = dominant_secondary(geometry_delay, advantage_db)
        tracking_owner = owner
        secondary_ratio = ratio
        profiles = np.stack([
            two_path_complex_profile(
                EXPECTED_TAPS,
                authentic_center_chips=float(design["centers"][event, prn]),
                secondary_delay_chips=float(secondary_delay[event, prn]),
                secondary_amplitude_ratio=ratio,
                relative_phase_rad=float(design["phases"][event, prn]),
                complex_noise=design["noise"][event, prn],
            )
            for event in range(len(secondary_delay))
            for prn in range(len(los))
        ]).reshape(len(secondary_delay), len(los), len(EXPECTED_TAPS))
        estimates, distances, _ = estimator.estimate(
            complex_profile_features(profiles.reshape(-1, len(EXPECTED_TAPS)), prompt_index=4)
        )
        estimates = estimates.reshape(len(secondary_delay), len(los))
        distances = distances.reshape(len(secondary_delay), len(los))
        pair_spoof: list[float] = []
        pair_multipath: list[float] = []
        for event in range(len(secondary_delay)):
            permutation = design["permutations"][event]
            spoof_fit = fit_clock_centered_geometry(los, estimates[event])
            multipath_fit = fit_clock_centered_geometry(los, estimates[event, permutation])
            oracle_spoof = fit_clock_centered_geometry(los, secondary_delay[event])
            oracle_multipath = fit_clock_centered_geometry(los, secondary_delay[event, permutation])
            spoof_score = -float(spoof_fit.clock_centered_normalized_residual)
            multipath_score = -float(multipath_fit.clock_centered_normalized_residual)
            oracle_spoof_score = -float(oracle_spoof.clock_centered_normalized_residual)
            oracle_multipath_score = -float(oracle_multipath.clock_centered_normalized_residual)
            pair_spoof.append(spoof_score)
            pair_multipath.append(multipath_score)
            spoof_scores.append(spoof_score)
            multipath_scores.append(multipath_score)
            oracle_spoof_scores.append(oracle_spoof_score)
            oracle_multipath_scores.append(oracle_multipath_score)
            eligible = np.abs(secondary_delay[event]) >= float(config["evaluation"]["delay_sign_min_abs_truth_chips"])
            correct = int(np.sum(np.sign(estimates[event, eligible]) == np.sign(secondary_delay[event, eligible])))
            sign_correct += correct
            sign_eligible += int(eligible.sum())
            realized = np.abs(secondary_delay[event])
            common = {
                "cell_index": cell_index,
                "pair_id": pair_id,
                "event_index": event,
                "event_key": f"{pair_id}-e{event:03d}",
                "spatial_separation_m": separation_m,
                "spatial_separation_chips": separation_chips,
                "counterfeit_advantage_db": advantage_db,
                "tracking_owner": owner,
                "secondary_to_primary_amplitude_ratio": ratio,
                "realized_abs_delay_median_chips": float(np.median(realized)),
                "realized_abs_delay_q75_chips": float(np.quantile(realized, 0.75)),
                "realized_abs_delay_max_chips": float(np.max(realized)),
                "realized_delay_std_chips": float(np.std(secondary_delay[event])),
                "median_template_distance": float(np.median(distances[event])),
            }
            event_rows.extend((
                {
                    **common, "event_class": "coherent_spoof", "label": 1,
                    "clock_centered_residual": float(spoof_fit.clock_centered_normalized_residual),
                    "score": spoof_score,
                    "oracle_score": oracle_spoof_score,
                },
                {
                    **common, "event_class": "matched_independent_multipath", "label": 0,
                    "clock_centered_residual": float(multipath_fit.clock_centered_normalized_residual),
                    "score": multipath_score,
                    "oracle_score": oracle_multipath_score,
                },
            ))
        labels = np.r_[np.ones(len(pair_spoof), dtype=int), np.zeros(len(pair_multipath), dtype=int)]
        geometry_auc = float(roc_auc_score(labels, np.r_[pair_spoof, pair_multipath]))
        geometry_aucs[pair_id] = geometry_auc
        per_geometry_scores[pair_id] = (pair_spoof, pair_multipath)
        if float(np.median(-np.asarray(pair_multipath)) - np.median(-np.asarray(pair_spoof))) > 0:
            positive_separations += 1

    spoof_array = np.asarray(spoof_scores)
    multipath_array = np.asarray(multipath_scores)
    labels = np.r_[np.ones(len(spoof_array), dtype=int), np.zeros(len(multipath_array), dtype=int)]
    pooled_auc = float(roc_auc_score(labels, np.r_[spoof_array, multipath_array]))
    oracle_auc = float(roc_auc_score(labels, np.r_[oracle_spoof_scores, oracle_multipath_scores]))
    low, high = paired_bootstrap_auc(
        spoof_array, multipath_array,
        int(config["evaluation"]["bootstrap_repetitions"]),
        int(config["evaluation"]["bootstrap_seed"]) + cell_index,
    )
    sign_accuracy = float(sign_correct / sign_eligible) if sign_eligible else None
    rules = config["evaluation"]["observable_cell_rule"]
    gates = {
        "pooled_complex_auc": pooled_auc >= float(rules["minimum_pooled_complex_auc"]),
        "each_geometry_complex_auc": min(geometry_aucs.values()) >= float(rules["minimum_each_geometry_complex_auc"]),
        "positive_separation_geometry_count": positive_separations >= int(rules["minimum_positive_separation_geometry_count"]),
        "complex_delay_sign_accuracy": sign_accuracy is not None and sign_accuracy >= float(rules["minimum_complex_delay_sign_accuracy"]),
    }
    cell = {
        "cell_index": cell_index,
        "spatial_separation_m": separation_m,
        "spatial_separation_chips": separation_chips,
        "counterfeit_advantage_db": advantage_db,
        "tracking_owner": tracking_owner,
        "secondary_to_primary_amplitude_ratio": secondary_ratio,
        "paired_event_count": len(spoof_array),
        "pooled_complex_auc": pooled_auc,
        "paired_bootstrap_ci95_low": low,
        "paired_bootstrap_ci95_high": high,
        "minimum_geometry_complex_auc": min(geometry_aucs.values()),
        "maximum_geometry_complex_auc": max(geometry_aucs.values()),
        "positive_separation_geometry_count": positive_separations,
        "geometry_count": len(geometry_aucs),
        "complex_delay_sign_accuracy": sign_accuracy,
        "sign_eligible_delay_count": sign_eligible,
        "oracle_auc": oracle_auc,
        "median_multipath_minus_spoof_residual": float(np.median(-multipath_array) - np.median(-spoof_array)),
        "observable": bool(all(gates.values())),
        **{f"gate_{key}": bool(value) for key, value in gates.items()},
        **{f"auc_{key}": value for key, value in geometry_aucs.items()},
    }
    return event_rows, cell


def boundary_rows(cells: list[dict[str, Any]], powers: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for power in powers:
        ordered = sorted(
            (row for row in cells if float(row["counterfeit_advantage_db"]) == power),
            key=lambda row: float(row["spatial_separation_m"]),
        )
        observable = [row for row in ordered if row["observable"]]
        minimum = float(observable[0]["spatial_separation_m"]) if observable else None
        after = [row for row in ordered if minimum is not None and float(row["spatial_separation_m"]) >= minimum]
        below = [row for row in ordered if minimum is not None and float(row["spatial_separation_m"]) < minimum and not row["observable"]]
        rows.append({
            "counterfeit_advantage_db": power,
            "tracking_owner": ordered[0]["tracking_owner"],
            "secondary_to_primary_amplitude_ratio": ordered[0]["secondary_to_primary_amplitude_ratio"],
            "minimum_observable_separation_m": minimum,
            "minimum_observable_separation_chips": (
                None if not observable else observable[0]["spatial_separation_chips"]
            ),
            "largest_nonobservable_below_boundary_m": (
                None if not below else below[-1]["spatial_separation_m"]
            ),
            "observable_cell_count": len(observable),
            "tested_cell_count": len(ordered),
            "monotone_observable_at_and_above_boundary": bool(after) and all(row["observable"] for row in after),
        })
    return rows


def select_rf_anchors(
    cells: list[dict[str, Any]], boundaries: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    boundary_by_power = {float(row["counterfeit_advantage_db"]): row for row in boundaries}
    separations = [float(value) for value in config["grid"]["spatial_separation_m"]]
    for power in config["evaluation"]["representative_rf_anchor_power_advantages_db"]:
        power = float(power)
        boundary = boundary_by_power[power]["minimum_observable_separation_m"]
        candidates = {float(row["spatial_separation_m"]): row for row in cells if float(row["counterfeit_advantage_db"]) == power}
        if boundary is None:
            nearest = min(candidates.values(), key=lambda row: abs(float(row["pooled_complex_auc"]) - 0.8))
            anchors.append({
                "counterfeit_advantage_db": power,
                "role": "unresolved_nearest_auc_target",
                "spatial_separation_m": nearest["spatial_separation_m"],
                "screen_pooled_complex_auc": nearest["pooled_complex_auc"],
                "screen_observable": nearest["observable"],
            })
            continue
        index = separations.index(float(boundary))
        choices = []
        if index > 0:
            choices.append(("below_boundary", separations[index - 1]))
        choices.append(("at_boundary", separations[index]))
        if index + 1 < len(separations):
            choices.append(("above_boundary", separations[index + 1]))
        for role, separation in choices:
            row = candidates[separation]
            anchors.append({
                "counterfeit_advantage_db": power,
                "role": role,
                "spatial_separation_m": separation,
                "target_offset_enu_m": [0.8 * separation, 0.6 * separation, 0.0],
                "screen_pooled_complex_auc": row["pooled_complex_auc"],
                "screen_minimum_geometry_auc": row["minimum_geometry_complex_auc"],
                "screen_observable": row["observable"],
                "proposed_duration_seconds": 30,
                "proposed_spoof_start_seconds": 10.0,
                "proposed_transition_seconds": 5.0,
                "proposed_power_ramp_seconds": 5.0,
                "proposed_initial_advantage_db": -30.0,
                "proposed_final_advantage_db": power,
            })
    return anchors


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    fields = list(rows[0])
    extras = sorted({key for row in rows for key in row} - set(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields + extras, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def plot_boundary(path: Path, cells: list[dict[str, Any]], boundaries: list[dict[str, Any]], config: dict[str, Any]) -> None:
    separations = [float(value) for value in config["grid"]["spatial_separation_m"]]
    powers = [float(value) for value in config["grid"]["counterfeit_advantage_db"]]
    by_key = {(float(row["counterfeit_advantage_db"]), float(row["spatial_separation_m"])): row for row in cells}
    auc = np.asarray([[by_key[(power, separation)]["pooled_complex_auc"] for separation in separations] for power in powers])
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    image = axes[0].imshow(auc, origin="lower", aspect="auto", vmin=0.5, vmax=1.0, cmap="viridis")
    axes[0].set_xticks(range(len(separations)), [f"{value:g}" for value in separations])
    axes[0].set_yticks(range(len(powers)), [f"{value:+g}" for value in powers])
    axes[0].set_xlabel("Spatial separation (m)")
    axes[0].set_ylabel("Counterfeit advantage (dB)")
    axes[0].set_title("Clock-centered CGC AUC")
    for y, power in enumerate(powers):
        for x, separation in enumerate(separations):
            row = by_key[(power, separation)]
            axes[0].text(x, y, f"{row['pooled_complex_auc']:.2f}{'*' if row['observable'] else ''}", ha="center", va="center", fontsize=7, color="white" if auc[y, x] < 0.78 else "black")
    figure.colorbar(image, ax=axes[0], label="Spoof vs matched multipath AUC")
    boundary_values = [next(row["minimum_observable_separation_m"] for row in boundaries if float(row["counterfeit_advantage_db"]) == power) for power in powers]
    valid = [(power, value) for power, value in zip(powers, boundary_values) if value is not None]
    if valid:
        axes[1].plot([item[0] for item in valid], [item[1] for item in valid], marker="o", linewidth=1.8)
    axes[1].set_xticks(powers, [f"{value:+g}" for value in powers], rotation=45)
    axes[1].set_xlabel("Counterfeit advantage (dB)")
    axes[1].set_ylabel("Minimum observable separation (m)")
    axes[1].set_title("Exploratory observability boundary")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, max(separations) * 1.08)
    figure.suptitle("CGC physical observability screen (* = all cell rules passed)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    context = validate_config(config)
    root = context["output_root"]
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    estimator = build_estimator(context["template"])
    los_sets, los_provenance = load_los(context["template"], int(config["evaluation"]["minimum_prns"]))
    events = int(config["generator"]["events_per_geometry_per_cell"])
    designs = {
        pair_id: nuisance_design(len(los), events, config["generator"], int(config["generator"]["seed"]) + index)
        for index, (pair_id, los) in enumerate(los_sets.items())
    }
    event_rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    cell_index = 0
    for power in context["powers"]:
        for separation in context["separations"]:
            print(f"[boundary] power={power:+g} dB separation={separation:g} m", flush=True)
            rows, cell = evaluate_cell(
                separation, power,
                chip_length_m=context["chip_length"], los_sets=los_sets,
                designs=designs, estimator=estimator, config=config,
                cell_index=cell_index,
            )
            event_rows.extend(rows)
            cells.append(cell)
            cell_index += 1
    boundaries = boundary_rows(cells, context["powers"])
    anchors = select_rf_anchors(cells, boundaries, config)
    event_path = root / "event_scores.csv"
    cell_path = root / "cell_summary.csv"
    boundary_path = root / "power_boundary.csv"
    anchor_path = root / "rf_anchor_plan.json"
    plot_path = root / "observability_boundary.png"
    write_csv(event_path, event_rows)
    write_csv(cell_path, cells)
    write_csv(boundary_path, boundaries)
    write_json(anchor_path, {"schema": "gnss-doppler-lab.cgc-rf-anchor-plan", "schema_version": 1, "anchors": anchors})
    plot_boundary(plot_path, cells, boundaries, config)
    summary = {
        "schema": "gnss-doppler-lab.cgc-observability-boundary-result",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "frozen_candidate": config["frozen_candidate"],
        "los_provenance": los_provenance,
        "grid_cell_count": len(cells),
        "paired_event_count_per_cell": events * len(los_sets),
        "observable_cell_count": sum(bool(row["observable"]) for row in cells),
        "boundary_by_power": boundaries,
        "rf_anchor_plan": anchors,
        "data_boundary": config["data_boundary"],
        "claim_boundary": config["claim_boundary"],
        "artifacts": {
            "event_scores": {"path": str(event_path.resolve()), "sha256": sha256(event_path), "row_count": len(event_rows)},
            "cell_summary": {"path": str(cell_path.resolve()), "sha256": sha256(cell_path), "row_count": len(cells)},
            "power_boundary": {"path": str(boundary_path.resolve()), "sha256": sha256(boundary_path), "row_count": len(boundaries)},
            "rf_anchor_plan": {"path": str(anchor_path.resolve()), "sha256": sha256(anchor_path), "row_count": len(anchors)},
            "boundary_plot": {"path": str(plot_path.resolve()), "sha256": sha256(plot_path)},
        },
    }
    summary_path = root / "summary.json"
    write_json(summary_path, summary)
    print(json.dumps({
        "summary": str(summary_path),
        "observable_cell_count": summary["observable_cell_count"],
        "boundary_by_power": boundaries,
        "rf_anchor_plan": anchors,
    }, indent=2, sort_keys=True))
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config_path = repo_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if args.validate_only:
        print("observability-boundary config, candidate, and six train LOS inputs verified")
        return 0
    run(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
