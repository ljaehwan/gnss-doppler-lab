#!/usr/bin/env python3
"""Audit CGC finite-support behavior across nested seven-to-ten PRN subsets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
for search_root in (ROOT / "scripts", ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import audit_cgc_seven_prn_support as seven  # noqa: E402
import run_cgc_real_detection as frozen  # noqa: E402
from run_gcmr_texbat_external import preflight_ds4_alternate  # noqa: E402
from gnss_doppler_lab.clock_centered_geometry import fit_clock_centered_geometry  # noqa: E402
from gnss_doppler_lab.gcmr_geometry import (  # noqa: E402
    ephemeris_health_selection,
    parse_gnss_sdr_gps_ephemeris_xml,
    satellite_observation,
)


DEFAULT_CONFIG = ROOT / "configs/experiments/cgc_prn_support_sweep_v1.json"


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def verify(path: str | Path, expected: str, label: str) -> Path:
    target = resolve(path)
    if not target.is_file():
        raise FileNotFoundError(f"{label} missing: {target}")
    observed = seven.sha256(target)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {observed} != {expected}")
    return target


def stable_subset(
    entries: list[dict[str, Any]], *, scenario: str, trial: int, support: int,
) -> list[dict[str, Any]]:
    """Return nested deterministic subsets: the N set contains the N-1 set."""
    if support < 5 or len(entries) < support:
        raise ValueError("requested support is unavailable")

    def rank(row: dict[str, Any]) -> str:
        text = f"{scenario}:{trial}:{int(row['prn'])}".encode()
        return hashlib.sha256(text).hexdigest()

    return sorted(entries, key=rank)[:support]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git_release() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    if status.stdout.strip():
        raise ValueError("support sweep must run from a clean committed worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _apply_persistence(
    rows: list[dict[str, Any]], *, window: int, required: int,
) -> None:
    for region in sorted({str(row["region"]) for row in rows}):
        selected = sorted(
            (row for row in rows if row["region"] == region),
            key=lambda row: int(row["bin_index"]),
        )
        bins = np.asarray([row["bin_index"] for row in selected], dtype=np.int64)
        legacy = frozen.persistent_alarm(
            np.asarray([row["legacy_raw_alarm"] for row in selected], dtype=bool),
            bins, window=window, required=required,
        )
        normalized = frozen.persistent_alarm(
            np.asarray([row["partial_f_raw_alarm"] for row in selected], dtype=bool),
            bins, window=window, required=required,
        )
        for row, old_value, new_value in zip(selected, legacy, normalized):
            row["legacy_persistent_alarm"] = bool(old_value)
            row["partial_f_persistent_alarm"] = bool(new_value)


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        raise ValueError("cannot calculate rate from empty rows")
    return float(np.mean([bool(row[key]) for row in rows]))


def _median(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        raise ValueError("cannot calculate median from empty rows")
    return float(np.median([float(row[key]) for row in rows]))


def _delay(
    rows: list[dict[str, Any]], key: str, attack_onset_s: float,
) -> float | None:
    detections = [row for row in rows if bool(row[key])]
    if not detections:
        return None
    return float(min(int(row["bin_index"]) + 1 for row in detections) - attack_onset_s)


def _range(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize empty values")
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
    }


def _primary_geometry(
    resolved: dict[str, Any],
    groups: dict[tuple[str, int], list[dict[str, Any]]],
    scenario: str,
) -> tuple[
    dict[int, Any],
    dict[tuple[int, int], tuple[float, float, float]],
]:
    sources = {source["name"]: source for source in resolved["sources"]}
    if scenario not in sources:
        raise ValueError(f"primary scenario absent from frozen config: {scenario}")
    source = sources[scenario]
    ephemerides = parse_gnss_sdr_gps_ephemeris_xml(source["paths"]["ephemeris"])
    tracked = {
        int(row["prn"])
        for (name, _), rows in groups.items() if name == scenario
        for row in rows
    }
    preflight = preflight_ds4_alternate(
        source["paths"]["nmea"].parent,
        ephemerides,
        configured_tow0_s=float(source["tow0_s"]),
        tracked_prns=tracked,
        min_prns=7,
    )
    healthy, _ = ephemeris_health_selection(
        ephemerides, tracked_prns=tracked, min_prns=7
    )
    receiver_ecef = preflight["receiver_position_contract"]["ecef"]
    los_by_key: dict[tuple[int, int], tuple[float, float, float]] = {}
    for (name, bin_index), entries in groups.items():
        if name != scenario:
            continue
        tow = (float(source["tow0_s"]) + bin_index + 0.5) % 604800.0
        for row in entries:
            prn = int(row["prn"])
            if prn in healthy:
                los_by_key[(bin_index, prn)] = satellite_observation(
                    receiver_ecef, healthy[prn], tow
                ).los_ecef
    return healthy, los_by_key


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "gnss-doppler-lab.cgc-prn-support-sweep-config":
        raise ValueError("unsupported support-sweep config schema")
    release_commit = _git_release()
    inputs = config["inputs"]
    frozen_config_path = verify(
        inputs["frozen_config"], inputs["frozen_config_sha256"], "frozen config"
    )
    frozen_summary_path = verify(
        inputs["frozen_summary"], inputs["frozen_summary_sha256"], "frozen summary"
    )
    partial_audit_path = verify(
        inputs["partial_f_audit"], inputs["partial_f_audit_sha256"],
        "partial-F audit",
    )
    frozen_config = json.loads(frozen_config_path.read_text(encoding="utf-8"))
    resolved = frozen.validate_config(frozen_config)
    frozen_summary = json.loads(frozen_summary_path.read_text(encoding="utf-8"))
    delay_record = frozen_summary["artifacts"]["delay_estimates"]
    delay_path = verify(
        delay_record["path"], inputs["delay_estimates_sha256"], "delay estimates"
    )
    if delay_record["sha256"] != inputs["delay_estimates_sha256"]:
        raise ValueError("delay-estimate provenance drifted")
    partial_audit = json.loads(partial_audit_path.read_text(encoding="utf-8"))
    rules = config["frozen_rules"]
    if float(partial_audit["partial_f"]["p_value_alarm_threshold"]) != float(
        rules["partial_f_p_alarm_threshold"]
    ):
        raise ValueError("partial-F threshold drifted from sealed audit")
    if float(frozen_summary["thresholds"]["residual_alarm_threshold"]) != float(
        rules["legacy_residual_alarm_threshold"]
    ):
        raise ValueError("legacy residual threshold drifted")

    sweep = config["sweep"]
    sizes = [int(value) for value in sweep["subset_sizes"]]
    if sizes != sorted(set(sizes)) or sizes[0] <= 4:
        raise ValueError("subset sizes must be unique, ordered, and greater than four")
    trials = int(sweep["deterministic_trials"])
    if trials < 1:
        raise ValueError("deterministic_trials must be positive")
    primary = str(sweep["primary_scenario"])
    groups = seven.load_delay_groups(delay_path)
    healthy, los_by_key = _primary_geometry(resolved, groups, primary)

    metric_rows: list[dict[str, Any]] = []
    legacy_threshold = float(rules["legacy_residual_alarm_threshold"])
    partial_threshold = float(rules["partial_f_p_alarm_threshold"])
    window = int(rules["persistence_window_bins"])
    required = int(rules["persistence_required_bins"])
    attack_onset = float(sweep["attack_onset_seconds"])
    for support in sizes:
        for trial in range(trials):
            scored: list[dict[str, Any]] = []
            for (scenario, bin_index), entries in sorted(groups.items()):
                if scenario != primary:
                    continue
                valid = [row for row in entries if int(row["prn"]) in healthy]
                if len(valid) < support:
                    continue
                chosen = stable_subset(
                    valid, scenario=scenario, trial=trial, support=support
                )
                los = np.asarray([
                    los_by_key[(bin_index, int(row["prn"]))] for row in chosen
                ], dtype=np.float64)
                delays = np.asarray([float(row["delay"]) for row in chosen])
                fit = fit_clock_centered_geometry(los, delays)
                if int(fit.rank) != 4:
                    raise ValueError(
                        f"rank-deficient geometry: bin {bin_index} N={support}"
                    )
                residual = float(fit.clock_centered_normalized_residual)
                partial_p = seven.partial_f_p_value(residual, support)
                design = np.column_stack((-los, np.ones(support)))
                scored.append({
                    "region": frozen.source_region(
                        scenario, str(chosen[0]["role"]), float(bin_index)
                    ),
                    "bin_index": bin_index,
                    "residual": residual,
                    "partial_f_p_value": partial_p,
                    "design_condition_number": float(np.linalg.cond(design)),
                    "legacy_raw_alarm": residual <= legacy_threshold,
                    "partial_f_raw_alarm": partial_p <= partial_threshold,
                })
            _apply_persistence(scored, window=window, required=required)
            pre = [row for row in scored if row["region"] == "stable_pre"]
            post = [row for row in scored if row["region"] == "stable_post"]
            expected_pre = int(sweep["expected_primary_pre_bins_per_trial"])
            expected_post = int(sweep["expected_primary_post_bins_per_trial"])
            if len(pre) != expected_pre or len(post) != expected_post:
                raise ValueError(
                    f"primary support drifted at N={support}: "
                    f"pre={len(pre)}, post={len(post)}"
                )
            legacy_delay = _delay(post, "legacy_persistent_alarm", attack_onset)
            partial_delay = _delay(post, "partial_f_persistent_alarm", attack_onset)
            partial_pre_rate = _rate(pre, "partial_f_persistent_alarm")
            metric_rows.append({
                "prn_count": support,
                "trial": trial,
                "primary_pre_bin_count": len(pre),
                "primary_post_bin_count": len(post),
                "legacy_pre_raw_alarm_rate": _rate(pre, "legacy_raw_alarm"),
                "partial_f_pre_raw_alarm_rate": _rate(pre, "partial_f_raw_alarm"),
                "legacy_pre_persistent_alarm_rate": _rate(
                    pre, "legacy_persistent_alarm"
                ),
                "partial_f_pre_persistent_alarm_rate": partial_pre_rate,
                "pre_median_residual": _median(pre, "residual"),
                "pre_median_partial_f_p_value": _median(pre, "partial_f_p_value"),
                "pre_median_design_condition_number": _median(
                    pre, "design_condition_number"
                ),
                "legacy_ds7_persistent_detection": legacy_delay is not None,
                "legacy_ds7_detection_delay_s": legacy_delay,
                "partial_f_ds7_persistent_detection": partial_delay is not None,
                "partial_f_ds7_detection_delay_s": partial_delay,
                "partial_f_specificity_gate_passed": partial_pre_rate <= float(
                    config["decision_gates"][
                        "maximum_primary_pre_persistent_alarm_rate"
                    ]
                ),
            })

    curve_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for support in sizes:
        selected = [row for row in metric_rows if int(row["prn_count"]) == support]

        def values(key: str) -> list[float]:
            return [float(row[key]) for row in selected if row[key] is not None]

        partial_delays = values("partial_f_ds7_detection_delay_s")
        legacy_delays = values("legacy_ds7_detection_delay_s")
        support_summary = {
            "trial_count": len(selected),
            "legacy_pre_persistent_alarm_rate": _range(values(
                "legacy_pre_persistent_alarm_rate"
            )),
            "partial_f_pre_persistent_alarm_rate": _range(values(
                "partial_f_pre_persistent_alarm_rate"
            )),
            "partial_f_specificity_trial_pass_rate": float(np.mean([
                bool(row["partial_f_specificity_gate_passed"]) for row in selected
            ])),
            "pre_residual": _range(values("pre_median_residual")),
            "pre_design_condition_number": _range(values(
                "pre_median_design_condition_number"
            )),
            "legacy_ds7_detection_trial_rate": float(np.mean([
                bool(row["legacy_ds7_persistent_detection"]) for row in selected
            ])),
            "partial_f_ds7_detection_trial_rate": float(np.mean([
                bool(row["partial_f_ds7_persistent_detection"]) for row in selected
            ])),
            "legacy_ds7_detection_delay_s": (
                _range(legacy_delays) if legacy_delays else None
            ),
            "partial_f_ds7_detection_delay_s": (
                _range(partial_delays) if partial_delays else None
            ),
        }
        summaries[str(support)] = support_summary
        curve_rows.append({
            "prn_count": support,
            "trial_count": len(selected),
            "legacy_pre_far_min": support_summary[
                "legacy_pre_persistent_alarm_rate"
            ]["minimum"],
            "legacy_pre_far_median": support_summary[
                "legacy_pre_persistent_alarm_rate"
            ]["median"],
            "legacy_pre_far_max": support_summary[
                "legacy_pre_persistent_alarm_rate"
            ]["maximum"],
            "partial_f_pre_far_min": support_summary[
                "partial_f_pre_persistent_alarm_rate"
            ]["minimum"],
            "partial_f_pre_far_median": support_summary[
                "partial_f_pre_persistent_alarm_rate"
            ]["median"],
            "partial_f_pre_far_max": support_summary[
                "partial_f_pre_persistent_alarm_rate"
            ]["maximum"],
            "median_pre_residual": support_summary["pre_residual"]["median"],
            "median_design_condition_number": support_summary[
                "pre_design_condition_number"
            ]["median"],
            "legacy_detection_trial_rate": support_summary[
                "legacy_ds7_detection_trial_rate"
            ],
            "partial_f_detection_trial_rate": support_summary[
                "partial_f_ds7_detection_trial_rate"
            ],
            "partial_f_detection_delay_median_s": (
                support_summary["partial_f_ds7_detection_delay_s"]["median"]
                if support_summary["partial_f_ds7_detection_delay_s"] else None
            ),
            "partial_f_specificity_trial_pass_rate": support_summary[
                "partial_f_specificity_trial_pass_rate"
            ],
        })

    legacy_far_medians = [float(row["legacy_pre_far_median"]) for row in curve_rows]
    partial_far_medians = [
        float(row["partial_f_pre_far_median"]) for row in curve_rows
    ]
    residual_medians = [float(row["median_pre_residual"]) for row in curve_rows]
    gates_config = config["decision_gates"]
    gates = {
        "partial_f_specificity_each_n": all(
            float(row["partial_f_specificity_trial_pass_rate"]) >= float(
                gates_config["minimum_specificity_trial_pass_rate_each_n"]
            )
            for row in curve_rows
        ),
        "partial_f_detection_each_n": all(
            float(row["partial_f_detection_trial_rate"]) >= float(
                gates_config["minimum_ds7_detection_trial_rate_each_n"]
            )
            for row in curve_rows
        ),
        "legacy_n7_far_above_n10": (
            legacy_far_medians[0] > legacy_far_medians[-1]
            if gates_config["require_legacy_median_pre_alarm_rate_n7_above_n10"]
            else True
        ),
        "legacy_n7_residual_below_n10": (
            residual_medians[0] < residual_medians[-1]
            if gates_config["require_legacy_median_pre_residual_n7_below_n10"]
            else True
        ),
    }

    unsupported: dict[str, Any] = {}
    for support_value in sweep["unsupported_sizes_to_report"]:
        support = int(support_value)
        pre_bins = 0
        post_bins = 0
        for (scenario, bin_index), entries in groups.items():
            if scenario != primary:
                continue
            valid_count = sum(
                int(row["prn"]) in healthy for row in entries
            )
            if valid_count < support:
                continue
            region = frozen.source_region(
                primary, str(entries[0]["role"]), float(bin_index)
            )
            pre_bins += region == "stable_pre"
            post_bins += region == "stable_post"
        unsupported[str(support)] = {
            "primary_pre_bin_count": pre_bins,
            "primary_post_bin_count": post_bins,
            "reason": (
                "insufficient support for the frozen 60-pre/161-post-bin comparison"
            ),
        }

    output = resolve(config["output_root"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "trial_metrics.csv"
    curve_path = output / "support_curve.csv"
    write_csv(metrics_path, metric_rows)
    write_csv(curve_path, curve_rows)
    result = {
        "schema": "gnss-doppler-lab.cgc-prn-support-sweep-result",
        "schema_version": 1,
        "status": (
            "FINITE_SUPPORT_MECHANISM_SUPPORTED"
            if all(gates.values()) else "FINITE_SUPPORT_MECHANISM_NOT_SUPPORTED"
        ),
        "all_gates_passed": all(gates.values()),
        "gates": gates,
        "subset_sizes": sizes,
        "unsupported_sizes": unsupported,
        "deterministic_trials_per_size": trials,
        "threshold_refitting": False,
        "release_commit": release_commit,
        "trend": {
            "legacy_pre_far_spearman_rho_vs_prn_count": float(
                spearmanr(sizes, legacy_far_medians).statistic
            ),
            "legacy_pre_residual_spearman_rho_vs_prn_count": float(
                spearmanr(sizes, residual_medians).statistic
            ),
            "partial_f_median_pre_far_range": float(
                max(partial_far_medians) - min(partial_far_medians)
            ),
        },
        "by_prn_count": summaries,
        "claim_boundary": (
            "nested deterministic subsets of previously sealed DS7 delay "
            "estimates; a finite-support mechanism audit, not a new RF recording"
        ),
        "inputs": {
            "config": {
                "path": str(config_path), "sha256": seven.sha256(config_path)
            },
            "frozen_summary": {
                "path": str(frozen_summary_path),
                "sha256": seven.sha256(frozen_summary_path),
            },
            "delay_estimates": {
                "path": str(delay_path), "sha256": seven.sha256(delay_path)
            },
            "partial_f_audit": {
                "path": str(partial_audit_path),
                "sha256": seven.sha256(partial_audit_path),
            },
        },
        "artifacts": {
            "trial_metrics": {
                "path": str(metrics_path),
                "sha256": seven.sha256(metrics_path),
                "row_count": len(metric_rows),
            },
            "support_curve": {
                "path": str(curve_path),
                "sha256": seven.sha256(curve_path),
                "row_count": len(curve_rows),
            },
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().config.resolve())
