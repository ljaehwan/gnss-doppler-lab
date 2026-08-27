#!/usr/bin/env python3
"""Run the frozen real-IQ CGC spoof-alarm evaluation."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_cgc_rf_challenge_pilot as pilot  # noqa: E402
from run_gcmr_texbat_external import preflight_ds4_alternate  # noqa: E402
from gnss_doppler_lab.clock_centered_geometry import fit_clock_centered_geometry  # noqa: E402
from gnss_doppler_lab.correlator_geometry import complex_profile_features  # noqa: E402
from gnss_doppler_lab.gcmr_geometry import (  # noqa: E402
    ephemeris_health_selection,
    parse_gnss_sdr_gps_ephemeris_xml,
    satellite_observation,
)

DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/cgc_real_detection_v1.json"
PROTOCOL_PATH = REPO_ROOT / "docs/results/cgc_real_detection_protocol_v1.md"
RELEASE_TOKEN = "RELEASE-CGC-REAL-DETECTION-V1"
RELEASE_INPUTS = (
    "configs/experiments/cgc_real_detection_v1.json",
    "docs/results/cgc_real_detection_protocol_v1.md",
    "scripts/run_cgc_real_detection.py",
)
EXPECTED_ARRAYS = {
    "complex_iq", "sample_count", "time_s", "prn", "channel",
    "segment_index", "cn0_db_hz",
}
EXPECTED_SOURCE_ROSTER = [
    ("cleanStatic", "calibration_only"),
    ("cleanDynamic", "locked_normal"),
    ("ds7", "primary_attack"),
    ("ds1", "secondary_attack"),
    ("ds2", "secondary_attack"),
    ("ds3", "secondary_attack"),
]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _verify_record(record: dict[str, str], label: str) -> Path:
    path = _repo_path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    observed = _sha256(path)
    if observed != record["sha256"]:
        raise ValueError(f"{label} hash mismatch: {observed}")
    return path


def _validate_fixed_contract(config: dict[str, Any]) -> None:
    if config.get("schema") != "gnss-doppler-lab.cgc-real-detection-config" or config.get("schema_version") != 1:
        raise ValueError("unsupported real detection config")
    experiment = config.get("experiment", {})
    if experiment.get("name") != "cgc-real-detection-v1":
        raise ValueError("experiment identity drifted")
    if experiment.get("primary_attack") != "TEXBAT DS7 carrier-aligned matched-power time push":
        raise ValueError("primary attack drifted")
    if experiment.get("primary_attack_cgc_outcome_accessed_before_freeze") is not False:
        raise ValueError("DS7 must be outcome-unseen before freeze")
    if experiment.get("secondary_attack_cgc_direction_accessed_before_freeze") is not True:
        raise ValueError("secondary prior access must be declared")

    analysis = config.get("analysis", {})
    expected_analysis = {
        "bin_seconds": 1.0,
        "minimum_prns": 8,
        "epoch_chunk_size": 50000,
        "calibration_interval_seconds": [330.0, 420.0],
        "residual_alarm_quantile": 0.05,
        "residual_alarm_comparison": "less_than_or_equal",
        "persistence_window_bins": 5,
        "persistence_required_bins": 3,
        "multipath_enrichment_metric": "q75 across PRNs of median epoch early-late magnitude asymmetry",
        "multipath_enrichment_calibration_quantile": 0.8,
        "stable_pre_seconds": [30.0, 90.0],
        "excluded_transition_seconds": [90.0, 110.0],
        "stable_post_start_seconds": 110.0,
        "score_available_at": "bin_end",
    }
    if any(analysis.get(key) != value for key, value in expected_analysis.items()):
        raise ValueError("analysis contract drifted")

    gates = config.get("primary_gates", {})
    expected_gates = {
        "minimum_calibration_bins": 80,
        "minimum_locked_normal_bins": 25,
        "minimum_primary_pre_bins": 50,
        "minimum_primary_post_bins": 150,
        "maximum_pooled_negative_persistent_alarm_rate": 0.05,
        "maximum_multipath_enriched_persistent_alarm_rate": 0.1,
        "primary_attack_must_have_persistent_detection": True,
        "maximum_primary_detection_delay_seconds": 60.0,
    }
    if any(gates.get(key) != value for key, value in expected_gates.items()):
        raise ValueError("primary gate contract drifted")

    frozen = config.get("frozen_candidate", {})
    if frozen.get("residual") != "SSE_LOS_plus_clock / SSE_clock_only":
        raise ValueError("residual law drifted")
    if frozen.get("spoof_score") != "negative clock-centered residual":
        raise ValueError("spoof score drifted")
    for key in ("clock_centered_module", "correlator_geometry_module", "geometry_module", "template_config"):
        _verify_record(frozen[key], key)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    _validate_fixed_contract(config)
    sources = config.get("sources")
    if not isinstance(sources, list):
        raise ValueError("sources must be a list")
    if [(row.get("name"), row.get("role")) for row in sources] != EXPECTED_SOURCE_ROSTER:
        raise ValueError("source roster or roles drifted")
    resolved = []
    for row in sources:
        paths = {
            key: _verify_record(row[key], f"{row['name']} {key}")
            for key in ("complex_epoch_npz", "export_manifest", "ephemeris", "nmea")
        }
        manifest = json.loads(paths["export_manifest"].read_text(encoding="utf-8"))
        output = manifest.get("output", {})
        feature = manifest.get("feature_schema", {})
        if feature.get("tensor") != "complex_iq" or output.get("shape", [None, None])[1:] != [9, 2]:
            raise ValueError(f"complex nine-tap manifest drifted: {row['name']}")
        if output.get("sha256") != row["complex_epoch_npz"]["sha256"]:
            raise ValueError(f"NPZ manifest hash drifted: {row['name']}")
        resolved.append({**row, "paths": paths, "manifest": manifest})
    output_root = _repo_path(config["output_root"])
    if output_root != REPO_ROOT / "artifacts/cgc_real_detection_v1":
        raise ValueError("output root drifted")
    template = json.loads(_verify_record(config["frozen_candidate"]["template_config"], "template config").read_text(encoding="utf-8"))
    return {"sources": resolved, "output_root": output_root, "template": template}


def early_late_asymmetry(complex_iq: np.ndarray) -> np.ndarray:
    """Scale-free magnitude asymmetry over the four symmetric tap pairs."""
    values = np.asarray(complex_iq)
    if values.ndim != 3 or values.shape[1:] != (9, 2) or not np.isfinite(values).all():
        raise ValueError("complex_iq must be finite [epoch,9,2]")
    magnitude = np.hypot(values[:, :, 0], values[:, :, 1]).astype(np.float64)
    paired = np.abs(magnitude[:, :4] - magnitude[:, -1:4:-1]).sum(axis=1)
    scale = magnitude.sum(axis=1)
    return np.divide(paired, scale, out=np.zeros_like(paired), where=scale > 1e-12)


def persistent_alarm(raw: np.ndarray, bins: np.ndarray, *, window: int, required: int) -> np.ndarray:
    """Return causal k-of-n alarms, resetting across non-consecutive bins."""
    alarms = np.asarray(raw, dtype=bool)
    indices = np.asarray(bins, dtype=np.int64)
    if alarms.ndim != 1 or indices.shape != alarms.shape:
        raise ValueError("raw alarms and bins must be matching vectors")
    if window < 1 or required < 1 or required > window:
        raise ValueError("invalid persistence rule")
    result = np.zeros(len(alarms), dtype=bool)
    for end in range(window - 1, len(alarms)):
        start = end - window + 1
        if np.array_equal(indices[start : end + 1], np.arange(indices[start], indices[start] + window)):
            result[end] = int(alarms[start : end + 1].sum()) >= required
    return result


def source_region(name: str, role: str, bin_start_s: float) -> str:
    value = float(bin_start_s)
    if role == "calibration_only":
        return "calibration" if 330.0 <= value < 420.0 else "development_excluded"
    if role == "locked_normal":
        return "locked_normal"
    if 30.0 <= value < 90.0:
        return "stable_pre"
    if value >= 110.0:
        return "stable_post"
    return "excluded_transition"


def _estimate_epochs(
    estimator: Any, complex_iq: np.ndarray, chunk_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(complex_iq)
    estimates = np.empty(count, dtype=np.float64)
    distances = np.empty(count, dtype=np.float64)
    asymmetry = np.empty(count, dtype=np.float64)
    for start in range(0, count, chunk_size):
        end = min(count, start + chunk_size)
        chunk = complex_iq[start:end]
        profiles = chunk[:, :, 0].astype(np.float64) + 1j * chunk[:, :, 1].astype(np.float64)
        features = complex_profile_features(profiles, prompt_index=4)
        estimates[start:end], distances[start:end], _ = estimator.estimate(features)
        asymmetry[start:end] = early_late_asymmetry(chunk)
    return estimates, distances, asymmetry


def evaluate_source(
    source: dict[str, Any], analysis: dict[str, Any], estimator: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    name, role = source["name"], source["role"]
    with np.load(source["paths"]["complex_epoch_npz"], allow_pickle=False) as loaded:
        if set(loaded.files) != EXPECTED_ARRAYS:
            raise ValueError(f"NPZ arrays drifted: {name}")
        arrays = {key: loaded[key] for key in loaded.files}
    complex_iq = arrays["complex_iq"]
    count = len(complex_iq)
    if complex_iq.shape != (count, 9, 2):
        raise ValueError(f"NPZ complex shape mismatch: {name}")
    if any(arrays[key].shape != (count,) for key in EXPECTED_ARRAYS - {"complex_iq"}):
        raise ValueError(f"NPZ vector shape mismatch: {name}")
    if count != int(source["manifest"]["output"]["row_count"]):
        raise ValueError(f"NPZ row count mismatch: {name}")
    times = np.asarray(arrays["time_s"], dtype=np.float64)
    prns = np.asarray(arrays["prn"], dtype=np.int64)
    if not np.isfinite(times).all() or np.any(prns < 1) or np.any(prns > 32):
        raise ValueError(f"invalid time or PRN: {name}")

    ephemerides = parse_gnss_sdr_gps_ephemeris_xml(source["paths"]["ephemeris"])
    tracked = set(int(value) for value in np.unique(prns))
    preflight = preflight_ds4_alternate(
        source["paths"]["nmea"].parent,
        ephemerides,
        configured_tow0_s=float(source["tow0_s"]),
        tracked_prns=tracked,
        min_prns=int(analysis["minimum_prns"]),
    )
    healthy, health = ephemeris_health_selection(
        ephemerides, tracked_prns=tracked, min_prns=int(analysis["minimum_prns"])
    )
    receiver_ecef = preflight["receiver_position_contract"]["ecef"]
    estimates, distances, asymmetry = _estimate_epochs(
        estimator, complex_iq, int(analysis["epoch_chunk_size"])
    )

    bins = np.floor(times / float(analysis["bin_seconds"])).astype(np.int64)
    valid = np.isin(prns, np.asarray(sorted(healthy), dtype=np.int64))
    valid_bins, valid_prns = bins[valid], prns[valid]
    valid_estimates, valid_distances = estimates[valid], distances[valid]
    valid_asymmetry = asymmetry[valid]
    order = np.lexsort((valid_prns, valid_bins))
    valid_bins, valid_prns = valid_bins[order], valid_prns[order]
    valid_estimates, valid_distances = valid_estimates[order], valid_distances[order]
    valid_asymmetry = valid_asymmetry[order]
    keys = valid_bins * 64 + valid_prns
    boundaries = np.r_[0, np.flatnonzero(np.diff(keys)) + 1, len(keys)]

    delay_rows: list[dict[str, Any]] = []
    by_bin: dict[int, list[tuple[int, float, float]]] = {}
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        bin_index = int(valid_bins[start])
        prn = int(valid_prns[start])
        delay = float(np.median(valid_estimates[start:end]))
        prn_asymmetry = float(np.median(valid_asymmetry[start:end]))
        delay_rows.append({
            "scenario": name,
            "role": role,
            "bin_index": bin_index,
            "bin_start_s": float(bin_index),
            "prn": f"G{prn:02d}",
            "epoch_count": int(end - start),
            "estimated_delay_chips": delay,
            "median_template_distance": float(np.median(valid_distances[start:end])),
            "median_early_late_asymmetry": prn_asymmetry,
        })
        by_bin.setdefault(bin_index, []).append((prn, delay, prn_asymmetry))

    geometry_rows: list[dict[str, Any]] = []
    for bin_index, entries in sorted(by_bin.items()):
        if len(entries) < int(analysis["minimum_prns"]):
            continue
        tow = (float(source["tow0_s"]) + float(bin_index) + 0.5) % 604800.0
        los = np.asarray([
            satellite_observation(receiver_ecef, healthy[prn], tow).los_ecef
            for prn, _, _ in entries
        ], dtype=np.float64)
        delays = np.asarray([delay for _, delay, _ in entries], dtype=np.float64)
        fit = fit_clock_centered_geometry(los, delays)
        geometry_rows.append({
            "scenario": name,
            "role": role,
            "bin_index": bin_index,
            "bin_start_s": float(bin_index),
            "bin_end_s": float(bin_index + 1),
            "region": source_region(name, role, float(bin_index)),
            "prn_count": len(entries),
            "los_tow_s": tow,
            "clock_centered_geometry_residual": fit.clock_centered_normalized_residual,
            "spoof_score": -fit.clock_centered_normalized_residual,
            "directional_geometry_coherence": fit.directional_coherence,
            "legacy_zero_referenced_residual": fit.normalized_residual,
            "fit_rank": fit.rank,
            "clock_only_bias_chips": fit.clock_only_bias_chips,
            "q75_prn_early_late_asymmetry": float(np.quantile(
                np.asarray([value for _, _, value in entries]), 0.75
            )),
        })
    metadata = {
        "scenario": name,
        "role": role,
        "input_row_count": count,
        "healthy_tracked_prns": health["healthy_tracked_prns"],
        "preflight": preflight,
    }
    return delay_rows, geometry_rows, metadata


def _region_rows(rows: list[dict[str, Any]], scenario: str, region: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["scenario"] == scenario and row["region"] == region]


def apply_alarm_rules(
    rows: list[dict[str, Any]], analysis: dict[str, Any]
) -> dict[str, float]:
    calibration = _region_rows(rows, "cleanStatic", "calibration")
    if not calibration:
        raise ValueError("calibration region is empty")
    residual_threshold = float(np.quantile(
        np.asarray([row["clock_centered_geometry_residual"] for row in calibration], dtype=np.float64),
        float(analysis["residual_alarm_quantile"]),
    ))
    distortion_threshold = float(np.quantile(
        np.asarray([row["q75_prn_early_late_asymmetry"] for row in calibration], dtype=np.float64),
        float(analysis["multipath_enrichment_calibration_quantile"]),
    ))
    for row in rows:
        row["residual_alarm_threshold"] = residual_threshold
        row["multipath_enrichment_threshold"] = distortion_threshold
        row["raw_spoof_alarm"] = bool(
            float(row["clock_centered_geometry_residual"]) <= residual_threshold
        )
        row["multipath_enriched"] = bool(
            float(row["q75_prn_early_late_asymmetry"]) >= distortion_threshold
        )
        row["persistent_spoof_alarm"] = False

    window = int(analysis["persistence_window_bins"])
    required = int(analysis["persistence_required_bins"])
    groups = sorted({(row["scenario"], row["region"]) for row in rows})
    for scenario, region in groups:
        group = sorted(
            (row for row in rows if row["scenario"] == scenario and row["region"] == region),
            key=lambda row: int(row["bin_index"]),
        )
        persistent = persistent_alarm(
            np.asarray([row["raw_spoof_alarm"] for row in group], dtype=bool),
            np.asarray([row["bin_index"] for row in group], dtype=np.int64),
            window=window,
            required=required,
        )
        for row, value in zip(group, persistent):
            row["persistent_spoof_alarm"] = bool(value)

    negative_regions = {"locked_normal", "stable_pre"}
    for row in rows:
        if row["persistent_spoof_alarm"]:
            classification = "spoof_alarm"
        elif row["region"] in negative_regions and row["multipath_enriched"]:
            classification = "multipath_enriched_negative"
        else:
            classification = "no_spoof_alarm"
        row["detector_classification"] = classification
    return {
        "residual_alarm_threshold": residual_threshold,
        "multipath_enrichment_threshold": distortion_threshold,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "bin_count": 0,
            "raw_alarm_count": 0,
            "raw_alarm_rate": None,
            "persistent_alarm_count": 0,
            "persistent_alarm_rate": None,
            "multipath_enriched_bin_count": 0,
            "multipath_enriched_persistent_alarm_count": 0,
            "multipath_enriched_persistent_alarm_rate": None,
            "median_residual": None,
            "median_distortion": None,
        }
    raw = np.asarray([row["raw_spoof_alarm"] for row in rows], dtype=bool)
    persistent = np.asarray([row["persistent_spoof_alarm"] for row in rows], dtype=bool)
    enriched = np.asarray([row["multipath_enriched"] for row in rows], dtype=bool)
    enriched_count = int(enriched.sum())
    enriched_alarm_count = int((enriched & persistent).sum())
    return {
        "bin_count": len(rows),
        "raw_alarm_count": int(raw.sum()),
        "raw_alarm_rate": float(raw.mean()),
        "persistent_alarm_count": int(persistent.sum()),
        "persistent_alarm_rate": float(persistent.mean()),
        "multipath_enriched_bin_count": enriched_count,
        "multipath_enriched_persistent_alarm_count": enriched_alarm_count,
        "multipath_enriched_persistent_alarm_rate": (
            float(enriched_alarm_count / enriched_count) if enriched_count else None
        ),
        "median_residual": float(np.median([
            row["clock_centered_geometry_residual"] for row in rows
        ])),
        "median_distortion": float(np.median([
            row["q75_prn_early_late_asymmetry"] for row in rows
        ])),
        "minimum_prn_count": min(int(row["prn_count"]) for row in rows),
    }


def _first_detection(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    detected = sorted(
        (row for row in rows if row["persistent_spoof_alarm"]),
        key=lambda row: float(row["bin_end_s"]),
    )
    if not detected:
        return None, None
    first = float(detected[0]["bin_end_s"])
    return first, first - 100.0


def build_evaluation(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    calibration = _region_rows(rows, "cleanStatic", "calibration")
    locked = _region_rows(rows, "cleanDynamic", "locked_normal")
    primary_pre = _region_rows(rows, "ds7", "stable_pre")
    primary_post = _region_rows(rows, "ds7", "stable_post")
    primary_negatives = locked + primary_pre
    first_detection_s, detection_delay_s = _first_detection(primary_post)

    labels = np.r_[
        np.zeros(len(primary_negatives), dtype=np.int64),
        np.ones(len(primary_post), dtype=np.int64),
    ]
    scores = np.asarray([
        row["spoof_score"] for row in primary_negatives + primary_post
    ], dtype=np.float64)
    descriptive_auc = float(roc_auc_score(labels, scores))

    gates_config = config["primary_gates"]
    negative_summary = summarize_rows(primary_negatives)
    enriched_rate = negative_summary["multipath_enriched_persistent_alarm_rate"]
    gates = {
        "minimum_calibration_bins": len(calibration) >= int(gates_config["minimum_calibration_bins"]),
        "minimum_locked_normal_bins": len(locked) >= int(gates_config["minimum_locked_normal_bins"]),
        "minimum_primary_pre_bins": len(primary_pre) >= int(gates_config["minimum_primary_pre_bins"]),
        "minimum_primary_post_bins": len(primary_post) >= int(gates_config["minimum_primary_post_bins"]),
        "maximum_pooled_negative_persistent_alarm_rate": (
            negative_summary["persistent_alarm_rate"]
            <= float(gates_config["maximum_pooled_negative_persistent_alarm_rate"])
        ),
        "maximum_multipath_enriched_persistent_alarm_rate": (
            enriched_rate is not None
            and enriched_rate
            <= float(gates_config["maximum_multipath_enriched_persistent_alarm_rate"])
        ),
        "primary_attack_must_have_persistent_detection": first_detection_s is not None,
        "maximum_primary_detection_delay_seconds": (
            detection_delay_s is not None
            and detection_delay_s
            <= float(gates_config["maximum_primary_detection_delay_seconds"])
        ),
    }

    per_scenario: dict[str, Any] = {}
    for name in ("ds7", "ds1", "ds2", "ds3"):
        pre = _region_rows(rows, name, "stable_pre")
        post = _region_rows(rows, name, "stable_post")
        first_s, delay_s = _first_detection(post)
        if pre and post:
            scenario_labels = np.r_[
                np.zeros(len(pre), dtype=np.int64),
                np.ones(len(post), dtype=np.int64),
            ]
            scenario_scores = np.asarray([
                row["spoof_score"] for row in pre + post
            ], dtype=np.float64)
            auc = float(roc_auc_score(scenario_labels, scenario_scores))
        else:
            auc = None
        per_scenario[name] = {
            "role": "primary_attack" if name == "ds7" else "secondary_attack_prior_direction_known",
            "stable_pre": summarize_rows(pre),
            "stable_post": summarize_rows(post),
            "first_persistent_detection_bin_end_s": first_s,
            "persistent_detection_delay_from_nominal_onset_s": delay_s,
            "descriptive_serial_bin_auc": auc,
        }

    passed = all(gates.values())
    return {
        "status": "REAL_SPOOF_DETECTION_SUPPORTED" if passed else "REAL_SPOOF_DETECTION_NOT_SUPPORTED",
        "all_primary_gates_passed": passed,
        "primary_gates": gates,
        "calibration": summarize_rows(calibration),
        "locked_normal": summarize_rows(locked),
        "primary_negative_pool": negative_summary,
        "primary_attack_post": summarize_rows(primary_post),
        "primary_first_persistent_detection_bin_end_s": first_detection_s,
        "primary_detection_delay_from_nominal_onset_s": detection_delay_s,
        "primary_descriptive_serial_bin_auc": descriptive_auc,
        "per_attack_scenario": per_scenario,
    }



def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0])
    extras = sorted({key for row in rows for key in row} - set(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields + extras, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _committed_release() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        _git("ls-files", "--error-unmatch", relative)
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative], cwd=REPO_ROOT
        ).returncode
        if dirty:
            raise ValueError(f"release input is not committed and clean: {relative}")
    return {
        "head_commit": _git("rev-parse", "HEAD"),
        "input_commits": {
            relative: _git("log", "-1", "--format=%H", "--", relative)
            for relative in RELEASE_INPUTS
        },
        "runner_sha256": _sha256(Path(__file__).resolve()),
    }


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    context = validate_config(config)
    root = context["output_root"]
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    state_path = root / "release_state.json"
    state = {
        "schema": "gnss-doppler-lab.cgc-real-detection-release-state",
        "schema_version": 1,
        "phase": "released_before_outcome_access",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": _sha256(PROTOCOL_PATH)},
        "commits": _committed_release(),
        "metrics_emitted": False,
    }
    _write_json(state_path, state)

    estimator = pilot._estimator(context["template"])
    all_delays: list[dict[str, Any]] = []
    all_geometry: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for source in context["sources"]:
        print(f"[real-detection] {source['name']} ({source['role']})", flush=True)
        delays, geometry, source_metadata = evaluate_source(
            source, config["analysis"], estimator
        )
        all_delays.extend(delays)
        all_geometry.extend(geometry)
        metadata.append(source_metadata)

    thresholds = apply_alarm_rules(all_geometry, config["analysis"])
    evaluation = build_evaluation(all_geometry, config)
    state["phase"] = "metrics_emitted"
    state["metrics_emitted"] = True
    state["metrics_emitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(state_path, state)

    delay_path = root / "delay_estimates.csv"
    score_path = root / "detection_scores.csv"
    _write_csv(delay_path, all_delays)
    _write_csv(score_path, all_geometry)
    result = {
        "schema": "gnss-doppler-lab.cgc-real-detection-result",
        "schema_version": 1,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": _sha256(PROTOCOL_PATH)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        "release_state": {"path": str(state_path.resolve()), "sha256": _sha256(state_path)},
        "thresholds": thresholds,
        "primary_evaluation": evaluation,
        "source_metadata": metadata,
        "artifacts": {
            "delay_estimates": {
                "path": str(delay_path.resolve()), "sha256": _sha256(delay_path),
                "row_count": len(all_delays),
            },
            "detection_scores": {
                "path": str(score_path.resolve()), "sha256": _sha256(score_path),
                "row_count": len(all_geometry),
            },
        },
        "claim_boundary": config["claim_boundary"],
        "post_release_tuning_or_retest": False,
    }
    summary_path = root / "summary.json"
    _write_json(summary_path, result)
    print(json.dumps({
        "summary": str(summary_path),
        "status": evaluation["status"],
        "thresholds": thresholds,
        "primary": evaluation,
    }, indent=2, sort_keys=True))
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--release-token", choices=[RELEASE_TOKEN])
    args = parser.parse_args(argv)
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    if args.validate_only:
        print("real detection config and pinned inputs verified; no CGC outcome accessed")
        return 0
    run(DEFAULT_CONFIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

