#!/usr/bin/env python3
"""Run the score-free support preflight for JammerTest JT23-17.1.6."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for search_root in (ROOT / "scripts", ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_jammertest2023_jt17_receiver as receiver_runner  # noqa: E402
from gnss_doppler_lab.gcmr_geometry import (  # noqa: E402
    ephemeris_health_selection,
    parse_gnss_sdr_gps_ephemeris_xml,
)


DEFAULT_CONFIG = ROOT / "configs/experiments/jammertest2023_jt17_cgc_v1.json"
PROTOCOL = ROOT / "docs/results/jammertest2023_jt17_cgc_protocol_v1.md"
RELEASE_TOKEN = "RELEASE-JAMMERTEST2023-JT17-SUPPORT-V2"
RELEASE_INPUTS = (
    "configs/experiments/jammertest2023_jt17_cgc_v1.json",
    "docs/results/jammertest2023_jt17_cgc_protocol_v1.md",
    "scripts/run_jammertest2023_jt17_receiver.py",
    "scripts/audit_jammertest2023_jt17_support.py",
    "scripts/run_jammertest2023_jt17_cgc_detection.py",
)
COMPLEX_TAP_NAMES = tuple(
    name
    for label in ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")
    for name in (f"tap_I_{label}", f"tap_Q_{label}")
)


def resolve(value: str | Path) -> Path:
    return receiver_runner.resolve(value)


def sha256(path: str | Path) -> str:
    return receiver_runner.sha256(path)


def validate_config(config: dict[str, Any]) -> None:
    receiver_runner.validate_config(config)
    support = config["support"]
    expected = {
        "bin_seconds": 1.0,
        "minimum_epochs_per_prn_bin": 40,
        "minimum_prns": 8,
        "minimum_bins": {"clean": 60, "aligned_spoof": 60, "carryoff_onset": 20},
        "require_complex_nine_tap_schema": True,
        "score_access": False,
        "delay_template_access": False,
        "output_root": "/home/ubuntu/hdd_data/jammertest2023/analysis/jt23_17_1_6_cgc_v1/support_preflight_v2",
    }
    if support != expected:
        raise ValueError("JammerTest support contract drifted")
    regions = config["analysis"]["regions_seconds"]
    if regions != {
        "clean": [40.0, 200.0],
        "aligned_spoof": [246.0, 500.0],
        "carryoff_onset": [526.0, 556.0],
    }:
        raise ValueError("JammerTest support intervals drifted")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def committed_release() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        _git("ls-files", "--error-unmatch", relative)
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT)
        if dirty.returncode:
            raise ValueError(f"release input is not committed and clean: {relative}")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "inputs": {relative: {"sha256": sha256(ROOT / relative)} for relative in RELEASE_INPUTS},
    }


def infer_recording_start_tow(
    observables_path: Path, *, cadence_s: float, max_residual_s: float,
    minimum_stable_rows: int = 100,
) -> dict[str, Any]:
    """Fit TOW zero on the longest score-free cadence-consistent segment."""
    with h5py.File(observables_path, "r") as handle:
        if "RX_time" not in handle:
            raise ValueError("observables MAT is missing RX_time")
        rx = np.asarray(handle["RX_time"], dtype=np.float64)
    if rx.ndim == 1:
        rx = rx[:, None]
    rows: list[int] = []
    values: list[float] = []
    for row_index, row in enumerate(rx):
        usable = row[np.isfinite(row) & (row > 0.0)]
        if usable.size:
            rows.append(row_index)
            values.append(float(np.median(usable)))
    if len(rows) < 100:
        raise ValueError("observables contain fewer than 100 valid timing rows")
    y = np.asarray(values, dtype=np.float64)
    unwrapped = y.copy()
    for index in range(1, len(unwrapped)):
        while unwrapped[index] - unwrapped[index - 1] < -302400.0:
            unwrapped[index:] += 604800.0
        while unwrapped[index] - unwrapped[index - 1] > 302400.0:
            unwrapped[index:] -= 604800.0
    row_array = np.asarray(rows, dtype=np.int64)
    x = row_array.astype(np.float64) * float(cadence_s)
    intercept_samples = unwrapped - x
    boundaries = np.flatnonzero(
        np.abs(np.diff(intercept_samples)) > float(max_residual_s)
    ) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(rows)]
    segments = [(int(start), int(stop)) for start, stop in zip(starts, stops)]
    stable_start, stable_stop = max(
        segments, key=lambda pair: (pair[1] - pair[0], pair[0])
    )
    if stable_stop - stable_start < int(minimum_stable_rows):
        raise ValueError(
            "observables contain no sufficiently long cadence-consistent segment"
        )
    stable_rows = row_array[stable_start:stable_stop]
    stable_values = unwrapped[stable_start:stable_stop]
    stable_x = stable_rows.astype(np.float64) * float(cadence_s)
    intercept = float(np.median(stable_values - stable_x))
    residual = stable_values - (intercept + stable_x)
    maximum = float(np.max(np.abs(residual)))
    if maximum > float(max_residual_s) + 1e-12:
        raise ValueError(f"observables TOW fit residual {maximum} exceeds frozen limit")
    return {
        "recording_start_tow_s": intercept % 604800.0,
        "valid_row_count": len(rows),
        "stable_row_count": int(stable_stop - stable_start),
        "excluded_clock_settle_row_count": int(
            len(rows) - (stable_stop - stable_start)
        ),
        "clock_step_count": int(len(boundaries)),
        "first_valid_row_index": rows[0],
        "first_valid_rx_time_s": values[0],
        "first_stable_row_index": int(stable_rows[0]),
        "last_stable_row_index": int(stable_rows[-1]),
        "first_full_stable_bin_index": int(
            np.ceil(stable_rows[0] * float(cadence_s))
        ),
        "last_full_stable_bin_exclusive": int(
            np.floor((stable_rows[-1] + 1) * float(cadence_s))
        ),
        "cadence_s": float(cadence_s),
        "maximum_absolute_fit_residual_s": maximum,
        "fit_method": (
            "median intercept on longest cadence-consistent segment with "
            "frozen 0.02 s slope after GPS-week unwrap"
        ),
    }


def audit_tracking_schema_and_counts(
    run_dir: Path, *, internal_rate_hz: int, minimum_epochs: int
) -> dict[str, Any]:
    paths = sorted((run_dir / "raw").glob("epl_tracking_ch_*.mat"))
    if not paths:
        raise ValueError("receiver has no tracking MAT files")
    per_channel: dict[tuple[int, int, str], int] = {}
    valid_file_count = 0
    for path in paths:
        with h5py.File(path, "r") as handle:
            if "PRN" not in handle or "PRN_start_sample_count" not in handle:
                continue
            missing = [name for name in COMPLEX_TAP_NAMES if name not in handle]
            if missing:
                raise ValueError(f"complex-nine-tap schema missing in {path.name}: {missing}")
            prns = np.asarray(handle["PRN"]).reshape(-1)
            samples = np.asarray(handle["PRN_start_sample_count"]).reshape(-1)
        if len(prns) != len(samples):
            raise ValueError(f"tracking identity/sample length mismatch: {path}")
        if prns.shape == (2,) and np.array_equal(prns, np.asarray([1, 0])):
            continue
        valid_file_count += 1
        for raw_prn, raw_sample in zip(prns, samples):
            prn = int(raw_prn)
            sample = float(raw_sample)
            if not 1 <= prn <= 32 or not np.isfinite(sample) or sample < 0.0:
                continue
            bin_index = int(np.floor(sample / float(internal_rate_hz)))
            key = (bin_index, prn, path.name)
            per_channel[key] = per_channel.get(key, 0) + 1
    consolidated: dict[tuple[int, int], int] = {}
    for (bin_index, prn, _channel), count in per_channel.items():
        key = (bin_index, prn)
        consolidated[key] = max(consolidated.get(key, 0), count)
    eligible = {
        key: count for key, count in consolidated.items() if count >= int(minimum_epochs)
    }
    by_bin: dict[int, list[int]] = {}
    for (bin_index, prn), _count in eligible.items():
        by_bin.setdefault(bin_index, []).append(prn)
    return {
        "tracking_mat_file_count": len(paths),
        "tracking_mat_files_with_valid_prns": valid_file_count,
        "complex_nine_tap_schema_present": True,
        "tap_values_read": False,
        "eligible_prns_by_bin": {
            str(key): sorted(set(value)) for key, value in sorted(by_bin.items())
        },
    }


def interval_support(
    eligible_by_bin: dict[str, list[int]], healthy: set[int], interval: list[float],
    *, minimum_prns: int, minimum_bins: int,
) -> dict[str, Any]:
    lo, hi = map(float, interval)
    counts: dict[int, int] = {}
    prns_by_bin: dict[str, list[str]] = {}
    for bin_index in range(int(np.floor(lo)), int(np.ceil(hi))):
        prns = sorted(healthy.intersection(eligible_by_bin.get(str(bin_index), [])))
        counts[bin_index] = len(prns)
        prns_by_bin[str(bin_index)] = [f"G{prn:02d}" for prn in prns]
    primary = [key for key, count in counts.items() if count >= minimum_prns]
    return {
        "interval_seconds": [lo, hi],
        "minimum_prns": minimum_prns,
        "minimum_bins": minimum_bins,
        "primary_bin_count": len(primary),
        "primary_bins": primary,
        "minimum_observed_prn_count": min(counts.values(), default=0),
        "maximum_observed_prn_count": max(counts.values(), default=0),
        "healthy_eligible_prns_by_bin": prns_by_bin,
        "support_eligible": len(primary) >= minimum_bins,
        "status": "SUPPORT_ELIGIBLE" if len(primary) >= minimum_bins else "INSUFFICIENT_SUPPORT",
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    release = committed_release()
    dataset, receiver, support = config["dataset"], config["receiver"], config["support"]
    iq = resolve(dataset["iq_path"])
    if not iq.is_file() or iq.stat().st_size != int(dataset["iq_bytes"]):
        raise ValueError("JammerTest IQ identity failed before support audit")
    run_dir = resolve(receiver["output_root"]) / receiver["run_id"]
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "gnss-doppler-lab.jammertest2023-jt17-receiver.v1":
        raise ValueError("receiver manifest schema mismatch")
    if manifest["source"].get("iq_sha256") != dataset["iq_sha256"]:
        raise ValueError("receiver manifest does not pin the published full-file hash")
    if manifest["receiver"].get("executable_sha256") != receiver["executable_sha256"]:
        raise ValueError("receiver executable identity drifted")
    if manifest["tracking"].get("tap_count") != 9 or manifest["tracking"].get("tap_spacing_chips") != 0.125:
        raise ValueError("receiver manifest is not complex nine-tap")
    audit = audit_tracking_schema_and_counts(
        run_dir,
        internal_rate_hz=int(receiver["internal_sample_rate_hz"]),
        minimum_epochs=int(support["minimum_epochs_per_prn_bin"]),
    )
    ephemerides = parse_gnss_sdr_gps_ephemeris_xml(run_dir / "gps_ephemeris.xml")
    tracked = {
        int(prn) for prns in audit["eligible_prns_by_bin"].values() for prn in prns
    }
    healthy_map, health = ephemeris_health_selection(
        ephemerides, tracked_prns=tracked, min_prns=int(support["minimum_prns"])
    )
    healthy = set(healthy_map)
    timing = infer_recording_start_tow(
        run_dir / "raw" / "observables.mat",
        cadence_s=float(config["analysis"]["observables_cadence_s"]),
        max_residual_s=float(config["analysis"]["observables_tow_fit_max_residual_s"]),
        minimum_stable_rows=int(config["analysis"]["observables_minimum_stable_rows"]),
    )
    stable_eligible = {
        key: value for key, value in audit["eligible_prns_by_bin"].items()
        if timing["first_full_stable_bin_index"] <= int(key)
        < timing["last_full_stable_bin_exclusive"]
    }
    intervals = {
        name: interval_support(
            stable_eligible, healthy, interval,
            minimum_prns=int(support["minimum_prns"]),
            minimum_bins=int(support["minimum_bins"][name]),
        )
        for name, interval in config["analysis"]["regions_seconds"].items()
    }
    eligible = all(row["support_eligible"] for row in intervals.values())
    result = {
        "schema": "gnss-doppler-lab.jammertest2023-jt17-support-result.v2",
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release": release,
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "protocol": {"path": str(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "receiver_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "score_accessed": False,
        "delay_template_accessed": False,
        "tap_values_read": False,
        "detector_loaded": False,
        "tracking_support": audit,
        "recording_timing": timing,
        "healthy_ephemeris_prns": [f"G{prn:02d}" for prn in sorted(healthy)],
        "ephemeris_health": health,
        "intervals": intervals,
        "decision": "SUPPORT_ELIGIBLE" if eligible else "INSUFFICIENT_SUPPORT",
        "claim_boundary": "Score-free input-support decision only; no spoof-detection result.",
    }
    output = resolve(support["output_root"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    summary = output / "summary.json"
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(summary), "decision": result["decision"],
        "healthy_ephemeris_prns": result["healthy_ephemeris_prns"],
        "primary_bins": {name: row["primary_bin_count"] for name, row in intervals.items()},
        "score_accessed": False,
    }, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release-token", required=True)
    args = parser.parse_args()
    if args.release_token != RELEASE_TOKEN:
        raise ValueError("release token mismatch")
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
