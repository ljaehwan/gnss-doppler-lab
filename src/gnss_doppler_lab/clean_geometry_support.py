"""Clean-only satellite-support gate for external GPS L1 receiver runs."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np


COMPLEX_TAP_LABELS = ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")


def _validate_parameters(
    *, start_s: float, end_s: float, bin_seconds: float, minimum_epochs: int,
    minimum_primary_prns: int, secondary_boundary_prns: int,
    minimum_primary_bins: int,
) -> None:
    numeric = (start_s, end_s, bin_seconds)
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("time parameters must be finite")
    if start_s < 0.0 or end_s <= start_s or bin_seconds <= 0.0:
        raise ValueError("invalid analysis interval or bin width")
    if minimum_epochs < 1 or minimum_primary_prns < 5:
        raise ValueError("minimum epochs must be positive and primary PRNs must be at least five")
    if secondary_boundary_prns < 5 or secondary_boundary_prns >= minimum_primary_prns:
        raise ValueError("secondary boundary must be below the primary PRN threshold")
    if minimum_primary_bins < 1:
        raise ValueError("minimum primary-bin count must be positive")


def _manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if "source" not in document or "sample_rate_hz" not in document["source"]:
        raise ValueError(f"receiver manifest has no source sample rate: {path}")
    return document


def _prn_number(value: object) -> int | None:
    text = str(value).strip().upper()
    if text.startswith("G"):
        text = text[1:]
    try:
        number = int(float(text))
    except (TypeError, ValueError, OverflowError):
        return None
    return number if 1 <= number <= 32 else None


def _channel_number(path: Path, fallback: int) -> int:
    match = re.search(r"_ch_(\d+)\.mat$", path.name)
    return int(match.group(1)) if match else fallback


def _complex_schema_present(handle: h5py.File) -> bool:
    return all(
        f"tap_{component}_{label}" in handle
        for label in COMPLEX_TAP_LABELS
        for component in ("I", "Q")
    )


def _mat_counts(
    run_dir: Path, manifest: dict[str, Any], *, start_s: float, end_s: float,
    bin_seconds: float, require_complex_nine_tap: bool,
) -> tuple[dict[tuple[int, int, int], int], set[int], dict[str, Any]] | None:
    tracking = manifest.get("tracking", {})
    raw_dir = run_dir / str(tracking.get("raw_directory", "raw"))
    paths = sorted(raw_dir.glob("epl_tracking_ch_*.mat"))
    if not paths:
        return None
    sample_rate_hz = float(manifest["source"]["sample_rate_hz"])
    time_offset_s = float(manifest["source"].get("start_offset_s", 0.0))
    if sample_rate_hz <= 0.0 or not math.isfinite(sample_rate_hz):
        raise ValueError("receiver sample rate must be finite and positive")
    counts: dict[tuple[int, int, int], int] = {}
    discovered: set[int] = set()
    checked_nonempty = 0
    observed_rows = 0
    for fallback_channel, path in enumerate(paths):
        channel = _channel_number(path, fallback_channel)
        with h5py.File(path, "r") as handle:
            if "PRN" not in handle or "PRN_start_sample_count" not in handle:
                raise ValueError(f"tracking MAT lacks PRN/time datasets: {path}")
            raw_prns = np.asarray(handle["PRN"]).reshape(-1)
            sample_counts = np.asarray(handle["PRN_start_sample_count"]).reshape(-1)
            if len(raw_prns) != len(sample_counts):
                raise ValueError(f"tracking MAT PRN/time length mismatch: {path}")
            if raw_prns.shape == (2,) and np.array_equal(raw_prns, np.asarray([1, 0])):
                continue
            prns = np.asarray([_prn_number(value) or 0 for value in raw_prns], dtype=np.int16)
            time_s = sample_counts.astype(np.float64) / sample_rate_hz + time_offset_s
            selected = (prns > 0) & (time_s >= start_s) & (time_s < end_s)
            if not np.any(selected):
                continue
            if require_complex_nine_tap and not _complex_schema_present(handle):
                raise ValueError(f"tracking MAT lacks complex nine-tap datasets: {path}")
            checked_nonempty += 1
            selected_prns = prns[selected]
            selected_bins = np.floor(time_s[selected] / bin_seconds).astype(np.int64)
            observed_rows += int(len(selected_prns))
            for bin_index, prn in zip(selected_bins.tolist(), selected_prns.tolist()):
                key = (int(bin_index), int(prn), channel)
                counts[key] = counts.get(key, 0) + 1
                discovered.add(int(prn))
    return counts, discovered, {
        "kind": "tracking_mat",
        "path": str(raw_dir.resolve()),
        "file_count": len(paths),
        "nonempty_files_in_interval": checked_nonempty,
        "observed_epoch_rows": observed_rows,
        "complex_nine_tap_evidence": "verified_dataset_names" if require_complex_nine_tap else "not_required",
    }


def _csv_counts(
    run_dir: Path, manifest: dict[str, Any], *, start_s: float, end_s: float,
    bin_seconds: float, require_complex_nine_tap: bool,
) -> tuple[dict[tuple[int, int, int], int], set[int], dict[str, Any]]:
    path = run_dir / str(manifest.get("tracking", {}).get("csv", "tracking.csv"))
    if not path.is_file():
        raise FileNotFoundError(f"no tracking MAT files or tracking CSV: {run_dir}")
    tracking = manifest.get("tracking", {})
    if require_complex_nine_tap:
        if int(tracking.get("tap_count", 0)) != 9:
            raise ValueError("tracking CSV manifest does not attest nine correlator taps")
        component_order = tracking.get("component_order")
        tap_schema = tracking.get("tap_schema", {})
        if component_order != ["I", "Q"] and tap_schema.get("component_order") != ["I", "Q"]:
            raise ValueError("tracking CSV manifest does not attest complex I/Q taps")
    counts: dict[tuple[int, int, int], int] = {}
    discovered: set[int] = set()
    observed_rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"time_s", "channel", "prn"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"tracking CSV lacks required columns: {path}")
        for row in reader:
            time_s = float(row["time_s"])
            if time_s < start_s or time_s >= end_s:
                continue
            prn = _prn_number(row["prn"])
            if prn is None:
                continue
            channel = int(row["channel"])
            bin_index = int(math.floor(time_s / bin_seconds))
            key = (bin_index, prn, channel)
            counts[key] = counts.get(key, 0) + 1
            discovered.add(prn)
            observed_rows += 1
    return counts, discovered, {
        "kind": "tracking_csv",
        "path": str(path.resolve()),
        "observed_epoch_rows": observed_rows,
        "complex_nine_tap_evidence": "receiver_manifest" if require_complex_nine_tap else "not_required",
    }


def audit_clean_geometry_support(
    receiver_run_dir: str | Path, *, start_s: float, end_s: float,
    bin_seconds: float = 1.0, minimum_epochs: int = 200,
    minimum_primary_prns: int = 8, secondary_boundary_prns: int = 7,
    minimum_primary_bins: int = 60, require_complex_nine_tap: bool = True,
) -> dict[str, Any]:
    """Count conservative per-bin PRN support without computing detector scores."""
    _validate_parameters(
        start_s=start_s, end_s=end_s, bin_seconds=bin_seconds,
        minimum_epochs=minimum_epochs, minimum_primary_prns=minimum_primary_prns,
        secondary_boundary_prns=secondary_boundary_prns,
        minimum_primary_bins=minimum_primary_bins,
    )
    run_dir = Path(receiver_run_dir).resolve()
    manifest = _manifest(run_dir)
    loaded = _mat_counts(
        run_dir, manifest, start_s=start_s, end_s=end_s, bin_seconds=bin_seconds,
        require_complex_nine_tap=require_complex_nine_tap,
    )
    if loaded is None:
        loaded = _csv_counts(
            run_dir, manifest, start_s=start_s, end_s=end_s, bin_seconds=bin_seconds,
            require_complex_nine_tap=require_complex_nine_tap,
        )
    channel_counts, discovered, source = loaded

    # A duplicated PRN on several channels must not inflate its epoch support.
    per_prn_bin: dict[tuple[int, int], int] = {}
    for (bin_index, prn, _channel), count in channel_counts.items():
        key = (bin_index, prn)
        per_prn_bin[key] = max(per_prn_bin.get(key, 0), count)

    maximum_epoch_count = max(per_prn_bin.values(), default=0)
    telemetry_density_sufficient = maximum_epoch_count >= minimum_epochs

    first_bin = int(math.floor(start_s / bin_seconds))
    final_bin = int(math.ceil(end_s / bin_seconds))
    support_by_bin: dict[int, int] = {}
    eligible_prns_by_bin: dict[int, list[int]] = {}
    for bin_index in range(first_bin, final_bin):
        prns = sorted(
            prn for (candidate_bin, prn), count in per_prn_bin.items()
            if candidate_bin == bin_index and count >= minimum_epochs
        )
        support_by_bin[bin_index] = len(prns)
        eligible_prns_by_bin[bin_index] = prns

    primary_bins = [index for index, count in support_by_bin.items() if count >= minimum_primary_prns]
    secondary_bins = [index for index, count in support_by_bin.items() if count == secondary_boundary_prns]
    distribution: dict[str, int] = {}
    for count in support_by_bin.values():
        key = str(count)
        distribution[key] = distribution.get(key, 0) + 1
    support_eligible = telemetry_density_sufficient and len(primary_bins) >= minimum_primary_bins
    if not telemetry_density_sufficient:
        status = "INSUFFICIENT_TELEMETRY_DENSITY"
    elif support_eligible:
        status = "SUPPORT_ELIGIBLE"
    else:
        status = "INSUFFICIENT_SUPPORT"
    return {
        "schema": "gnss-doppler-lab.clean-geometry-support-audit",
        "schema_version": 1,
        "receiver_run_dir": str(run_dir),
        "score_accessed": False,
        "attack_payload_accessed": False,
        "rules": {
            "analysis_interval_seconds": [start_s, end_s],
            "bin_seconds": bin_seconds,
            "minimum_epochs_per_prn_bin": minimum_epochs,
            "minimum_primary_prns": minimum_primary_prns,
            "secondary_boundary_prns": secondary_boundary_prns,
            "minimum_primary_bins": minimum_primary_bins,
            "duplicate_channel_rule": "maximum epoch count per PRN/bin",
            "require_complex_nine_tap": require_complex_nine_tap,
        },
        "source": source,
        "discovered_prns": [f"G{prn:02d}" for prn in sorted(discovered)],
        "discovered_prn_count": len(discovered),
        "maximum_epoch_count_per_prn_bin": maximum_epoch_count,
        "telemetry_density_sufficient": telemetry_density_sufficient,
        "maximum_eligible_prns": max(support_by_bin.values(), default=0),
        "eligible_bin_count_by_prn_count": dict(sorted(distribution.items(), key=lambda row: int(row[0]))),
        "primary_bin_count": len(primary_bins),
        "primary_bins": primary_bins,
        "secondary_boundary_bin_count": len(secondary_bins),
        "secondary_boundary_bins": secondary_bins,
        "eligible_prns_by_bin": {str(key): value for key, value in eligible_prns_by_bin.items()},
        "support_eligible": support_eligible,
        "status": status,
    }
