from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from gnss_doppler_lab.clean_geometry_support import (
    COMPLEX_TAP_LABELS,
    audit_clean_geometry_support,
)


def _manifest(run_dir: Path, *, csv_evidence: bool = False) -> None:
    tracking = {"raw_directory": "raw", "tap_count": 9}
    if csv_evidence:
        tracking["component_order"] = ["I", "Q"]
    (run_dir / "manifest.json").write_text(json.dumps({
        "source": {"sample_rate_hz": 1000},
        "tracking": tracking,
    }))


def _mat(path: Path, prn: int, times_s: np.ndarray, *, complex_taps: bool = True) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("PRN", data=np.full(len(times_s), prn))
        handle.create_dataset("PRN_start_sample_count", data=times_s * 1000)
        if complex_taps:
            for label in COMPLEX_TAP_LABELS:
                handle.create_dataset(f"tap_I_{label}", data=np.ones(len(times_s)))
                handle.create_dataset(f"tap_Q_{label}", data=np.ones(len(times_s)))


def test_mat_gate_accepts_eight_supported_prns(tmp_path: Path) -> None:
    run_dir = tmp_path / "receiver"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)
    _manifest(run_dir)
    times = np.arange(0.05, 0.30, 0.001)
    for channel, prn in enumerate(range(1, 9)):
        _mat(raw / f"epl_tracking_ch_{channel}.mat", prn, times)
    result = audit_clean_geometry_support(
        run_dir, start_s=0.0, end_s=1.0, minimum_epochs=200,
        minimum_primary_bins=1,
    )
    assert result["status"] == "SUPPORT_ELIGIBLE"
    assert result["maximum_eligible_prns"] == 8
    assert result["primary_bin_count"] == 1
    assert result["score_accessed"] is False
    assert result["attack_payload_accessed"] is False


def test_duplicate_channels_do_not_combine_epoch_counts(tmp_path: Path) -> None:
    run_dir = tmp_path / "receiver"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)
    _manifest(run_dir)
    first = np.arange(0.05, 0.20, 0.001)
    second = np.arange(0.30, 0.45, 0.001)
    _mat(raw / "epl_tracking_ch_0.mat", 1, first)
    _mat(raw / "epl_tracking_ch_1.mat", 1, second)
    result = audit_clean_geometry_support(
        run_dir, start_s=0.0, end_s=1.0, minimum_epochs=200,
        minimum_primary_bins=1,
    )
    assert result["maximum_eligible_prns"] == 0
    assert result["maximum_epoch_count_per_prn_bin"] == max(len(first), len(second))
    assert result["status"] == "INSUFFICIENT_TELEMETRY_DENSITY"


def test_csv_fallback_uses_manifest_complex_tap_attestation(tmp_path: Path) -> None:
    run_dir = tmp_path / "receiver"
    run_dir.mkdir()
    _manifest(run_dir, csv_evidence=True)
    with (run_dir / "tracking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", "channel", "prn"])
        writer.writeheader()
        for channel, prn in enumerate(range(1, 8)):
            for time_s in np.arange(0.05, 0.30, 0.001):
                writer.writerow({"time_s": time_s, "channel": channel, "prn": f"G{prn:02d}"})
    result = audit_clean_geometry_support(
        run_dir, start_s=0.0, end_s=1.0, minimum_epochs=200,
        minimum_primary_bins=1,
    )
    assert result["source"]["kind"] == "tracking_csv"
    assert result["secondary_boundary_bin_count"] == 1
    assert result["status"] == "INSUFFICIENT_SUPPORT"


def test_gate_rejects_missing_complex_taps(tmp_path: Path) -> None:
    run_dir = tmp_path / "receiver"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)
    _manifest(run_dir)
    _mat(raw / "epl_tracking_ch_0.mat", 1, np.arange(0.05, 0.30, 0.001), complex_taps=False)
    with pytest.raises(ValueError, match="complex nine-tap"):
        audit_clean_geometry_support(
            run_dir, start_s=0.0, end_s=1.0, minimum_epochs=200,
            minimum_primary_bins=1,
        )
