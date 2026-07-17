from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from gnss_doppler_lab.tracking_peaks import TrackingPeakSeries
from gnss_doppler_lab.tracking_feature_windows import (
    compute_tracking_window_feature_records,
    export_receiver_run_tracking_feature_csv,
)


def _series_with_five_taps() -> TrackingPeakSeries:
    return TrackingPeakSeries(
        prn="G05",
        channel=0,
        sample_rate_hz=10,
        time_s=np.array([0.0, 0.25, 0.5, 0.75, 0.99], dtype=np.float64),
        tap_names=("E", "P", "L"),
        magnitudes=np.array(
            [
                [4.0, 10.0, 3.0],
                [4.2, 10.0, 3.1],
                [4.4, 10.0, 3.2],
                [4.6, 10.0, 3.3],
                [4.8, 10.0, 3.4],
            ],
            dtype=np.float64,
        ),
        carrier_doppler_hz=np.array([100.0, 101.0, 103.0, 106.0, 110.0], dtype=np.float64),
        cn0_db_hz=np.array([45.0, 45.5, 46.0, 46.5, 47.0], dtype=np.float64),
        prompt_i=np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64),
        prompt_q=np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64),
        code_error_chips=np.array([-0.1, -0.05, 0.0, 0.05, 0.1], dtype=np.float64),
        code_freq_chips=np.array([1023000.0, 1023000.2, 1023000.4, 1023000.6, 1023000.8], dtype=np.float64),
        source_mat_path=Path("/tmp/epl_tracking_ch_0.mat"),
    )


def _write_tracking_mat(path: Path, *, prn: int, samples: list[int]) -> None:
    with h5py.File(path, "w") as handle:
        values = {
            "PRN": np.array(samples, dtype=np.uint32) * 0 + prn,
            "PRN_start_sample_count": np.array(samples, dtype=np.uint64),
            "carrier_doppler_hz": np.array([100.0, 101.0, 103.0, 106.0, 110.0], dtype=np.float32),
            "CN0_SNV_dB_Hz": np.array([45.0, 45.5, 46.0, 46.5, 47.0], dtype=np.float32),
            "Prompt_I": np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32),
            "Prompt_Q": np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "code_error_chips": np.array([-0.1, -0.05, 0.0, 0.05, 0.1], dtype=np.float32),
            "code_freq_chips": np.array([1023000.0, 1023000.2, 1023000.4, 1023000.6, 1023000.8], dtype=np.float32),
            "abs_VE": np.array([1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float32),
            "abs_E": np.array([4.0, 4.2, 4.4, 4.6, 4.8], dtype=np.float32),
            "abs_P": np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32),
            "abs_L": np.array([3.0, 3.1, 3.2, 3.3, 3.4], dtype=np.float32),
            "abs_VL": np.array([0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float32),
        }
        for key, value in values.items():
            handle.create_dataset(key, data=value.reshape(-1, 1))


def _receiver_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "paper_run"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)
    _write_tracking_mat(raw / "epl_tracking_ch_0.mat", prn=5, samples=[0, 2, 5, 7, 9])
    manifest = {
        "source": {"sample_rate_hz": 10},
        "tracking": {"raw_directory": "raw", "prns": ["G05"]},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def test_compute_tracking_window_feature_records_returns_expected_feature_statistics() -> None:
    records = compute_tracking_window_feature_records(
        _series_with_five_taps(),
        receiver_run_id="paper-run",
        window_s=1.0,
        stride_s=0.5,
        min_epochs=4,
        label="normal",
    )

    assert len(records) == 1
    record = records[0]
    assert record.run_id == "paper-run"
    assert record.prn == "G05"
    assert record.label == "normal"
    assert record.epoch_count == 5
    assert record.window_start_s == pytest.approx(0.0)
    assert record.window_end_s == pytest.approx(1.0)
    assert record.near_sym_mean == pytest.approx(0.1573450754, rel=1e-6)
    assert record.sharp_narrow_mean == pytest.approx(1.2399998760, rel=1e-6)
    assert record.doppler_std == pytest.approx(3.6331804249, rel=1e-6)
    assert record.prompt_mag_cv == pytest.approx(0.00243853127, rel=1e-6)


def test_export_receiver_run_tracking_feature_csv_writes_paper_ready_rows(tmp_path: Path) -> None:
    run_dir = _receiver_run(tmp_path)
    output = tmp_path / "tracking_feature_windows.csv"

    written = export_receiver_run_tracking_feature_csv(
        run_dir,
        output_path=output,
        window_s=1.0,
        stride_s=0.5,
        min_epochs=4,
        label="normal",
    )

    with written.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert written == output
    assert len(rows) == 1
    assert rows[0]["run_id"] == "paper_run"
    assert rows[0]["prn"] == "G05"
    assert rows[0]["label"] == "normal"
    assert rows[0]["epoch_count"] == "5"
    assert "near_sym_mean" in rows[0]
    assert "code_err_std" in rows[0]
    assert rows[0]["segment_index"] == "0"


def test_current_paper_schema_is_explicit_three_tap_epl_and_rejects_five_tap_placeholder() -> None:
    from gnss_doppler_lab.isolation_forest_baseline import MORPHOLOGY_FEATURE_COLUMNS, FEATURE_GROUPS
    assert MORPHOLOGY_FEATURE_COLUMNS == [
        "near_sym_mean", "near_sym_std", "sharp_narrow_mean",
        "sharp_narrow_std", "sharp_narrow_slope",
    ]
    assert len(FEATURE_GROUPS["combined"]) == 11
    series = _series_with_five_taps()
    from dataclasses import replace
    five = replace(series, tap_names=("VE", "E", "P", "L", "VL"), magnitudes=np.column_stack((np.zeros(5), series.magnitudes, np.zeros(5))))
    with pytest.raises(ValueError, match="exactly real E/P/L"):
        compute_tracking_window_feature_records(five, receiver_run_id="bad")


def test_collection_never_windows_across_prn_or_time_gap_segments_and_honors_filter(tmp_path: Path) -> None:
    from gnss_doppler_lab.tracking_feature_windows import collect_receiver_run_tracking_feature_records
    from test_tracking_peaks import _multi_prn_receiver_run
    run_dir = _multi_prn_receiver_run(tmp_path)
    records = collect_receiver_run_tracking_feature_records(run_dir, window_s=1.0, stride_s=1.0, min_epochs=2)
    assert [(r.prn, r.segment_index, r.window_index, r.epoch_count) for r in records] == [
        ("G31", 0, 0, 2), ("G31", 2, 0, 2), ("G01", 1, 0, 2), ("G02", 3, 0, 2)
    ]
    filtered = collect_receiver_run_tracking_feature_records(run_dir, window_s=1.0, stride_s=1.0, min_epochs=2, prns=["G01"])
    assert {(r.prn, r.segment_index) for r in filtered} == {("G01", 1)}
    assert all(r.window_end_s - r.window_start_s == pytest.approx(1.0) for r in records)
