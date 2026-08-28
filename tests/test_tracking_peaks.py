
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from gnss_doppler_lab.tracking_peaks import (
    available_tracking_prns,
    load_receiver_tracking_peak_series,
    render_tracking_peak_dashboard,
)


PNG_SIGNATURE = bytes([137, 80, 78, 71, 13, 10, 26, 10])


def _peak_tracking_mat(path: Path, *, prn: int, samples: list[int]) -> None:
    with h5py.File(path, "w") as handle:
        values = {
            "PRN": np.array(samples, dtype=np.uint32) * 0 + prn,
            "PRN_start_sample_count": np.array(samples, dtype=np.uint64),
            "carrier_doppler_hz": np.array([1200.0, 1205.0, 1210.0], dtype=np.float32),
            "CN0_SNV_dB_Hz": np.array([44.0, 45.0, 46.0], dtype=np.float32),
            "Prompt_I": np.array([10.0, 11.0, 12.0], dtype=np.float32),
            "Prompt_Q": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "code_error_chips": np.array([-0.1, -0.05, 0.0], dtype=np.float32),
            "code_freq_chips": np.array([1023000.5, 1023000.75, 1023001.0], dtype=np.float32),
            "abs_VE": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "abs_E": np.array([3.0, 4.0, 5.0], dtype=np.float32),
            "abs_P": np.array([9.0, 10.0, 11.0], dtype=np.float32),
            "abs_L": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            "abs_VL": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }
        for key, value in values.items():
            handle.create_dataset(key, data=value.reshape(-1, 1))


def _receiver_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "receiver_run"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)
    _peak_tracking_mat(raw / "epl_tracking_ch_0.mat", prn=5, samples=[2600, 5200, 7800])
    _peak_tracking_mat(raw / "epl_tracking_ch_1.mat", prn=18, samples=[2600, 5200, 7800])
    manifest = {
        "source": {"sample_rate_hz": 2_600_000},
        "tracking": {"raw_directory": "raw", "prns": ["G05", "G18"]},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


def test_available_tracking_prns_reads_manifest_order(tmp_path: Path) -> None:
    run_dir = _receiver_run(tmp_path)

    assert available_tracking_prns(run_dir) == ["G05", "G18"]


def test_available_tracking_prns_rejects_empty_matlab_sentinel(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty_receiver_run"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)
    with h5py.File(raw / "epl_tracking_ch_0.mat", "w") as handle:
        handle.create_dataset("PRN", data=np.asarray([1, 0], dtype=np.uint64))
    (run_dir / "manifest.json").write_text(json.dumps({
        "source": {"sample_rate_hz": 10},
        "tracking": {"raw_directory": "raw"},
    }))

    assert available_tracking_prns(run_dir) == []
    with pytest.raises(FileNotFoundError, match="G01"):
        load_receiver_tracking_peak_series(run_dir, "G01")


def test_load_receiver_tracking_peak_series_extracts_real_prn_peak_slice(tmp_path: Path) -> None:
    run_dir = _receiver_run(tmp_path)

    series = load_receiver_tracking_peak_series(run_dir, "G05", max_epochs=2)

    assert series.prn == "G05"
    assert series.channel == 0
    assert series.tap_names == ("E", "P", "L")
    assert series.magnitudes.shape == (2, 3)
    assert np.allclose(series.time_s, np.array([0.001, 0.002]))
    assert np.allclose(series.magnitudes[0], np.array([3.0, 9.0, 4.0]))


def test_receiver_start_offset_is_added_to_tracking_time(tmp_path: Path) -> None:
    run_dir = _receiver_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"]["start_offset_s"] = 7.0
    manifest_path.write_text(json.dumps(manifest))

    series = load_receiver_tracking_peak_series(run_dir, "G05", max_epochs=2)

    assert np.allclose(series.time_s, np.array([7.001, 7.002]))


def test_render_tracking_peak_dashboard_writes_png(tmp_path: Path) -> None:
    run_dir = _receiver_run(tmp_path)
    output = tmp_path / "tracking_peak_dashboard.png"

    series = load_receiver_tracking_peak_series(run_dir, "G18")
    render_tracking_peak_dashboard(series, output_path=output, title="Tracking peak dashboard")

    assert output.read_bytes().startswith(PNG_SIGNATURE)
    assert output.stat().st_size > 10_000


def _multi_prn_receiver_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "multi_prn_run"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)
    prns = np.array([31, 31, 1, 1, 31, 31, 2, 2], dtype=np.uint32)
    samples = np.array([0, 10, 20, 30, 100, 110, 120, 130], dtype=np.uint64)
    with h5py.File(raw / "epl_tracking_ch_5.mat", "w") as handle:
        n = len(prns)
        values = {
            "PRN": prns,
            "PRN_start_sample_count": samples,
            "carrier_doppler_hz": np.arange(n, dtype=np.float32) + 100,
            "CN0_SNV_dB_Hz": np.arange(n, dtype=np.float32) + 40,
            "Prompt_I": np.arange(n, dtype=np.float32) + 10,
            "Prompt_Q": np.arange(n, dtype=np.float32) + 1,
            "code_error_chips": np.arange(n, dtype=np.float32) / 100,
            "code_freq_chips": np.arange(n, dtype=np.float32) + 1023000,
            "abs_E": np.arange(n, dtype=np.float32) + 3,
            "abs_P": np.arange(n, dtype=np.float32) + 9,
            "abs_L": np.arange(n, dtype=np.float32) + 4,
        }
        for key, value in values.items():
            handle.create_dataset(key, data=value.reshape(-1, 1))
    manifest = {"source": {"sample_rate_hz": 10}, "tracking": {"raw_directory": "raw", "prns": ["G31"]}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


def test_multi_prn_channel_discovers_all_epoch_prns_and_filters_requested_prn(tmp_path: Path) -> None:
    run_dir = _multi_prn_receiver_run(tmp_path)
    assert available_tracking_prns(run_dir) == ["G31", "G01", "G02"]
    series = load_receiver_tracking_peak_series(run_dir, "G01")
    assert series.prn == "G01"
    assert series.channel == 5
    assert series.time_s.tolist() == [2.0, 3.0]
    assert series.carrier_doppler_hz.tolist() == [102.0, 103.0]


def test_same_prn_reappearance_is_returned_as_distinct_segments(tmp_path: Path) -> None:
    from gnss_doppler_lab.tracking_peaks import load_receiver_tracking_peak_series_segments
    segments = load_receiver_tracking_peak_series_segments(_multi_prn_receiver_run(tmp_path), "G31")
    assert [segment.segment_index for segment in segments] == [0, 2]
    assert [segment.time_s.tolist() for segment in segments] == [[0.0, 1.0], [10.0, 11.0]]


def _nine_tap_tracking_mat(
    path: Path, *, prn: int, samples: list[int], complex_taps: bool = False
) -> None:
    with h5py.File(path, "w") as handle:
        n = len(samples)
        values = {
            "PRN": np.array(samples, dtype=np.uint32) * 0 + prn,
            "PRN_start_sample_count": np.array(samples, dtype=np.uint64),
            "carrier_doppler_hz": np.arange(n, dtype=np.float32) + 100.0,
            "CN0_SNV_dB_Hz": np.arange(n, dtype=np.float32) + 45.0,
            "Prompt_I": np.arange(n, dtype=np.float32) + 10.0,
            "Prompt_Q": np.arange(n, dtype=np.float32),
            "code_error_chips": np.arange(n, dtype=np.float32) / 100.0,
            "code_freq_chips": np.arange(n, dtype=np.float32) + 1023000.0,
            "abs_E4": np.arange(n, dtype=np.float32) * 0.1 + 1.0,
            "abs_E3": np.arange(n, dtype=np.float32) * 0.1 + 2.0,
            "abs_E2": np.arange(n, dtype=np.float32) * 0.1 + 3.0,
            "abs_E": np.arange(n, dtype=np.float32) * 0.1 + 4.0,
            "abs_P": np.arange(n, dtype=np.float32) * 0.1 + 10.0,
            "abs_L": np.arange(n, dtype=np.float32) * 0.1 + 5.0,
            "abs_L2": np.arange(n, dtype=np.float32) * 0.1 + 3.5,
            "abs_L3": np.arange(n, dtype=np.float32) * 0.1 + 2.5,
            "abs_L4": np.arange(n, dtype=np.float32) * 0.1 + 1.5,
        }
        if complex_taps:
            for label, magnitude_name in (
                ("E4", "abs_E4"), ("E3", "abs_E3"), ("E2", "abs_E2"),
                ("E", "abs_E"), ("P", "abs_P"), ("L", "abs_L"),
                ("L2", "abs_L2"), ("L3", "abs_L3"), ("L4", "abs_L4"),
            ):
                values[f"tap_I_{label}"] = values[magnitude_name].copy()
                values[f"tap_Q_{label}"] = np.zeros(n, dtype=np.float32)
        for key, value in values.items():
            handle.create_dataset(key, data=value.reshape(-1, 1))


def test_load_receiver_tracking_peak_series_can_select_real_nine_tap_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "nine_tap_run"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)
    _nine_tap_tracking_mat(raw / "epl_tracking_ch_0.mat", prn=5, samples=[10, 20, 30])
    (run_dir / "manifest.json").write_text(json.dumps({
        "source": {"sample_rate_hz": 10},
        "tracking": {"raw_directory": "raw", "prns": ["G05"], "tap_count": 9},
    }))

    series = load_receiver_tracking_peak_series(run_dir, "G05", tap_count=9)

    assert series.tap_names == ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")
    assert series.magnitudes.shape == (3, 9)
    assert np.allclose(series.magnitudes[0], [1.0, 2.0, 3.0, 4.0, 10.0, 5.0, 3.5, 2.5, 1.5])


def test_load_receiver_tracking_peak_series_requires_consistent_complex_nine_taps(tmp_path: Path) -> None:
    run_dir = tmp_path / "complex_nine_tap_run"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)
    mat = raw / "epl_tracking_ch_0.mat"
    _nine_tap_tracking_mat(mat, prn=5, samples=[10, 20, 30], complex_taps=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "source": {"sample_rate_hz": 10},
        "tracking": {"raw_directory": "raw", "prns": ["G05"], "tap_count": 9},
    }))

    series = load_receiver_tracking_peak_series(
        run_dir, "G05", tap_count=9, require_complex_taps=True
    )

    assert series.has_complex_taps
    assert series.complex_taps.shape == (3, 9)
    assert np.allclose(np.abs(series.complex_taps), series.magnitudes)
    assert np.allclose(series.complex_taps.imag, 0.0)


def test_complex_nine_tap_requirement_rejects_missing_or_partial_dump(tmp_path: Path) -> None:
    run_dir = tmp_path / "magnitude_only_nine_tap_run"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)
    mat = raw / "epl_tracking_ch_0.mat"
    _nine_tap_tracking_mat(mat, prn=5, samples=[10, 20, 30])
    (run_dir / "manifest.json").write_text(json.dumps({
        "source": {"sample_rate_hz": 10},
        "tracking": {"raw_directory": "raw", "prns": ["G05"], "tap_count": 9},
    }))
    with pytest.raises(ValueError, match="missing required complex"):
        load_receiver_tracking_peak_series(
            run_dir, "G05", tap_count=9, require_complex_taps=True
        )
    with h5py.File(mat, "a") as handle:
        handle.create_dataset("tap_I_E4", data=np.ones((3, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="partial complex"):
        load_receiver_tracking_peak_series(run_dir, "G05", tap_count=9)
