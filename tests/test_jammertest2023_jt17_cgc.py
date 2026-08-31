from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RECEIVER = load("jt17_receiver", "scripts/run_jammertest2023_jt17_receiver.py")
SUPPORT = load("jt17_support", "scripts/audit_jammertest2023_jt17_support.py")
DETECTOR = load("jt17_detector", "scripts/run_jammertest2023_jt17_cgc_detection.py")
CONFIG = json.loads(RECEIVER.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_frozen_config_validates_for_all_phases() -> None:
    RECEIVER.validate_config(CONFIG)
    SUPPORT.validate_config(CONFIG)
    DETECTOR.validate_config(CONFIG)


def test_interleaved_ibyte_item_count_and_receiver_contract() -> None:
    assert RECEIVER.ibyte_source_item_count(560.0, 30_690_000) == 34_372_800_000
    text = RECEIVER.render_config(Path("/tmp/source.iq"), Path("/tmp/out"), CONFIG)
    assert "SignalSource.item_type=ibyte" in text
    assert "DataTypeAdapter.implementation=Ibyte_To_Complex" in text
    assert "InputFilter.decimation_factor=5" in text
    assert "GNSS-SDR.internal_fs_sps=6138000" in text
    assert "Tracking_1C.tap_count=9" in text
    assert "Tracking_1C.tap_spacing_chips=0.125" in text


def test_observables_tow_intercept_fit(tmp_path: Path) -> None:
    path = tmp_path / "observables.mat"
    tow0 = 345_678.25
    rows = np.arange(800, dtype=np.float64)
    rx = np.zeros((800, 4), dtype=np.float64)
    rx[100:, :] = (tow0 + rows[100:] * 0.02)[:, None]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("RX_time", data=rx)
    result = SUPPORT.infer_recording_start_tow(
        path, cadence_s=0.02, max_residual_s=0.021
    )
    assert abs(result["recording_start_tow_s"] - tow0) < 1e-9
    assert result["first_valid_row_index"] == 100
    assert result["maximum_absolute_fit_residual_s"] < 1e-9


def test_observables_tow_fit_excludes_initial_clock_settle(tmp_path: Path) -> None:
    path = tmp_path / "observables-step.mat"
    tow0 = 311_646.78
    rows = np.arange(2_000, dtype=np.float64)
    rx = (tow0 + rows * 0.02)[:, None]
    rx[:250] -= 0.08
    with h5py.File(path, "w") as handle:
        handle.create_dataset("RX_time", data=rx)
    result = SUPPORT.infer_recording_start_tow(
        path, cadence_s=0.02, max_residual_s=0.021, minimum_stable_rows=1_000
    )
    assert abs(result["recording_start_tow_s"] - tow0) < 1e-9
    assert result["clock_step_count"] == 1
    assert result["first_stable_row_index"] == 250
    assert result["excluded_clock_settle_row_count"] == 250


def test_region_boundaries_and_motion_label_are_frozen() -> None:
    assert CONFIG["dataset"]["official_spoof_rf_onset_s"] == 226.0
    assert CONFIG["dataset"]["planned_carryoff_motion_onset_s"] == 526.0
    assert DETECTOR.region(40.0, CONFIG) == "clean"
    assert DETECTOR.region(199.0, CONFIG) == "clean"
    assert DETECTOR.region(226.0, CONFIG) == "excluded"
    assert DETECTOR.region(246.0, CONFIG) == "aligned_spoof"
    assert DETECTOR.region(499.0, CONFIG) == "aligned_spoof"
    assert DETECTOR.region(526.0, CONFIG) == "carryoff_onset"
    assert DETECTOR.region(555.0, CONFIG) == "carryoff_onset"
    assert DETECTOR.region(556.0, CONFIG) == "excluded"


def test_threshold_and_persistence_are_unchanged() -> None:
    detector = CONFIG["frozen_detector"]
    assert detector["partial_f_p_alarm_threshold"] == 0.06028418845288192
    assert detector["persistence_window_bins"] == 5
    assert detector["persistence_required_bins"] == 3
    assert CONFIG["support"]["minimum_epochs_per_prn_bin"] == 40
