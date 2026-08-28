import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RECEIVER = _load("run_gnss_openif_s1_receiver", "scripts/run_gnss_openif_s1_receiver.py")
EVALUATE = _load("evaluate_gnss_openif_s1_cgc", "scripts/evaluate_gnss_openif_s1_cgc.py")


def test_receiver_config_translates_and_decimates_labsat_if(tmp_path: Path) -> None:
    iq = tmp_path / "s1.bin"
    output = tmp_path / "receiver"
    config = RECEIVER.render_config(
        iq_path=iq, output_dir=output, duration_s=10.0,
        start_offset_s=2.5, channel_count=16,
    )

    assert "SignalSource.item_type=ibyte" in config
    assert "SignalSource.sampling_frequency=58000000" in config
    assert "SignalSource.seconds_to_skip=2.5" in config
    assert "SignalSource.samples=1160000000" in config
    assert "InputFilter.implementation=Freq_Xlating_Fir_Filter" in config
    # The official IF magnitude is 4.58 MHz, but the recorded I+jQ spectrum
    # places GPS L1 at -4.58 MHz.
    assert "InputFilter.IF=-4580000" in config
    assert "InputFilter.decimation_factor=10" in config
    assert "GNSS-SDR.internal_fs_sps=5800000" in config
    assert "Acquisition_1C.pfa=0.01" in config
    assert "Acquisition_1C.max_dwells=5" in config
    assert "Tracking_1C.tap_count=9" in config
    assert "Channel0.satellite=22" in config
    assert "Channel1.satellite=5" in config


def test_receiver_full_file_uses_zero_source_item_limit(tmp_path: Path) -> None:
    config = RECEIVER.render_config(
        iq_path=tmp_path / "s1.bin",
        output_dir=tmp_path / "receiver",
        duration_s=0.0,
    )
    assert "SignalSource.samples=0" in config


def test_ground_truth_parser_and_interpolation(tmp_path: Path) -> None:
    path = tmp_path / "gt.txt"
    path.write_text(
        "header\nunits\n"
        "1000 2353 10 22 114 3 1 2 3\n"
        "1001 2353 11 22 114 3 3 4 5\n",
        encoding="utf-8",
    )
    truth = EVALUATE.load_ground_truth(path)
    assert EVALUATE.interpolate_ecef(truth, 10.5).tolist() == [2.0, 3.0, 4.0]


def test_support_normalized_alarm_rule_is_causal_three_of_five() -> None:
    rows = [
        {
            "bin_index": index,
            "prn_count": 7,
            "clock_centered_geometry_residual": residual,
            "q75_prn_early_late_asymmetry": 0.08,
        }
        for index, residual in enumerate([0.7, 0.05, 0.7, 0.05, 0.05])
    ]
    config = {
        "frozen_detector": {
            "residual_alarm_threshold": 0.33,
            "multipath_enrichment_threshold": 0.063,
            "persistence_window_bins": 5,
            "persistence_required_bins": 3,
        },
        "support_normalization": {
            "partial_f_p_alarm_threshold": 0.06028418845288192,
        },
    }
    EVALUATE.apply_frozen_alarm(rows, config)
    assert [row["persistent_spoof_alarm"] for row in rows] == [False, False, False, False, True]
    assert rows[-1]["detector_classification"] == "spoof_alarm"


def test_partial_f_corrects_for_available_satellite_support() -> None:
    residual = 0.2
    seven_prn_p = EVALUATE.partial_f_p_value(residual, 7)
    twelve_prn_p = EVALUATE.partial_f_p_value(residual, 12)

    assert seven_prn_p > twelve_prn_p
    assert seven_prn_p > 0.06028418845288192
    assert twelve_prn_p < 0.06028418845288192


def test_profile_rows_consolidates_duplicate_prn_channels(monkeypatch) -> None:
    class Estimator:
        def estimate(self, features):
            count = len(features)
            return np.full(count, 0.1), np.full(count, 0.2), None

    epoch_count = 300
    segment = SimpleNamespace(
        time_s=np.linspace(1.001, 1.299, epoch_count),
        complex_taps=np.ones((epoch_count, 9), dtype=np.complex64),
        cn0_db_hz=np.full(epoch_count, 40.0),
    )
    monkeypatch.setattr(EVALUATE, "available_tracking_prns", lambda _: ["G09"])
    monkeypatch.setattr(
        EVALUATE,
        "load_receiver_tracking_peak_series_segments",
        lambda *args, **kwargs: [segment, segment],
    )

    rows = EVALUATE.profile_rows(
        Path("unused"),
        Estimator(),
        bin_seconds=1.0,
        minimum_epochs=200,
        start_s=0.0,
        end_s=2.0,
    )

    assert len(rows) == 1
    assert rows[0]["epoch_count"] == epoch_count
