from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/run_tuni_gps_partial_spoof_external.py"
CONFIG = ROOT / "configs/experiments/tuni_gps_partial_spoof_external_v1.json"


def _load():
    spec = importlib.util.spec_from_file_location("tuni_gps_external", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def test_render_receiver_config_is_contiguous_big_endian_complex9(tmp_path: Path) -> None:
    text = MODULE.render_receiver_config(
        iq_path=tmp_path / "input.bin",
        output_dir=tmp_path / "receiver",
        duration_s=149.99916,
    )
    assert "SignalSource.item_type=ishort" in text
    assert "DataTypeAdapter.swap_endian=true" in text
    assert "SignalSource.samples=0" in text
    assert "SignalSource.sampling_frequency=50000000" in text
    assert "Resampler.sample_freq_out=5000000" in text
    assert "Channels_1C.count=31" in text
    assert "Tracking_1C.tap_count=9" in text
    assert "Tracking_1C.tap_spacing_chips=0.125" in text
    assert "PVT.nmea_output_enabled=true" in text
    assert "PVT.nmea_dump_filename=nmea_pvt.nmea" in text
    assert "SignalSource.seconds_to_skip" not in text


def test_render_receiver_config_rejects_short_or_nonintegral_rate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duration"):
        MODULE.render_receiver_config(
            iq_path=tmp_path / "input.bin", output_dir=tmp_path, duration_s=10.0
        )
    with pytest.raises(ValueError, match="integral"):
        MODULE.render_receiver_config(
            iq_path=tmp_path / "input.bin", output_dir=tmp_path,
            duration_s=100.0, input_rate_hz=50_000_001,
        )


def test_clean_static_position_falls_back_to_same_run_pvt_log(tmp_path: Path) -> None:
    receiver_log = tmp_path / "receiver.log"
    receiver_log.write_text(
        "Current receiver time: 59 s\n"
        "Position at t using 4 observations is Lat = 1.0 [deg], "
        "Long = 2.0 [deg], Height = 3.0 [m]\n"
        "Current receiver time: 60 s\n"
        "Position at t using 4 observations is Lat = 61.1 [deg], "
        "Long = 23.8 [deg], Height = 180.0 [m]\n"
        "Current receiver time: 1 min 40 s\n"
        "Position at t using 5 observations is Lat = 61.3 [deg], "
        "Long = 24.0 [deg], Height = 200.0 [m]\n"
        "Current receiver time: 2 min 20 s\n"
        "Position at t using 5 observations is Lat = 9.0 [deg], "
        "Long = 9.0 [deg], Height = 9.0 [m]\n",
        encoding="utf-8",
    )
    result = MODULE.clean_static_position(
        tmp_path / "missing.nmea", 0.0, receiver_log
    )
    assert result["llh"] == pytest.approx((61.2, 23.9, 190.0))
    assert result["sample_count"] == 2
    assert result["relative_time_range_s"] == [60.0, 100.0]
    assert "display log" in result["source"]


def _summary(*, bins: int = 100, target: bool = True, detected: bool = False, rate: float = 0.0):
    return {
        "primary_bin_count": bins,
        "all_documented_spoof_prns_tracked": target,
        "persistent_alarm_rate": rate,
        "detected": detected,
    }


def test_terminal_decision_supported() -> None:
    config = json.loads(CONFIG.read_text())
    summaries = {
        "C-5": _summary(rate=0.04),
        "SS-17": _summary(detected=False),
        "SS-18": _summary(detected=True),
        "SS-20": _summary(detected=True),
    }
    result = MODULE.terminal_decision(summaries, config)
    assert result["decision"] == "REAL_PARTIAL_SPOOF_TRANSFER_SUPPORTED"
    assert result["clean_specificity_pass"] is True
    assert result["attack_sensitivity_pass"] is True


def test_terminal_decision_specificity_only() -> None:
    config = json.loads(CONFIG.read_text())
    summaries = {
        "C-5": _summary(rate=0.0),
        "SS-17": _summary(detected=True),
        "SS-18": _summary(detected=False),
        "SS-20": _summary(detected=False),
    }
    result = MODULE.terminal_decision(summaries, config)
    assert result["decision"] == "SPECIFICITY_ONLY_DETECTION_NOT_SUPPORTED"


def test_terminal_decision_fails_closed_on_missing_target_or_bins() -> None:
    config = json.loads(CONFIG.read_text())
    summaries = {
        "C-5": _summary(rate=0.0),
        "SS-17": _summary(target=False, detected=True),
        "SS-18": _summary(detected=True),
        "SS-20": _summary(bins=59, detected=True),
    }
    result = MODULE.terminal_decision(summaries, config)
    assert result["decision"] == "INSUFFICIENT_SUPPORT"
    assert result["support_sufficient"] is False


def test_frozen_config_encodes_partial_spoofer_boundary() -> None:
    config = json.loads(CONFIG.read_text())
    MODULE.validate_config(config)
    assert config["analysis"]["minimum_primary_prns"] == 8
    assert config["analysis"]["secondary_boundary_prns"] == 7
    assert config["analysis"]["analysis_interval_seconds"] == [30.0, 140.0]
    assert [row["spoofed_prns"] for row in config["dataset"]["scenarios"]] == [
        [], [1], [1, 2], [1, 2, 21, 32]
    ]
