from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "doppler_silent_hold_development",
    ROOT / "scripts" / "audit_cgc_doppler_silent_hold_development.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(doppler: float, code_range_m: float) -> dict[str, float]:
    return {"carrier_doppler_hz": doppler, "code_range_m": code_range_m}


def test_truth_bin_separates_velocity_from_displacement() -> None:
    authentic = {(12.5, prn): row(100.0 + prn, 20_000_000.0) for prn in range(1, 9)}
    spoof = {(12.5, prn): row(100.1 + prn, 20_000_050.0) for prn in range(1, 9)}
    metrics = MODULE.truth_bin_metrics(authentic, spoof, 12)
    assert metrics["tu_oracle_prn_count"] == 0
    assert metrics["tu_oracle_available"] is False
    assert metrics["median_abs_code_offset_m"] == 50.0
    assert metrics["maximum_abs_code_offset_chips"] < 0.5


def test_truth_bin_marks_tu_input_support() -> None:
    authentic = {(7.5, prn): row(100.0, 10_000.0) for prn in range(1, 7)}
    spoof = {(7.5, prn): row(104.0 if prn <= 5 else 102.0, 10_020.0) for prn in range(1, 7)}
    metrics = MODULE.truth_bin_metrics(authentic, spoof, 7)
    assert metrics["tu_oracle_prn_count"] == 5
    assert metrics["tu_oracle_available"] is True


def test_longest_consecutive_bins() -> None:
    assert MODULE.longest_consecutive_bins([]) == 0
    assert MODULE.longest_consecutive_bins([12, 13, 15, 16, 17, 17]) == 3
    assert MODULE.longest_consecutive_bins([14, 13, 12]) == 3


def test_phase_boundaries_and_boolean_parser() -> None:
    assert [MODULE.phase_for_bin(value) for value in (4, 5, 9, 10, 11, 12)] == [
        "baseline", "pull-off", "pull-off", "guard", "guard", "hold"
    ]
    assert MODULE.parse_bool("True") is True
    assert MODULE.parse_bool("false") is False
