from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cgc_three_regime_complementarity",
    ROOT / "scripts/audit_cgc_three_regime_complementarity.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def truth_rows(code_offset_m: float, code_rate_difference: float, carrier_difference: float):
    authentic = {}
    spoof = {}
    for prn in range(1, 7):
        key = (7.5, prn)
        authentic[key] = {
            "code_range_m": 20_000_000.0,
            "code_frequency_hz": 1_023_000.0,
            "carrier_doppler_hz": -2000.0,
        }
        spoof[key] = {
            "code_range_m": 20_000_000.0 + code_offset_m,
            "code_frequency_hz": 1_023_000.0 + code_rate_difference,
            "carrier_doppler_hz": -2000.0 + carrier_difference,
        }
    return authentic, spoof


def test_consistent_doppler_has_tu_input_without_code_carrier_mismatch() -> None:
    authentic, spoof = truth_rows(50.0, 0.02, 30.8)
    result = MODULE.truth_observability_metrics(authentic, spoof, 7)
    assert result["tu_oracle_input_available"] is True
    assert result["median_abs_code_offset_m"] == 50.0
    assert result["median_abs_code_carrier_mismatch_equivalent_hz"] == pytest.approx(0.0, abs=1e-6)


def test_locked_doppler_removes_tu_input_but_exposes_consistency_mismatch() -> None:
    authentic, spoof = truth_rows(50.0, 0.02, 0.0)
    result = MODULE.truth_observability_metrics(authentic, spoof, 7)
    assert result["tu_oracle_input_available"] is False
    assert result["tu_oracle_prn_count"] == 0
    assert result["median_abs_code_carrier_mismatch_equivalent_hz"] == pytest.approx(30.8, abs=1e-6)


def test_position_hold_keeps_displacement_after_rate_observables_collapse() -> None:
    authentic, spoof = truth_rows(50.0, 0.0, 0.0)
    result = MODULE.truth_observability_metrics(authentic, spoof, 7)
    assert result["median_abs_code_offset_m"] == 50.0
    assert result["tu_oracle_input_available"] is False
    assert result["median_abs_code_carrier_mismatch_equivalent_hz"] == 0.0


def test_phase_and_regime_labels() -> None:
    assert [MODULE.phase_for_bin(index) for index in (4, 5, 9, 10, 11, 12)] == [
        "baseline",
        "pull-off",
        "pull-off",
        "guard",
        "guard",
        "hold",
    ]
    assert MODULE.regime_name("carrier-coupled", "pull-off") == "consistent_pull_off"
    assert MODULE.regime_name("doppler-locked", "pull-off") == "locked_pull_off"
    assert MODULE.regime_name("doppler-locked", "hold") == "locked_position_hold"
