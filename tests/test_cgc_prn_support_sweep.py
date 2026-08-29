from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_cgc_prn_support_sweep.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "audit_cgc_prn_support_sweep", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SWEEP = _load()


def _entries(count: int) -> list[dict[str, int]]:
    return [{"prn": value} for value in range(1, count + 1)]


def test_stable_subsets_are_deterministic_and_nested() -> None:
    entries = _entries(12)
    seven = SWEEP.stable_subset(
        entries, scenario="ds7", trial=4, support=7
    )
    ten = SWEEP.stable_subset(
        entries, scenario="ds7", trial=4, support=10
    )

    assert seven == SWEEP.stable_subset(
        entries, scenario="ds7", trial=4, support=7
    )
    assert seven == ten[:7]


def test_stable_subset_rejects_unavailable_support() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        SWEEP.stable_subset(
            _entries(7), scenario="ds7", trial=0, support=8
        )


def test_delay_is_causal_bin_endpoint_relative_to_attack_onset() -> None:
    rows = [
        {"bin_index": 110, "alarm": False},
        {"bin_index": 111, "alarm": True},
    ]

    assert SWEEP._delay(rows, "alarm", 100.0) == 12.0
    assert SWEEP._delay(rows[:1], "alarm", 100.0) is None


def test_range_reports_minimum_median_and_maximum() -> None:
    assert SWEEP._range([0.3, 0.1, 0.2]) == {
        "minimum": 0.1,
        "median": 0.2,
        "maximum": 0.3,
    }
