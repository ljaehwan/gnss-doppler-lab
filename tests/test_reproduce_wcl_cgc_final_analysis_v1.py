from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reproduce_wcl_cgc_final_analysis_v1",
    ROOT / "scripts/reproduce_wcl_cgc_final_analysis_v1.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_rebased_path_keeps_relative_location_under_moved_root() -> None:
    old = Path("/old/frozen/pairs/p1/pair_complete.json")
    actual = MODULE.rebased_path(old, Path("/old/frozen"), Path("/new/data"))
    assert actual == Path("/new/data/pairs/p1/pair_complete.json")


def test_rebased_path_rejects_record_outside_frozen_root() -> None:
    with pytest.raises(ValueError, match="escapes"):
        MODULE.rebased_path(
            Path("/unrelated/manifest.json"),
            Path("/old/frozen"),
            Path("/new/data"),
        )


def test_logical_comparison_reports_each_contract_independently() -> None:
    reference = {
        "decision": "SUPPORTED",
        "aggregates": {"auc": 0.98},
        "gates": {"auc": True},
        "pairs": [{"pair_id": "p1"}],
    }
    assert all(MODULE.logical_comparison(reference.copy(), reference).values())
    changed = {**reference, "aggregates": {"auc": 0.97}}
    result = MODULE.logical_comparison(changed, reference)
    assert result == {
        "decision": True,
        "aggregates": False,
        "gates": True,
        "pairs": True,
    }
