from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_cgc_rf_tokyo_replication.py"
SPEC = importlib.util.spec_from_file_location("run_cgc_rf_tokyo_replication", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def config():
    return json.loads((ROOT / "configs/experiments/cgc_rf_tokyo_replication_v1.json").read_text())


def row(replica_id, power, auc, direction=0.93):
    return {
        "replica_id": replica_id,
        "final_advantage_db": power,
        "serial_bin_auc": auc,
        "median_absolute_direction_cosine": direction,
        "spoof_bin_count": 10,
        "multipath_bin_count": 10,
        "minimum_spoof_prn_count": 9,
        "minimum_multipath_prn_count": 9,
    }


def grid(high_aucs, low_aucs=None):
    low_aucs = low_aucs or [0.95] * 5
    rows = []
    for replica_id, low, high in zip(MODULE.REPLICA_IDS, low_aucs, high_aucs):
        rows.extend((row(replica_id, -6.0, low), row(replica_id, 3.0, high)))
    return rows


def test_config_freezes_five_new_seed_pairs():
    doc = config()
    assert tuple(row["replica_id"] for row in doc["replicas"]) == MODULE.REPLICA_IDS
    assert tuple(row["receiver_seed"] for row in doc["replicas"]) == MODULE.RECEIVER_SEEDS
    assert tuple(row["multipath_seed"] for row in doc["replicas"]) == MODULE.MULTIPATH_SEEDS
    assert doc["analysis"]["primary_aperture_taps"] == 9


def test_systematic_high_power_blind_spot_decision():
    result = MODULE.evaluate_replication(grid([0.42, 0.55, 0.61, 0.70, 0.91]), config()["analysis"])
    assert result["decision"] == "SYSTEMATIC_HIGH_POWER_BLIND_SPOT"
    assert result["plus3_auc_fail_count"] == 4
    assert result["original_observed_tokyo_cell_in_primary_counts"] is False


def test_single_realization_exception_decision():
    result = MODULE.evaluate_replication(grid([0.91, 0.92, 0.93, 0.94, 0.50]), config()["analysis"])
    assert result["decision"] == "SINGLE_REALIZATION_EXCEPTION"
    assert result["plus3_auc_fail_count"] == 1


def test_mixed_pattern_remains_unresolved():
    result = MODULE.evaluate_replication(grid([0.45, 0.55, 0.90, 0.91, 0.92]), config()["analysis"])
    assert result["decision"] == "MIXED_OR_UNRESOLVED"


def test_incomplete_grid_is_rejected():
    rows = grid([0.4] * 5)
    try:
        MODULE.evaluate_replication(rows[:-1], config()["analysis"])
    except ValueError as exc:
        assert "complete five-by-two grid" in str(exc)
    else:
        raise AssertionError("incomplete grid should fail closed")
