import csv
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_cgc_normal_detector_freeze_audit.py"
CONFIG = (
    ROOT
    / "configs/experiments/cgc_normal_detector_freeze_audit_v1.json"
)


def load_module():
    name = "cgc_normal_detector_freeze_audit_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_config_pins_only_five_train_normal_sources():
    module = load_module()
    context = module.validate_config(frozen_config(), verify_iq=False)

    assert [pair["paired_group_id"] for pair in context["pairs"]] == [
        f"pv1-pair-{index:03d}" for index in range(2, 7)
    ]
    assert set(context["sources"]) == set(module.EXPECTED_PAIR_IDS)


def test_lopo_threshold_separates_synthetic_complete_streams():
    module = load_module()
    rows = []
    bases = {
        "normal": -0.90,
        "independent_multipath": -0.75,
        "carryoff_spoof": -0.15,
    }
    for pair_id in module.EXPECTED_PAIR_IDS:
        for scenario, base in bases.items():
            for bin_index in range(3):
                rows.append({
                    "pair_id": pair_id,
                    "scenario": scenario,
                    "bin_index": bin_index,
                    "score": base + 0.01 * bin_index,
                    "fit_rank": 4,
                })

    result = module.lopo_evaluate(
        rows,
        pair_ids=module.EXPECTED_PAIR_IDS,
        consecutive=2,
        gates=module.EXPECTED_GATES,
        finite_fraction=1.0,
        full_rank_fraction=1.0,
    )

    assert result["cross_validated_macro_balanced_accuracy"] == 1.0
    assert result["normal_persistent_alarm_pair_count"] == 0
    assert result["multipath_persistent_alarm_pair_count"] == 0
    assert result["spoof_persistent_detection_pair_count"] == 5
    assert result["all_support_gates_passed"] is True
    assert -0.75 < result["final_threshold"] < -0.13


def test_persistence_resets_across_missing_or_benign_bins():
    module = load_module()
    threshold = -0.5
    separated = [
        {"bin_index": 1, "score": -0.1},
        {"bin_index": 3, "score": -0.1},
    ]
    interrupted = [
        {"bin_index": 1, "score": -0.1},
        {"bin_index": 2, "score": -0.8},
        {"bin_index": 3, "score": -0.1},
    ]
    consecutive = [
        {"bin_index": 1, "score": -0.1},
        {"bin_index": 2, "score": -0.1},
    ]

    assert module.persistent_alarm(separated, threshold, 2) is False
    assert module.persistent_alarm(interrupted, threshold, 2) is False
    assert module.persistent_alarm(consecutive, threshold, 2) is True


def test_forbidden_access_and_threshold_drift_are_rejected():
    module = load_module()
    leaked = copy.deepcopy(frozen_config())
    leaked["data_boundary"]["test_pairs_accessed"] = True
    with pytest.raises(ValueError, match="forbidden data-access"):
        module.validate_config(leaked)

    drifted = copy.deepcopy(frozen_config())
    drifted["threshold_selection"]["persistence_consecutive_bins"] = 3
    with pytest.raises(ValueError, match="threshold-selection"):
        module.validate_config(drifted)


def test_csv_writer_preserves_union_of_mixed_row_fields(tmp_path):
    module = load_module()
    path = tmp_path / "mixed.csv"
    module._write_csv(path, [
        {"pair_id": "p1", "normal_only": 1},
        {"pair_id": "p2", "replication_only": 2},
    ])

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["pair_id", "normal_only", "replication_only"]
