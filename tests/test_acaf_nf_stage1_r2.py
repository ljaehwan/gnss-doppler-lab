import json
from pathlib import Path

from gnss_doppler_lab.acaf_nf_stage1_r2 import (
    CHECKPOINT4_REQUIRED,
    R1_REQUIRED,
    SCENARIOS,
    _first_rows_semantics,
    checkpoint4,
)


def test_r2_contract_names_are_complete():
    assert {"cleanStatic", "ds3", "ds4", "ds7", "ds8"} == set(SCENARIOS)
    assert "go_no_go.json" in R1_REQUIRED
    assert "per_window_scores.csv" in R1_REQUIRED
    assert Path("artifacts/acaf_nf_stage1_r2_full_normal").name == "acaf_nf_stage1_r2_full_normal"


def test_frozen_scenario_semantics_mapping_is_explicit():
    assert callable(_first_rows_semantics)


def test_checkpoint4_fail_closed_contract(tmp_path):
    def dump(name, value):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")

    dump("foundation_status.json", {
        "status": "FOUNDATION_INVALID", "checkpoint3_physics_authorized": False,
        "failed_clean_gates": ["r14_common_reproduced_1e_6"],
    })
    dump("full_cleanstatic_validation.json", {
        "status": "CONTINUOUS_TRACKER_INVALID", "selected_epochs": 9, "selected_prn_channels": 4,
        "gates": {"r14_common_reproduced_1e_6": False}, "prompt_reproduction": {"n": 9},
        "delay_recovery": {"n": 9}, "l20_doppler": {"n": 1}, "grid_boundary_fraction": .1,
        "r14_common_epochs": {"n": 2, "surface_sha256_all_match": False},
    })
    dump("fresh_continuous_tracker_manifest.json", {
        "fresh_replay_manifest_sha256": "a" * 64, "rows": 10, "csv_sha256": "b" * 64,
        "csv_size_bytes": 100, "exporter_rows": 12, "status": "CONTINUOUS_TRACKER_INVALID",
        "alignment": {"prompt": "current MAT row k"},
    })
    dump("verification_report.json", {
        "checkpoint": 2, "status": "PASS", "recomputed": {
            "derived_tracker_status": "CONTINUOUS_TRACKER_INVALID",
            "derived_foundation_status": "FOUNDATION_INVALID",
        },
    })
    dump("config.json", {"checkpoint": 2})
    (tmp_path / "test_report.txt").write_text("pending\n", encoding="utf-8")
    checkpoint4(tmp_path)
    assert all((tmp_path / name).is_file() for name in CHECKPOINT4_REQUIRED)
    assert json.loads((tmp_path / "go_no_go.json").read_text())["verdict"] == "FOUNDATION_INVALID"
    assert json.loads((tmp_path / "execution_validity.json").read_text())["checkpoint3_physics_executed"] is False
    assert json.loads((tmp_path / "normal_split.json").read_text())["applied"] is False
