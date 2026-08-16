import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

from gnss_doppler_lab.trace_native_1ms import ACTION_VALUE_FIELDS, read_records

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2b_stable_handoff_repair"


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_diagnosis_precedes_and_binds_fail_closed_repair():
    diagnosis = json.loads((ARTIFACT / "diagnosis.json").read_text())
    plan = json.loads((ARTIFACT / "repair_plan_preregistered.json").read_text())
    assert diagnosis["status"] == "COMPLETE_BEFORE_RECEIVER_REPAIR"
    assert diagnosis["reproduced_parent_failure"]["quality_filtered_common_causal_pairs"] == 79
    assert diagnosis["reproduced_parent_failure"]["stable_quality_common_at_least_4_prn_epochs"] == 0
    assert plan["trace_math_threshold_tolerance_or_window_changes_authorized"] is False
    assert plan["attack_performance_read_or_computed"] is False


def test_frozen_handoffs_restore_exact_target_aligned_action_state():
    manifest = json.loads((ARTIFACT / "handoffs/manifest.json").read_text())
    for scenario in manifest["scenarios"].values():
        with (ROOT / scenario["handoff_path"]).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == scenario["channel_count"] >= 4
        assert [int(row["channel"]) for row in rows] == list(range(len(rows)))
        selected = {int(item["selected_absolute_raw_sample"]): Path(item["path"]) for item in scenario["source_files"] if item["status"] == "SELECTED"}
        for row in rows:
            absolute = int(row["source_raw_interval_start_sample"])
            _, records = read_records(selected[absolute])
            source = records[np.flatnonzero(records["raw_interval_start_sample"] == absolute)[0]]
            assert int(row["source_channel"]) == int(source["channel"])
            assert int(row["prn"]) == int(source["prn"])
            for field in ACTION_VALUE_FIELDS:
                assert float(row[field]) == float(source[f"action_used_{field}"])
            assert int(row["interval_length_samples"]) == int(source["action_used_interval_length_samples"])


def test_r2b_config_uses_only_available_frozen_channels():
    source = (ROOT / "scripts/run_trace_stage0_r2a.py").read_text()
    assert '"Channels_1C.count": str(len(rows))' in source
    assert '"Channels.in_acquisition": str(len(rows))' in source
