from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/gcspo_stage0_r1_frozen_completion"


def test_r1_physical_control_audit_reconstructs_scientific_g5_rows():
    source_path = ARTIFACT / "physical_controls.json"
    source = json.loads(source_path.read_text())
    audit = json.loads((ARTIFACT / "physical_controls_audit.json").read_text())

    assert audit["input_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert source["overall_status"] == "PASS"
    assert audit["generator_status_meaning"].endswith("not scientific specificity")
    assert audit["scientific_overall_status"] == "FAIL"
    assert audit["threshold_or_score_changed"] is False
    assert audit["code_change_required"] is False

    failed = []
    for row in source["results"]:
        passed = (row["specificity_ratio"] <= 0.25 if row["id"] == "CLOCK_DRIFT"
                  else row["persistent_alarm_ratio"] <= 0.10
                  and row["max_consecutive_alarms"] < 10)
        if not passed:
            failed.append((row["id"], row["level"], row["block_id"],
                           row["persistent_alarm_ratio"], row["max_consecutive_alarms"]))

    recorded = [(row["id"], row["level"], row["block_id"],
                 row["persistent_alarm_ratio"], row["max_consecutive_alarms"])
                for row in audit["failed_rows"]]
    assert len(source["results"]) == audit["rows_total"] == 312
    assert len(failed) == audit["rows_fail"] == 33
    assert audit["rows_pass"] == 279
    assert recorded == failed
