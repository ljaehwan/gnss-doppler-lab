import json
from pathlib import Path

from scripts.verify_crid_r1_terminal_drain import verify

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts/crid_stage0_r1_terminal_drain_repair"

def test_r1_compact_artifact_verifies():
    assert verify(ART)["status"]=="PASS"

def test_r1_does_not_change_crid_science_or_authorize_attack():
    contract=json.loads((ART/"termination_contract.json").read_text())
    final=json.loads((ART/"final_verdict.json").read_text())
    assert contract["scientific_configuration_changed"] is False
    assert contract["attack_data_accessed"] is False
    assert final["attack_evaluation_started"] is False
    assert final["status"]=="READY_FOR_FROZEN_CRID_RESUME"

def test_every_configuration_used_only_graceful_exit():
    four=json.loads((ART/"four_configuration_completion.json").read_text())
    for row in four["configurations"].values():
        assert row["exit_code"]==0
        assert row["exact_input_bytes"]==4_500_000_000
        assert row["dump_count"]==10
        assert row["sigterm_sent"] is False
        assert row["sigkill_sent"] is False
    assert four["absolute_raw_endpoint_contract"]["maximum_absolute_endpoint_delta_samples"]<=1
