import csv,json
from pathlib import Path

from scripts.verify_crid_r2_frozen_evaluation import verify

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts/crid_stage0_r2_frozen_evaluation"

def test_compact_r2_artifact_verifies():
    assert verify(ART)["status"]=="PASS"

def test_phase_a_blocks_attack_and_stage1():
    final=json.loads((ART/"final_verdict.json").read_text())
    assert final["verdict"]=="INCONCLUSIVE_RECEIVER_REPLAY_OR_ALIGNMENT"
    assert final["phase_b"]=="NOT_AUTHORIZED"
    assert final["attack_payload_opened"] is False
    assert final["attack_payload_bytes_read"]==0
    assert final["neural_stage1_implemented"] is False

def test_frozen_control_grid_is_fully_inventoried_not_falsely_completed():
    with (ART/"physical_control_metrics.csv").open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    required=[row for row in rows if row["frozen_required"]=="True"]
    assert len(required)==66
    for domain in ("TEX","OAK"):
        domain_rows=[row for row in required if row["domain"]==domain]
        assert sum(row["kind"]=="negative" for row in domain_rows)==15
        assert sum(row["kind"]=="positive" for row in domain_rows)==18
        assert sum(row["conforming"]=="True" for row in domain_rows if row["kind"]=="negative")==9
        assert not any(row["conforming"]=="True" for row in domain_rows if row["kind"]=="positive")

def test_r1_reuse_is_diagnostic_only_and_termination_valid():
    replay=json.loads((ART/"replay_completion.json").read_text())
    assert replay["r1_reuse"]["scientific_use"]=="DIAGNOSTIC_ONLY_NONCONFORMING_STIMULUS"
    for row in replay["r1_reuse"]["configurations"].values():
        assert row["status"]=="PASS" and row["exit_code"]==0
        assert row["sigterm_sent"] is False and row["sigkill_sent"] is False
