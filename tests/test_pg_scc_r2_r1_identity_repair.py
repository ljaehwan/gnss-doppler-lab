from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1"
PREDECESSOR_SUPPORT = ROOT / "artifacts/pg_scc_stage0_r2_repair_followup/support_preflight.json"
CYCLE_PREREG = "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/preregistration.json"


def load_runner():
    path = ROOT / "scripts/run_pg_scc_root_cause_audit.py"
    spec = importlib.util.spec_from_file_location("pg_scc_r1_identity_cycle_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_identity_serialization_drift_reproduces_r1_mismatch(monkeypatch):
    runner = load_runner()
    full_paths = runner._git_changed_paths_since_r1()
    assert CYCLE_PREREG in full_paths

    monkeypatch.setattr(
        runner,
        "_git_changed_paths_since_r1",
        lambda: set(full_paths) - {CYCLE_PREREG},
    )
    before = runner.verify_r1_identity()

    monkeypatch.setattr(runner, "_git_changed_paths_since_r1", lambda: set(full_paths))
    after = runner.verify_r1_identity()

    assert before["status"] == "PASS"
    assert after["status"] == "PASS"
    assert before["phase2_changed_paths"] != after["phase2_changed_paths"]
    assert CYCLE_PREREG not in before["phase2_changed_paths"]
    assert CYCLE_PREREG in after["phase2_changed_paths"]
    stable_fields = set(before) - {"phase2_changed_paths"}
    assert {key: before[key] for key in stable_fields} == {
        key: after[key] for key in stable_fields
    }


def test_predecessor_committed_report_mismatch_is_exactly_r1_identity():
    runner = load_runner()
    predecessor = json.loads(PREDECESSOR_SUPPORT.read_text(encoding="utf-8"))
    repaired = runner.build_metadata_support_preflight_report()
    differing = sorted(
        key for key in set(predecessor) | set(repaired)
        if predecessor.get(key) != repaired.get(key)
    )
    assert differing == ["r1_identity"]
    assert repaired["protected_score_fields_read"] == 0
    assert repaired["protected_score_fields_projected_or_read"] == 0


def test_cycle1_committed_preflight_bytes_remain_frozen():
    manifest = json.loads((CYCLE / "implementation_manifest_sha256.json").read_text())
    relative = "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/support_preflight.json"
    actual = hashlib.sha256((CYCLE / "support_preflight.json").read_bytes()).hexdigest()
    assert actual == manifest["files"][relative]


def test_cycle1_terminal_record_forbids_retry_and_reuse():
    attempt = json.loads((CYCLE / "attempt_state.json").read_text(encoding="utf-8"))
    assert attempt["state"] == "TERMINAL_FAIL_CLOSED"
    assert attempt["retry_forbidden"] is True
    assert attempt["protected_loader_invoked"] is True
    assert attempt["partial_scientific_outputs_inspected"] is False
    assert attempt["partial_scientific_outputs_reused"] is False


def test_cycle_identity_paths_are_explicitly_authorized():
    runner = load_runner()
    required = {
        "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/config.json",
        "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/implementation_manifest_sha256.json",
        "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/pre_run_state.json",
        "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/predecessor_failure_binding.json",
        "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/preregistration.json",
        "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/r1_failure_binding.json",
        "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/semantic_diff_audit.json",
        "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/source_commit.json",
        "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/support_preflight.json",
        "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/zero_protected_access.json",
        "scripts/verify_pg_scc_r2_r1_identity_repair.py",
        "scripts/verify_pg_scc_r2_r1_identity_semantic_diff.py",
        "tests/test_pg_scc_r2_r1_identity_repair.py",
    }
    assert required <= runner.PHASE2_ALLOWED_CHANGED_PATHS
