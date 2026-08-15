from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.verify_gcspo_r2_evidence import DEFAULT_EVIDENCE_DIR, verify


def test_gcspo_r2_evidence_manifest_and_source_integrity():
    base = DEFAULT_EVIDENCE_DIR
    source = json.loads((base / "source_artifact_integrity.json").read_text())
    assert source["status"] == "PASS"
    assert source["actual_manifest_sha256"] == "ad6bbcd34c3889aa393d8699eec4e48c2dcc59095a5a0e3e632442b0bc7205cd"
    assert source["actual_file_count"] == 68
    manifest = json.loads((base / "evidence_bundle_manifest.json").read_text())
    assert manifest["bundle_size_bytes"] < 50 * 1024 * 1024
    assert manifest["final_evidence_judgement"] == "EVIDENCE_VERIFIED"


def test_gcspo_r2_independent_metric_recalculation():
    result = verify(DEFAULT_EVIDENCE_DIR)
    assert result["final_evidence_judgement"] == "EVIDENCE_VERIFIED"
    assert result["checks_failed"] == 0
    ds3 = result["recomputed_metrics"]["DS3"]
    ds7 = result["recomputed_metrics"]["DS7"]
    assert ds3["full_events"] == 907
    assert abs(ds3["pre_onset_fpr_q99"] - 0.14893617021276595) < 1e-12
    assert abs(ds3["roc_auc"] - 0.8267983789260385) < 1e-12
    assert ds7["full_events"] == 533
    assert abs(ds7["roc_auc"] - 0.5991366738610512) < 1e-12


def test_gcspo_r2_exact_support_row_accounting():
    result = verify(DEFAULT_EVIDENCE_DIR)
    checks = {c["name"]: c for c in result["checks"]}
    assert checks["DS3 B0 exact-common"]["actual"] == 643
    assert checks["DS7 B0 exact-common"]["actual"] == 396
    assert checks["complete epoch+PRN+window identity malformed rows"]["actual"] == 0
    assert checks["DS3 A0 method_only"]["status"] == "PASS"
    assert checks["DS7 A0 full_only"]["status"] == "PASS"


def test_gcspo_r2_runner_seven_phase_terminal_state():
    runner = json.loads((DEFAULT_EVIDENCE_DIR / "runner_phase_evidence.json").read_text())
    assert runner["all_required_latest_successful"] is True
    assert len(runner["required_phases"]) == 7
    for phase, info in runner["required_phases"].items():
        assert info["status"] == "succeeded", phase
        assert info["exit_code"] == 0, phase
        assert info["final_heartbeat_exists"] is True, phase
        assert info["contract_command_exists"] is True, phase
        assert info["contract_repository_path"], phase
        assert info["phase_terminal_evidence_ok"] is True, phase
    # Failed retries must remain visible rather than being overwritten.
    failed = {(x["phase"], x["run_id"]) for x in runner["failed_attempts_preserved"]}
    assert ("gcspo-r2-cleanstatic-normal-model", "20260815T103959Z-gcspo-r2-cleanstatic-normal-model") in failed
    assert ("gcspo-r2-ds3-evaluation", "20260815T115002Z-gcspo-r2-ds3-evaluation") in failed


def test_gcspo_r2_scientific_source_code_not_changed_in_this_worktree():
    changed = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], text=True).splitlines()
    forbidden_prefixes = (
        "src/gnss_doppler_lab/gcspo_r2_runner.py",
        "src/gnss_doppler_lab/gcspo_core.py",
        "src/gnss_doppler_lab/gcspo_clean.py",
        "src/gnss_doppler_lab/gcspo_b0.py",
        "src/gnss_doppler_lab/gcspo_statistics.py",
    )
    assert not [p for p in changed if p.startswith(forbidden_prefixes)]
