from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_qset", ROOT / "scripts/verify_qset_gnss_stage0a.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ARTIFACT = ROOT / MODULE.ARTIFACT_REL


def test_blocked_artifact_validates() -> None:
    result = MODULE.validate_artifact(ARTIFACT)
    assert result["verdict"] == MODULE.VERDICT
    assert result["stage0b_authorized"] is False


def test_access_tamper_fails_closed() -> None:
    audit = MODULE.load_json(ARTIFACT / "access_audit.json")
    audit["attack_payload"]["bytes_read"] = 1
    with pytest.raises(MODULE.VerificationError):
        MODULE.validate_access(audit)


def test_download_total_tamper_fails_closed() -> None:
    plan = MODULE.load_json(ARTIFACT / "dataset_download_plan.json")
    plan["minimum_totals"]["raw_bytes"] += 1
    with pytest.raises(MODULE.VerificationError):
        MODULE.validate_download_plan(plan)


def test_automatic_download_flag_fails_closed() -> None:
    plan = MODULE.load_json(ARTIFACT / "dataset_download_plan.json")
    plan["automatic_download_performed"] = True
    with pytest.raises(MODULE.VerificationError):
        MODULE.validate_download_plan(plan)


def test_stage0b_authorization_tamper_fails_closed() -> None:
    final = MODULE.load_json(ARTIFACT / "final_verdict.json")
    final["stage0b_authorized"] = True
    with pytest.raises(MODULE.VerificationError):
        MODULE.validate_final(final)


def test_manifest_detects_byte_tamper(tmp_path: Path) -> None:
    target = tmp_path / "x"
    target.write_bytes(b"before")
    before = MODULE.compact_manifest(tmp_path)
    target.write_bytes(b"after")
    assert before != MODULE.compact_manifest(tmp_path)


def test_official_scenario_incompatibility_is_explicit() -> None:
    preflight = MODULE.load_json(ARTIFACT / "dataset_preflight.json")
    assert preflight["selection_conclusion"]["gps_l1_priority_feasible_for_exact_1_3_5_contract"] is False
    assert preflight["scenario_role_inventory"][6]["independence"] == "FAIL_BYTE_IDENTICAL_OFFICIAL_MD5"
