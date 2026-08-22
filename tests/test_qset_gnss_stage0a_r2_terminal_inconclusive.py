from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from gnss_doppler_lab import qset_stage0a_r2 as Q

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_qset_r2_terminal",
    ROOT / "scripts/verify_qset_gnss_stage0a_r2_terminal_inconclusive.py",
)
assert SPEC and SPEC.loader
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def failure_fixture() -> dict:
    files: list[dict] = []
    return {
        "status": "FAIL_INSUFFICIENT_GALILEO_RECEIVER_SUPPORT",
        "trace_validation": {
            "status": "FAIL",
            "tracked_prn_count": 3,
            "tracked_prns": [9, 30, 36],
            "finite_failures": 0,
            "cadence_failures": 0,
            "causal_failures": 0,
        },
        "receiver": {"terminal_drain": True, "program_ended": True},
        "support_gate": {"pass": False},
        "score_computed": False,
        "downstream_attack_scenarios_opened": [],
        "output_set": {
            "files": files,
            "file_count": 0,
            "aggregate_sha256": Q.canonical_sha(files),
        },
    }


def test_terminal_failure_accepts_exact_three_prn_fail_closed_evidence() -> None:
    VERIFIER.validate_terminal_failure(failure_fixture())


def test_terminal_failure_rejects_support_or_score_tamper() -> None:
    changed = copy.deepcopy(failure_fixture())
    changed["trace_validation"]["tracked_prn_count"] = 5
    with pytest.raises(VERIFIER.VerificationError):
        VERIFIER.validate_terminal_failure(changed)
    changed = copy.deepcopy(failure_fixture())
    changed["score_computed"] = True
    with pytest.raises(VERIFIER.VerificationError):
        VERIFIER.validate_terminal_failure(changed)


def test_terminal_access_audit_rejects_downstream_access() -> None:
    audit = {
        "status": "PASS",
        "attack_access_after_freeze_only": True,
        "attack_payload": {"allowlisted_scenarios": ["SS-1"], "bytes_read": 59_999_664_000},
        "unopened_allowlisted_scenarios": ["SS-3", "SS-5", "SS-11"],
        "unallowlisted_tuni2025_raw": {key: 0 for key in ("stats", "hashes", "opens", "mmaps", "bytes_read")},
        "attack_scientific_operations": {"feature_windows": 0, "scores": 0, "attack_evaluations": 0},
    }
    VERIFIER.validate_access(audit)
    audit["unopened_allowlisted_scenarios"] = ["SS-5", "SS-11"]
    with pytest.raises(VERIFIER.VerificationError):
        VERIFIER.validate_access(audit)


def test_terminal_manifest_detects_tamper(tmp_path: Path) -> None:
    target = tmp_path / "evidence.json"
    target.write_bytes(b"frozen")
    files = [{"path": target.name, "size_bytes": target.stat().st_size, "sha256": Q.sha256_file(target)}]
    manifest = {"status": "PASS", "files": files, "aggregate_sha256": Q.canonical_sha(files)}
    VERIFIER.validate_manifest(tmp_path, manifest)
    target.write_bytes(b"changed")
    with pytest.raises(VERIFIER.VerificationError):
        VERIFIER.validate_manifest(tmp_path, manifest)


def test_terminal_reporting_source_cannot_open_raw_or_score() -> None:
    source = (ROOT / "src/gnss_doppler_lab/qset_stage0a_r2_terminal_inconclusive.py").read_text(encoding="utf-8")
    for forbidden_call in ("evaluate_attack(", "extract_window_features(", "scenario_path(", "md5_file("):
        assert forbidden_call not in source
