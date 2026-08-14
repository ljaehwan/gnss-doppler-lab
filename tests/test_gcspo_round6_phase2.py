"""Round-6 packaging/freeze tests; no protected path is inspected."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnss_doppler_lab.gcspo_round6_verify import (
    SOURCE_COMMIT, compare_round6_a5_runs, verify_round6_a5,
)


ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "artifacts/gcspo_stage0_static_rerun"


def test_packaged_round6_signed_chains_and_parity_reconstruct():
    result = verify_round6_a5(ARTIFACT)
    assert result["status"] == "PASS"
    assert result["witnessed"]["source_commit"] == SOURCE_COMMIT
    assert result["witnessed"]["independence"] == {
        "status": "EXTERNALLY_WITNESSED",
        "distinct": [
            "nonce", "PREPARED envelope", "PREPARED signature",
            "COMPLETED signature", "process identity", "process interval",
            "execution receipt",
        ],
    }
    assert [row["prepared"]["backend"] for row in result["witnessed"]["runs"]] == [
        "cuda", "cuda", "cpu",
    ]
    assert result["parity"]["same_backend"] == "BYTE_IDENTICAL"
    assert result["parity"]["cpu_cuda"]["status"] == "WITHIN_PREREGISTERED_TOLERANCE"


def test_round6_index_is_repository_relative_and_private_key_free():
    index = json.loads((ARTIFACT / "round6_a5_provenance.json").read_text())
    assert index["source_commit"] == SOURCE_COMMIT
    assert all(not Path(path).is_absolute() for path in index["evidence_roots"])
    packaged_names = {
        path.relative_to(ARTIFACT).as_posix()
        for path in (ARTIFACT / "round6_provenance").glob("**/*") if path.is_file()
    }
    assert not any("private" in name.lower() or name.endswith((".key", ".pem")) for name in packaged_names)
    assert not any("completion_observation.json" in name for name in packaged_names)


def test_round6_parity_fails_closed_on_tolerance_excess(monkeypatch):
    result = verify_round6_a5(ARTIFACT)
    verified = result["witnessed"]
    cpu = next(row for row in verified["runs"] if row["prepared"]["backend"] == "cpu")
    original = Path.read_text

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path == cpu["output_dir"] / "a5_numeric_trace.json":
            document = json.loads(text)
            document["thresholds"][next(iter(document["thresholds"]))] += 1.0
            return json.dumps(document)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="tolerance|mismatch"):
        compare_round6_a5_runs(verified)


def test_round5_unsigned_failure_remains_excluded_from_round6_freeze():
    audit = json.loads((ARTIFACT / "round6_audit_report.json").read_text())
    rejection = audit["round5_rejected_evidence"]
    assert rejection["status"] == "REJECTED_UNSIGNED_FAIL_CLOSED"
    assert rejection["retrospective_acceptance"] is False
    assert rejection["excluded_from_round6"] is True
    assert audit["protected"]["rows_opened"] == 0
    assert audit["protected"]["bytes_opened"] == 0
    assert audit["attack_run_count"] == 0
    assert audit["push_performed"] is False
