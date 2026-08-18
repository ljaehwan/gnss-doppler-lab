from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "closure_verifier", ROOT / "scripts/verify_crisp_provenance_closure.py"
)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


def closure():
    return json.loads(VERIFY.CLOSURE_PATH.read_text())


def test_scientific_result_sha_binding():
    doc = closure()
    VERIFY.validate_document(doc)
    assert doc["scientific_result_sha"] == VERIFY.SCIENCE_SHA
    assert subprocess.run(
        ["git", "cat-file", "-e", f"{VERIFY.SCIENCE_SHA}^{{commit}}"], cwd=ROOT
    ).returncode == 0


def test_git_ancestry():
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", VERIFY.PREREG_SHA, VERIFY.SCIENCE_SHA],
        cwd=ROOT,
    ).returncode == 0


def test_pending_flag_removed():
    verdict = json.loads((VERIFY.ARTIFACT / "final_verdict.json").read_text())
    assert verdict["result_commit_pending"] is False


def test_original_and_final_science_hash_equality():
    doc = closure()
    VERIFY.verify_hash_bindings(doc)
    assert doc["original_scientific_file_hashes"] == doc["final_scientific_file_hashes"]


def test_forbidden_metric_mutation_detection():
    with pytest.raises(VERIFY.ClosureError, match="forbidden"):
        VERIFY.ensure_allowed_changes({"artifacts/crisp_stage0_static/scenario_metrics.csv"})


def test_forbidden_detector_code_mutation_detection():
    with pytest.raises(VERIFY.ClosureError, match="forbidden"):
        VERIFY.ensure_allowed_changes({"src/gnss_doppler_lab/crisp.py"})


def test_manifest_tamper_detection(tmp_path: Path):
    target = tmp_path / "value.txt"
    target.write_text("changed")
    manifest = {
        "files": [{"path": "value.txt", "size_bytes": 8, "sha256": "0" * 64}]
    }
    with pytest.raises(VERIFY.ClosureError, match="manifest tamper"):
        VERIFY.verify_manifest(tmp_path, manifest)


def test_missing_scientific_result_sha_detection():
    doc = closure()
    del doc["scientific_result_sha"]
    with pytest.raises(VERIFY.ClosureError, match="missing closure fields"):
        VERIFY.validate_document(doc)


def test_fake_self_referential_closure_sha_rejected():
    doc = closure()
    doc["provenance_closure_commit_sha"] = "PENDING_COMMIT"
    with pytest.raises(VERIFY.ClosureError, match="self-referential"):
        VERIFY.validate_document(doc)


def test_preserved_no_go_verdict_and_recommendation():
    VERIFY.validate_verdict()
    verdict = json.loads((VERIFY.ARTIFACT / "final_verdict.json").read_text())
    assert verdict["verdict"] == VERIFY.NO_GO
    assert verdict["go_checks"]["core_detection"] is False
    assert verdict["recommended_next_action"] == VERIFY.RECOMMENDATION


def test_execution_source_evidence():
    VERIFY.verify_evidence()
    assert closure()["execution_source_sha"] == VERIFY.PREREG_SHA


def test_deterministic_verifier_message():
    assert VERIFY.pass_message() == VERIFY.pass_message() == "PROVENANCE_CLOSURE_VERIFIED"
