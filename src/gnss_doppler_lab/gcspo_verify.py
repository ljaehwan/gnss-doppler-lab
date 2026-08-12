"""Fail-closed clean and final artifact verification for GCSPO Stage-0."""
from __future__ import annotations

import json
from pathlib import Path

from .gcspo_verify_artifacts import reconstruct_final_evidence
from .gcspo_verify_reconstruct import validate_access_ledger, verify_evidence_document
from .gcspo_artifacts import FROZEN_HASHES, VALID_SCIENCE_REQUIRED, sha256_file, verify_artifact_manifest

METHODS = {"A0", "A1", "A2", "A3", "A4", "A5", "Full"}
FINAL_REQUIRED = set(VALID_SCIENCE_REQUIRED)


def _json(root: Path, name: str):
    path = root / name
    if not path.is_file():
        raise ValueError(f"required artifact missing: {name}")
    return json.loads(path.read_text())


def verify_frozen(root: str | Path):
    artifact = Path(root)
    observed = {name: sha256_file(artifact / name) for name in FROZEN_HASHES}
    if observed != FROZEN_HASHES:
        raise ValueError("frozen artifact checksum mismatch")
    doc_hash = sha256_file(artifact.parents[1] / "docs/GCSPO_STAGE0.md")
    if doc_hash != "d5cb40d436b55cd58cff3063e018b19b4f8296f5af5e96331b77af663324314f":
        raise ValueError("frozen GCSPO document checksum mismatch")
    return observed


def verify_clean_ready(root: str | Path):
    artifact = Path(root)
    clean, preflight = _json(artifact, "clean_only_report.json"), _json(artifact, "preflight_report.json")
    if clean.get("run_status") != "CLEAN_ONLY_PASS" or preflight.get("overall_status") != "PASS":
        raise ValueError("clean/preflight status is not PASS")
    if clean.get("protected_attack_rows_read") is not False or clean.get("attack_access_count") != 0 or preflight.get("attack_access_count") != 0:
        raise ValueError("clean-only artifacts indicate protected access")
    if set(clean.get("all_methods", [])) != METHODS:
        raise ValueError("clean-only methods are incomplete")
    if clean.get("deterministic_rerun") != "PASS":
        raise ValueError("deterministic rerun is not PASS")
    return {"schema": "gnss-doppler-lab.gcspo-stage0.verifier-report.v1", "phase": "clean-ready", "status": "PASS"}


def verify_final(root: str | Path, *, strict: bool = False):
    artifact = Path(root)
    verdict = _json(artifact, "final_verdict.json")
    if verdict.get("verdict") not in {"GO_FOR_NEURAL_STAGE1", "NO_GO_PHYSICAL_HYPOTHESIS"} or verdict.get("protected_run_count") != 1:
        raise ValueError("scientific verdict/protected count mismatch")
    ledger = artifact / "access_ledger.jsonl"
    records = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.is_file() else []
    validate_access_ledger(records)
    reconstructed = reconstruct_final_evidence(artifact)
    if verdict.get("evidence") != reconstructed: raise ValueError("reported evidence differs from artifact reconstruction")
    verify_evidence_document(verdict)
    if strict:
        actual = {path.relative_to(artifact).as_posix() for path in artifact.rglob("*") if path.is_file()}
        missing = sorted((FINAL_REQUIRED - {"verifier_report.json", "fresh_clone_verifier_report.json"}) - actual)
        allowed = FINAL_REQUIRED | {"implementation_manifest.json", "data_manifest.json", "run_manifest.json", "file_access_trace.jsonl"}
        extras = sorted(path for path in actual if path not in allowed and not path.startswith("plots/"))
        if missing or extras:
            raise ValueError(f"final artifact exact-set mismatch: missing={missing} extras={extras}")
        if not any(path.startswith("plots/") for path in actual): raise ValueError("final artifact plots are absent")
        verify_frozen(artifact)
    verify_artifact_manifest(artifact)
    return {"schema": "gnss-doppler-lab.gcspo-stage0.verifier-report.v1", "phase": "final", "status": "PASS", "protected_run_count": 1}
