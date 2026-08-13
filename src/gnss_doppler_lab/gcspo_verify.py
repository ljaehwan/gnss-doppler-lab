"""Fail-closed clean and final artifact verification for GCSPO Stage-0."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .gcspo_verify_artifacts import reconstruct_final_evidence
from .gcspo_verify_reconstruct import validate_access_ledger, verify_evidence_document
from .gcspo_artifacts import FROZEN_HASHES, VALID_SCIENCE_REQUIRED, sha256_file, verify_artifact_manifest
from .gcspo_freeze import verify_review_candidate_record

METHODS = {"A0", "A1", "A2", "A3", "A4", "A5", "Full"}
FINAL_REQUIRED = set(VALID_SCIENCE_REQUIRED)
_HEX64 = re.compile(r"[0-9a-f]{64}")


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
    if doc_hash != "3ff447ee62e32c16249bd34a0b134d27d1040b0e78e01cb753b24de3b6db6fa0":
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
    recovery = preflight.get("synthetic_physical_recovery")
    if not isinstance(recovery, dict) or recovery.get("overall_status") != "PASS":
        raise ValueError("synthetic physical recovery proof is not PASS")
    if recovery.get("var_transfer_application_count") != 1:
        raise ValueError("synthetic recovery VAR transfer count mismatch")
    observed = recovery.get("maximum_scaled_state_error")
    allowed = recovery.get("tolerance", {}).get("maximum_scaled_state_error")
    if not isinstance(observed, (int, float)) or not isinstance(allowed, (int, float)) or observed > allowed:
        raise ValueError("synthetic recovery tolerance mismatch")
    verify_reproduction_manifests(artifact)
    candidate = artifact / "implementation_manifest.json"
    if candidate.is_file():
        record = json.loads(candidate.read_text())
        verify_review_candidate_record(record, target_commit=record.get("target_commit"))
    return {"schema": "gnss-doppler-lab.gcspo-stage0.verifier-report.v1", "phase": "clean-ready", "status": "PASS"}


def verify_reproduction_manifests(root: str | Path):
    """Authenticate two durable, content-addressed clean reproduction records."""
    artifact = Path(root); documents = []
    required = {"schema", "run_id", "source_freeze_sha", "commands", "config_identities",
                "dependencies", "runtime_canonicalization", "scientific_files",
                "boundary_and_reset_evidence", "comparison"}
    for index in (1, 2):
        path = artifact / f"reproduction_run_{index}.json"
        if not path.is_file(): raise ValueError("reproduction manifest is absent")
        doc = json.loads(path.read_text())
        if set(doc) != required or doc.get("schema") != "gnss-doppler-lab.gcspo-stage0.reproduction-run.v1":
            raise ValueError("reproduction manifest schema/keys mismatch")
        if doc.get("run_id") != f"a5-segment-repair-run-{index}" or len(str(doc.get("source_freeze_sha", ""))) != 40:
            raise ValueError("reproduction run/source identity mismatch")
        if not isinstance(doc.get("commands"), list) or not doc["commands"]:
            raise ValueError("reproduction commands are absent")
        if doc.get("runtime_canonicalization") != ["preflight_report.json:started_utc",
                                                     "preflight_report.json:finished_utc"]:
            raise ValueError("reproduction runtime canonicalization changed")
        files = doc.get("scientific_files")
        if not isinstance(files, dict) or not files:
            raise ValueError("reproduction scientific identities are absent")
        for name, identity in files.items():
            if not isinstance(name, str) or set(identity) != {"sha256", "size_bytes"}:
                raise ValueError("reproduction scientific identity is malformed")
            if _HEX64.fullmatch(str(identity["sha256"])) is None or isinstance(identity["size_bytes"], bool) or identity["size_bytes"] <= 0:
                raise ValueError("reproduction scientific identity is incomplete")
        evidence = doc.get("boundary_and_reset_evidence", {})
        if evidence.get("b0_half_open_test") != "test_b0_scheduled_windows_are_exactly_half_open_with_boundary_epsilon_and_reuse":
            raise ValueError("B0 boundary reproduction evidence is absent")
        if evidence.get("a5_gap_reset_test") != "test_a5_reacquisition_forms_independent_segments_and_preserves_actual_support":
            raise ValueError("A5 gap/reset reproduction evidence is absent")
        documents.append(doc)
    first, second = documents
    if first["scientific_files"] != second["scientific_files"]:
        raise ValueError("reproduction scientific identities differ")
    expected_comparison = "BYTE_IDENTICAL_AFTER_EXPLICIT_TIMESTAMP_CANONICALIZATION"
    if any(doc["comparison"] != {"peer_run_id": documents[1 - index]["run_id"], "result": expected_comparison}
           for index, doc in enumerate(documents)):
        raise ValueError("reproduction comparison result mismatch")
    for name, identity in first["scientific_files"].items():
        path = artifact / name
        if not path.is_file() or path.stat().st_size != identity["size_bytes"]:
            raise ValueError(f"reproduction canonical identity absent: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != identity["sha256"]:
            raise ValueError(f"reproduction canonical scientific identity mismatch: {name}")
    return {"status": "PASS", "run_count": 2, "comparison": expected_comparison,
            "scientific_files": first["scientific_files"]}


def verify_final(root: str | Path, *, strict: bool = False):
    artifact = Path(root)
    verdict = _json(artifact, "final_verdict.json")
    if verdict.get("verdict") not in {"GO_FOR_NEURAL_STAGE1", "NO_GO_PHYSICAL_HYPOTHESIS"} or verdict.get("protected_run_count") != 1:
        raise ValueError("scientific verdict/protected count mismatch")
    ledger = artifact / "access_ledger.jsonl"
    records = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.is_file() else []
    validate_access_ledger(records)
    verify_reproduction_manifests(artifact)
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
