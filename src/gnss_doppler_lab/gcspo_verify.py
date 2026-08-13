"""Fail-closed clean and final artifact verification for GCSPO Stage-0."""
from __future__ import annotations

from datetime import datetime, timezone
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


def _timestamp(value, label):
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"reproduction {label} timestamp is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"reproduction {label} timestamp is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"reproduction {label} timestamp is not UTC")
    return parsed


def _identity(path, identity, label):
    path = Path(path)
    if (not path.is_file() or not isinstance(identity, dict) or
            set(identity) != {"sha256", "size_bytes"} or
            _HEX64.fullmatch(str(identity.get("sha256", ""))) is None or
            isinstance(identity.get("size_bytes"), bool) or
            not isinstance(identity.get("size_bytes"), int) or identity["size_bytes"] <= 0):
        raise ValueError(f"reproduction {label} identity is incomplete")
    payload = path.read_bytes()
    if len(payload) != identity["size_bytes"] or hashlib.sha256(payload).hexdigest() != identity["sha256"]:
        raise ValueError(f"reproduction {label} identity mismatch")
    return payload


def _identity_map(rows, *, root, label):
    if not isinstance(rows, dict) or not rows:
        raise ValueError(f"reproduction {label} identities are absent")
    observed = {}
    for name, row in rows.items():
        if not isinstance(name, str) or not isinstance(row, dict) or set(row) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"reproduction {label} identity is malformed")
        path = Path(row["path"])
        if not path.is_absolute():
            path = root / path
        identity = {"sha256": row["sha256"], "size_bytes": row["size_bytes"]}
        _identity(path, identity, f"{label}:{name}")
        observed[name] = identity
    return observed


def verify_reproduction_manifests(root: str | Path):
    """Prove two separate successful A5 executions and their durable outputs."""
    artifact = Path(root).resolve(); repo = artifact.parents[1]
    documents = []
    required = {"schema", "run_id", "started_utc", "finished_utc", "source_freeze_sha",
                "starting_git", "command", "scratch_root", "environment",
                "source_identities", "input_identities", "output_bundle", "execution",
                "runtime_canonicalization", "canonical_promotion", "peer_comparison",
                "failed_attempts", "unchanged_dependency_identities"}
    for index in (1, 2):
        path = artifact / f"reproduction_run_{index}.json"
        if not path.is_file():
            raise ValueError("reproduction manifest is absent")
        doc = json.loads(path.read_text())
        if set(doc) != required or doc.get("schema") != "gnss-doppler-lab.gcspo-stage0.reproduction-run.v2":
            raise ValueError("reproduction manifest schema/keys mismatch")
        if not isinstance(doc.get("run_id"), str) or not doc["run_id"]:
            raise ValueError("reproduction run identity is absent")
        if _timestamp(doc["started_utc"], "start") >= _timestamp(doc["finished_utc"], "finish"):
            raise ValueError("reproduction execution did not finish after start")
        source_sha = str(doc.get("source_freeze_sha", ""))
        git = doc.get("starting_git")
        if (len(source_sha) != 40 or not isinstance(git, dict) or
                git != {"sha": source_sha, "branch": "research/gcspo-stage0-static-rerun",
                        "clean": True, "status_porcelain": []}):
            raise ValueError("reproduction starting Git identity/clean status mismatch")
        command = doc.get("command")
        scratch = doc.get("scratch_root")
        if (not isinstance(command, str) or not command or not isinstance(scratch, str) or
                scratch not in command or "<" in command or ">" in command or
                "PLACEHOLDER" in command.upper()):
            raise ValueError("reproduction exact command/scratch root is invalid")
        environment = doc.get("environment")
        if not isinstance(environment, dict) or not environment.get("python_executable") or not environment.get("packages"):
            raise ValueError("reproduction environment identity is incomplete")
        _identity_map(doc["source_identities"], root=repo, label="source")
        _identity_map(doc["input_identities"], root=artifact, label="input")
        unchanged = _identity_map(doc["unchanged_dependency_identities"], root=artifact,
                                  label="unchanged dependency")
        needed = {"Full_A1", "A2_A3_A4", "B0", "controls"}
        if not needed <= set(unchanged):
            raise ValueError("Full/A1-A4/B0/control dependency authentication is incomplete")
        execution = doc.get("execution")
        if execution != {"exit_status": 0, "status": "ACCEPTED", "completed": True}:
            raise ValueError("reproduction execution was not separately completed and accepted")
        failed = doc.get("failed_attempts")
        if (not isinstance(failed, dict) or set(failed) != {"count", "dispositions"} or
                failed["count"] != len(failed["dispositions"])):
            raise ValueError("reproduction failed-attempt disposition is incomplete")
        if doc.get("runtime_canonicalization") != []:
            raise ValueError("reproduction canonicalization includes unapproved runtime fields")
        bundle = doc.get("output_bundle")
        if not isinstance(bundle, dict) or set(bundle) != {"path", "files"}:
            raise ValueError("reproduction durable output bundle is malformed")
        bundle_path = (artifact / bundle["path"]).resolve()
        if artifact not in bundle_path.parents or not bundle_path.is_dir():
            raise ValueError("reproduction durable output bundle is absent")
        output_payloads = {}
        for name, identity in bundle["files"].items():
            output_payloads[name] = _identity(bundle_path / name, identity, f"bundle:{name}")
        if set(output_payloads) != {"clean_a5_report.json", "thresholds.json"}:
            raise ValueError("reproduction output bundle file set mismatch")
        promotion = doc.get("canonical_promotion")
        if (not isinstance(promotion, dict) or
                set(promotion) != {"status", "files"} or
                promotion["status"] != "BYTE_IDENTICAL_TO_BOTH_ACCEPTED_RUNS" or
                promotion["files"] != bundle["files"]):
            raise ValueError("reproduction canonical promotion binding mismatch")
        documents.append({"doc": doc, "bundle_path": bundle_path,
                          "payloads": output_payloads})
    first, second = documents
    run_ids = [row["doc"]["run_id"] for row in documents]
    scratch_roots = [row["doc"]["scratch_root"] for row in documents]
    bundle_paths = [str(row["bundle_path"]) for row in documents]
    if len(set(run_ids)) != 2 or len(set(scratch_roots)) != 2 or len(set(bundle_paths)) != 2:
        raise ValueError("reproduction runs lack separate IDs/scratch roots/output locations")
    expected = "BYTE_IDENTICAL_OUTPUT_SNAPSHOTS"
    for index, row in enumerate(documents):
        peer = row["doc"]["peer_comparison"]
        if peer != {"peer_run_id": documents[1 - index]["doc"]["run_id"],
                    "result": expected,
                    "files": ["clean_a5_report.json", "thresholds.json"]}:
            raise ValueError("reproduction peer comparison binding mismatch")
    if first["payloads"] != second["payloads"]:
        raise ValueError("reproduction output snapshots are not byte-identical")
    for name, payload in first["payloads"].items():
        canonical = artifact / name
        if canonical.read_bytes() != payload:
            raise ValueError(f"canonical promoted output differs from accepted runs: {name}")
    return {"status": "PASS", "run_count": 2, "comparison": expected,
            "run_ids": run_ids, "scratch_roots": scratch_roots,
            "bundle_paths": bundle_paths, "canonical_bound_to_both_runs": True,
            "scientific_files": first["doc"]["output_bundle"]["files"]}


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
