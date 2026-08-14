"""Read-only verification of the Round-8 successor freeze."""
from __future__ import annotations

from pathlib import Path
import subprocess

from .gcspo_round6_verify import (
    EXPECTED_NUMERIC_FIELD_COUNT,
    EXPECTED_NUMERIC_PATH_SHA256,
    ROUND7_REJECTED_FREEZE_COMMIT,
    SOURCE_COMMIT,
    _identity,
    strict_json_load,
    strict_json_loads,
    verify_round6_a5,
)


ROUND8_REPAIR_COMMIT = "40a4dc4cff94aa47875a83a9187e0959096e3f0f"
ROUND7_VERDICT_SHA256 = "a4ab0bde9430d0a61d0631e5c5e3faa5f99fa3dd8769ea657c4214d261c0e48d"
ROUND7_VERDICT_SIZE_BYTES = 5323
README_SHA256 = "eea2e10885d66bfc762f33b2e25147ab07b1bbceace505078e8770e4cdc18ac2"
FREEZE_RELATIVE = "artifacts/gcspo_stage0_static_rerun/round8_freeze_manifest.json"
ROUND8_JSON_FILES = (
    "round8_audit_report.json",
    "round8_evidence_manifest.json",
    "round8_freeze_manifest.json",
    "round8_green_report.json",
    "round8_red_report.json",
    "round8_repair_review_handoff.json",
)
ANCESTRY = (
    "a9a6f03a8fe984ee75c15fbcf81f7c04c5ab2e46",
    SOURCE_COMMIT,
    "532c4dd014432b787553b199a93f01ddaa294c01",
    "c525b8d1db7b1d7c5955af805900366a87a5a877",
    "093cb0d8456f0bdf795e0a9833693d58e6e54bbd",
    ROUND7_REJECTED_FREEZE_COMMIT,
    "70826965bd96e8c103704ca4d0916890e5dc69c4",
    ROUND8_REPAIR_COMMIT,
)


def _git(repo: Path, *arguments: str, text: bool = True):
    return subprocess.run(
        ["git", *arguments], cwd=repo, check=True, capture_output=True, text=text,
    ).stdout


def _require_contract(document: dict, label: str) -> None:
    contract = document.get("numeric_trace_contract")
    if contract != {
        "expected_numeric_field_count": EXPECTED_NUMERIC_FIELD_COUNT,
        "expected_numeric_path_sha256": EXPECTED_NUMERIC_PATH_SHA256,
        "path_digest_encoding": "SHA256_OF_SEQUENCE_OF_UINT64_BE_LENGTH_PLUS_COMPACT_UTF8_JSON_PATH_TOKENS",
        "coverage_requirement": "POSITIVE_AND_EXACT_COUNT_AND_EXACT_PATH_DIGEST",
    }:
        raise ValueError(f"Round-8 {label} numeric trace contract mismatch")


def _verify_evidence_manifest(artifact: Path) -> dict:
    repo = artifact.parents[1]
    manifest = strict_json_load(artifact / "round8_evidence_manifest.json")
    if (manifest.get("schema") !=
            "gnss-doppler-lab.gcspo-stage0.round8-successor-evidence-manifest.v1" or
            manifest.get("source_commit") != SOURCE_COMMIT or
            manifest.get("rejected_round7_freeze_commit") != ROUND7_REJECTED_FREEZE_COMMIT or
            manifest.get("repair_commit") != ROUND8_REPAIR_COMMIT or
            manifest.get("manifest_excludes_self") is not True or
            manifest.get("self_binding_exclusion") !=
            "artifacts/gcspo_stage0_static_rerun/round8_evidence_manifest.json"):
        raise ValueError("Round-8 evidence manifest contract mismatch")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Round-8 evidence manifest file set is empty")
    paths = [row.get("path") for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Round-8 evidence paths are not sorted and unique")
    for row in rows:
        if set(row) != {"path", "sha256", "size_bytes"}:
            raise ValueError("Round-8 evidence identity row is malformed")
        path = (repo / row["path"]).resolve(strict=True)
        if repo not in path.parents or _identity(path) != {
                "sha256": row["sha256"], "size_bytes": row["size_bytes"]}:
            raise ValueError(f"Round-8 evidence identity mismatch: {row['path']}")
    return manifest


def verify_round8_freeze(artifact_root: str | Path, expected_freeze_commit: str) -> dict:
    artifact = Path(artifact_root).resolve(strict=True)
    repo = artifact.parents[1]
    head = _git(repo, "rev-parse", "HEAD").strip()
    if head != expected_freeze_commit:
        raise ValueError("Round-8 review HEAD does not equal expected freeze commit")
    status = _git(
        repo, "status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching",
    )
    if status:
        raise ValueError("Round-8 review worktree or ignored runtime state is not clean")

    # Decode every new package document before any PASS-capable reconstruction.
    for name in ROUND8_JSON_FILES:
        strict_json_load(artifact / name)
    historical = strict_json_loads(
        _git(repo, "show", f"{ROUND7_REJECTED_FREEZE_COMMIT}:"
             "artifacts/gcspo_stage0_static_rerun/round7_freeze_manifest.json"),
        label="historical Round-7 freeze manifest",
    )

    manifest_path = artifact / "round8_freeze_manifest.json"
    manifest = strict_json_load(manifest_path)
    if (manifest.get("schema") != "gnss-doppler-lab.gcspo-stage0.round8-successor-freeze.v1" or
            manifest.get("state") != "READY_FOR_FRESH_INDEPENDENT_READ_ONLY_REVIEW" or
            manifest.get("source_commit") != SOURCE_COMMIT or
            manifest.get("rejected_round7_freeze_commit") != ROUND7_REJECTED_FREEZE_COMMIT or
            manifest.get("freeze_parent_sha") != ROUND8_REPAIR_COMMIT or
            manifest.get("ancestry") != list(ANCESTRY) + ["COMMIT_CONTAINING_THIS_MANIFEST"] or
            manifest.get("manifest_excludes_self") is not True or
            manifest.get("self_binding_exclusions") != [FREEZE_RELATIVE] or
            manifest.get("protected_access_authorized") is not False):
        raise ValueError("Round-8 successor freeze contract mismatch")
    _require_contract(manifest, "freeze")

    parent = _git(repo, "rev-parse", f"{expected_freeze_commit}^").strip()
    if parent != ROUND8_REPAIR_COMMIT:
        raise ValueError("Round-8 successor freeze parent mismatch")
    for ancestor in ANCESTRY:
        if subprocess.run(
                ["git", "merge-base", "--is-ancestor", ancestor, expected_freeze_commit],
                cwd=repo, capture_output=True).returncode:
            raise ValueError(f"Round-8 successor ancestry mismatch: {ancestor}")
    committed_manifest = _git(repo, "show", f"{expected_freeze_commit}:{FREEZE_RELATIVE}", text=False)
    if manifest_path.read_bytes() != committed_manifest:
        raise ValueError("Round-8 freeze manifest is not from expected commit")

    changes = _git(
        repo, "diff", "--name-only", ROUND7_REJECTED_FREEZE_COMMIT, expected_freeze_commit,
    ).splitlines()
    successor_changes = sorted(set(changes) - {FREEZE_RELATIVE})
    if manifest.get("successor_change_paths") != successor_changes:
        raise ValueError("Round-8 successor change coverage differs from exact Git diff")
    prior_paths = {row["path"] for row in historical.get("files", [])}
    required_paths = sorted(
        prior_paths | set(successor_changes) |
        {"artifacts/gcspo_stage0_static_rerun/round7_freeze_manifest.json"}
    )
    rows = manifest.get("files")
    if not isinstance(rows, list) or [row.get("path") for row in rows] != required_paths:
        raise ValueError("Round-8 freeze file set is not the predecessor/change union")
    if len(required_paths) != len(set(required_paths)):
        raise ValueError("Round-8 freeze paths are duplicated")

    for row in rows:
        if set(row) != {"path", "sha256", "size_bytes"}:
            raise ValueError("Round-8 freeze identity row is malformed")
        path = (repo / row["path"]).resolve(strict=True)
        if repo not in path.parents:
            raise ValueError(f"Round-8 freeze path escapes repository: {row['path']}")
        if row["path"].endswith(".json"):
            strict_json_load(path)
        if _identity(path) != {"sha256": row["sha256"], "size_bytes": row["size_bytes"]}:
            raise ValueError(f"Round-8 freeze identity mismatch: {row['path']}")
        committed = _git(repo, "show", f"{expected_freeze_commit}:{row['path']}", text=False)
        if path.read_bytes() != committed:
            raise ValueError(f"Round-8 freeze member is not committed: {row['path']}")

    verdict = artifact / "round8_round7_independent_review_rejection.txt"
    if _identity(verdict) != {
            "sha256": ROUND7_VERDICT_SHA256, "size_bytes": ROUND7_VERDICT_SIZE_BYTES}:
        raise ValueError("Round-7 reviewer verdict is not preserved verbatim")
    handoff = strict_json_load(artifact / "round8_repair_review_handoff.json")
    audit = strict_json_load(artifact / "round8_audit_report.json")
    for document, label in ((handoff, "handoff"), (audit, "audit")):
        if (document.get("state") != "READY_FOR_FRESH_INDEPENDENT_READ_ONLY_REVIEW" or
                document.get("freeze_parent_sha") != ROUND8_REPAIR_COMMIT or
                document.get("final_freeze_sha_binding") != "COMMIT_CONTAINING_THIS_DOCUMENT"):
            raise ValueError(f"Round-8 {label} commit binding mismatch")
        _require_contract(document, label)
    rejection = handoff.get("round7_rejection_evidence", {})
    if rejection != {
        "path": "artifacts/gcspo_stage0_static_rerun/round8_round7_independent_review_rejection.txt",
        "reviewed_sha": ROUND7_REJECTED_FREEZE_COMMIT,
        "sha256": ROUND7_VERDICT_SHA256,
        "size_bytes": ROUND7_VERDICT_SIZE_BYTES,
        "verdict": "INDEPENDENT_REVIEW_REJECT",
    }:
        raise ValueError("Round-8 handoff rejection evidence binding mismatch")
    if _identity(artifact / "README.md")["sha256"] != README_SHA256:
        raise ValueError("Round-8 frozen README identity mismatch")
    marker = artifact.parent / f".{artifact.name}.protected_run_started.json"
    if marker.exists() or (artifact / "access_ledger.jsonl").exists():
        raise ValueError("Round-8 protected marker or ledger exists")

    evidence = _verify_evidence_manifest(artifact)
    result = verify_round6_a5(artifact)
    parity = result["parity"]["cpu_cuda"]
    if (parity.get("numeric_field_count") != EXPECTED_NUMERIC_FIELD_COUNT or
            parity.get("numeric_path_sha256") != EXPECTED_NUMERIC_PATH_SHA256 or
            parity.get("numeric_coverage_status") != "EXACT_CONTRACT_MATCH"):
        raise ValueError("Round-8 reconstructed parity coverage contract mismatch")
    return {
        "status": "PASS",
        "freeze_commit": expected_freeze_commit,
        "freeze_parent_sha": parent,
        "source_commit": SOURCE_COMMIT,
        "freeze_file_count": len(rows),
        "successor_change_file_count": len(successor_changes),
        "successor_evidence_file_count": len(evidence["files"]),
        "evidence_file_count": len(result["manifest"]["files"]),
        "signature_count": 6,
        "run_count": len(result["witnessed"]["runs"]),
        "independence": result["witnessed"]["independence"],
        "parity": result["parity"],
        "protected_rows_opened": 0,
        "protected_bytes_opened": 0,
        "attack_run_count": 0,
        "push_performed": False,
    }
