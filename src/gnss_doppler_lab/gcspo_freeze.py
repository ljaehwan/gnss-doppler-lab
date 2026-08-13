"""Byte-bound implementation freeze and atomic protected-attempt claim."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


def _identity(path):
    path = Path(path).resolve(strict=True)
    if path.is_symlink() or not path.is_file(): raise ValueError("freeze member must be a regular file")
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def build_freeze_record(*, target_commit, config_sha256, implementation_files, clean_files, review_evidence):
    if len(target_commit) != 40 or len(config_sha256) != 64:
        raise ValueError("freeze commit/config identity is incomplete")
    if review_evidence.get("status") != "PASS" or not review_evidence.get("reviewer"):
        raise ValueError("independent review evidence is incomplete")
    implementation = sorted((_identity(path) for path in implementation_files), key=lambda row: row["path"])
    clean = sorted((_identity(path) for path in clean_files), key=lambda row: row["path"])
    if not implementation or not clean: raise ValueError("freeze file sets are incomplete")
    return {"schema": "gnss-doppler-lab.gcspo-stage0.implementation-freeze.v2",
            "validity_state": "VALID_FOR_PROTECTED_ACCESS", "target_commit": target_commit,
            "config_sha256": config_sha256, "implementation_files": implementation,
            "clean_scientific_artifacts": clean, "review_evidence": dict(review_evidence),
            "manifest_excludes_self": True}


def build_review_candidate_record(*, target_commit, config_sha256, implementation_files,
                                  clean_files, rejected_freeze_commit,
                                  prior_rejected_freeze_commits=()):
    """Build a non-self-referential repair freeze awaiting independent rereview."""
    if len(target_commit) != 40 or len(rejected_freeze_commit) != 40 or target_commit == rejected_freeze_commit:
        raise ValueError("repair target/rejected freeze identity is incomplete")
    rejected = [rejected_freeze_commit, *prior_rejected_freeze_commits]
    if (any(not isinstance(value, str) or len(value) != 40 for value in rejected) or
            len(rejected) != len(set(rejected)) or target_commit in rejected):
        raise ValueError("rejected freeze ancestry binding is incomplete")

    if len(config_sha256) != 64:
        raise ValueError("repair config identity is incomplete")
    implementation = sorted((_identity(path) for path in implementation_files), key=lambda row: row["path"])
    clean = sorted((_identity(path) for path in clean_files), key=lambda row: row["path"])
    if not implementation or not clean:
        raise ValueError("repair freeze file sets are incomplete")
    return {"schema": "gnss-doppler-lab.gcspo-stage0.implementation-freeze.v3",
            "validity_state": "AWAITING_INDEPENDENT_REREVIEW", "target_commit": target_commit,
            "config_sha256": config_sha256, "implementation_files": implementation,
            "clean_scientific_artifacts": clean,
            "review_evidence": {"status": "REPAIRS_COMPLETE_AWAITING_REREVIEW",
                                "rejected_freeze_commit": rejected_freeze_commit,
                                "rejected_freeze_commits": rejected},
            "manifest_excludes_self": True, "protected_access_authorized": False}


def verify_review_candidate_record(record, *, target_commit):
    if record.get("schema") != "gnss-doppler-lab.gcspo-stage0.implementation-freeze.v3":
        raise ValueError("repair freeze schema mismatch")
    if record.get("validity_state") != "AWAITING_INDEPENDENT_REREVIEW" or record.get("target_commit") != target_commit:
        raise ValueError("repair freeze target/state mismatch")
    if record.get("protected_access_authorized") is not False or record.get("manifest_excludes_self") is not True:
        raise ValueError("repair freeze access/self-reference contract mismatch")
    review = record.get("review_evidence", {})
    rejected = review.get("rejected_freeze_commits")
    if (review.get("status") != "REPAIRS_COMPLETE_AWAITING_REREVIEW" or
            not isinstance(rejected, list) or not rejected or len(rejected) != len(set(rejected)) or
            any(not isinstance(value, str) or len(value) != 40 for value in rejected) or
            review.get("rejected_freeze_commit") != rejected[0]):
        raise ValueError("repair freeze review evidence mismatch")
    _verify_rows(record.get("implementation_files"), "implementation")
    _verify_rows(record.get("clean_scientific_artifacts"), "clean artifact")
    return True


def _verify_rows(rows, kind):
    if not isinstance(rows, list) or not rows: raise ValueError(f"{kind} freeze set is empty")
    paths = [row.get("path") for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)): raise ValueError(f"{kind} paths are not sorted/unique")
    for row in rows:
        observed = _identity(row["path"])
        if observed != row: raise ValueError(f"{kind} hash mismatch: {row['path']}")


def verify_freeze_record(record, *, target_commit):
    if record.get("schema") != "gnss-doppler-lab.gcspo-stage0.implementation-freeze.v2":
        raise ValueError("freeze schema mismatch")
    if record.get("validity_state") != "VALID_FOR_PROTECTED_ACCESS" or record.get("target_commit") != target_commit:
        raise ValueError("freeze target/state mismatch")
    if len(str(record.get("config_sha256", ""))) != 64 or record.get("manifest_excludes_self") is not True:
        raise ValueError("freeze config/self-reference contract mismatch")
    review = record.get("review_evidence", {})
    if review.get("status") != "PASS" or not review.get("reviewer"):
        raise ValueError("freeze independent review evidence mismatch")
    _verify_rows(record.get("implementation_files"), "implementation")
    _verify_rows(record.get("clean_scientific_artifacts"), "clean artifact")
    return True


def claim_protected_attempt(marker_path, payload):
    """Create the one-shot marker exactly once and durably."""
    path = Path(marker_path); path.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema": "gnss-doppler-lab.gcspo-stage0.protected-run-start.v1",
                "protected_run_count": 1, **dict(payload)}
    data = (json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(data): offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try: os.fsync(parent)
    finally: os.close(parent)
    return document


def validate_then_claim(marker_path, *, freeze, expected_commit, live_remote_sha,
                        implementation_hashes, clean_hashes):
    verify_freeze_record(freeze, target_commit=expected_commit)
    if live_remote_sha != expected_commit: raise ValueError("live remote does not equal freeze target")
    expected_implementation = {row["path"]: row["sha256"] for row in freeze["implementation_files"]}
    expected_clean = {row["path"]: row["sha256"] for row in freeze["clean_scientific_artifacts"]}
    if implementation_hashes != expected_implementation or clean_hashes != expected_clean:
        raise ValueError("recomputed freeze hashes differ")
    return claim_protected_attempt(marker_path, {"target_commit": expected_commit,
                                                  "live_remote_sha": live_remote_sha})


def _runtime_relevant_ignored(status):
    relevant = []
    root_suffixes = {".py", ".pyc", ".so", ".pth", ".json", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
    for line in status.splitlines():
        if not line.startswith("!! "): continue
        path = line[3:].strip().strip('"')
        candidate = Path(path)
        in_runtime_tree = path.startswith("src/") or path.startswith("scripts/")
        root_runtime_file = len(candidate.parts) == 1 and (candidate.suffix.lower() in root_suffixes or
                                                          candidate.name in {".env", "sitecustomize.py", "usercustomize.py"})
        generated_residue = (candidate.name == ".pytest_cache" or
                             "__pycache__" in candidate.parts or candidate.suffix.lower() == ".pyc" or
                             (len(candidate.parts) == 1 and candidate.name.startswith("iq.bin.") and
                              candidate.suffix == ".tmp"))
        config_resource = ("config" in candidate.name.lower() and candidate.suffix.lower() in root_suffixes
                           and not path.startswith("artifacts/"))
        if in_runtime_tree or root_runtime_file or generated_residue or config_resource: relevant.append(line)
    return relevant


def live_remote_snapshot(repo_root, remote, branch):
    root = Path(repo_root)
    query = subprocess.run(["git", "ls-remote", remote, f"refs/heads/{branch}"], cwd=root,
                           check=True, text=True, capture_output=True).stdout.strip().splitlines()
    if len(query) != 1 or len(query[0].split()) != 2:
        raise ValueError("live remote branch identity is absent or ambiguous")
    remote_sha = query[0].split()[0]
    local_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                               text=True, capture_output=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=root,
                            check=True, text=True, capture_output=True).stdout.strip()
    ignored = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all", "--ignored"], cwd=root,
                             check=True, text=True, capture_output=True).stdout.strip()
    runtime_ignored = _runtime_relevant_ignored(ignored)
    fully_clean = not status and not runtime_ignored
    synchronized = len(remote_sha) == 40 and remote_sha == local_sha and fully_clean
    return {"query": "git ls-remote", "remote": remote, "branch": branch,
            "local_sha": local_sha, "remote_sha": remote_sha, "ahead": 0 if synchronized else None,
            "behind": 0 if synchronized else None, "tracked_source_clean": fully_clean,
            "git_status": status.splitlines(), "runtime_relevant_ignored": runtime_ignored,
            "synchronized": synchronized}


def validate_protected_manifest_inventory(inventory, *, required=("DS3", "DS4", "DS7", "DS8")):
    rows = {row.get("id"): row for row in inventory.get("scenario_inventory", []) if isinstance(row, dict)}
    result = {}
    for scenario in required:
        row = rows.get(scenario, {})
        sha = row.get("receiver_manifest_sha256"); size = row.get("receiver_manifest_size_bytes")
        path = row.get("receiver_manifest_path")
        if not isinstance(path, str) or not path or len(str(sha or "")) != 64 or isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"{scenario} authenticated manifest identity is incomplete")
        result[scenario] = {"path": path, "sha256": sha, "size_bytes": size}
    return result
