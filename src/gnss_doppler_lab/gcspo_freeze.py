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
        config_resource = ("config" in candidate.name.lower() and candidate.suffix.lower() in root_suffixes
                           and not path.startswith("artifacts/"))
        if in_runtime_tree or root_runtime_file or config_resource: relevant.append(line)
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
