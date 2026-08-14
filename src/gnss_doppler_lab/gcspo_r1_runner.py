"""Fail-closed R1 protected runner for the preregistered frozen completion."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .gcspo_artifacts import FROZEN_HASHES, canonical_write_json, quarantine_failed_final_verdict
from .gcspo_capabilities import validate_preaccess_capabilities
from .gcspo_core import AccessGate
from .gcspo_evaluate import validate_clean_contrast_preaccess
from .gcspo_freeze import validate_protected_manifest_inventory
from .gcspo_r1_support import (
    exact_b0_full_contrast_r1,
    install_r1_support_adapter,
)

INVOCATION_ID = "gcspo-stage0-r1-frozen-completion-0acca83b5245429b"
INVOCATION_NONCE = "97109aa102565cb9a5b7c8864ea766c4780b525c1be190dc56129b69f92a1535"
ARTIFACT_RELATIVE = "artifacts/gcspo_stage0_r1_frozen_completion"
CONFIG_RELATIVE = f"config/gcspo_r1_completion/{INVOCATION_ID}"
FREEZE_RELATIVE = f"{CONFIG_RELATIVE}/execution_freeze.json"
RECEIPT_RELATIVE = f"{CONFIG_RELATIVE}/independent_review_receipt.json"
MARKER_RELATIVE = f"{ARTIFACT_RELATIVE}/.protected_run_started.json"
BRANCH = "research/gcspo-stage0-r1-frozen-completion"


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=True, text=True,
                            capture_output=True)
    return result.stdout.strip()


def _git_bytes(root: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=root,
                            capture_output=True)
    if result.returncode:
        raise ValueError(f"required Git object absent: {commit}:{relative}")
    return result.stdout


def _strict_object(payload: bytes, label: str) -> dict:
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           ValueError(f"nonfinite JSON in {label}: {token}")))
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def install_and_verify_adapter() -> None:
    """Bind the exact-support contract into evaluator and verifier namespaces."""
    from . import gcspo_evaluate, gcspo_verify_artifacts

    install_r1_support_adapter()
    bindings = (
        gcspo_evaluate.exact_b0_full_contrast,
        gcspo_verify_artifacts.exact_b0_full_contrast,
    )
    if any(binding is not exact_b0_full_contrast_r1 for binding in bindings):
        raise ValueError("R1 evaluator/verifier adapter identity mismatch")


def verify_zero_access_state(artifact_dir: str | Path, marker: str | Path) -> None:
    artifact = Path(artifact_dir)
    marker_path = Path(marker)
    if marker_path.exists():
        raise ValueError("protected marker already exists")
    ledger = artifact / "access_ledger.jsonl"
    if ledger.exists() and ledger.stat().st_size:
        raise ValueError("protected access ledger is nonempty")
    if (artifact / "final_verdict.json").exists():
        raise ValueError("protected final verdict already exists")


def claim_once(marker: str | Path, *, wrapper_commit: str, target_commit: str) -> dict:
    document = {
        "schema": "gnss-doppler-lab.gcspo-stage0.r1-protected-run-start.v1",
        "protected_run_count": 1,
        "invocation_id": INVOCATION_ID,
        "nonce": INVOCATION_NONCE,
        "authorization_wrapper_commit": wrapper_commit,
        "target_commit": target_commit,
    }
    payload = (json.dumps(document, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    path = Path(marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return document


def _verify_identity_rows(root: Path, target: str, rows: object, label: str) -> list[dict]:
    if type(rows) is not list or not rows:
        raise ValueError(f"{label} must be a nonempty list")
    paths = []
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"{label} row schema mismatch")
        payload = _git_bytes(root, target, row["path"])
        if _sha(payload) != row["sha256"] or len(payload) != row["size_bytes"]:
            raise ValueError(f"{label} identity mismatch: {row['path']}")
        paths.append(row["path"])
    if paths != sorted(set(paths)):
        raise ValueError(f"{label} paths must be sorted and unique")
    return rows


def verify_execution_freeze(root: str | Path, *, check_remote: bool = True) -> dict:
    root = Path(root).resolve(strict=True)
    if _git(root, "status", "--porcelain"):
        raise ValueError("target worktree is not clean")
    wrapper = _git(root, "rev-parse", "HEAD")
    target = _git(root, "rev-parse", "HEAD^")
    changed = _git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", wrapper).splitlines()
    if sorted(changed) != sorted((FREEZE_RELATIVE, RECEIPT_RELATIVE)):
        raise ValueError("authorization wrapper exact diff mismatch")

    freeze = _strict_object(_git_bytes(root, wrapper, FREEZE_RELATIVE), "R1 execution freeze")
    receipt = _strict_object(_git_bytes(root, wrapper, RECEIPT_RELATIVE), "R1 review receipt")
    if freeze.get("schema") != "gnss-doppler-lab.gcspo-stage0.r1-execution-freeze.v1":
        raise ValueError("R1 execution freeze schema mismatch")
    if freeze.get("state") != "VALID_FOR_PROTECTED_ACCESS" or freeze.get("target_commit") != target:
        raise ValueError("R1 execution freeze target/state mismatch")
    if freeze.get("wrapper_commit_binding") != "COMMIT_CONTAINING_THIS_DOCUMENT":
        raise ValueError("R1 execution freeze wrapper binding mismatch")
    if receipt.get("schema") != "gnss-doppler-lab.gcspo-stage0.r1-independent-review.v1":
        raise ValueError("R1 review receipt schema mismatch")
    if (receipt.get("state") != "APPROVED" or receipt.get("passed") is not True or
            receipt.get("target_commit") != target or
            receipt.get("invocation_id") != INVOCATION_ID or
            receipt.get("nonce") != INVOCATION_NONCE or receipt.get("findings") != []):
        raise ValueError("R1 independent review receipt mismatch")

    _verify_identity_rows(root, target, freeze.get("runtime_files"), "runtime freeze")
    clean_rows = _verify_identity_rows(root, target, freeze.get("clean_files"), "clean input freeze")

    science = _strict_object(
        _git_bytes(root, target, f"{ARTIFACT_RELATIVE}/frozen_science_hashes.json"),
        "frozen science hashes",
    )
    for relative, expected in science.get("frozen_scientific_code", {}).items():
        if _sha(_git_bytes(root, target, relative)) != expected:
            raise ValueError(f"frozen science hash changed: {relative}")

    if check_remote:
        branch = _git(root, "branch", "--show-current")
        if branch != BRANCH:
            raise ValueError("R1 branch mismatch")
        subprocess.run(["git", "fetch", "origin", BRANCH], cwd=root, check=True,
                       capture_output=True, text=True)
        remote = _git(root, "rev-parse", f"origin/{BRANCH}")
        counts = _git(root, "rev-list", "--left-right", "--count",
                      f"HEAD...origin/{BRANCH}").split()
        if remote != wrapper or counts != ["0", "0"]:
            raise ValueError("R1 remote synchronization mismatch")
    return {"wrapper_commit": wrapper, "target_commit": target,
            "freeze": freeze, "receipt": receipt, "clean_rows": clean_rows}


def preflight(root: str | Path, *, check_remote: bool = True) -> dict:
    root = Path(root).resolve(strict=True)
    install_and_verify_adapter()
    checked = verify_execution_freeze(root, check_remote=check_remote)
    artifact = root / ARTIFACT_RELATIVE
    marker = root / MARKER_RELATIVE
    verify_zero_access_state(artifact, marker)

    prereg = _strict_object((artifact / "completion_preregistration.json").read_bytes(),
                            "R1 preregistration")
    if (prereg.get("invocation", {}).get("id") != INVOCATION_ID or
            prereg.get("invocation", {}).get("nonce") != INVOCATION_NONCE):
        raise ValueError("R1 preregistration invocation mismatch")
    inventory = _strict_object((artifact / "data_inventory.json").read_bytes(), "inventory")
    capabilities = validate_preaccess_capabilities(
        _strict_object((artifact / "protected_capabilities.json").read_bytes(), "capabilities"))
    manifest_identities = validate_protected_manifest_inventory(
        inventory, required=tuple(capabilities["available"]))
    for scenario, capability in capabilities["available"].items():
        if manifest_identities[scenario] != capability["manifest_identity"]:
            raise ValueError(f"{scenario} inventory/capability identity mismatch")

    clean_identities = []
    for row in checked["clean_rows"]:
        clean_identities.append({"path": str(root / row["path"]),
                                 "sha256": row["sha256"],
                                 "size_bytes": row["size_bytes"]})
    validate_clean_contrast_preaccess(artifact, clean_identities)
    return {**checked, "artifact": artifact, "marker": marker,
            "inventory": inventory, "capabilities": capabilities,
            "manifest_identities": manifest_identities,
            "clean_identities": clean_identities}


def _write_full_manifest(artifact: Path) -> None:
    path = artifact / "r1_artifact_manifest.json"
    rows = []
    for item in sorted(artifact.rglob("*")):
        if not item.is_file() or item == path:
            continue
        payload = item.read_bytes()
        rows.append({"path": item.relative_to(artifact).as_posix(),
                     "sha256": _sha(payload), "size_bytes": len(payload)})
    canonical_write_json(path, {
        "schema": "gnss-doppler-lab.gcspo-stage0.r1-artifact-manifest.v1",
        "invocation_id": INVOCATION_ID,
        "files": rows,
    })


def protected(root: str | Path, *, check_remote: bool = True) -> int:
    root = Path(root).resolve(strict=True)
    claimed = False
    artifact = root / ARTIFACT_RELATIVE
    try:
        checked = preflight(root, check_remote=check_remote)
        claim_once(checked["marker"], wrapper_commit=checked["wrapper_commit"],
                   target_commit=checked["target_commit"])
        claimed = True
        canonical_write_json(artifact / "protected_run_provenance.json", {
            "schema": "gnss-doppler-lab.gcspo-stage0.r1-protected-run-provenance.v1",
            "invocation_id": INVOCATION_ID, "nonce": INVOCATION_NONCE,
            "authorization_wrapper_commit": checked["wrapper_commit"],
            "target_commit": checked["target_commit"],
            "receipt_path": RECEIPT_RELATIVE, "execution_freeze_path": FREEZE_RELATIVE,
        })
        gate = AccessGate(artifact / "access_ledger.jsonl")
        gate.set_preflight(clean_only_pass=True, reviews_pass=True,
                           freeze_sha=checked["wrapper_commit"], frozen_hashes=FROZEN_HASHES)
        gate.set_remote_sync(local_sha=checked["wrapper_commit"],
                             remote_sha=checked["wrapper_commit"], ahead=0, behind=0,
                             clean=True)
        from . import gcspo_evaluate
        verdict = gcspo_evaluate.run_one_shot(
            artifact_dir=artifact, repo_root=root, inventory=checked["inventory"],
            gate=gate, manifest_identities=checked["manifest_identities"],
            clean_identities=checked["clean_identities"],
            capabilities=checked["capabilities"],
        )
        _write_full_manifest(artifact)
        print(f"PROTECTED_ONE_SHOT_PASS verdict={verdict}", flush=True)
        return 0
    except Exception as exc:
        if claimed:
            quarantine_failed_final_verdict(artifact)
        print(str(exc), file=sys.stderr)
        return 2


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    return protected(root)
