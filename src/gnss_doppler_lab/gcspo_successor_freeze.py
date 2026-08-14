from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

SCHEMA = "gnss-doppler-lab.gcspo-stage0.successor-implementation-freeze.v1"
CONTROL_SCHEMA = "gnss-doppler-lab.gcspo-stage0.successor-invocation-control.v1"
HANDOFF_SCHEMA = "gnss-doppler-lab.gcspo-stage0.successor-freeze-review-handoff.v1"
CONFIG_SHA256 = "0db816116b95b41db8b7af7379cd7411cc52d43b6428ae00ab02d6ccac19f4ad"
PREREGISTRATION_SHA256 = "715e11965854f785487e9d2c747718747c1d31cdd8603696ea7af126a45a70da"
PREDECESSOR_FREEZE_COMMIT = "0ab94567938234ca925f0bb8fbaece41e7d5e4a3"
INVALID_EVIDENCE_COMMIT = "58943724ea278b754d388ec2dca0f3666ed6c8a2"
PREREGISTRATION_COMMIT = "c2a6447cd2d8f989848d7e016bfe8c17a87f64ea"
INVOCATION_ID = "gcspo-stage0-successor-7be48c4411644ff3a9ec41c7701dfa01"
INVOCATION_NONCE = "c8b66154edba6ce2fc2418d97186a1c452ab8a3ac68afb8f040eedd25c6d23c3"
ARTIFACT_RELATIVE = f"artifacts/gcspo_stage0_static_rerun_successor/{INVOCATION_ID}"
CONTROL_RELATIVE = f"{ARTIFACT_RELATIVE}/successor_control.json"
MANIFEST_RELATIVE = f"{ARTIFACT_RELATIVE}/implementation_manifest.json"
HANDOFF_RELATIVE = f"{ARTIFACT_RELATIVE}/review_handoff.json"
INVALID_ROOT_RELATIVE = "artifacts/gcspo_stage0_static_rerun"

STALE_CROSS_GENERATION_PATHS = (
    "scripts/run_gcspo_clean_a5.py",
    "src/gnss_doppler_lab/gcspo_provenance.py",
    "src/gnss_doppler_lab/gcspo_verify.py",
    "tests/test_gcspo_round4_freeze_repairs.py",
)
LATER_GCSPO_PATHS = (
    "scripts/build_gcspo_round8_package.py",
    "scripts/run_gcspo_a5_witnessed.py",
    "scripts/verify_gcspo_round6_freeze.py",
    "scripts/verify_gcspo_round8_freeze.py",
    "src/gnss_doppler_lab/gcspo_round5_verify.py",
    "src/gnss_doppler_lab/gcspo_round6_verify.py",
    "src/gnss_doppler_lab/gcspo_round8_verify.py",
    "src/gnss_doppler_lab/gcspo_witness.py",
    "tests/test_gcspo_round5_child_receipt.py",
    "tests/test_gcspo_round5_witnessed_provenance.py",
    "tests/test_gcspo_round6_phase2.py",
    "tests/test_gcspo_round6_successor.py",
    "tests/test_gcspo_round8_repairs.py",
)


def _run(repo: Path, *args: str, text: bool = True):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=text).stdout


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_bytes(payload: bytes, label: str) -> dict:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_strict_pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               ValueError(f"non-finite JSON constant: {value}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _tree_bytes(repo: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=repo,
                            capture_output=True)
    if result.returncode:
        raise ValueError(f"path absent from target Git tree: {relative}")
    return result.stdout


def _identity(relative: str, payload: bytes) -> dict:
    return {"path": relative, "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload)}


def _in_scope(relative: str) -> bool:
    path = Path(relative)
    return bool(
        (relative.startswith("scripts/") and path.suffix == ".py" and "gcspo" in path.name) or
        (relative.startswith("src/gnss_doppler_lab/") and path.suffix == ".py" and path.name.startswith("gcspo")) or
        (relative.startswith("tests/") and path.suffix == ".py" and path.name.startswith("test_gcspo")) or
        relative == "config/gnss_gcspo_witness_ed25519.pub"
    )


def implementation_paths_at_commit(repo: str | Path, target_commit: str) -> list[str]:
    root = Path(repo).resolve(strict=True)
    check = subprocess.run(["git", "cat-file", "-e", f"{target_commit}^{{commit}}"],
                           cwd=root, capture_output=True)
    if check.returncode:
        raise ValueError("target Git commit is absent")
    names = _run(root, "ls-tree", "-r", "--name-only", target_commit).splitlines()
    paths = sorted(path for path in names if _in_scope(path))
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("deterministic implementation scope is empty or duplicated")
    return paths


def build_successor_manifest(repo: str | Path, *, target_commit: str, invocation_id: str,
                             nonce: str, predecessor_freeze_commit: str,
                             invalid_evidence_commit: str, config_sha256: str,
                             preregistration_sha256: str) -> dict:
    root = Path(repo).resolve(strict=True)
    paths = implementation_paths_at_commit(root, target_commit)
    rows = [_identity(path, _tree_bytes(root, target_commit, path)) for path in paths]
    return {
        "schema": SCHEMA,
        "state": "READY_FOR_FRESH_INDEPENDENT_READ_ONLY_REVIEW",
        "target_commit": target_commit,
        "wrapper_commit_binding": "COMMIT_CONTAINING_THIS_DOCUMENT",
        "wrapper_parent_sha": target_commit,
        "invocation_id": invocation_id,
        "nonce": nonce,
        "same_invocation_retry": False,
        "predecessor_freeze_commit": predecessor_freeze_commit,
        "invalid_evidence_commit": invalid_evidence_commit,
        "config_sha256": config_sha256,
        "preregistration_sha256": preregistration_sha256,
        "scope_policy": "EXACT_SORTED_GIT_LS_TREE_GCSPO_SCRIPTS_LIBRARY_TESTS_AND_WITNESS_PUBLIC_KEY",
        "implementation_paths": paths,
        "files": rows,
        "manifest_excludes_self": True,
        "protected_access_authorized": False,
        "protected_access_count": 0,
        "protected_marker_present": False,
        "protected_ledger_size_bytes": 0,
    }


def verify_successor_manifest(manifest: dict, repo: str | Path, *,
                              expected_target_commit: str,
                              wrapper_commit: str | None = None,
                              require_clean_worktree: bool = False,
                              require_repository_coverage: bool = False) -> bool:
    root = Path(repo).resolve(strict=True)
    if (manifest.get("schema") != SCHEMA or
            manifest.get("state") != "READY_FOR_FRESH_INDEPENDENT_READ_ONLY_REVIEW" or
            manifest.get("target_commit") != expected_target_commit or
            manifest.get("wrapper_parent_sha") != expected_target_commit or
            manifest.get("wrapper_commit_binding") != "COMMIT_CONTAINING_THIS_DOCUMENT" or
            manifest.get("manifest_excludes_self") is not True or
            manifest.get("protected_access_authorized") is not False or
            manifest.get("same_invocation_retry") is not False):
        raise ValueError("successor manifest target/state contract mismatch")
    if not isinstance(manifest.get("nonce"), str) or len(manifest["nonce"]) != 64:
        raise ValueError("successor invocation nonce is invalid")
    expected_paths = implementation_paths_at_commit(root, expected_target_commit)
    if manifest.get("implementation_paths") != expected_paths:
        raise ValueError("implementation path set has missing or extra row")
    rows = manifest.get("files")
    if not isinstance(rows, list) or [row.get("path") for row in rows] != expected_paths:
        raise ValueError("implementation row set has missing or extra row")
    for row in rows:
        if set(row) != {"path", "sha256", "size_bytes"}:
            raise ValueError("implementation row shape mismatch")
        expected = _identity(row["path"], _tree_bytes(root, expected_target_commit, row["path"]))
        if row != expected:
            raise ValueError("implementation target identity hash/size mismatch: %s" % row.get("path"))
    if require_repository_coverage:
        required = set(STALE_CROSS_GENERATION_PATHS) | set(LATER_GCSPO_PATHS)
        missing = sorted(required - set(expected_paths))
        if missing:
            raise ValueError(f"required stale/later GCSPO coverage missing: {missing}")
    if wrapper_commit is not None:
        parent = _run(root, "rev-parse", f"{wrapper_commit}^").strip()
        if parent != expected_target_commit:
            raise ValueError("wrapper immediate parent does not equal target commit")
        wrapper_changes = _run(root, "diff-tree", "--no-commit-id", "--name-only", "-r", wrapper_commit).splitlines()
        changed_implementation = sorted(path for path in wrapper_changes if _in_scope(path))
        if changed_implementation:
            raise ValueError(f"post-target implementation source change in wrapper: {changed_implementation}")
    if require_clean_worktree:
        for relative in expected_paths:
            path = root / relative
            if not path.is_file() or path.is_symlink() or path.read_bytes() != _tree_bytes(
                    root, expected_target_commit, relative):
                raise ValueError(f"working-tree shadow differs from target: {relative}")
        status = _run(root, "status", "--porcelain=v1", "--untracked-files=all")
        if status:
            raise ValueError("working-tree shadow/status is not clean")
    return True


def _file_identity(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required regular file absent: {path}")
    payload = path.read_bytes()
    return {"path": path.as_posix(), "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload)}


def verify_successor_freeze(repo: str | Path, artifact_root: str | Path,
                            expected_wrapper_commit: str) -> dict:
    root = Path(repo).resolve(strict=True)
    artifact = Path(artifact_root).resolve(strict=True)
    head = _run(root, "rev-parse", "HEAD").strip()
    if head != expected_wrapper_commit:
        raise ValueError("review HEAD does not equal expected wrapper commit")
    control_path = artifact / "successor_control.json"
    manifest_path = artifact / "implementation_manifest.json"
    handoff_path = artifact / "review_handoff.json"
    control = strict_json_bytes(control_path.read_bytes(), "successor control")
    manifest = strict_json_bytes(manifest_path.read_bytes(), "implementation manifest")
    handoff = strict_json_bytes(handoff_path.read_bytes(), "review handoff")
    for path, relative in ((control_path, CONTROL_RELATIVE), (manifest_path, MANIFEST_RELATIVE),
                           (handoff_path, HANDOFF_RELATIVE)):
        if path.read_bytes() != _tree_bytes(root, expected_wrapper_commit, relative):
            raise ValueError(f"wrapper document is not committed exactly: {relative}")
    if (control.get("schema") != CONTROL_SCHEMA or
            control.get("state") != "IMMUTABLY_PREREGISTERED_MANIFEST_REPAIR_ONLY" or
            control.get("commit_binding") != "COMMIT_CONTAINING_THIS_DOCUMENT"):
        raise ValueError("successor control contract mismatch")
    invocation = control.get("invocation", {})
    ancestry = control.get("ancestry", {})
    frozen = control.get("frozen_inputs", {})
    if invocation != {"invocation_id": INVOCATION_ID, "nonce": INVOCATION_NONCE,
                      "same_invocation_retry": False, "new_identity_required": True}:
        raise ValueError("successor invocation identity contract mismatch")
    if (ancestry.get("approved_predecessor_freeze_commit") != PREDECESSOR_FREEZE_COMMIT or
            ancestry.get("failed_invocation_evidence_commit") != INVALID_EVIDENCE_COMMIT or
            frozen != {"config_sha256": CONFIG_SHA256,
                       "preregistration_sha256": PREREGISTRATION_SHA256}):
        raise ValueError("successor ancestry/frozen hash contract mismatch")
    target = manifest.get("target_commit")
    if not isinstance(target, str) or len(target) != 40:
        raise ValueError("successor target commit is invalid")
    verify_successor_manifest(
        manifest, root, expected_target_commit=target, wrapper_commit=expected_wrapper_commit,
        require_clean_worktree=True, require_repository_coverage=True,
    )
    if (manifest.get("invocation_id") != INVOCATION_ID or
            manifest.get("nonce") != INVOCATION_NONCE or
            manifest.get("predecessor_freeze_commit") != PREDECESSOR_FREEZE_COMMIT or
            manifest.get("invalid_evidence_commit") != INVALID_EVIDENCE_COMMIT or
            manifest.get("config_sha256") != CONFIG_SHA256 or
            manifest.get("preregistration_sha256") != PREREGISTRATION_SHA256):
        raise ValueError("manifest does not match immutable control evidence")
    for ancestor in (PREDECESSOR_FREEZE_COMMIT, INVALID_EVIDENCE_COMMIT, PREREGISTRATION_COMMIT):
        if subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, target],
                          cwd=root, capture_output=True).returncode:
            raise ValueError(f"target ancestry mismatch: {ancestor}")
    if _tree_bytes(root, target, CONTROL_RELATIVE) != control_path.read_bytes():
        raise ValueError("control evidence changed after preregistration target")
    changes = _run(root, "diff-tree", "--no-commit-id", "--name-only", "-r",
                   expected_wrapper_commit).splitlines()
    if sorted(changes) != sorted([MANIFEST_RELATIVE, HANDOFF_RELATIVE]):
        raise ValueError("wrapper contains changes outside manifest and handoff")
    if (handoff.get("schema") != HANDOFF_SCHEMA or
            handoff.get("state") != "READY_FOR_FRESH_INDEPENDENT_READ_ONLY_REVIEW" or
            handoff.get("wrapper_commit_binding") != "COMMIT_CONTAINING_THIS_DOCUMENT" or
            handoff.get("target_commit") != target or
            handoff.get("wrapper_parent_sha") != target or
            handoff.get("invocation_id") != INVOCATION_ID):
        raise ValueError("review handoff binding mismatch")
    old = root / INVALID_ROOT_RELATIVE
    preservation = control.get("failed_invocation_preservation", {})
    expected_names = preservation.get("exact_regular_file_set")
    observed_names = sorted(p.relative_to(old).as_posix() for p in old.rglob("*") if p.is_file())
    if observed_names != expected_names:
        raise ValueError("invalid artifact root exact-set contract mismatch")
    expected_old = preservation.get("identities")
    observed_old = [_file_identity(p) for p in sorted(old.rglob("*")) if p.is_file()]
    if observed_old != expected_old:
        raise ValueError("invalid artifact root identity contract mismatch")
    config = artifact / "config.json"
    prereg = artifact / "preregistration.json"
    if (_file_identity(config)["sha256"] != CONFIG_SHA256 or
            _file_identity(prereg)["sha256"] != PREREGISTRATION_SHA256):
        raise ValueError("successor config/preregistration immutable hash mismatch")
    old_marker = root / control["prior_protected_state"]["marker_path"]
    new_marker = root / control["successor_namespace"]["marker_path"]
    old_ledger = root / control["prior_protected_state"]["ledger_path"]
    new_ledger = root / control["successor_namespace"]["ledger_path"]
    if old_marker.exists() or new_marker.exists() or old_ledger.stat().st_size != 0 or new_ledger.stat().st_size != 0:
        raise ValueError("protected marker/ledger state is not pre-access empty")
    if (old / "final_verdict.json").exists() or (artifact / "final_verdict.json").exists():
        raise ValueError("final verdict exists before protected access")
    return {
        "status": "PASS", "wrapper_commit": expected_wrapper_commit,
        "target_commit": target, "wrapper_parent_sha": target,
        "manifest_row_count": len(manifest["files"]),
        "stale_row_coverage": len(STALE_CROSS_GENERATION_PATHS),
        "later_gcspo_row_coverage": len(LATER_GCSPO_PATHS),
        "config_sha256": CONFIG_SHA256, "preregistration_sha256": PREREGISTRATION_SHA256,
        "protected_access_count": 0, "protected_marker_present": False,
        "protected_ledger_size_bytes": 0, "push_performed": False,
    }
