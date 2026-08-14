from __future__ import annotations

import ast
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
REJECTED_WRAPPER_COMMIT = "078a8c5739c92e877763583ce2ca23dad4f433f9"
REJECTED_TARGET_COMMIT = "1bfc3d2b64ad43a9db78081f3abc482bdf5d022f"
REQUIRED_INTERNAL_DEPENDENCY_PATHS = (
    "src/gnss_doppler_lab/gcmr_geometry.py",
    "src/gnss_doppler_lab/trajectory.py",
)

MANIFEST_KEYS = frozenset({
    "schema", "state", "target_commit", "wrapper_commit_binding", "wrapper_parent_sha",
    "invocation_id", "nonce", "same_invocation_retry", "predecessor_freeze_commit",
    "invalid_evidence_commit", "config_sha256", "preregistration_sha256", "scope_policy",
    "implementation_paths", "internal_import_closure_paths", "files", "manifest_excludes_self",
    "protected_access_authorized", "protected_access_count", "protected_marker_present",
    "protected_ledger_size_bytes",
})
CONTROL_KEYS = frozenset({
    "schema", "state", "commit_binding", "invocation", "ancestry", "repair_scope",
    "frozen_inputs", "failed_invocation_preservation", "prior_protected_state",
    "successor_namespace", "commit_topology", "prohibitions",
})
HANDOFF_KEYS = frozenset({
    "schema", "state", "wrapper_commit_binding", "wrapper_parent_sha", "target_commit",
    "invocation_id", "nonce", "same_invocation_retry", "predecessor_freeze_commit",
    "invalid_evidence_commit", "repair_scope", "config_sha256", "preregistration_sha256",
    "red_green", "manifest_coverage", "protected", "prior_independent_rejection",
    "invalid_artifact_root_preserved", "push_performed", "independent_review_command",
})

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


PACKAGE_NAME = "gnss_doppler_lab"
PACKAGE_ROOT = "src/gnss_doppler_lab/"


def _in_scope(relative: str) -> bool:
    path = Path(relative)
    return bool(
        (relative.startswith("scripts/") and path.suffix == ".py" and "gcspo" in path.name) or
        (relative.startswith(PACKAGE_ROOT) and path.suffix == ".py" and path.name.startswith("gcspo")) or
        (relative.startswith("tests/") and path.suffix == ".py" and path.name.startswith("test_gcspo")) or
        relative == "config/gnss_gcspo_witness_ed25519.pub"
    )


def _module_name(relative: str) -> str | None:
    if not relative.startswith(PACKAGE_ROOT) or not relative.endswith(".py"):
        return None
    parts = relative[len(PACKAGE_ROOT):-3].split("/")
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join((PACKAGE_NAME, *parts))


def _resolve_internal_module(module: str, modules: dict[str, str]) -> str | None:
    if module == PACKAGE_NAME or module.startswith(PACKAGE_NAME + "."):
        return modules.get(module)
    return None


def _absolute_from_module(current: str | None, level: int, module: str | None) -> str | None:
    if level == 0:
        return module
    if current is None:
        raise ValueError("relative import outside an internal package module")
    package_parts = current.split(".")[:-1]
    if level > len(package_parts):
        raise ValueError("relative internal import escapes package")
    base = package_parts[:len(package_parts) - level + 1]
    if module:
        base.extend(module.split("."))
    return ".".join(base)


def _internal_import_paths(relative: str, payload: bytes,
                           modules: dict[str, str]) -> set[str]:
    current = _module_name(relative)
    try:
        tree = ast.parse(payload, filename=relative)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"cannot parse Python import graph at target: {relative}") from exc
    dependencies: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolved = _resolve_internal_module(alias.name, modules)
                if resolved is not None:
                    dependencies.add(resolved)
                elif alias.name == PACKAGE_NAME or alias.name.startswith(PACKAGE_NAME + "."):
                    raise ValueError(f"unresolved internal import in {relative}: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_from_module(current, node.level, node.module)
            if base is None:
                continue
            base_path = _resolve_internal_module(base, modules)
            alias_paths = {
                resolved for alias in node.names if alias.name != "*"
                for resolved in [_resolve_internal_module(f"{base}.{alias.name}", modules)]
                if resolved is not None
            }
            if base_path is not None:
                dependencies.add(base_path)
            dependencies.update(alias_paths)
            if ((base == PACKAGE_NAME or base.startswith(PACKAGE_NAME + ".")) and
                    base_path is None and not alias_paths):
                raise ValueError(f"unresolved internal import in {relative}: {base}")
    return dependencies


def _implementation_scope_at_commit(repo: Path, target_commit: str) -> tuple[list[str], list[str]]:
    check = subprocess.run(["git", "cat-file", "-e", f"{target_commit}^{{commit}}"],
                           cwd=repo, capture_output=True)
    if check.returncode:
        raise ValueError("target Git commit is absent")
    names = _run(repo, "ls-tree", "-r", "--name-only", target_commit).splitlines()
    base_paths = sorted(path for path in names if _in_scope(path))
    if not base_paths or len(base_paths) != len(set(base_paths)):
        raise ValueError("deterministic implementation scope is empty or duplicated")
    modules = {
        module: path for path in names
        for module in [_module_name(path)] if module is not None
    }
    selected = set(base_paths)
    pending = sorted(path for path in selected if path.endswith(".py"))
    parsed: set[str] = set()
    while pending:
        relative = pending.pop(0)
        if relative in parsed:
            continue
        parsed.add(relative)
        for dependency in sorted(_internal_import_paths(
                relative, _tree_bytes(repo, target_commit, relative), modules)):
            if dependency not in selected:
                selected.add(dependency)
                pending.append(dependency)
        pending.sort()
    closure = sorted(selected - set(base_paths))
    return sorted(selected), closure


def implementation_paths_at_commit(repo: str | Path, target_commit: str) -> list[str]:
    root = Path(repo).resolve(strict=True)
    return _implementation_scope_at_commit(root, target_commit)[0]


def internal_dependency_paths_at_commit(repo: str | Path, target_commit: str) -> list[str]:
    root = Path(repo).resolve(strict=True)
    return _implementation_scope_at_commit(root, target_commit)[1]

def _verify_exact_record(value: object, expected: dict, label: str) -> None:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} exact schema/key set mismatch")
    for key, expected_value in expected.items():
        observed = value[key]
        if type(observed) is not type(expected_value) or observed != expected_value:
            raise ValueError(f"{label} type/value mismatch: {key}")


def verify_manifest_protected_state(manifest: dict) -> None:
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise ValueError("successor manifest exact schema/key set mismatch")
    _verify_exact_record(
        {key: manifest[key] for key in (
            "protected_access_authorized", "protected_access_count",
            "protected_marker_present", "protected_ledger_size_bytes")},
        {"protected_access_authorized": False, "protected_access_count": 0,
         "protected_marker_present": False, "protected_ledger_size_bytes": 0},
        "successor manifest protected state",
    )


def verify_control_protected_state(control: dict) -> None:
    if not isinstance(control, dict) or set(control) != CONTROL_KEYS:
        raise ValueError("successor control exact schema/key set mismatch")
    _verify_exact_record(control.get("prior_protected_state"), {
        "protected_access_count": 0, "protected_rows_opened": 0,
        "protected_bytes_opened": 0,
        "marker_path": "artifacts/.gcspo_stage0_static_rerun.protected_run_started.json",
        "marker_present": False,
        "ledger_path": "artifacts/gcspo_stage0_static_rerun/access_ledger.jsonl",
        "ledger_present": True, "ledger_size_bytes": 0,
        "final_verdict_present": False,
    }, "successor control prior protected state")
    _verify_exact_record(control.get("successor_namespace"), {
        "artifact_root": ARTIFACT_RELATIVE,
        "marker_path": (
            "artifacts/gcspo_stage0_static_rerun_successor/."
            f"{INVOCATION_ID}.protected_run_started.json"),
        "marker_present": False,
        "ledger_path": f"{ARTIFACT_RELATIVE}/access_ledger.jsonl",
        "ledger_size_bytes": 0, "protected_access_authorized": False,
    }, "successor control namespace protected state")


def verify_handoff_protected_state(handoff: dict) -> None:
    if not isinstance(handoff, dict) or set(handoff) != HANDOFF_KEYS:
        raise ValueError("review handoff exact schema/key set mismatch")
    _verify_exact_record(handoff.get("protected"), {
        "access_count": 0, "marker_present": False,
        "ledger_size_bytes": 0, "authorized": False,
    }, "review handoff protected state")


def build_successor_manifest(repo: str | Path, *, target_commit: str, invocation_id: str,
                             nonce: str, predecessor_freeze_commit: str,
                             invalid_evidence_commit: str, config_sha256: str,
                             preregistration_sha256: str) -> dict:
    root = Path(repo).resolve(strict=True)
    paths, closure = _implementation_scope_at_commit(root, target_commit)
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
        "scope_policy": "EXACT_SORTED_GIT_TREE_GCSPO_ROOTS_PLUS_TRANSITIVE_INTERNAL_PYTHON_IMPORT_CLOSURE",
        "implementation_paths": paths,
        "internal_import_closure_paths": closure,
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
    verify_manifest_protected_state(manifest)
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
    expected_paths, expected_closure = _implementation_scope_at_commit(root, expected_target_commit)
    if manifest.get("scope_policy") != (
            "EXACT_SORTED_GIT_TREE_GCSPO_ROOTS_PLUS_TRANSITIVE_INTERNAL_PYTHON_IMPORT_CLOSURE"):
        raise ValueError("implementation scope policy mismatch")
    if manifest.get("implementation_paths") != expected_paths:
        raise ValueError("implementation path set has missing or extra row")
    if manifest.get("internal_import_closure_paths") != expected_closure:
        raise ValueError("internal import closure path set has missing or extra row")
    missing_direct_dependencies = sorted(
        set(REQUIRED_INTERNAL_DEPENDENCY_PATHS) - set(expected_closure))
    if missing_direct_dependencies:
        raise ValueError(
            f"required direct internal dependency missing: {missing_direct_dependencies}")
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
        wrapper_paths = implementation_paths_at_commit(root, wrapper_commit)
        exact_implementation_paths = set(expected_paths) | set(wrapper_paths)
        changed_implementation = sorted(
            path for path in wrapper_changes if path in exact_implementation_paths)
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
    verify_control_protected_state(control)
    verify_handoff_protected_state(handoff)
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
    if handoff.get("prior_independent_rejection") != {
            "wrapper_commit": REJECTED_WRAPPER_COMMIT,
            "target_commit": REJECTED_TARGET_COMMIT,
            "verdict": "REJECT",
            "blocking_findings": [
                "TRANSITIVE_INTERNAL_IMPORT_CLOSURE_MISSING",
                "PROTECTED_STATE_SCHEMA_TYPE_VALUE_NOT_STRICT",
            ]}:
        raise ValueError("prior independent rejection evidence mismatch")
    if subprocess.run(["git", "merge-base", "--is-ancestor", REJECTED_WRAPPER_COMMIT, target],
                      cwd=root, capture_output=True).returncode:
        raise ValueError("rejected wrapper is not preserved in target ancestry")
    old = root / INVALID_ROOT_RELATIVE
    preservation = control.get("failed_invocation_preservation", {})
    expected_names = preservation.get("exact_regular_file_set")
    observed_names = sorted(p.relative_to(old).as_posix() for p in old.rglob("*") if p.is_file())
    if observed_names != expected_names:
        raise ValueError("invalid artifact root exact-set contract mismatch")
    expected_old = preservation.get("identities")
    observed_old = []
    for expected in expected_old if isinstance(expected_old, list) else ():
        relative = expected.get("path")
        if not isinstance(relative, str):
            raise ValueError("invalid artifact root identity row is malformed")
        path = root / relative
        payload = path.read_bytes() if path.is_file() and not path.is_symlink() else b""
        observed_old.append({"path": relative, "sha256": hashlib.sha256(payload).hexdigest(),
                             "size_bytes": len(payload)})
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
