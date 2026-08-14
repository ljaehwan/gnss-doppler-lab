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
SECOND_REJECTED_WRAPPER_COMMIT = "34462807a2029a0979c9e65162246b799406c562"
SECOND_REJECTED_TARGET_COMMIT = "9121843f1d4835884af91883369dca62848c8dcd"
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
    "latest_independent_rejection",
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


def _absolute_from_module(current: str | None, current_is_package: bool,
                          level: int, module: str | None) -> str | None:
    if level == 0:
        return module
    if current is None:
        raise ValueError("relative import outside an internal package module")
    package = current if current_is_package else current.rpartition(".")[0]
    package_parts = package.split(".") if package else []
    parents = level - 1
    if parents >= len(package_parts):
        raise ValueError("relative internal import escapes package")
    base = package_parts[:len(package_parts) - parents]
    if module:
        base.extend(module.split("."))
    return ".".join(base)


def _statically_exported_names(payload: bytes, relative: str) -> set[str]:
    try:
        tree = ast.parse(payload, filename=relative)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"cannot parse Python import graph at target: {relative}") from exc
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for child in ast.walk(target):
                    if isinstance(child, ast.Name):
                        names.add(child.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
    return names


def _internal_import_paths(relative: str, payload: bytes,
                           modules: dict[str, str], packages: set[str],
                           package_names: set[str],
                           exports: dict[str, set[str]]) -> set[str]:
    current = _module_name(relative)
    current_is_package = bool(current is not None and current in packages)
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
            base = _absolute_from_module(current, current_is_package, node.level, node.module)
            if base is None:
                continue
            internal = base == PACKAGE_NAME or base.startswith(PACKAGE_NAME + ".")
            if not internal:
                continue
            base_path = _resolve_internal_module(base, modules)
            if base_path is not None:
                dependencies.add(base_path)
            elif base not in package_names:
                raise ValueError(f"unresolved internal import in {relative}: {base}")
            for alias in node.names:
                if alias.name == "*":
                    if base_path is None:
                        raise ValueError(
                            f"unresolved internal import in {relative}: {base}.*")
                    continue
                candidate = f"{base}.{alias.name}"
                candidate_path = _resolve_internal_module(candidate, modules)
                if candidate_path is not None:
                    dependencies.add(candidate_path)
                elif base in package_names:
                    if base not in packages or alias.name not in exports.get(base, set()):
                        raise ValueError(
                            f"unresolved internal import in {relative}: {candidate}")
                # A name imported from an ordinary module is a symbol, not a
                # candidate module dependency, and is deliberately not rejected.
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
    modules: dict[str, str] = {}
    packages: set[str] = set()
    exports: dict[str, set[str]] = {}
    for path in names:
        module = _module_name(path)
        if module is None:
            continue
        if module in modules:
            raise ValueError(f"duplicate internal module path in target Git tree: {module}")
        modules[module] = path
        payload = _tree_bytes(repo, target_commit, path)
        exports[module] = _statically_exported_names(payload, path)
        if path.endswith("/__init__.py"):
            packages.add(module)
    package_names = set(packages)
    for module in modules:
        parts = module.split(".")
        package_names.update(".".join(parts[:index]) for index in range(1, len(parts)))
    selected = set(base_paths)
    pending = sorted(path for path in selected if path.endswith(".py"))
    parsed: set[str] = set()
    while pending:
        relative = pending.pop(0)
        if relative in parsed:
            continue
        parsed.add(relative)
        for dependency in sorted(_internal_import_paths(
                relative, _tree_bytes(repo, target_commit, relative), modules, packages, package_names, exports)):
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

def _require_exact_keys(value: object, keys: set[str] | frozenset[str], label: str) -> dict:
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{label} exact schema/key set mismatch")
    return value


def _require_exact_type(value: object, expected_type: type, label: str) -> None:
    if type(value) is not expected_type:
        raise ValueError(f"{label} exact type mismatch")


def _require_int(value: object, label: str, *, minimum: int = 0) -> None:
    _require_exact_type(value, int, label)
    if value < minimum:
        raise ValueError(f"{label} value is invalid")


def _require_string(value: object, label: str, *, length: int | None = None) -> None:
    _require_exact_type(value, str, label)
    if length is not None and len(value) != length:
        raise ValueError(f"{label} length is invalid")


def _require_string_list(value: object, label: str) -> None:
    _require_exact_type(value, list, label)
    for index, item in enumerate(value):
        _require_string(item, f"{label}[{index}]")


def _verify_exact_record(value: object, expected: dict, label: str) -> None:
    record = _require_exact_keys(value, set(expected), label)
    for key, expected_value in expected.items():
        observed = record[key]
        if type(observed) is not type(expected_value) or observed != expected_value:
            raise ValueError(f"{label} type/value mismatch: {key}")


def _verify_identity_rows(rows: object, label: str) -> None:
    _require_exact_type(rows, list, label)
    for index, row in enumerate(rows):
        record = _require_exact_keys(row, {"path", "sha256", "size_bytes"},
                                     f"{label}[{index}]")
        _require_string(record["path"], f"{label}[{index}].path")
        _require_string(record["sha256"], f"{label}[{index}].sha256", length=64)
        _require_int(record["size_bytes"], f"{label}[{index}].size_bytes")


def verify_manifest_protected_state(manifest: dict) -> None:
    record = _require_exact_keys(manifest, MANIFEST_KEYS, "successor manifest")
    fixed = {
        "schema": SCHEMA,
        "state": "READY_FOR_FRESH_INDEPENDENT_READ_ONLY_REVIEW",
        "wrapper_commit_binding": "COMMIT_CONTAINING_THIS_DOCUMENT",
        "same_invocation_retry": False,
        "scope_policy":
            "EXACT_SORTED_GIT_TREE_GCSPO_ROOTS_PLUS_TRANSITIVE_INTERNAL_PYTHON_IMPORT_CLOSURE",
        "manifest_excludes_self": True,
        "protected_access_authorized": False,
        "protected_access_count": 0,
        "protected_marker_present": False,
        "protected_ledger_size_bytes": 0,
    }
    for key, expected in fixed.items():
        if type(record[key]) is not type(expected) or record[key] != expected:
            raise ValueError(f"successor manifest schema type/value mismatch: {key}")
    for key in ("target_commit", "wrapper_parent_sha",
                "predecessor_freeze_commit", "invalid_evidence_commit"):
        _require_string(record[key], f"successor manifest {key}", length=40)
    for key in ("nonce", "config_sha256", "preregistration_sha256"):
        _require_string(record[key], f"successor manifest {key}", length=64)
    _require_string(record["invocation_id"], "successor manifest invocation_id")
    _require_string_list(record["implementation_paths"],
                         "successor manifest implementation_paths")
    _require_string_list(record["internal_import_closure_paths"],
                         "successor manifest internal_import_closure_paths")
    _verify_identity_rows(record["files"], "successor manifest files")


def verify_control_protected_state(control: dict) -> None:
    record = _require_exact_keys(control, CONTROL_KEYS, "successor control")
    fixed_top = {
        "schema": CONTROL_SCHEMA,
        "state": "IMMUTABLY_PREREGISTERED_MANIFEST_REPAIR_ONLY",
        "commit_binding": "COMMIT_CONTAINING_THIS_DOCUMENT",
    }
    for key, expected in fixed_top.items():
        if type(record[key]) is not str or record[key] != expected:
            raise ValueError(f"successor control type/value mismatch: {key}")
    _verify_exact_record(record["invocation"], {
        "invocation_id": INVOCATION_ID, "nonce": INVOCATION_NONCE,
        "same_invocation_retry": False, "new_identity_required": True,
    }, "successor control invocation")
    ancestry = _require_exact_keys(record["ancestry"], {
        "approved_predecessor_freeze_commit", "failed_invocation_evidence_commit",
        "failed_invocation_archive", "failed_target_commit",
    }, "successor control ancestry")
    _verify_exact_record(ancestry, {
        "approved_predecessor_freeze_commit": PREDECESSOR_FREEZE_COMMIT,
        "failed_invocation_evidence_commit": INVALID_EVIDENCE_COMMIT,
        "failed_invocation_archive": "/tmp/gcspo-stage0-static-rerun-invalid-0ab9456/",
        "failed_target_commit": PREDECESSOR_FREEZE_COMMIT,
    }, "successor control ancestry")
    repair = _require_exact_keys(record["repair_scope"],
        {"only", "required_detection", "forbidden_changes"},
        "successor control repair scope")
    _require_string(repair["only"], "successor control repair scope only")
    _require_string_list(repair["required_detection"],
                         "successor control required detection")
    _require_string_list(repair["forbidden_changes"],
                         "successor control forbidden changes")
    _verify_exact_record(record["frozen_inputs"], {
        "config_sha256": CONFIG_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
    }, "successor control frozen inputs")
    preservation = _require_exact_keys(record["failed_invocation_preservation"], {
        "artifact_root", "exact_regular_file_set", "identities",
        "overwrite_forbidden",
    }, "successor control failed invocation preservation")
    if preservation["artifact_root"] != INVALID_ROOT_RELATIVE or type(
            preservation["artifact_root"]) is not str:
        raise ValueError("successor control preservation artifact root type/value mismatch")
    _require_string_list(preservation["exact_regular_file_set"],
                         "successor control preserved file set")
    _verify_identity_rows(preservation["identities"],
                          "successor control preserved identities")
    if preservation["overwrite_forbidden"] is not True:
        raise ValueError("successor control overwrite prohibition mismatch")
    _verify_exact_record(record["prior_protected_state"], {
        "protected_access_count": 0, "protected_rows_opened": 0,
        "protected_bytes_opened": 0,
        "marker_path": "artifacts/.gcspo_stage0_static_rerun.protected_run_started.json",
        "marker_present": False,
        "ledger_path": "artifacts/gcspo_stage0_static_rerun/access_ledger.jsonl",
        "ledger_present": True, "ledger_size_bytes": 0,
        "final_verdict_present": False,
    }, "successor control prior protected state")
    _verify_exact_record(record["successor_namespace"], {
        "artifact_root": ARTIFACT_RELATIVE,
        "marker_path": ("artifacts/gcspo_stage0_static_rerun_successor/."
                        f"{INVOCATION_ID}.protected_run_started.json"),
        "marker_present": False,
        "ledger_path": f"{ARTIFACT_RELATIVE}/access_ledger.jsonl",
        "ledger_size_bytes": 0, "protected_access_authorized": False,
    }, "successor control namespace protected state")
    topology = _require_exact_keys(record["commit_topology"], {
        "successor_target", "freeze_wrapper", "self_reference_rule"},
        "successor control commit topology")
    for key, value in topology.items():
        _require_string(value, f"successor control commit topology {key}")
    _verify_exact_record(record["prohibitions"], {
        "protected_evaluation": True, "protected_entrypoint": True,
        "private_signing_key_search": True, "push": True,
    }, "successor control prohibitions")


def _verify_test_result(value: object, label: str, *, execution_root: bool) -> None:
    keys = {"command", "passed", "failed", "exit_code"}
    if execution_root:
        keys.add("execution_root")
    record = _require_exact_keys(value, keys, label)
    _require_string(record["command"], f"{label} command")
    for key in ("passed", "failed", "exit_code"):
        _require_int(record[key], f"{label} {key}")
    if execution_root:
        _require_string(record["execution_root"], f"{label} execution_root")


def _verify_red_report(value: object) -> None:
    record = _require_exact_keys(value, {
        "schema", "phase", "baseline_wrapper_commit", "baseline_target_commit",
        "independent_review_verdict", "command", "exit_code", "passed",
        "failed", "reproduced", "protected_access_count",
        "protected_marker_present", "protected_ledger_size_bytes", "push_performed",
    }, "review handoff RED report")
    if record["schema"] not in {
            "gnss-doppler-lab.gcspo-stage0.successor-blocker-red.v2",
            "gnss-doppler-lab.gcspo-stage0.successor-blocker-red.v3"} or type(
                record["schema"]) is not str:
        raise ValueError("review handoff RED report schema mismatch")
    for key, expected in {"phase": "RED", "independent_review_verdict": "REJECT",
                          "protected_access_count": 0,
                          "protected_marker_present": False,
                          "protected_ledger_size_bytes": 0,
                          "push_performed": False}.items():
        if type(record[key]) is not type(expected) or record[key] != expected:
            raise ValueError(f"review handoff RED report type/value mismatch: {key}")
    for key in ("baseline_wrapper_commit", "baseline_target_commit"):
        _require_string(record[key], f"review handoff RED {key}", length=40)
    _require_string(record["command"], "review handoff RED command")
    for key in ("exit_code", "passed", "failed"):
        _require_int(record[key], f"review handoff RED {key}")
    reproduced = record["reproduced"]
    if record["schema"].endswith("v2"):
        r = _require_exact_keys(reproduced, {"deterministic_closure_set_missing",
            "post_target_internal_dependency_mutations_accepted",
            "protected_manifest_tamper_variants_accepted", "protected_tamper_classes"},
            "review handoff RED reproduced v2")
        _require_int(r["deterministic_closure_set_missing"], "RED closure count")
        _require_int(r["protected_manifest_tamper_variants_accepted"],
                     "RED protected tamper count")
        _require_string_list(r["post_target_internal_dependency_mutations_accepted"],
                             "RED post-target paths")
        _require_string_list(r["protected_tamper_classes"], "RED tamper classes")
    else:
        r = _require_exact_keys(reproduced, {"package_init_valid_rejected",
            "package_init_missing_accepted", "manifest_size_type_mutations_accepted",
            "handoff_mutations_accepted", "artifact_root_aliases_accepted",
            "total_adversarial_failures", "tamper_classes"},
            "review handoff RED reproduced v3")
        for key in ("package_init_valid_rejected", "package_init_missing_accepted",
                    "handoff_mutations_accepted", "total_adversarial_failures"):
            _require_int(r[key], f"RED reproduced {key}")
        for key in ("manifest_size_type_mutations_accepted",
                    "artifact_root_aliases_accepted", "tamper_classes"):
            _require_string_list(r[key], f"RED reproduced {key}")


def _verify_green_report(value: object) -> None:
    record = _require_exact_keys(value, {
        "schema", "phase", "target_commit_binding", "focused", "relevant",
        "all_gcspo", "adversarial_rejections", "target_manifest_validation",
        "repair_scope", "scientific_files_changed", "config_sha256",
        "preregistration_sha256", "protected_access_count",
        "protected_marker_present", "protected_ledger_size_bytes", "push_performed",
    }, "review handoff GREEN report")
    if record["schema"] not in {
            "gnss-doppler-lab.gcspo-stage0.successor-blocker-green.v2",
            "gnss-doppler-lab.gcspo-stage0.successor-blocker-green.v3"} or type(
                record["schema"]) is not str:
        raise ValueError("review handoff GREEN report schema mismatch")
    for key, expected in {"phase": "GREEN_TESTED_TARGET",
                          "target_commit_binding": "COMMIT_CONTAINING_THIS_DOCUMENT",
                          "scientific_files_changed": False,
                          "config_sha256": CONFIG_SHA256,
                          "preregistration_sha256": PREREGISTRATION_SHA256,
                          "protected_access_count": 0,
                          "protected_marker_present": False,
                          "protected_ledger_size_bytes": 0,
                          "push_performed": False}.items():
        if type(record[key]) is not type(expected) or record[key] != expected:
            raise ValueError(f"review handoff GREEN type/value mismatch: {key}")
    _require_string(record["repair_scope"], "review handoff GREEN repair scope")
    _verify_test_result(record["focused"], "review handoff focused", execution_root=False)
    _verify_test_result(record["relevant"], "review handoff relevant", execution_root=True)
    _verify_test_result(record["all_gcspo"], "review handoff all GCSPO", execution_root=True)
    adversarial = record["adversarial_rejections"]
    if record["schema"].endswith("v2"):
        a = _require_exact_keys(adversarial, {"post_target_internal_dependency_mutations",
            "manifest_protected_state_mutations", "control_protected_state_mutations",
            "handoff_protected_state_mutations", "tamper_classes", "status"},
            "review handoff GREEN adversarial v2")
        for key in ("post_target_internal_dependency_mutations",
                    "manifest_protected_state_mutations",
                    "control_protected_state_mutations",
                    "handoff_protected_state_mutations"):
            _require_int(a[key], f"GREEN adversarial {key}")
    else:
        a = _require_exact_keys(adversarial, {"python_import_cases",
            "manifest_size_type_mutations", "handoff_recursive_schema_mutations",
            "artifact_root_aliases", "total_second_review_regressions",
            "prior_protected_mutation_rejections", "tamper_classes", "status"}, "review handoff GREEN adversarial v3")
        for key in ("python_import_cases", "manifest_size_type_mutations",
                    "handoff_recursive_schema_mutations", "artifact_root_aliases",
                    "total_second_review_regressions",
                    "prior_protected_mutation_rejections"):
            _require_int(a[key], f"GREEN adversarial {key}")
    _require_string_list(a["tamper_classes"], "GREEN tamper classes")
    if a["status"] != "PASS_FAIL_CLOSED" or type(a["status"]) is not str:
        raise ValueError("GREEN adversarial status mismatch")
    validation = _require_exact_keys(record["target_manifest_validation"], {
        "implementation_rows", "internal_import_closure_paths",
        "required_direct_dependencies_present", "missing", "extra", "status"},
        "review handoff GREEN target manifest validation")
    for key in ("implementation_rows", "missing", "extra"):
        _require_int(validation[key], f"GREEN target validation {key}")
    _require_string_list(validation["internal_import_closure_paths"],
                         "GREEN closure paths")
    _require_string_list(validation["required_direct_dependencies_present"],
                         "GREEN direct dependencies")
    if validation["status"] != "PASS" or type(validation["status"]) is not str:
        raise ValueError("GREEN target validation status mismatch")


def _verify_rejection(value: object, label: str) -> None:
    record = _require_exact_keys(value, {"wrapper_commit", "target_commit", "verdict",
                                        "blocking_findings"}, label)
    _require_string(record["wrapper_commit"], f"{label} wrapper", length=40)
    _require_string(record["target_commit"], f"{label} target", length=40)
    if record["verdict"] != "REJECT" or type(record["verdict"]) is not str:
        raise ValueError(f"{label} verdict mismatch")
    _require_string_list(record["blocking_findings"], f"{label} findings")


def verify_handoff_protected_state(handoff: dict) -> None:
    record = _require_exact_keys(handoff, HANDOFF_KEYS, "review handoff")
    fixed = {
        "schema": HANDOFF_SCHEMA,
        "state": "READY_FOR_FRESH_INDEPENDENT_READ_ONLY_REVIEW",
        "wrapper_commit_binding": "COMMIT_CONTAINING_THIS_DOCUMENT",
        "same_invocation_retry": False,
        "config_sha256": CONFIG_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "invalid_artifact_root_preserved": True, "push_performed": False,
    }
    for key, expected in fixed.items():
        if type(record[key]) is not type(expected) or record[key] != expected:
            raise ValueError(f"review handoff schema type/value mismatch: {key}")
    for key in ("wrapper_parent_sha", "target_commit", "predecessor_freeze_commit",
                "invalid_evidence_commit"):
        _require_string(record[key], f"review handoff {key}", length=40)
    _require_string(record["invocation_id"], "review handoff invocation_id")
    _require_string(record["nonce"], "review handoff nonce", length=64)
    _require_string(record["repair_scope"], "review handoff repair_scope")
    _require_string(record["independent_review_command"],
                    "review handoff independent review command")
    red_green = _require_exact_keys(record["red_green"], {"red", "green"},
                                    "review handoff red/green")
    _verify_red_report(red_green["red"]); _verify_green_report(red_green["green"])
    coverage = _require_exact_keys(record["manifest_coverage"], {
        "total_rows", "stale_rows_required_and_present",
        "later_gcspo_rows_required_and_present",
        "required_direct_internal_dependencies", "internal_import_closure_paths",
        "missing_rows", "extra_rows"}, "review handoff manifest coverage")
    for key in ("total_rows", "missing_rows", "extra_rows"):
        _require_int(coverage[key], f"review handoff manifest coverage {key}")
    for key in ("stale_rows_required_and_present",
                "later_gcspo_rows_required_and_present",
                "required_direct_internal_dependencies",
                "internal_import_closure_paths"):
        _require_string_list(coverage[key], f"review handoff manifest coverage {key}")
    _verify_exact_record(record["protected"], {
        "access_count": 0, "marker_present": False,
        "ledger_size_bytes": 0, "authorized": False,
    }, "review handoff protected state")
    _verify_rejection(record["prior_independent_rejection"],
                      "review handoff prior independent rejection")
    if "latest_independent_rejection" in record:
        _verify_rejection(record["latest_independent_rejection"],
                          "review handoff latest independent rejection")
    if record["repair_scope"] != (
            "PYTHON_IMPORT_RESOLUTION_EXACT_DOCUMENT_SCHEMA_AND_CANONICAL_ARTIFACT_ROOT_ONLY"):
        raise ValueError("review handoff repair scope value mismatch")
    if (record["invocation_id"] != INVOCATION_ID or record["nonce"] != INVOCATION_NONCE or
            record["predecessor_freeze_commit"] != PREDECESSOR_FREEZE_COMMIT or
            record["invalid_evidence_commit"] != INVALID_EVIDENCE_COMMIT or
            record["independent_review_command"] != (
                "python3 scripts/verify_gcspo_successor_freeze.py "
                "--expected-wrapper-commit $(git rev-parse HEAD)")):
        raise ValueError("review handoff immutable value contract mismatch")
    expected_coverage = {
        "total_rows": 69,
        "stale_rows_required_and_present": list(STALE_CROSS_GENERATION_PATHS),
        "later_gcspo_rows_required_and_present": list(LATER_GCSPO_PATHS),
        "required_direct_internal_dependencies": list(REQUIRED_INTERNAL_DEPENDENCY_PATHS),
        "internal_import_closure_paths": [
            "src/gnss_doppler_lab/__init__.py",
            "src/gnss_doppler_lab/gcmr_geometry.py",
            "src/gnss_doppler_lab/trajectory.py",
        ],
        "missing_rows": 0, "extra_rows": 0,
    }
    if record["manifest_coverage"] != expected_coverage:
        raise ValueError("review handoff manifest coverage value contract mismatch")
    red = record["red_green"]["red"]
    expected_red_values = {
        "schema": "gnss-doppler-lab.gcspo-stage0.successor-blocker-red.v3",
        "phase": "RED", "baseline_wrapper_commit": SECOND_REJECTED_WRAPPER_COMMIT,
        "baseline_target_commit": SECOND_REJECTED_TARGET_COMMIT,
        "independent_review_verdict": "REJECT",
        "command": ("/home/ubuntu/.venvs/cmte-a2/bin/python -m pytest -q "
                    "tests/test_gcspo_successor_manifest.py"),
        "exit_code": 1, "passed": 48, "failed": 15,
        "reproduced": {
            "package_init_valid_rejected": 1, "package_init_missing_accepted": 1,
            "manifest_size_type_mutations_accepted": ["float", "bool"],
            "handoff_mutations_accepted": 7,
            "artifact_root_aliases_accepted": [
                "external_symlink", "traversal", "absolute", "dot_alias"],
            "total_adversarial_failures": 15,
            "tamper_classes": ["resolution", "bool-as-int", "float-as-int",
                "boolean-value-flip", "extra", "missing", "type", "path-alias"],
        },
        "protected_access_count": 0, "protected_marker_present": False,
        "protected_ledger_size_bytes": 0, "push_performed": False,
    }
    if red != expected_red_values:
        raise ValueError("review handoff RED exact value contract mismatch")
    green = record["red_green"]["green"]
    expected_commands = {
        "focused": ("/home/ubuntu/.venvs/cmte-a2/bin/python -m pytest -q "
                    "tests/test_gcspo_successor_manifest.py", 63, False),
        "relevant": ("/home/ubuntu/.venvs/cmte-a2/bin/python -m pytest -q "
                    "tests/test_gcspo_successor_manifest.py tests/test_gcspo_freeze.py "
                    "tests/test_gcspo_round4_freeze_repairs.py tests/test_gcspo_verifier.py "
                    "tests/test_gcspo_stage0.py", 141, True),
        "all_gcspo": ("/home/ubuntu/.venvs/cmte-a2/bin/python -m pytest -q "
                    "tests/test_gcspo*.py", 337, True),
    }
    for key, (command, passed, has_root) in expected_commands.items():
        expected_result = {"command": command, "passed": passed, "failed": 0, "exit_code": 0}
        if has_root:
            expected_result["execution_root"] = (
                "isolated detached worktree with immutable committed package fixture; "
                "authoritative invalid root untouched")
        if green[key] != expected_result:
            raise ValueError(f"review handoff GREEN {key} exact value contract mismatch")
    if green["adversarial_rejections"] != {
            "python_import_cases": 2, "manifest_size_type_mutations": 2,
            "handoff_recursive_schema_mutations": 7, "artifact_root_aliases": 4,
            "total_second_review_regressions": 15,
            "prior_protected_mutation_rejections": 275,
            "tamper_classes": ["resolution", "bool-as-int", "float-as-int",
                "boolean-value-flip", "extra", "missing", "type", "path-alias"],
            "status": "PASS_FAIL_CLOSED"} or green["target_manifest_validation"] != {
            "implementation_rows": 69,
            "internal_import_closure_paths": expected_coverage["internal_import_closure_paths"],
            "required_direct_dependencies_present": list(REQUIRED_INTERNAL_DEPENDENCY_PATHS),
            "missing": 0, "extra": 0, "status": "PASS"} or green["repair_scope"] != (
            "PYTHON_IMPORT_RESOLUTION_EXACT_DOCUMENT_SCHEMA_AND_CANONICAL_ARTIFACT_ROOT_ONLY"):
        raise ValueError("review handoff GREEN exact value contract mismatch")

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



def _canonical_artifact_directory(root: Path, artifact_root: str | Path) -> Path:
    if not isinstance(artifact_root, (str, Path)):
        raise ValueError("artifact root canonical path contract requires a path string")
    spelling = str(artifact_root)
    if spelling != ARTIFACT_RELATIVE:
        raise ValueError(
            f"artifact root canonical repo-relative spelling must be exactly {ARTIFACT_RELATIVE}")
    relative = Path(spelling)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("artifact root canonical path contains a forbidden alias component")
    expected = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("artifact root canonical path contains a symlink component")
    if not expected.is_dir() or expected.resolve(strict=True) != expected:
        raise ValueError("artifact root canonical directory is absent or aliased")
    return expected


def _require_regular_file(root: Path, path: Path, label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes repository root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} has a symlink component")
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file")

def verify_successor_freeze(repo: str | Path, artifact_root: str | Path,
                            expected_wrapper_commit: str) -> dict:
    root = Path(repo).resolve(strict=True)
    artifact = _canonical_artifact_directory(root, artifact_root)
    head = _run(root, "rev-parse", "HEAD").strip()
    if head != expected_wrapper_commit:
        raise ValueError("review HEAD does not equal expected wrapper commit")
    control_path = artifact / "successor_control.json"
    manifest_path = artifact / "implementation_manifest.json"
    handoff_path = artifact / "review_handoff.json"
    for path, label in ((control_path, "successor control"),
                        (manifest_path, "implementation manifest"),
                        (handoff_path, "review handoff")):
        _require_regular_file(root, path, label)
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
    if _tree_bytes(root, SECOND_REJECTED_TARGET_COMMIT, CONTROL_RELATIVE) != control_path.read_bytes():
        raise ValueError("immutable successor control bytes changed from rejected target")
    for key, name in (("red", "red_report.json"), ("green", "green_report.json")):
        target_report = strict_json_bytes(
            _tree_bytes(root, target, f"{ARTIFACT_RELATIVE}/{name}"), f"target {key} report")
        if handoff["red_green"][key] != target_report:
            raise ValueError(f"review handoff {key} report differs from target evidence")
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
    if handoff.get("latest_independent_rejection") != {
            "wrapper_commit": SECOND_REJECTED_WRAPPER_COMMIT,
            "target_commit": SECOND_REJECTED_TARGET_COMMIT,
            "verdict": "REJECT",
            "blocking_findings": [
                "PACKAGE_INIT_RELATIVE_IMPORT_RESOLUTION_FAIL_OPEN",
                "DOCUMENT_SCHEMA_AND_PRIMITIVE_TYPE_FAIL_OPEN",
                "ARTIFACT_ROOT_CANONICAL_PATH_FAIL_OPEN",
            ]}:
        raise ValueError("latest independent rejection evidence mismatch")
    if subprocess.run(["git", "merge-base", "--is-ancestor",
                       SECOND_REJECTED_WRAPPER_COMMIT, target],
                      cwd=root, capture_output=True).returncode:
        raise ValueError("latest rejected wrapper is not preserved in target ancestry")
    old = root / INVALID_ROOT_RELATIVE
    current = root
    for part in Path(INVALID_ROOT_RELATIVE).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("invalid artifact root contains a symlink component")
    preservation = control.get("failed_invocation_preservation", {})
    expected_names = preservation.get("exact_regular_file_set")
    old_entries = list(old.rglob("*"))
    if any(path.is_symlink() for path in old_entries):
        raise ValueError("invalid artifact root contains a symlink entry")
    observed_names = sorted(p.relative_to(old).as_posix() for p in old_entries if p.is_file())
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
    ledger = artifact / "access_ledger.jsonl"
    for path, label in ((config, "successor config"), (prereg, "successor preregistration"),
                        (ledger, "successor access ledger")):
        _require_regular_file(root, path, label)
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
