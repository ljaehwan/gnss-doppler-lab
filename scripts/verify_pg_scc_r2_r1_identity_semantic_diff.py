#!/usr/bin/env python3
"""Fail-closed semantic guard for the PG-SCC R2 r1 identity repair."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = "scripts/run_pg_scc_root_cause_audit.py"
PREREGISTRATION_SHA = "174610776c69f9ab2bc085be191cbdc563934625"
ALLOWED_CHANGED_FUNCTIONS = {
    "_write_followup_attempt_state": "cycle-local attempt provenance only",
    "verify_preregistration": "cycle branch/base provenance gate",
    "verify_implementation_freeze": "cycle branch/namespace provenance gate",
    "validate_committed_support_preflight": "cycle committed-preflight namespace gate",
}
ALLOWED_NEW_FUNCTIONS = {}
ALLOWED_CHANGED_ASSIGNMENTS = {
    "OUTPUT",
    "PREREGISTRATION_SHA",
    "REQUIRED_BASE_SHA",
    "PHASE2_ALLOWED_CHANGED_PATHS",
    "REQUIRED_ARTIFACTS",
    "IMPLEMENTATION_FILES",
}
IMMUTABLE_SCIENTIFIC_FILES = (
    "src/gnss_doppler_lab/pg_scc.py",
    "src/gnss_doppler_lab/pg_scc_physics.py",
    "src/gnss_doppler_lab/pg_scc_selector.py",
)
MANIFEST_FILES = (
    "scripts/run_pg_scc_root_cause_audit.py",
    "scripts/verify_pg_scc_root_cause_audit.py",
    "scripts/verify_pg_scc_r2_repair_followup.py",
    "scripts/verify_pg_scc_r2_semantic_diff.py",
    "scripts/verify_pg_scc_r2_r1_identity_repair.py",
    "scripts/verify_pg_scc_r2_r1_identity_semantic_diff.py",
    "tests/test_pg_scc_root_cause_audit.py",
    "tests/test_pg_scc_r2_preflight.py",
    "tests/test_pg_scc_r2_repair_followup.py",
    "tests/test_pg_scc_r2_r1_identity_repair.py",
    "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/config.json",
    "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/source_commit.json",
    "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/r1_failure_binding.json",
    "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/support_preflight.json",
    "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/preregistration.json",
    "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/predecessor_failure_binding.json",
    "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/zero_protected_access.json",
)


def _git(*args: str, binary: bool = False) -> str | bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=not binary)


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_path(path: Path) -> str:
    return _digest_bytes(path.read_bytes())


def _node_hash(node: ast.AST) -> str:
    return _digest_bytes(ast.dump(node, annotate_fields=True, include_attributes=False).encode())


def _functions(source: str) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _assignments(source: str) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            result[node.target.id] = node.value
    return result


def _normalized_main_hash(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    parsed = ast.parse(ast.unparse(node)).body[0]
    assert isinstance(parsed, (ast.FunctionDef, ast.AsyncFunctionDef))
    parsed.body = [
        statement for statement in parsed.body
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "_write_followup_attempt_state"
        )
    ]
    return _node_hash(parsed)


def _allowed_path(path: str) -> bool:
    return (
        path == RUNNER
        or path in {
            "scripts/verify_pg_scc_r2_r1_identity_repair.py",
            "scripts/verify_pg_scc_r2_r1_identity_semantic_diff.py",
            "tests/test_pg_scc_r2_repair_followup.py",
            "tests/test_pg_scc_r2_r1_identity_repair.py",
        }
        or path.startswith("artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle1/")
    )


def audit(base: str) -> dict[str, Any]:
    errors: list[str] = []
    base_source = _git("show", f"{base}:{RUNNER}")
    assert isinstance(base_source, str)
    current_source = (ROOT / RUNNER).read_text(encoding="utf-8")
    before_functions = _functions(base_source)
    after_functions = _functions(current_source)
    function_report: dict[str, Any] = {}
    for name in sorted(set(before_functions) & set(after_functions)):
        before_hash = _node_hash(before_functions[name])
        after_hash = _node_hash(after_functions[name])
        match = before_hash == after_hash
        function_report[name] = {
            "base_ast_sha256": before_hash,
            "current_ast_sha256": after_hash,
            "equivalent": match,
            "classification": ALLOWED_CHANGED_FUNCTIONS.get(name, "SCIENTIFIC_OR_UTILITY_UNCHANGED"),
        }
        if not match and name not in ALLOWED_CHANGED_FUNCTIONS:
            errors.append(f"unexpected_function_change:{name}")
    missing = sorted(set(before_functions) - set(after_functions))
    new = sorted(set(after_functions) - set(before_functions))
    errors.extend(f"missing_function:{name}" for name in missing)
    errors.extend(f"unexpected_new_function:{name}" for name in new if name not in ALLOWED_NEW_FUNCTIONS)

    before_assignments = _assignments(base_source)
    after_assignments = _assignments(current_source)
    assignment_report: dict[str, Any] = {}
    for name in sorted(set(before_assignments) & set(after_assignments)):
        before_hash = _node_hash(before_assignments[name])
        after_hash = _node_hash(after_assignments[name])
        match = before_hash == after_hash
        assignment_report[name] = {"equivalent": match}
        if not match and name not in ALLOWED_CHANGED_ASSIGNMENTS:
            errors.append(f"unexpected_assignment_change:{name}")
    errors.extend(
        f"unexpected_new_assignment:{name}"
        for name in sorted(set(after_assignments) - set(before_assignments))
        if name not in ALLOWED_CHANGED_ASSIGNMENTS
    )

    scientific_files: dict[str, Any] = {}
    for relative in IMMUTABLE_SCIENTIFIC_FILES:
        base_bytes = _git("show", f"{base}:{relative}", binary=True)
        assert isinstance(base_bytes, bytes)
        current_hash = _digest_path(ROOT / relative)
        base_hash = _digest_bytes(base_bytes)
        scientific_files[relative] = {
            "base_sha256": base_hash,
            "current_sha256": current_hash,
            "byte_equivalent": current_hash == base_hash,
        }
        if current_hash != base_hash:
            errors.append(f"scientific_file_change:{relative}")

    changed = set(str(_git("diff", "--name-only", base, "--")).splitlines())
    changed.update(str(_git("ls-files", "--others", "--exclude-standard")).splitlines())
    unexpected_paths = sorted(path for path in changed if path and not _allowed_path(path))
    errors.extend(f"unexpected_path:{path}" for path in unexpected_paths)
    return {
        "allowed_changed_functions": ALLOWED_CHANGED_FUNCTIONS,
        "allowed_new_functions": ALLOWED_NEW_FUNCTIONS,
        "assignment_ast_equivalence": assignment_report,
        "base_sha": base,
        "errors": sorted(set(errors)),
        "function_ast_equivalence": function_report,
        "new_functions": new,
        "non_plot_scientific_functions_equivalent": not any(
            error.startswith("unexpected_function_change:") for error in errors
        ),
        "preregistration_sha": PREREGISTRATION_SHA,
        "schema": "pg_scc_stage0_r2_r1_identity_repair_semantic_diff.v1",
        "scientific_file_byte_equivalence": scientific_files,
        "status": "PASS" if not errors else "FAIL",
        "unexpected_changed_paths": unexpected_paths,
    }


def write_manifest(output: Path, base: str) -> None:
    files = {relative: _digest_path(ROOT / relative) for relative in MANIFEST_FILES}
    payload = {
        "base_sha": base,
        "files": files,
        "implementation_freeze_rule": "commit containing these exact hashes",
        "preregistration_sha": PREREGISTRATION_SHA,
        "protected_score_fields_read_before_freeze": 0,
        "schema": "pg_scc_stage0_r2_r1_identity_repair_implementation_manifest.v1",
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.base)
    target = (ROOT / args.output).resolve() if not args.output.is_absolute() else args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(target.with_name("implementation_manifest_sha256.json"), args.base)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
