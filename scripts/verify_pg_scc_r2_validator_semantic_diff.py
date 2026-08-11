#!/usr/bin/env python3
"""Fail-closed semantic guard for the PG-SCC R2 validator sequencing repair."""
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
PREREGISTRATION_SHA = "c7887316ed981d0e7cde74b2bbadeb1cf83bb233"
ALLOWED_CHANGED_FUNCTIONS = {
    "_write_followup_attempt_state": "validator-repair attempt-state schema binding only",
    "_validate_required_outputs": "metadata-only two-phase manifest finalization validation",
    "main": "metadata-only manifest finalization ordering",
    "build_metadata_support_preflight_report": "validator-repair source provenance binding only",
    "verify_preregistration": "validator-repair branch/base provenance gate",
    "verify_implementation_freeze": "validator-repair branch/namespace provenance gate",
    "validate_committed_support_preflight": "stable exact plus bounded operational-path validation",
}
ALLOWED_NEW_FUNCTIONS: dict[str, str] = {
    "_load_committed_preregistration": "hash-bound committed preregistration loader",
    "_operational_additions": "committed operational path extraction",
    "_freeze_dirty_errors": "closed-world freeze dirty-path gate",
    "_implementation_manifest_errors": "closed-world implementation manifest gate",
}
ALLOWED_CHANGED_ASSIGNMENTS = {
    "OUTPUT",
    "SOURCE_SHA256",
    "PREREGISTRATION_SHA",
    "PREREGISTRATION_BLOB_SHA256",
    "REQUIRED_BASE_SHA",
    "SCIENTIFIC_IMPLEMENTATION_SHA",
    "PHASE2_ALLOWED_CHANGED_PATHS",
    "REQUIRED_ARTIFACTS",
    "IMPLEMENTATION_FILES",
    "IMPLEMENTATION_MANIFEST_FILES",
}
IMMUTABLE_SCIENTIFIC_FILES = (
    "src/gnss_doppler_lab/pg_scc.py",
    "src/gnss_doppler_lab/pg_scc_physics.py",
    "src/gnss_doppler_lab/pg_scc_selector.py",
)
PREDECESSOR_FAIL_CLOSED_SHA = "68ab54677f5d0b4b55cc39279aec631f60f655a9"
PRESERVED_PREDECESSOR_PATHS = {
    "artifacts/pg_scc_stage0_r2_repair_followup/attempt_state.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/fail_closed_delivery_manifest_sha256.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/fail_closed_delivery_report.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/fresh_clone_verifier_report.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/normal_verifier_report.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/pre_run_state.json",
    "artifacts/pg_scc_stage0_r2_repair_followup/protected_attempt_traceback.txt",
    "artifacts/pg_scc_stage0_r2_repair_followup/related_tests_report.txt",
    "artifacts/pg_scc_stage0_r2_repair_followup/semantic_verifier_report.json",
}
MANIFEST_FILES = (
    "scripts/run_pg_scc_root_cause_audit.py",
    "scripts/verify_pg_scc_root_cause_audit.py",
    "scripts/verify_pg_scc_r2_repair_followup.py",
    "scripts/verify_pg_scc_r2_semantic_diff.py",
    "scripts/verify_pg_scc_r2_validator_repair.py",
    "scripts/verify_pg_scc_r2_validator_semantic_diff.py",
    "tests/test_pg_scc_root_cause_audit.py",
    "tests/test_pg_scc_r2_preflight.py",
    "tests/test_pg_scc_r2_repair_followup.py",
    "tests/test_pg_scc_r2_validator_repair.py",
    "artifacts/pg_scc_stage0_r2_validator_repair/config.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/source_commit.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/r1_failure_binding.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/support_preflight.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/preregistration.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/predecessor_failure_binding.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/validator_root_cause.json",
    "artifacts/pg_scc_stage0_r2_validator_repair/test_report.txt",
    "artifacts/pg_scc_stage0_r2_validator_repair/semantic_diff_audit.json",
    "artifacts/pg_scc_stage0_r2_validator_finalization_followup/preregistration.json",
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
        or path in PRESERVED_PREDECESSOR_PATHS
        or path in {
            "scripts/verify_pg_scc_r2_validator_repair.py",
            "scripts/verify_pg_scc_r2_validator_semantic_diff.py",
            "tests/test_pg_scc_r2_repair_followup.py",
            "tests/test_pg_scc_r2_validator_repair.py",
        }
        or path.startswith("artifacts/pg_scc_stage0_r2_validator_repair/")
        or path.startswith("artifacts/pg_scc_stage0_r2_validator_finalization_followup/")
    )


def audit(base: str) -> dict[str, Any]:
    errors: list[str] = []
    if base != "9839823c00cafc34fbbf1d6b1dbe069eb2c4e74d":
        errors.append("scientific_baseline_not_implementation_9839823")
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
    if function_report.get("_save_plots", {}).get("equivalent") is not True:
        errors.append("plot_function_changed")
    required_plot_fragment = 'if row["group"] == group and "mean_improvement_per_k" in row'
    if required_plot_fragment not in current_source:
        errors.append("plot_schema_filter_not_exact")

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

    predecessor_file_byte_equivalence = {}
    for relative in sorted(PRESERVED_PREDECESSOR_PATHS):
        expected_bytes = _git("show", f"{PREDECESSOR_FAIL_CLOSED_SHA}:{relative}", binary=True)
        assert isinstance(expected_bytes, bytes)
        expected_hash = _digest_bytes(expected_bytes)
        current_hash = _digest_path(ROOT / relative)
        predecessor_file_byte_equivalence[relative] = {
            "base_sha256": expected_hash,
            "byte_equivalent": current_hash == expected_hash,
            "current_sha256": current_hash,
        }
        if current_hash != expected_hash:
            errors.append(f"predecessor_fail_closed_file_change:{relative}")

    preregistration = json.loads(
        (ROOT / "artifacts/pg_scc_stage0_r2_validator_repair/preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    registered = preregistration["scientific_identity_invariants"]
    registered_function_hashes = registered["immutable_function_ast_sha256"]
    function_hash_matches = {}
    for name, expected_hash in registered_function_hashes.items():
        actual_hash = function_report.get(name, {}).get("current_ast_sha256")
        function_hash_matches[name] = actual_hash == expected_hash
        if actual_hash != expected_hash:
            errors.append(f"preregistered_function_ast:{name}")
    registered_file_hashes = registered["frozen_file_sha256"]
    file_hash_matches = {}
    for relative, expected_hash in registered_file_hashes.items():
        actual_hash = _digest_path(ROOT / relative)
        file_hash_matches[relative] = actual_hash == expected_hash
        if actual_hash != expected_hash:
            errors.append(f"preregistered_file_hash:{relative}")

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
        "preregistered_scientific_hashes": {
            "file_sha256_match": file_hash_matches,
            "function_ast_sha256_match": function_hash_matches,
        },
        "non_plot_scientific_functions_equivalent": not any(
            error.startswith("unexpected_function_change:") for error in errors
        ),
        "predecessor_fail_closed_file_byte_equivalence": predecessor_file_byte_equivalence,
        "preregistration_sha": PREREGISTRATION_SHA,
        "schema": "pg_scc_stage0_r2_validator_repair_semantic_diff.v1",
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
        "schema": "pg_scc_stage0_r2_validator_repair_implementation_manifest.v1",
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
