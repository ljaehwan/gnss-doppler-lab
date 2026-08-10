from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CYCLE2 = ROOT / "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle2"
CYCLE1_ATTEMPT_SHA256 = "23c5a418ca63fe56b821cba1ebd16efbd2ed5a3ae992e37fc9496ba2f5d9f1fe"


def load_module(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runner():
    return load_module("scripts/run_pg_scc_root_cause_audit.py", "pg_scc_cycle2_runner_test")


def _call_name(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    return statement.value.func.id if isinstance(statement.value.func, ast.Name) else None


def test_cycle1_failure_reproduced_and_manifest_first_repairs_it(tmp_path):
    runner = load_runner()
    runner.OUTPUT = tmp_path
    for name in runner.REQUIRED_ARTIFACTS:
        if name in {"plots", "artifact_manifest_sha256.json", "root_cause_verdict.json"}:
            continue
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    plots = tmp_path / "plots"
    plots.mkdir()
    for name in runner.REQUIRED_PLOTS:
        (plots / name).write_bytes(b"synthetic plot fixture")
    runner.dump_json(tmp_path / "root_cause_verdict.json", {
        "exactly_one_verdict": True,
        "root_causes": {name: {} for name in runner.REQUIRED_ROOT_CAUSES},
        "verdict": "TERMINATE_PG_SCC",
    })

    with pytest.raises(RuntimeError, match=r"artifact_manifest_sha256\.json"):
        runner._validate_required_outputs()

    manifest = runner.finalize_manifest(tmp_path)
    runner._validate_required_outputs()
    assert (tmp_path / "artifact_manifest_sha256.json").is_file()
    assert "artifact_manifest_sha256.json" not in manifest


def test_main_contains_only_the_exact_adjacent_ordering_repair():
    source = (ROOT / "scripts/run_pg_scc_root_cause_audit.py").read_text(encoding="utf-8")
    main = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    names = [_call_name(statement) for statement in main.body]
    pairs = [
        index for index in range(len(names) - 1)
        if names[index:index + 2] == ["finalize_manifest", "_validate_required_outputs"]
    ]
    assert len(pairs) == 1


def test_cycle2_semantic_guard_normalizes_only_manifest_order():
    guard = load_module(
        "scripts/verify_pg_scc_r2_r1_identity_cycle2_semantic_diff.py",
        "pg_scc_cycle2_semantic_test",
    )
    report = guard.audit("2ddeccbfe934deeff527a165469fdafa1618ca33")
    assert report["status"] == "PASS"
    assert report["errors"] == []
    assert report["function_ast_equivalence"]["main"]["equivalent"] is True
    assert report["function_ast_equivalence"]["main"]["manifest_before_schema_validation"] is True
    assert report["scientific_file_byte_equivalence"]
    assert all(item["byte_equivalent"] for item in report["scientific_file_byte_equivalence"].values())


def test_cycle2_preflight_exactly_matches_metadata_only_reconstruction():
    runner = load_runner()
    committed = json.loads((CYCLE2 / "support_preflight.json").read_text(encoding="utf-8"))
    reconstructed = runner.build_metadata_support_preflight_report()
    assert committed == reconstructed
    assert committed["status"] == "PASS"
    assert committed["r1_identity"]["status"] == "PASS"
    assert committed["r1_identity"]["disallowed_changed_paths"] == []
    assert committed["protected_score_fields_projected_or_read"] == 0
    runner.validate_committed_support_preflight(committed)


def test_cycle2_binding_proves_failed_outputs_were_not_inspected_or_reused():
    predecessor = json.loads((CYCLE2 / "predecessor_failure_binding.json").read_text())
    zero = json.loads((CYCLE2 / "zero_protected_access.json").read_text())
    assert predecessor["authorized_failure_metadata_only"]["attempt_state_sha256"] == CYCLE1_ATTEMPT_SHA256
    assert predecessor["immutability"]["partial_scientific_outputs_inspected"] is False
    assert predecessor["immutability"]["partial_scientific_outputs_reused"] is False
    assert zero["partial_cycle1_scientific_outputs_inspected"] is False
    assert zero["partial_cycle1_scientific_outputs_reused"] is False
    assert zero["protected_score_fields_projected_or_read_for_cycle2_diagnosis"] == 0
