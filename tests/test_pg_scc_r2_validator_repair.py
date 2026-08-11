from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PRE_RUN_PATH = "artifacts/pg_scc_stage0_r2_validator_repair/pre_run_state.json"
ATTEMPT_PATH = "artifacts/pg_scc_stage0_r2_validator_repair/attempt_state.json"


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "pg_scc_root_cause_audit_r2_validator_repair",
        ROOT / "scripts/run_pg_scc_root_cause_audit.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_validator_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_pg_scc_r2_validator_repair_tests",
        ROOT / "scripts/verify_pg_scc_r2_validator_repair.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_preflight() -> dict[str, object]:
    formula_identity = {
        "status": "PASS",
        "function_ast_sha256": {"metric_bundle": {"match": True}},
        "frozen_artifact_sha256": {"masks.json": {"match": True}},
        "frozen_source_sha256": {"src/gnss_doppler_lab/pg_scc.py": {"match": True}},
    }
    return {
        "schema": "pg_scc_stage0_r2_support_preflight.v1",
        "status": "PASS",
        "eligible_event_count": 260,
        "excluded_event_count": 4,
        "eligible_pooled_event_identities": [
            {"scenario": "cleanStatic", "phase": "holdout", "second": 1}
        ],
        "excluded_pooled_event_identities": [
            {"scenario": "cleanStatic", "phase": "holdout", "second": 2}
        ],
        "pooled_event_unique_prn_counts": [
            {
                "eligible": True,
                "identity": {"scenario": "cleanStatic", "phase": "holdout", "second": 1},
                "prn_count": 4,
                "prns": [3, 7, 11, 19],
            }
        ],
        "per_detector_raw_row_support_hashes": {"pg_scc_k9": "raw-support-hash"},
        "per_detector_eligible_event_support_hashes": {"pg_scc_k9": "event-support-hash"},
        "protected_score_fields_read": 0,
        "protected_score_fields_projected_or_read": 0,
        "frozen_formula_identity": formula_identity,
        "r1_identity": {
            "status": "PASS",
            "binding_hashes_verified": 8,
            "preserved_file_sha256": {"binding": "frozen"},
            "current_preserved_file_sha256": {"binding": "frozen"},
            "disallowed_paths_unchanged": True,
            "disallowed_changed_paths": [],
            "phase2_changed_paths": [
                "artifacts/pg_scc_stage0_r2_validator_repair/preregistration.json"
            ],
            "unchanged_code_identity": formula_identity,
        },
    }


def reconstruct_with_paths(report: dict[str, object], *paths: str) -> dict[str, object]:
    reconstructed = copy.deepcopy(report)
    identity = reconstructed["r1_identity"]
    assert isinstance(identity, dict)
    identity["phase2_changed_paths"] = sorted(
        set(identity["phase2_changed_paths"]) | set(paths)
    )
    return reconstructed


def install_reconstruction(monkeypatch, runner, reconstructed: dict[str, object]) -> None:
    monkeypatch.setattr(runner, "build_metadata_support_preflight_report", lambda: reconstructed)


def write_preflight(tmp_path: Path, report: dict[str, object]) -> Path:
    path = tmp_path / "support_preflight.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_validator_gate_1_committed_preflight_only_passes(monkeypatch):
    runner = load_runner()
    committed = synthetic_preflight()
    install_reconstruction(monkeypatch, runner, copy.deepcopy(committed))
    assert runner.validate_committed_support_preflight(committed) == committed


def test_validator_gate_2_pre_run_operational_addition_passes(monkeypatch):
    runner = load_runner()
    committed = synthetic_preflight()
    reconstructed = reconstruct_with_paths(committed, PRE_RUN_PATH)
    install_reconstruction(monkeypatch, runner, reconstructed)
    assert runner.validate_committed_support_preflight(committed) == reconstructed


def test_validator_gate_3_attempt_state_operational_addition_preserves_stable_pass(monkeypatch):
    runner = load_runner()
    committed = synthetic_preflight()
    reconstructed = reconstruct_with_paths(committed, PRE_RUN_PATH, ATTEMPT_PATH)
    install_reconstruction(monkeypatch, runner, reconstructed)
    validated = runner.validate_committed_support_preflight(committed)
    assert validated["status"] == "PASS"
    assert validated["r1_identity"]["disallowed_changed_paths"] == []
    assert validated["r1_identity"]["disallowed_paths_unchanged"] is True


def test_validator_gate_4_arbitrary_disallowed_source_addition_fails(monkeypatch):
    runner = load_runner()
    committed = synthetic_preflight()
    reconstructed = reconstruct_with_paths(
        committed, "src/gnss_doppler_lab/unpreregistered_scientific_change.py"
    )
    install_reconstruction(monkeypatch, runner, reconstructed)
    with pytest.raises(RuntimeError, match="FAIL_CLOSED_SUPPORT_PREFLIGHT"):
        runner.validate_committed_support_preflight(committed)


def test_validator_gate_5_support_event_count_change_fails(monkeypatch):
    runner = load_runner()
    committed = synthetic_preflight()
    reconstructed = copy.deepcopy(committed)
    reconstructed["eligible_event_count"] = 261
    install_reconstruction(monkeypatch, runner, reconstructed)
    with pytest.raises(RuntimeError, match="FAIL_CLOSED_SUPPORT_PREFLIGHT"):
        runner.validate_committed_support_preflight(committed)


def test_validator_raw_support_hash_change_fails(monkeypatch):
    runner = load_runner()
    committed = synthetic_preflight()
    reconstructed = copy.deepcopy(committed)
    reconstructed["per_detector_raw_row_support_hashes"]["pg_scc_k9"] = "changed"
    install_reconstruction(monkeypatch, runner, reconstructed)
    with pytest.raises(RuntimeError, match="FAIL_CLOSED_SUPPORT_PREFLIGHT"):
        runner.validate_committed_support_preflight(committed)


def test_validator_gate_6_protected_score_read_count_change_fails(monkeypatch):
    runner = load_runner()
    committed = synthetic_preflight()
    reconstructed = copy.deepcopy(committed)
    reconstructed["protected_score_fields_projected_or_read"] = 1
    install_reconstruction(monkeypatch, runner, reconstructed)
    with pytest.raises(RuntimeError, match="FAIL_CLOSED_SUPPORT_PREFLIGHT"):
        runner.validate_committed_support_preflight(committed)


def test_validator_gate_7_loader_remains_closed_until_validator_passes(monkeypatch, tmp_path):
    runner = load_runner()
    committed = synthetic_preflight()
    reconstructed = reconstruct_with_paths(committed, "src/unpreregistered_change.py")
    install_reconstruction(monkeypatch, runner, reconstructed)
    calls: list[str] = []

    def protected_loader():
        calls.append("protected")
        return {"loaded": True}

    with pytest.raises(RuntimeError, match="FAIL_CLOSED_SUPPORT_PREFLIGHT"):
        runner.load_protected_inputs_after_preflight(
            preflight_path=write_preflight(tmp_path, committed),
            protected_loader=protected_loader,
        )
    assert calls == []


def test_validator_gate_8_loader_called_exactly_once_after_validator_passes(
    monkeypatch, tmp_path
):
    runner = load_runner()
    committed = synthetic_preflight()
    reconstructed = reconstruct_with_paths(committed, PRE_RUN_PATH)
    install_reconstruction(monkeypatch, runner, reconstructed)
    calls: list[str] = []

    def protected_loader():
        calls.append("protected")
        return {"loaded": True}

    loaded = runner.load_protected_inputs_after_preflight(
        preflight_path=write_preflight(tmp_path, committed),
        protected_loader=protected_loader,
    )
    assert loaded == {"loaded": True}
    assert calls == ["protected"]


def test_local_preregistration_tamper_cannot_expand_operational_allowlist(
    monkeypatch, tmp_path
):
    runner = load_runner()
    committed = synthetic_preflight()
    forbidden = "src/gnss_doppler_lab/tampered_allowlist_expansion.py"
    reconstructed = reconstruct_with_paths(committed, forbidden)
    install_reconstruction(monkeypatch, runner, reconstructed)

    tampered = json.loads(
        (ROOT / "artifacts/pg_scc_stage0_r2_validator_repair/preregistration.json").read_text()
    )
    allowlist = tampered["dynamic_operational_path_contract"][
        "allowed_only_when_added_after_committed_preflight_and_subset_of_this_set"
    ]
    allowlist.append(forbidden)
    allowlist.sort()
    (tmp_path / "preregistration.json").write_text(json.dumps(tampered), encoding="utf-8")
    runner.OUTPUT = tmp_path

    with pytest.raises(RuntimeError, match="dynamic operational path mismatch"):
        runner.validate_committed_support_preflight(committed)


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("schema", "tampered", "implementation_manifest_schema"),
        ("base_sha", "0" * 40, "implementation_manifest_base"),
        ("preregistration_sha", "0" * 40, "implementation_manifest_preregistration"),
    ],
)
def test_manifest_identity_tamper_fails(monkeypatch, field, value, expected_error):
    verifier = load_validator_verifier()
    root = ROOT / "artifacts/pg_scc_stage0_r2_validator_repair"
    manifest = json.loads((root / "implementation_manifest_sha256.json").read_text())
    manifest[field] = value
    original_load = verifier._load
    monkeypatch.setattr(
        verifier,
        "_load",
        lambda path: (
            manifest
            if path.name == "implementation_manifest_sha256.json"
            else original_load(path)
        ),
    )
    assert expected_error in verifier._manifest_errors(root)


def test_manifest_key_deletion_fails(monkeypatch):
    verifier = load_validator_verifier()
    root = ROOT / "artifacts/pg_scc_stage0_r2_validator_repair"
    manifest = json.loads((root / "implementation_manifest_sha256.json").read_text())
    manifest["files"].pop(
        "artifacts/pg_scc_stage0_r2_validator_repair/preregistration.json"
    )
    original_load = verifier._load
    monkeypatch.setattr(
        verifier,
        "_load",
        lambda path: (
            manifest
            if path.name == "implementation_manifest_sha256.json"
            else original_load(path)
        ),
    )
    assert "implementation_manifest_key_set" in verifier._manifest_errors(root)


@pytest.mark.parametrize(
    "relative",
    [
        "artifacts/pg_scc_stage0_r2_validator_repair/preregistration.json",
        "artifacts/pg_scc_stage0_r2_validator_repair/implementation_manifest_sha256.json",
        "artifacts/pg_scc_stage0_r2_validator_repair/semantic_diff_audit.json",
        "artifacts/pg_scc_stage0_r2_validator_repair/r1_failure_binding.json",
    ],
)
def test_runner_rejects_modified_tracked_freeze_metadata(relative):
    runner = load_runner()
    assert runner._freeze_dirty_errors(f" M {relative}") == [f" M {relative}"]


def test_runner_rejects_unregistered_dirty_namespace_file():
    runner = load_runner()
    relative = "artifacts/pg_scc_stage0_r2_validator_repair/unregistered.tmp"
    assert runner._freeze_dirty_errors(f"?? {relative}") == [f"?? {relative}"]


def test_runner_allows_only_untracked_preregistered_pre_run_state():
    runner = load_runner()
    assert runner._freeze_dirty_errors(f"?? {PRE_RUN_PATH}") == []


@pytest.mark.parametrize(
    "status_line",
    [
        " M artifacts/pg_scc_stage0_r2_validator_repair/preregistration.json",
        " M artifacts/pg_scc_stage0_r2_validator_repair/implementation_manifest_sha256.json",
        " M artifacts/pg_scc_stage0_r2_validator_repair/semantic_diff_audit.json",
        " M artifacts/pg_scc_stage0_r2_validator_repair/r1_failure_binding.json",
        "?? artifacts/pg_scc_stage0_r2_validator_repair/unregistered.tmp",
    ],
)
def test_independent_verifier_rejects_freeze_dirty_tamper(status_line):
    verifier = load_validator_verifier()
    assert verifier._preexecution_dirty_errors(status_line) == [status_line]


def test_independent_verifier_allows_untracked_preregistered_pre_run_state():
    verifier = load_validator_verifier()
    assert verifier._preexecution_dirty_errors(f"?? {PRE_RUN_PATH}") == []
