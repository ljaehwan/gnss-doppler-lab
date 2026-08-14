from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from gnss_doppler_lab.gcspo_freeze import verify_freeze_record
from gnss_doppler_lab.gcspo_successor_launch import (
    ARTIFACT_RELATIVE,
    AUTHORIZATION_RELATIVE,
    BRANCH,
    CLEAN_BUNDLE_RELATIVE_PATHS,
    CONTROL_RELATIVE,
    DYNAMIC_RUNTIME_PATHS,
    HISTORICAL_TARGET_COMMIT,
    HISTORICAL_WRAPPER_COMMIT,
    INVOCATION_ID,
    INVOCATION_NONCE,
    RECEIPT_RELATIVE,
    TARGET_DIFF_ALLOWLIST,
    authorize_launch,
    build_review_authorization_documents,
    identity,
    prepare_successor_valid_artifact_manifest,
    runtime_rows_at_commit,
    strict_json_bytes,
    verify_destination_bundle,
    verify_historical_pair,
    verify_launch_control,
    verify_sync_snapshot,
    verify_wrapper_documents,
    verify_zero_protected_state,
)

ROOT = Path(__file__).parents[1]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                          capture_output=True).stdout.strip()


def _control() -> dict:
    return strict_json_bytes((ROOT / CONTROL_RELATIVE).read_bytes(), "launch control")


def _approval_documents(target: str = "a" * 40):
    return build_review_authorization_documents(
        _control(), target_commit=target, reviewer="independent-reviewer",
        review_command="python -m pytest -q tests/test_gcspo_successor_launch.py",
        passed=31, findings=[], evidence_sha256="b" * 64,
        runtime_rows=[{"path": "scripts/frozen.py", "sha256": "c" * 64,
                       "size_bytes": 1}],
    )


def test_old_successor_v1_cannot_pass_legacy_v2_verifier():
    with pytest.raises(ValueError, match="schema"):
        verify_freeze_record({
            "schema": "gnss-doppler-lab.gcspo-stage0.successor-implementation-freeze.v1",
            "validity_state": "VALID_FOR_PROTECTED_ACCESS",
            "target_commit": "a" * 40,
        }, target_commit="a" * 40)


def test_historical_f8_13945_pair_and_exact_69_rows_are_verified_from_git_objects():
    report = verify_historical_pair(ROOT)
    assert report == {
        "wrapper_commit": HISTORICAL_WRAPPER_COMMIT,
        "target_commit": HISTORICAL_TARGET_COMMIT,
        "manifest_rows": 69,
        "protected_access_authorized": False,
    }


def test_historical_unauthorized_successor_cannot_launch():
    report = verify_historical_pair(ROOT)
    assert report["protected_access_authorized"] is False
    with pytest.raises(PermissionError, match="independent|authorization|wrapper"):
        authorize_launch(
            _control(), None, None, target_commit=HISTORICAL_WRAPPER_COMMIT,
            wrapper_commit=HISTORICAL_WRAPPER_COMMIT,
            wrapper_parent=HISTORICAL_TARGET_COMMIT, wrapper_changed_paths=[],
            local_sha=HISTORICAL_WRAPPER_COMMIT, remote_sha=HISTORICAL_WRAPPER_COMMIT,
            branch=BRANCH, clean=True, marker_present=False, ledger_size=0,
            verdict_present=False,
        )


def test_target_control_is_exact_and_awaiting_review_not_authorized():
    control = _control()
    assert verify_launch_control(control)
    assert control["state"] == "AWAITING_INDEPENDENT_REVIEW"
    assert control["authorization"] == {
        "authorized": False,
        "execution_freeze_present": False,
        "independent_review_receipt_present": False,
    }
    assert control["protected_state"] == {
        "access_count": 0, "authorized": False, "final_verdict_present": False,
        "ledger_size_bytes": 0, "marker_present": False,
    }


def test_control_binds_frozen_history_science_and_fresh_identity():
    control = _control()
    assert control["historical_successor"] == {
        "implementation_target_commit": HISTORICAL_TARGET_COMMIT,
        "wrapper_commit": HISTORICAL_WRAPPER_COMMIT,
        "manifest_rows": 69,
        "structurally_approved": True,
        "protected_access_authorized": False,
    }
    assert control["invocation"] == {
        "id": INVOCATION_ID, "nonce": INVOCATION_NONCE, "same_invocation_retry": False,
    }
    assert control["predecessor"]["freeze_commit"] == "0ab94567938234ca925f0bb8fbaece41e7d5e4a3"
    assert control["predecessor"]["invalid_evidence_commit"] == "58943724ea278b754d388ec2dca0f3666ed6c8a2"
    assert control["scientific_inputs"]["config_sha256"] == "0db816116b95b41db8b7af7379cd7411cc52d43b6428ae00ab02d6ccac19f4ad"
    assert control["scientific_inputs"]["preregistration_sha256"] == "715e11965854f785487e9d2c747718747c1d31cdd8603696ea7af126a45a70da"


def test_control_contains_exact_target_diff_allowlist_and_bundle_destinations():
    control = _control()
    assert control["target_diff_allowlist"] == list(TARGET_DIFF_ALLOWLIST)
    assert [row["destination"] for row in control["clean_bundle"]["files"]] == [
        f"{ARTIFACT_RELATIVE}/{relative}" for relative in CLEAN_BUNDLE_RELATIVE_PATHS
    ]


def test_builder_returns_receipt_and_execution_freeze_but_does_not_write_them():
    receipt_path = ROOT / RECEIPT_RELATIVE
    authorization_path = ROOT / AUTHORIZATION_RELATIVE
    assert not receipt_path.exists() and not authorization_path.exists()
    receipt, execution = _approval_documents()
    assert receipt["state"] == "APPROVED"
    assert execution["state"] == "VALID_FOR_PROTECTED_ACCESS"
    assert not receipt_path.exists() and not authorization_path.exists()


def test_receipt_has_exact_types_keys_and_target_invocation_binding():
    target = "c" * 40
    receipt, _ = _approval_documents(target)
    assert set(receipt) == {
        "schema", "state", "reviewer", "target_commit", "invocation_id", "nonce",
        "evidence",
    }
    assert all(type(receipt[key]) is str for key in (
        "schema", "state", "reviewer", "target_commit", "invocation_id", "nonce"))
    assert receipt["target_commit"] == target
    assert receipt["invocation_id"] == INVOCATION_ID
    assert receipt["nonce"] == INVOCATION_NONCE
    assert set(receipt["evidence"]) == {"command", "passed", "findings", "sha256"}
    assert type(receipt["evidence"]["passed"]) is int
    assert type(receipt["evidence"]["findings"]) is list


@pytest.mark.parametrize("mutation", [
    "receipt_extra", "receipt_bool_count", "wrong_target", "wrong_invocation",
    "wrong_nonce", "not_approved", "execution_extra", "wrong_receipt_path",
])
def test_receipt_execution_exact_schema_and_bindings_fail_closed(mutation):
    target = "d" * 40
    receipt, execution = _approval_documents(target)
    if mutation == "receipt_extra": receipt["extra"] = False
    elif mutation == "receipt_bool_count": receipt["evidence"]["passed"] = True
    elif mutation == "wrong_target": receipt["target_commit"] = "e" * 40
    elif mutation == "wrong_invocation": receipt["invocation_id"] = "other"
    elif mutation == "wrong_nonce": receipt["nonce"] = "f" * 64
    elif mutation == "not_approved": receipt["state"] = "REJECTED"
    elif mutation == "execution_extra": execution["extra"] = False
    elif mutation == "wrong_receipt_path": execution["receipt_path"] = "elsewhere.json"
    else: raise AssertionError(mutation)
    with pytest.raises((ValueError, PermissionError), match="receipt|execution|review|binding|schema|type|target|invocation|nonce"):
        verify_wrapper_documents(_control(), receipt, execution, target_commit=target)


def test_authorization_requires_immediate_child_and_receipt_only_wrapper():
    target, wrapper = "1" * 40, "2" * 40
    receipt, execution = _approval_documents(target)
    common = dict(
        target_commit=target, wrapper_commit=wrapper, wrapper_parent=target,
        wrapper_changed_paths=[RECEIPT_RELATIVE, AUTHORIZATION_RELATIVE],
        local_sha=wrapper, remote_sha=wrapper, branch=BRANCH, clean=True,
        marker_present=False, ledger_size=0, verdict_present=False,
    )
    result = authorize_launch(_control(), receipt, execution, **common)
    assert result == {"authorized": True, "freeze_sha": wrapper, "run_identity": wrapper,
                      "target_commit": target, "invocation_id": INVOCATION_ID}
    with pytest.raises(PermissionError, match="immediate parent"):
        authorize_launch(_control(), receipt, execution, **{**common, "wrapper_parent": "3" * 40})
    with pytest.raises(PermissionError, match="receipt|execution|wrapper"):
        authorize_launch(_control(), receipt, execution,
                         **{**common, "wrapper_changed_paths": [*common["wrapper_changed_paths"], "extra.py"]})


@pytest.mark.parametrize("field,value", [
    ("local_sha", "3" * 40), ("remote_sha", "3" * 40), ("branch", "wrong"),
    ("clean", False), ("marker_present", True), ("ledger_size", 1),
    ("verdict_present", True),
])
def test_authorization_requires_exact_sync_clean_and_zero_state(field, value):
    target, wrapper = "1" * 40, "2" * 40
    receipt, execution = _approval_documents(target)
    args = dict(target_commit=target, wrapper_commit=wrapper, wrapper_parent=target,
                wrapper_changed_paths=[RECEIPT_RELATIVE, AUTHORIZATION_RELATIVE],
                local_sha=wrapper, remote_sha=wrapper, branch=BRANCH, clean=True,
                marker_present=False, ledger_size=0, verdict_present=False)
    args[field] = value
    with pytest.raises(PermissionError, match="sync|branch|clean|marker|ledger|verdict|authorization"):
        authorize_launch(_control(), receipt, execution, **args)


def test_clean_bundle_rows_are_exact_destination_hash_size_and_source_commit():
    control = _control()
    rows = control["clean_bundle"]["files"]
    assert len(rows) == len(CLEAN_BUNDLE_RELATIVE_PATHS) == 21
    for row in rows:
        assert set(row) == {
            "destination", "predecessor_manifest_authenticated", "sha256", "size_bytes",
            "source_commit", "source_path",
        }
        assert row["source_commit"] == "0ab94567938234ca925f0bb8fbaece41e7d5e4a3"
        payload = (ROOT / row["destination"]).read_bytes()
        assert row["sha256"] == hashlib.sha256(payload).hexdigest()
        assert row["size_bytes"] == len(payload)


def _tiny_bundle(tmp_path: Path):
    root = tmp_path / "bundle"; root.mkdir()
    (root / "a.json").write_bytes(b"a\n")
    (root / "nested").mkdir(); (root / "nested/b.json").write_bytes(b"b\n")
    rows = [
        {**identity("a.json", b"a\n"), "destination": "a.json"},
        {**identity("nested/b.json", b"b\n"), "destination": "nested/b.json"},
    ]
    for row in rows: row.pop("path")
    return root, rows


@pytest.mark.parametrize("tamper", ["symlink", "missing", "extra", "hash", "size", "bytes"])
def test_bundle_symlink_missing_extra_hash_size_and_tamper_rejected(tmp_path, tamper):
    root, rows = _tiny_bundle(tmp_path)
    if tamper == "symlink":
        (root / "a.json").unlink(); (root / "a.json").symlink_to(root / "nested/b.json")
    elif tamper == "missing": (root / "a.json").unlink()
    elif tamper == "extra": (root / "extra.json").write_text("{}\n")
    elif tamper == "hash": rows[0]["sha256"] = "0" * 64
    elif tamper == "size": rows[0]["size_bytes"] += 1
    elif tamper == "bytes": (root / "a.json").write_bytes(b"x\n")
    with pytest.raises(ValueError, match="symlink|missing|extra|identity|hash|size|regular|bundle"):
        verify_destination_bundle(root, rows, expected_paths=["a.json", "nested/b.json"])


def test_dynamic_b0_runtime_closure_contains_required_scripts_and_assets():
    assert DYNAMIC_RUNTIME_PATHS == (
        "artifacts/ai_morph_gru_cleanStatic_q70_frame/prn_local_gru_predictor.pt",
        "artifacts/ai_morph_gru_cleanStatic_q70_frame/validation_prn_node_scores.csv",
        "scripts/eval_btail_support_gate.py",
        "scripts/score_texbat_prn_node_gru.py",
        "scripts/train_prn_node_gru.py",
    )
    rows = runtime_rows_at_commit(ROOT, HISTORICAL_WRAPPER_COMMIT)
    assert set(DYNAMIC_RUNTIME_PATHS).issubset({row["path"] for row in rows})
    assert all(set(row) == {"path", "sha256", "size_bytes"} for row in rows)


def test_sync_snapshot_requires_fixed_branch_exact_sha_clean_zero_divergence():
    sha = "a" * 40
    assert verify_sync_snapshot({
        "branch": BRANCH, "local_sha": sha, "remote_sha": sha,
        "ahead": 0, "behind": 0, "clean": True,
    }, expected_sha=sha)
    for key, value in (("branch", "main"), ("remote_sha", "b" * 40),
                       ("ahead", 1), ("behind", 1), ("clean", False)):
        snapshot = {"branch": BRANCH, "local_sha": sha, "remote_sha": sha,
                    "ahead": 0, "behind": 0, "clean": True}
        snapshot[key] = value
        with pytest.raises(PermissionError, match="sync|branch|clean"):
            verify_sync_snapshot(snapshot, expected_sha=sha)


def test_zero_markers_ledgers_and_verdicts_are_required(tmp_path):
    paths = {
        "markers": ["a/.marker", "b/.marker"],
        "ledgers": ["a/ledger", "b/ledger"],
        "verdicts": ["a/final_verdict.json", "b/final_verdict.json"],
    }
    for relative in paths["ledgers"]:
        path = tmp_path / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"")
    assert verify_zero_protected_state(tmp_path, **paths)
    for kind, relative, payload in (
        ("markers", paths["markers"][0], b"{}"),
        ("ledgers", paths["ledgers"][0], b"x"),
        ("verdicts", paths["verdicts"][0], b"{}"),
    ):
        case = tmp_path / kind; case.mkdir()
        for ledger in paths["ledgers"]:
            path = case / ledger; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"")
        path = case / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
        with pytest.raises(PermissionError, match="marker|ledger|verdict"):
            verify_zero_protected_state(case, **paths)


def test_successor_runner_has_fixed_paths_no_caller_artifact_or_config_and_no_legacy_verifier():
    source = (ROOT / "scripts/run_gcspo_stage0_successor.py").read_text()
    assert "--artifact-dir" not in source and "--config" not in source
    assert "verify_freeze_record" not in source
    assert "write_fail_closed_invalid" not in source
    assert "ARTIFACT_RELATIVE" in source and "CONTROL_RELATIVE" in source


def test_all_preclaim_failure_occurs_before_claim_and_creates_no_invalid_artifact(monkeypatch):
    path = ROOT / "scripts/run_gcspo_stage0_successor.py"
    spec = importlib.util.spec_from_file_location("successor_runner", path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(module)
    called = []
    monkeypatch.setattr(module, "preflight", lambda: (_ for _ in ()).throw(PermissionError("sealed")))
    monkeypatch.setattr(module, "claim", lambda *_: called.append("claimed"))
    assert module.main() == 2
    assert called == []


def test_successor_packaging_preserves_additions_and_manifests_them(tmp_path):
    from gnss_doppler_lab.gcspo_artifacts import (
        MANIFEST_EXCLUSIONS, VALID_SCIENCE_REQUIRED,
    )

    for relative in VALID_SCIENCE_REQUIRED - MANIFEST_EXCLUSIONS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture\n")
    plot = tmp_path / "plots/score.csv"
    plot.parent.mkdir(parents=True, exist_ok=True)
    plot.write_bytes(b"score\n")

    additions = {
        "launch_provenance.json": b"launch\n",
        "protected_run_provenance.json": b"protected\n",
        "protected_control_status.json": b"controls\n",
        "reproductions/run-1/report.json": b"reproduction\n",
    }
    for relative, payload in additions.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    manifest = prepare_successor_valid_artifact_manifest(tmp_path)

    manifested = {row["path"] for row in manifest["files"]}
    assert additions.keys() <= manifested
    for relative, payload in additions.items():
        assert (tmp_path / relative).read_bytes() == payload


def test_lower_level_evaluator_does_not_mutate_frozen_physical_controls():
    source = (ROOT / "src/gnss_doppler_lab/gcspo_evaluate.py").read_text()
    assert 'canonical_write_json(artifact / "physical_controls.json", controls)' not in source
    assert 'canonical_write_json(artifact / "protected_control_status.json"' in source


def test_target_allowlist_has_no_review_receipt_or_execution_authorization():
    assert RECEIPT_RELATIVE not in TARGET_DIFF_ALLOWLIST
    assert AUTHORIZATION_RELATIVE not in TARGET_DIFF_ALLOWLIST
    assert all("approval" not in path for path in TARGET_DIFF_ALLOWLIST)


def test_strict_json_rejects_duplicate_keys_and_nonfinite():
    with pytest.raises(ValueError): strict_json_bytes(b'{"a":1,"a":2}', "fixture")
    with pytest.raises(ValueError): strict_json_bytes(b'{"a":NaN}', "fixture")
