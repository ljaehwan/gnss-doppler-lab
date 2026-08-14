from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess

import pytest

from gnss_doppler_lab.gcspo_successor_freeze import (
    build_successor_manifest,
    implementation_paths_at_commit,
    verify_control_protected_state,
    verify_handoff_protected_state,
    verify_successor_manifest,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, text=True,
                          capture_output=True).stdout.strip()


def _repo(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "successor@test.invalid")
    _git(repo, "config", "user.name", "Successor Test")
    paths = {
        "scripts/run_gcspo_clean_a5.py": "print(1)\n",
        "src/gnss_doppler_lab/gcspo_verify.py": (
            "from gnss_doppler_lab import gcmr_geometry\n"
            "from gnss_doppler_lab.trajectory import Trajectory\n"
            "VALUE = 1\n"
        ),
        "src/gnss_doppler_lab/gcmr_geometry.py": "from .internal_math import VALUE\n",
        "src/gnss_doppler_lab/trajectory.py": "class Trajectory: pass\n",
        "src/gnss_doppler_lab/internal_math.py": "VALUE = 1\n",
        "tests/test_gcspo_round4_freeze_repairs.py": "def test_old(): pass\n",
        "notes/not_gcspo.txt": "outside scope\n",
    }
    for relative, payload in paths.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "target")
    return repo, _git(repo, "rev-parse", "HEAD")


def _manifest(repo: Path, target: str) -> dict:
    return build_successor_manifest(
        repo, target_commit=target, invocation_id="inv-unique",
        nonce="1" * 64, predecessor_freeze_commit="a" * 40,
        invalid_evidence_commit="b" * 40,
        config_sha256="c" * 64, preregistration_sha256="d" * 64,
    )


def test_scope_is_deterministic_exact_git_tree_set_with_transitive_internal_imports(tmp_path):
    repo, target = _repo(tmp_path)
    assert implementation_paths_at_commit(repo, target) == [
        "scripts/run_gcspo_clean_a5.py",
        "src/gnss_doppler_lab/gcmr_geometry.py",
        "src/gnss_doppler_lab/gcspo_verify.py",
        "src/gnss_doppler_lab/internal_math.py",
        "src/gnss_doppler_lab/trajectory.py",
        "tests/test_gcspo_round4_freeze_repairs.py",
    ]
    assert _manifest(repo, target)["implementation_paths"] == implementation_paths_at_commit(repo, target)


def test_collusive_transitive_dependency_omission_fails_closed(tmp_path):
    repo, target = _repo(tmp_path)
    manifest = _manifest(repo, target)
    omitted = "src/gnss_doppler_lab/internal_math.py"
    manifest["implementation_paths"].remove(omitted)
    manifest["internal_import_closure_paths"].remove(omitted)
    manifest["files"] = [row for row in manifest["files"] if row["path"] != omitted]
    with pytest.raises(ValueError, match="path set|closure|missing"):
        verify_successor_manifest(manifest, repo, expected_target_commit=target)


def test_unresolved_new_internal_import_fails_closed(tmp_path):
    repo, _ = _repo(tmp_path)
    source = repo / "src/gnss_doppler_lab/gcspo_verify.py"
    source.write_text(source.read_text() + "import gnss_doppler_lab.not_tracked\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "unresolved internal dependency")
    with pytest.raises(ValueError, match="unresolved internal import"):
        implementation_paths_at_commit(repo, _git(repo, "rev-parse", "HEAD"))


@pytest.mark.parametrize("tamper", ["missing", "extra", "hash", "size"])
def test_missing_extra_and_identity_mismatch_fail_closed(tmp_path, tamper):
    repo, target = _repo(tmp_path)
    manifest = _manifest(repo, target)
    if tamper == "missing":
        manifest["files"].pop()
    elif tamper == "extra":
        manifest["files"].append({"path": "scripts/extra_gcspo.py", "sha256": "0" * 64, "size_bytes": 1})
    elif tamper == "hash":
        manifest["files"][0]["sha256"] = "0" * 64
    else:
        manifest["files"][0]["size_bytes"] += 1
    with pytest.raises(ValueError, match="row|set|identity|hash|size|missing|extra"):
        verify_successor_manifest(manifest, repo, expected_target_commit=target)


def test_post_target_source_commit_is_not_silently_accepted(tmp_path):
    repo, target = _repo(tmp_path)
    manifest = _manifest(repo, target)
    (repo / "src/gnss_doppler_lab/gcspo_verify.py").write_text("VALUE = 2\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "post-target source mutation")
    with pytest.raises(ValueError, match="parent|target|post-target"):
        verify_successor_manifest(
            manifest, repo, expected_target_commit=target,
            wrapper_commit=_git(repo, "rev-parse", "HEAD"),
        )


@pytest.mark.parametrize("relative", [
    "src/gnss_doppler_lab/gcmr_geometry.py",
    "src/gnss_doppler_lab/trajectory.py",
    "src/gnss_doppler_lab/internal_math.py",
])
def test_post_target_internal_dependency_mutation_fails_closed(tmp_path, relative):
    repo, target = _repo(tmp_path)
    manifest = _manifest(repo, target)
    path = repo / relative
    path.write_text(path.read_text() + "MUTATED = True\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "post-target internal dependency mutation")
    with pytest.raises(ValueError, match="post-target.*source|implementation"):
        verify_successor_manifest(
            manifest, repo, expected_target_commit=target,
            wrapper_commit=_git(repo, "rev-parse", "HEAD"),
        )


def _tamper_protected(manifest: dict, mutation: str) -> None:
    if mutation == "count_bool":
        manifest["protected_access_count"] = True
    elif mutation == "count_nonzero":
        manifest["protected_access_count"] = 1
    elif mutation == "count_type":
        manifest["protected_access_count"] = "0"
    elif mutation == "count_missing":
        manifest.pop("protected_access_count")
    elif mutation == "marker_true":
        manifest["protected_marker_present"] = True
    elif mutation == "marker_type":
        manifest["protected_marker_present"] = 0
    elif mutation == "marker_missing":
        manifest.pop("protected_marker_present")
    elif mutation == "ledger_bool":
        manifest["protected_ledger_size_bytes"] = True
    elif mutation == "ledger_nonzero":
        manifest["protected_ledger_size_bytes"] = 1
    elif mutation == "ledger_type":
        manifest["protected_ledger_size_bytes"] = 0.0
    elif mutation == "extra":
        manifest["protected_rows_opened"] = 0
    else:
        raise AssertionError(mutation)


@pytest.mark.parametrize("mutation", [
    "count_bool", "count_nonzero", "count_type", "count_missing",
    "marker_true", "marker_type", "marker_missing",
    "ledger_bool", "ledger_nonzero", "ledger_type", "extra",
])
def test_manifest_protected_state_bool_missing_extra_type_value_tamper_fails_closed(
        tmp_path, mutation):
    repo, target = _repo(tmp_path)
    manifest = _manifest(repo, target)
    _tamper_protected(manifest, mutation)
    with pytest.raises(ValueError, match="protected|schema|key|shape|contract"):
        verify_successor_manifest(manifest, repo, expected_target_commit=target)


def _committed_document(name: str) -> dict:
    root = Path(__file__).parents[1]
    artifact = (root / "artifacts/gcspo_stage0_static_rerun_successor" /
                "gcspo-stage0-successor-7be48c4411644ff3a9ec41c7701dfa01")
    document = json.loads((artifact / name).read_text())
    if name == "review_handoff.json":
        document["prior_independent_rejection"] = {
            "wrapper_commit": "078a8c5739c92e877763583ce2ca23dad4f433f9",
            "target_commit": "1bfc3d2b64ad43a9db78081f3abc482bdf5d022f",
            "verdict": "REJECT",
            "blocking_findings": [
                "TRANSITIVE_INTERNAL_IMPORT_CLOSURE_MISSING",
                "PROTECTED_STATE_SCHEMA_TYPE_VALUE_NOT_STRICT",
            ],
        }
    return document


@pytest.mark.parametrize("mutation", [
    "top_missing", "top_extra", "prior_count_bool", "prior_count_missing",
    "prior_extra", "prior_marker_true", "prior_marker_type", "prior_ledger_nonzero",
    "prior_ledger_type", "successor_authorized_type", "successor_marker_true",
    "successor_ledger_nonzero",
])
def test_control_protected_state_exact_schema_type_and_value_fails_closed(mutation):
    control = _committed_document("successor_control.json")
    if mutation == "top_missing":
        control.pop("prohibitions")
    elif mutation == "top_extra":
        control["protected_state_note"] = False
    elif mutation == "prior_count_bool":
        control["prior_protected_state"]["protected_access_count"] = True
    elif mutation == "prior_count_missing":
        control["prior_protected_state"].pop("protected_access_count")
    elif mutation == "prior_extra":
        control["prior_protected_state"]["extra"] = 0
    elif mutation == "prior_marker_true":
        control["prior_protected_state"]["marker_present"] = True
    elif mutation == "prior_marker_type":
        control["prior_protected_state"]["marker_present"] = 0
    elif mutation == "prior_ledger_nonzero":
        control["prior_protected_state"]["ledger_size_bytes"] = 1
    elif mutation == "prior_ledger_type":
        control["prior_protected_state"]["ledger_size_bytes"] = 0.0
    elif mutation == "successor_authorized_type":
        control["successor_namespace"]["protected_access_authorized"] = 0
    elif mutation == "successor_marker_true":
        control["successor_namespace"]["marker_present"] = True
    elif mutation == "successor_ledger_nonzero":
        control["successor_namespace"]["ledger_size_bytes"] = 1
    else:
        raise AssertionError(mutation)
    with pytest.raises(ValueError, match="control|protected|schema|key|type|value"):
        verify_control_protected_state(control)


@pytest.mark.parametrize("mutation", [
    "top_missing", "top_extra", "count_bool", "count_missing", "protected_extra",
    "count_nonzero", "marker_true", "marker_type", "ledger_bool", "ledger_nonzero",
    "ledger_type", "authorized_type",
])
def test_handoff_protected_state_exact_schema_type_and_value_fails_closed(mutation):
    handoff = _committed_document("review_handoff.json")
    if mutation == "top_missing":
        handoff.pop("push_performed")
    elif mutation == "top_extra":
        handoff["protected_state_note"] = False
    elif mutation == "count_bool":
        handoff["protected"]["access_count"] = True
    elif mutation == "count_missing":
        handoff["protected"].pop("access_count")
    elif mutation == "protected_extra":
        handoff["protected"]["rows_opened"] = 0
    elif mutation == "count_nonzero":
        handoff["protected"]["access_count"] = 1
    elif mutation == "marker_true":
        handoff["protected"]["marker_present"] = True
    elif mutation == "marker_type":
        handoff["protected"]["marker_present"] = 0
    elif mutation == "ledger_bool":
        handoff["protected"]["ledger_size_bytes"] = True
    elif mutation == "ledger_nonzero":
        handoff["protected"]["ledger_size_bytes"] = 1
    elif mutation == "ledger_type":
        handoff["protected"]["ledger_size_bytes"] = 0.0
    elif mutation == "authorized_type":
        handoff["protected"]["authorized"] = 0
    else:
        raise AssertionError(mutation)
    with pytest.raises(ValueError, match="handoff|protected|schema|key|type|value"):
        verify_handoff_protected_state(handoff)


def test_working_tree_shadow_fails_closed(tmp_path):
    repo, target = _repo(tmp_path)
    manifest = _manifest(repo, target)
    (repo / "scripts/run_gcspo_clean_a5.py").write_text("print(2)\n")
    with pytest.raises(ValueError, match="working.tree|shadow|worktree"):
        verify_successor_manifest(manifest, repo, expected_target_commit=target,
                                  require_clean_worktree=True)


def test_wrapper_immediate_parent_must_be_target(tmp_path):
    repo, target = _repo(tmp_path)
    manifest = _manifest(repo, target)
    (repo / "middle.txt").write_text("middle\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "middle")
    (repo / "wrapper.json").write_text("{}\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "wrapper")
    with pytest.raises(ValueError, match="wrapper.*parent|parent.*target"):
        verify_successor_manifest(
            manifest, repo, expected_target_commit=target,
            wrapper_commit=_git(repo, "rev-parse", "HEAD"),
        )
