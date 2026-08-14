from __future__ import annotations

import copy
from pathlib import Path
import subprocess

import pytest

from gnss_doppler_lab.gcspo_successor_freeze import (
    build_successor_manifest,
    implementation_paths_at_commit,
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
        "src/gnss_doppler_lab/gcspo_verify.py": "VALUE = 1\n",
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


def test_scope_is_deterministic_exact_git_tree_set(tmp_path):
    repo, target = _repo(tmp_path)
    assert implementation_paths_at_commit(repo, target) == [
        "scripts/run_gcspo_clean_a5.py",
        "src/gnss_doppler_lab/gcspo_verify.py",
        "tests/test_gcspo_round4_freeze_repairs.py",
    ]
    assert _manifest(repo, target)["implementation_paths"] == implementation_paths_at_commit(repo, target)


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
