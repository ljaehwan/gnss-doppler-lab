"""Round-4 pre-attack regressions using only synthetic temporary files.

These tests never discover or open protected payload paths and never invoke a
protected runner or claim a protected attempt.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, text=True,
                          capture_output=True).stdout.strip()


def _init_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "round4@test.invalid")
    _git(repo, "config", "user.name", "Round Four")
    (repo / "tracked.txt").write_text("target bytes\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "target")
    return repo, _git(repo, "rev-parse", "HEAD")


def _ready_gate(tmp_path: Path):
    from gnss_doppler_lab.gcspo_access import AccessGate

    gate = AccessGate(tmp_path / "ledger.jsonl")
    gate.set_preflight(clean_only_pass=True, reviews_pass=True,
                       freeze_sha="a" * 40, frozen_hashes={"config": "b" * 64})
    gate.set_remote_sync(local_sha="a" * 40, remote_sha="a" * 40,
                         ahead=0, behind=0, clean=True)
    return gate


def test_consumer_receives_authenticated_snapshot_not_mutated_live_descriptor(tmp_path, monkeypatch):
    """Same-inode/same-size mutation after authentication cannot reach parsing."""
    import gnss_doppler_lab.gcspo_access as access

    source = tmp_path / "synthetic.bin"
    authenticated = b"AUTHENTICATED-SNAPSHOT"
    unauthenticated = b"UNAUTHENTICATED-BYTES!"
    assert len(authenticated) == len(unauthenticated)
    source.write_bytes(authenticated)
    gate = _ready_gate(tmp_path)
    gate.register_pinned(
        source, expected_sha256=hashlib.sha256(authenticated).hexdigest(),
        expected_size=len(authenticated), kind="SYNTHETIC")

    real_lseek = access.os.lseek
    mutated = False

    def mutate_before_rewind(fd, offset, whence):
        nonlocal mutated
        if not mutated and offset == 0 and whence == os.SEEK_SET:
            mutated = True
            with source.open("r+b", buffering=0) as handle:
                handle.write(unauthenticated)
                handle.flush()
                os.fsync(handle.fileno())
        return real_lseek(fd, offset, whence)

    monkeypatch.setattr(access.os, "lseek", mutate_before_rewind)
    observed = gate.consume(source, scenario="DS3", phase="transition",
                            purpose="synthetic immutable snapshot",
                            consumer=lambda handle: handle.read())
    assert observed == authenticated
    assert mutated is True


def test_mutation_during_snapshot_acquisition_fails_without_consumer_success(tmp_path, monkeypatch):
    import gnss_doppler_lab.gcspo_access as access

    source = tmp_path / "synthetic.bin"
    original = b"A" * (2 * (1 << 20))
    source.write_bytes(original)
    gate = _ready_gate(tmp_path)
    gate.register_pinned(source, expected_sha256=hashlib.sha256(original).hexdigest(),
                         expected_size=len(original), kind="SYNTHETIC")
    real_read = access.os.read
    reads = 0

    def racing_read(fd, count):
        nonlocal reads
        block = real_read(fd, count)
        reads += 1
        if reads == 1:
            with source.open("r+b", buffering=0) as handle:
                handle.seek(1 << 20)
                handle.write(b"B" * (1 << 20))
                handle.flush()
                os.fsync(handle.fileno())
        return block

    monkeypatch.setattr(access.os, "read", racing_read)
    consumed = False

    def consumer(_handle):
        nonlocal consumed
        consumed = True

    with pytest.raises(ValueError, match="identity mismatch"):
        gate.consume(source, scenario="DS7", phase="transition",
                     purpose="synthetic acquisition race", consumer=consumer)
    assert consumed is False
    rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert sum(row.get("outcome") == "SUCCESS" for row in rows) == 0
    assert not (tmp_path / "protected_run_started.json").exists()


def _record_for(path: Path, target: str) -> dict:
    payload = path.read_bytes()
    row = {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
           "size_bytes": len(payload)}
    return {
        "schema": "gnss-doppler-lab.gcspo-stage0.implementation-freeze.v3",
        "validity_state": "AWAITING_INDEPENDENT_REREVIEW",
        "target_commit": target,
        "config_sha256": "c" * 64,
        "implementation_files": [row],
        "clean_scientific_artifacts": [row],
        "review_evidence": {
            "status": "REPAIRS_COMPLETE_AWAITING_REREVIEW",
            "rejected_freeze_commit": "a" * 40,
            "rejected_freeze_commits": ["a" * 40],
        },
        "manifest_excludes_self": True,
        "protected_access_authorized": False,
        "delivery_state": "ATTACK_NOT_AUTHORIZED_UNTIL_CONTROLLER_NONFORCE_PUSH_AND_EXACT_REMOTE_SYNC",
    }


@pytest.mark.parametrize("mode", ["target_differs", "target_lacks_row"])
def test_review_candidate_rows_are_bound_to_declared_git_target_tree(tmp_path, mode):
    from gnss_doppler_lab.gcspo_freeze import verify_review_candidate_record

    repo, target = _init_repo(tmp_path)
    if mode == "target_differs":
        member = repo / "tracked.txt"
        member.write_text("working tree bytes\n")
    else:
        member = repo / "wrapper-only.txt"
        member.write_text("wrapper only\n")
    record = _record_for(member, target)
    with pytest.raises(ValueError, match="target.*tree|Git.*tree|absent.*target"):
        verify_review_candidate_record(record, target_commit=target,
                                       repo_root=repo, wrapper_commit=None)


def test_review_wrapper_parent_must_equal_manifest_target(tmp_path):
    from gnss_doppler_lab.gcspo_freeze import verify_review_candidate_record

    repo, target = _init_repo(tmp_path)
    (repo / "other.txt").write_text("middle\n")
    _git(repo, "add", "other.txt")
    _git(repo, "commit", "-qm", "middle")
    member = repo / "tracked.txt"
    record = _record_for(member, target)
    (repo / "implementation_manifest.json").write_text(json.dumps(record) + "\n")
    _git(repo, "add", "implementation_manifest.json")
    _git(repo, "commit", "-qm", "wrapper")
    wrapper = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="wrapper.*parent|immediate.*parent"):
        verify_review_candidate_record(record, target_commit=target,
                                       repo_root=repo, wrapper_commit=wrapper)


def _challenge_repo(tmp_path: Path):
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "round4@test.invalid")
    _git(repo, "config", "user.name", "Round Four")
    (repo / "runner.py").write_text("print('synthetic run')\n")
    challenge = {
        "schema": "gnss-doppler-lab.gcspo-stage0.a5-challenges.v1",
        "tolerance": {"absolute": 1e-10, "relative": 1e-10},
        "source_files": ["runner.py"],
        "runs": [
            {"id": "cuda-1", "nonce": "1" * 64, "backend": "cuda"},
            {"id": "cuda-2", "nonce": "2" * 64, "backend": "cuda"},
            {"id": "cpu-1", "nonce": "3" * 64, "backend": "cpu"},
        ],
    }
    (repo / "challenge.json").write_text(json.dumps(challenge, sort_keys=True) + "\n")
    _git(repo, "add", "runner.py", "challenge.json")
    _git(repo, "commit", "-qm", "challenge")
    return repo, _git(repo, "rev-parse", "HEAD")


def _synthetic_completed_runs(tmp_path: Path):
    from gnss_doppler_lab.gcspo_provenance import complete_run, prepare_run

    repo, source_commit = _challenge_repo(tmp_path)
    roots = []
    for index, (run_id, backend) in enumerate((("cuda-1", "cuda"),
                                                ("cuda-2", "cuda"),
                                                ("cpu-1", "cpu")), 1):
        evidence = tmp_path / f"evidence-{run_id}"
        scratch = tmp_path / f"scratch-{run_id}"
        input_file = tmp_path / f"input-{run_id}.json"
        input_file.write_text("{}\n")
        prepared = prepare_run(
            repo_root=repo, source_commit=source_commit,
            challenge_path=repo / "challenge.json", challenge_id=run_id,
            backend=backend, argv=["python", "runner.py", "--backend", backend],
            scratch_root=scratch, evidence_root=evidence,
            input_files={"input": input_file},
            output_names=("clean_a5_report.json", "thresholds.json", "a5_numeric_trace.json"),
        )
        output = scratch / "output"
        output.mkdir()
        scientific = {"lambda": 100.0, "lambda_objectives": [{"lambda": 100.0, "mean_gcv": 1.0}],
                      "thresholds": {"q99": 2.0, "q995": 3.0},
                      "calibration": [{"score": 1.0, "rss": 2.0, "gcv": 3.0,
                                       "effective_dof": 4.0, "rank": 5}],
                      "holdout": [{"score": 1.5, "rss": 2.5, "gcv": 3.5,
                                   "effective_dof": 4.5, "rank": 5}]}
        (output / "clean_a5_report.json").write_text(json.dumps(scientific, sort_keys=True) + "\n")
        (output / "thresholds.json").write_text(json.dumps({"A5": scientific["thresholds"]}, sort_keys=True) + "\n")
        (output / "a5_numeric_trace.json").write_text(json.dumps({
            "backend": backend, "lambda": 100.0,
            "lambda_objectives": scientific["lambda_objectives"],
            "calibration": [{**scientific["calibration"][0], "state": [1.0, 2.0]}],
            "holdout": [{**scientific["holdout"][0], "state": [2.0, 3.0]}],
        }, sort_keys=True) + "\n")
        stdout = evidence / "stdout.txt"
        stderr = evidence / "stderr.txt"
        stdout.write_text(f"run={run_id}\n")
        stderr.write_text("")
        complete_run(
            prepared_path=prepared, pid=1000 + index,
            started_utc=f"2026-08-14T00:00:0{index}.000001Z",
            finished_utc=f"2026-08-14T00:00:0{index}.000002Z",
            exit_code=0, backend_truth={"requested": backend, "resolved": backend,
                                        "cuda_available": True,
                                        "device": "Synthetic GPU" if backend == "cuda" else "cpu"},
            stdout_path=stdout, stderr_path=stderr, output_dir=output,
        )
        roots.append(evidence)
    return repo, source_commit, roots


def test_causal_provenance_and_full_workload_cpu_cuda_parity_pass_synthetic(tmp_path):
    from gnss_doppler_lab.gcspo_provenance import (compare_full_a5_runs, verify_causal_runs,
                                                   verify_round4_unsigned_runs)

    repo, source_commit, roots = _synthetic_completed_runs(tmp_path)
    with pytest.raises(ValueError, match="Round-4 unsigned.*rejected"):
        verify_causal_runs(roots, repo_root=repo, source_commit=source_commit,
                           challenge_path="challenge.json")
    verified = verify_round4_unsigned_runs(roots, repo_root=repo, source_commit=source_commit,
                                           challenge_path="challenge.json")
    report = compare_full_a5_runs(verified)
    assert report["status"] == "PASS"
    assert report["same_backend"] == "BYTE_IDENTICAL"
    assert report["cpu_cuda"]["selected_lambda_unchanged"] is True
    assert report["cpu_cuda"]["thresholds_unchanged"] is True


@pytest.mark.parametrize("tamper", [
    "duplicate_nonce", "same_process_identity", "missing_prepared", "missing_completed",
    "modified_transcript", "nonzero_exit", "preexisting_output", "source_mismatch",
])
def test_causal_provenance_rejects_tamper_copy_and_incomplete_cases(tmp_path, tamper):
    from gnss_doppler_lab.gcspo_provenance import (prepare_run,
                                                   verify_causal_runs)

    if tamper == "preexisting_output":
        repo, source_commit = _challenge_repo(tmp_path)
        scratch = tmp_path / "scratch"
        (scratch / "output").mkdir(parents=True)
        input_file = tmp_path / "input.json"
        input_file.write_text("{}\n")
        with pytest.raises(ValueError, match="pre-existing|fresh scratch"):
            prepare_run(repo_root=repo, source_commit=source_commit,
                        challenge_path=repo / "challenge.json", challenge_id="cuda-1",
                        backend="cuda", argv=["python", "runner.py"], scratch_root=scratch,
                        evidence_root=tmp_path / "evidence", input_files={"input": input_file},
                        output_names=("clean_a5_report.json",))
        return

    repo, source_commit, roots = _synthetic_completed_runs(tmp_path)
    if tamper == "source_mismatch":
        (repo / "runner.py").write_text("print('changed')\n")
    elif tamper == "missing_prepared":
        (roots[0] / "prepared.json").unlink()
    elif tamper == "missing_completed":
        (roots[0] / "completed.json").unlink()
    elif tamper == "modified_transcript":
        (roots[0] / "stdout.txt").write_text("tampered\n")
    else:
        path = roots[1] / "completed.json"
        doc = json.loads(path.read_text())
        if tamper == "duplicate_nonce":
            first = json.loads((roots[0] / "prepared.json").read_text())
            prepared = json.loads((roots[1] / "prepared.json").read_text())
            prepared["nonce"] = first["nonce"]
            (roots[1] / "prepared.json").write_text(json.dumps(prepared, sort_keys=True) + "\n")
        elif tamper == "same_process_identity":
            doc["pid"] = json.loads((roots[0] / "completed.json").read_text())["pid"]
            path.write_text(json.dumps(doc, sort_keys=True) + "\n")
        elif tamper == "nonzero_exit":
            doc["exit_code"] = 7
            path.write_text(json.dumps(doc, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="provenance|nonce|process|PREPARED|COMPLETED|transcript|exit|source"):
        verify_causal_runs(roots, repo_root=repo, source_commit=source_commit,
                           challenge_path="challenge.json")


def test_remote_precondition_requires_symbolic_branch_and_exact_ahead_behind(tmp_path, monkeypatch):
    from gnss_doppler_lab.gcspo_freeze import live_remote_snapshot

    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake(command, **_kwargs):
        if "ls-remote" in command:
            return Result("a" * 40 + "\trefs/heads/topic\n")
        if command[-2:] == ["rev-parse", "HEAD"]:
            return Result("a" * 40 + "\n")
        if command[-2:] == ["symbolic-ref", "--short"]:
            return Result("wrong-branch\n")
        if "rev-list" in command:
            return Result("0\t0\n")
        if "status" in command:
            return Result("")
        raise AssertionError(command)

    monkeypatch.setattr("gnss_doppler_lab.gcspo_freeze.subprocess.run", fake)
    snapshot = live_remote_snapshot(tmp_path, "origin", "topic")
    assert snapshot["synchronized"] is False
    assert snapshot["branch"] == "topic"
    assert snapshot["symbolic_branch"] == "wrong-branch"

