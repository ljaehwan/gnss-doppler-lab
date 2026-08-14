#!/usr/bin/env python3
"""Launch one clean A5 child only after external PREPARED verification.

This process never signs an envelope.  It records an unsigned observation for
the external controller to independently check, canonicalize as COMPLETED, and
sign only after this exact command has exited.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.gcspo_provenance import canonical_json_bytes
from gnss_doppler_lab.gcspo_witness import validate_prepared_for_launch


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _identity(path, *, allow_empty=True):
    payload = Path(path).read_bytes()
    if not allow_empty and not payload:
        raise ValueError(f"witness observation refuses empty file: {path}")
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _write_exclusive(path, document):
    payload = canonical_json_bytes(document)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _git(*arguments):
    return subprocess.run(["git", *arguments], cwd=ROOT, check=True, text=True,
                          capture_output=True).stdout.strip()


def _proc_start_ticks(pid):
    tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(tail[19])


def _argument(argv, name):
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"prepared child argv lacks {name}") from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-envelope", required=True, type=Path)
    parser.add_argument("--prepared-signature", required=True, type=Path)
    parser.add_argument("--challenge-file", required=True, type=Path)
    args = parser.parse_args()

    untrusted = json.loads(args.prepared_envelope.read_text())
    source_commit = untrusted.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("PREPARED source commit is absent before signature verification")
    loaded = validate_prepared_for_launch(
        args.prepared_envelope, args.prepared_signature, repo_root=ROOT,
        source_commit=source_commit, challenge_path=args.challenge_file,
        observed_start_utc=_utc_now())
    prepared, challenge = loaded["document"], loaded["context"]["document"]
    row = loaded["row"]
    controller_argv = [sys.executable, str(Path(__file__).absolute()), *sys.argv[1:]]
    if row.get("controller_argv") != controller_argv:
        raise ValueError("controller did not launch the exact challenged witness command")

    scratch = Path(prepared["scratch_root"])
    output = Path(prepared["output_root"])
    evidence = Path(prepared["evidence_root"])
    scratch.mkdir(parents=True)
    output.mkdir()
    (scratch / "tmp").mkdir()
    evidence.mkdir(parents=True)
    shutil.copy2(args.prepared_envelope, evidence / "prepared.json")
    shutil.copy2(args.prepared_signature, evidence / "prepared.json.sig")
    for name, record in challenge.get("input_files", {}).items():
        source = ROOT / record["path"]
        if _identity(source, allow_empty=False) != {"sha256": record["sha256"],
                                                    "size_bytes": record["size_bytes"]}:
            raise ValueError(f"challenged input identity mismatch: {name}")
    seed = challenge["input_files"]
    shutil.copy2(ROOT / seed["config"]["path"], output / "config.json")
    shutil.copy2(ROOT / seed["thresholds"]["path"], output / "thresholds.json")

    stdout_path, stderr_path = evidence / "stdout.txt", evidence / "stderr.txt"
    observed_start = _utc_now()
    monotonic_start = time.monotonic_ns()
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        child = subprocess.Popen(prepared["argv"], cwd=ROOT,
                                 env=dict(prepared["environment"]),
                                 stdout=stdout, stderr=stderr)
        process_identity = {"pid": child.pid,
                            "proc_start_ticks": _proc_start_ticks(child.pid)}
        exit_code = child.wait()
    elapsed = time.monotonic_ns() - monotonic_start
    observed_end = _utc_now()
    receipt_path = evidence / "execution_receipt.json"
    bundle = evidence / "bundle"
    outputs = {}
    if exit_code == 0:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("process_identity") != process_identity:
            raise ValueError("child receipt process identity differs from observed process")
        bundle.mkdir()
        for name in prepared["scientific_output_names"]:
            source = output / name
            outputs[name] = _identity(source, allow_empty=False)
            shutil.copy2(source, bundle / name)
            if _identity(bundle / name, allow_empty=False) != outputs[name]:
                raise RuntimeError("durable scientific bundle copy identity mismatch")
    worktree = _git("status", "--porcelain", "--untracked-files=all").encode()
    observation = {
        "schema": "gnss-doppler-lab.gcspo-stage0.a5-controller-completion-observation.v1",
        "status": "OBSERVED_SUCCESS" if exit_code == 0 else "OBSERVED_FAILURE",
        "run_id": prepared["run_id"], "nonce": prepared["nonce"],
        "source_commit": source_commit, "challenge_path": prepared["challenge_path"],
        "challenge": prepared["challenge"],
        "prepared_envelope": _identity(evidence / "prepared.json", allow_empty=False),
        "prepared_signature": _identity(evidence / "prepared.json.sig", allow_empty=False),
        "controller_argv": controller_argv,
        "argv": prepared["argv"], "environment": prepared["environment"],
        "backend": prepared["backend"], "observed_exit_code": exit_code,
        "process_identity": process_identity,
        "observed_process_start_utc": observed_start,
        "observed_process_end_utc": observed_end,
        "observed_elapsed_ns": elapsed,
        "stdout": _identity(stdout_path), "stderr": _identity(stderr_path),
        "execution_receipt": _identity(receipt_path, allow_empty=False) if receipt_path.is_file() else None,
        "backend_truth": _identity(bundle / "a5_backend_truth.json", allow_empty=False) if bundle.is_dir() else None,
        "scientific_outputs": outputs,
        "post_run_source": {
            "head_commit": _git("rev-parse", "HEAD"),
            "worktree_status": {"sha256": hashlib.sha256(worktree).hexdigest(),
                                "size_bytes": len(worktree)},
            "source_identities": prepared["source_identities"],
        },
    }
    _write_exclusive(evidence / "completion_observation.json", observation)
    if exit_code != 0:
        raise RuntimeError(f"witnessed A5 child exit is nonzero: {exit_code}")
    if worktree:
        raise RuntimeError("witnessed A5 post-run worktree is not clean")
    print(f"WITNESSED_A5_OBSERVED id={prepared['run_id']} backend={prepared['backend']} "
          f"pid={process_identity['pid']} receipt={receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
