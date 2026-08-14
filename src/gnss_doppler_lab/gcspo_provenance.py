"""Causal clean-A5 launch provenance and full-workload parity verification."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def _canonical(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(path, *, allow_empty=True):
    payload = Path(path).read_bytes()
    if not allow_empty and not payload:
        raise ValueError(f"provenance identity is empty: {path}")
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _check_identity(path, identity, label, *, allow_empty=True):
    if not isinstance(identity, dict) or set(identity) != {"sha256", "size_bytes"}:
        raise ValueError(f"provenance {label} identity is malformed")
    observed = _identity(path, allow_empty=allow_empty)
    if observed != identity:
        raise ValueError(f"provenance {label} identity mismatch")
    return observed


def _git(repo, *arguments, binary=False):
    result = subprocess.run(["git", *arguments], cwd=repo, check=True,
                            capture_output=True, text=not binary)
    return result.stdout if binary else result.stdout.strip()


def _timestamp(value, label):
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"provenance {label} timestamp is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"provenance {label} timestamp is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"provenance {label} timestamp is not UTC")
    return parsed


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _exclusive_json(path, document):
    path = Path(path)
    data = _canonical(document) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def _challenge(repo_root, source_commit, challenge_path):
    repo = Path(repo_root).resolve(strict=True)
    current = Path(challenge_path)
    if not current.is_absolute():
        current = repo / current
    relative = current.resolve(strict=True).relative_to(repo).as_posix()
    committed = _git(repo, "show", f"{source_commit}:{relative}", binary=True)
    if current.read_bytes() != committed:
        raise ValueError("provenance challenge source mismatch")
    document = json.loads(committed)
    if document.get("schema") != "gnss-doppler-lab.gcspo-stage0.a5-challenges.v1":
        raise ValueError("provenance challenge schema mismatch")
    runs = document.get("runs")
    if not isinstance(runs, list) or len(runs) < 3:
        raise ValueError("provenance challenge run set is incomplete")
    ids = [row.get("id") for row in runs]
    nonces = [row.get("nonce") for row in runs]
    if (len(ids) != len(set(ids)) or len(nonces) != len(set(nonces)) or
            any(not isinstance(value, str) or len(value) != 64 for value in nonces)):
        raise ValueError("provenance challenge nonce identity is invalid")
    tolerance = document.get("tolerance")
    if (not isinstance(tolerance, dict) or set(tolerance) != {"absolute", "relative"} or
            any(not isinstance(tolerance[key], (int, float)) or tolerance[key] <= 0
                for key in tolerance)):
        raise ValueError("provenance preregistered parity tolerance is invalid")
    source = {}
    for item in document.get("source_files", []):
        if not isinstance(item, str) or Path(item).is_absolute() or ".." in Path(item).parts:
            raise ValueError("provenance source path is invalid")
        payload = _git(repo, "show", f"{source_commit}:{item}", binary=True)
        path = repo / item
        if path.read_bytes() != payload:
            raise ValueError("provenance source mismatch")
        source[item] = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    if not source:
        raise ValueError("provenance source identities are absent")
    return document, relative, _identity(current, allow_empty=False), source


def prepare_run(*, repo_root, source_commit, challenge_path, challenge_id, backend,
                argv, scratch_root, evidence_root, input_files, output_names):
    repo = Path(repo_root).resolve(strict=True)
    if len(source_commit) != 40 or _git(repo, "rev-parse", "HEAD") != source_commit:
        raise ValueError("provenance source commit mismatch")
    challenge, relative, challenge_identity, source = _challenge(
        repo, source_commit, challenge_path)
    matching = [row for row in challenge["runs"] if row["id"] == challenge_id]
    if len(matching) != 1 or matching[0].get("backend") != backend:
        raise ValueError("provenance challenge/backend mismatch")
    if backend not in {"cpu", "cuda"}:
        raise ValueError("provenance backend is invalid")
    if not isinstance(argv, list) or not argv or any(not isinstance(value, str) or not value for value in argv):
        raise ValueError("provenance argv is invalid")
    scratch = Path(scratch_root)
    evidence = Path(evidence_root)
    if scratch.exists() or evidence.exists():
        raise ValueError("fresh scratch required; pre-existing output or evidence found")
    scratch.mkdir(parents=True)
    evidence.mkdir(parents=True)
    inputs = {}
    for name, path in sorted(input_files.items()):
        inputs[name] = {"path": str(Path(path).resolve(strict=True)),
                        **_identity(path, allow_empty=False)}
    names = list(output_names)
    if not names or len(names) != len(set(names)) or any(Path(name).name != name for name in names):
        raise ValueError("provenance output contract is invalid")
    document = {
        "schema": "gnss-doppler-lab.gcspo-stage0.a5-run-prepared.v1",
        "status": "PREPARED", "source_commit": source_commit,
        "challenge_path": relative, "challenge_identity": challenge_identity,
        "challenge_id": challenge_id, "nonce": matching[0]["nonce"],
        "backend": backend, "argv": argv, "scratch_root": str(scratch.resolve()),
        "evidence_root": str(evidence.resolve()), "prepared_utc": utc_now(),
        "source_identities": source, "input_identities": inputs,
        "output_names": names,
    }
    document["prepared_sha256"] = hashlib.sha256(_canonical(document)).hexdigest()
    return _exclusive_json(evidence / "prepared.json", document)


def _load_prepared(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError("provenance PREPARED predecessor is absent")
    document = json.loads(path.read_text())
    digest = document.pop("prepared_sha256", None)
    if digest != hashlib.sha256(_canonical(document)).hexdigest():
        raise ValueError("provenance PREPARED predecessor hash mismatch")
    document["prepared_sha256"] = digest
    if document.get("status") != "PREPARED":
        raise ValueError("provenance PREPARED predecessor state mismatch")
    return document


def complete_run(*, prepared_path, pid, started_utc, finished_utc, exit_code,
                 backend_truth, stdout_path, stderr_path, output_dir):
    prepared = _load_prepared(prepared_path)
    if _timestamp(started_utc, "start") >= _timestamp(finished_utc, "finish"):
        raise ValueError("provenance launch did not finish after start")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("provenance process identity is invalid")
    output = Path(output_dir).resolve(strict=True)
    if output != Path(prepared["scratch_root"]).resolve() / "output":
        raise ValueError("provenance output does not belong to prepared scratch root")
    outputs = {}
    for name in prepared["output_names"]:
        outputs[name] = _identity(output / name, allow_empty=False)
    truth = dict(backend_truth)
    if truth.get("requested") != prepared["backend"] or truth.get("resolved") != prepared["backend"]:
        raise ValueError("provenance backend truth mismatch")
    durable = Path(prepared_path).parent / "bundle"
    if durable.exists():
        raise ValueError("provenance durable output bundle pre-exists")
    durable.mkdir()
    for name in prepared["output_names"]:
        shutil.copy2(output / name, durable / name)
        if _identity(durable / name, allow_empty=False) != outputs[name]:
            raise RuntimeError("provenance durable output copy identity mismatch")
    document = {
        "schema": "gnss-doppler-lab.gcspo-stage0.a5-run-completed.v1",
        "status": "COMPLETED" if exit_code == 0 else "FAILED",
        "prepared_sha256": prepared["prepared_sha256"],
        "source_commit": prepared["source_commit"], "challenge_id": prepared["challenge_id"],
        "nonce": prepared["nonce"], "backend": prepared["backend"],
        "argv": prepared["argv"], "scratch_root": prepared["scratch_root"],
        "pid": pid, "started_utc": started_utc, "finished_utc": finished_utc,
        "exit_code": int(exit_code), "backend_truth": truth,
        "stdout": _identity(stdout_path), "stderr": _identity(stderr_path),
        "outputs": outputs,
    }
    document["completed_sha256"] = hashlib.sha256(_canonical(document)).hexdigest()
    path = Path(prepared_path).parent / "completed.json"
    _exclusive_json(path, document)
    if exit_code != 0:
        raise RuntimeError("provenance child exit is nonzero")
    return path


def _load_completed(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError("provenance COMPLETED launch record is absent")
    document = json.loads(path.read_text())
    digest = document.pop("completed_sha256", None)
    if digest != hashlib.sha256(_canonical(document)).hexdigest():
        raise ValueError("provenance COMPLETED hash mismatch")
    document["completed_sha256"] = digest
    return document


def verify_causal_runs(evidence_roots, *, repo_root, source_commit, challenge_path):
    repo = Path(repo_root).resolve(strict=True)
    challenge, relative, challenge_identity, source = _challenge(
        repo, source_commit, challenge_path)
    expected = {row["id"]: row for row in challenge["runs"]}
    roots = [Path(root).resolve(strict=True) for root in evidence_roots]
    if len(roots) != len(expected) or len(roots) != len(set(roots)):
        raise ValueError("provenance run evidence set is incomplete or duplicated")
    verified = []
    for root in roots:
        prepared = _load_prepared(root / "prepared.json")
        completed = _load_completed(root / "completed.json")
        if (prepared["source_commit"] != source_commit or prepared["challenge_path"] != relative or
                prepared["challenge_identity"] != challenge_identity or
                prepared["source_identities"] != source):
            raise ValueError("provenance source/challenge binding mismatch")
        expected_row = expected.get(prepared["challenge_id"])
        if expected_row is None or prepared["nonce"] != expected_row["nonce"] or prepared["backend"] != expected_row["backend"]:
            raise ValueError("provenance nonce/backend challenge mismatch")
        if (completed["prepared_sha256"] != prepared["prepared_sha256"] or
                any(completed[key] != prepared[key] for key in
                    ("source_commit", "challenge_id", "nonce", "backend", "argv", "scratch_root"))):
            raise ValueError("provenance COMPLETED/PREPARED chain mismatch")
        if completed["status"] != "COMPLETED" or completed["exit_code"] != 0:
            raise ValueError("provenance child exit is nonzero or incomplete")
        if not (_timestamp(completed["started_utc"], "start") < _timestamp(completed["finished_utc"], "finish")):
            raise ValueError("provenance causal timestamp ordering mismatch")
        if completed["backend_truth"].get("requested") != prepared["backend"] or completed["backend_truth"].get("resolved") != prepared["backend"]:
            raise ValueError("provenance backend truth mismatch")
        _check_identity(root / "stdout.txt", completed["stdout"], "stdout transcript")
        _check_identity(root / "stderr.txt", completed["stderr"], "stderr transcript")
        output = root / "bundle"
        for name, identity in completed["outputs"].items():
            _check_identity(output / name, identity, f"output:{name}", allow_empty=False)
        verified.append({"root": root, "prepared": prepared, "completed": completed,
                         "output_dir": output})
    nonces = [row["prepared"]["nonce"] for row in verified]
    pids = [row["completed"]["pid"] for row in verified]
    if len(nonces) != len(set(nonces)):
        raise ValueError("provenance duplicate nonce")
    if len(pids) != len(set(pids)):
        raise ValueError("provenance duplicate process identity")
    if sorted(row["prepared"]["backend"] for row in verified) != ["cpu", "cuda", "cuda"]:
        raise ValueError("provenance backend run count mismatch")
    return {"source_commit": source_commit, "challenge": challenge, "runs": verified}


def _walk_numeric(first, second, path=""):
    if isinstance(first, bool) or isinstance(second, bool):
        if first != second:
            raise ValueError(f"CPU/CUDA nonnumeric mismatch at {path}")
        return []
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return [(path, float(first), float(second))]
    if type(first) is not type(second):
        raise ValueError(f"CPU/CUDA structure mismatch at {path}")
    if isinstance(first, dict):
        if set(first) != set(second):
            raise ValueError(f"CPU/CUDA keys mismatch at {path}")
        result = []
        for key in sorted(first):
            result.extend(_walk_numeric(first[key], second[key], f"{path}.{key}" if path else key))
        return result
    if isinstance(first, list):
        if len(first) != len(second):
            raise ValueError(f"CPU/CUDA length mismatch at {path}")
        result = []
        for index, (left, right) in enumerate(zip(first, second)):
            result.extend(_walk_numeric(left, right, f"{path}[{index}]"))
        return result
    if first != second:
        raise ValueError(f"CPU/CUDA nonnumeric mismatch at {path}")
    return []


def compare_full_a5_runs(verified):
    runs = verified["runs"]
    cuda = [row for row in runs if row["prepared"]["backend"] == "cuda"]
    cpu = [row for row in runs if row["prepared"]["backend"] == "cpu"]
    if len(cuda) != 2 or len(cpu) != 1:
        raise ValueError("CPU/CUDA full workload count mismatch")
    names = cuda[0]["prepared"]["output_names"]
    for name in names:
        if (cuda[0]["output_dir"] / name).read_bytes() != (cuda[1]["output_dir"] / name).read_bytes():
            raise ValueError(f"same-backend A5 output is not byte-identical: {name}")
    cuda_report = json.loads((cuda[0]["output_dir"] / "clean_a5_report.json").read_text())
    cpu_report = json.loads((cpu[0]["output_dir"] / "clean_a5_report.json").read_text())
    if cuda_report.get("lambda") != cpu_report.get("lambda"):
        raise ValueError("CPU/CUDA selected lambda changed")
    if cuda_report.get("thresholds") != cpu_report.get("thresholds"):
        raise ValueError("CPU/CUDA thresholds changed")
    cuda_trace = json.loads((cuda[0]["output_dir"] / "a5_numeric_trace.json").read_text())
    cpu_trace = json.loads((cpu[0]["output_dir"] / "a5_numeric_trace.json").read_text())
    pairs = _walk_numeric(cuda_trace, cpu_trace)
    tolerance = verified["challenge"]["tolerance"]
    maximum_absolute = 0.0
    maximum_relative = 0.0
    maximum_field = None
    for field, left, right in pairs:
        absolute = abs(left - right)
        relative = absolute / max(abs(left), abs(right), 1.0)
        if absolute > maximum_absolute:
            maximum_absolute, maximum_field = absolute, field
        maximum_relative = max(maximum_relative, relative)
        if absolute > tolerance["absolute"] and relative > tolerance["relative"]:
            raise ValueError(f"CPU/CUDA parity tolerance exceeded at {field}")
    return {
        "schema": "gnss-doppler-lab.gcspo-stage0.a5-full-workload-parity.v1",
        "status": "PASS", "source_commit": verified["source_commit"],
        "run_count": 3, "backends": ["cuda", "cuda", "cpu"],
        "same_backend": "BYTE_IDENTICAL",
        "same_backend_files": names,
        "cpu_cuda": {
            "selected_lambda_unchanged": True, "thresholds_unchanged": True,
            "numeric_field_count": len(pairs), "maximum_absolute_delta": maximum_absolute,
            "maximum_relative_delta": maximum_relative, "maximum_delta_field": maximum_field,
            "tolerance": tolerance,
        },
    }

