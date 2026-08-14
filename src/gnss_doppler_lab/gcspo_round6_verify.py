"""Read-only reconstruction of the packaged Round-6 witness freeze."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

from .gcspo_provenance import verify_witnessed_runs


SOURCE_COMMIT = "9774fe1048e467808b53769f94a717507fac5a38"
CHALLENGE_PATH = "artifacts/gcspo_stage0_static_rerun/a5_round6_challenges.json"
RUN_IDS = ("round6-cuda-1", "round6-cuda-2", "round6-cpu-1")
EVIDENCE_FILES = (
    "prepared.json", "prepared.json.sig", "completed.json", "completed.json.sig",
    "execution_receipt.json", "stdout.txt", "stderr.txt",
    "bundle/clean_a5_report.json", "bundle/thresholds.json",
    "bundle/a5_numeric_trace.json", "bundle/a5_backend_truth.json",
)


def _identity(path: Path) -> dict[str, int | str]:
    payload = path.read_bytes()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _load(path: Path) -> dict:
    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    document = json.loads(path.read_text(), parse_constant=reject_constant)
    if not isinstance(document, dict):
        raise ValueError(f"round-6 document is not an object: {path}")
    return document


def _finite_number(value, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"CPU/CUDA numeric value is not a number at {path}")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"CPU/CUDA numeric value is not finite at {path}") from exc
    if not math.isfinite(result):
        raise ValueError(f"CPU/CUDA non-finite numeric value at {path}")
    return result


def _finite_derived(value: float, name: str, path: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"CPU/CUDA non-finite derived {name} at {path}")
    return value


def _comparison_metrics(left, right, path: str) -> dict[str, float]:
    left = _finite_number(left, f"{path}.left")
    right = _finite_number(right, f"{path}.right")
    delta = _finite_derived(left - right, "delta", path)
    absolute = _finite_derived(abs(delta), "absolute value", path)
    scale = _finite_derived(max(abs(left), abs(right), 1.0), "scale", path)
    relative = _finite_derived(absolute / scale, "relative value", path)
    return {"delta": delta, "absolute": absolute, "scale": scale, "relative": relative}


def _numeric_pairs(first, second, path=""):
    if isinstance(first, bool) or isinstance(second, bool):
        if first != second:
            raise ValueError(f"CPU/CUDA nonnumeric mismatch at {path}")
        return
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        yield path, _finite_number(first, f"{path}.left"), _finite_number(second, f"{path}.right")
        return
    if type(first) is not type(second):
        raise ValueError(f"CPU/CUDA structure mismatch at {path}")
    if isinstance(first, dict):
        if set(first) != set(second):
            raise ValueError(f"CPU/CUDA keys mismatch at {path}")
        for key in sorted(first):
            yield from _numeric_pairs(first[key], second[key], f"{path}.{key}" if path else key)
        return
    if isinstance(first, list):
        if len(first) != len(second):
            raise ValueError(f"CPU/CUDA length mismatch at {path}")
        for index, (left, right) in enumerate(zip(first, second)):
            yield from _numeric_pairs(left, right, f"{path}[{index}]")
        return
    if first != second:
        raise ValueError(f"CPU/CUDA nonnumeric mismatch at {path}")


def compare_round6_a5_runs(verified: dict) -> dict:
    """Apply the preregistered Round-4 tolerance to signed Round-6 outputs."""
    runs = verified["runs"]
    cuda = [row for row in runs if row["prepared"]["backend"] == "cuda"]
    cpu = [row for row in runs if row["prepared"]["backend"] == "cpu"]
    if len(cuda) != 2 or len(cpu) != 1:
        raise ValueError("CPU/CUDA full workload count mismatch")
    names = cuda[0]["prepared"]["scientific_output_names"]
    if any(row["prepared"]["scientific_output_names"] != names for row in runs):
        raise ValueError("Round-6 scientific output contracts differ")
    for name in names:
        if (cuda[0]["output_dir"] / name).read_bytes() != (cuda[1]["output_dir"] / name).read_bytes():
            raise ValueError(f"same-backend A5 output is not byte-identical: {name}")

    cuda_report = _load(cuda[0]["output_dir"] / "clean_a5_report.json")
    cpu_report = _load(cpu[0]["output_dir"] / "clean_a5_report.json")
    cuda_lambda = _finite_number(cuda_report.get("lambda"), "clean_a5_report.lambda.cuda")
    cpu_lambda = _finite_number(cpu_report.get("lambda"), "clean_a5_report.lambda.cpu")
    if cuda_lambda != cpu_lambda:
        raise ValueError("CPU/CUDA selected lambda changed")
    tolerance = verified["challenge"]["tolerance"]
    absolute_tolerance = _finite_number(tolerance.get("absolute"), "challenge.tolerance.absolute")
    relative_tolerance = _finite_number(tolerance.get("relative"), "challenge.tolerance.relative")
    if set(cuda_report.get("thresholds", {})) != set(cpu_report.get("thresholds", {})):
        raise ValueError("CPU/CUDA threshold keys changed")
    threshold_deltas = {}
    for name in sorted(cuda_report["thresholds"]):
        left = _finite_number(cuda_report["thresholds"][name], f"thresholds.{name}.cuda")
        right = _finite_number(cpu_report["thresholds"][name], f"thresholds.{name}.cpu")
        metrics = _comparison_metrics(left, right, f"thresholds.{name}")
        absolute, relative = metrics["absolute"], metrics["relative"]
        threshold_deltas[name] = {"absolute": absolute, "relative": relative}
        if absolute > absolute_tolerance and relative > relative_tolerance:
            raise ValueError(f"CPU/CUDA threshold parity tolerance exceeded: {name}")

    cuda_trace = _load(cuda[0]["output_dir"] / "a5_numeric_trace.json")
    cpu_trace = _load(cpu[0]["output_dir"] / "a5_numeric_trace.json")
    cuda_trace.pop("backend", None)
    cpu_trace.pop("backend", None)
    numeric_field_count = 0
    maximum_absolute = 0.0
    maximum_relative = 0.0
    maximum_field = None
    for field, left, right in _numeric_pairs(cuda_trace, cpu_trace):
        numeric_field_count += 1
        metrics = _comparison_metrics(left, right, field)
        absolute, relative = metrics["absolute"], metrics["relative"]
        if absolute > maximum_absolute:
            maximum_absolute, maximum_field = absolute, field
        maximum_relative = max(maximum_relative, relative)
        if absolute > absolute_tolerance and relative > relative_tolerance:
            raise ValueError(f"CPU/CUDA parity tolerance exceeded at {field}")
    return {
        "schema": "gnss-doppler-lab.gcspo-stage0.round6-a5-parity.v1",
        "status": "PASS", "source_commit": verified["source_commit"],
        "run_count": 3, "backends": ["cuda", "cuda", "cpu"],
        "same_backend": "BYTE_IDENTICAL", "same_backend_files": names,
        "cpu_cuda": {
            "status": "WITHIN_PREREGISTERED_TOLERANCE",
            "selected_lambda_unchanged": True,
            "threshold_keys_unchanged": True,
            "numeric_field_count": numeric_field_count,
            "maximum_absolute_delta": maximum_absolute,
            "maximum_relative_delta": maximum_relative,
            "maximum_delta_field": maximum_field,
            "threshold_deltas": threshold_deltas,
            "tolerance": tolerance,
            "tolerance_status": "PRESERVED_UNCHANGED_FROM_ROUND4",
        },
    }


def _verify_evidence_manifest(artifact: Path, manifest_path: Path) -> dict:
    manifest = _load(manifest_path)
    if (manifest.get("schema") != "gnss-doppler-lab.gcspo-stage0.round6-evidence-manifest.v1" or
            manifest.get("source_commit") != SOURCE_COMMIT):
        raise ValueError("round-6 evidence manifest schema/source mismatch")
    expected = sorted(
        f"round6_provenance/{run_id}/{name}"
        for run_id in RUN_IDS for name in EVIDENCE_FILES
    )
    rows = manifest.get("files")
    if not isinstance(rows, list) or [row.get("path") for row in rows] != expected:
        raise ValueError("round-6 evidence manifest file set/order mismatch")
    for row in rows:
        if set(row) != {"path", "sha256", "size_bytes"}:
            raise ValueError("round-6 evidence identity row is malformed")
        if _identity(artifact / row["path"]) != {"sha256": row["sha256"], "size_bytes": row["size_bytes"]}:
            raise ValueError(f"round-6 packaged evidence hash mismatch: {row['path']}")
    return manifest


def verify_round6_a5(artifact_root: str | Path) -> dict:
    artifact = Path(artifact_root).resolve(strict=True)
    repo = artifact.parents[1]
    index = _load(artifact / "round6_a5_provenance.json")
    if index.get("schema") != "gnss-doppler-lab.gcspo-stage0.round6-a5-provenance-index.v1":
        raise ValueError("round-6 signed A5 provenance index schema mismatch")
    if index.get("source_commit") != SOURCE_COMMIT or index.get("challenge_path") != CHALLENGE_PATH:
        raise ValueError("round-6 signed A5 provenance source/challenge mismatch")
    relative_roots = index.get("evidence_roots")
    if relative_roots != [f"artifacts/gcspo_stage0_static_rerun/round6_provenance/{name}" for name in RUN_IDS]:
        raise ValueError("round-6 signed A5 evidence root set/order mismatch")
    roots = []
    for relative in relative_roots:
        path = (repo / relative).resolve(strict=True)
        if repo not in path.parents:
            raise ValueError("round-6 signed A5 evidence path escapes repository")
        roots.append(path)
    manifest_relative = index.get("evidence_manifest")
    manifest_path = (repo / str(manifest_relative)).resolve(strict=True)
    if repo not in manifest_path.parents:
        raise ValueError("round-6 evidence manifest path escapes repository")
    manifest = _verify_evidence_manifest(artifact, manifest_path)
    verified = verify_witnessed_runs(
        roots, repo_root=repo, source_commit=SOURCE_COMMIT,
        challenge_path=CHALLENGE_PATH,
    )
    parity = compare_round6_a5_runs(verified)
    parity_path = (repo / str(index.get("parity_report"))).resolve(strict=True)
    if repo not in parity_path.parents or _load(parity_path) != parity:
        raise ValueError("round-6 signed A5 parity report differs from reconstruction")
    return {"status": "PASS", "manifest": manifest, "witnessed": verified, "parity": parity}


def verify_round6_freeze(artifact_root: str | Path, expected_freeze_commit: str) -> dict:
    artifact = Path(artifact_root).resolve(strict=True)
    repo = artifact.parents[1]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                          text=True, capture_output=True).stdout.strip()
    if head != expected_freeze_commit:
        raise ValueError("round-6 review HEAD does not equal expected freeze commit")
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo,
                      check=True, text=True, capture_output=True).stdout:
        raise ValueError("round-6 review worktree is not clean")
    manifest_path = artifact / "round6_freeze_manifest.json"
    manifest = _load(manifest_path)
    if (manifest.get("schema") != "gnss-doppler-lab.gcspo-stage0.round6-packaging-freeze.v1" or
            manifest.get("state") != "READY_FOR_SINGLE_INDEPENDENT_READ_ONLY_REVIEW" or
            manifest.get("source_commit") != SOURCE_COMMIT or
            manifest.get("manifest_excludes_self") is not True or
            manifest.get("protected_access_authorized") is not False):
        raise ValueError("round-6 packaging freeze contract mismatch")
    committed_manifest = subprocess.run(
        ["git", "show", f"{expected_freeze_commit}:artifacts/gcspo_stage0_static_rerun/round6_freeze_manifest.json"],
        cwd=repo, check=True, capture_output=True,
    ).stdout
    if manifest_path.read_bytes() != committed_manifest:
        raise ValueError("round-6 freeze manifest is not from expected commit")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("round-6 packaging freeze file set is empty")
    paths = [row.get("path") for row in rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("round-6 packaging freeze paths are not sorted/unique")
    for row in rows:
        path = (repo / str(row.get("path"))).resolve(strict=True)
        if repo not in path.parents or _identity(path) != {
                "sha256": row.get("sha256"), "size_bytes": row.get("size_bytes")}:
            raise ValueError(f"round-6 freeze hash mismatch: {row.get('path')}")
        committed = subprocess.run(["git", "show", f"{expected_freeze_commit}:{row['path']}"],
                                   cwd=repo, check=True, capture_output=True).stdout
        if path.read_bytes() != committed:
            raise ValueError(f"round-6 freeze file is not from expected commit: {row['path']}")
    result = verify_round6_a5(artifact)
    return {
        "status": "PASS", "freeze_commit": expected_freeze_commit,
        "source_commit": SOURCE_COMMIT,
        "evidence_file_count": len(result["manifest"]["files"]),
        "run_count": len(result["witnessed"]["runs"]),
        "independence": result["witnessed"]["independence"],
        "parity": result["parity"],
    }
