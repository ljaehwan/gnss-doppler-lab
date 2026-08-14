"""Round-8 public-entry regressions for the binding Round-7 rejection."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

import pytest

from gnss_doppler_lab import gcspo_round6_verify as round6_verify


ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "artifacts/gcspo_stage0_static_rerun"
ROUND7_FREEZE = "b54383b799f47fd1a849126d3f21fe6c643eb209"
TRACE_SCHEMA = "gnss-doppler-lab.gcspo-stage0.a5-numeric-trace.v1"


@pytest.fixture(scope="module")
def witnessed_runs():
    return round6_verify.verify_round6_a5(ARTIFACT)["witnessed"]


def _compare_synthetic(monkeypatch, witnessed_runs, cuda_trace, cpu_trace):
    cuda_paths = {
        row["output_dir"] / "a5_numeric_trace.json"
        for row in witnessed_runs["runs"] if row["prepared"]["backend"] == "cuda"
    }
    cpu_path = next(
        row["output_dir"] / "a5_numeric_trace.json"
        for row in witnessed_runs["runs"] if row["prepared"]["backend"] == "cpu"
    )
    original = Path.read_text

    def with_backend(document, backend):
        if isinstance(document, dict):
            return {**document, "backend": backend}
        return document

    def changed(path, *args, **kwargs):
        if path in cuda_paths:
            return json.dumps(with_backend(cuda_trace, "cuda"))
        if path == cpu_path:
            return json.dumps(with_backend(cpu_trace, "cpu"))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", changed)
    return round6_verify.compare_round6_a5_runs(witnessed_runs)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (True, True),
        (False, False),
        (True, 1),
        (1, True),
        (False, 0),
        (0, False),
        ({"nested": True}, {"nested": True}),
        ([1, True, 2], [1, True, 2]),
        ({"items": [False]}, {"items": [False]}),
    ],
)
def test_public_parity_rejects_boolean_numeric_trace_leaves(
        monkeypatch, witnessed_runs, left, right):
    with pytest.raises(ValueError, match="bool|boolean"):
        _compare_synthetic(
            monkeypatch, witnessed_runs,
            {"schema": TRACE_SCHEMA, "payload": left},
            {"schema": TRACE_SCHEMA, "payload": right},
        )


@pytest.mark.parametrize(
    "payload",
    [{}, [], {"nested": {}}, {"nested": []}, [{"nested": []}]],
)
def test_public_parity_rejects_empty_or_nested_empty_trace_documents(
        monkeypatch, witnessed_runs, payload):
    with pytest.raises(ValueError, match="empty|coverage|count|schema|object"):
        _compare_synthetic(monkeypatch, witnessed_runs, payload, payload)


def test_public_parity_rejects_both_sides_omitting_same_required_numeric_field(
        monkeypatch, witnessed_runs):
    omitted = {"schema": TRACE_SCHEMA, "calibration": [{"availability_s": 1.0}]}
    with pytest.raises(ValueError, match="coverage|count|path|schema"):
        _compare_synthetic(monkeypatch, witnessed_runs, omitted, omitted)


def test_public_parity_rejects_unexpected_numeric_field_coverage(
        monkeypatch, witnessed_runs):
    unexpected = {"schema": TRACE_SCHEMA, "unexpected_numeric": 1.0}
    with pytest.raises(ValueError, match="coverage|count|path|schema"):
        _compare_synthetic(monkeypatch, witnessed_runs, unexpected, unexpected)


def test_public_parity_rejects_matching_unapproved_nonnumeric_metadata(
        monkeypatch, witnessed_runs):
    unexpected = {"schema": TRACE_SCHEMA, "label": "same-on-both-sides", "value": 1.0}
    with pytest.raises(ValueError, match="nonnumeric|schema|coverage"):
        _compare_synthetic(monkeypatch, witnessed_runs, unexpected, unexpected)


def test_public_parity_rejects_numeric_path_change_with_same_field_count(
        monkeypatch, witnessed_runs):
    original = Path.read_text

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.name == "a5_numeric_trace.json" and "round6_provenance" in path.parts:
            assert '"thresholds":' in text
            return text.replace('"thresholds":', '"thresholds_changed":', 1)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="path/schema digest"):
        round6_verify.compare_round6_a5_runs(witnessed_runs)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_public_strict_loader_rejects_nonfinite_constants(constant):
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        round6_verify.strict_json_loads('{"value":' + constant + '}', label="synthetic")


def test_public_parity_rejects_huge_integer_numeric_coercion(
        monkeypatch, witnessed_runs):
    huge = 10 ** 400
    trace = {"schema": TRACE_SCHEMA, "value": huge}
    with pytest.raises(ValueError, match="not finite|not a number"):
        _compare_synthetic(monkeypatch, witnessed_runs, trace, trace)


def test_public_parity_rejects_subtraction_overflow(monkeypatch, witnessed_runs):
    with pytest.raises(ValueError, match="derived delta"):
        _compare_synthetic(
            monkeypatch, witnessed_runs,
            {"schema": TRACE_SCHEMA, "value": 1e308},
            {"schema": TRACE_SCHEMA, "value": -1e308},
        )


def _duplicate_schema_member(text: str) -> str:
    match = re.search(r'"schema"\s*:\s*"[^"]+"', text)
    assert match is not None
    opening = text.index("{")
    return text[:opening + 1] + match.group(0) + "," + text[opening + 1:]


@pytest.mark.parametrize(
    "relative",
    [
        "round6_a5_provenance.json",
        "round6_evidence_manifest.json",
        "round6_a5_parity.json",
    ],
)
def test_public_package_verifier_rejects_duplicate_package_metadata(
        monkeypatch, relative):
    target = ARTIFACT / relative
    original = Path.read_text

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        return _duplicate_schema_member(text) if path == target else text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="duplicate"):
        round6_verify.verify_round6_a5(ARTIFACT)


@pytest.mark.parametrize(
    "suffix",
    [
        "bundle/clean_a5_report.json",
        "bundle/a5_numeric_trace.json",
        "bundle/a5_backend_truth.json",
    ],
)
def test_public_package_verifier_rejects_duplicate_scientific_or_backend_json(
        monkeypatch, suffix):
    original = Path.read_text

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        return _duplicate_schema_member(text) if path.as_posix().endswith(suffix) else text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="duplicate"):
        round6_verify.verify_round6_a5(ARTIFACT)


def test_public_package_verifier_rejects_duplicate_challenge_json(monkeypatch):
    from gnss_doppler_lab import gcspo_witness

    challenge = ARTIFACT / "a5_round6_challenges.json"
    duplicate = _duplicate_schema_member(challenge.read_text()).encode()
    original_git = gcspo_witness._git
    original_bytes = Path.read_bytes
    original_text = Path.read_text

    def changed_git(repo, *arguments, binary=False):
        if arguments == ("show", f"{round6_verify.SOURCE_COMMIT}:{round6_verify.CHALLENGE_PATH}"):
            return duplicate if binary else duplicate.decode().strip()
        return original_git(repo, *arguments, binary=binary)

    def changed_bytes(path, *args, **kwargs):
        return duplicate if path == challenge else original_bytes(path, *args, **kwargs)

    def changed_text(path, *args, **kwargs):
        return duplicate.decode() if path == challenge else original_text(path, *args, **kwargs)

    monkeypatch.setattr(gcspo_witness, "_git", changed_git)
    monkeypatch.setattr(Path, "read_bytes", changed_bytes)
    monkeypatch.setattr(Path, "read_text", changed_text)
    with pytest.raises(ValueError, match="duplicate"):
        round6_verify.verify_round6_a5(ARTIFACT)


def test_public_strict_loader_rejects_duplicate_nested_key():
    malformed = '{"outer":{"value":1,"value":1}}'
    with pytest.raises(ValueError, match="duplicate"):
        round6_verify.strict_json_loads(malformed, label="nested synthetic package metadata")


@pytest.mark.parametrize(
    "relative",
    [
        "round6_audit_report.json",
        "round6_freeze_manifest.json",
        "round6_phase1_handoff.json",
        "round6_phase2_review_handoff.json",
        "round7_audit_report.json",
        "round7_freeze_manifest.json",
        "round7_evidence_manifest.json",
        "round7_repair_evidence.json",
        "round7_repair_review_handoff.json",
        "round6_independent_review_rejection.json",
    ],
)
def test_public_freeze_verifier_rejects_duplicate_freeze_audit_handoff_json(
        monkeypatch, relative):
    target = ARTIFACT / relative
    original = Path.read_text
    real_run = subprocess.run

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        return _duplicate_schema_member(text) if path == target else text

    def clean_status(arguments, *args, **kwargs):
        if arguments == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(arguments, 0, stdout=ROUND7_FREEZE + "\n", stderr="")
        if arguments[:3] == ["git", "status", "--porcelain"]:
            return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")
        return real_run(arguments, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", changed)
    monkeypatch.setattr(subprocess, "run", clean_status)
    with pytest.raises(ValueError, match="duplicate"):
        round6_verify.verify_round6_freeze(ARTIFACT, ROUND7_FREEZE)


def test_public_freeze_verifier_rejects_duplicate_historical_manifest(monkeypatch):
    real_run = subprocess.run
    historical = (
        f"{round6_verify.REJECTED_FREEZE_COMMIT}:"
        "artifacts/gcspo_stage0_static_rerun/round6_freeze_manifest.json"
    )

    def changed_run(arguments, *args, **kwargs):
        if arguments == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(arguments, 0, stdout=ROUND7_FREEZE + "\n", stderr="")
        if arguments[:3] == ["git", "status", "--porcelain"]:
            return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")
        result = real_run(arguments, *args, **kwargs)
        if arguments == ["git", "show", historical]:
            duplicate = _duplicate_schema_member(result.stdout)
            return subprocess.CompletedProcess(arguments, 0, stdout=duplicate, stderr=result.stderr)
        return result

    monkeypatch.setattr(subprocess, "run", changed_run)
    with pytest.raises(ValueError, match="duplicate"):
        round6_verify.verify_round6_freeze(ARTIFACT, ROUND7_FREEZE)
