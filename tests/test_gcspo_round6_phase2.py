"""Round-6 packaging/freeze tests; no protected path is inspected."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from gnss_doppler_lab import gcspo_round6_verify as round6_verify
from gnss_doppler_lab.gcspo_round6_verify import (
    SOURCE_COMMIT, compare_round6_a5_runs, verify_round6_a5,
)


ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "artifacts/gcspo_stage0_static_rerun"


@pytest.fixture(scope="module")
def witnessed_round6_runs():
    return verify_round6_a5(ARTIFACT)["witnessed"]


def test_packaged_round6_signed_chains_and_parity_reconstruct():
    result = verify_round6_a5(ARTIFACT)
    assert result["status"] == "PASS"
    assert result["witnessed"]["source_commit"] == SOURCE_COMMIT
    assert result["witnessed"]["independence"] == {
        "status": "EXTERNALLY_WITNESSED",
        "distinct": [
            "nonce", "PREPARED envelope", "PREPARED signature",
            "COMPLETED signature", "process identity", "process interval",
            "execution receipt",
        ],
    }
    assert [row["prepared"]["backend"] for row in result["witnessed"]["runs"]] == [
        "cuda", "cuda", "cpu",
    ]
    assert result["parity"]["same_backend"] == "BYTE_IDENTICAL"
    assert result["parity"]["cpu_cuda"]["status"] == "WITHIN_PREREGISTERED_TOLERANCE"


def test_round6_index_is_repository_relative_and_private_key_free():
    index = json.loads((ARTIFACT / "round6_a5_provenance.json").read_text())
    assert index["source_commit"] == SOURCE_COMMIT
    assert all(not Path(path).is_absolute() for path in index["evidence_roots"])
    packaged_names = {
        path.relative_to(ARTIFACT).as_posix()
        for path in (ARTIFACT / "round6_provenance").glob("**/*") if path.is_file()
    }
    assert not any("private" in name.lower() or name.endswith((".key", ".pem")) for name in packaged_names)
    assert not any("completion_observation.json" in name for name in packaged_names)


def test_round6_parity_fails_closed_on_tolerance_excess(monkeypatch):
    result = verify_round6_a5(ARTIFACT)
    verified = result["witnessed"]
    cpu = next(row for row in verified["runs"] if row["prepared"]["backend"] == "cpu")
    original = Path.read_text

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path == cpu["output_dir"] / "a5_numeric_trace.json":
            document = json.loads(text)
            document["thresholds"][next(iter(document["thresholds"]))] += 1.0
            return json.dumps(document)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="tolerance|mismatch"):
        compare_round6_a5_runs(verified)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_round6_json_loader_rejects_nonstandard_nonfinite_constants(tmp_path, constant):
    document = tmp_path / "synthetic.json"
    document.write_text('{"value":' + constant + '}')
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        round6_verify._load(document)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_round6_numeric_pair_rejects_nonfinite_compared_values(value):
    with pytest.raises(ValueError, match="non-finite numeric value"):
        list(round6_verify._numeric_pairs(value, 0.0, "synthetic.value"))


def test_round6_threshold_comparison_rejects_bool(monkeypatch, witnessed_round6_runs):
    cpu = next(
        row for row in witnessed_round6_runs["runs"]
        if row["prepared"]["backend"] == "cpu"
    )
    original = Path.read_text

    def changed(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path == cpu["output_dir"] / "clean_a5_report.json":
            document = json.loads(text)
            document["thresholds"][next(iter(document["thresholds"]))] = True
            return json.dumps(document)
        return text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError, match="bool|numeric"):
        compare_round6_a5_runs(witnessed_round6_runs)


def test_round6_comparison_rejects_overflow_derived_nonfinite():
    with pytest.raises(ValueError, match="derived delta"):
        round6_verify._comparison_metrics(1e308, -1e308, "synthetic.overflow")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("delta", float("nan")),
        ("absolute value", float("inf")),
        ("scale", float("-inf")),
        ("relative value", float("nan")),
    ],
)
def test_round6_rejects_each_nonfinite_derived_metric(name, value):
    with pytest.raises(ValueError, match=f"derived {name}"):
        round6_verify._finite_derived(value, name, "synthetic.derived")


def test_round6_comparison_metrics_are_strictly_finite():
    metrics = round6_verify._comparison_metrics(3.0, -1.0, "synthetic.finite")
    assert metrics == {"delta": 4.0, "absolute": 4.0, "scale": 3.0, "relative": 4.0 / 3.0}
    assert all(math.isfinite(value) for value in metrics.values())


def test_round5_unsigned_failure_remains_excluded_from_round6_freeze():
    audit = json.loads((ARTIFACT / "round6_audit_report.json").read_text())
    rejection = audit["round5_rejected_evidence"]
    assert rejection["status"] == "REJECTED_UNSIGNED_FAIL_CLOSED"
    assert rejection["retrospective_acceptance"] is False
    assert rejection["excluded_from_round6"] is True
    assert audit["protected"]["rows_opened"] == 0
    assert audit["protected"]["bytes_opened"] == 0
    assert audit["attack_run_count"] == 0
    assert audit["push_performed"] is False
