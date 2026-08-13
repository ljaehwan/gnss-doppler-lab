"""Executable regressions for the third pre-attack GCSPO freeze repair.

All file fixtures are synthetic or committed clean/candidate metadata.  This
module never opens protected attack payload bytes and never claims an attempt.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "artifacts/gcspo_stage0_static_rerun"


def _ready_gate(tmp_path: Path):
    from gnss_doppler_lab.gcspo_access import AccessGate

    gate = AccessGate(tmp_path / "access_ledger.jsonl")
    gate.set_preflight(
        clean_only_pass=True,
        reviews_pass=True,
        freeze_sha="a" * 40,
        frozen_hashes={"config": "b" * 64},
    )
    gate.set_remote_sync(
        local_sha="a" * 40,
        remote_sha="a" * 40,
        ahead=0,
        behind=0,
        clean=True,
    )
    return gate


def _file_identity(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _register_receiver_manifest(gate, path: Path) -> None:
    sha256, size = _file_identity(path)
    gate.register_pinned(
        path,
        expected_sha256=sha256,
        expected_size=size,
        kind="RECEIVER_MANIFEST",
    )


def test_sidecar_size_mismatch_fails_before_claim_ledger_or_child_read(tmp_path, monkeypatch):
    import gnss_doppler_lab.gcspo_access as access

    manifest = tmp_path / "receiver-manifest.json"
    manifest.write_text("{}\n")
    child = tmp_path / "epl_tracking_ch_0.mat"
    child.write_bytes(b"synthetic-child")
    child_sha, child_size = _file_identity(child)
    sidecar = [{
        "canonical_path": str(child.resolve()),
        "sha256": child_sha,
        "size_bytes": child_size,
    }]
    gate = _ready_gate(tmp_path)
    _register_receiver_manifest(gate, manifest)
    child.write_bytes(b"short")

    real_open = access.os.open
    child_read_opens = []

    def observed_open(path, flags, *args, **kwargs):
        if Path(path) == child and flags & os.O_RDONLY == os.O_RDONLY:
            child_read_opens.append(str(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(access.os, "open", observed_open)
    with pytest.raises(ValueError, match="size|identity"):
        gate.register_sidecar_children(manifest, sidecar)

    assert child_read_opens == []
    assert not (tmp_path / "access_ledger.jsonl").exists()


def test_registered_child_replacement_is_rejected_preclaim_even_when_bytes_match(tmp_path):
    manifest = tmp_path / "receiver-manifest.json"
    manifest.write_text("{}\n")
    child = tmp_path / "epl_tracking_ch_0.mat"
    child.write_bytes(b"same-science-bytes")
    child_sha, child_size = _file_identity(child)
    gate = _ready_gate(tmp_path)
    _register_receiver_manifest(gate, manifest)
    gate.register_sidecar_children(manifest, [{
        "canonical_path": str(child.resolve()),
        "sha256": child_sha,
        "size_bytes": child_size,
    }])
    replacement = tmp_path / "replacement.mat"
    replacement.write_bytes(b"same-science-bytes")
    os.replace(replacement, child)
    called = False

    def consume(_handle):
        nonlocal called
        called = True

    with pytest.raises(RuntimeError, match="replaced|identity"):
        gate.consume(
            child,
            scenario="DS3",
            phase="transition",
            purpose="synthetic race",
            consumer=consume,
        )
    assert called is False
    assert not (tmp_path / "access_ledger.jsonl").exists()


def test_child_toctou_between_precheck_and_open_closes_claim_without_science(tmp_path, monkeypatch):
    import gnss_doppler_lab.gcspo_access as access

    manifest = tmp_path / "receiver-manifest.json"
    manifest.write_text("{}\n")
    child = tmp_path / "epl_tracking_ch_0.mat"
    child.write_bytes(b"original-science")
    child_sha, child_size = _file_identity(child)
    replacement = tmp_path / "replacement.mat"
    replacement.write_bytes(b"original-science")
    gate = _ready_gate(tmp_path)
    _register_receiver_manifest(gate, manifest)
    gate.register_sidecar_children(manifest, [{
        "canonical_path": str(child.resolve()),
        "sha256": child_sha,
        "size_bytes": child_size,
    }])
    real_open = access.os.open
    raced = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal raced
        if Path(path) == child and not raced:
            raced = True
            os.replace(replacement, child)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(access.os, "open", racing_open)
    called = False

    def consume(_handle):
        nonlocal called
        called = True

    with pytest.raises(RuntimeError, match="replaced|identity"):
        gate.consume(
            child,
            scenario="DS7",
            phase="transition",
            purpose="synthetic TOCTOU",
            consumer=consume,
        )
    assert called is False
    records = [json.loads(line) for line in (tmp_path / "access_ledger.jsonl").read_text().splitlines()]
    assert [row["record_type"] for row in records] == ["PRE", "POST"]
    assert records[-1]["outcome"] == "PATH_REPLACED"
    assert all(row.get("outcome") != "SUCCESS" for row in records)


def _full_row(*, phase="transition", start=1.0, prns=(1, 2, 3, 4)):
    epochs = tuple(range(50))
    support = tuple((epoch, tuple(prns)) for epoch in epochs)
    return {
        "phase": phase,
        "window_start_s": start,
        "availability_s": start + 1.0,
        "score": 9.0,
        "prns": list(prns),
        "epoch_ids": epochs,
        "epoch_prn_support": support,
    }


def _b0_rows(*, phase="transition", start=1.0, prns=(1, 2, 3, 4), score=True):
    epochs = tuple(range(50))
    rows = []
    for prn in prns:
        row = {
            "phase": phase,
            "window_start_s": start,
            "availability_s": start + 1.0,
            "prn": prn,
            "epoch_ids": epochs,
            "epoch_prn_support": tuple((epoch, (prn,)) for epoch in epochs),
        }
        if score:
            row["event_score"] = float(prn)
        rows.append(row)
    return rows


def test_protected_b0_rejects_empty_partial_duplicate_missing_and_unsupported_support():
    from gnss_doppler_lab.gcspo_evaluate import integrate_protected_b0

    methods = {name: [_full_row()] for name in ("A1", "A2", "A3", "A4", "A5", "Full")}
    with pytest.raises(ValueError, match="empty|support"):
        integrate_protected_b0(methods, [], score_column="event_score")
    with pytest.raises(ValueError, match="exact|support"):
        integrate_protected_b0(methods, _b0_rows(prns=(1, 2, 3)), score_column="event_score")
    with pytest.raises(ValueError, match="duplicate"):
        integrate_protected_b0(methods, _b0_rows() + _b0_rows(prns=(1,)), score_column="event_score")
    with pytest.raises(ValueError, match="score"):
        integrate_protected_b0(methods, _b0_rows(score=False), score_column="event_score")
    with pytest.raises(ValueError, match="join|support"):
        integrate_protected_b0(
            methods,
            _b0_rows() + _b0_rows(start=2.0),
            score_column="event_score",
        )


def test_protected_b0_exact_support_succeeds_and_all_mandatory_methods_are_nonempty():
    from gnss_doppler_lab.gcspo_evaluate import (
        integrate_protected_b0,
        validate_protected_method_support,
    )

    methods = {name: [_full_row()] for name in ("A1", "A2", "A3", "A4", "A5", "Full")}
    integrated = integrate_protected_b0(methods, _b0_rows(), score_column="event_score")
    validation = validate_protected_method_support(
        integrated,
        required_phases=("transition",),
    )
    assert validation == {
        "methods": ["A0", "A1", "A2", "A3", "A4", "A5", "Full"],
        "phase_counts": {name: {"transition": 1} for name in
                         ("A0", "A1", "A2", "A3", "A4", "A5", "Full")},
    }
    assert integrated["A0"][0]["epoch_prn_support"] == _full_row()["epoch_prn_support"]


def test_protected_b0_builder_rejects_non_access_gate_even_with_synthetic_rows():
    from gnss_doppler_lab.gcspo_b0 import SAMPLE_RATE_HZ, TAP_FIELDS, build_protected_scheduled_node_table

    class FakeGate:
        def read_h5(self, *_args, **_kwargs):
            samples = np.asarray([0, SAMPLE_RATE_HZ // 4, SAMPLE_RATE_HZ // 2,
                                  3 * SAMPLE_RATE_HZ // 4], dtype=np.int64)
            values = {"PRN": np.full(4, 3), "PRN_start_sample_count": samples}
            values.update({name: np.ones(4) for name in TAP_FIELDS})
            return values

    with pytest.raises(TypeError, match="AccessGate"):
        build_protected_scheduled_node_table(
            [Path("synthetic.mat")],
            gate=FakeGate(),
            scenario="DS3",
            roles={"transition": (0.0, 2.0)},
        )


def _relation_row(*, scenario, method, score=10.0):
    epochs = tuple(range(50))
    support = tuple((epoch, (1, 2, 3, 4)) for epoch in epochs)
    return {
        "scenario": scenario,
        "phase": "transition",
        "method": method,
        "score": score,
        "window_start_s": 0.0,
        "availability_s": 1.0,
        "phase_start_s": 0.0,
        "phase_end_s": 10.0,
        "label": True,
        "prns": (1, 2, 3, 4),
        "epoch_ids": epochs,
        "epoch_prn_support": support,
    }


def test_reconstruction_rejects_missing_mandatory_ds7_relation_evidence():
    from gnss_doppler_lab.gcspo_verify_artifacts import reconstruct_relation_evidence

    rows = [
        _relation_row(scenario="DS3", method="Full"),
        _relation_row(scenario="DS3", method="PER_PRN_TEMPORAL_SHIFT", score=5.0),
    ]
    capabilities = {
        "available": {"DS3": {}, "DS7": {}},
        "unavailable": {
            "DS4": {"status": "LIMITED_TRANSITION_ONLY"},
            "DS8": {"status": "UNAVAILABLE"},
        },
    }
    with pytest.raises(ValueError, match="mandatory.*DS7|DS7.*mandatory"):
        reconstruct_relation_evidence({"rows": rows}, capabilities=capabilities)


def _invalid_relation_final_document():
    destruction = {
        "policy": {},
        "required_available_scenarios": ["DS3", "DS7"],
        "scenario_results": {
            "DS3": {"status": "AVAILABLE", "mandatory": True, "lcb": 1.0,
                    "median_relative_loss": 0.5},
            "DS4": {"status": "LIMITED_TRANSITION_ONLY", "mandatory": False},
            "DS7": {"status": "UNAVAILABLE", "mandatory": True,
                    "reason": "relation score-loss support is empty"},
            "DS8": {"status": "UNAVAILABLE", "mandatory": False},
        },
    }
    evidence = {
        "clean_holdout_fpr": 0.0,
        "external_pre_fpr": {"DS3": 0.0},
        "incremental_lcb": {"Full-A1": 1.0, "Full-A2": 1.0},
        "destruction": destruction,
        "persistence": {
            "DS3": {"ratio": 1.0, "delay_s": 0.0},
            "DS7_DS8": {"ratio": 1.0, "delay_s": 0.0},
        },
        "controls": [
            {"id": "CLOCK_DRIFT", "specificity_ratio": 0.0,
             "persistent_alarm_ratio": 0.0, "max_consecutive_alarms": 0},
            {"id": "OTHER", "specificity_ratio": 0.0,
             "persistent_alarm_ratio": 0.0, "max_consecutive_alarms": 0},
        ],
        "shared": {"full_pauc": 1.0, "a5_pauc": 0.5,
                   "full_median_edf": 1.0, "a5_median_edf": 2.0},
    }
    from gnss_doppler_lab.gcspo_statistics import compute_scientific_gates

    gates = compute_scientific_gates(evidence)
    return {
        "scientific_status": "VALID_SCIENCE",
        "protected_run_count": 1,
        "verdict": "NO_GO_PHYSICAL_HYPOTHESIS",
        "evidence": evidence,
        "gates": gates,
    }


def test_final_evidence_validation_cannot_package_missing_relation_as_valid_no_go():
    from gnss_doppler_lab.gcspo_verify_reconstruct import verify_evidence_document

    with pytest.raises(ValueError, match="mandatory.*relation|relation.*mandatory"):
        verify_evidence_document(_invalid_relation_final_document())


def _run_a5_backend(monkeypatch, *, cuda: bool, terms, smoothness: float):
    import torch
    import gnss_doppler_lab.gcspo_a5 as a5

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    segments = [{"prn": 1, "epoch_ids": (0,), "state_start": 0, "state_stop": 2}]
    return a5.a5_spectral_scores(terms, segments, (smoothness,))[0]


def _assert_a5_equal(first, second):
    np.testing.assert_allclose(first["state"], second["state"], rtol=1e-11, atol=1e-11)
    for key in ("rss", "score", "gcv", "effective_dof"):
        assert first[key] == pytest.approx(second[key], rel=1e-11, abs=1e-11)
    assert first["rank"] == second["rank"]


def test_a5_exact_reviewer_negative_mode_cpu_cuda_parity(monkeypatch):
    import gnss_doppler_lab.gcspo_a5 as a5

    monkeypatch.setattr(
        a5,
        "a5_segment_prior_precision",
        lambda *_args, **_kwargs: np.eye(2, dtype=float) * 1.0e8,
    )
    terms = (np.diag([-0.09, 1.0e8]), np.asarray([1.0, 0.0]), 100.0, 20)
    cpu = _run_a5_backend(monkeypatch, cuda=False, terms=terms, smoothness=1.0e-9)
    cuda = _run_a5_backend(monkeypatch, cuda=True, terms=terms, smoothness=1.0e-9)

    np.testing.assert_allclose(cpu["state"], [10.0, 0.0], rtol=1e-11, atol=1e-11)
    assert cpu["rss"] == pytest.approx(71.0, rel=1e-11, abs=1e-11)
    assert cpu["score"] == pytest.approx(26.0042677294, rel=1e-10)
    _assert_a5_equal(cpu, cuda)


@pytest.mark.parametrize("small_eigenvalue", [-5.0e-16, 0.0, 5.0e-16, 5.0e-10])
def test_a5_cpu_cuda_near_zero_and_rank_boundary_parity(monkeypatch, small_eigenvalue):
    terms = (
        np.diag([small_eigenvalue, 1.0]),
        np.asarray([small_eigenvalue, 0.25]),
        2.0,
        20,
    )
    cpu = _run_a5_backend(monkeypatch, cuda=False, terms=terms, smoothness=1.0)
    cuda = _run_a5_backend(monkeypatch, cuda=True, terms=terms, smoothness=1.0)
    _assert_a5_equal(cpu, cuda)


def test_a5_materially_negative_rss_fails_closed_on_both_backends(monkeypatch):
    terms = (np.eye(2), np.asarray([2.0, 0.0]), 1.0, 20)
    with pytest.raises(ValueError, match="negative RSS"):
        _run_a5_backend(monkeypatch, cuda=False, terms=terms, smoothness=1.0)
    with pytest.raises(ValueError, match="negative RSS"):
        _run_a5_backend(monkeypatch, cuda=True, terms=terms, smoothness=1.0)


@pytest.mark.parametrize("offset", [-1.0e-15, 1.0e-15])
def test_a5_scale_aware_roundoff_rss_is_nonnegative_and_backend_identical(monkeypatch, offset):
    smoothness = 1.0e-8
    state = 1.0 / (1.0 + smoothness)
    yty = 2.0 * state - state * state + offset
    terms = (np.eye(2), np.asarray([1.0, 0.0]), yty, 20)
    cpu = _run_a5_backend(monkeypatch, cuda=False, terms=terms, smoothness=smoothness)
    cuda = _run_a5_backend(monkeypatch, cuda=True, terms=terms, smoothness=smoothness)
    assert cpu["rss"] >= 0.0
    _assert_a5_equal(cpu, cuda)


def test_reproduction_evidence_authenticates_two_distinct_durable_executions():
    from gnss_doppler_lab.gcspo_verify import verify_reproduction_manifests

    result = verify_reproduction_manifests(ARTIFACT)
    assert result["status"] == "PASS"
    assert len(set(result["run_ids"])) == 2
    assert len(set(result["scratch_roots"])) == 2
    assert len(set(result["bundle_paths"])) == 2
    assert result["comparison"] == "BYTE_IDENTICAL_OUTPUT_SNAPSHOTS"
    assert result["canonical_bound_to_both_runs"] is True


@pytest.mark.parametrize("path", [
    ".pytest_cache/",
    "tests/__pycache__/",
    "src/gnss_doppler_lab/cache.pyc",
    "iq.bin.synthetic.tmp",
])
def test_runtime_hygiene_marks_all_generated_residue_relevant(path):
    from gnss_doppler_lab.gcspo_freeze import _runtime_relevant_ignored

    assert _runtime_relevant_ignored(f"!! {path}\n") == [f"!! {path}"]
