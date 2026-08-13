"""Focused regressions for the Stage-0 static rerun pre-freeze contract.

Every fixture is synthetic or reads only repository source/preregistration bytes.
No protected receiver payload is opened by this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).parents[1]
LEGACY_ROOT = ROOT / "artifacts/gcspo_stage0_static"
RERUN_ROOT = ROOT / "artifacts/gcspo_stage0_static_rerun"
LEGACY_HASHES = {
    "README.md": "cb24096d8160e0ebea1e78aec416ba3abdd678940d2d829ebf2c4ad351bda8df",
    "config.json": "919353cbf66230df506a9eb672d366dc61450b6637003f470939c0d3c91ee30e",
    "data_inventory.json": "4faffaede28119f7655da25b44129b09e76f1bb49ec5169861b6336abaea3631",
    "preregistration.json": "2390fddb2048db9c333dbb9d7a7bae1c3a174fa59a144902cf9743ad21501a03",
    "source_commit.json": "38215e854859dd18816625d089bac5ff8d1e7378882abdd42d70ce19c8e895d3",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rerun_artifact_and_branch_isolation_keeps_legacy_tree_immutable():
    assert {name: _sha(LEGACY_ROOT / name) for name in LEGACY_HASHES} == LEGACY_HASHES
    assert all((RERUN_ROOT / name).is_file() for name in LEGACY_HASHES)
    runner = (ROOT / "scripts/run_gcspo_stage0.py").read_text()
    verifier = (ROOT / "scripts/verify_gcspo_stage0.py").read_text()
    finalizer = (ROOT / "scripts/finalize_gcspo_clean.py").read_text()
    fresh = (ROOT / "scripts/verify_gcspo_fresh_clone.py").read_text()
    for source in (runner, verifier, finalizer, fresh):
        assert "artifacts/gcspo_stage0_static\"" not in source
        assert "gcspo_stage0_static_rerun" in source
    assert "research/gcspo-stage0-static-rerun" in runner
    assert "research/gcspo-stage0-phase2-implementation" not in runner


def _write_clean_fixture(root: Path, guard_value: float):
    from gnss_doppler_lab.gcspo_clean import Q_FIELDS, SAMPLE_RATE_HZ

    raw = root / "raw"
    raw.mkdir(parents=True)
    seconds = np.asarray([30.0, 60.0, 100.0, 139.0, 140.0, 150.0, 209.0])
    samples = np.rint(seconds * SAMPLE_RATE_HZ).astype(np.int64)
    amplitude = np.asarray([1.0, 1.2, 1.4, 1.6, guard_value, guard_value, guard_value])
    for channel in range(11):
        with h5py.File(raw / f"epl_tracking_ch_{channel}.mat", "w") as handle:
            handle["PRN_start_sample_count"] = samples
            handle["PRN"] = np.full(len(samples), channel + 1)
            for field in Q_FIELDS:
                if field == "I_E":
                    handle[field] = amplitude
                elif field == "code_freq_chips":
                    handle[field] = np.full(len(samples), 1_023_000.0)
                else:
                    handle[field] = np.zeros(len(samples))


def test_normalization_epsilon_uses_only_train_fit_not_guard_or_validation(tmp_path):
    from gnss_doppler_lab.gcspo_clean import load_cleanstatic_mat

    first, second = tmp_path / "first", tmp_path / "second"
    _write_clean_fixture(first, 100.0)
    _write_clean_fixture(second, 1e-9)
    assert load_cleanstatic_mat(first, start_s=30, end_s=210).epsilons == load_cleanstatic_mat(
        second, start_s=30, end_s=210
    ).epsilons


def test_signed_epl_coordinates_and_equal_norm_directions_reach_full_observer():
    from gnss_doppler_lab.gcspo_clean import Q_FIELDS, signed_q
    from gnss_doppler_lab.gcspo_core import SharedVAR, Whitener, build_physical_loading, build_state_prior_precision
    from gnss_doppler_lab.gcspo_full import FULL_INPUT_COORDINATES, _score_terms, _window_normal_terms

    columns = {name: np.asarray([float(index + 1)]) for index, name in enumerate(Q_FIELDS)}
    base = signed_q(columns, epsilon=np.asarray([0.25]))[0]
    for coordinate, name in enumerate(Q_FIELDS[:6]):
        changed = dict(columns)
        changed[name] = -changed[name]
        flipped = signed_q(changed, epsilon=np.asarray([0.25]))[0]
        different = np.flatnonzero(~np.isclose(base, flipped, rtol=0, atol=0))
        assert different.tolist() == [coordinate]
    assert FULL_INPUT_COORDINATES == Q_FIELDS

    los = np.asarray([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3)

    class Geometry:
        def loading(self, epoch, prn):
            row = los[int(prn) - 1]
            return row, build_physical_loading(row, validated_rows={
                "code_error_chips", "pll_phase_error_cycles",
                "carrier_doppler_hz", "code_frequency_offset_chips_s",
            })

    epochs = tuple(range(50))
    prns = {epoch: [1, 2, 3, 4] for epoch in epochs}
    positive = {(epoch, prn): np.r_[np.zeros(6), 1.0, 0.0, 0.0, 0.0]
                for epoch in epochs for prn in prns[epoch]}
    negative = {key: -value for key, value in positive.items()}
    model = SharedVAR(np.zeros(10), np.zeros((1, 10, 10)))
    whitener = Whitener(np.zeros(10), np.eye(10), np.eye(10))
    args = (epochs, prns, Geometry(), model, whitener, np.zeros((10, 10)))
    plus = _score_terms(_window_normal_terms(args[0], args[1], positive, *args[2:]),
                        build_state_prior_precision(epoch_count=50, smoothness=1e-6))
    minus = _score_terms(_window_normal_terms(args[0], args[1], negative, *args[2:]),
                         build_state_prior_precision(epoch_count=50, smoothness=1e-6))
    assert np.linalg.norm(np.concatenate(list(positive.values()))) == pytest.approx(
        np.linalg.norm(np.concatenate(list(negative.values())))
    )
    assert plus["state"] == pytest.approx(-minus["state"], abs=1e-9)


def test_end_to_end_synthetic_recovery_is_fail_closed_for_sign_units_and_rank():
    from gnss_doppler_lab.gcspo_core import SharedVAR, Whitener
    from gnss_doppler_lab.gcspo_transfer import prove_synthetic_physical_recovery

    model = SharedVAR(np.zeros(10), np.zeros((1, 10, 10)))
    whitener = Whitener(np.zeros(10), np.eye(10), np.eye(10))
    passing = prove_synthetic_physical_recovery(model, whitener, np.zeros((10, 10)))
    assert passing["overall_status"] == "PASS"
    assert passing["var_transfer_application_count"] == 1
    assert passing["maximum_scaled_state_error"] <= passing["tolerance"]["maximum_scaled_state_error"]
    for kwargs in ({"source_sign": 1.0}, {"unit_scale": 1000.0}, {"degenerate_geometry": True}):
        assert prove_synthetic_physical_recovery(
            model, whitener, np.zeros((10, 10)), **kwargs
        )["overall_status"] == "FAIL"


def test_half_open_real_scorer_excludes_phase_endpoint():
    from gnss_doppler_lab.gcspo_clean import window_endpoints

    assert 150.0 not in window_endpoints(110.0, 150.0)
    assert 149.5 in window_endpoints(110.0, 150.0)


def test_scenario_relation_policy_and_single_class_score_loss_contrast():
    from gnss_doppler_lab.gcspo_statistics import RELATION_POLICY, paired_score_loss_bootstrap

    assert RELATION_POLICY["DS3"]["primary"] == "PER_PRN_TEMPORAL_SHIFT"
    assert RELATION_POLICY["DS7"]["primary"] == "PER_PRN_TEMPORAL_SHIFT"
    assert RELATION_POLICY["DS8"]["primary"] == "PER_PRN_TEMPORAL_SHIFT"
    assert RELATION_POLICY["DS4"]["primary"] == "LOS_SHUFFLE"
    assert RELATION_POLICY["DS4"]["requires_established"] is True
    rows = []
    support = tuple((epoch, (1, 2, 3, 4)) for epoch in range(50))
    for slot in range(12):
        common = {"scenario": "DS3", "phase": "established", "label": True,
                  "window_start_s": 195.0 + slot * .5, "availability_s": 196.0 + slot * .5,
                  "phase_start_s": 195.0, "phase_end_s": 260.0,
                  "epoch_ids": tuple(range(50)), "prns": (1, 2, 3, 4),
                  "epoch_prn_support": support}
        rows += [{**common, "method": "Full", "score": 10.0},
                 {**common, "method": "PER_PRN_TEMPORAL_SHIFT", "score": 5.0}]
    report = paired_score_loss_bootstrap(rows, "Full", "PER_PRN_TEMPORAL_SHIFT", replicates=2000, seed=23)
    assert report["lcb_95"] > 0 and report["median_relative_loss"] >= 0.25


def test_b0_protected_exact_support_adapter_executes_without_entering_full_design():
    from gnss_doppler_lab.gcspo_b0 import adapt_b0_exact_support

    b0 = pd.DataFrame({"window_start_s": [1.0] * 4, "prn": [1, 2, 3, 4],
                       "event_score": [0.1, 0.2, 0.3, 0.4]})
    full = pd.DataFrame({"window_start_s": [1.0], "prns": [[1, 2, 3, 4]], "score": [99.0]})
    rows = adapt_b0_exact_support(b0, full, score_column="event_score")
    assert rows == [{"window_start_s": 1.0, "prns": [1, 2, 3, 4], "score": 0.25}]
    assert "TAP_FIELDS" not in (ROOT / "src/gnss_doppler_lab/gcspo_full.py").read_text()


def test_capability_sidecar_schema_binds_source_children_and_timeline():
    from gnss_doppler_lab.gcspo_capabilities import validate_capability_sidecar

    document = {
        "schema": "gnss-doppler-lab.gcspo-stage0.capability-sidecar.v1",
        "scenario": "DS7", "purpose": "FULL_SIGNED_EPL_GEOMETRY",
        "producer": {"identity": "complex9", "source_sha256": "a" * 64},
        "root_manifest": {"canonical_path": "/sealed/manifest.json", "sha256": "b" * 64,
                          "size_bytes": 123, "adapter": "explicit-child-bindings-v1"},
        "children": [{"canonical_path": "/sealed/raw/epl_tracking_ch_0.mat", "sha256": "c" * 64,
                      "size_bytes": 456, "scenario": "DS7", "purpose": "signed_epl_tracking"}],
        "fields": [{"name": "I_E", "producer": "complex9", "source_sha256": "a" * 64,
                    "unit": "signed correlator accumulation", "sign_proof": "source-bound",
                    "cadence_s": 0.02, "role": "FULL_SIGNED_EPL", "direct_loading": "covariance"}],
        "timeline": {"official_document": {"identity": "TEXBAT", "sha256": "d" * 64},
                     "raw_iq": {"canonical_path": "/sealed/ds7.bin", "sha256": "e" * 64,
                                "size_bytes": 1000, "byte_zero": 0, "sample_rate_hz": 25_000_000},
                     "processed_sample_count": 500, "recording_relative_seconds": 0.00002,
                     "rx_time_s": 477900.00002, "nmea_time": "authenticated",
                     "gps_time": {"week": 2000, "tow_s": 477900.00002},
                     "onset_s": 110.0, "pull_off_s": 150.0},
    }
    assert validate_capability_sidecar(document)["status"] == "AVAILABLE"
    broken = json.loads(json.dumps(document)); broken["fields"][0].pop("sign_proof")
    with pytest.raises(ValueError, match="field|sign"):
        validate_capability_sidecar(broken)


def test_evaluator_has_no_hardcoded_validated_rows_or_placeholder_availability():
    source = (ROOT / "src/gnss_doppler_lab/gcspo_evaluate.py").read_text()
    assert "VALIDATED =" not in source
    assert "UNAVAILABLE_EXACT_ADAPTER" not in source
    assert 'manifest["files"]' not in source


def test_canonical_json_rejects_nan_and_clean_ready_requires_recovery_proof(tmp_path):
    from gnss_doppler_lab.gcspo_artifacts import canonical_write_json
    from gnss_doppler_lab.gcspo_verify import verify_clean_ready

    with pytest.raises(ValueError):
        canonical_write_json(tmp_path / "bad.json", {"score": float("nan")})
    (tmp_path / "clean_only_report.json").write_text(json.dumps({
        "run_status": "CLEAN_ONLY_PASS", "protected_attack_rows_read": False,
        "attack_access_count": 0, "all_methods": ["A0", "A1", "A2", "A3", "A4", "A5", "Full"],
        "deterministic_rerun": "PASS",
    }))
    (tmp_path / "preflight_report.json").write_text(json.dumps({
        "overall_status": "PASS", "attack_access_count": 0,
        "synthetic_physical_recovery": {"overall_status": "FAIL"},
    }))
    with pytest.raises(ValueError, match="recovery"):
        verify_clean_ready(tmp_path)
