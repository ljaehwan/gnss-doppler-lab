from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
from gnss_doppler_lab import qset_stage0a_r2 as Q
from gnss_doppler_lab import qset_stage0a_r2_execution as E
from gnss_doppler_lab import qset_stage0a_r2_evaluation as V


def test_prompt_relative_normalization_is_phase_and_amplitude_invariant() -> None:
    rng = np.random.default_rng(3); taps = rng.normal(size=(200, 9)) + 1j * rng.normal(size=(200, 9)); taps[:, 4] += 4
    left, valid_left = Q.normalized_complex_taps(taps); right, valid_right = Q.normalized_complex_taps(taps * (7.0 * np.exp(1j * 1.2)))
    assert valid_left.all() and valid_right.all(); np.testing.assert_allclose(left, right, atol=1e-12)


def test_morphology_feature_has_frozen_shape_and_is_finite() -> None:
    rng = np.random.default_rng(7); taps = rng.normal(size=(250, 9)) + 1j * rng.normal(size=(250, 9)); taps[:, 4] += 5
    feature = Q.morphology_feature(taps); assert feature.shape == (len(Q.FEATURE_NAMES),); assert np.isfinite(feature).all()


def test_global_robust_model_and_clipped_score() -> None:
    rng = np.random.default_rng(9); values = rng.normal(size=(100, len(Q.FEATURE_NAMES))); model = Q.fit_robust_model(values); scores = Q.local_scores(values, model)
    assert len(scores) == 100 and np.isfinite(scores).all(); assert "prn" not in model


def test_dynamic_panel_is_not_fixed_to_ten_slots() -> None:
    model = {"median": [0.0] * len(Q.FEATURE_NAMES), "scale": [1.0] * len(Q.FEATURE_NAMES)}
    rows = [{"feature": np.ones(len(Q.FEATURE_NAMES)) * prn, "scenario": "x", "prn": prn, "window_start_s": 0, "window_end_s": 1, "epoch_count": 250, "cn0_median": 40.0, "lock_median": 1.0} for prn in (2, 5, 11, 14, 30, 36)]
    windows = Q.dynamic_windows(rows, model); assert len(windows) == 1; assert windows[0]["prns"] == [2, 5, 11, 14, 30, 36]


def test_mean_dilution_is_exact_k_over_m() -> None:
    reference = {"median": [0.0] * 4, "scale": [1.0] * 4}; base = np.arange(10, dtype=float); shifted = base.copy(); shifted[:3] += 2
    before = Q.aggregate_scores(base, reference); after = Q.aggregate_scores(shifted, reference); assert after["MEAN"] - before["MEAN"] == pytest.approx(0.6)


def test_causal_persistence_and_strict_threshold() -> None:
    continuous, warm = Q.persistence([0, 2, 2, 0], [1, 2, 3, 4]); assert warm.tolist() == [True, True, False, False]; assert continuous[2] == 2 and continuous[3] == 2
    calibrated = Q.calibrate_threshold([0, 1, 2, 3], [1, 2, 3, 4]); assert calibrated["comparison"] == ">"


def test_normalized_pauc_uses_supported_numpy_api() -> None:
    assert Q.normalized_pauc([0.0, 0.1, 0.2, 0.3], [0.7, 0.8, 0.9], 0.01) == pytest.approx(1.0)


def test_manifest_tamper_detection(tmp_path: Path) -> None:
    target = tmp_path / "x"; target.write_bytes(b"before"); manifest = E.output_manifest(tmp_path); target.write_bytes(b"after")
    with pytest.raises(Q.QSetError): E.verify_manifest(tmp_path, manifest)


def test_frozen_file_binding_tamper_detection(tmp_path: Path) -> None:
    target = tmp_path / "feature.npz"
    target.write_bytes(b"frozen")
    binding = {"size_bytes": target.stat().st_size, "sha256": Q.sha256_file(target)}
    V.verify_file_binding(target, binding, "synthetic feature")
    target.write_bytes(b"tampered")
    with pytest.raises(Q.QSetError):
        V.verify_file_binding(target, binding, "synthetic feature")


def test_receiver_config_and_patch_are_frozen() -> None:
    assert Q.sha256_file(Q.CONFIG_TEMPLATE) == Q.CONFIG_SHA256
    text = Q.GALILEO_PATCH.read_text(); assert "pilot-tracked Galileo 1B" in text and "tap_count=9" in text
    config = E.render_config(Path("/tmp/in"), Path("/tmp/trace_"), 100, "C-1"); assert "Tracking_1B.tap_count=9" in config and "SignalSource.enable_terminal_drain=true" in config


def test_access_audit_tamper_fails_compact_verifier() -> None:
    spec = importlib.util.spec_from_file_location("verify_qset_r2", ROOT / "scripts/verify_qset_gnss_stage0a_r2.py"); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    audit = {"status": "PASS", "attack_access_after_freeze_only": True, "attack_payload": {"allowlisted_scenarios": ["SS-1", "SS-3", "SS-5", "SS-11"]}, "unallowlisted_tuni2025_raw": {key: 0 for key in ("stats", "hashes", "opens", "mmaps", "bytes_read")}}
    module.validate_access(audit); changed = copy.deepcopy(audit); changed["unallowlisted_tuni2025_raw"]["bytes_read"] = 1
    with pytest.raises(module.VerificationError): module.validate_access(changed)


def test_preregistration_forbids_neural_and_shortcut_inputs() -> None:
    prereg = json.loads((Q.ARTIFACT / "preregistration.json").read_text()); feature = prereg["feature_contract"]
    assert feature["supervised_or_neural_model"] is False; assert "PRN identity" in feature["excluded_inputs"]; assert "absolute C/N0" in feature["excluded_inputs"]
