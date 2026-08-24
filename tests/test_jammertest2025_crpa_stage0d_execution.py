from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from gnss_doppler_lab.jammertest_crpa_stage0b import circular_shift_batch, phase_randomize_batch
from gnss_doppler_lab.jammertest_crpa_stage0d import MODEL_NAMES
from gnss_doppler_lab.jammertest_crpa_stage0d_execution import (
    PowerResidualizer,
    compute_features,
    feature_error,
    fit_pipeline,
    verify_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/jammertest2025_crpa_stage0d_true_spoof_discrimination"


def read_json(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_train_only_scaler_and_residualizer() -> None:
    rng = np.random.default_rng(20250823)
    values = (rng.normal(size=(8, 4, 1024)) + 1j * rng.normal(size=(8, 4, 1024))).astype(np.complex128)
    power = np.asarray([-3.0, -2.0, -1.0, 0.0, 50.0, 60.0, 70.0, 80.0])
    labels = np.asarray([0, 0, 1, 1])
    features = compute_features(values, power)
    train = np.arange(4)
    pipeline = fit_pipeline("M2R", features, power, train, labels)
    assert pipeline.residualizer.power_mean == np.mean(power[train])
    assert pipeline.residualizer.power_mean != np.mean(power)
    train_residual = pipeline.residualizer.transform(power[train], features.m2[train])
    assert np.allclose(pipeline.scaler.mean_, train_residual.mean(axis=0))


def test_residualizer_does_not_depend_on_test_values() -> None:
    train_power = np.asarray([-2.0, -1.0, 0.0, 1.0])
    train_values = np.column_stack((train_power, train_power ** 2))
    first = PowerResidualizer.fit(train_power, train_values)
    second = PowerResidualizer.fit(train_power, train_values)
    test_power = np.asarray([100.0, 200.0])
    test_values = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(first.transform(test_power, test_values), second.transform(test_power, test_values))


def test_m2_permutation_and_gain_phase_invariance_synthetic() -> None:
    rng = np.random.default_rng(23)
    waveform = rng.normal(size=(10, 1, 1024)) + 1j * rng.normal(size=(10, 1, 1024))
    steering = np.exp(1j * np.asarray([0.0, 0.2, 0.9, 1.7]))[None, :, None]
    coherent = (waveform * steering).astype(np.complex128)
    incoherent = (rng.normal(size=(10, 4, 1024)) + 1j * rng.normal(size=(10, 4, 1024))).astype(np.complex128)
    for values in (coherent, incoherent):
        before = compute_features(values).m2
        assert feature_error(before, compute_features(values[:, [2, 0, 3, 1], :]).m2)["allclose_pass"]
        for gain in (0.1, 1.0, 3.7, 10.0):
            for phase in (0.0, 0.37, 1.123, 2.9):
                after = compute_features(values * (gain * np.exp(1j * phase))).m2
                assert feature_error(before, after)["allclose_pass"]


def test_destruction_controls_are_reproducible() -> None:
    rng = np.random.default_rng(9)
    values = (rng.normal(size=(6, 4, 1024)) + 1j * rng.normal(size=(6, 4, 1024))).astype(np.complex64)
    shifted_a = circular_shift_batch(values, np.random.default_rng(10))
    shifted_b = circular_shift_batch(values, np.random.default_rng(10))
    randomized_a = phase_randomize_batch(values, np.random.default_rng(11))
    randomized_b = phase_randomize_batch(values, np.random.default_rng(11))
    assert np.array_equal(shifted_a, shifted_b)
    assert np.array_equal(randomized_a, randomized_b)


def test_actual_oof_predictions_are_complete_and_threshold_frozen() -> None:
    with (ARTIFACT / "out_of_fold_predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    actual = [row for row in rows if row["experiment"] == "actual_oof"]
    for model in MODEL_NAMES:
        selected = [row for row in actual if row["model"] == model]
        indices = [int(row["sample_index"]) for row in selected]
        assert len(indices) == 288
        assert len(set(indices)) == 288
        assert sum(row["class_name"] == "Spoof" for row in selected) == 124
        assert sum(row["class_name"] == "Prn" for row in selected) == 164
    assert all(float(row["threshold"]) == 0.5 for row in rows)
    assert all(int(row["prediction"]) == int(float(row["probability"]) >= 0.5) for row in rows)


def test_power_gate_prevents_track_b_scoring() -> None:
    matched = read_json("matched_metrics.json")
    sensitivity = read_json("caliper_sensitivity.json")
    assert matched["track_b_executed"] is False
    assert all(value is None for value in matched["matched_models"].values())
    assert sensitivity["spatial_scoring_executed"] is False
    assert all(row["test_pair_count"] == 0 for row in sensitivity["calipers"].values())


def test_invariance_and_final_claim_contract() -> None:
    assert read_json("invariance_results.json")["all_contracts_pass"] is True
    verdict = read_json("final_verdict.json")
    assert verdict["verdict"] == "SPOOF_EVALUATION_INVALID_NO_RECEIVED_POWER_OVERLAP"
    assert verdict["track_b_executed"] is False
    assert verdict["ready_for_wcl"] is False
    assert verdict["recording_independent_generalization"] is False


def test_committed_artifact_verifies() -> None:
    assert verify_artifact(ARTIFACT) == []


def test_manifest_detects_byte_flip(tmp_path: Path) -> None:
    copied = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, copied)
    path = copied / "final_verdict.json"
    path.write_bytes(path.read_bytes() + b"X")
    assert "manifest hash mismatch: final_verdict.json" in verify_artifact(copied)
