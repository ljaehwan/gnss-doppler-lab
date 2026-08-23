from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from gnss_doppler_lab.jammertest_crpa_stage0b import (
    circular_shift_batch,
    mismatch_batch,
    mismatch_source_map,
    phase_randomize_batch,
)
from gnss_doppler_lab.jammertest_crpa_stage0c import VERDICTS
from gnss_doppler_lab.jammertest_crpa_stage0c_execution import (
    fit_fixed_logistic,
    numerical_invariance_grid,
    verify_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/jammertest2025_crpa_stage0c_spatial_discrimination"


def read_json(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_global_gain_phase_invariance_all_16_combinations() -> None:
    rng = np.random.default_rng(20250823)
    incoherent = (
        rng.normal(size=(12, 4, 1024))
        + 1j * rng.normal(size=(12, 4, 1024))
    ).astype(np.complex128)
    waveform = rng.normal(size=(12, 1, 1024)) + 1j * rng.normal(size=(12, 1, 1024))
    steering = np.exp(1j * np.asarray([0.0, 0.3, 1.1, 2.0]))[None, :, None]
    coherent = (waveform * steering).astype(np.complex128)
    for values in (coherent, incoherent):
        result = numerical_invariance_grid(values)
        assert result["all_16_combinations_pass"] is True
        assert len(result["combinations"]) == 16


def test_destruction_controls_are_deterministic_and_preserve_contracts() -> None:
    rng = np.random.default_rng(11)
    values = (
        rng.normal(size=(8, 4, 1024))
        + 1j * rng.normal(size=(8, 4, 1024))
    ).astype(np.complex64)
    rows = [
        {"area": 1, "class_id": 4, "transmit_power_dbm": 30}
        for _ in range(len(values))
    ]
    mapping = mismatch_source_map(rows)
    positions = np.arange(len(values))
    mismatch = mismatch_batch(values, mapping, positions)
    assert mismatch.shape == values.shape
    assert not np.array_equal(mismatch, values)

    shifted_1 = circular_shift_batch(values, np.random.default_rng(12))
    shifted_2 = circular_shift_batch(values, np.random.default_rng(12))
    assert np.array_equal(shifted_1, shifted_2)
    assert np.allclose(np.mean(np.abs(shifted_1) ** 2, axis=-1), np.mean(np.abs(values) ** 2, axis=-1))

    randomized_1 = phase_randomize_batch(values, np.random.default_rng(13))
    randomized_2 = phase_randomize_batch(values, np.random.default_rng(13))
    assert np.array_equal(randomized_1, randomized_2)
    assert np.allclose(
        np.mean(np.abs(randomized_1) ** 2, axis=-1),
        np.mean(np.abs(values) ** 2, axis=-1),
        rtol=2e-6,
        atol=2e-6,
    )


def test_scaler_is_fit_on_training_values_only() -> None:
    train = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    labels = np.asarray([0, 0, 1, 1])
    held_out = np.asarray([[1000.0, 2000.0]])
    fitted = fit_fixed_logistic(train, labels)
    assert np.array_equal(fitted.scaler.mean_, train.mean(axis=0))
    assert not np.array_equal(fitted.scaler.mean_, np.vstack((train, held_out)).mean(axis=0))


def test_numerical_and_destruction_artifacts_are_complete() -> None:
    invariance = read_json("numerical_invariance.json")
    assert invariance["all_16_combinations_pass"] is True
    assert len(invariance["combinations"]) == 16
    destruction = read_json("destruction_classifier_results.json")
    assert set(destruction) == {"mismatched", "circular_shift", "fourier_phase_randomized"}
    for result in destruction.values():
        assert set(result) == {"retrained", "actual_trained_cross_apply"}
        assert set(result["retrained"]) == {"M2", "M3"}
        assert set(result["actual_trained_cross_apply"]) == {"M2", "M3"}


def test_oof_is_complete_unique_and_uses_frozen_threshold() -> None:
    with (ARTIFACT / "out_of_fold_predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    actual = [
        row for row in rows
        if row["experiment"] == "actual_oof" and row["view"] == "actual" and row["training_view"] == "actual"
    ]
    by_evaluation = {
        evaluation: [row for row in actual if row["evaluation"] == evaluation]
        for evaluation in ("primary", "sensitivity_a")
    }
    assert len(by_evaluation["primary"]) == 260 * 4
    assert len(by_evaluation["sensitivity_a"]) == 324 * 4
    keys = {(row["evaluation"], row["model"], row["sample_index"]) for row in actual}
    assert len(keys) == len(actual)
    assert all(row["final_role"] == "test" for row in actual)
    assert all(float(row["threshold"]) == 0.5 for row in rows)
    assert all(int(row["prediction"]) == (float(row["probability"]) >= 0.5) for row in rows)
    contract = read_json("feature_contract.json")
    assert contract["threshold"] == 0.5
    assert contract["threshold_source"] == "fixed before training; no test-label use"


def test_final_claims_and_access_are_restricted() -> None:
    verdict = read_json("final_verdict.json")
    assert verdict["verdict"] in VERDICTS
    assert verdict["recording_provenance_blocked"] is True
    assert verdict["clean_detector_success"] is False
    assert verdict["general_spoof_detector_success"] is False
    assert verdict["ready_for_wcl"] is False
    access = read_json("access_audit.json")
    assert access["redownloaded_bytes"] == 0
    assert access["copied_raw_bytes"] == 0
    assert access["raw_objects_opened"] == 1
    assert access["selected_snapshot_count"] == 3588
    assert all(access[name] == 0 for name in ("innosense_bytes", "texbat_bytes", "oakbat_bytes", "tuni_bytes"))


def test_committed_artifact_verifies() -> None:
    assert verify_artifact(ARTIFACT) == []


def test_manifest_detects_byte_flip(tmp_path: Path) -> None:
    copied = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, copied)
    readme = copied / "README.md"
    readme.write_bytes(readme.read_bytes() + b"X")
    assert "manifest hash mismatch: README.md" in verify_artifact(copied)
