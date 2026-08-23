from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from gnss_doppler_lab.jammertest_crpa_split_audit import assess_blocked_split_feasibility
from gnss_doppler_lab.jammertest_crpa_stage0b import (
    EXPECTED_BYTES,
    EXPECTED_SHA256,
    block_bootstrap_mean_difference,
    circular_shift_batch,
    compute_features,
    mismatch_batch,
    mismatch_source_map,
    open_crpa_memmap,
    phase_randomize_batch,
    verify_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/jammertest2025_crpa_stage0b_bounded_validation"


def coherent_data(seed: int = 7, snapshots: int = 24) -> np.ndarray:
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(snapshots, 1, 1_024)) + 1j * rng.normal(
        size=(snapshots, 1, 1_024)
    )
    phases = np.exp(1j * np.asarray([0.0, 0.3, -0.7, 1.1]))[None, :, None]
    return (source * phases).astype(np.complex64)


def test_read_only_npy_open_disallows_pickle(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.npy"
    np.save(path, coherent_data(snapshots=2), allow_pickle=False)
    array = open_crpa_memmap(path)
    assert isinstance(array, np.memmap)
    assert array.mode == "r"
    assert array.shape == (2, 4, 1_024)


def test_coherent_tuple_exceeds_shift_and_phase_destruction() -> None:
    x = coherent_data()
    rng = np.random.default_rng(11)
    actual = compute_features(x).mean_coherence.mean()
    shifted = compute_features(circular_shift_batch(x, rng)).mean_coherence.mean()
    randomized = compute_features(phase_randomize_batch(x, rng)).mean_coherence.mean()
    assert actual > 0.999
    assert actual - shifted > 0.8
    assert actual - randomized > 0.8


def test_destructors_preserve_channel_spectrum() -> None:
    x = coherent_data(snapshots=8)
    rng = np.random.default_rng(19)
    shifted = circular_shift_batch(x, rng)
    randomized = phase_randomize_batch(x, rng)
    expected = np.abs(np.fft.fft(x, axis=-1))
    assert np.allclose(np.abs(np.fft.fft(shifted, axis=-1)), expected, rtol=1e-5, atol=1e-5)
    assert np.allclose(np.abs(np.fft.fft(randomized, axis=-1)), expected, rtol=1e-5, atol=1e-5)


def test_global_gain_phase_preserves_primary_covariance_but_exposes_condition_instability() -> None:
    x = coherent_data().astype(np.complex128)
    before = compute_features(x)
    after = compute_features(x * (3.7 * np.exp(1j * 1.123)))
    assert np.allclose(before.eigenvalue_fractions, after.eigenvalue_fractions, rtol=0, atol=1e-12)
    assert np.allclose(before.coherences, after.coherences, rtol=0, atol=1e-12)
    assert np.allclose(before.mean_coherence, after.mean_coherence, rtol=0, atol=1e-12)
    assert not np.allclose(before.condition_number, after.condition_number, rtol=0, atol=1e-12)


def test_mismatch_is_within_stratum_permutation() -> None:
    x = coherent_data(snapshots=12)
    rows = [
        {"area": 1, "class_id": 4, "transmit_power_dbm": 25.0}
        for _ in range(len(x))
    ]
    mapping = mismatch_source_map(rows)
    positions = np.arange(len(x))
    mismatched = mismatch_batch(x, mapping, positions)
    for channel in range(4):
        assert sorted(mapping[:, channel].tolist()) == positions.tolist()
        assert np.array_equal(
            np.sort_complex(mismatched[:, channel, 0]),
            np.sort_complex(x[:, channel, 0]),
        )
        assert np.all(mapping[:, channel] != positions)


def test_block_bootstrap_detects_positive_paired_difference() -> None:
    indices = np.arange(4_096)
    actual = np.ones(len(indices))
    control = np.zeros(len(indices))
    result = block_bootstrap_mean_difference(
        actual, control, indices, 128, replicates=200
    )
    assert result["mean_difference"] == 1.0
    assert result["ci95_low"] == 1.0
    assert result["actual_significantly_higher"] is True


def test_split_audit_rejects_single_block_power_cell() -> None:
    rows = []
    for power in (15.0, 25.0, 30.0, 35.0, 40.0):
        rows.extend(
            {"sample_index": int(power) * 100 + offset, "area": 1,
             "transmit_power_dbm": power, "class_name": "Spoof", "class_id": 4}
            for offset in (0, 40)
        )
        rows.append(
            {"sample_index": int(power) * 100 + 10, "area": 1,
             "transmit_power_dbm": power, "class_name": "CW", "class_id": 0}
        )
    result = assess_blocked_split_feasibility(rows, 32)
    assert result["balanced_block_disjoint_split_feasible"] is False
    assert any(
        cell["binary_class"] == "negative"
        and not cell["necessary_train_test_condition_passed"]
        for cell in result["cells"]
    )


def test_committed_artifact_verifies() -> None:
    assert verify_artifact(ARTIFACT) == []


def test_final_verdict_contract() -> None:
    verdict = json.loads((ARTIFACT / "final_verdict.json").read_text())
    assert verdict["verdict"] == "INCONCLUSIVE_SPATIAL_SIGNAL_PROVENANCE_BLOCKED"
    assert verdict["classification_run"] is False
    assert verdict["ready_for_wcl_declared"] is False
    classifier = json.loads((ARTIFACT / "classifier_not_run.json").read_text())
    assert classifier["power_only_baseline"] == "NOT_RUN"
    assert classifier["snapshot_random_split_used"] is False


def test_access_and_download_binding() -> None:
    access = json.loads((ARTIFACT / "access_audit.json").read_text())
    integrity = json.loads((ARTIFACT / "download_integrity.json").read_text())
    assert access["downloaded_payload_bytes"] == EXPECTED_BYTES
    assert access["other_lfs_payload_bytes"] == 0
    assert access["innosense_hdf5_bytes"] == 0
    assert integrity["actual_sha256"] == EXPECTED_SHA256


def test_manifest_detects_byte_flip(tmp_path: Path) -> None:
    copied = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, copied)
    readme = copied / "README.md"
    readme.write_bytes(readme.read_bytes() + b"X")
    assert any("manifest hash mismatch: README.md" in error for error in verify_artifact(copied))


def test_download_hash_tamper_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, copied)
    path = copied / "download_integrity.json"
    value = json.loads(path.read_text())
    value["actual_sha256"] = "0" * 64
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    errors = verify_artifact(copied)
    assert "download hash binding mismatch" in errors


def test_forbidden_access_tamper_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, copied)
    path = copied / "access_audit.json"
    value = json.loads(path.read_text())
    value["innosense_hdf5_bytes"] = 1
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    assert "forbidden access nonzero: innosense_hdf5_bytes" in verify_artifact(copied)
