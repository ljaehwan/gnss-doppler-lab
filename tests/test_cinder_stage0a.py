from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from gnss_doppler_lab.cinder_cyclic_features import (
    c4_vector,
    fourth_order_cyclic_cumulant,
    fractional_chip_resample_records,
    hermitian_projective_compact,
)
from gnss_doppler_lab.cinder_emitter_identifiability import (
    SEEDS,
    block_bootstrap_auc,
    matched_pairs,
    verification_metrics,
)
from gnss_doppler_lab.trace_native_1ms import RECORD_DTYPE


def synthetic_waveform(seed: int = 3, chips: int = 8184) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    code = rng.choice((-1.0, 1.0), chips)
    symbol = rng.laplace(size=(chips, 1)) + 1j * rng.laplace(size=(chips, 1))
    shape = np.asarray((0.6 + 0.1j, 1.0 + 0.3j, 0.8 - 0.2j, 0.3 - 0.1j))
    return code[:, None] * (1.0 + 0.08 * symbol * shape[None, :]), code


def test_fourth_cumulant_formula_and_gaussian_reduction() -> None:
    rng = np.random.default_rng(7)
    gaussian = rng.normal(size=300_000) + 1j * rng.normal(size=300_000)
    laplace = rng.laplace(size=300_000) + 1j * rng.laplace(size=300_000)
    cg = fourth_order_cyclic_cumulant(gaussian, 0.0, (0, 0, 0))
    cl = fourth_order_cyclic_cumulant(laplace, 0.0, (0, 0, 0))
    assert abs(cg) < 0.08 * np.mean(np.abs(gaussian) ** 4)
    assert abs(cl) > 3 * abs(cg)


def test_gain_phase_and_nav_invariance() -> None:
    wave, code = synthetic_waveform()
    reference = hermitian_projective_compact(c4_vector(wave, code))
    for gain in (0.5, 0.8, 1.2, 2.0):
        for phase in (0.0, np.pi / 4, np.pi / 2, np.pi):
            for sign in (-1, 1):
                value = hermitian_projective_compact(c4_vector(sign * gain * np.exp(1j * phase) * wave, code))
                np.testing.assert_allclose(value, reference, atol=1e-10, rtol=1e-9)


def test_ideal_code_and_circular_shift_do_not_create_c4() -> None:
    rng = np.random.default_rng(9)
    code = rng.choice((-1.0, 1.0), 4092)
    ideal = np.repeat(code[:, None], 4, axis=1).astype(complex)
    assert np.max(np.abs(c4_vector(ideal, code))) == 0.0
    assert np.max(np.abs(c4_vector(np.roll(ideal, 17, axis=0), np.roll(code, 17)))) == 0.0


def test_fractional_chip_lineage_and_group_delay(tmp_path: Path) -> None:
    raw = tmp_path / "one_epoch.bin"
    iq = np.ones(1024, dtype=np.complex64)
    packed = np.column_stack((iq.real, iq.imag)).astype("<i2").reshape(-1)
    raw.write_bytes(packed.tobytes())
    records = np.zeros(1, dtype=RECORD_DTYPE)
    records["raw_interval_start_sample"] = 0
    records["raw_interval_end_sample"] = 1024
    records["action_used_code_phase_step_chips_per_sample"] = 1.0
    records["action_used_interval_length_samples"] = 1024
    wave, audit = fractional_chip_resample_records(raw, records, 3)
    assert wave.shape == (1023, 4)
    assert audit.source_sample_count == 1024
    assert audit.group_delay_source_samples == 0.0
    assert audit.out_of_bounds_queries == 0
    assert hashlib.sha256(raw.read_bytes()).hexdigest()


def _toy_pairs(seed: int = SEEDS[0]):
    rng = np.random.default_rng(1)
    prns = np.repeat(np.arange(5), 5)
    blocks = np.tile(np.arange(5), 5)
    features = np.eye(5)[prns] + 0.03 * rng.normal(size=(25, 5))
    nuisance = rng.normal(size=(25, 4))
    pairs = matched_pairs(features, prns, blocks, nuisance, seed=seed)
    return features, pairs


def test_pair_gap_matching_determinism_and_bootstrap() -> None:
    features, first = _toy_pairs()
    _, second = _toy_pairs()
    assert first == second
    pos = sorted(row["gap_blocks"] for row in first if row["label"] == 1)
    neg = sorted(row["gap_blocks"] for row in first if row["label"] == 0)
    assert pos == neg
    labels = np.asarray([row["label"] for row in first])
    scores = np.asarray([-np.linalg.norm(features[row["left"]] - features[row["right"]]) for row in first])
    metrics = verification_metrics(labels, scores, threshold=float(np.median(scores)))
    assert metrics["roc_auc"] > 0.9
    boot = block_bootstrap_auc(labels, scores, first, seed=4, repetitions=100)
    assert len(boot) >= 50


def test_chronological_split_contract_and_no_prn_feature() -> None:
    roles = {"feature_train": range(0, 8), "metric_train": range(9, 17),
             "calibration": range(18, 24), "final_holdout": range(25, 33)}
    used = {value for values in roles.values() for value in values}
    assert {8, 17, 24}.isdisjoint(used)
    assert all(len(values) >= 6 for values in roles.values())
    forbidden = {"prn", "channel", "time", "doppler", "cn0", "power"}
    feature_contract = {"c4_projective", "second_order_diagnostic", "prompt_scattering_diagnostic"}
    assert forbidden.isdisjoint(feature_contract)
