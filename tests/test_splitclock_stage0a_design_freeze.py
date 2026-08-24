from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnss_doppler_lab.splitclock_stage0a import (
    GALILEO_E1_WAVELENGTH_M,
    REQUIRED_OBSERVABLES,
    carrier_phase_radians_to_increment_m,
    doppler_hz_to_range_rate_mps,
    inject_secondary_clock,
    score_window,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/splitclock_stage0a_clean_identifiability"


def synthetic_panel(seed: int = 7, prns: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(10, dtype=float)
    panel = np.empty((10, prns, 3), dtype=float)
    common = np.column_stack((0.2 * time, np.full(10, 0.2), np.full(10, 0.2)))
    panel[:] = common[:, None, :]
    panel += rng.normal(scale=np.asarray([0.02, 0.005, 0.005]), size=panel.shape)
    return panel


def test_design_freeze_has_zero_data_or_score_access() -> None:
    design = json.loads((ARTIFACT / "design_freeze.json").read_text())
    assert design["status"] == "PRE_RAW_DESIGN_FREEZE"
    assert design["raw_feature_bytes_read"] == 0
    assert design["score_operations"] == 0
    assert set(design["observable_contract"]["required"]) == set(REQUIRED_OBSERVABLES)
    assert all(value == 0 for value in design["attack_access"].values())


def test_receiver_contract_is_reference_only_v3() -> None:
    receiver = json.loads((ARTIFACT / "receiver_source_binding.json").read_text())
    assert receiver["reference_only"] is True
    assert receiver["merge_performed"] is False
    assert receiver["qset_feature_threshold_score_reused"] is False
    assert receiver["v3_parameters"] == {
        "coherent_integration_ms": 8,
        "concurrent_acquisition_channels": 12,
        "pfa": "0.00001",
    }


def test_sign_and_unit_conversion() -> None:
    assert np.isclose(GALILEO_E1_WAVELENGTH_M, 0.190293672798365)
    doppler = np.asarray([1.0, -2.0])
    assert np.allclose(doppler_hz_to_range_rate_mps(doppler), -GALILEO_E1_WAVELENGTH_M * doppler)
    phase = np.asarray([0.0, 2.0 * np.pi, 4.0 * np.pi])
    assert np.allclose(carrier_phase_radians_to_increment_m(phase), -GALILEO_E1_WAVELENGTH_M)


def test_k1_normal_and_all_prn_clock_absorption() -> None:
    clean = synthetic_panel()
    all_prns = np.arange(clean.shape[1])
    common = inject_secondary_clock(clean, all_prns, d0_m=10, velocity_mps=0.5, acceleration_mps2=0)
    clean_score = score_window(clean).score
    common_score = score_window(common).score
    assert np.isfinite(clean_score)
    assert abs(common_score - clean_score) < 2.0


def test_k2_secondary_clock_recovery_and_coherent_injection() -> None:
    clean = synthetic_panel()
    subset = np.asarray([0, 2, 5])
    injected = inject_secondary_clock(clean, subset, d0_m=0, velocity_mps=0.5, acceleration_mps2=0)
    result = score_window(injected)
    recovered = result.labels == result.labels[subset[0]]
    assert result.score > score_window(clean).score
    assert np.sum(recovered[subset]) == len(subset)
    assert np.sum(recovered) in (len(subset), clean.shape[1] - len(subset))


def test_prn_permutation_and_label_switching_invariance() -> None:
    panel = inject_secondary_clock(synthetic_panel(), np.asarray([0, 2, 5]), d0_m=0, velocity_mps=0.5, acceleration_mps2=0)
    permutation = np.asarray([6, 2, 0, 7, 4, 5, 1, 3])
    first = score_window(panel)
    second = score_window(panel[:, permutation])
    assert np.isclose(first.score, second.score, rtol=1e-10, atol=1e-10)
    unpermuted = np.empty_like(second.labels)
    unpermuted[permutation] = second.labels
    assert np.array_equal(first.labels, unpermuted) or np.array_equal(first.labels, 1 - unpermuted)


def test_temporal_membership_destruction_reduces_score() -> None:
    clean = synthetic_panel()
    persistent = inject_secondary_clock(clean, np.asarray([0, 1, 2]), d0_m=0, velocity_mps=0.5, acceleration_mps2=0)
    destroyed = clean.copy()
    for epoch in range(10):
        subset = np.asarray([(epoch + offset) % 8 for offset in range(3)])
        one = inject_secondary_clock(clean[epoch : epoch + 1], subset, d0_m=0, velocity_mps=0.5 * epoch, acceleration_mps2=0)
        destroyed[epoch] = one[0]
    assert score_window(persistent).score > score_window(destroyed).score


def test_dynamic_panel_and_cycle_slip_mask_fail_closed() -> None:
    panel = synthetic_panel(prns=4)
    with np.testing.assert_raises(ValueError):
        score_window(panel)
    panel = synthetic_panel()
    mask = np.ones_like(panel, dtype=bool)
    mask[4, 1] = False
    with np.testing.assert_raises(ValueError):
        score_window(panel, mask)


def test_split_threshold_and_verdict_are_frozen_without_attack_tuning() -> None:
    design = json.loads((ARTIFACT / "design_freeze.json").read_text())
    assert design["model"]["window_seconds"] == 10
    assert design["model"]["fit_fraction"] == 0.7
    assert design["threshold"]["quantile"] == 0.99
    assert design["threshold"]["method"] == "higher"
    assert design["threshold"]["positive_control_tuning"] is False
    assert design["threshold"]["persistence_consecutive_exceedances"] == 3
    assert design["verdict_precedence"][0:2] == ["observable_unavailable", "panel_unsupported"]
