from __future__ import annotations

import numpy as np
import pytest

from gnss_doppler_lab.acaf_nf_stage1_r2a_l20_foundation_audit import (
    DELAY_GRID_CHIPS,
    DOPPLER_GRID_HZ,
    State,
    clean_only_guard,
    complex_caf_surface,
    numerical_equivalence,
    r14_l20_aggregate,
    r2_l20_aggregate,
    same_assignment,
    score_power_surface,
    support_deltas_are_causal,
)


def _state(row: int, *, prn: int = 3, start: int | None = None) -> State:
    return State(
        channel=2,
        prn=prn,
        tracker_row=row,
        state_row=row - 1,
        raw_start_sample=(row - 1) * 25_000 if start is None else start,
        code_freq_chips=1_023_000.0,
        carrier_doppler_hz=700.0,
        aux1=0.0,
        prompt_i=1.0,
        prompt_q=0.0,
        cn0_db_hz=40.0,
        carrier_lock=0.95,
    )


def test_r14_r2_l20_numerical_equivalence_at_1e_12():
    rng = np.random.default_rng(20260808)
    surfaces = rng.normal(size=(20, 11, 17)) + 1j * rng.normal(size=(20, 11, 17))
    report = numerical_equivalence(surfaces, 1e-12)
    assert report["status"] == "PASS"
    assert report["aggregate_max_abs_delta"] <= 1e-12
    assert np.allclose(r14_l20_aggregate(surfaces), r2_l20_aggregate(surfaces), atol=1e-12, rtol=0)


def test_real_imag_magnitude_and_power_are_not_interchangeable():
    surfaces = np.ones((20, 11, 17), dtype=np.complex128)
    surfaces[:, 4, 7] = 10j
    aggregate = r2_l20_aggregate(surfaces)
    assert np.argmax(aggregate) == np.ravel_multi_index((4, 7), aggregate.shape)
    assert not np.array_equal(np.mean(surfaces.real, axis=0), aggregate)
    assert np.all(aggregate >= 0)


def test_doppler_wipeoff_sign_recovers_known_residual():
    state = _state(10)
    n = 25_000
    t = np.arange(n) / 25_000_000.0
    # The raw tone has tracker Doppler + 100 Hz.  carrier_sign=-1 removes it.
    code_state = State(**{**state.__dict__, "carrier_doppler_hz": 700.0})
    from gnss_doppler_lab.acaf_nf_stage0_r13_reconstruction import code_replica

    code = code_replica(3, n, 25_000_000.0, 1_023_000.0, 0.0, -1, 0.0, replica_direction=1)[0]
    iq = code * np.exp(1j * 2 * np.pi * 800.0 * t)
    correct = np.abs(complex_caf_surface(iq, code_state, carrier_sign=-1)) ** 2
    score = score_power_surface(correct)
    assert score["peak_doppler_offset_hz"] == 100.0
    wrong = np.abs(complex_caf_surface(iq, code_state, carrier_sign=1)) ** 2
    assert score_power_surface(wrong)["peak_doppler_offset_hz"] != 100.0


def test_k_kminus1_sample_counter_contract_and_reassignment_exclusion():
    states = [_state(i) for i in range(20, 40)]
    assert same_assignment(states)
    assert states[0].state_row == states[0].tracker_row - 1
    assert states[0].raw_start_sample == states[0].state_row * 25_000
    reassigned = list(states)
    reassigned[10] = _state(30, prn=8)
    assert not same_assignment(reassigned)


def test_contiguous_causal_l20_and_variable_support_lengths():
    starts_25k = [i * 25_000 for i in range(20)]
    starts_24999 = [i * 24_999 for i in range(20)]
    assert support_deltas_are_causal(starts_25k)
    assert not support_deltas_are_causal(starts_24999)
    assert support_deltas_are_causal(starts_24999, allow_r14_overlap=True)
    broken = list(starts_25k)
    broken[11] += 25_000
    assert not support_deltas_are_causal(broken)


def test_clean_only_and_no_threshold_or_model_fitting_contract():
    clean_only_guard("cleanStatic", ["/authenticated/cleanStatic.bin"])
    for scenario in ("ds3", "ds4", "ds7", "ds8"):
        with pytest.raises(ValueError):
            clean_only_guard(scenario)
        with pytest.raises(ValueError):
            clean_only_guard("cleanStatic", [f"/raw/{scenario}.bin"])


def test_full_normal_streaming_selection_is_deterministic():
    rng = np.random.default_rng(7)
    surfaces = rng.normal(size=(20, len(DOPPLER_GRID_HZ), len(DELAY_GRID_CHIPS))) + 1j * rng.normal(
        size=(20, len(DOPPLER_GRID_HZ), len(DELAY_GRID_CHIPS))
    )
    first = numerical_equivalence(surfaces)
    second = numerical_equivalence(surfaces.copy())
    assert first == second
