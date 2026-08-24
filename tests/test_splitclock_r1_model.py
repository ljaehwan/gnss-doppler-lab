from __future__ import annotations

import struct
from datetime import datetime, timedelta, timezone

import numpy as np

from gnss_doppler_lab.splitclock_r1_geometry import Ephemeris, geodetic_to_ecef, satellite_state, trace_cadence
from gnss_doppler_lab.splitclock_r1_model import inject_clock, score_window


SCALES = np.asarray([0.1, 0.05, 0.05])
PROCESS = np.asarray([0.5, 0.1])


def panel(seed=4):
    rng = np.random.default_rng(seed); time = np.arange(10)[:, None]; values = np.zeros((10, 6, 3))
    values[:, :, 0] = 0.2 * time + rng.normal(0, 0.02, (10, 6))
    values[:, :, 1] = 0.2 + rng.normal(0, 0.01, (10, 6))
    values[:, :, 2] = 0.2 + rng.normal(0, 0.01, (10, 6))
    return values, np.ones_like(values, dtype=bool)


def two_clock():
    values, valid = panel(); time = np.arange(10)[:, None]
    values[:, 3:, 0] += 0.8 * time; values[:, 3:, 1:] += 0.8
    return values, valid


def test_k1_data_does_not_reward_k2_after_mdl():
    values, valid = panel(); result = score_window(values, valid, SCALES, PROCESS)
    assert result.score < 0


def test_clear_two_clock_increases_score_with_soft_membership():
    baseline = score_window(*panel(), SCALES, PROCESS); result = score_window(*two_clock(), SCALES, PROCESS)
    assert result.score > baseline.score + 20
    assert np.all((result.memberships > 0) & (result.memberships < 1))
    assert np.isclose(sum(result.cluster_masses), np.sum(result.eligible))
    assert min(result.cluster_masses) >= 2


def test_prn_permutation_invariance():
    values, valid = two_clock(); permutation = np.asarray([2, 5, 0, 4, 1, 3])
    left = score_window(values, valid, SCALES, PROCESS); right = score_window(values[:, permutation], valid[:, permutation], SCALES, PROCESS)
    assert abs(left.score - right.score) <= 1e-10


def test_dynamic_panel_and_missing_modalities():
    values, valid = two_clock(); valid[2, 0, 2] = False; values[4, 1, 1] = np.nan; valid[8, 5, 2] = False
    result = score_window(values, valid, SCALES, PROCESS)
    assert result.n_valid == int(np.sum(result.evaluation_mask))
    assert result.n_valid < 3 * 6 * 3


def test_heldout_never_changes_fit_or_restart_selection():
    values, valid = two_clock(); left = score_window(values, valid, SCALES, PROCESS)
    mutated = values.copy(); mutated[7:] += 100.0
    right = score_window(mutated, valid, SCALES, PROCESS)
    assert left.fit_digest == right.fit_digest
    assert left.selected_restart == right.selected_restart
    assert np.array_equal(left.memberships, right.memberships)


def test_deterministic_reproduction():
    values, valid = two_clock(); left = score_window(values, valid, SCALES, PROCESS); right = score_window(values, valid, SCALES, PROCESS)
    assert left.score == right.score and left.fit_digest == right.fit_digest
    assert np.array_equal(left.memberships, right.memberships)


def test_all_prn_coherent_path_is_not_k2_evidence():
    values, valid = panel(); coherent = inject_clock(values, np.arange(6), 3, 10.0, 0.5, 0.0)
    assert score_window(coherent, valid, SCALES, PROCESS).score < 0


def test_d0_transition_is_applied_once_to_carrier_increment():
    values = np.zeros((10, 6, 3)); injected = inject_clock(values, np.asarray([0, 1, 2]), 3, 10.0, 0.1, 0.0)
    assert np.allclose(injected[3, :3, 2], 10.0)
    assert np.allclose(injected[4:, :3, 2], 0.1)
    assert np.allclose(injected[3:, :3, 0], 10.0 + 0.1 * np.arange(7)[:, None])


def test_single_modality_ablations_are_supported():
    values, valid = two_clock()
    for modalities in ((0,), (1,), (2,)):
        assert np.isfinite(score_window(values, valid, SCALES, PROCESS, modalities=modalities).score)


def test_synthetic_broadcast_geometry_is_finite():
    toc = datetime(2025, 1, 1, tzinfo=timezone.utc)
    eph = Ephemeris(1, toc, 1e-4, 1e-12, 0.0, 10.0, 1e-9, 0.1, 1e-6, 0.001, 1e-6, 5440.0, 345600.0, 1e-8, 1.0, 1e-8, 0.96, 100.0, 0.5, -5e-9, 1e-10, 2347, 0.0, 1e-9, 2e-9)
    position, clock, eccentric = satellite_state(eph, toc + timedelta(seconds=10))
    assert np.isfinite(position).all() and 2e7 < np.linalg.norm(position) < 4e7
    assert np.isfinite(clock) and np.isfinite(eccentric)
    assert np.allclose(geodetic_to_ecef(0, 0, 0), [6378137.0, 0.0, 0.0])


def test_trace_cadence_is_parsed_not_injected(tmp_path):
    header = struct.Struct("<8sIIIIdff64s48s9fI")
    payload = header.pack(b"TRC1MS02", 2, 192, 416, 0x01020304, 4e6, 0.125, 0.004, b"C-1", b"0" * 40, *([-0.5] * 9), 0)
    for channel in range(12): (tmp_path / f"trace_native_1ms_ch_{channel}.bin").write_bytes(payload)
    result = trace_cadence(tmp_path)
    assert np.isclose(result["native_trace_cadence_ms"], 4.0, atol=1e-6)
    assert result["native_trace_nominal_cadence_ms"] == 4
    assert result["acquisition_coherent_integration_ms"] == 8
    assert result["semantic_separation"]
