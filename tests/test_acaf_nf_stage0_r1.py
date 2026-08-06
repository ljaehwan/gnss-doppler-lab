import numpy as np
import pytest
from gnss_doppler_lab.acaf_nf_stage0_r1 import (
    GPS_CA_CHIP_RATE_HZ, ca_code, sampled_replica, caf_complex_grid,
    TrackerCenter, complex_coordinate_indices, normalize_complex_caf,
    select_complex_coordinates, tracker_center_from_row,
)


def test_prn1_first_ten_chips_match_is_gps_200_reference():
    # IS-GPS-200 PRN 1 C/A sequence, 0 -> +1 and 1 -> -1 mapping.
    # Frozen reference prefix for this ICD-conformant G2-stage convention.
    assert ca_code(1)[:10].tolist() == [-1, -1, 1, 1, -1, 1, 1, 1, 1, 1]


def test_one_ms_replica_advances_one_complete_ca_period_at_25_msps():
    fs = 25_000_000
    code = sampled_replica(3, fs, fs // 1000, code_phase_chips=0.0, code_rate_hz=GPS_CA_CHIP_RATE_HZ)
    # sample just before the final chip boundary belongs to chip 1022, not chip 0/1.
    assert code[-1] == ca_code(3)[1022]
    assert len(np.unique(np.floor(np.arange(fs // 1000) * GPS_CA_CHIP_RATE_HZ / fs))) == 1023


def test_complex_caf_recovers_synthetic_center():
    fs = 1_000_000
    n = fs // 1000
    x = sampled_replica(6, fs, n, code_phase_chips=0.25, code_rate_hz=GPS_CA_CHIP_RATE_HZ)
    t = np.arange(n) / fs
    x = x * np.exp(2j * np.pi * 100.0 * t)
    c = caf_complex_grid(x, TrackerCenter(6, 0.25, 100.0, GPS_CA_CHIP_RATE_HZ, 0), fs,
                         np.array([-0.125, 0.0, 0.125]), np.array([-50.0, 0.0, 50.0]))
    assert np.unravel_index(np.abs(c).argmax(), c.shape) == (1, 1)


def test_gain_and_global_phase_do_not_change_normalized_caf():
    fs = 1_000_000
    x = sampled_replica(3, fs, fs // 1000, code_phase_chips=0.0, code_rate_hz=GPS_CA_CHIP_RATE_HZ)
    center = TrackerCenter(3, 0.0, 0.0, GPS_CA_CHIP_RATE_HZ, 0)
    a, _ = normalize_complex_caf(caf_complex_grid(x, center, fs, np.array([0.0]), np.array([0.0])))
    b, _ = normalize_complex_caf(caf_complex_grid(2.0j * x, center, fs, np.array([0.0]), np.array([0.0])))
    assert np.allclose(a, b, atol=1e-10)


def test_query_budget_is_complex_coordinate_not_real_imag_scalar():
    field_shape = (1, 10)
    idx = select_complex_coordinates(np.arange(40.0).reshape(4, 10), field_shape, 9)
    assert len(idx) == 9
    assert len(set(idx)) == 9
    assert all(0 <= i < field_shape[0] * field_shape[1] for i in idx)


def test_tracker_center_requires_all_requested_lineage_fields():
    row = {"PRN": 3, "PRN_start_sample_count": 25000, "carrier_doppler_hz": 100.0,
           "code_freq_chips": GPS_CA_CHIP_RATE_HZ, "aux1": 0.125}
    c = tracker_center_from_row(row, 25_000_000)
    assert c.prn == 3 and c.sample_count == 25000 and c.code_phase_chips == pytest.approx(0.125)
    with pytest.raises(ValueError):
        tracker_center_from_row({"PRN": 3}, 25_000_000)
