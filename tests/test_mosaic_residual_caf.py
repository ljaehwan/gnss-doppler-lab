import numpy as np

from gnss_doppler_lab.mosaic_iq_injector import InjectionTheta, inject_counterfeit, sampled_prn_replica
from gnss_doppler_lab.mosaic_parameter_recovery import delay_direction_accuracy, estimate_from_residual_caf, median_abs_error
from gnss_doppler_lab.mosaic_residual_caf import h0_residual, residual_caf


def test_h0_residual_removes_single_source_and_variable_prn_count():
    fs = 1_023_000
    r1 = sampled_prn_replica(1, fs, 1023)
    r2 = sampled_prn_replica(2, fs, 1023)
    y = 2 * r1 - 0.5j * r2
    residual, alpha = h0_residual(y, np.column_stack([r1, r2]))
    assert alpha.shape == (2,)
    assert np.linalg.norm(residual) / np.linalg.norm(y) < 1e-6


def test_residual_caf_recovers_injected_delay_and_doppler_on_synthetic_unit_window():
    fs = 1_023_000
    clean = sampled_prn_replica(3, fs, 1023)
    theta = InjectionTheta(4.0, 1000.0, -3.0, 0.0)
    y, _ = inject_counterfeit(clean, 3, fs, theta, nav_bits=np.ones(1023))
    residual, _ = h0_residual(y, clean)
    delays = np.array([-4.0, 0.0, 4.0])
    dopplers = np.array([-1000.0, 0.0, 1000.0])
    caf = residual_caf(residual, 3, fs, delays, dopplers, nav_bits=np.ones(1023))
    est = estimate_from_residual_caf(caf)
    assert est.delay_chips == 4.0
    assert est.doppler_hz == 1000.0
    assert est.observable


def test_recovery_metrics_helpers():
    assert delay_direction_accuracy(np.array([-1, 1]), np.array([-0.5, 0.2])) == 1.0
    assert median_abs_error(np.array([0.01, 0.2]), np.array([0.0, 0.15]), minimum_abs=0.1) == 0.05000000000000002


def test_prn_permutation_invariance_of_joint_residual():
    fs = 1_023_000
    reps = [sampled_prn_replica(p, fs, 1023) for p in (1, 2, 3)]
    y = reps[0] + 2 * reps[1] - reps[2]
    r_a, _ = h0_residual(y, np.column_stack(reps))
    r_b, _ = h0_residual(y, np.column_stack([reps[2], reps[0], reps[1]]))
    assert np.linalg.norm(r_a - r_b) < 1e-8
