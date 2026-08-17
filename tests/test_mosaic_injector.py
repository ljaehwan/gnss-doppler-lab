import numpy as np
import pytest

from gnss_doppler_lab.mosaic_iq_injector import (
    InjectionTheta,
    inject_counterfeit,
    quantize_interleaved_int8,
    sampled_prn_replica,
)


def test_sample_rate_specific_code_stepping_and_1ms():
    replica = sampled_prn_replica(5, 5_000_000, 5000)
    assert replica.shape == (5000,)
    assert set(np.unique(replica.real)) == {-1.0, 1.0}


def test_delay_doppler_and_carrier_phase_signs_are_deterministic():
    fs = 1_023_000
    a = sampled_prn_replica(1, fs, 1023, code_phase_chips=0.0, doppler_hz=0.0)
    b = sampled_prn_replica(1, fs, 1023, code_phase_chips=1.0, doppler_hz=0.0)
    assert np.allclose(b[:-1], a[1:])
    c = sampled_prn_replica(1, fs, 8, doppler_hz=fs / 8, carrier_phase_rad=np.pi / 2)
    assert np.isclose(c[0] / sampled_prn_replica(1, fs, 1)[0], 1j)


def test_injection_identity_zero_amplitude_and_count_preservation():
    clean = np.ones(64, dtype=np.complex128) * (2 + 1j)
    theta = InjectionTheta(0.0, 0.0, -np.inf, 0.0)
    out, metrics = inject_counterfeit(clean, 1, 1_023_000, theta, nav_bits=np.ones(clean.size))
    assert out.shape == clean.shape
    assert np.allclose(out, clean)
    assert metrics["amplitude_scale"] == 0.0


def test_injection_refuses_missing_nav_bits():
    with pytest.raises(ValueError, match="navigation-bit provenance"):
        inject_counterfeit(np.ones(8, dtype=complex), 1, 1_023_000, InjectionTheta(0, 0, -3, 0), nav_bits=None)


def test_quantization_clipping_and_deterministic_injection():
    iq = np.array([1 + 2j, 200 - 200j], dtype=np.complex128)
    payload, q = quantize_interleaved_int8(iq)
    assert len(payload) == 4
    assert q["clipping_rate"] == 0.5
    clean = np.arange(16, dtype=float) + 1j * np.arange(16, dtype=float)
    theta = InjectionTheta(0.1, 25.0, -6.0, np.pi)
    a, _ = inject_counterfeit(clean, 2, 5_000_000, theta, nav_bits=np.ones(clean.size))
    b, _ = inject_counterfeit(clean, 2, 5_000_000, theta, nav_bits=np.ones(clean.size))
    assert np.array_equal(a, b)
