import numpy as np
import pytest

from gnss_doppler_lab.doppler_observability import (
    dominant_doppler_peaks,
    local_probe,
    normalized_doppler_envelope,
)


def test_envelope_collapses_one_code_period_and_normalizes() -> None:
    magnitude = np.array([[1, 2, 99, 99], [3, 1, 99, 99], [2, 1, 99, 99]], dtype=float)
    result = normalized_doppler_envelope(magnitude, samples_per_code=2)
    np.testing.assert_allclose(result, [2 / 3, 1, 2 / 3])


def test_dominant_peaks_reject_sidelobes_and_keep_two_carriers() -> None:
    bins = np.arange(-200, 201, 10, dtype=float)
    profile = np.zeros_like(bins)
    profile[np.where(bins == -60)[0][0]] = 1.0
    profile[np.where(bins == 70)[0][0]] = 0.91
    profile[np.where(bins == 130)[0][0]] = 0.25
    peaks = dominant_doppler_peaks(bins, profile)
    assert peaks.frequencies_hz == (-60.0, 70.0)


def test_local_probe_reports_nearest_local_maximum() -> None:
    bins = np.arange(-100, 101, 10, dtype=float)
    profile = np.linspace(0, 1, bins.size)
    frequency, value = local_probe(bins, profile, 35.0, half_width_hz=15.0)
    assert frequency == 50.0
    assert value == pytest.approx(profile[bins == 50][0])


def test_envelope_rejects_zero_surface() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        normalized_doppler_envelope(np.zeros((3, 4)), samples_per_code=4)
