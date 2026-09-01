"""Diagnostics for whether two same-PRN carrier hypotheses are observable."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class DominantDopplerPeaks:
    frequencies_hz: tuple[float, ...]
    normalized_heights: tuple[float, ...]


def normalized_doppler_envelope(
    magnitude: np.ndarray,
    *,
    samples_per_code: int,
) -> np.ndarray:
    """Collapse one C/A-code period of a delay--Doppler surface over delay."""
    values = np.asarray(magnitude, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("magnitude must be a non-empty two-dimensional array")
    if not 0 < samples_per_code <= values.shape[1]:
        raise ValueError("samples_per_code is outside the delay axis")
    envelope = np.max(values[:, :samples_per_code], axis=1)
    peak = float(np.max(envelope))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("Doppler envelope has no positive finite peak")
    return envelope / peak


def dominant_doppler_peaks(
    doppler_bins_hz: np.ndarray,
    normalized_envelope: np.ndarray,
    *,
    minimum_height: float = 0.5,
    minimum_prominence: float = 0.1,
    minimum_separation_hz: float = 60.0,
) -> DominantDopplerPeaks:
    """Return strong local maxima, ordered by decreasing peak height."""
    bins = np.asarray(doppler_bins_hz, dtype=float)
    profile = np.asarray(normalized_envelope, dtype=float)
    if bins.ndim != 1 or profile.ndim != 1 or bins.size != profile.size or bins.size < 3:
        raise ValueError("Doppler bins and envelope must be equal-length one-dimensional arrays")
    spacing = np.diff(bins)
    if np.any(spacing <= 0) or not np.allclose(spacing, spacing[0]):
        raise ValueError("Doppler bins must be strictly increasing and uniformly spaced")
    distance = max(1, int(np.ceil(float(minimum_separation_hz) / float(spacing[0]))))
    indices, _ = find_peaks(
        profile,
        height=float(minimum_height),
        prominence=float(minimum_prominence),
        distance=distance,
    )
    ordered = sorted(indices, key=lambda index: float(profile[index]), reverse=True)
    return DominantDopplerPeaks(
        tuple(float(bins[index]) for index in ordered),
        tuple(float(profile[index]) for index in ordered),
    )


def local_probe(
    doppler_bins_hz: np.ndarray,
    normalized_envelope: np.ndarray,
    expected_hz: float,
    *,
    half_width_hz: float = 25.0,
) -> tuple[float, float]:
    """Return the strongest envelope value near an expected Doppler."""
    bins = np.asarray(doppler_bins_hz, dtype=float)
    profile = np.asarray(normalized_envelope, dtype=float)
    selected = np.flatnonzero(np.abs(bins - float(expected_hz)) <= float(half_width_hz))
    if selected.size == 0:
        raise ValueError("expected Doppler is outside the probe grid")
    index = int(selected[np.argmax(profile[selected])])
    return float(bins[index]), float(profile[index])
