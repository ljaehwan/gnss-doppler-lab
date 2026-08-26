"""Physics-only correlation-profile and cross-satellite geometry utilities."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


EPSILON = 1e-12


def triangular_correlation(offset_chips: np.ndarray | Iterable[float]) -> np.ndarray:
    """Ideal GPS L1 C/A autocorrelation magnitude on the chip axis."""
    offset = np.asarray(offset_chips, dtype=np.float64)
    if not np.isfinite(offset).all():
        raise ValueError("correlation offsets must be finite")
    return np.maximum(0.0, 1.0 - np.abs(offset))


def normalize_profiles(profiles: np.ndarray) -> np.ndarray:
    """L2-normalize one profile or a row-major profile matrix."""
    values = np.asarray(profiles, dtype=np.float64)
    if values.ndim not in (1, 2) or not values.size or not np.isfinite(values).all():
        raise ValueError("profiles must be a nonempty finite vector or matrix")
    matrix = values[None, :] if values.ndim == 1 else values
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= EPSILON):
        raise ValueError("correlation profile has zero norm")
    normalized = matrix / norms
    return normalized[0] if values.ndim == 1 else normalized


def complex_profile_features(
    profiles: np.ndarray, *, prompt_index: int
) -> np.ndarray:
    """L2-normalized complex taps, phase-aligned to a real-positive prompt."""
    values = np.asarray(profiles, dtype=np.complex128)
    if values.ndim not in (1, 2) or not values.size or not np.isfinite(values).all():
        raise ValueError("complex profiles must be a nonempty finite vector or matrix")
    matrix = values[None, :] if values.ndim == 1 else values
    if isinstance(prompt_index, bool) or not 0 <= int(prompt_index) < matrix.shape[1]:
        raise ValueError("prompt index is outside the complex profile")
    prompt_index = int(prompt_index)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= EPSILON) or np.any(np.abs(matrix[:, prompt_index]) <= EPSILON):
        raise ValueError("complex profile or prompt has zero norm")
    normalized = matrix / norms
    rotation = np.exp(-1j * np.angle(normalized[:, prompt_index]))[:, None]
    aligned = normalized * rotation
    features = np.concatenate((aligned.real, aligned.imag), axis=1)
    return features[0] if values.ndim == 1 else features


def two_path_complex_profile(
    tap_offsets_chips: np.ndarray | Iterable[float],
    *,
    authentic_center_chips: float,
    secondary_delay_chips: float,
    secondary_amplitude_ratio: float,
    relative_phase_rad: float,
    complex_noise: np.ndarray | None = None,
) -> np.ndarray:
    """Complex authentic-plus-replica correlator profile before magnitude loss."""
    taps = np.asarray(tuple(tap_offsets_chips), dtype=np.float64)
    center = float(authentic_center_chips)
    delay = float(secondary_delay_chips)
    ratio = float(secondary_amplitude_ratio)
    phase = float(relative_phase_rad)
    if (
        taps.ndim != 1
        or not len(taps)
        or not np.isfinite(taps).all()
        or not all(math.isfinite(value) for value in (center, delay, ratio, phase))
        or ratio < 0
    ):
        raise ValueError("invalid finite two-path profile parameters")
    authentic = triangular_correlation(taps - center)
    secondary = triangular_correlation(taps - center - delay)
    complex_profile = authentic.astype(np.complex128)
    complex_profile += ratio * np.exp(1j * phase) * secondary
    if complex_noise is not None:
        noise = np.asarray(complex_noise, dtype=np.complex128)
        if noise.shape != complex_profile.shape or not np.isfinite(noise).all():
            raise ValueError("complex noise must match the tap profile")
        complex_profile += noise
    return complex_profile


def two_path_magnitude_profile(
    tap_offsets_chips: np.ndarray | Iterable[float],
    *,
    authentic_center_chips: float,
    secondary_delay_chips: float,
    secondary_amplitude_ratio: float,
    relative_phase_rad: float,
    complex_noise: np.ndarray | None = None,
) -> np.ndarray:
    """Magnitude of authentic plus one delayed coherent replica."""
    return np.abs(
        two_path_complex_profile(
            tap_offsets_chips,
            authentic_center_chips=authentic_center_chips,
            secondary_delay_chips=secondary_delay_chips,
            secondary_amplitude_ratio=secondary_amplitude_ratio,
            relative_phase_rad=relative_phase_rad,
            complex_noise=complex_noise,
        )
    )


@dataclass(frozen=True)
class CorrelatorTemplateBank:
    profiles: np.ndarray
    delays_chips: np.ndarray
    centers_chips: np.ndarray
    amplitude_ratios: np.ndarray
    phases_rad: np.ndarray

    def __post_init__(self) -> None:
        rows = len(self.profiles)
        if self.profiles.ndim != 2 or rows == 0:
            raise ValueError("template profiles must be a nonempty matrix")
        for values in (
            self.delays_chips,
            self.centers_chips,
            self.amplitude_ratios,
            self.phases_rad,
        ):
            if values.shape != (rows,) or not np.isfinite(values).all():
                raise ValueError("template metadata length mismatch")


def build_template_bank(
    tap_offsets_chips: Iterable[float],
    *,
    delays_chips: Iterable[float],
    centers_chips: Iterable[float],
    amplitude_ratios: Iterable[float],
    phases_rad: Iterable[float],
) -> CorrelatorTemplateBank:
    """Build a deterministic normalized magnitude-profile dictionary."""
    taps = np.asarray(tuple(tap_offsets_chips), dtype=np.float64)
    axes = [
        np.asarray(tuple(values), dtype=np.float64)
        for values in (delays_chips, centers_chips, amplitude_ratios, phases_rad)
    ]
    if taps.ndim != 1 or not len(taps) or any(axis.ndim != 1 or not len(axis) for axis in axes):
        raise ValueError("template axes must be nonempty vectors")
    delay_axis, center_axis, ratio_axis, phase_axis = axes
    if not all(np.isfinite(axis).all() for axis in [taps, *axes]) or np.any(ratio_axis < 0):
        raise ValueError("template axes must be finite and amplitudes nonnegative")
    profiles, metadata = [], []
    for delay in delay_axis:
        for center in center_axis:
            for ratio in ratio_axis:
                for phase in phase_axis:
                    profiles.append(
                        normalize_profiles(
                            two_path_magnitude_profile(
                                taps,
                                authentic_center_chips=float(center),
                                secondary_delay_chips=float(delay),
                                secondary_amplitude_ratio=float(ratio),
                                relative_phase_rad=float(phase),
                            )
                        )
                    )
                    metadata.append((delay, center, ratio, phase))
    meta = np.asarray(metadata, dtype=np.float64)
    return CorrelatorTemplateBank(
        profiles=np.asarray(profiles, dtype=np.float64),
        delays_chips=meta[:, 0],
        centers_chips=meta[:, 1],
        amplitude_ratios=meta[:, 2],
        phases_rad=meta[:, 3],
    )


def build_complex_template_bank(
    tap_offsets_chips: Iterable[float],
    *,
    prompt_index: int,
    delays_chips: Iterable[float],
    centers_chips: Iterable[float],
    amplitude_ratios: Iterable[float],
    phases_rad: Iterable[float],
) -> CorrelatorTemplateBank:
    """Build a deterministic prompt-phase-aligned complex-tap dictionary."""
    taps = np.asarray(tuple(tap_offsets_chips), dtype=np.float64)
    axes = [
        np.asarray(tuple(values), dtype=np.float64)
        for values in (delays_chips, centers_chips, amplitude_ratios, phases_rad)
    ]
    if taps.ndim != 1 or not len(taps) or any(axis.ndim != 1 or not len(axis) for axis in axes):
        raise ValueError("template axes must be nonempty vectors")
    delay_axis, center_axis, ratio_axis, phase_axis = axes
    if not all(np.isfinite(axis).all() for axis in [taps, *axes]) or np.any(ratio_axis < 0):
        raise ValueError("template axes must be finite and amplitudes nonnegative")
    if isinstance(prompt_index, bool) or not 0 <= int(prompt_index) < len(taps):
        raise ValueError("prompt index is outside the template tap layout")
    profiles, metadata = [], []
    for delay in delay_axis:
        for center in center_axis:
            for ratio in ratio_axis:
                for phase in phase_axis:
                    profiles.append(
                        complex_profile_features(
                            two_path_complex_profile(
                                taps,
                                authentic_center_chips=float(center),
                                secondary_delay_chips=float(delay),
                                secondary_amplitude_ratio=float(ratio),
                                relative_phase_rad=float(phase),
                            ),
                            prompt_index=int(prompt_index),
                        )
                    )
                    metadata.append((delay, center, ratio, phase))
    meta = np.asarray(metadata, dtype=np.float64)
    return CorrelatorTemplateBank(
        profiles=np.asarray(profiles, dtype=np.float64),
        delays_chips=meta[:, 0],
        centers_chips=meta[:, 1],
        amplitude_ratios=meta[:, 2],
        phases_rad=meta[:, 3],
    )


class TemplateDelayEstimator:
    """Nearest-template signed secondary-delay estimator."""

    def __init__(self, bank: CorrelatorTemplateBank) -> None:
        self.bank = bank
        self._tree = cKDTree(bank.profiles)

    def estimate(self, profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normalized = normalize_profiles(profiles)
        matrix = normalized[None, :] if normalized.ndim == 1 else normalized
        distance, index = self._tree.query(matrix, k=1)
        estimates = self.bank.delays_chips[np.asarray(index, dtype=np.int64)]
        if normalized.ndim == 1:
            return estimates[:1], np.asarray(distance)[:1], np.asarray(index)[:1]
        return estimates, np.asarray(distance), np.asarray(index)


@dataclass(frozen=True)
class GeometryFit:
    theta: np.ndarray
    predicted_delays_chips: np.ndarray
    normalized_residual: float
    coherence: float
    rank: int


def fit_common_geometry(
    los_enu: np.ndarray,
    signed_delays_chips: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> GeometryFit:
    """Fit delay_j = [-u_j^T, 1] [Delta r/L_chip, clock]^T."""
    los = np.asarray(los_enu, dtype=np.float64)
    delays = np.asarray(signed_delays_chips, dtype=np.float64)
    if (
        los.ndim != 2
        or los.shape[1] != 3
        or delays.shape != (len(los),)
        or len(los) < 5
        or not np.isfinite(los).all()
        or not np.isfinite(delays).all()
    ):
        raise ValueError("geometry fit requires at least five finite LOS-delay rows")
    norms = np.linalg.norm(los, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("LOS rows must have unit norm")
    design = np.column_stack((-los, np.ones(len(los), dtype=np.float64)))
    if weights is None:
        weight = np.ones(len(los), dtype=np.float64)
    else:
        weight = np.asarray(weights, dtype=np.float64)
        if weight.shape != delays.shape or not np.isfinite(weight).all() or np.any(weight <= 0):
            raise ValueError("geometry weights must be finite and positive")
    root_weight = np.sqrt(weight)
    theta, _, rank, _ = np.linalg.lstsq(
        design * root_weight[:, None], delays * root_weight, rcond=None
    )
    predicted = design @ theta
    residual_energy = float(np.sum(weight * (delays - predicted) ** 2))
    observed_energy = float(np.sum(weight * delays**2))
    normalized = min(1.0, max(0.0, residual_energy / (observed_energy + EPSILON)))
    return GeometryFit(
        theta=theta,
        predicted_delays_chips=predicted,
        normalized_residual=normalized,
        coherence=1.0 - normalized,
        rank=int(rank),
    )


def random_derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a deterministic-under-RNG permutation with no fixed points."""
    if isinstance(size, bool) or int(size) != size or size < 2:
        raise ValueError("derangement size must be an integer of at least two")
    size = int(size)
    identity = np.arange(size)
    for _ in range(100):
        candidate = rng.permutation(size)
        if np.all(candidate != identity):
            return candidate
    shift = int(rng.integers(1, size))
    return np.roll(identity, shift)


def profile_width_variance(
    profiles: np.ndarray, tap_offsets_chips: Iterable[float]
) -> np.ndarray:
    """Normalized magnitude-profile variance in chip-squared units."""
    values = np.asarray(profiles, dtype=np.float64)
    matrix = values[None, :] if values.ndim == 1 else values
    taps = np.asarray(tuple(tap_offsets_chips), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(taps) or np.any(matrix < 0):
        raise ValueError("nonnegative profiles and matching tap offsets are required")
    total = matrix.sum(axis=1)
    if np.any(total <= EPSILON):
        raise ValueError("profile magnitude sum must be positive")
    centroid = (matrix * taps[None, :]).sum(axis=1) / total
    width = (matrix * (taps[None, :] - centroid[:, None]) ** 2).sum(axis=1) / total
    return width
