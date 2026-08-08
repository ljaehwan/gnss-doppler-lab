"""Physical complex-CAF model used by PG-SCC.

The module is deliberately recording agnostic.  It provides a same-PRN
analytic second-source response, covariance-whitened one/two-source fits, and
an actual raw-IQ sparse correlator.  No attack labels or PRN identities enter
the detector or coordinate selector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from gnss_doppler_lab.acaf_nf_stage0_r13_reconstruction import code_replica, carrier_wipeoff
from gnss_doppler_lab.acaf_nf_stage1_r2a_l20_foundation_audit import FS_HZ, SUPPORT_SAMPLES, State

DELAYS = np.arange(-1.0, 1.0001, 0.125, dtype=np.float64)
DOPPLERS = np.arange(-250.0, 250.0001, 50.0, dtype=np.float64)
SHAPE = (len(DOPPLERS), len(DELAYS))
N_COORDINATES = int(np.prod(SHAPE))
CENTER = int(np.ravel_multi_index((5, 8), SHAPE))
COORDINATES = np.asarray([(tau, doppler) for doppler in DOPPLERS for tau in DELAYS])
DEFAULT_SEARCH = tuple(
    (float(tau), float(doppler))
    for doppler in (-150.0, -75.0, 0.0, 75.0, 150.0)
    for tau in (-0.75, -0.375, -0.125, 0.0, 0.125, 0.375, 0.75)
    if tau != 0.0 or doppler != 0.0
)


@dataclass(frozen=True)
class FitResult:
    score: float
    rss_h0: float
    rss_h1: float
    delta_tau_chips: float
    delta_doppler_hz: float
    beta_h0: complex
    beta_h1_auth: complex
    beta_h1_second: complex


def coordinate_index(delay_chips: float, doppler_hz: float) -> int:
    """Return the exact flattened grid index for one complex coordinate."""
    delay = np.flatnonzero(np.isclose(DELAYS, float(delay_chips), atol=1e-12))
    doppler = np.flatnonzero(np.isclose(DOPPLERS, float(doppler_hz), atol=1e-12))
    if delay.size != 1 or doppler.size != 1:
        raise ValueError("coordinate is not on the frozen dense grid")
    return int(np.ravel_multi_index((int(doppler[0]), int(delay[0])), SHAPE))


def normalize_complex(surface: np.ndarray, mode: str = "prompt_phase", epsilon: float = 1e-9) -> np.ndarray:
    """Suppress global gain/phase while retaining relative complex shape."""
    value = np.asarray(surface, dtype=np.complex128).reshape(-1)
    if value.size != N_COORDINATES or not np.isfinite(value).all():
        raise ValueError("finite 187-coordinate complex CAF required")
    if mode == "prompt_phase":
        prompt = value[CENTER]
        if abs(prompt) <= epsilon:
            raise ValueError("prompt is too small for phase alignment")
        return value * np.exp(-1j * np.angle(prompt)) / (abs(prompt) + epsilon)
    if mode == "local_energy":
        scale = float(np.sqrt(np.mean(np.abs(value) ** 2)))
        if scale <= epsilon:
            raise ValueError("CAF energy is too small for normalization")
        return value / (scale + epsilon)
    raise ValueError(f"unknown normalization mode: {mode}")


def analytic_same_prn_template(delta_tau_chips: float, delta_doppler_hz: float) -> np.ndarray:
    """Ideal 1-ms GPS L1 C/A complex CAF for a same-PRN source.

    The code term is the triangular main lobe of the same-code periodic ACF.
    The carrier term is the exact rectangular-window complex sinc response.
    It is evaluated directly at every coordinate, so no array shifting,
    wrapping, or zero padding is used.
    """
    delay_error = DELAYS[None, :] - float(delta_tau_chips)
    code = np.maximum(1.0 - np.abs(delay_error), 0.0)
    frequency_error = DOPPLERS[:, None] - float(delta_doppler_hz)
    seconds = 1e-3
    carrier = np.sinc(frequency_error * seconds) * np.exp(1j * np.pi * frequency_error * seconds)
    value = code * carrier
    peak = np.max(np.abs(value))
    if peak <= 0:
        raise ValueError("second-source template is outside the modeled support")
    return (value / peak).reshape(-1).astype(np.complex128)


def estimate_complex_covariance(clean: np.ndarray, auth_template: np.ndarray, shrinkage: float = 0.35) -> np.ndarray:
    """Estimate a clean-only Hermitian covariance with diagonal shrinkage."""
    values = np.asarray(clean, dtype=np.complex128)
    template = np.asarray(auth_template, dtype=np.complex128).reshape(-1)
    if values.ndim != 2 or values.shape[1] != N_COORDINATES or values.shape[0] < 4:
        raise ValueError("clean covariance requires at least four 187-coordinate CAFs")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0,1]")
    denom = np.vdot(template, template).real
    alpha = (values @ np.conj(template)) / max(denom, 1e-12)
    residual = values - alpha[:, None] * template[None, :]
    residual -= residual.mean(axis=0, keepdims=True)
    sample = residual.conj().T @ residual / max(values.shape[0] - 1, 1)
    diagonal = np.diag(np.maximum(np.real(np.diag(sample)), 1e-7))
    covariance = (1.0 - shrinkage) * sample + shrinkage * diagonal
    floor = max(float(np.median(np.real(np.diag(covariance)))) * 1e-4, 1e-8)
    covariance = covariance + np.eye(N_COORDINATES) * floor
    return (covariance + covariance.conj().T) / 2


def _gls(design: np.ndarray, y: np.ndarray, precision: np.ndarray, ridge: float) -> tuple[float, np.ndarray]:
    if precision.ndim == 1:
        gram = design.conj().T @ (precision[:, None] * design) + np.eye(design.shape[1]) * float(ridge)
        rhs = design.conj().T @ (precision * y)
    else:
        gram = design.conj().T @ precision @ design + np.eye(design.shape[1]) * float(ridge)
        rhs = design.conj().T @ precision @ y
    beta = np.linalg.solve(gram, rhs)
    residual = y - design @ beta
    weighted = precision * residual if precision.ndim == 1 else precision @ residual
    rss = float(np.real(np.vdot(residual, weighted)))
    return max(rss, 0.0), beta


def two_source_glrt(
    surface: np.ndarray,
    auth_template: np.ndarray,
    covariance: np.ndarray,
    *,
    indices: Sequence[int] | None = None,
    search: Sequence[tuple[float, float]] = DEFAULT_SEARCH,
    ridge: float = 1e-3,
) -> FitResult:
    """Whitened complex H0/H1 likelihood improvement on identical support."""
    y_full = np.asarray(surface, dtype=np.complex128).reshape(-1)
    auth_full = np.asarray(auth_template, dtype=np.complex128).reshape(-1)
    cov_full = np.asarray(covariance, dtype=np.complex128)
    selected = np.arange(N_COORDINATES) if indices is None else np.asarray(indices, dtype=np.int64)
    if y_full.size != N_COORDINATES or auth_full.size != N_COORDINATES:
        raise ValueError("surface and authentic template must have 187 complex coordinates")
    if cov_full.shape != (N_COORDINATES, N_COORDINATES):
        raise ValueError("covariance must be 187x187")
    if selected.size < 3 or selected.size != np.unique(selected).size:
        raise ValueError("at least three unique complex coordinates are required")
    y = y_full[selected]
    auth = auth_full[selected]
    selected_covariance = cov_full[np.ix_(selected, selected)]
    off_diagonal = selected_covariance - np.diag(np.diag(selected_covariance))
    precision = (
        1.0 / np.maximum(np.real(np.diag(selected_covariance)), 1e-12)
        if np.count_nonzero(off_diagonal) == 0 else np.linalg.inv(selected_covariance)
    )
    rss0, beta0 = _gls(auth[:, None], y, precision, ridge)
    best_rss, best_beta, best_offset = rss0, np.asarray([beta0[0], 0j]), (0.0, 0.0)
    for tau, doppler in search:
        second = analytic_same_prn_template(tau, doppler)[selected]
        rss, beta = _gls(np.column_stack((auth, second)), y, precision, ridge)
        if rss < best_rss:
            best_rss, best_beta, best_offset = rss, beta, (float(tau), float(doppler))
    # Per-complex-coordinate improvement is comparable across K.
    score = max(rss0 - best_rss, 0.0) / selected.size
    return FitResult(
        score=float(score), rss_h0=float(rss0 / selected.size), rss_h1=float(best_rss / selected.size),
        delta_tau_chips=best_offset[0], delta_doppler_hz=best_offset[1],
        beta_h0=complex(beta0[0]), beta_h1_auth=complex(best_beta[0]), beta_h1_second=complex(best_beta[1]),
    )


def one_source_residual(
    surface: np.ndarray, auth_template: np.ndarray, covariance: np.ndarray, indices: Sequence[int] | None = None
) -> float:
    selected = np.arange(N_COORDINATES) if indices is None else np.asarray(indices, dtype=np.int64)
    y = np.asarray(surface, complex).reshape(-1)[selected]
    auth = np.asarray(auth_template, complex).reshape(-1)[selected]
    cov = np.asarray(covariance, complex)[np.ix_(selected, selected)]
    rss, _ = _gls(auth[:, None], y, np.linalg.inv(cov), 1e-3)
    return float(rss / selected.size)


def inject_same_prn_second_source(
    clean_surface: np.ndarray,
    *,
    delta_tau_chips: float,
    delta_doppler_hz: float,
    relative_amplitude: float,
    relative_phase_rad: float,
    noise_sigma: float,
    rng: np.random.Generator,
    normalization: str,
) -> np.ndarray:
    """Build synthetic H1 at complex-correlator level with the same PRN ACF."""
    base = np.asarray(clean_surface, dtype=np.complex128).reshape(-1)
    second = analytic_same_prn_template(delta_tau_chips, delta_doppler_hz)
    prompt_scale = max(abs(base[CENTER]), 1e-9)
    value = base + prompt_scale * float(relative_amplitude) * np.exp(1j * float(relative_phase_rad)) * second
    if noise_sigma > 0:
        noise = rng.normal(size=value.size) + 1j * rng.normal(size=value.size)
        value = value + float(noise_sigma) * prompt_scale * noise / np.sqrt(2.0)
    return normalize_complex(value, normalization)


def complex_correlator_coordinates(iq: np.ndarray, state: State, indices: Iterable[int]) -> np.ndarray:
    """Compute exactly K raw-IQ complex correlations at frozen coordinates."""
    values = np.asarray(iq, dtype=np.complex128)
    selected = np.asarray(list(indices), dtype=np.int64)
    if values.shape != (SUPPORT_SAMPLES,) or selected.size == 0 or selected.size != np.unique(selected).size:
        raise ValueError("one 1-ms support and unique nonempty coordinates are required")
    result = np.empty(selected.size, dtype=np.complex128)
    for position, index in enumerate(selected):
        if not 0 <= int(index) < N_COORDINATES:
            raise ValueError("coordinate index outside dense grid")
        di, ci = np.unravel_index(int(index), SHAPE)
        replica = code_replica(
            state.prn, values.size, FS_HZ, state.code_freq_chips, state.aux1,
            -1, float(DELAYS[ci]), replica_direction=1,
        )[0]
        wipe = carrier_wipeoff(
            values.size, FS_HZ, state.carrier_doppler_hz, float(DOPPLERS[di]), -1,
        )[0]
        result[position] = np.sum(values * wipe * replica)
    return result
