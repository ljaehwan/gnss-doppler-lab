"""Analytic one-epoch action warp used by TRACE Stage-0.

The receiver row ``t`` contains correlators from the interval that just ended
and the updated DLL/PLL/NCO state used for interval ``t + 1``.  The code action
is expressed as a signed local-replica displacement in chips and the carrier
action as a signed wipe-off phase in radians.
"""

from __future__ import annotations

import numpy as np

TAP_COORDS_CHIPS = np.arange(-0.5, 0.5001, 0.125, dtype=np.float64)


def prompt_normalize(taps: np.ndarray, epsilon: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Prompt-reference complex taps and return a stable quality mask."""
    values = np.asarray(taps, dtype=np.complex128)
    if values.shape[-1] != 9:
        raise ValueError("TRACE full input must contain exactly nine complex taps")
    prompt = values[..., 4]
    power = np.abs(prompt) ** 2
    finite = np.isfinite(values.real).all(axis=-1) & np.isfinite(values.imag).all(axis=-1)
    valid = finite & (power > float(epsilon))
    normalized = values * np.conj(prompt)[..., None] / (power[..., None] + float(epsilon))
    normalized = np.where(valid[..., None], normalized, np.nan + 1j * np.nan)
    return normalized, valid


def _interp_complex_no_extrapolation(
    values: np.ndarray, source_coordinates: np.ndarray, target_coordinates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate complex values without zero padding/extrapolation."""
    source = np.asarray(source_coordinates, dtype=np.float64)
    target = np.asarray(target_coordinates, dtype=np.float64)
    vector = np.asarray(values, dtype=np.complex128)
    if vector.shape != source.shape or source.ndim != 1:
        raise ValueError("values and source_coordinates must be matching vectors")
    valid = (target >= source[0]) & (target <= source[-1])
    out = np.full(target.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    out[valid] = np.interp(target[valid], source, vector.real) + 1j * np.interp(
        target[valid], source, vector.imag
    )
    return out, valid


def warp_complex_taps(
    taps: np.ndarray,
    code_action_chips: float,
    carrier_action_rad: float,
    tap_coordinates_chips: np.ndarray = TAP_COORDS_CHIPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the source-signed code-coordinate and common carrier action.

    ``W(c,u)[k] = exp(-j*phi_u) c(tau_k + delta_tau_u)``.  Coordinates outside
    the measured aperture are invalid, never padded.  Positive code action is
    a positive displacement of the receiver's local replica coordinate.
    """
    coords = np.asarray(tap_coordinates_chips, dtype=np.float64)
    shifted = coords + float(code_action_chips)
    warped, valid = _interp_complex_no_extrapolation(np.asarray(taps), coords, shifted)
    warped[valid] *= np.exp(-1j * float(carrier_action_rad))
    return warped, valid


def receiver_action(
    code_freq_chips_s: float,
    carrier_doppler_hz: float,
    epoch_duration_s: float,
    nominal_code_rate_chips_s: float = 1_023_000.0,
    carrier_aiding_hz: float = 1_575_420_000.0,
) -> tuple[float, float]:
    """Convert dumped next-interval NCO state to the analytic warp action.

    GNSS-SDR sets ``code_freq = nominal - DLL_filter + carrier_aiding`` and
    ``carrier_phase_step = 2*pi*doppler/fs``.  The returned code displacement
    removes the deterministic carrier-aiding term, leaving the DLL action.
    """
    dt = float(epoch_duration_s)
    aided_nominal = float(nominal_code_rate_chips_s) * (
        1.0 + float(carrier_doppler_hz) / float(carrier_aiding_hz)
    )
    code_action = (float(code_freq_chips_s) - aided_nominal) * dt
    carrier_action = 2.0 * np.pi * float(carrier_doppler_hz) * dt
    return code_action, carrier_action
