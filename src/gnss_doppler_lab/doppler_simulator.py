"""Receiver-level GNSS carrier Doppler simulation utilities.

The functions in this module generate *observation-level* Doppler values from
satellite ephemeris-like states and receiver trajectory states.  They do not
model or transmit RF signals; they are intended for defensive receiver-side
spoofing-detection experiments.
"""

from __future__ import annotations

import numpy as np

SPEED_OF_LIGHT_MPS = 299_792_458.0
GPS_L1_HZ = 1_575_420_000.0


def _as_float_array(x: np.ndarray | list[float] | float) -> np.ndarray:
    return np.asarray(x, dtype=float)


def los_unit_vectors(
    satellite_positions_ecef_m: np.ndarray,
    receiver_position_ecef_m: np.ndarray,
) -> np.ndarray:
    """Return line-of-sight unit vectors from receiver to satellites.

    Args:
        satellite_positions_ecef_m: Array shaped ``(n_sats, 3)``.
        receiver_position_ecef_m: Array shaped ``(3,)``.

    Returns:
        Array shaped ``(n_sats, 3)`` whose rows are unit vectors pointing from
        the receiver toward each satellite.
    """
    sat_pos = _as_float_array(satellite_positions_ecef_m)
    rx_pos = _as_float_array(receiver_position_ecef_m)
    if sat_pos.ndim != 2 or sat_pos.shape[-1] != 3:
        raise ValueError("satellite_positions_ecef_m must have shape (n_sats, 3)")
    if rx_pos.shape != (3,):
        raise ValueError("receiver_position_ecef_m must have shape (3,)")

    vectors = sat_pos - rx_pos
    ranges = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(ranges <= 0.0):
        raise ValueError("zero range between receiver and satellite is invalid")
    return vectors / ranges


def carrier_doppler_hz(
    satellite_positions_ecef_m: np.ndarray,
    satellite_velocities_ecef_mps: np.ndarray,
    receiver_position_ecef_m: np.ndarray,
    receiver_velocity_ecef_mps: np.ndarray,
    *,
    carrier_frequency_hz: float = GPS_L1_HZ,
    clock_drift_hz: float | np.ndarray = 0.0,
    noise_hz: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Compute carrier Doppler observations for one epoch.

    The sign convention is
    ``f_D = -f_c/c * (v_sat - v_rx) dot los + clock_drift + noise``.
    With this convention, a receiver moving toward a stationary satellite has
    positive Doppler.
    """
    sat_pos = _as_float_array(satellite_positions_ecef_m)
    sat_vel = _as_float_array(satellite_velocities_ecef_mps)
    rx_vel = _as_float_array(receiver_velocity_ecef_mps)
    if sat_vel.shape != sat_pos.shape:
        raise ValueError("satellite_velocities_ecef_mps must match satellite positions shape")
    if rx_vel.shape != (3,):
        raise ValueError("receiver_velocity_ecef_mps must have shape (3,)")
    if carrier_frequency_hz <= 0:
        raise ValueError("carrier_frequency_hz must be positive")

    los = los_unit_vectors(sat_pos, receiver_position_ecef_m)
    relative_velocity = sat_vel - rx_vel
    range_rate_mps = np.sum(relative_velocity * los, axis=-1)
    doppler = -(carrier_frequency_hz / SPEED_OF_LIGHT_MPS) * range_rate_mps
    return doppler + _as_float_array(clock_drift_hz) + _as_float_array(noise_hz)


def doppler_observation_matrix(
    satellite_positions_ecef_m: np.ndarray,
    satellite_velocities_ecef_mps: np.ndarray,
    receiver_positions_ecef_m: np.ndarray,
    receiver_velocities_ecef_mps: np.ndarray,
    *,
    carrier_frequency_hz: float = GPS_L1_HZ,
    clock_drift_hz: float | np.ndarray = 0.0,
    noise_hz: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Simulate Doppler for multiple epochs and satellites.

    Args:
        satellite_positions_ecef_m: Shape ``(n_epochs, n_sats, 3)``.
        satellite_velocities_ecef_mps: Shape ``(n_epochs, n_sats, 3)``.
        receiver_positions_ecef_m: Shape ``(n_epochs, 3)``.
        receiver_velocities_ecef_mps: Shape ``(n_epochs, 3)``.
        clock_drift_hz: Scalar or shape ``(n_epochs,)`` or ``(n_epochs, n_sats)``.
        noise_hz: Scalar or shape ``(n_epochs, n_sats)``.

    Returns:
        Doppler matrix shaped ``(n_epochs, n_sats)``.
    """
    sat_pos = _as_float_array(satellite_positions_ecef_m)
    sat_vel = _as_float_array(satellite_velocities_ecef_mps)
    rx_pos = _as_float_array(receiver_positions_ecef_m)
    rx_vel = _as_float_array(receiver_velocities_ecef_mps)
    if sat_pos.ndim != 3 or sat_pos.shape[-1] != 3:
        raise ValueError("satellite_positions_ecef_m must have shape (n_epochs, n_sats, 3)")
    if sat_vel.shape != sat_pos.shape:
        raise ValueError("satellite_velocities_ecef_mps must match satellite positions shape")
    if rx_pos.shape != (sat_pos.shape[0], 3):
        raise ValueError("receiver_positions_ecef_m must have shape (n_epochs, 3)")
    if rx_vel.shape != (sat_pos.shape[0], 3):
        raise ValueError("receiver_velocities_ecef_mps must have shape (n_epochs, 3)")

    clock = _as_float_array(clock_drift_hz)
    noise_arr = _as_float_array(noise_hz)
    out = np.empty(sat_pos.shape[:2], dtype=float)
    for epoch in range(sat_pos.shape[0]):
        epoch_clock = clock[epoch] if clock.ndim == 1 else clock
        epoch_noise = noise_arr[epoch] if noise_arr.ndim == 2 else noise_arr
        out[epoch] = carrier_doppler_hz(
            sat_pos[epoch],
            sat_vel[epoch],
            rx_pos[epoch],
            rx_vel[epoch],
            carrier_frequency_hz=carrier_frequency_hz,
            clock_drift_hz=epoch_clock,
            noise_hz=epoch_noise,
        )
    return out


def add_common_spoofing_drift(
    authentic_doppler_hz: np.ndarray,
    common_drift_hz: float | np.ndarray,
    *,
    prn_offsets_hz: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Create a simple spoofed Doppler observation matrix.

    This receiver-observation model adds an epoch-common drift/bias and optional
    per-PRN offsets.  It is useful as the first defensive simulation step for
    testing whether multi-PRN covariance/rank features respond to common-mode
    spoofing-like distortion.
    """
    authentic = _as_float_array(authentic_doppler_hz)
    if authentic.ndim != 2:
        raise ValueError("authentic_doppler_hz must have shape (n_epochs, n_sats)")

    common = _as_float_array(common_drift_hz)
    if common.ndim == 1:
        common = common[:, None]
    return authentic + common + _as_float_array(prn_offsets_hz)
