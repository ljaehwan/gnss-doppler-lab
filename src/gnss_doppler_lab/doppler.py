from __future__ import annotations

from gnss_doppler_lab.coordinates import dot, sub, unit

SPEED_OF_LIGHT_MPS = 299_792_458.0


def compute_doppler_hz(
    receiver_ecef: tuple[float, float, float],
    receiver_velocity_ecef_mps: tuple[float, float, float],
    satellite_ecef: tuple[float, float, float],
    satellite_velocity_ecef_mps: tuple[float, float, float],
    carrier_frequency_hz: float,
) -> float:
    line_of_sight = unit(sub(satellite_ecef, receiver_ecef))
    relative_velocity = sub(satellite_velocity_ecef_mps, receiver_velocity_ecef_mps)
    range_rate_mps = dot(relative_velocity, line_of_sight)
    return -(range_rate_mps / SPEED_OF_LIGHT_MPS) * carrier_frequency_hz
