from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

from gnss_doppler_lab.coordinates import Vector3

EARTH_ROTATION_RAD_S = 7.2921159e-5
EARTH_RADIUS_M = 6_378_137.0
GPS_ALTITUDE_M = 20_200_000.0
GPS_ORBIT_RADIUS_M = EARTH_RADIUS_M + GPS_ALTITUDE_M
GPS_INCLINATION_RAD = 0.9599310886  # 55 deg
GPS_ORBIT_PERIOD_S = 43_082.0
GPS_MEAN_MOTION_RAD_S = 2.0 * pi / GPS_ORBIT_PERIOD_S


@dataclass(slots=True)
class SatelliteState:
    prn: str
    position_ecef_m: Vector3
    velocity_ecef_mps: Vector3


def _rotate_z(angle_rad: float, vector: Vector3) -> Vector3:
    x, y, z = vector
    c = cos(angle_rad)
    s = sin(angle_rad)
    return (c * x - s * y, s * x + c * y, z)


def _cross_omega_r(position: Vector3) -> Vector3:
    x, y, z = position
    omega = EARTH_ROTATION_RAD_S
    return (-omega * y, omega * x, 0.0)


def _satellite_state(index: int, elapsed_s: float, num_satellites: int, orbital_planes: int) -> SatelliteState:
    satellites_per_plane = max(1, num_satellites // orbital_planes)
    plane_index = index // satellites_per_plane
    slot_index = index % satellites_per_plane

    raan = 2.0 * pi * plane_index / orbital_planes
    plane_phase = 2.0 * pi * slot_index / satellites_per_plane
    inter_plane_bias = (plane_index % 2) * (pi / satellites_per_plane)
    argument_of_latitude = plane_phase + inter_plane_bias + GPS_MEAN_MOTION_RAD_S * elapsed_s

    x_orb = GPS_ORBIT_RADIUS_M * cos(argument_of_latitude)
    y_orb = GPS_ORBIT_RADIUS_M * sin(argument_of_latitude)
    vx_orb = -GPS_ORBIT_RADIUS_M * GPS_MEAN_MOTION_RAD_S * sin(argument_of_latitude)
    vy_orb = GPS_ORBIT_RADIUS_M * GPS_MEAN_MOTION_RAD_S * cos(argument_of_latitude)

    cos_raan = cos(raan)
    sin_raan = sin(raan)
    cos_inc = cos(GPS_INCLINATION_RAD)
    sin_inc = sin(GPS_INCLINATION_RAD)

    position_eci = (
        x_orb * cos_raan - y_orb * cos_inc * sin_raan,
        x_orb * sin_raan + y_orb * cos_inc * cos_raan,
        y_orb * sin_inc,
    )
    velocity_eci = (
        vx_orb * cos_raan - vy_orb * cos_inc * sin_raan,
        vx_orb * sin_raan + vy_orb * cos_inc * cos_raan,
        vy_orb * sin_inc,
    )

    earth_rotation_angle = EARTH_ROTATION_RAD_S * elapsed_s
    position_ecef = _rotate_z(earth_rotation_angle, position_eci)
    velocity_rotated = _rotate_z(earth_rotation_angle, velocity_eci)
    velocity_ecef = (
        velocity_rotated[0] - _cross_omega_r(position_ecef)[0],
        velocity_rotated[1] - _cross_omega_r(position_ecef)[1],
        velocity_rotated[2],
    )

    return SatelliteState(
        prn=f"G{index + 1:02d}",
        position_ecef_m=position_ecef,
        velocity_ecef_mps=velocity_ecef,
    )


def generate_satellite_constellation(elapsed_s: float, num_satellites: int, orbital_planes: int) -> list[SatelliteState]:
    return [
        _satellite_state(index, elapsed_s, num_satellites=num_satellites, orbital_planes=orbital_planes)
        for index in range(num_satellites)
    ]
