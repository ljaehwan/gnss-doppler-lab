from __future__ import annotations

from math import atan2, cos, degrees, radians, sin, sqrt

WGS84_A = 6_378_137.0
WGS84_E2 = 6.69437999014e-3


Vector3 = tuple[float, float, float]


def dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(v: Vector3) -> float:
    return sqrt(dot(v, v))


def sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(v: Vector3, factor: float) -> Vector3:
    return (v[0] * factor, v[1] * factor, v[2] * factor)


def unit(v: Vector3) -> Vector3:
    magnitude = norm(v)
    return (v[0] / magnitude, v[1] / magnitude, v[2] / magnitude)


def geodetic_to_ecef(latitude_deg: float, longitude_deg: float, altitude_m: float) -> Vector3:
    lat = radians(latitude_deg)
    lon = radians(longitude_deg)
    sin_lat = sin(lat)
    cos_lat = cos(lat)
    sin_lon = sin(lon)
    cos_lon = cos(lon)
    n = WGS84_A / sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + altitude_m) * cos_lat * cos_lon
    y = (n + altitude_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + altitude_m) * sin_lat
    return (x, y, z)


def enu_basis(latitude_deg: float, longitude_deg: float) -> tuple[Vector3, Vector3, Vector3]:
    lat = radians(latitude_deg)
    lon = radians(longitude_deg)
    east = (-sin(lon), cos(lon), 0.0)
    north = (-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat))
    up = (cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat))
    return east, north, up


def enu_to_ecef_vector(
    east_mps: float,
    north_mps: float,
    up_mps: float,
    latitude_deg: float,
    longitude_deg: float,
) -> Vector3:
    east_axis, north_axis, up_axis = enu_basis(latitude_deg, longitude_deg)
    return add(add(scale(east_axis, east_mps), scale(north_axis, north_mps)), scale(up_axis, up_mps))


def ecef_to_enu(vector: Vector3, latitude_deg: float, longitude_deg: float) -> Vector3:
    east_axis, north_axis, up_axis = enu_basis(latitude_deg, longitude_deg)
    return (dot(vector, east_axis), dot(vector, north_axis), dot(vector, up_axis))


def azimuth_elevation(receiver_ecef: Vector3, satellite_ecef: Vector3, latitude_deg: float, longitude_deg: float) -> tuple[float, float, float]:
    line_of_sight = sub(satellite_ecef, receiver_ecef)
    east, north, up = ecef_to_enu(line_of_sight, latitude_deg, longitude_deg)
    horizontal = sqrt(east * east + north * north)
    azimuth_deg = (degrees(atan2(east, north)) + 360.0) % 360.0
    elevation_deg = degrees(atan2(up, horizontal))
    return azimuth_deg, elevation_deg, norm(line_of_sight)
