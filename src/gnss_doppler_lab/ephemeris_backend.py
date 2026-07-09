"""RINEX NAV-backed GPS broadcast ephemeris utilities.

This module keeps the scope intentionally defensive and receiver-side:
- parse GPS broadcast ephemerides from RINEX NAV;
- generate satellite ECEF position/velocity for requested epochs;
- compute receiver geometry metadata such as elevation/azimuth;
- prepare hover/static receiver inputs for observation-level Doppler simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
import re
from typing import Iterable

import numpy as np

GPS_MU_M3PS2 = 3.986005e14
GPS_OMEGA_EARTH_RADPS = 7.2921151467e-5
GPS_WEEK_SECONDS = 604_800.0
WGS84_A_M = 6_378_137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@dataclass(frozen=True)
class GpsBroadcastEphemeris:
    prn: str
    constellation: str
    toc: datetime
    gps_week: int
    toe_s: float
    af0: float
    af1: float
    af2: float
    iode: float
    crs: float
    delta_n: float
    m0: float
    cuc: float
    eccentricity: float
    cus: float
    sqrt_a: float
    cic: float
    omega0: float
    cis: float
    i0: float
    crc: float
    omega: float
    omega_dot: float
    idot: float
    tgd: float = 0.0


def _parse_float(field: str) -> float:
    text = field.strip().replace("D", "E")
    return float(text) if text else 0.0


def _parse_prn_and_epoch(line: str) -> tuple[str, datetime]:
    head = line[:23].split()
    if len(head) < 7:
        raise ValueError(f"invalid RINEX NAV epoch header: {line!r}")

    if len(head[0]) == 1 and len(head) >= 8 and head[1].isdigit():
        prn = f"{head[0]}{int(head[1]):02d}"
        year, month, day, hour, minute, second = head[2:8]
    else:
        prn = head[0]
        year, month, day, hour, minute, second = head[1:7]

    year_i = int(year)
    if year_i < 100:
        year_i += 2000 if year_i < 80 else 1900
    return prn, datetime(
        year_i,
        int(month),
        int(day),
        int(hour),
        int(minute),
        int(float(second)),
        tzinfo=timezone.utc,
    )


def _line_fields(line: str, *, first_line: bool) -> list[float]:
    payload = line[23:] if first_line else line[3:]
    matches = re.findall(r"[+-]?\d+\.\d+(?:[DdEe][+-]?\d+)?", payload)
    expected = 3 if first_line else 4
    if len(matches) < expected:
        raise ValueError(f"invalid RINEX NAV numeric field line: {line!r}")
    return [_parse_float(field) for field in matches[:expected]]


def parse_rinex_nav_file(path: str | Path) -> list[GpsBroadcastEphemeris]:
    """Parse GPS broadcast-ephemeris records from a RINEX NAV file."""
    lines = Path(path).read_text().splitlines()
    try:
        header_end = next(i for i, line in enumerate(lines) if "END OF HEADER" in line)
    except StopIteration as exc:
        raise ValueError("RINEX NAV file missing END OF HEADER") from exc

    records: list[GpsBroadcastEphemeris] = []
    i = header_end + 1
    while i < len(lines):
        line0 = lines[i]
        if not line0.strip():
            i += 1
            continue
        try:
            prn, toc = _parse_prn_and_epoch(line0)
        except ValueError:
            i += 1
            continue
        constellation = prn[:1]
        if constellation != "G":
            i += 1
            continue
        if i + 7 >= len(lines):
            raise ValueError("truncated GPS RINEX NAV record")
        l0 = _line_fields(line0, first_line=True)
        l1 = _line_fields(lines[i + 1], first_line=False)
        l2 = _line_fields(lines[i + 2], first_line=False)
        l3 = _line_fields(lines[i + 3], first_line=False)
        l4 = _line_fields(lines[i + 4], first_line=False)
        l5 = _line_fields(lines[i + 5], first_line=False)
        l6 = _line_fields(lines[i + 6], first_line=False)

        records.append(
            GpsBroadcastEphemeris(
                prn=prn,
                constellation=constellation,
                toc=toc,
                gps_week=int(round(l5[2])),
                toe_s=l3[0],
                af0=l0[0],
                af1=l0[1],
                af2=l0[2],
                iode=l1[0],
                crs=l1[1],
                delta_n=l1[2],
                m0=l1[3],
                cuc=l2[0],
                eccentricity=l2[1],
                cus=l2[2],
                sqrt_a=l2[3],
                cic=l3[1],
                omega0=l3[2],
                cis=l3[3],
                i0=l4[0],
                crc=l4[1],
                omega=l4[2],
                omega_dot=l4[3],
                idot=l5[0],
                tgd=l6[2],
            )
        )
        i += 8
    return records


def gps_prns_in_nav_file(path: str | Path) -> list[str]:
    """Return sorted GPS PRN identifiers available in a NAV file."""
    return sorted({record.prn for record in parse_rinex_nav_file(path)})


def _gps_week_and_sow(time_utc: datetime) -> tuple[int, float]:
    if time_utc.tzinfo is None:
        time_utc = time_utc.replace(tzinfo=timezone.utc)
    else:
        time_utc = time_utc.astimezone(timezone.utc)
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    delta = time_utc - gps_epoch
    total_seconds = delta.total_seconds()
    week = int(total_seconds // GPS_WEEK_SECONDS)
    sow = total_seconds - week * GPS_WEEK_SECONDS
    return week, sow


def _wrap_gps_time(seconds: float) -> float:
    if seconds > GPS_WEEK_SECONDS / 2:
        return seconds - GPS_WEEK_SECONDS
    if seconds < -GPS_WEEK_SECONDS / 2:
        return seconds + GPS_WEEK_SECONDS
    return seconds


def _solve_kepler(mean_anomaly_rad: float, eccentricity: float, *, tol: float = 1e-12, max_iter: int = 20) -> float:
    eccentric_anomaly = mean_anomaly_rad
    for _ in range(max_iter):
        residual = eccentric_anomaly - eccentricity * sin(eccentric_anomaly) - mean_anomaly_rad
        derivative = 1.0 - eccentricity * cos(eccentric_anomaly)
        step = residual / derivative
        eccentric_anomaly -= step
        if abs(step) < tol:
            break
    return eccentric_anomaly


def satellite_position_ecef_m(record: GpsBroadcastEphemeris, time_utc: datetime) -> np.ndarray:
    """Compute GPS broadcast-orbit satellite ECEF position at a UTC epoch."""
    target_week, target_sow = _gps_week_and_sow(time_utc)
    tk = _wrap_gps_time((target_week - record.gps_week) * GPS_WEEK_SECONDS + (target_sow - record.toe_s))

    semi_major_axis = record.sqrt_a**2
    mean_motion_0 = sqrt(GPS_MU_M3PS2 / semi_major_axis**3)
    mean_motion = mean_motion_0 + record.delta_n
    mean_anomaly = record.m0 + mean_motion * tk
    eccentric_anomaly = _solve_kepler(mean_anomaly, record.eccentricity)

    sin_e = sin(eccentric_anomaly)
    cos_e = cos(eccentric_anomaly)
    true_anomaly = atan2(
        sqrt(1.0 - record.eccentricity**2) * sin_e,
        cos_e - record.eccentricity,
    )
    argument_of_latitude = true_anomaly + record.omega
    cos_2u = cos(2.0 * argument_of_latitude)
    sin_2u = sin(2.0 * argument_of_latitude)

    corrected_u = argument_of_latitude + record.cuc * cos_2u + record.cus * sin_2u
    corrected_r = semi_major_axis * (1.0 - record.eccentricity * cos_e) + record.crc * cos_2u + record.crs * sin_2u
    corrected_i = record.i0 + record.idot * tk + record.cic * cos_2u + record.cis * sin_2u

    x_orb = corrected_r * cos(corrected_u)
    y_orb = corrected_r * sin(corrected_u)
    omega = record.omega0 + (record.omega_dot - GPS_OMEGA_EARTH_RADPS) * tk - GPS_OMEGA_EARTH_RADPS * record.toe_s

    cos_omega = cos(omega)
    sin_omega = sin(omega)
    cos_i = cos(corrected_i)
    sin_i = sin(corrected_i)

    return np.array(
        [
            x_orb * cos_omega - y_orb * cos_i * sin_omega,
            x_orb * sin_omega + y_orb * cos_i * cos_omega,
            y_orb * sin_i,
        ],
        dtype=float,
    )


def satellite_velocity_ecef_mps(record: GpsBroadcastEphemeris, time_utc: datetime, dt_seconds: float = 0.5) -> np.ndarray:
    """Approximate ECEF velocity by central difference around the epoch."""
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be positive")
    dt = timedelta(seconds=dt_seconds)
    before = time_utc.astimezone(timezone.utc) - dt
    after = time_utc.astimezone(timezone.utc) + dt
    return (satellite_position_ecef_m(record, after) - satellite_position_ecef_m(record, before)) / (2.0 * dt_seconds)


def _best_record_for_time(records: Iterable[GpsBroadcastEphemeris], prn: str, time_utc: datetime) -> GpsBroadcastEphemeris:
    matching = [record for record in records if record.prn == prn]
    if not matching:
        raise KeyError(f"no ephemeris found for {prn}")
    target_week, target_sow = _gps_week_and_sow(time_utc)
    return min(
        matching,
        key=lambda record: abs(
            _wrap_gps_time((target_week - record.gps_week) * GPS_WEEK_SECONDS + (target_sow - record.toe_s))
        ),
    )


def satellite_states_for_times(
    records: list[GpsBroadcastEphemeris],
    times_utc: Iterable[datetime],
    prns: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return satellite ECEF position/velocity arrays for requested epochs."""
    times = list(times_utc)
    if not times:
        raise ValueError("times_utc must not be empty")
    prn_list = sorted(set(prns or [record.prn for record in records]))
    positions = np.empty((len(times), len(prn_list), 3), dtype=float)
    velocities = np.empty_like(positions)
    for epoch_idx, time_utc in enumerate(times):
        for prn_idx, prn in enumerate(prn_list):
            record = _best_record_for_time(records, prn, time_utc)
            positions[epoch_idx, prn_idx] = satellite_position_ecef_m(record, time_utc)
            velocities[epoch_idx, prn_idx] = satellite_velocity_ecef_mps(record, time_utc)
    return positions, velocities, prn_list


def geodetic_to_ecef(latitude_deg: float, longitude_deg: float, altitude_m: float) -> np.ndarray:
    """Convert WGS-84 geodetic coordinates to ECEF."""
    lat = radians(latitude_deg)
    lon = radians(longitude_deg)
    sin_lat = sin(lat)
    cos_lat = cos(lat)
    sin_lon = sin(lon)
    cos_lon = cos(lon)
    prime_vertical = WGS84_A_M / sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (prime_vertical + altitude_m) * cos_lat * cos_lon
    y = (prime_vertical + altitude_m) * cos_lat * sin_lon
    z = (prime_vertical * (1.0 - WGS84_E2) + altitude_m) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_geodetic(receiver_ecef_m: np.ndarray) -> tuple[float, float, float]:
    """Convert ECEF to approximate WGS-84 geodetic latitude/longitude/altitude."""
    x, y, z = np.asarray(receiver_ecef_m, dtype=float)
    lon = atan2(y, x)
    p = sqrt(x**2 + y**2)
    lat = atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(5):
        sin_lat = sin(lat)
        n = WGS84_A_M / sqrt(1.0 - WGS84_E2 * sin_lat**2)
        alt = p / cos(lat) - n
        lat = atan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
    sin_lat = sin(lat)
    n = WGS84_A_M / sqrt(1.0 - WGS84_E2 * sin_lat**2)
    alt = p / cos(lat) - n
    return lat, lon, alt


def elevation_azimuth_deg(satellite_position_ecef_m: np.ndarray, receiver_ecef_m: np.ndarray) -> tuple[float, float]:
    """Return elevation and azimuth angles for a receiver-satellite geometry."""
    rx = np.asarray(receiver_ecef_m, dtype=float)
    sat = np.asarray(satellite_position_ecef_m, dtype=float)
    delta = sat - rx
    lat, lon, _ = ecef_to_geodetic(rx)

    sin_lat = sin(lat)
    cos_lat = cos(lat)
    sin_lon = sin(lon)
    cos_lon = cos(lon)

    transform = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ]
    )
    east, north, up = transform @ delta
    horizontal = sqrt(east**2 + north**2)
    elevation = np.degrees(atan2(up, horizontal))
    azimuth = (np.degrees(atan2(east, north)) + 360.0) % 360.0
    return elevation, azimuth


def _validate_requested_times_within_nav_coverage(
    records: Iterable[GpsBroadcastEphemeris],
    times_utc: Iterable[datetime],
    *,
    max_distance_hours: float = 12.0,
) -> None:
    records_list = list(records)
    times = list(times_utc)
    if not records_list:
        raise ValueError("no GPS ephemeris records available in NAV file")
    if not times:
        raise ValueError("times_utc must not be empty")

    record_times = [record.toc.astimezone(timezone.utc) for record in records_list]
    request_times = [time_utc.astimezone(timezone.utc) for time_utc in times]
    max_distance_seconds = max_distance_hours * 3600.0

    for request_time in request_times:
        nearest_seconds = min(abs((request_time - record_time).total_seconds()) for record_time in record_times)
        if nearest_seconds > max_distance_seconds:
            coverage_start = min(record_times)
            coverage_end = max(record_times)
            request_start = min(request_times)
            request_end = max(request_times)
            raise ValueError(
                "requested time window is outside NAV coverage: "
                f"request=[{request_start.isoformat()} .. {request_end.isoformat()}], "
                f"nav=[{coverage_start.isoformat()} .. {coverage_end.isoformat()}], "
                f"nearest_record_gap_hours={nearest_seconds / 3600.0:.3f}"
            )



def visible_satellite_states(
    records: list[GpsBroadcastEphemeris],
    times_utc: Iterable[datetime],
    receiver_position_ecef_m: np.ndarray,
    *,
    elevation_mask_deg: float = 10.0,
) -> dict[str, np.ndarray | list[str]]:
    """Prepare hover/static receiver arrays plus visibility metadata."""
    times = list(times_utc)
    _validate_requested_times_within_nav_coverage(records, times)
    sat_positions, sat_velocities, prns = satellite_states_for_times(records, times)
    receiver_position = np.asarray(receiver_position_ecef_m, dtype=float)
    receiver_positions = np.repeat(receiver_position[None, :], len(times), axis=0)
    receiver_velocities = np.zeros_like(receiver_positions)

    elevations = np.empty(sat_positions.shape[:2], dtype=float)
    azimuths = np.empty_like(elevations)
    for epoch_idx in range(sat_positions.shape[0]):
        for sat_idx in range(sat_positions.shape[1]):
            elevation, azimuth = elevation_azimuth_deg(sat_positions[epoch_idx, sat_idx], receiver_position)
            elevations[epoch_idx, sat_idx] = elevation
            azimuths[epoch_idx, sat_idx] = azimuth
    visibility_mask = elevations >= elevation_mask_deg
    return {
        "times_utc": np.array(times, dtype=object),
        "prns": prns,
        "satellite_positions_ecef_m": sat_positions,
        "satellite_velocities_ecef_mps": sat_velocities,
        "receiver_positions_ecef_m": receiver_positions,
        "receiver_velocities_ecef_mps": receiver_velocities,
        "elevation_deg": elevations,
        "azimuth_deg": azimuths,
        "visibility_mask": visibility_mask,
    }


def simulate_hover_doppler_scenario(
    rinex_nav_path: str | Path,
    *,
    start_time_utc: datetime,
    duration_seconds: float,
    sample_rate_hz: float,
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
    elevation_mask_deg: float = 10.0,
) -> dict[str, np.ndarray | list[str] | float]:
    """Build a static/hover receiver Doppler dataset from a RINEX NAV file."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    n_epochs = max(1, int(round(duration_seconds * sample_rate_hz)))
    dt_seconds = 1.0 / sample_rate_hz
    times = [
        start_time_utc.astimezone(timezone.utc) + timedelta(seconds=idx * dt_seconds)
        for idx in range(n_epochs)
    ]
    receiver_ecef = geodetic_to_ecef(latitude_deg, longitude_deg, altitude_m)
    scenario = visible_satellite_states(
        parse_rinex_nav_file(rinex_nav_path),
        times,
        receiver_ecef,
        elevation_mask_deg=elevation_mask_deg,
    )
    from gnss_doppler_lab.doppler_simulator import doppler_observation_matrix

    doppler_hz = doppler_observation_matrix(
        scenario["satellite_positions_ecef_m"],
        scenario["satellite_velocities_ecef_mps"],
        scenario["receiver_positions_ecef_m"],
        scenario["receiver_velocities_ecef_mps"],
    )
    return {
        **scenario,
        "sample_rate_hz": sample_rate_hz,
        "doppler_hz": doppler_hz,
    }
