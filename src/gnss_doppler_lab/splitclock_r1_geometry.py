"""Galileo E1B geometry and observable reconstruction for SPLITCLOCK R1."""

from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .splitclock_r1_contract import GALILEO_E1_WAVELENGTH_M, SPEED_OF_LIGHT_MPS

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
MU = 3.986004418e14
OMEGA_E = 7.2921151467e-5
RELATIVISTIC_F = -4.442807633e-10
WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
TRACE_HEADER = struct.Struct("<8sIIIIdff64s48s9fI")
TRACE_MAGIC = b"TRC1MS02"


@dataclass(frozen=True)
class Observation:
    system_time: datetime
    utc_time: datetime
    receiver_epoch: int
    prn: int
    pseudorange_m: float
    carrier_cycles: float
    doppler_hz: float
    cn0_db_hz: float
    lli: int


@dataclass(frozen=True)
class Ephemeris:
    prn: int
    toc: datetime
    af0: float
    af1: float
    af2: float
    crs: float
    delta_n: float
    m0: float
    cuc: float
    eccentricity: float
    cus: float
    sqrt_a: float
    toe: float
    cic: float
    omega0: float
    cis: float
    i0: float
    crc: float
    omega: float
    omega_dot: float
    idot: float
    week: int
    health: float
    bgd_e5a_e1_s: float
    bgd_e5b_e1_s: float


@dataclass
class ScenarioPanel:
    scenario: str
    epochs: list[datetime]
    prns: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    cn0: np.ndarray
    cycle_slip: np.ndarray
    reacquisition: np.ndarray
    geometry_rows: list[dict[str, Any]]
    alignment_errors_s: list[float]


def _float(text: str) -> float:
    value = text.strip().replace("D", "E")
    return float(value) if value else float("nan")


def _nav_fields(line: str, start: int) -> list[float]:
    return [_float(line[index : index + 19]) for index in range(start, min(len(line), start + 4 * 19), 19)]


def parse_rinex_observations(path: Path) -> tuple[list[Observation], dict[str, Any]]:
    lines = path.read_text(encoding="ascii").splitlines()
    end = next(i for i, line in enumerate(lines) if "END OF HEADER" in line)
    leap_rows = [line for line in lines[:end] if "LEAP SECONDS" in line]
    if len(leap_rows) != 1:
        raise ValueError("RINEX observation header must contain one LEAP SECONDS row")
    leap_seconds = int(leap_rows[0][:6])
    time_rows = [line for line in lines[:end] if "TIME OF FIRST OBS" in line]
    if len(time_rows) != 1 or "GAL" not in time_rows[0]:
        raise ValueError("RINEX observations are not explicitly Galileo system time")
    type_rows = [line for line in lines[:end] if "SYS / # / OBS TYPES" in line]
    if len(type_rows) != 1 or "E    4 C1B L1B D1B S1B" not in type_rows[0]:
        raise ValueError("RINEX observable tuple differs from C1B/L1B/D1B/S1B")
    observations: list[Observation] = []
    system_time = None
    receiver_epoch = -1
    for line in lines[end + 1 :]:
        if line.startswith(">"):
            parts = line[1:].split()
            second = float(parts[5]); whole = int(second)
            system_time = datetime(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), whole, tzinfo=timezone.utc) + timedelta(seconds=second - whole)
            receiver_epoch += 1
        elif line.startswith("E") and system_time is not None:
            fields = [line[3 + 16 * j : 3 + 16 * (j + 1)] for j in range(4)]
            values = [_float(field[:14]) for field in fields]
            lli_text = fields[1][14:15].strip()
            observations.append(Observation(system_time, system_time - timedelta(seconds=leap_seconds), receiver_epoch, int(line[1:3]), *values, int(lli_text) if lli_text else 0))
    return observations, {"leap_seconds": leap_seconds, "time_system": "GAL", "observation_types": ["C1B", "L1B", "D1B", "S1B"]}


def parse_rinex_navigation(path: Path) -> tuple[list[Ephemeris], dict[str, Any]]:
    lines = path.read_text(encoding="ascii").splitlines()
    end = next(i for i, line in enumerate(lines) if "END OF HEADER" in line)
    values: list[Ephemeris] = []
    index = end + 1
    while index < len(lines):
        if not lines[index].startswith("E"):
            index += 1; continue
        block = lines[index : index + 8]
        if len(block) != 8:
            raise ValueError("truncated Galileo navigation block")
        parts = block[0][:23].split()
        toc = datetime(int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6]), tzinfo=timezone.utc)
        clock = _nav_fields(block[0], 23)
        rows = [_nav_fields(line, 4) for line in block[1:]]
        if any(len(row) < 4 for row in rows[:6]) or len(clock) < 3:
            raise ValueError("malformed Galileo navigation fields")
        values.append(Ephemeris(
            int(parts[0][1:]), toc, clock[0], clock[1], clock[2], rows[0][1], rows[0][2], rows[0][3],
            rows[1][0], rows[1][1], rows[1][2], rows[1][3], rows[2][0], rows[2][1], rows[2][2], rows[2][3],
            rows[3][0], rows[3][1], rows[3][2], rows[3][3], rows[4][0], int(rows[4][2]), rows[5][0], rows[5][1], rows[5][2],
        ))
        index += 8
    if not values:
        raise ValueError("no Galileo ephemerides")
    return values, {"record_count": len(values), "prns": sorted({value.prn for value in values}), "format": "RINEX3_GALILEO_NAV"}


def parse_gpx_positions(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    root = ET.parse(path).getroot()
    points = []
    for element in root.iter():
        if not element.tag.endswith("trkpt"):
            continue
        lat = float(element.attrib["lat"]); lon = float(element.attrib["lon"])
        elevation = next((float(child.text) for child in element if child.tag.endswith("ele") and child.text), None)
        timestamp = next((child.text for child in element if child.tag.endswith("time") and child.text), None)
        if elevation is None or timestamp is None:
            continue
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        points.append((dt.timestamp(), *geodetic_to_ecef(lat, lon, elevation)))
    if len(points) < 2:
        raise ValueError("GPX has fewer than two complete points")
    array = np.asarray(points, dtype=float)
    order = np.argsort(array[:, 0])
    array = array[order]
    return array[:, 0], array[:, 1:4], {"source": str(path), "coordinate_system": "WGS84_geodetic_to_ECEF", "point_count": len(array)}


def geodetic_to_ecef(latitude_deg: float, longitude_deg: float, height_m: float) -> np.ndarray:
    latitude = math.radians(latitude_deg); longitude = math.radians(longitude_deg)
    e2 = WGS84_F * (2.0 - WGS84_F)
    n = WGS84_A / math.sqrt(1.0 - e2 * math.sin(latitude) ** 2)
    return np.asarray([(n + height_m) * math.cos(latitude) * math.cos(longitude), (n + height_m) * math.cos(latitude) * math.sin(longitude), (n * (1.0 - e2) + height_m) * math.sin(latitude)])


def receiver_position(timestamp: float, times: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, float]:
    if timestamp < times[0] - 1.0 or timestamp > times[-1] + 1.0:
        raise ValueError("receiver position outside GPX support")
    nearest = float(np.min(np.abs(times - timestamp)))
    return np.asarray([np.interp(timestamp, times, positions[:, axis]) for axis in range(3)]), nearest


def system_seconds(timestamp: datetime) -> float:
    return (timestamp - GPS_EPOCH).total_seconds()


def wrap_week(seconds: float) -> float:
    while seconds > 302400.0: seconds -= 604800.0
    while seconds < -302400.0: seconds += 604800.0
    return seconds


def select_ephemeris(ephemerides: list[Ephemeris], prn: int, timestamp: datetime) -> Ephemeris:
    candidates = [value for value in ephemerides if value.prn == prn and np.isfinite(value.health)]
    if not candidates:
        raise ValueError(f"no ephemeris for E{prn:02d}")
    sow = system_seconds(timestamp) % 604800.0
    selected = min(candidates, key=lambda value: abs(wrap_week(sow - value.toe)))
    if abs(wrap_week(sow - selected.toe)) > 7200.0:
        raise ValueError(f"ephemeris age exceeds 7200 s for E{prn:02d}")
    return selected


def satellite_state(eph: Ephemeris, timestamp: datetime) -> tuple[np.ndarray, float, float]:
    sow = system_seconds(timestamp) % 604800.0
    tk = wrap_week(sow - eph.toe)
    semi_major = eph.sqrt_a ** 2
    mean_motion = math.sqrt(MU / semi_major ** 3) + eph.delta_n
    mean_anomaly = eph.m0 + mean_motion * tk
    eccentric = mean_anomaly
    for _ in range(20):
        update = (eccentric - eph.eccentricity * math.sin(eccentric) - mean_anomaly) / (1.0 - eph.eccentricity * math.cos(eccentric))
        eccentric -= update
        if abs(update) < 1e-13: break
    true_anomaly = math.atan2(math.sqrt(1.0 - eph.eccentricity ** 2) * math.sin(eccentric), math.cos(eccentric) - eph.eccentricity)
    phi = true_anomaly + eph.omega
    du = eph.cus * math.sin(2.0 * phi) + eph.cuc * math.cos(2.0 * phi)
    dr = eph.crs * math.sin(2.0 * phi) + eph.crc * math.cos(2.0 * phi)
    di = eph.cis * math.sin(2.0 * phi) + eph.cic * math.cos(2.0 * phi)
    u = phi + du; radius = semi_major * (1.0 - eph.eccentricity * math.cos(eccentric)) + dr; inclination = eph.i0 + eph.idot * tk + di
    omega = eph.omega0 + (eph.omega_dot - OMEGA_E) * tk - OMEGA_E * eph.toe
    x_orbit = radius * math.cos(u); y_orbit = radius * math.sin(u)
    position = np.asarray([x_orbit * math.cos(omega) - y_orbit * math.cos(inclination) * math.sin(omega), x_orbit * math.sin(omega) + y_orbit * math.cos(inclination) * math.cos(omega), y_orbit * math.sin(inclination)])
    dt_clock = wrap_week((timestamp - eph.toc).total_seconds())
    clock = eph.af0 + eph.af1 * dt_clock + eph.af2 * dt_clock ** 2 + RELATIVISTIC_F * eph.eccentricity * eph.sqrt_a * math.sin(eccentric)
    return position, clock, eccentric


def apparent_range(ephs: list[Ephemeris], prn: int, receive_system_time: datetime, pseudorange_m: float, receive_utc_timestamp: float, gpx_times: np.ndarray, gpx_positions: np.ndarray) -> tuple[float, dict[str, Any]]:
    travel = pseudorange_m / SPEED_OF_LIGHT_MPS
    transmit = receive_system_time - timedelta(seconds=travel)
    eph = select_ephemeris(ephs, prn, transmit)
    sat, clock_s, _ = satellite_state(eph, transmit)
    sat_plus, _, _ = satellite_state(eph, transmit + timedelta(seconds=0.5))
    sat_minus, _, _ = satellite_state(eph, transmit - timedelta(seconds=0.5))
    sat_velocity = sat_plus - sat_minus
    receiver, alignment = receiver_position(receive_utc_timestamp, gpx_times, gpx_positions)
    for _ in range(2):
        theta = OMEGA_E * travel; c = math.cos(theta); s = math.sin(theta)
        rotated = np.asarray([c * sat[0] + s * sat[1], -s * sat[0] + c * sat[1], sat[2]])
        geometric = float(np.linalg.norm(rotated - receiver)); travel = geometric / SPEED_OF_LIGHT_MPS
    value = geometric - SPEED_OF_LIGHT_MPS * clock_s
    return value, {"satellite_position_ecef_m": rotated.tolist(), "satellite_velocity_ecef_mps": sat_velocity.tolist(), "satellite_clock_s": clock_s, "geometric_range_m": geometric, "sagnac_correction_m": geometric - float(np.linalg.norm(sat - receiver)), "receiver_position_ecef_m": receiver.tolist(), "alignment_error_s": alignment, "ephemeris_toe_sow": eph.toe}


def trace_cadence(receiver_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(receiver_dir.glob("trace_native_1ms_ch_*.bin")):
        with path.open("rb") as stream: payload = stream.read(TRACE_HEADER.size)
        if len(payload) != TRACE_HEADER.size: raise ValueError(f"truncated TRACE header: {path}")
        unpacked = TRACE_HEADER.unpack(payload)
        if unpacked[0] != TRACE_MAGIC: raise ValueError(f"invalid TRACE magic: {path}")
        rows.append({"path": str(path), "sample_rate_hz": unpacked[5], "tap_spacing_chips": unpacked[6], "coherent_integration_s": unpacked[7], "record_size": unpacked[3], "header_size": unpacked[2]})
    if len(rows) != 12: raise ValueError("expected 12 native TRACE files")
    cadences = np.asarray([row["coherent_integration_s"] * 1000.0 for row in rows])
    consistent = bool(np.max(cadences) - np.min(cadences) <= 1e-6)
    return {"status": "PASS" if consistent else "FAIL", "native_trace_cadence_ms": float(np.median(cadences)), "native_trace_nominal_cadence_ms": int(round(float(np.median(cadences)))), "acquisition_coherent_integration_ms": 8, "semantic_separation": True, "files": rows}


def build_panel(scenario: str, observation_path: Path, navigation_path: Path, gpx_path: Path) -> tuple[ScenarioPanel, dict[str, Any]]:
    observations, obs_meta = parse_rinex_observations(observation_path)
    ephemerides, nav_meta = parse_rinex_navigation(navigation_path)
    gpx_times, gpx_positions, gpx_meta = parse_gpx_positions(gpx_path)
    epochs = sorted({value.system_time for value in observations}); prns = np.asarray(sorted({value.prn for value in observations}), dtype=int)
    epoch_index = {value: index for index, value in enumerate(epochs)}; prn_index = {value: index for index, value in enumerate(prns)}
    shape = (len(epochs), len(prns), 3); raw = np.full(shape, np.nan); valid = np.zeros(shape, dtype=bool); cn0 = np.full(shape[:2], np.nan)
    slip = np.zeros(shape[:2], dtype=bool); reacquisition = np.zeros(shape[:2], dtype=bool); apparent = np.full(shape[:2], np.nan)
    geometry_rows: list[dict[str, Any]] = []; alignment_errors: list[float] = []
    obs_by_key = {(value.system_time, value.prn): value for value in observations}
    for observation in observations:
        t = epoch_index[observation.system_time]; p = prn_index[observation.prn]
        try:
            predicted, audit = apparent_range(ephemerides, observation.prn, observation.system_time, observation.pseudorange_m, observation.utc_time.timestamp(), gpx_times, gpx_positions)
            plus, _ = apparent_range(ephemerides, observation.prn, observation.system_time + timedelta(seconds=0.5), observation.pseudorange_m, observation.utc_time.timestamp() + 0.5, gpx_times, gpx_positions)
            minus, _ = apparent_range(ephemerides, observation.prn, observation.system_time - timedelta(seconds=0.5), observation.pseudorange_m, observation.utc_time.timestamp() - 0.5, gpx_times, gpx_positions)
        except ValueError:
            continue
        apparent[t, p] = predicted; raw[t, p, 0] = observation.pseudorange_m - predicted; raw[t, p, 1] = -GALILEO_E1_WAVELENGTH_M * observation.doppler_hz - (plus - minus); valid[t, p, :2] = np.isfinite(raw[t, p, :2]); cn0[t, p] = observation.cn0_db_hz; slip[t, p] = observation.lli != 0
        alignment_errors.append(audit["alignment_error_s"]); geometry_rows.append({"scenario": scenario, "receiver_epoch": t, "prn": observation.prn, **audit})
    for t in range(1, len(epochs)):
        if (epochs[t] - epochs[t - 1]).total_seconds() != 1.0: continue
        for p, prn in enumerate(prns):
            left = obs_by_key.get((epochs[t - 1], int(prn))); right = obs_by_key.get((epochs[t], int(prn)))
            if left is None or right is None or not np.isfinite(apparent[t - 1 : t + 1, p]).all():
                reacquisition[t, p] = right is not None; continue
            raw[t, p, 2] = GALILEO_E1_WAVELENGTH_M * (right.carrier_cycles - left.carrier_cycles) - (apparent[t, p] - apparent[t - 1, p])
            valid[t, p, 2] = np.isfinite(raw[t, p, 2]) and not slip[t, p]
            if abs(GALILEO_E1_WAVELENGTH_M * (right.carrier_cycles - left.carrier_cycles) + GALILEO_E1_WAVELENGTH_M * 0.5 * (left.doppler_hz + right.doppler_hz)) > 5.0:
                slip[t, p] = True; valid[t, p, 2] = False
    panel = ScenarioPanel(scenario, epochs, prns, raw, valid, cn0, slip, reacquisition, geometry_rows, alignment_errors)
    return panel, {"observation": obs_meta, "navigation": nav_meta, "receiver_position": gpx_meta}
