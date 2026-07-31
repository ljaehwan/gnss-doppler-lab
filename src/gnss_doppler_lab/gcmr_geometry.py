"""GPS broadcast-ephemeris geometry for GCMR Doppler features.

Implements the user algorithm in IS-GPS-200: broadcast orbit propagation in
ECEF, finite-difference velocity, receiver look angles, and static-receiver L1
Doppler.  Inputs are GPS time-of-week (TOW); satellite clock corrections and
signal travel-time/Sagnac corrections are intentionally outside this layer.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping
import xml.etree.ElementTree as ET

from .trajectory import ecef_to_llh

GPS_WEEK_SECONDS = 604_800.0
GPS_HALF_WEEK_SECONDS = GPS_WEEK_SECONDS / 2.0
GPS_MU_M3_S2 = 3.986005e14
EARTH_ROTATION_RAD_S = 7.2921151467e-5
SPEED_OF_LIGHT_M_S = 299_792_458.0
GPS_L1_FREQUENCY_HZ = 1_575_420_000.0
GPS_L1_WAVELENGTH_M = SPEED_OF_LIGHT_M_S / GPS_L1_FREQUENCY_HZ


@dataclass(frozen=True)
class GpsEphemeris:
    """Required orbital subset of GNSS-SDR's ``Gps_Ephemeris`` record."""
    PRN: int
    M_0: float
    delta_n: float
    ecc: float
    sqrtA: float
    OMEGA_0: float
    i_0: float
    omega: float
    OMEGAdot: float
    idot: float
    Cuc: float
    Cus: float
    Crc: float
    Crs: float
    Cic: float
    Cis: float
    toe: float
    WN: int
    toc: float | None = None
    decoded_tow: float | None = None
    SV_health: int | None = None
    SV_accuracy: float | None = None
    fit_interval_flag: int | None = None

    @property
    def prn(self): return self.PRN
    @property
    def sqrt_a(self): return self.sqrtA
    @property
    def omega_dot(self): return self.OMEGAdot
    @property
    def week(self): return self.WN


@dataclass(frozen=True)
class LookAngles:
    los_ecef: tuple[float, float, float]
    range_m: float
    elevation_deg: float
    azimuth_deg: float


@dataclass(frozen=True)
class SatelliteObservation:
    prn: int
    position_ecef_m: tuple[float, float, float]
    velocity_ecef_mps: tuple[float, float, float]
    los_ecef: tuple[float, float, float]
    range_m: float
    elevation_deg: float
    azimuth_deg: float
    range_rate_mps: float
    predicted_l1_doppler_hz: float


_XML_FIELDS = (
    "PRN", "M_0", "delta_n", "ecc", "sqrtA", "OMEGA_0", "i_0",
    "omega", "OMEGAdot", "idot", "Cuc", "Cus", "Crc", "Crs", "Cic",
    "Cis", "toe", "WN",
)
_OPTIONAL_XML_FIELDS = {"toc": "toc", "tow": "decoded_tow", "SV_health": "SV_health",
                        "SV_accuracy": "SV_accuracy", "fit_interval_flag": "fit_interval_flag"}


def _finite(name, value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _local_name(tag):
    return tag.rsplit("}", 1)[-1]


def _direct_text(element, name):
    for child in element:
        if _local_name(child.tag) == name:
            if child.text is None or not child.text.strip():
                break
            return child.text.strip()
    raise ValueError(f"ephemeris is missing {name}")


def parse_gnss_sdr_gps_ephemeris_xml(path) -> dict[int, GpsEphemeris]:
    """Parse a GNSS-SDR Boost ``gps_ephemeris.xml`` archive.

    Only the orbital fields needed by this geometry layer are accepted.  A
    malformed archive, incomplete record, duplicate PRN, or nonfinite value is
    rejected rather than producing partial geometry.
    """
    try:
        root = ET.parse(Path(path)).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"invalid GPS ephemeris XML: {exc}") from exc
    maps = [node for node in root.iter() if "ephemeris_map" in _local_name(node.tag)]
    if len(maps) != 1:
        raise ValueError("GPS ephemeris XML must contain exactly one ephemeris_map")
    map_node = maps[0]
    items = [node for node in map_node if _local_name(node.tag) == "item"]
    if not items:
        raise ValueError("GPS ephemeris XML contains no ephemerides")
    result = {}
    for item in items:
        second = next((node for node in item if _local_name(node.tag) == "second"), None)
        if second is None:
            raise ValueError("ephemeris map item is missing second")
        raw = {name: _direct_text(second, name) for name in _XML_FIELDS}
        values = {name: _finite(name, raw[name]) for name in _XML_FIELDS}
        direct = {_local_name(child.tag): child.text.strip() for child in second
                  if child.text is not None and child.text.strip()}
        for xml_name, field_name in _OPTIONAL_XML_FIELDS.items():
            if xml_name in direct:
                values[field_name] = _finite(xml_name, direct[xml_name])
        prn_value, week_value = values["PRN"], values["WN"]
        if not prn_value.is_integer() or not 1 <= prn_value <= 32:
            raise ValueError("PRN must be an integer in [1, 32]")
        if not week_value.is_integer() or week_value < 0:
            raise ValueError("WN must be a nonnegative integer")
        values["PRN"], values["WN"] = int(prn_value), int(week_value)
        for name in ("SV_health", "fit_interval_flag"):
            if name in values:
                if not values[name].is_integer(): raise ValueError(f"{name} must be an integer")
                values[name] = int(values[name])
        eph = GpsEphemeris(**values)
        _validate_ephemeris(eph)
        map_prn = _finite("map PRN", _direct_text(item, "first"))
        if not map_prn.is_integer() or int(map_prn) != eph.PRN:
            raise ValueError("ephemeris map PRN does not match record PRN")
        if eph.PRN in result:
            raise ValueError(f"duplicate ephemeris for PRN {eph.PRN}")
        result[eph.PRN] = eph
    return result


def _validate_ephemeris(eph):
    if not isinstance(eph, GpsEphemeris):
        raise ValueError("ephemeris must be GpsEphemeris")
    for field in _XML_FIELDS:
        _finite(field, getattr(eph, field))
    prn = _finite("PRN", eph.PRN)
    week = _finite("WN", eph.WN)
    if isinstance(eph.PRN, bool) or not prn.is_integer() or not 1 <= prn <= 32:
        raise ValueError("PRN must be an integer in [1, 32]")
    if isinstance(eph.WN, bool) or not week.is_integer() or week < 0:
        raise ValueError("WN must be a nonnegative integer")
    if not 0.0 <= eph.ecc <= 0.1:
        raise ValueError("ecc must be in the plausible GPS range [0, 0.1]")
    if eph.sqrtA <= 0.0:
        raise ValueError("sqrtA must be positive")
    if not 0.0 <= eph.toe < GPS_WEEK_SECONDS:
        raise ValueError("toe must be a GPS time-of-week")
    for name in ("toc", "decoded_tow"):
        value = getattr(eph, name)
        if value is not None and not 0.0 <= _finite(name, value) < GPS_WEEK_SECONDS:
            raise ValueError(f"{name} must be a GPS time-of-week")
    for name in ("SV_health", "fit_interval_flag"):
        value = getattr(eph, name)
        if value is not None and (isinstance(value, bool) or not _finite(name, value).is_integer()):
            raise ValueError(f"{name} must be an integer")
    if eph.SV_accuracy is not None: _finite("SV_accuracy", eph.SV_accuracy)

def ephemeris_health_selection(ephemerides, *, tracked_prns, min_prns=None):
    # Only an explicit zero broadcast health word is scientifically healthy.
    tracked = sorted(set(tracked_prns))
    if any(isinstance(prn, bool) or not isinstance(prn, int) or not 1 <= prn <= 32 for prn in tracked):
        raise ValueError("tracked PRNs must be integers in [1, 32]")
    healthy = {}
    excluded = {}
    for prn, eph in sorted(ephemerides.items()):
        _validate_ephemeris(eph)
        if int(prn) != eph.PRN: raise ValueError(f"ephemeris key does not match PRN {eph.PRN}")
        if eph.SV_health == 0: healthy[eph.PRN] = eph
        else: excluded[eph.PRN] = eph.SV_health
    healthy_tracked = sorted(set(tracked).intersection(healthy))
    report = {"health_acceptance_rule": "SV_health == 0",
        "healthy_ephemeris_prns": sorted(healthy), "healthy_tracked_prns": healthy_tracked,
        "excluded_ephemeris_health_by_prn": excluded,
        "excluded_tracking_prns": sorted(set(tracked).difference(healthy)),
        "tracked_prns_without_ephemeris": sorted(set(tracked).difference(ephemerides))}
    if min_prns is not None:
        if isinstance(min_prns, bool) or int(min_prns) != min_prns or min_prns < 1: raise ValueError("min_prns must be a positive integer")
        if len(healthy_tracked) < int(min_prns): raise ValueError(f"fewer than min healthy tracked PRNs: {len(healthy_tracked)} < {int(min_prns)}")
    return healthy, report


def validate_ephemeris_time_alignment(ephemerides, *, full_gps_week, recording_start_tow_s,
                                      max_toe_age_s, week_modulus=1024):
    """Fail closed on broadcast epoch/week mismatch; snapshot TOW is informational.

    Saved GNSS-SDR maps may be end-of-run snapshots, so decoded TOW availability
    and its relation to recording start are reported but are not a causality gate.
    """
    if not ephemerides: raise ValueError("at least one ephemeris is required")
    week = _finite("full_gps_week", full_gps_week)
    modulus = _finite("week_modulus", week_modulus)
    if not week.is_integer() or week < 0: raise ValueError("full GPS week must be a nonnegative integer")
    if not modulus.is_integer() or modulus <= 0: raise ValueError("week modulus must be a positive integer")
    start = _tow(recording_start_tow_s)
    maximum = _finite("max_toe_age_s", max_toe_age_s)
    if maximum < 0 or maximum > GPS_HALF_WEEK_SECONDS: raise ValueError("max_toe_age_s must be in [0, 302400]")
    decoded=[]; toe_ages={}; toc_available=[]; health={}; fit={}
    for prn,eph in sorted(ephemerides.items()):
        _validate_ephemeris(eph)
        if int(prn) != eph.PRN: raise ValueError(f"ephemeris key does not match PRN {eph.PRN}")
        if eph.WN != int(week) % int(modulus):
            raise ValueError(f"GPS week modulo mismatch for PRN {eph.PRN}: WN {eph.WN}")
        age=abs(_week_delta(start,eph.toe));toe_ages[eph.PRN]=age
        if age > maximum + 1e-9: raise ValueError(f"ephemeris toe age exceeds maximum for PRN {eph.PRN}")
        if eph.SV_health is not None: health[eph.PRN]=eph.SV_health
        if eph.fit_interval_flag is not None: fit[eph.PRN]=eph.fit_interval_flag
        if eph.toc is not None: toc_available.append(eph.PRN)
        if eph.decoded_tow is not None: decoded.append(eph.decoded_tow)
    if not decoded: relation="unavailable"
    elif all(_week_delta(value,start) >= 0 for value in decoded): relation="after_recording_start_allowed_offline_oracle"
    else: relation="mixed_or_before_recording_start_allowed_offline_oracle"
    return {"full_gps_week":int(week),"week_modulus":int(modulus),
            "recording_start_tow_s":start,"max_toe_age_s":maximum,
            "toe_age_s_by_prn":toe_ages,"toc_available_prns":toc_available,
            "health_by_prn":health,"fit_interval_flag_by_prn":fit,
            "decoded_snapshot_available":bool(decoded),"decoded_snapshot_tow_s":decoded,
            "decoded_snapshot_relation":relation}


def _tow(value):
    tow = _finite("tow", value)
    if not 0.0 <= tow < GPS_WEEK_SECONDS:
        raise ValueError("tow must be in [0, 604800)")
    return tow


def _week_delta(tow, reference):
    return (tow - reference + GPS_HALF_WEEK_SECONDS) % GPS_WEEK_SECONDS - GPS_HALF_WEEK_SECONDS


def _solve_kepler(mean_anomaly, eccentricity, *, max_iterations=20, tolerance=1e-14):
    """Solve E - e sin(E) = M by checked Newton iteration."""
    mean = _finite("mean anomaly", mean_anomaly); ecc = _finite("eccentricity", eccentricity)
    if not 0.0 <= ecc <= 0.1:
        raise ValueError("eccentricity must be in the plausible GPS range [0, 0.1]")
    if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    tol = _finite("tolerance", tolerance)
    if tol <= 0: raise ValueError("tolerance must be positive")
    anomaly = mean
    for _ in range(int(max_iterations)):
        residual = anomaly - ecc * math.sin(anomaly) - mean
        derivative = 1.0 - ecc * math.cos(anomaly)
        updated = anomaly - residual / derivative
        if abs(updated - anomaly) <= tol:
            return updated
        anomaly = updated
    raise RuntimeError("Kepler solver did not converge")


def satellite_position_ecef(ephemeris: GpsEphemeris, tow) -> tuple[float, float, float]:
    """Propagate one GPS broadcast orbit to TOW using the IS-GPS-200 model."""
    _validate_ephemeris(ephemeris)
    tow = _tow(tow)
    tk = _week_delta(tow, ephemeris.toe)
    semi_major = ephemeris.sqrtA ** 2
    mean_motion = math.sqrt(GPS_MU_M3_S2 / semi_major ** 3) + ephemeris.delta_n
    mean_anomaly = ephemeris.M_0 + mean_motion * tk
    eccentric_anomaly = _solve_kepler(mean_anomaly, ephemeris.ecc)
    true_anomaly = math.atan2(
        math.sqrt(1.0 - ephemeris.ecc ** 2) * math.sin(eccentric_anomaly),
        math.cos(eccentric_anomaly) - ephemeris.ecc,
    )
    phi = true_anomaly + ephemeris.omega
    sin2, cos2 = math.sin(2.0 * phi), math.cos(2.0 * phi)
    argument = phi + ephemeris.Cus * sin2 + ephemeris.Cuc * cos2
    radius = semi_major * (1.0 - ephemeris.ecc * math.cos(eccentric_anomaly)) + ephemeris.Crs * sin2 + ephemeris.Crc * cos2
    inclination = ephemeris.i_0 + ephemeris.idot * tk + ephemeris.Cis * sin2 + ephemeris.Cic * cos2
    x_orbit, y_orbit = radius * math.cos(argument), radius * math.sin(argument)
    ascending = ephemeris.OMEGA_0 + (ephemeris.OMEGAdot - EARTH_ROTATION_RAD_S) * tk - EARTH_ROTATION_RAD_S * ephemeris.toe
    result = (
        x_orbit * math.cos(ascending) - y_orbit * math.cos(inclination) * math.sin(ascending),
        x_orbit * math.sin(ascending) + y_orbit * math.cos(inclination) * math.cos(ascending),
        y_orbit * math.sin(inclination),
    )
    if not all(math.isfinite(value) for value in result):
        raise ValueError("propagated satellite position is nonfinite")
    return result


def satellite_velocity_ecef(ephemeris, tow, *, difference_s=0.5):
    """Obtain ECEF velocity by symmetric finite difference across GPS TOW."""
    tow = _tow(tow)
    step = _finite("difference_s", difference_s)
    if not 0.0 < step < GPS_HALF_WEEK_SECONDS:
        raise ValueError("difference_s must be positive and less than half a GPS week")
    before = satellite_position_ecef(ephemeris, (tow - step) % GPS_WEEK_SECONDS)
    after = satellite_position_ecef(ephemeris, (tow + step) % GPS_WEEK_SECONDS)
    result = tuple((high - low) / (2.0 * step) for low, high in zip(before, after))
    if not all(math.isfinite(value) for value in result):
        raise ValueError("propagated satellite velocity is nonfinite")
    return result


def _ecef(name, values):
    try:
        if len(values) != 3: raise ValueError
        result = tuple(_finite(f"{name}[{index}]", value) for index, value in enumerate(values))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three finite values") from exc
    return result


def look_angles(receiver_ecef, satellite_ecef) -> LookAngles:
    """Derive receiver-to-satellite LOS and WGS-84 azimuth/elevation."""
    receiver = _ecef("receiver_ecef", receiver_ecef)
    satellite = _ecef("satellite_ecef", satellite_ecef)
    if math.dist((0.0, 0.0, 0.0), receiver) < 1.0:
        raise ValueError("receiver_ecef must not be at the Earth center")
    delta = tuple(s - r for s, r in zip(satellite, receiver))
    range_m = math.dist((0.0, 0.0, 0.0), delta)
    if range_m <= 0.0:
        raise ValueError("receiver and satellite positions must differ")
    los = tuple(value / range_m for value in delta)
    latitude, longitude, _ = ecef_to_llh(*receiver)
    lat, lon = math.radians(latitude), math.radians(longitude)
    east = -math.sin(lon) * los[0] + math.cos(lon) * los[1]
    north = (-math.sin(lat) * math.cos(lon) * los[0]
             - math.sin(lat) * math.sin(lon) * los[1] + math.cos(lat) * los[2])
    up = (math.cos(lat) * math.cos(lon) * los[0]
          + math.cos(lat) * math.sin(lon) * los[1] + math.sin(lat) * los[2])
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, up))))
    azimuth = 0.0 if math.hypot(east, north) < 1e-14 else math.degrees(math.atan2(east, north)) % 360.0
    return LookAngles(los, range_m, elevation, azimuth)


def satellite_observation(receiver_ecef, ephemeris, tow, *, difference_s=0.5):
    """Return static-receiver geometry and predicted GPS L1 Doppler."""
    receiver = _ecef("receiver_ecef", receiver_ecef)
    position = satellite_position_ecef(ephemeris, tow)
    velocity = satellite_velocity_ecef(ephemeris, tow, difference_s=difference_s)
    angles = look_angles(receiver, position)
    range_rate = sum(v * los for v, los in zip(velocity, angles.los_ecef))
    doppler = -range_rate / GPS_L1_WAVELENGTH_M
    if not math.isfinite(doppler):
        raise ValueError("predicted Doppler is nonfinite")
    return SatelliteObservation(ephemeris.PRN, position, velocity, angles.los_ecef,
                                angles.range_m, angles.elevation_deg,
                                angles.azimuth_deg, range_rate, doppler)


def common_clock_removed_residuals(observed_hz: Mapping[int, float], predicted_hz: Mapping[int, float], *, visible_prns: Iterable[int] | None = None) -> dict[int, float]:
    """Subtract the across-visible-PRN median from Doppler residuals.

    Residual sign is ``observed - predicted``.  Missing PRNs and nonfinite
    measurements fail closed so a changing satellite set cannot silently bias
    the common-clock estimate.
    """
    prns = list(visible_prns if visible_prns is not None else observed_hz.keys())
    if not prns:
        raise ValueError("at least one visible PRN is required")
    if len(set(prns)) != len(prns):
        raise ValueError("visible PRNs must be unique")
    raw = {}
    for prn in prns:
        if prn not in observed_hz or prn not in predicted_hz:
            raise ValueError(f"missing observed or predicted Doppler for PRN {prn}")
        raw[prn] = _finite(f"observed Doppler for PRN {prn}", observed_hz[prn]) - _finite(f"predicted Doppler for PRN {prn}", predicted_hz[prn])
    common = median(raw.values())
    return {prn: value - common for prn, value in raw.items()}


def predict_static_l1_doppler(receiver_ecef, ephemeris, tow, *, difference_s=0.5):
    """Predict GPS L1 Doppler in Hz for a receiver stationary in ECEF."""
    return satellite_observation(
        receiver_ecef, ephemeris, tow, difference_s=difference_s
    ).predicted_l1_doppler_hz
