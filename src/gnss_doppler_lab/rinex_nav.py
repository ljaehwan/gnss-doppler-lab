"""Strict RINEX 2 GPS broadcast-navigation selection for offline validation."""
from __future__ import annotations

import gzip
import math
from pathlib import Path

from .gcmr_geometry import GPS_HALF_WEEK_SECONDS, GPS_WEEK_SECONDS, GpsEphemeris


def _rinex_float(text: str) -> float:
    value = float(text.replace("D", "E").replace("d", "e"))
    if not math.isfinite(value):
        raise ValueError("RINEX navigation value must be finite")
    return value


def _orbit_values(line: str) -> list[float]:
    padded = line.rstrip("\n").ljust(79)
    return [_rinex_float(padded[3 + 19 * index:22 + 19 * index])
            for index in range(4)]


def _week_delta(value: float, reference: float) -> float:
    return (value - reference + GPS_HALF_WEEK_SECONDS) % GPS_WEEK_SECONDS - GPS_HALF_WEEK_SECONDS


def parse_rinex2_gps_nav_gz(
    path: str | Path, *, full_gps_week: int, target_tow_s: float,
    maximum_toe_age_s: float,
) -> dict[int, GpsEphemeris]:
    """Select the nearest healthy/flagged broadcast record for each GPS PRN."""
    target = float(target_tow_s)
    maximum = float(maximum_toe_age_s)
    if not 0.0 <= target < GPS_WEEK_SECONDS:
        raise ValueError("target_tow_s must be a GPS time-of-week")
    if not 0.0 <= maximum <= GPS_HALF_WEEK_SECONDS:
        raise ValueError("maximum_toe_age_s is outside the supported range")
    try:
        with gzip.open(Path(path), "rt", encoding="ascii") as handle:
            lines = handle.readlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"invalid gzip RINEX navigation file: {exc}") from exc
    header_end = next(
        (index for index, line in enumerate(lines) if "END OF HEADER" in line),
        None,
    )
    if header_end is None:
        raise ValueError("RINEX navigation file is missing END OF HEADER")
    candidates: dict[int, list[GpsEphemeris]] = {}
    index = header_end + 1
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        if index + 7 >= len(lines):
            raise ValueError("truncated RINEX GPS navigation record")
        block = lines[index:index + 8]
        index += 8
        try:
            prn = int(block[0][0:2])
            line2, line3, line4 = map(_orbit_values, block[1:4])
            line5, line6, line7 = map(_orbit_values, block[4:7])
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed RINEX GPS navigation record") from exc
        if not 1 <= prn <= 32:
            raise ValueError(f"RINEX GPS PRN is outside [1, 32]: {prn}")
        broadcast_week = int(round(line6[2]))
        if broadcast_week % 1024 != int(full_gps_week) % 1024:
            continue
        eph = GpsEphemeris(
            PRN=prn,
            M_0=line2[3],
            delta_n=line2[2],
            ecc=line3[1],
            sqrtA=line3[3],
            OMEGA_0=line4[2],
            i_0=line5[0],
            omega=line5[2],
            OMEGAdot=line5[3],
            idot=line6[0],
            Cuc=line3[0],
            Cus=line3[2],
            Crc=line5[1],
            Crs=line2[1],
            Cic=line4[1],
            Cis=line4[3],
            toe=line4[0],
            WN=broadcast_week % 1024,
            SV_health=int(round(line7[1])),
            SV_accuracy=line7[0],
        )
        if abs(_week_delta(eph.toe, target)) <= maximum:
            candidates.setdefault(prn, []).append(eph)
    selected: dict[int, GpsEphemeris] = {}
    for prn, records in sorted(candidates.items()):
        selected[prn] = min(
            records,
            key=lambda eph: (
                abs(_week_delta(eph.toe, target)),
                _week_delta(eph.toe, target) > 0.0,
            ),
        )
    if not selected:
        raise ValueError("no time-aligned GPS ephemerides in RINEX navigation file")
    return selected
