"""WGS-84 trajectory generation and strict gps-sdr-sim CSV validation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
# Deliberately below the WGS-84 polar radius (~6,356.8 km) so ordinary
# below-sea-level trajectories remain valid while deep-Earth inputs do not.
MIN_PHYSICAL_ECEF_RADIUS_M = 6_000_000.0
SAMPLE_RATE_HZ = 10
GENERATOR_SCHEMA = "gnss-doppler-lab.trajectory"
GENERATOR_VERSION = 2
DOIS = {
    "straight": "10.1109/TRO.2007.898976",
    "circle": "10.1109/TRO.2007.898976",
    "parallel-sweep": "10.1109/ICRA.2011.5979707",
}


def _finite(name, value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def llh_to_ecef(latitude_deg, longitude_deg, altitude_m):
    lat = math.radians(latitude_deg)
    lon = math.radians(longitude_deg)
    n = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    return ((n + altitude_m) * math.cos(lat) * math.cos(lon),
            (n + altitude_m) * math.cos(lat) * math.sin(lon),
            (n * (1.0 - WGS84_E2) + altitude_m) * math.sin(lat))


def ecef_to_llh(x, y, z):
    """Convert ECEF to LLH, including a stable polar-axis branch."""
    p = math.hypot(x, y)
    if p < 1e-9:
        if abs(z) < 1e-9:
            raise ValueError("ECEF origin has undefined latitude/longitude")
        b = WGS84_A_M * (1.0 - WGS84_F)
        return (90.0 if z > 0 else -90.0), 0.0, abs(z) - b
    lon = math.atan2(y, x)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(12):
        n = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - n
        updated = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
        if abs(updated - lat) < 1e-14:
            lat = updated
            break
        lat = updated
    n = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    h = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), h


def enu_to_llh(east_m, north_m, up_m, latitude_deg, longitude_deg, altitude_m):
    x0, y0, z0 = llh_to_ecef(latitude_deg, longitude_deg, altitude_m)
    p, l = math.radians(latitude_deg), math.radians(longitude_deg)
    x = x0 - math.sin(l)*east_m - math.sin(p)*math.cos(l)*north_m + math.cos(p)*math.cos(l)*up_m
    y = y0 + math.cos(l)*east_m - math.sin(p)*math.sin(l)*north_m + math.cos(p)*math.sin(l)*up_m
    z = z0 + math.cos(p)*north_m + math.sin(p)*up_m
    return ecef_to_llh(x, y, z)


def llh_to_enu(latitude_deg, longitude_deg, altitude_m, origin_latitude_deg, origin_longitude_deg, origin_altitude_m):
    x, y, z = llh_to_ecef(latitude_deg, longitude_deg, altitude_m)
    x0, y0, z0 = llh_to_ecef(origin_latitude_deg, origin_longitude_deg, origin_altitude_m)
    dx, dy, dz = x-x0, y-y0, z-z0
    p, l = math.radians(origin_latitude_deg), math.radians(origin_longitude_deg)
    return (-math.sin(l)*dx + math.cos(l)*dy,
            -math.sin(p)*math.cos(l)*dx - math.sin(p)*math.sin(l)*dy + math.cos(p)*dz,
            math.cos(p)*math.cos(l)*dx + math.cos(p)*math.sin(l)*dy + math.sin(p)*dz)


def read_trajectory(path, duration_seconds, coordinate_system="llh"):
    duration = _finite("duration_seconds", duration_seconds)
    expected_float = duration * SAMPLE_RATE_HZ
    if duration <= 0 or not expected_float.is_integer():
        raise ValueError("duration_seconds must produce an integer number of 10 Hz rows")
    expected = int(expected_float)
    rows = []
    try:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            for line_number, row in enumerate(csv.reader(stream), 1):
                if len(row) != 4:
                    raise ValueError(f"row {line_number}: expected exactly 4 columns")
                try:
                    values = tuple(float(value) for value in row)
                except ValueError as exc:
                    raise ValueError(f"row {line_number}: header/non-numeric value forbidden") from exc
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"row {line_number}: values must be finite")
                expected_time = (line_number - 1) / SAMPLE_RATE_HZ
                if not math.isclose(values[0], expected_time, rel_tol=0.0, abs_tol=1e-8):
                    raise ValueError(f"row {line_number}: timestamp must be {expected_time:.1f}")
                if coordinate_system == "llh" and not (-90 <= values[1] <= 90 and -180 <= values[2] <= 180):
                    raise ValueError(f"row {line_number}: LLH out of range")
                if (coordinate_system == "ecef" and
                        math.hypot(values[1], values[2], values[3]) < MIN_PHYSICAL_ECEF_RADIUS_M):
                    raise ValueError(
                        f"row {line_number}: ECEF geocentric radius must be at least "
                        f"{MIN_PHYSICAL_ECEF_RADIUS_M:.0f} m"
                    )
                rows.append(values)
    except OSError as exc:
        raise ValueError(str(exc)) from exc
    if len(rows) != expected:
        raise ValueError(f"trajectory requires exactly {expected} rows for {duration:g} s at 10 Hz (found {len(rows)})")
    return rows


def _sweep(distance, leg_length, lane_spacing):
    """C1 lawnmower path: straight legs joined by tangent semicircles."""
    period = 2*leg_length + math.pi*lane_spacing
    cycle = int(distance // period)
    q = distance - cycle*period
    north = 2*cycle*lane_spacing
    if q <= leg_length:
        return q, north
    q -= leg_length
    turn_length = math.pi*lane_spacing/2
    if q <= turn_length:
        angle = q/(lane_spacing/2)
        return leg_length + lane_spacing/2*math.sin(angle), north + lane_spacing/2*(1-math.cos(angle))
    q -= turn_length
    if q <= leg_length:
        return leg_length-q, north+lane_spacing
    q -= leg_length
    angle = q/(lane_spacing/2)
    return -lane_spacing/2*math.sin(angle), north+lane_spacing + lane_spacing/2*(1-math.cos(angle))


def _atomic_write(path, data):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except Exception:
        try: os.unlink(name)
        except FileNotFoundError: pass
        raise


def generate_trajectory(kind, output, *, latitude_deg, longitude_deg, altitude_m,
                        duration_seconds, speed_mps, radius_m=20, leg_length_m=100,
                        lane_spacing_m=20, heading_deg=0, laps=None):
    if kind not in DOIS:
        raise ValueError(f"unknown trajectory kind: {kind}")
    lat = _finite("latitude_deg", latitude_deg)
    lon = _finite("longitude_deg", longitude_deg)
    alt = _finite("altitude_m", altitude_m)
    duration = _finite("duration_seconds", duration_seconds)
    speed = _finite("speed_mps", speed_mps)
    heading = _finite("heading_deg", heading_deg)
    if not -89.9 <= lat <= 89.9:
        raise ValueError("latitude_deg must be in [-89.9, 89.9] for local ENU generation")
    if not -180 <= lon <= 180:
        raise ValueError("longitude_deg must be in [-180, 180]")
    samples_float = duration*SAMPLE_RATE_HZ
    if duration <= 0 or duration > 300 or not samples_float.is_integer():
        raise ValueError("duration_seconds must be positive, <= 300, and represent an integer number of 10 Hz samples")
    if speed <= 0:
        raise ValueError("speed_mps must be positive")
    if kind == "circle":
        radius = _finite("radius_m", radius_m)
        if radius <= 0: raise ValueError("radius_m must be positive")
        if laps is not None:
            lap_count = _finite("laps", laps)
            if lap_count <= 0 or not lap_count.is_integer():
                raise ValueError("laps must be a positive integer")
            angular_rate = 2*math.pi*lap_count/duration
            effective_speed = angular_rate*radius
        else:
            angular_rate = speed/radius
            lap_count = angular_rate*duration/(2*math.pi)
            effective_speed = speed
    elif kind == "parallel-sweep":
        leg_length = _finite("leg_length_m", leg_length_m)
        lane_spacing = _finite("lane_spacing_m", lane_spacing_m)
        if leg_length <= 0 or lane_spacing <= 0: raise ValueError("sweep distances must be positive")
        effective_speed = speed
        lap_count = None
    else:
        effective_speed = speed
        lap_count = None

    sample_count = int(samples_float)
    hd = math.radians(heading)
    rows = []
    for index in range(sample_count):
        time_s = index/SAMPLE_RATE_HZ
        distance = effective_speed*time_s
        if kind == "straight": east0, north0 = 0.0, distance
        elif kind == "circle":
            angle = angular_rate*time_s
            east0, north0 = radius*math.sin(angle), radius*(1-math.cos(angle))
        else: east0, north0 = _sweep(distance, leg_length, lane_spacing)
        east = east0*math.cos(hd) + north0*math.sin(hd)
        north = -east0*math.sin(hd) + north0*math.cos(hd)
        out_lat, out_lon, out_alt = enu_to_llh(east, north, 0, lat, lon, alt)
        rows.append((time_s, out_lat, out_lon, out_alt))
    csv_bytes = "".join(f"{t:.1f},{a:.9f},{o:.9f},{h:.4f}\n" for t,a,o,h in rows).encode()
    digest = hashlib.sha256(csv_bytes).hexdigest()
    distance = effective_speed*duration
    closure_error = (2*radius*abs(math.sin(math.pi*lap_count)) if kind == "circle" else None)
    closed = kind == "circle" and closure_error < 1e-9
    metadata = {
        "generator": {"schema": GENERATOR_SCHEMA, "version": GENERATOR_VERSION},
        "scenario": kind,
        "motion_model": "constant-radius circular motion" if kind == "circle" else ("C1 tangent straight/semicircle sweep" if kind == "parallel-sweep" else "constant-speed straight motion"),
        "coordinate_reference": {"datum": "WGS-84", "frame": "local ENU tangent frame converted to geodetic LLH", "semi_major_axis_m": WGS84_A_M, "flattening": WGS84_F},
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "actual_row_count": len(rows), "actual_start_time_s": rows[0][0], "actual_end_time_s": rows[-1][0],
        "csv_sha256": digest,
        "parameters": {"latitude_deg": lat, "longitude_deg": lon, "altitude_m": alt, "duration_seconds": duration, "requested_speed_mps": speed, "heading_deg": heading},
        "effective": {"distance_m": distance, "speed_mps": effective_speed, "laps": lap_count, "arc_radians": (angular_rate*duration if kind == "circle" else None), "closed_orbit": closed, "closure_error_m": closure_error},
        "literature": {"doi": DOIS[kind], "note": "Trajectory family is literature-motivated; speed, altitude, radius, duration, heading, and sweep dimensions are controlled parameters of this study."},
    }
    if kind == "circle": metadata["parameters"]["radius_m"] = radius
    if kind == "parallel-sweep": metadata["parameters"].update(leg_length_m=leg_length, lane_spacing_m=lane_spacing)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix(".json")
    # Pair policy: remove the old sidecar first. A failure may leave a CSV without a
    # sidecar, but can never leave an apparently valid stale CSV/JSON pair.
    sidecar.unlink(missing_ok=True)
    _atomic_write(output, csv_bytes)
    _atomic_write(sidecar, (json.dumps(metadata, indent=2, sort_keys=True)+"\n").encode())
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=DOIS)
    parser.add_argument("output")
    for name in ("latitude_deg", "longitude_deg", "altitude_m", "duration_seconds", "speed_mps"):
        parser.add_argument("--"+name.replace("_", "-"), type=float, required=True)
    parser.add_argument("--radius-m", type=float, default=20)
    parser.add_argument("--leg-length-m", type=float, default=100)
    parser.add_argument("--lane-spacing-m", type=float, default=20)
    parser.add_argument("--heading-deg", type=float, default=0)
    parser.add_argument("--laps", type=float, help="circle only: positive integer full laps; overrides effective speed")
    args = parser.parse_args(argv)
    generate_trajectory(args.kind, args.output, **{k:v for k,v in vars(args).items() if k not in ("kind", "output")})


if __name__ == "__main__": main()
