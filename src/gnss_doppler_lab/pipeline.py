from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import timedelta
from math import cos, radians, sin
from pathlib import Path

from gnss_doppler_lab.config_loader import load_scenario_config
from gnss_doppler_lab.coordinates import add, enu_to_ecef_vector, geodetic_to_ecef, scale
from gnss_doppler_lab.doppler import compute_doppler_hz
from gnss_doppler_lab.satellites import generate_satellite_constellation
from gnss_doppler_lab.visibility import visible_satellites


@dataclass(slots=True)
class PipelineResult:
    output_dir: Path
    records_written: int
    visible_satellite_count: int


def _receiver_kinematics(config) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    receiver_start = geodetic_to_ecef(config.latitude_deg, config.longitude_deg, config.altitude_m)
    heading_rad = radians(config.receiver_heading_deg)
    east_mps = config.receiver_speed_mps * sin(heading_rad)
    north_mps = config.receiver_speed_mps * cos(heading_rad)
    velocity_ecef = enu_to_ecef_vector(
        east_mps=east_mps,
        north_mps=north_mps,
        up_mps=config.climb_rate_mps,
        latitude_deg=config.latitude_deg,
        longitude_deg=config.longitude_deg,
    )
    return receiver_start, velocity_ecef, (east_mps, north_mps, config.climb_rate_mps)


def run_visibility_pipeline(config_path: Path, output_root: Path) -> PipelineResult:
    config = load_scenario_config(config_path)
    output_dir = output_root / config.scenario_name
    output_dir.mkdir(parents=True, exist_ok=True)

    receiver_start, velocity_ecef, _ = _receiver_kinematics(config)
    timeseries_rows: list[dict[str, float | str | int]] = []
    summary: dict[str, dict[str, float | int]] = {}

    for epoch_index in range(config.epoch_count):
        elapsed_s = epoch_index * config.sample_period_s
        epoch_time = config.start_time + timedelta(seconds=elapsed_s)
        receiver_ecef = add(receiver_start, scale(velocity_ecef, elapsed_s))
        satellites = generate_satellite_constellation(
            elapsed_s=elapsed_s,
            num_satellites=config.num_satellites,
            orbital_planes=config.orbital_planes,
        )
        visible = visible_satellites(
            receiver_ecef=receiver_ecef,
            satellites=satellites,
            latitude_deg=config.latitude_deg,
            longitude_deg=config.longitude_deg,
            mask_angle_deg=config.mask_angle_deg,
        )
        visible_lookup = {record.prn: record for record in visible}

        for satellite in satellites:
            if satellite.prn not in visible_lookup:
                continue
            visibility_record = visible_lookup[satellite.prn]
            doppler_hz = compute_doppler_hz(
                receiver_ecef=receiver_ecef,
                receiver_velocity_ecef_mps=velocity_ecef,
                satellite_ecef=satellite.position_ecef_m,
                satellite_velocity_ecef_mps=satellite.velocity_ecef_mps,
                carrier_frequency_hz=config.carrier_frequency_hz,
            )
            timeseries_rows.append(
                {
                    "epoch_index": epoch_index,
                    "time_utc": epoch_time.isoformat().replace("+00:00", "Z"),
                    "elapsed_s": round(elapsed_s, 3),
                    "prn": satellite.prn,
                    "azimuth_deg": round(visibility_record.azimuth_deg, 3),
                    "elevation_deg": round(visibility_record.elevation_deg, 3),
                    "range_m": round(visibility_record.range_m, 3),
                    "doppler_hz": round(doppler_hz, 3),
                }
            )
            per_satellite = summary.setdefault(
                satellite.prn,
                {
                    "samples": 0,
                    "max_elevation_deg": visibility_record.elevation_deg,
                    "min_elevation_deg": visibility_record.elevation_deg,
                },
            )
            per_satellite["samples"] += 1
            per_satellite["max_elevation_deg"] = max(per_satellite["max_elevation_deg"], visibility_record.elevation_deg)
            per_satellite["min_elevation_deg"] = min(per_satellite["min_elevation_deg"], visibility_record.elevation_deg)

    _write_csv(output_dir / "doppler_timeseries.csv", timeseries_rows)
    summary_rows = [
        {
            "prn": prn,
            "samples": values["samples"],
            "min_elevation_deg": round(values["min_elevation_deg"], 3),
            "max_elevation_deg": round(values["max_elevation_deg"], 3),
        }
        for prn, values in sorted(summary.items())
    ]
    _write_csv(output_dir / "visibility_summary.csv", summary_rows)

    return PipelineResult(
        output_dir=output_dir,
        records_written=len(timeseries_rows),
        visible_satellite_count=len(summary_rows),
    )


def _write_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
