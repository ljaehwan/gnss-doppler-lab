from __future__ import annotations

from pathlib import Path

import yaml

from gnss_doppler_lab.scenario import ScenarioConfig


def load_scenario_config(path: Path) -> ScenarioConfig:
    data = yaml.safe_load(path.read_text())
    return ScenarioConfig(
        scenario_name=data["scenario_name"],
        latitude_deg=float(data["latitude_deg"]),
        longitude_deg=float(data["longitude_deg"]),
        altitude_m=float(data["altitude_m"]),
        start_time_utc=str(data["start_time_utc"]),
        duration_s=float(data["duration_s"]),
        sample_rate_hz=float(data["sample_rate_hz"]),
        receiver_speed_mps=float(data.get("receiver_speed_mps", 0.0)),
        receiver_heading_deg=float(data.get("receiver_heading_deg", 0.0)),
        climb_rate_mps=float(data.get("climb_rate_mps", 0.0)),
        mask_angle_deg=float(data.get("mask_angle_deg", 10.0)),
        num_satellites=int(data.get("num_satellites", 24)),
        orbital_planes=int(data.get("orbital_planes", 6)),
        carrier_frequency_hz=float(data.get("carrier_frequency_hz", 1_575_420_000.0)),
    )
