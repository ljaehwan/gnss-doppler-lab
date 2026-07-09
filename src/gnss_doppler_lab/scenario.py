from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(slots=True)
class ScenarioConfig:
    scenario_name: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    start_time_utc: str
    duration_s: float
    sample_rate_hz: float
    receiver_speed_mps: float = 0.0
    receiver_heading_deg: float = 0.0
    climb_rate_mps: float = 0.0
    mask_angle_deg: float = 10.0
    num_satellites: int = 24
    orbital_planes: int = 6
    carrier_frequency_hz: float = 1_575_420_000.0

    @property
    def sample_period_s(self) -> float:
        return 1.0 / self.sample_rate_hz

    @property
    def epoch_count(self) -> int:
        return int(round(self.duration_s * self.sample_rate_hz)) + 1

    @property
    def start_time(self) -> datetime:
        normalized = self.start_time_utc.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).astimezone(timezone.utc)
