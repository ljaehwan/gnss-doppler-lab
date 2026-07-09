from dataclasses import dataclass


@dataclass(slots=True)
class ScenarioConfig:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    start_time_utc: str
    duration_s: float
    sample_rate_hz: float
