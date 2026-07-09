from pathlib import Path

from gnss_doppler_lab.scenario import ScenarioConfig


def test_scenario_config_fields() -> None:
    scenario = ScenarioConfig(
        scenario_name="seoul_poc",
        latitude_deg=37.5665,
        longitude_deg=126.9780,
        altitude_m=120.0,
        start_time_utc="2026-07-09T00:00:00Z",
        duration_s=10.0,
        sample_rate_hz=1.0,
        receiver_speed_mps=22.0,
        receiver_heading_deg=45.0,
        climb_rate_mps=0.5,
    )

    assert scenario.sample_period_s == 1.0
    assert scenario.epoch_count == 11
    assert scenario.start_time.year == 2026


def test_scenario_config_loader_file_exists() -> None:
    assert Path("configs/seoul_poc.yaml").exists()
