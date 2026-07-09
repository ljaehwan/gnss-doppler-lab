from gnss_doppler_lab.scenario import ScenarioConfig


def test_scenario_config_fields() -> None:
    scenario = ScenarioConfig(
        latitude_deg=37.5665,
        longitude_deg=126.9780,
        altitude_m=120.0,
        start_time_utc="2026-07-09T00:00:00Z",
        duration_s=60.0,
        sample_rate_hz=5.0,
    )

    assert scenario.sample_rate_hz == 5.0
    assert scenario.duration_s == 60.0
