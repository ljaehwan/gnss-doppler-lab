from pathlib import Path

from gnss_doppler_lab.pipeline import run_visibility_pipeline


def test_run_visibility_pipeline_writes_expected_outputs(tmp_path: Path) -> None:
    result = run_visibility_pipeline(
        config_path=Path("configs/seoul_poc.yaml"),
        output_root=tmp_path,
    )

    summary_path = result.output_dir / "visibility_summary.csv"
    timeseries_path = result.output_dir / "doppler_timeseries.csv"

    assert summary_path.exists()
    assert timeseries_path.exists()
    assert result.records_written > 0
    assert result.visible_satellite_count > 0

    summary_text = summary_path.read_text()
    timeseries_text = timeseries_path.read_text()

    assert "prn" in summary_text
    assert "doppler_hz" in timeseries_text
    assert "G" in summary_text or "G" in timeseries_text


def test_run_visibility_pipeline_supports_rinex_nav_backend(tmp_path: Path) -> None:
    nav_text = """     3.05           NAVIGATION DATA     GPS                         RINEX VERSION / TYPE
END OF HEADER
G01 2024 01 01 00 00 00 1.234567890123D-04 0.000000000000D+00 0.000000000000D+00
    1.000000000000D+01 2.000000000000D+01 0.000000000000D+00 1.000000000000D+00
    0.000000000000D+00 1.000000000000D-02 0.000000000000D+00 5.153795890810D+03
    0.000000000000D+00 4.320000000000D+05 1.000000000000D+00 0.000000000000D+00
    9.400000000000D-01 0.000000000000D+00 5.000000000000D-01 -8.000000000000D-09
    0.000000000000D+00 0.000000000000D+00 2.230000000000D+03 0.000000000000D+00
    0.000000000000D+00 0.000000000000D+00 -2.000000000000D-08 0.000000000000D+00
    0.000000000000D+00 0.000000000000D+00
G03 2024 01 01 00 00 00 -2.345678901234D-04 0.000000000000D+00 0.000000000000D+00
    1.500000000000D+01 1.500000000000D+01 0.000000000000D+00 2.000000000000D+00
    0.000000000000D+00 2.000000000000D-02 0.000000000000D+00 5.153600000000D+03
    0.000000000000D+00 4.320000000000D+05 1.200000000000D+00 0.000000000000D+00
    9.500000000000D-01 0.000000000000D+00 6.000000000000D-01 -8.500000000000D-09
    0.000000000000D+00 0.000000000000D+00 2.230000000000D+03 0.000000000000D+00
    0.000000000000D+00 0.000000000000D+00 -1.000000000000D-08 0.000000000000D+00
    0.000000000000D+00 0.000000000000D+00
"""
    nav_path = tmp_path / "sample.nav"
    nav_path.write_text(nav_text)

    config_path = tmp_path / "rinex_nav.yaml"
    config_path.write_text(
        "\n".join(
            [
                "scenario_name: rinex_nav_poc",
                "latitude_deg: 37.5665",
                "longitude_deg: 126.9780",
                "altitude_m: 120.0",
                'start_time_utc: "2024-01-01T00:00:00Z"',
                "duration_s: 1.0",
                "sample_rate_hz: 1.0",
                "mask_angle_deg: -90.0",
                f'rinex_nav_path: "{nav_path}"',
                "",
            ]
        )
    )

    result = run_visibility_pipeline(config_path=config_path, output_root=tmp_path)

    summary_path = result.output_dir / "visibility_summary.csv"
    timeseries_path = result.output_dir / "doppler_timeseries.csv"

    assert summary_path.exists()
    assert timeseries_path.exists()
    assert result.records_written > 0
    assert result.visible_satellite_count == 2
    assert "G01" in summary_path.read_text()
