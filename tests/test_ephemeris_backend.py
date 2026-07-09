from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from gnss_doppler_lab.doppler_simulator import doppler_observation_matrix
from gnss_doppler_lab.ephemeris_backend import (
    geodetic_to_ecef,
    gps_prns_in_nav_file,
    parse_rinex_nav_file,
    satellite_states_for_times,
    simulate_hover_doppler_scenario,
    visible_satellite_states,
)


def _write_sample_nav(tmp_path):
    nav_text = """     3.05           NAVIGATION DATA     GPS                         RINEX VERSION / TYPE
END OF HEADER
G01 2024 01 01 00 00 00 1.234567890123D-04 0.000000000000D+00 0.000000000000D+00
     1.000000000000D+02 1.000000000000D+01 4.000000000000D-09 0.000000000000D+00
     1.000000000000D-06 1.000000000000D-02 1.000000000000D-06 5.153795490500D+03
     0.000000000000D+00 1.000000000000D-08 1.000000000000D+00 1.000000000000D-08
     9.400000000000D+01 2.100000000000D+00 5.000000000000D-09 -8.000000000000D-09
     0.000000000000D+00 2.300000000000D+03 0.000000000000D+00 2.400000000000D+00
     0.000000000000D+00 0.000000000000D+00 2.000000000000D-08 0.000000000000D+00
     0.000000000000D+00 0.000000000000D+00 0.000000000000D+00 0.000000000000D+00
G 3 2024 01 01 00 00 00 2.000000000000D-04 0.000000000000D+00 0.000000000000D+00
     1.200000000000D+02 -1.200000000000D+01 5.000000000000D-09 8.000000000000D-01
    -2.000000000000D-06 8.000000000000D-03 2.000000000000D-06 5.153650000000D+03
     1.000000000000D+05 -2.000000000000D-08 9.500000000000D-01 2.000000000000D-08
     8.800000000000D+01 1.000000000000D+00 4.000000000000D-09 -7.000000000000D-09
     0.000000000000D+00 2.300000000000D+03 0.000000000000D+00 2.400000000000D+00
     0.000000000000D+00 0.000000000000D+00 1.000000000000D-08 0.000000000000D+00
     0.000000000000D+00 0.000000000000D+00 0.000000000000D+00 0.000000000000D+00
"""
    nav_path = tmp_path / "sample.nav"
    nav_path.write_text(nav_text)
    return nav_path


def test_parse_rinex_nav_file_extracts_gps_records(tmp_path):
    nav_path = _write_sample_nav(tmp_path)

    records = parse_rinex_nav_file(nav_path)

    assert len(records) == 2
    assert {record.prn for record in records} == {"G01", "G03"}
    assert all(record.constellation == "G" for record in records)


def test_gps_prns_in_nav_file_reports_available_gps_satellites(tmp_path):
    nav_path = _write_sample_nav(tmp_path)

    prns = gps_prns_in_nav_file(nav_path)

    assert prns == ["G01", "G03"]


def test_satellite_states_for_times_returns_finite_ecef_states(tmp_path):
    nav_path = _write_sample_nav(tmp_path)
    records = parse_rinex_nav_file(nav_path)
    times = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=offset) for offset in (0, 60)]

    positions, velocities, prns = satellite_states_for_times(records, times)

    assert positions.shape == (2, 2, 3)
    assert velocities.shape == (2, 2, 3)
    assert prns == ["G01", "G03"]
    assert np.isfinite(positions).all()
    assert np.isfinite(velocities).all()
    radii = np.linalg.norm(positions, axis=-1)
    assert np.all(radii > 2.0e7)
    assert np.all(radii < 3.0e7)


def test_visible_satellite_states_filters_by_elevation_and_supports_hover_5hz(tmp_path):
    nav_path = _write_sample_nav(tmp_path)
    records = parse_rinex_nav_file(nav_path)
    times = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=0.2 * i) for i in range(5)]
    receiver_ecef = geodetic_to_ecef(37.5665, 126.9780, 120.0)

    visible = visible_satellite_states(records, times, receiver_ecef, elevation_mask_deg=-90.0)

    assert visible["satellite_positions_ecef_m"].shape == (5, 2, 3)
    assert visible["satellite_velocities_ecef_mps"].shape == (5, 2, 3)
    assert visible["receiver_positions_ecef_m"].shape == (5, 3)
    assert visible["receiver_velocities_ecef_mps"].shape == (5, 3)
    assert visible["receiver_velocities_ecef_mps"].sum() == 0.0
    assert visible["visibility_mask"].shape == (5, 2)
    assert np.all(visible["visibility_mask"])

    doppler = doppler_observation_matrix(
        visible["satellite_positions_ecef_m"],
        visible["satellite_velocities_ecef_mps"],
        visible["receiver_positions_ecef_m"],
        visible["receiver_velocities_ecef_mps"],
    )

    assert doppler.shape == (5, 2)
    assert np.isfinite(doppler).all()


def test_simulate_hover_doppler_scenario_builds_5hz_static_receiver_dataset(tmp_path):
    nav_path = _write_sample_nav(tmp_path)
    scenario = simulate_hover_doppler_scenario(
        nav_path,
        start_time_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        duration_seconds=1.0,
        sample_rate_hz=5.0,
        latitude_deg=37.5665,
        longitude_deg=126.9780,
        altitude_m=120.0,
        elevation_mask_deg=-90.0,
    )

    assert scenario["sample_rate_hz"] == 5.0
    assert scenario["receiver_positions_ecef_m"].shape == (5, 3)
    assert scenario["receiver_velocities_ecef_mps"].shape == (5, 3)
    assert np.allclose(scenario["receiver_velocities_ecef_mps"], 0.0)
    assert scenario["doppler_hz"].shape == (5, 2)
    assert np.isfinite(scenario["doppler_hz"]).all()



def test_simulate_hover_doppler_scenario_rejects_time_outside_nav_coverage(tmp_path):
    nav_path = _write_sample_nav(tmp_path)

    with pytest.raises(ValueError, match="outside NAV coverage"):
        simulate_hover_doppler_scenario(
            nav_path,
            start_time_utc=datetime(2027, 5, 12, 13, 0, 0, tzinfo=timezone.utc),
            duration_seconds=60.0,
            sample_rate_hz=1.0,
            latitude_deg=37.5665,
            longitude_deg=126.9780,
            altitude_m=120.0,
            elevation_mask_deg=-90.0,
        )
