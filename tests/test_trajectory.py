import hashlib
import json
import math
import os

import pytest

from gnss_doppler_lab.trajectory import (
    MIN_PHYSICAL_ECEF_RADIUS_M, WGS84_A_M, ecef_to_llh, enu_to_llh,
    generate_trajectory, llh_to_ecef, llh_to_enu, read_trajectory,
)
from gnss_doppler_lab.rf_config import ConfigError, TrajectoryPosition, load_rf_config

ORIGIN = (37.5665, 126.978, 50.0)


def _generate(tmp_path, kind, duration=10, speed=3, **kwargs):
    path = tmp_path / f"{kind}.csv"
    meta = generate_trajectory(kind, path, latitude_deg=ORIGIN[0], longitude_deg=ORIGIN[1],
                               altitude_m=ORIGIN[2], duration_seconds=duration,
                               speed_mps=speed, **kwargs)
    rows = read_trajectory(path, duration)
    enu = [llh_to_enu(*row[1:], *ORIGIN) for row in rows]
    return path, meta, rows, enu


def test_exact_duration_contract_and_strict_no_prefix_truncation(tmp_path):
    path, meta, rows, _ = _generate(tmp_path, "straight", duration=10)
    assert len(rows) == 100
    assert [rows[0][0], rows[-1][0]] == [0.0, 9.9]
    assert meta["actual_row_count"] == 100
    assert meta["actual_start_time_s"] == 0.0
    assert meta["actual_end_time_s"] == 9.9
    path.write_text(path.read_text() + "10.0,37,127,50\n")
    with pytest.raises(ValueError, match="exactly 100"):
        read_trajectory(path, 10)


def test_ecef_trajectory_rejects_obvious_earth_interior_coordinates(tmp_path):
    path = tmp_path / "ecef.csv"
    for coordinates in ((0, 0, 0), (MIN_PHYSICAL_ECEF_RADIUS_M - 1, 0, 0)):
        path.write_text(f"0.0,{coordinates[0]},{coordinates[1]},{coordinates[2]}\n")
        with pytest.raises(
            ValueError,
            match=rf"row 1: ECEF geocentric radius must be at least {MIN_PHYSICAL_ECEF_RADIUS_M:.0f} m",
        ):
            read_trajectory(path, 0.1, coordinate_system="ecef")


@pytest.mark.parametrize("llh", [(0, 0, 0), (37.5665, 126.978, 12000)])
def test_ecef_trajectory_accepts_surface_and_aviation_coordinates(tmp_path, llh):
    path = tmp_path / "ecef.csv"
    x, y, z = llh_to_ecef(*llh)
    path.write_text(f"0.0,{x},{y},{z}\n")
    assert read_trajectory(path, 0.1, coordinate_system="ecef") == [
        pytest.approx((0.0, x, y, z))
    ]


def test_wgs84_known_points_round_trip_and_enu_displacement():
    assert llh_to_ecef(0, 0, 0) == pytest.approx((WGS84_A_M, 0, 0), abs=1e-8)
    north_pole = llh_to_ecef(90, 0, 0)
    assert ecef_to_llh(*north_pole) == pytest.approx((90, 0, 0), abs=1e-7)
    for point in ((0, 0, 0), ORIGIN, (-33.9, 151.2, 1234.5)):
        assert ecef_to_llh(*llh_to_ecef(*point)) == pytest.approx(point, abs=2e-7)
    moved = enu_to_llh(12.3, -45.6, 7.8, *ORIGIN)
    assert llh_to_enu(*moved, *ORIGIN) == pytest.approx((12.3, -45.6, 7.8), abs=2e-7)


def test_straight_si_speed_and_heading_after_csv_quantization(tmp_path):
    _, _, _, enu = _generate(tmp_path, "straight", duration=5, speed=4, heading_deg=90)
    velocities = [((b[0]-a[0])*10, (b[1]-a[1])*10) for a, b in zip(enu, enu[1:])]
    assert [sum(v[0] for v in velocities)/len(velocities), sum(v[1] for v in velocities)/len(velocities)] == pytest.approx([4, 0], abs=0.015)
    assert max(abs(v[1]) for v in velocities) < 0.03


def test_circle_radius_speed_direction_and_closed_laps(tmp_path):
    _, meta, _, enu = _generate(tmp_path, "circle", duration=8, speed=999,
                                 radius_m=10, laps=1)
    center = (0, 10)
    radii = [math.hypot(p[0]-center[0], p[1]-center[1]) for p in enu]
    assert radii == pytest.approx([10]*len(radii), abs=0.015)
    expected_speed = 2*math.pi*10/8
    step_speeds = [math.dist(a[:2], b[:2])*10 for a,b in zip(enu, enu[1:])]
    assert step_speeds == pytest.approx([expected_speed]*len(step_speeds), abs=0.02)
    assert enu[1][0] > enu[0][0]  # clockwise from the southern point in EN coordinates
    assert meta["effective"]["closed_orbit"] is True
    assert meta["effective"]["laps"] == 1
    assert meta["effective"]["closure_error_m"] == pytest.approx(0, abs=1e-12)
    # The endpoint is excluded by the time contract; extrapolated full-lap endpoint closes.
    assert math.dist(enu[0][:2], (0, 0)) < 0.015


def test_non_closed_circle_metadata_never_claims_closed(tmp_path):
    _, meta, _, _ = _generate(tmp_path, "circle", duration=3, speed=2, radius_m=10)
    assert meta["motion_model"] == "constant-radius circular motion"
    assert meta["effective"]["closed_orbit"] is False
    assert meta["effective"]["arc_radians"] == pytest.approx(.6)
    assert meta["effective"]["closure_error_m"] > 0


def test_sweep_straight_semicircle_boundaries_are_c1_after_quantization(tmp_path):
    # speed=1 places boundaries at t=2 and t=2+pi; compare one-sided velocity.
    _, _, _, enu = _generate(tmp_path, "parallel-sweep", duration=9, speed=1,
                              leg_length_m=2, lane_spacing_m=2)
    velocity = [((b[0]-a[0])*10, (b[1]-a[1])*10) for a,b in zip(enu, enu[1:])]
    for boundary_index in (20, round((2+math.pi)*10), round((4+math.pi)*10)):
        before, after = velocity[boundary_index-1], velocity[boundary_index]
        assert math.dist(before, after) < 0.12
        assert math.hypot(*before) == pytest.approx(1, abs=.01)
        assert math.hypot(*after) == pytest.approx(1, abs=.01)


def test_sidecar_reproducibility_metadata_and_hash(tmp_path):
    path, meta, _, _ = _generate(tmp_path, "circle", duration=2, speed=2, radius_m=5)
    saved = json.loads(path.with_suffix(".json").read_text())
    assert saved == meta
    assert saved["csv_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert saved["generator"] == {"schema": "gnss-doppler-lab.trajectory", "version": 2}
    assert saved["coordinate_reference"]["datum"] == "WGS-84"
    assert saved["literature"]["doi"]
    assert "controlled parameters" in saved["literature"]["note"]


def test_atomic_pair_policy_removes_stale_sidecar_on_publish_failure(tmp_path, monkeypatch):
    path, _, _, _ = _generate(tmp_path, "straight", duration=1, speed=1)
    sidecar = path.with_suffix(".json")
    real_replace = os.replace
    calls = 0
    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2: raise OSError("sidecar publish failed")
        return real_replace(source, destination)
    monkeypatch.setattr(os, "replace", fail_second)
    with pytest.raises(OSError, match="sidecar"):
        generate_trajectory("straight", path, latitude_deg=ORIGIN[0], longitude_deg=ORIGIN[1],
                            altitude_m=ORIGIN[2], duration_seconds=1, speed_mps=2)
    assert path.exists() and not sidecar.exists()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("kwargs", [
    {"latitude_deg": 90}, {"longitude_deg": 181}, {"altitude_m": math.nan},
    {"heading_deg": math.inf}, {"speed_mps": math.nan}, {"duration_seconds": 1.01},
])
def test_generator_rejects_nonfinite_ranges_and_polar_origins(tmp_path, kwargs):
    args = dict(latitude_deg=0, longitude_deg=0, altitude_m=0,
                duration_seconds=1, speed_mps=1, heading_deg=0)
    args.update(kwargs)
    with pytest.raises(ValueError): generate_trajectory("straight", tmp_path/"x.csv", **args)


def test_only_kind_used_parameters_are_validated(tmp_path):
    _generate(tmp_path, "straight", duration=1, speed=1,
              radius_m=math.nan, leg_length_m=math.nan, lane_spacing_m=math.nan)


def test_dynamic_config_validation_relative_path_and_integer_contract(tmp_path):
    trajectory = tmp_path/"motion.csv"
    generate_trajectory("straight", trajectory, latitude_deg=37.5, longitude_deg=127,
                        altitude_m=10, duration_seconds=1, speed_mps=1)
    config = tmp_path/"c.yml"
    config.write_text("""version: 1
scenario:
  name: dyn
  constellation: GPS
  signal: L1CA
  utc: 2025-01-01T00:00:00Z
  duration_seconds: 1
  position: {type: trajectory, path: motion.csv, coordinate_system: llh}
input: {rinex_nav: nav.rnx}
output: {root: out, rf_sample_rate_hz: 2600000}
""")
    loaded = load_rf_config(config)
    assert isinstance(loaded.scenario.position, TrajectoryPosition)
    assert loaded.scenario.position.path == trajectory.resolve()
    assert loaded.scenario.position.csv_sha256 == hashlib.sha256(trajectory.read_bytes()).hexdigest()
    assert loaded.scenario.position.metadata_path == trajectory.with_suffix(".json")
    assert loaded.scenario.position.metadata_sha256
    for old, new in (("duration_seconds: 1", "duration_seconds: 1.5"),
                     ("rf_sample_rate_hz: 2600000", "rf_sample_rate_hz: 2600000.5")):
        config.write_text(config.read_text().replace(old, new))
        with pytest.raises(ConfigError, match="integer"):
            load_rf_config(config)
        config.write_text(config.read_text().replace(new, old))


def test_generated_sidecar_hash_mismatch_rejected(tmp_path):
    trajectory = tmp_path / "motion.csv"
    generate_trajectory(
        "straight", trajectory, latitude_deg=37.5, longitude_deg=127,
        altitude_m=10, duration_seconds=1, speed_mps=1,
    )
    trajectory.write_text(trajectory.read_text().replace("37.500000000", "37.500000001", 1))
    config = tmp_path / "c.yml"
    config.write_text("""version: 1
scenario:
  name: mismatch
  constellation: GPS
  signal: L1CA
  utc: 2025-01-01T00:00:00Z
  duration_seconds: 1
  position: {type: trajectory, path: motion.csv, coordinate_system: llh}
input: {rinex_nav: nav.rnx}
output: {root: out, rf_sample_rate_hz: 2600000}
""")
    with pytest.raises(ConfigError, match="csv_sha256 does not match"):
        load_rf_config(config)
