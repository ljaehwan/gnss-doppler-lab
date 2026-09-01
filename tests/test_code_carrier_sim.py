from datetime import datetime, timezone
from pathlib import Path

import pytest

from gnss_doppler_lab.code_carrier_sim import (
    DecoupledSimulationRequest,
    summarize_truth_triplet,
)
from gnss_doppler_lab.gps_sdr_sim import SimulatorError


HEADER = "time_s,prn,code_range_m,carrier_range_m,relative_code_range_m,code_rate_mps,carrier_rate_mps,code_frequency_hz,carrier_doppler_hz\n"


def write_motion(path: Path, rows: int = 10):
    path.write_text("".join(f"{i/10:.1f},37.0,127.0,100.0\n" for i in range(rows)))


def test_request_requires_carrier_reference_only_for_locked(tmp_path: Path):
    nav = tmp_path / "nav"; nav.write_text("x")
    code = tmp_path / "code.csv"; write_motion(code)
    utc = datetime(2022, 1, 1, tzinfo=timezone.utc)
    request = DecoupledSimulationRequest(nav, code, None, utc, 1, 1_000_000, "doppler_locked")
    with pytest.raises(SimulatorError, match="requires carrier_motion"):
        request.validate()


def test_truth_triplet_proves_code_equal_and_carrier_locked(tmp_path: Path):
    authentic = tmp_path / "auth.csv"
    coupled = tmp_path / "coupled.csv"
    locked = tmp_path / "locked.csv"
    authentic.write_text(HEADER + "10.0,5,100,100,0,1,1,1023000,20\n")
    coupled.write_text(HEADER + "10.0,5,140,140,0,4,4,1022990,35\n")
    locked.write_text(HEADER + "10.0,5,140,100,40,4,1,1022990,20\n")
    result = summarize_truth_triplet(authentic, coupled, locked, hold_start_seconds=10)
    assert result["locked_vs_coupled_code_range_max_abs_m"] == 0
    assert result["locked_vs_authentic_carrier_rate_max_abs_mps"] == 0
    assert result["locked_code_vs_carrier_hold_max_abs_m"] == 40
    assert result["coupled_vs_locked_carrier_doppler_max_abs_hz"] == 15
