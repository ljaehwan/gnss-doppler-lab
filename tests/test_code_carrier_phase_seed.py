from datetime import datetime, timezone
from pathlib import Path

import pytest

from gnss_doppler_lab.code_carrier_sim import DecoupledSimulationRequest
from gnss_doppler_lab.gps_sdr_sim import SimulatorError


def write_motion(path: Path, rows: int = 10) -> None:
    path.write_text(
        "".join(f"{index / 10:.1f},37.0,127.0,100.0\n" for index in range(rows)),
        encoding="ascii",
    )


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_request_rejects_invalid_carrier_phase_seed(tmp_path: Path, seed) -> None:
    nav = tmp_path / "nav"
    nav.write_text("    18                                                      LEAP SECONDS\n                                                            END OF HEADER\n", encoding="ascii")
    code = tmp_path / "code.csv"
    write_motion(code)
    request = DecoupledSimulationRequest(
        nav,
        code,
        None,
        datetime(2022, 1, 1, tzinfo=timezone.utc),
        1,
        1_000_000,
        "coupled",
        carrier_phase_seed=seed,
    )
    with pytest.raises(SimulatorError, match="non-negative integer"):
        request.validate()


def test_request_accepts_integer_carrier_phase_seed(tmp_path: Path) -> None:
    nav = tmp_path / "nav"
    nav.write_text("    18                                                      LEAP SECONDS\n                                                            END OF HEADER\n", encoding="ascii")
    code = tmp_path / "code.csv"
    write_motion(code)
    request = DecoupledSimulationRequest(
        nav,
        code,
        None,
        datetime(2022, 1, 1, tzinfo=timezone.utc),
        1,
        1_000_000,
        "coupled",
        carrier_phase_seed=2026090201,
    )
    assert request.validate()["code_rows"] == 10
