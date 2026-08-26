from pathlib import Path
from types import SimpleNamespace

import pytest

from gnss_doppler_lab.satellite_multipath import (
    PATCH_CONTRACT,
    PrnMultipathGpsSdrSimRunner,
    SatelliteMultipathEcho,
    independent_echoes,
    validate_echoes,
    write_multipath_spec,
)


def test_independent_echoes_are_reproducible_and_prn_specific():
    first = independent_echoes([8, 1, 21], seed=20260901)
    second = independent_echoes([21, 8, 1], seed=20260901)
    assert first == second
    assert [echo.prn for echo in first] == [1, 8, 21]
    assert len({echo.delay_chips for echo in first}) == 3
    assert len({echo.phase_deg for echo in first}) == 3
    with pytest.raises(ValueError, match="unique"):
        independent_echoes([8, 8], seed=20260901)


def test_validate_echoes_rejects_duplicate_prns_and_subsample_delay():
    duplicate = [
        SatelliteMultipathEcho(8, 0.5, 0.4, 0.0),
        SatelliteMultipathEcho(8, 0.6, 0.3, 90.0),
    ]
    with pytest.raises(ValueError, match="unique PRNs"):
        validate_echoes(duplicate, 25_000_000)
    with pytest.raises(ValueError, match="less than one RF sample"):
        validate_echoes(
            [SatelliteMultipathEcho(8, 0.5, 0.4, 0.0)],
            1_000_000,
        )


def test_write_multipath_spec_has_explicit_sample_delay_and_no_overwrite(tmp_path):
    path = tmp_path / "multipath.csv"
    echo = SatelliteMultipathEcho(8, 0.5, 0.4, 45.0)
    write_multipath_spec(path, [echo], sample_rate_hz=25_000_000)
    fields = path.read_text(encoding="ascii").strip().split(",")
    assert fields[0] == "8"
    assert float(fields[1]) == pytest.approx(12.218963831867)
    assert fields[2:] == ["0.400000000000", "45.000000000000"]
    with pytest.raises(FileExistsError):
        write_multipath_spec(path, [echo], sample_rate_hz=25_000_000)


def test_patched_runner_inserts_multipath_before_output_argument(tmp_path):
    runner = PrnMultipathGpsSdrSimRunner(
        "/sim",
        [SatelliteMultipathEcho(8, 0.5, 0.4, 45.0)],
    )
    spec = tmp_path / "multipath.csv"
    runner._active_spec_path = spec
    config = SimpleNamespace(
        scenario=SimpleNamespace(
            position=SimpleNamespace(
                latitude_deg=37.5,
                longitude_deg=127.0,
                altitude_m=42.0,
            ),
            duration_seconds=2,
        ),
        output=SimpleNamespace(rf_sample_rate_hz=25_000_000),
    )
    time = SimpleNamespace(simulator_input_calendar="2026/07/11,03:04:23")
    command = runner.build_command(config, tmp_path / "iq.bin", "nav.rnx", time)
    assert command[-4:] == ["-m", "multipath.csv", "-o", str(tmp_path / "iq.bin")]
    assert command[0] == "/sim"
    assert runner.cli_contract.endswith(PATCH_CONTRACT)
