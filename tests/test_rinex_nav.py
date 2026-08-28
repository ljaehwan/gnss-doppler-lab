from __future__ import annotations

import gzip

from gnss_doppler_lab.rinex_nav import parse_rinex2_gps_nav_gz


def _nav_line(values: list[float]) -> str:
    return "   " + "".join(f"{value:19.12E}".replace("E", "D") for value in values) + "\n"


def _record(prn: int, *, toe: float, health: int = 0) -> str:
    return "".join([
        f"{prn:2d}".ljust(80) + "\n",
        _nav_line([1.0, -80.0, 4.5e-9, 1.0]),
        _nav_line([1.0e-6, 0.01, 2.0e-6, 5153.7955]),
        _nav_line([toe, 3.0e-8, 0.7, 4.0e-8]),
        _nav_line([0.94, 200.0, -0.3, -8.0e-9]),
        _nav_line([1.0e-10, 1.0, 2353.0, 0.0]),
        _nav_line([2.0, float(health), -1.0e-9, 1.0]),
        _nav_line([388000.0, 4.0, 0.0, 0.0]),
    ])


def test_parse_rinex2_gps_nav_selects_nearest_record_and_week_modulo(tmp_path) -> None:
    path = tmp_path / "brdc0440.25n.gz"
    text = (
        "     2.11           N: GPS NAV DATA                         RINEX VERSION / TYPE\n"
        "                                                            END OF HEADER\n"
        + _record(22, toe=360000.0)
        + _record(22, toe=388800.0)
        + _record(5, toe=388800.0, health=1)
    )
    with gzip.open(path, "wt", encoding="ascii") as handle:
        handle.write(text)

    ephemerides = parse_rinex2_gps_nav_gz(
        path,
        full_gps_week=2353,
        target_tow_s=388470.0,
        maximum_toe_age_s=7200.0,
    )

    assert sorted(ephemerides) == [5, 22]
    assert ephemerides[22].toe == 388800.0
    assert ephemerides[22].WN == 2353 % 1024
    assert ephemerides[22].SV_health == 0
    assert ephemerides[5].SV_health == 1
