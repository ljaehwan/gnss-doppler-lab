import json
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.rf_config import ConfigError, load_rf_config
from gnss_doppler_lab.spoofing import (
    SpoofingError,
    build_spoofing_rows,
    filter_rinex2_nav,
    generate_spoofing_iq,
    mix_iq_files,
    power_envelope,
)


def _rinex2_nav(path: Path):
    header = (
        "     2.10           N: GPS NAV DATA                         RINEX VERSION / TYPE\n"
        "    18                                                      LEAP SECONDS\n"
        "                                                            END OF HEADER\n"
    )
    records = []
    for prn in (1, 3, 7):
        first = f"{prn:2d} 22  1  1  0  0  0.0 0.0 0.0 0.0".ljust(80) + "\n"
        records.append(first + (" " * 79 + "\n") * 7)
    path.write_text(header + "".join(records))


def test_filter_rinex2_nav_keeps_header_and_selected_prn_records(tmp_path):
    source = tmp_path / "source.nav"
    output = tmp_path / "filtered.nav"
    _rinex2_nav(source)

    result = filter_rinex2_nav(source, output, (3, 7))

    text = output.read_text()
    assert result == {"available_prns": [1, 3, 7], "selected_prns": [3, 7]}
    assert "END OF HEADER" in text
    assert " 3 22" in text
    assert " 7 22" in text
    assert " 1 22" not in text


def test_filter_rinex2_nav_rejects_unknown_or_duplicate_prns(tmp_path):
    source = tmp_path / "source.nav"
    _rinex2_nav(source)
    with pytest.raises(SpoofingError, match="not present"):
        filter_rinex2_nav(source, tmp_path / "x.nav", (9,))
    with pytest.raises(SpoofingError, match="duplicate"):
        filter_rinex2_nav(source, tmp_path / "x.nav", (3, 3))


def test_build_carryoff_rows_match_truth_before_attack_and_reach_enu_offset():
    authentic = tuple((i / 10, 37.5665, 126.9780, 120.0) for i in range(50))

    rows = build_spoofing_rows(
        authentic,
        coordinate_system="llh",
        attack_type="carry_off",
        start_seconds=1.0,
        transition_seconds=2.0,
        target_offset_enu_m=(100.0, 0.0, 10.0),
    )

    assert rows[:11] == authentic[:11]
    from gnss_doppler_lab.trajectory import llh_to_enu
    east, north, up = llh_to_enu(*rows[-1][1:], *authentic[-1][1:])
    assert east == pytest.approx(100.0, abs=0.02)
    assert north == pytest.approx(0.0, abs=0.02)
    assert up == pytest.approx(10.0, abs=0.02)


def test_build_spoofing_rows_supports_ecef_truth_and_rejects_invalid_timing():
    from gnss_doppler_lab.trajectory import llh_to_ecef
    xyz = llh_to_ecef(37.5665, 126.9780, 120.0)
    authentic = tuple((i / 10, *xyz) for i in range(20))
    rows = build_spoofing_rows(
        authentic,
        coordinate_system="ecef",
        attack_type="abrupt",
        start_seconds=1.0,
        transition_seconds=0.0,
        target_offset_enu_m=(5.0, 0.0, 0.0),
    )
    assert rows[9] == authentic[9]
    assert rows[10] != authentic[10]
    with pytest.raises(SpoofingError, match="transition_seconds"):
        build_spoofing_rows(
            authentic,
            coordinate_system="ecef",
            attack_type="carry_off",
            start_seconds=1.0,
            transition_seconds=0.0,
            target_offset_enu_m=(5.0, 0.0, 0.0),
        )


def test_power_envelope_uses_amplitude_db_and_linear_ramp():
    envelope = power_envelope(
        sample_count=8,
        sample_rate_hz=2,
        start_seconds=1.0,
        ramp_seconds=2.0,
        initial_advantage_db=-20.0,
        final_advantage_db=0.0,
    )
    assert envelope[:2].tolist() == [0.0, 0.0]
    assert envelope[2] == pytest.approx(0.1)
    assert envelope[6] == pytest.approx(1.0)
    assert envelope[-1] == pytest.approx(1.0)


def test_mix_iq_files_streams_iq8_and_records_clipping(tmp_path):
    authentic = tmp_path / "auth.bin"
    spoofing = tmp_path / "spoof.bin"
    output = tmp_path / "mixed.bin"
    np.array([[10, -10], [20, -20], [30, -30], [40, -40]], dtype=np.int8).tofile(authentic)
    np.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.int8).tofile(spoofing)

    report = mix_iq_files(
        authentic,
        spoofing,
        output,
        sample_rate_hz=2,
        start_seconds=1.0,
        ramp_seconds=0.0,
        initial_advantage_db=0.0,
        final_advantage_db=0.0,
        fixed_scale=0.5,
        chunk_complex_samples=2,
    )

    mixed = np.fromfile(output, dtype=np.int8).reshape(-1, 2)
    assert mixed.tolist() == [[5, -5], [10, -10], [30, 0], [40, 0]]
    assert report["complex_samples"] == 4
    assert report["clipped_components"] == 0
    assert report["fixed_scale"] == 0.5


def test_mix_iq_files_rejects_mismatched_or_odd_sources(tmp_path):
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    a.write_bytes(b"1234")
    b.write_bytes(b"12")
    with pytest.raises(SpoofingError, match="same byte length"):
        mix_iq_files(a, b, out, sample_rate_hz=2, start_seconds=0, ramp_seconds=0,
                     initial_advantage_db=0, final_advantage_db=0)
    b.write_bytes(b"123")
    a.write_bytes(b"123")
    with pytest.raises(SpoofingError, match="even"):
        mix_iq_files(a, b, out, sample_rate_hz=2, start_seconds=0, ramp_seconds=0,
                     initial_advantage_db=0, final_advantage_db=0)


def _spoofing_config(tmp_path: Path) -> Path:
    nav = tmp_path / "input.nav"
    _rinex2_nav(nav)
    config = tmp_path / "spoof.yaml"
    config.write_text('''version: 2
scenario:
  name: seoul-carryoff
  constellation: GPS
  signal: L1CA
  utc: "2022-01-01T00:00:00Z"
  duration_seconds: 5
  position:
    type: static
    latitude_deg: 37.5665
    longitude_deg: 126.9780
    altitude_m: 120
input:
  rinex_nav: input.nav
output:
  root: runs
  rf_sample_rate_hz: 2600000
  sample_format: s8_iq
spoofing:
  attack_type: carry_off
  start_seconds: 1
  transition_seconds: 2
  target_offset_enu_m:
    east_m: 100
    north_m: 0
    up_m: 10
  prn_selection:
    mode: explicit
    prns: [3, 7]
  power:
    initial_advantage_db: -15
    final_advantage_db: 3
    ramp_seconds: 2
  keep_component_iq: true
simulator:
  executable: /tools/gps-sdr-sim
''')
    return config


def test_load_version2_spoofing_config(tmp_path):
    cfg = load_rf_config(_spoofing_config(tmp_path))
    assert cfg.version == 2
    assert cfg.spoofing.attack_type == "carry_off"
    assert cfg.spoofing.target_offset_enu_m == (100.0, 0.0, 10.0)
    assert cfg.spoofing.prn_selection.mode == "explicit"
    assert cfg.spoofing.prn_selection.prns == (3, 7)
    assert cfg.spoofing.power.final_advantage_db == 3.0
    assert cfg.spoofing.keep_component_iq is True


@pytest.mark.parametrize("old,new,match", [
    ("transition_seconds: 2", "transition_seconds: 0", "transition_seconds"),
    ("start_seconds: 1", "start_seconds: 5", "start_seconds"),
    ("prns: [3, 7]", "prns: [3, 3]", "duplicate"),
    ("east_m: 100", "east_m: 0", "non-zero"),
    ("final_advantage_db: 3", "final_advantage_db: .inf", "finite"),
])
def test_rejects_invalid_spoofing_config(tmp_path, old, new, match):
    path = _spoofing_config(tmp_path)
    path.write_text(path.read_text().replace(old, new))
    if old == "east_m: 100":
        path.write_text(path.read_text().replace("up_m: 10", "up_m: 0"))
    with pytest.raises(ConfigError, match=match):
        load_rf_config(path)


def test_version2_requires_spoofing_block(tmp_path):
    path = _spoofing_config(tmp_path)
    text = path.read_text().split("spoofing:\n", 1)[0] + "simulator:\n  executable: /tools/gps-sdr-sim\n"
    path.write_text(text)
    with pytest.raises(ConfigError, match="spoofing"):
        load_rf_config(path)


def test_spoofing_pipeline_generates_two_ensembles_composite_truth_and_manifest(tmp_path):
    cfg = load_rf_config(_spoofing_config(tmp_path))

    class FakeRunner:
        identity = "fake-gps-sdr-sim"
        executable = "/fake/gps-sdr-sim"
        provenance = "test"
        cli_contract = "test-contract"

        def __init__(self):
            self.calls = []

        def run(self, config, output, log):
            self.calls.append((config, output, log))
            value = np.array([[10, -10], [20, -20], [30, -30], [40, -40]], dtype=np.int8)
            if "counterfeit" in output.name:
                value = np.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.int8)
            value.tofile(output)
            log.write_text("fake")
            return {"command": [self.executable, "-o", output.name], "actual_bytes": output.stat().st_size}

    runner = FakeRunner()
    manifest_path = generate_spoofing_iq(cfg, runner)

    assert len(runner.calls) == 2
    manifest = json.loads(manifest_path.read_text())
    run_dir = manifest_path.parent
    assert manifest["schema"] == {"name": "gnss-doppler-lab.spoofing", "version": 1}
    assert manifest["attack"]["type"] == "carry_off"
    assert manifest["attack"]["prn_selection"] == {"mode": "explicit", "prns": [3, 7]}
    assert manifest["attack"]["target_offset_enu_m"] == [100.0, 0.0, 10.0]
    assert manifest["truth"]["spoofing_trajectory_sha256"]
    assert manifest["truth"]["filtered_nav_sha256"]
    assert manifest["iq"]["composite"]["sha256"]
    assert (run_dir / manifest["iq"]["authentic"]["path"]).is_file()
    assert (run_dir / manifest["iq"]["counterfeit"]["path"]).is_file()
    assert (run_dir / manifest["iq"]["composite"]["path"]).is_file()
    assert (run_dir / manifest["truth"]["spoofing_trajectory_path"]).is_file()
    assert (run_dir / manifest["truth"]["filtered_nav_path"]).is_file()


def test_spoofing_pipeline_refuses_normal_config(tmp_path):
    path = _spoofing_config(tmp_path)
    path.write_text(path.read_text().replace("version: 2", "version: 1").split("spoofing:\n", 1)[0] + "simulator: {}\n")
    cfg = load_rf_config(path)
    with pytest.raises(SpoofingError, match="version-2 spoofing"):
        generate_spoofing_iq(cfg, object())


def test_cli_dispatches_version2_to_spoofing_pipeline(tmp_path, monkeypatch, capsys):
    from gnss_doppler_lab import cli

    config_path = _spoofing_config(tmp_path)
    expected = tmp_path / "manifest.json"
    runner = object()
    monkeypatch.setattr(cli, "GpsSdrSimRunner", lambda executable: runner)
    seen = {}

    def fake_generate(config, actual_runner):
        seen["config"] = config
        seen["runner"] = actual_runner
        return expected

    monkeypatch.setattr(cli, "generate_spoofing_iq", fake_generate, raising=False)
    code = cli.main(["generate", str(config_path)])
    assert code == 0
    assert seen["config"].version == 2
    assert seen["runner"] is runner
    assert capsys.readouterr().out.strip() == str(expected)


def test_receiver_selects_composite_iq_from_spoofing_manifest():
    from gnss_doppler_lab.gnss_sdr import _source_iq_info

    info, role = _source_iq_info({
        "iq": {
            "authentic": {"path": "auth.bin"},
            "counterfeit": {"path": "fake.bin"},
            "composite": {"path": "mixed.bin", "sha256": "abc", "rf_sample_rate_hz": 2600000},
        }
    })
    assert role == "composite"
    assert info["path"] == "mixed.bin"


def test_receiver_keeps_normal_flat_iq_manifest_compatibility():
    from gnss_doppler_lab.gnss_sdr import _source_iq_info

    original = {"path": "gps_l1ca_s8_iq.bin", "sha256": "abc", "rf_sample_rate_hz": 2600000}
    info, role = _source_iq_info({"iq": original})
    assert role == "normal"
    assert info is original
