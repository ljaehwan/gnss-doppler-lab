import json
from pathlib import Path
import subprocess
import hashlib
from dataclasses import replace

import pytest

from gnss_doppler_lab.rf_config import ConfigError, load_rf_config
from gnss_doppler_lab.gps_sdr_sim import GpsSdrSimRunner, SimulatorError
from gnss_doppler_lab.rf_pipeline import generate_iq


def config_file(tmp_path, *, position="static"):
    nav = tmp_path / "input.nav"
    nav.write_text("     2.10           N: GPS NAV DATA                         RINEX VERSION / TYPE\n")
    cfg = tmp_path / "scenario.yaml"
    cfg.write_text(f'''version: 1
scenario:
  name: seoul-normal
  constellation: GPS
  signal: L1CA
  utc: "2026-07-11T03:04:05Z"
  duration_seconds: 2
  position:
    type: {position}
    latitude_deg: 37.5
    longitude_deg: 127.0
    altitude_m: 42
input:
  rinex_nav: input.nav
output:
  root: runs
  rf_sample_rate_hz: 2600000
  sample_format: s8_iq
simulator:
  executable: /tools/gps-sdr-sim
''')
    return cfg, nav


def test_load_versioned_config_and_resolve_paths(tmp_path):
    cfg_path, nav = config_file(tmp_path)
    cfg = load_rf_config(cfg_path)
    assert cfg.version == 1
    assert cfg.input.rinex_nav == nav.resolve()
    assert cfg.output.root == (tmp_path / "runs").resolve()
    assert cfg.output.rf_sample_rate_hz == 2_600_000
    assert cfg.scenario.utc.isoformat() == "2026-07-11T03:04:05+00:00"


def test_rejects_unknown_version_and_trajectory(tmp_path):
    cfg_path, _ = config_file(tmp_path)
    text = cfg_path.read_text().replace("version: 1", "version: 2")
    cfg_path.write_text(text)
    with pytest.raises(ConfigError, match="version"):
        load_rf_config(cfg_path)
    cfg_path, _ = config_file(tmp_path, position="trajectory")
    with pytest.raises(ConfigError, match="not supported"):
        load_rf_config(cfg_path)


def test_discovery_precedence_and_command_contract(tmp_path, monkeypatch):
    cfg_path, nav = config_file(tmp_path)
    cfg = load_rf_config(cfg_path)
    monkeypatch.setenv("GPS_SDR_SIM", "/env/sim")
    runner = GpsSdrSimRunner(executable="/explicit/sim")
    assert runner.executable == "/explicit/sim"
    out = tmp_path / "x.bin"
    assert runner.build_command(cfg, out) == [
        "/explicit/sim", "-e", str(nav.resolve()), "-l", "37.50000000,127.00000000,42.000",
        "-t", "2026/07/11,03:04:05", "-d", "2", "-s", "2600000", "-b", "8", "-o", str(out),
    ]


def test_relative_explicit_executable_is_resolved_before_changing_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tool = tmp_path / "tools" / "gps-sdr-sim"
    tool.parent.mkdir()
    tool.write_text("binary")

    runner = GpsSdrSimRunner(executable="tools/gps-sdr-sim")

    assert runner.executable == str(tool.resolve())


def test_runner_uses_safe_argv_atomic_output_and_logs(tmp_path, monkeypatch):
    cfg_path, _ = config_file(tmp_path)
    cfg = load_rf_config(cfg_path)
    final = tmp_path / "iq.bin"
    log = tmp_path / "sim.log"
    seen = {}
    def fake_run(argv, **kwargs):
        seen.update(argv=argv, kwargs=kwargs)
        (Path(kwargs["cwd"]) / argv[-1]).write_bytes(b"IQ")
        return subprocess.CompletedProcess(argv, 0, "ok", "warn")
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = GpsSdrSimRunner(executable="/sim").run(cfg, final, log)
    assert final.read_bytes() == b"IQ"
    assert result["actual_bytes"] == 2
    assert seen["kwargs"]["shell"] is False
    assert Path(seen["kwargs"]["cwd"]) == final.parent
    assert Path(seen["argv"][-1]).name == seen["argv"][-1]
    assert "stdout:\nok" in log.read_text()
    assert not list(tmp_path.glob("*.tmp"))


def test_runner_failure_and_empty_output_do_not_publish(tmp_path, monkeypatch):
    cfg_path, _ = config_file(tmp_path)
    cfg = load_rf_config(cfg_path)
    def failed(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 3, "", "bad")
    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(SimulatorError, match="exit code 3"):
        GpsSdrSimRunner(executable="/sim").run(cfg, tmp_path/"iq.bin", tmp_path/"log")
    def empty(argv, **kwargs):
        Path(argv[-1]).touch()
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(subprocess, "run", empty)
    with pytest.raises(SimulatorError, match="empty"):
        GpsSdrSimRunner(executable="/sim").run(cfg, tmp_path/"iq.bin", tmp_path/"log")


def test_runner_rejects_rinex3_before_execution(tmp_path, monkeypatch):
    cfg_path, nav = config_file(tmp_path)
    nav.write_text("     3.05           N: GNSS NAV DATA    M: MIXED            RINEX VERSION / TYPE\n")
    cfg = load_rf_config(cfg_path)

    def must_not_run(*args, **kwargs):
        raise AssertionError("subprocess must not run for an incompatible NAV file")

    monkeypatch.setattr(subprocess, "run", must_not_run)
    with pytest.raises(SimulatorError, match="RINEX 2"):
        GpsSdrSimRunner(executable="/sim").run(
            cfg, tmp_path / "iq.bin", tmp_path / "sim.log"
        )


def test_pipeline_deterministic_layout_and_manifest(tmp_path):
    cfg_path, nav = config_file(tmp_path)
    cfg = load_rf_config(cfg_path)
    class FakeRunner:
        identity = "fake-gps-sdr-sim"
        executable = "/fake/sim"
        def build_command(self, config, output): return [self.executable, "-o", str(output)]
        def run(self, config, output, log):
            output.write_bytes(b"1234")
            log.write_text("fake")
            return {"command": self.build_command(config, output), "actual_bytes": 4}
    manifest_path = generate_iq(cfg, FakeRunner())
    assert manifest_path == cfg.output.root / "seoul-normal_20260711T030405Z" / "manifest.json"
    m = json.loads(manifest_path.read_text())
    assert m["schema_version"] == 1
    assert m["input"]["rinex_nav_sha256"]
    assert m["iq"]["actual_bytes"] == 4
    assert m["iq"]["complex_samples"] == 2
    assert m["iq"]["actual_duration_seconds"] == pytest.approx(2 / 2_600_000)
    assert m["iq"]["sample_format"] == "s8_iq"
    assert m["simulator"]["identity"] == "fake-gps-sdr-sim"
    with pytest.raises(FileExistsError):
        generate_iq(cfg, FakeRunner())


@pytest.mark.parametrize("name", ["../escape", "a/b", ".", "", "bad name", "x\\y"])
def test_rejects_unsafe_scenario_names(tmp_path, name):
    cfg_path, _ = config_file(tmp_path)
    cfg_path.write_text(cfg_path.read_text().replace("name: seoul-normal", f"name: {json.dumps(name)}"))
    with pytest.raises(ConfigError, match="scenario.name"):
        load_rf_config(cfg_path)


@pytest.mark.parametrize("old,new", [
    ("duration_seconds: 2", "duration_seconds: 0"),
    ("duration_seconds: 2", "duration_seconds: 86401"),
    ("rf_sample_rate_hz: 2600000", "rf_sample_rate_hz: 999999"),
    ("latitude_deg: 37.5", "latitude_deg: .nan"),
    ("longitude_deg: 127.0", "longitude_deg: .inf"),
    ("altitude_m: 42", "altitude_m: -.inf"),
])
def test_rejects_values_outside_pinned_simulator_limits(tmp_path, old, new):
    cfg_path, _ = config_file(tmp_path)
    cfg_path.write_text(cfg_path.read_text().replace(old, new))
    with pytest.raises(ConfigError):
        load_rf_config(cfg_path)


def test_runner_stages_short_nav_name_and_returns_exact_argv(tmp_path, monkeypatch):
    deep = tmp_path / ("n" * 180)
    deep.mkdir()
    cfg_path, _ = config_file(deep)
    cfg = load_rf_config(cfg_path)
    seen = {}
    def fake_run(argv, **kwargs):
        seen["argv"] = list(argv)
        nav_arg = argv[argv.index("-e") + 1]
        assert not Path(nav_arg).is_absolute()
        assert len(nav_arg) < 32
        assert (Path(kwargs["cwd"]) / nav_arg).is_file()
        (Path(kwargs["cwd"]) / argv[-1]).write_bytes(b"IQ")
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    out = tmp_path / "out" / "iq.bin"
    result = GpsSdrSimRunner("/sim").run(cfg, out, tmp_path / "log")
    assert result["command"] == seen["argv"]
    assert not (out.parent / "nav.rnx").exists()


def test_manifest_records_unverified_executable_hash_and_cli_contract(tmp_path):
    cfg_path, _ = config_file(tmp_path)
    cfg = load_rf_config(cfg_path)
    executable = tmp_path / "arbitrary-sim"
    executable.write_bytes(b"not the pinned build")
    class FakeRunner:
        identity = "must-not-be-presented-as-pinned"
        cli_contract = "osqzss/gps-sdr-sim@28ca29a6719475195e3aabd5930c4ed02d67190f"
        provenance = "unverified"
        def run(self, config, output, log):
            output.write_bytes(b"IQ"); log.write_text("")
            return {"command": [self.executable, "--actual"], "actual_bytes": 2}
    runner = FakeRunner()
    runner.executable = str(executable)
    m = json.loads(generate_iq(cfg, runner).read_text())
    sim = m["simulator"]
    assert sim["provenance"] == "unverified"
    assert sim["cli_contract"] == FakeRunner.cli_contract
    assert sim["executable_sha256"] == hashlib.sha256(executable.read_bytes()).hexdigest()
    assert sim["command"] == [str(executable), "--actual"]


def test_pipeline_rejects_odd_s8_iq_byte_count(tmp_path):
    cfg_path, _ = config_file(tmp_path)
    cfg = load_rf_config(cfg_path)
    class OddRunner:
        executable = "/fake"; identity = "fake"
        def run(self, config, output, log):
            output.write_bytes(b"IQX"); log.write_text("")
            return {"command": [self.executable], "actual_bytes": 3}
    with pytest.raises(ValueError, match="even"):
        generate_iq(cfg, OddRunner())


def test_pipeline_defensively_keeps_run_directory_under_output_root(tmp_path):
    cfg_path, _ = config_file(tmp_path)
    cfg = load_rf_config(cfg_path)
    cfg = replace(cfg, scenario=replace(cfg.scenario, name="../escape"))
    class MustNotRun:
        def run(self, *args):
            raise AssertionError("unsafe run directory must be rejected before execution")
    with pytest.raises(ValueError, match="output root"):
        generate_iq(cfg, MustNotRun())
    assert not (tmp_path / "escape_20260711T030405Z").exists()


def test_runner_provenance_hashes_available_path_executable(tmp_path, monkeypatch):
    executable = tmp_path / "gps-sdr-sim"
    executable.write_bytes(b"arbitrary executable")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    runner = GpsSdrSimRunner()
    probe = runner.probe()
    assert runner.identity == "unverified gps-sdr-sim executable"
    assert probe["provenance"] == "unverified"
    assert probe["cli_contract"].endswith(GpsSdrSimRunner.pinned_commit)
    assert probe["executable_sha256"] == hashlib.sha256(executable.read_bytes()).hexdigest()
