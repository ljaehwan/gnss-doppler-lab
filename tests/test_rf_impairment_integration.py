import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from gnss_doppler_lab.rf_config import ConfigError, _strict_integer, load_rf_config
from gnss_doppler_lab.rf_pipeline import generate_iq


def make_config(tmp_path: Path, impairments: str) -> Path:
    (tmp_path / "input.nav").write_text(
        "     2.10           N: GPS NAV DATA                         RINEX VERSION / TYPE\n"
        "    18                                                      LEAP SECONDS\n"
        "                                                            END OF HEADER\n"
    )
    p = tmp_path / "config.yaml"
    p.write_text(f'''version: 1
scenario:
  name: rf-test
  constellation: GPS
  signal: L1CA
  utc: "2026-07-11T03:04:05Z"
  duration_seconds: 1
  position:
    type: static
    latitude_deg: 37.5
    longitude_deg: 127.0
    altitude_m: 42
input:
  rinex_nav: input.nav
output:
  root: runs
  rf_sample_rate_hz: 2600000
  sample_format: s8_iq
{impairments}
simulator:
  executable: /fake/sim
''')
    return p


def test_impairment_yaml_validation_and_clean_default(tmp_path):
    clean = load_rf_config(make_config(tmp_path, ""))
    assert clean.impairments.enabled is False
    cfg = load_rf_config(make_config(tmp_path, "impairments:\n  enabled: true\n  profile: open_sky_normal\n  seed: 99"))
    assert cfg.impairments.enabled and cfg.impairments.seed == 99
    for bad, match in [
        ("  mystery: 1", "unknown impairment"),
        ("  seed: -1", "seed"),
        ("  sample_snr_db: .nan", "finite"),
        ("  frontend_cutoff_hz: 1400000", "Nyquist"),
        ("  fading_depth: 1.1", "fading_depth"),
    ]:
        text = "impairments:\n  enabled: true\n  profile: explicit\n" + bad
        with pytest.raises(ConfigError, match=match):
            load_rf_config(make_config(tmp_path, text))


def test_strict_integer_preserves_full_uint64_range_and_script_seed_round_trip(tmp_path):
    assert _strict_integer(2**53 + 1, "seed") == 2**53 + 1
    assert _strict_integer(2**64 - 1, "seed") == 2**64 - 1
    mod = load_script()
    row = {"run_id":"candidate-high-seed", "utc":"2026-01-01T00:00:00Z", "rinex_nav":"nav.rnx",
           "latitude_deg":"37", "longitude_deg":"127", "altitude_m":"20", "duration_seconds":"300"}
    expected = mod.impairment_seed(row["run_id"])
    path = mod.write_rf_config(row, tmp_path, tmp_path/"rf", Path("sim"))
    parsed = yaml.safe_load(path.read_text())["impairments"]["seed"]
    assert parsed == expected
    assert _strict_integer(parsed, "impairments.seed") == expected


@pytest.mark.parametrize("mapping,match", [
    ("impairments: {}", "empty"),
    ("impairments:\n  enabled: true\n  profile: clean", "clean.*disabled"),
    ("impairments:\n  enabled: false\n  profile: open_sky_normal", "open_sky_normal.*enabled"),
    ("impairments:\n  enabled: false\n  profile: explicit", "explicit.*enabled"),
    ("impairments:\n  enabled: true", "identity"),
    ("impairments:\n  enabled: true\n  profile: open_sky_normal\n  sample_snr_db: -10", "only"),
])
def test_profile_semantics_reject_empty_contradictory_or_preset_overrides(tmp_path, mapping, match):
    with pytest.raises(ConfigError, match=match):
        load_rf_config(make_config(tmp_path, mapping))


def test_enabled_pipeline_uses_clean_temp_and_records_provenance(tmp_path):
    cfg = load_rf_config(make_config(tmp_path, "impairments:\n  enabled: true\n  profile: explicit\n  seed: 12\n  sample_snr_db: 20"))
    class Runner:
        identity = "fake"; executable = "/fake/sim"; provenance = "test"; cli_contract = None
        def run(self, config, output, log):
            assert output.name != "gps_l1ca_s8_iq.bin"
            output.write_bytes(bytes([20, -20 & 255]) * 2_600_000)
            log.write_text("ok")
            return {"command": [self.executable], "actual_bytes": output.stat().st_size}
    manifest_path = generate_iq(cfg, Runner())
    doc = json.loads(manifest_path.read_text())
    assert doc["schema_version"] == 3
    assert doc["impairments"]["seed"] == 12
    assert doc["impairments"]["clean_input"]["sha256"]
    assert doc["impairments"]["output"]["sha256"] == doc["iq"]["sha256"]
    assert "not per-PRN" in doc["impairments"]["cn0_caveat"]
    assert doc["impairments"]["requested"]["profile"] == "explicit"
    assert doc["iq"]["expected_bytes"] == 5_200_000
    assert not list(manifest_path.parent.glob("*clean*"))


def load_script():
    path = Path(__file__).parents[1] / "scripts" / "run_normal_v3_large_pipeline.py"
    spec = importlib.util.spec_from_file_location("normal_v3_pipeline", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_normal_v3_yaml_defaults_realistic_with_stable_diverse_seed(tmp_path):
    mod = load_script()
    common = {"utc":"2026-01-01T00:00:00Z", "rinex_nav":"nav.rnx", "latitude_deg":"37",
              "longitude_deg":"127", "altitude_m":"20", "duration_seconds":"300"}
    a = mod.write_rf_config({**common, "run_id":"candidate-a"}, tmp_path, tmp_path/"rf", Path("sim"))
    b = mod.write_rf_config({**common, "run_id":"candidate-b"}, tmp_path, tmp_path/"rf", Path("sim"))
    da, db = yaml.safe_load(a.read_text()), yaml.safe_load(b.read_text())
    assert da["impairments"]["profile"] == "open_sky_normal"
    assert da["impairments"]["seed"] != db["impairments"]["seed"]
    a.unlink()
    a2 = mod.write_rf_config({**common, "run_id":"candidate-a"}, tmp_path, tmp_path/"rf", Path("sim"))
    assert yaml.safe_load(a2.read_text())["impairments"]["seed"] == da["impairments"]["seed"]


def test_normal_v3_can_write_clean_yaml(tmp_path):
    mod = load_script()
    row = {"run_id":"candidate-a", "utc":"2026-01-01T00:00:00Z", "rinex_nav":"nav.rnx",
           "latitude_deg":"37", "longitude_deg":"127", "altitude_m":"20", "duration_seconds":"300"}
    p = mod.write_rf_config(row, tmp_path, tmp_path/"rf", Path("sim"), impairment_profile="clean")
    assert yaml.safe_load(p.read_text())["impairments"] == {"enabled": False, "profile": "clean"}
