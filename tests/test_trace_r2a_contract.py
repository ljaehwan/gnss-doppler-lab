import csv
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2a_reproducibility_repair"


def test_phase_a_handoffs_have_fixed_unique_channel_prn_and_pre_onset_sources():
    manifest = json.loads((ARTIFACT / "handoffs/manifest.json").read_text())
    assert manifest["guard_s_by_scenario"] == {"OAKBAT.OS3": 5.0, "TEXBAT.DS3": 5.0, "TEXBAT.cleanStatic": 30.0}
    for scenario in manifest["scenarios"].values():
        assert scenario["all_source_rows_pre_onset"] is True
        with Path(scenario["handoff_path"]).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert sorted(int(row["channel"]) for row in rows) == list(range(11))
        assert len({int(row["prn"]) for row in rows}) == 11
        minimum = int(scenario["guard_s_from_replay_start"] * 5_000_000)
        assert all(int(row["first_raw_interval_start_sample"]) >= minimum for row in rows)
        assert all(row["code_phase_convention"] == "residual_code_phase_chips_and_samples_zero_at_first_interval" for row in rows)
        assert all(row["carrier_phase_convention"].startswith("residual_carrier_phase_zero") for row in rows)
        assert all(row["acquisition_sample_stamp"] == "UNAVAILABLE_IN_PRESERVED_R2_RELEASE_SCHEMA" for row in rows)
        assert all(row["acquisition_metric"] == "UNAVAILABLE_IN_PRESERVED_R2_RELEASE_SCHEMA" for row in rows)


def test_receiver_repair_fails_closed_and_pins_raw_start():
    repair = ARTIFACT / "receiver_repair.diff"
    if not repair.exists():
        pytest.skip("repair diff is materialized by preregistration freeze")
    text = repair.read_text()
    assert "TRACE frozen handoff missed first raw sample" in text
    assert "TRACE frozen handoff PRN mismatch" in text
    assert "d_trace_handoff_first_raw_start_sample" in text
    assert "d_trace_handoff_carrier_doppler_hz" in text


def test_semantic_contract_does_not_relax_physical_values():
    path = ARTIFACT / "semantic_reproduction_contract.json"
    if not path.exists():
        pytest.skip("semantic contract is materialized by preregistration freeze")
    contract = json.loads(path.read_text())
    tolerance = contract["semantic_tolerances"]
    assert tolerance["float32_complex_taps_absolute"] == 0.0
    assert tolerance["float64_actions_and_state_absolute"] == 0.0
    assert tolerance["trace_score_absolute"] == 1e-12
    assert contract["decision_gate"]["physical_semantic_match_required"] is True


def test_frozen_configs_use_relative_dump_path_and_fixed_satellites():
    config_dir = ARTIFACT / "frozen_configs"
    if not config_dir.exists():
        pytest.skip("configs are materialized by preregistration freeze")
    for path in config_dir.rglob("*.conf"):
        text = path.read_text()
        assert "Tracking_1C.trace_dump_filename=trace_native_1ms_ch_" in text
        assert "Tracking_1C.trace_handoff_filename=" in text
        assert sum(line.startswith("Channel") and ".satellite=" in line for line in text.splitlines()) == 11
