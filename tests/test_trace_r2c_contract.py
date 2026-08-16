import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2c_terminal_drain_repair"
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_r2c_diagnosis_localizes_complete_terminal_pair_before_repair():
    diagnosis = json.loads((ARTIFACT / "diagnosis.json").read_text())
    mismatch = diagnosis["exact_mismatch"]
    assert diagnosis["status"] == "COMPLETE_BEFORE_RECEIVER_REPAIR"
    assert diagnosis["diagnosis_label"] == "FILE_SOURCE_STOP_PREEMPTS_TERMINAL_CHANNEL_DRAIN"
    assert mismatch["dataset"] == "TEXBAT.cleanStatic"
    assert mismatch["extra_native_record"]["channel"] == 0
    assert mismatch["extra_native_record"]["prn"] == 19
    assert mismatch["extra_native_record"]["raw_interval_end_sample"] < mismatch["raw_source_end_exclusive"]
    assert mismatch["block_key"]["rep3_pair_count"] - mismatch["block_key"]["rep4_pair_count"] == 1
    assert diagnosis["attack_performance_read_or_computed"] is False


def test_r2c_preregistration_preserves_frozen_science_and_requires_row_closure():
    plan = json.loads((ARTIFACT / "repair_plan_preregistered.json").read_text())
    frozen = plan["frozen_phase_a"]
    assert plan["repair_strategy"] == "OPT_IN_NATURAL_EOS_FLOWGRAPH_DRAIN"
    assert plan["trace_math_threshold_tolerance_window_or_gate_changes_authorized"] is False
    assert plan["source_range_or_duration_change_authorized"] is False
    assert plan["evaluation_code_change_authorized"] is False
    assert frozen["whole_replay_row_set_must_match"] is True
    assert frozen["terminal_row_counts_per_prn_channel_must_match"] is True
    assert frozen["trace_score_absolute_tolerance"] == 1e-12


def test_r2c_frozen_configs_enable_opt_in_terminal_drain():
    runner = load_script("run_trace_stage0_r2c.py")
    for name in runner.driver.SCENARIOS:
        text = runner.frozen_config_text(name)
        assert "SignalSource.enable_terminal_drain=true\n" in text
        assert "Tracking_1C.trace_dump=true\n" in text


def test_r2c_phase_b_dump_count_is_derived_from_frozen_handoff():
    source = (ROOT / "scripts/run_trace_stage0_r2a.py").read_text()
    phase_b = source[source.index("def run_phase_b_receiver") : source.index("def main")]
    assert '"expected_dump_file_count": len(rows)' in phase_b
    assert "len(record_counts) == len(rows)" in phase_b


def test_r2c_receiver_patch_waits_for_natural_eos_without_row_filtering():
    patch = (ARTIFACT / "receiver_repair.diff").read_text()
    assert "SignalSource.enable_terminal_drain" not in patch
    assert 'enable_terminal_drain_(configuration->property(role_ + ".enable_terminal_drain"s, false))' in patch
    assert 'command_event_make(200, action)' in patch
    assert 'LOG(INFO) << "Received action DRAIN"' in patch
    assert "flowgraph_->wait();" in patch
    assert "terminal_cutoff" not in patch
    assert "cn0_min" not in patch[patch.find("gnss_sdr_valve.cc") : patch.find("gnss_sdr_valve.h")]


def test_r2c_phase_a_requires_whole_row_set_and_terminal_count_identity():
    metrics = json.loads((ARTIFACT / "rep3_rep4_reproduction_metrics.json").read_text())
    audit = json.loads((ARTIFACT / "terminal_row_set_audit.json").read_text())
    checks = metrics["semantic_reproduction_gate"]["checks"]
    assert checks["whole_replay_row_set_identical"] == audit["whole_replay_row_set_identical"]
    assert checks["terminal_row_counts_per_prn_channel_identical"] == audit["terminal_row_counts_per_prn_channel_identical"]
    if metrics["phase_a_status"] == "PASS":
        assert audit["status"] == "PASS"
        assert audit["rep3_only_rows"] == audit["rep4_only_rows"] == 0
        assert audit["rep3"]["row_set_sha256"] == audit["rep4"]["row_set_sha256"]
        assert metrics["phase_b_authorized"] is True
    else:
        assert metrics["phase_b_authorized"] is False


def test_r2c_verdict_and_phase_b_metric_availability_are_fail_closed():
    metrics = json.loads((ARTIFACT / "rep3_rep4_reproduction_metrics.json").read_text())
    verdict = json.loads((ARTIFACT / "final_verdict.json").read_text())
    assert verdict["phase_a_passed"] == (metrics["phase_a_status"] == "PASS")
    assert verdict["phase_b_authorized"] == metrics["phase_b_authorized"]
    if not verdict["phase_b_authorized"]:
        assert verdict["phase_b_run"] is False
        assert verdict["attack_metrics_computed"] is False
        assert verdict["normal_fpr"]["status"] == "UNAVAILABLE"
