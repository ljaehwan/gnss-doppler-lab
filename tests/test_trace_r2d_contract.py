import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2d_oakbat_clean_support_repair"
PARENT = ROOT / "artifacts/trace_stage0_r2c_terminal_drain_repair"
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r2d_diagnosis_localizes_clean_handoff_support_failure():
    diagnosis = json.loads((ARTIFACT / "diagnosis.json").read_text())
    assert diagnosis["status"] == "COMPLETE_BEFORE_REPAIR"
    assert diagnosis["diagnosis_label"] == "OAKBAT_OS3_HANDOFF_INCOMPATIBLE_WITH_CLEANSTATIC"
    assert diagnosis["evidence"]["parent_selected_unique_prns"] == [10, 11, 27]
    assert all(value == 0 for value in diagnosis["evidence"]["parent_role_pair_counts"].values())
    assert diagnosis["attack_performance_read_or_computed"] is False


def test_r2d_preregistration_changes_only_normal_clean_support_mapping():
    plan = json.loads((ARTIFACT / "repair_plan_preregistered.json").read_text())
    assert plan["status"] == "SEALED_BEFORE_SUPPORT_ACQUISITION"
    assert plan["repair_strategy"] == "CLEANSTATIC_SPECIFIC_NORMAL_ONLY_TARGET_ALIGNED_HANDOFF"
    assert plan["support_acquisition"]["selection_guard_s"] == 30.0
    assert plan["support_acquisition"]["quality_or_score_used_for_selection"] is False
    assert plan["support_acquisition"]["attack_data_used"] is False
    assert "45/20/15/20 chronological clean split with 5 s guards" in plan["unchanged"]


def test_r2d_clean_handoff_is_normal_only_and_raw_time_selected():
    manifest = json.loads((ARTIFACT / "handoff_manifest.json").read_text())
    clean = manifest["scenarios"]["OAKBAT.cleanStatic"]
    assert clean["normal_only"] is True
    assert clean["attack_data_used"] is False
    assert clean["guard_absolute_sample"] == 150_000_000
    assert clean["raw_offset_samples"] == 0
    with (ARTIFACT / "handoffs/oakbat_cleanstatic.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) >= 4
    assert [int(row["channel"]) for row in rows] == list(range(len(rows)))
    assert all(int(row["source_raw_interval_start_sample"]) >= 150_000_000 for row in rows)


def test_r2d_configs_preserve_phase_a_and_only_repair_clean_phase_b_mapping():
    runner = load_script("run_trace_stage0_r2d.py")
    for name in runner.driver.SCENARIOS:
        assert runner.frozen_config_text(name) == (ARTIFACT / f"frozen_configs/{runner.driver.SCENARIOS[name]['slug']}.conf").read_text()
        assert "SignalSource.enable_terminal_drain=true\n" in runner.frozen_config_text(name)
    clean = runner.frozen_phase_b_config_text("OAKBAT.cleanStatic")
    assert "SignalSource.seconds_to_skip=0.0\n" in clean
    assert "Tracking_1C.trace_handoff_filename=" + str((ARTIFACT / "handoffs/oakbat_cleanstatic.csv").resolve()) in clean
    for name in ("OAKBAT.OS3", "OAKBAT.OS4"):
        text = runner.frozen_phase_b_config_text(name)
        assert "SignalSource.seconds_to_skip=90.0\n" in text
        assert "handoffs/oakbat_os3.csv" in text


def test_r2d_reuses_byte_identical_r2c_terminal_drain_and_frozen_scorer():
    build = json.loads((ARTIFACT / "receiver_build_manifest.json").read_text())
    parent = json.loads((PARENT / "receiver_build_manifest.json").read_text())
    assert build["status"] == "PASS_REUSED_BYTE_IDENTICAL_R2C"
    assert build["receiver_executable"]["sha256"] == parent["receiver_executable"]["sha256"]
    assert sha(ARTIFACT / "receiver_repair.diff") == sha(PARENT / "receiver_repair.diff")
    prereg = json.loads((ARTIFACT / "preregistration.json").read_text())
    assert prereg["phase_b_scorer"]["frozen_sha256"] == sha(ROOT / "scripts/evaluate_trace_r2_phase_b.py")


def test_r2d_final_artifacts_are_fail_closed_or_computed():
    verdict_path = ARTIFACT / "final_verdict.json"
    if not verdict_path.exists():
        return
    verdict = json.loads(verdict_path.read_text())
    phase_a = json.loads((ARTIFACT / "rep3_rep4_reproduction_metrics.json").read_text())
    metrics = json.loads((ARTIFACT / "phase_b_metrics.json").read_text())
    assert verdict["phase_a_passed"] == (phase_a["phase_a_status"] == "PASS")
    assert verdict["phase_b_authorized"] == phase_a["phase_b_authorized"]
    if verdict["verdict"] in {"GO_TRACE_PHYSICAL_HYPOTHESIS", "NO_GO_TRACE_PHYSICAL_HYPOTHESIS"}:
        assert verdict["attack_metrics_computed"] is True
        assert metrics["status"] == "AVAILABLE"
    else:
        assert verdict["attack_metrics_computed"] is False
        assert metrics["status"] == "UNAVAILABLE"
