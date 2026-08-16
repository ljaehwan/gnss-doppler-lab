import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2e_attack_support_repair"
PARENT = ROOT / "artifacts/trace_stage0_r2d_oakbat_clean_support_repair"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def test_r2e_preregistration_is_bounded_and_preserves_frozen_contract():
    prereg = json.loads((ARTIFACT / "preregistration.json").read_text())
    assert prereg["status"] in {
        "SEALED_BEFORE_REPAIRED_SUPPORT_ACQUISITION_OR_METRIC_EVALUATION",
        "FROZEN_AFTER_PREREGISTERED_PRE_ONSET_HANDOFF_REPAIR",
    }
    assert prereg["task_base_commit"] == "e66619e31a937186e522d8566711436e24f2b99d"
    assert prereg["repair_strategy"] == "SCENARIO_SPECIFIC_PRE_ONSET_TARGET_ALIGNED_HANDOFF"
    assert prereg["scenario_repairs"]["TEXBAT.DS7"]["support_duration_s"] + 90.0 < 110.0
    assert prereg["scenario_repairs"]["OAKBAT.OS4"]["support_duration_s"] + 90.0 < 120.0
    assert prereg["frozen_phase_b_scorer"]["sha256"] == sha(ROOT / "scripts/evaluate_trace_r2_phase_b.py")
    assert prereg["attack_scores_read_or_computed"] is False


def test_r2e_attack_handoffs_are_scenario_specific_pre_onset_and_raw_time_selected():
    manifest = json.loads((ARTIFACT / "handoff_manifest.json").read_text())
    expectations = {
        "TEXBAT.DS7": ("texbat_ds7.csv", 2_375_000_000, 2_750_000_000),
        "OAKBAT.OS4": ("oakbat_os4.csv", 475_000_000, 600_000_000),
    }
    for name, (filename, threshold, onset) in expectations.items():
        item = manifest["scenarios"][name]
        assert item["scenario_specific"] is True
        assert item["pre_onset_only"] is True
        assert item["trace_scores_used"] is False
        assert item["attack_outcomes_used"] is False
        assert item["channel_count"] >= 4
        assert item["selection_absolute_sample"] == threshold
        assert item["onset_absolute_sample"] == onset
        with (ARTIFACT / "handoffs" / filename).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert [int(row["channel"]) for row in rows] == list(range(len(rows)))
        assert all(threshold <= int(row["source_raw_interval_start_sample"]) < onset for row in rows)


def test_r2e_only_changes_ds7_os4_phase_b_handoff_mapping():
    r2e_config = json.loads((ARTIFACT / "config.json").read_text())
    assert r2e_config["only_support_mapping_changes"] == {
        "TEXBAT.DS7": {"before": "texbat_ds3.csv", "after": "texbat_ds7.csv"},
        "OAKBAT.OS4": {"before": "oakbat_os3.csv", "after": "oakbat_os4.csv"},
    }
    for path in sorted((ARTIFACT / "frozen_configs").glob("*.conf")):
        parent = PARENT / "frozen_configs" / path.name
        before, after = config(parent), config(path)
        assert set(before) == set(after)
        assert {key for key in before if before[key] != after[key]} == {"Tracking_1C.trace_handoff_filename"}
        assert Path(before["Tracking_1C.trace_handoff_filename"]).name == Path(after["Tracking_1C.trace_handoff_filename"]).name
    for path in sorted((ARTIFACT / "frozen_configs/phase_b").glob("*.conf")):
        parent = PARENT / "frozen_configs/phase_b" / path.name
        before, after = config(parent), config(path)
        assert set(before) == set(after)
        changed = {key for key in before if before[key] != after[key]}
        old_name = Path(before["Tracking_1C.trace_handoff_filename"]).name
        new_name = Path(after["Tracking_1C.trace_handoff_filename"]).name
        if path.stem == "texbat_ds7":
            assert (old_name, new_name) == ("texbat_ds3.csv", "texbat_ds7.csv")
            assert changed == {"Tracking_1C.trace_handoff_filename"} | {
                key for key in before if key.startswith("Channel") and key.endswith(".satellite")
                and before[key] != after[key]
            }
        elif path.stem == "oakbat_os4":
            assert (old_name, new_name) == ("oakbat_os3.csv", "oakbat_os4.csv")
            assert changed == {"Tracking_1C.trace_handoff_filename"} | {
                key for key in before if key.startswith("Channel") and key.endswith(".satellite")
                and before[key] != after[key]
            }
        else:
            assert changed == {"Tracking_1C.trace_handoff_filename"}
            assert old_name == new_name


def test_r2e_reuses_byte_identical_receiver_terminal_drain_and_scorer():
    build = json.loads((ARTIFACT / "receiver_build_manifest.json").read_text())
    parent = json.loads((PARENT / "receiver_build_manifest.json").read_text())
    assert build["status"] == "PASS_REUSED_BYTE_IDENTICAL_R2C_R2D"
    assert build["receiver_executable"]["sha256"] == parent["receiver_executable"]["sha256"]
    assert sha(ARTIFACT / "receiver_repair.diff") == sha(PARENT / "receiver_repair.diff")
    assert "SignalSource.enable_terminal_drain=true\n" in (ARTIFACT / "frozen_configs/texbat_cleanstatic.conf").read_text()


def test_r2e_repaired_support_and_terminal_outcome_if_available():
    for filename in ("ds7_attack_support_audit.json", "os4_attack_support_audit.json"):
        audit = json.loads((ARTIFACT / filename).read_text())
        if audit["schema"].endswith(".v2"):
            assert audit["status"] == "PASS"
            assert audit["frozen_support"]["pre_onset_four_prn_block_count"] > 0
            assert audit["frozen_support"]["post_onset_four_prn_block_count"] > 0
    verdict_path = ARTIFACT / "final_verdict.json"
    if verdict_path.exists():
        verdict = json.loads(verdict_path.read_text())
        metrics = json.loads((ARTIFACT / "phase_b_metrics.json").read_text())
        if verdict["verdict"] in {"GO_TRACE_PHYSICAL_HYPOTHESIS", "NO_GO_TRACE_PHYSICAL_HYPOTHESIS"}:
            assert verdict["attack_metrics_computed"] is True
            assert metrics["status"] == "AVAILABLE"
        else:
            assert verdict["attack_metrics_computed"] is False
            assert verdict["performance_claimed"] is False
            assert metrics["status"] == "UNAVAILABLE"
