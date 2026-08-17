import gzip
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/mosaic_stage0b_r0c_boundary_phase_extrapolation"


def verifier():
    path = ROOT / "scripts/verify_mosaic_stage0b_r0c.py"
    spec = importlib.util.spec_from_file_location("r0c_verify_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_phase_is_fit_from_direct_flags_and_all_holdout_flags_match():
    v = verifier()
    inventory = v.read_csv(ART / "direct_flag_inventory.csv")
    fit = v.read_csv(ART / "phase_fit_summary.csv")
    holdout = v.read_csv(ART / "phase_holdout_validation.csv")
    phases = v.validate_phase(inventory, fit, holdout)
    assert phases == {
        ("OAKBAT.cleanStatic", 10): 19,
        ("OAKBAT.cleanStatic", 11): 12,
        ("OAKBAT.cleanStatic", 21): 5,
        ("OAKBAT.cleanStatic", 24): 2,
        ("OAKBAT.cleanStatic", 27): 14,
        ("TEXBAT.cleanStatic", 3): 0,
        ("TEXBAT.cleanStatic", 13): 19,
        ("TEXBAT.cleanStatic", 16): 4,
        ("TEXBAT.cleanStatic", 19): 4,
        ("TEXBAT.cleanStatic", 30): 11,
    }
    assert sum(int(row["fit_direct_flags"]) for row in fit) == 819
    assert sum(int(row["holdout_direct_flags"]) for row in holdout) == 823
    assert all(row["prompt_nav_used_for_selection"] == "False" for row in fit)
    assert all(row["holdout_mismatches"] == "0" for row in holdout)


def test_corrected_mapping_is_exact_r0b_mapping_and_matches_extrapolated_phase():
    v = verifier()
    phases = v.validate_phase(
        v.read_csv(ART / "direct_flag_inventory.csv"),
        v.read_csv(ART / "phase_fit_summary.csv"),
        v.read_csv(ART / "phase_holdout_validation.csv"),
    )
    mapping = v.read_csv(ART / "corrected_bit_mapping.csv.gz", compressed=True)
    report = v.validate_mapping(mapping, phases)
    assert report == {"bits": 6000, "phase_matches": 6000, "prompt_agreements": 6000}


def test_tracking_continuity_is_proven_for_all_ten_prns():
    v = verifier()
    rows = v.read_csv(ART / "tracking_continuity.csv")
    v.validate_continuity(rows)
    assert len(rows) == 10
    assert all(row["status"] == "PASS" for row in rows)
    assert all(row["raw_join_min_samples"] == "-1" and row["raw_join_max_samples"] == "1" for row in rows)


def test_post_selection_prompt_and_nav_validation():
    prompt = json.loads((ART / "prompt_transition_validation.json").read_text())
    nav = json.loads((ART / "nav_structure_validation.json").read_text())
    assert prompt["phase_selection_input"] is False
    assert prompt["observed_prompt_transition_alignment"] == "754/754"
    assert nav["phase_selection_input"] is False
    assert (nav["parity_valid_words"], nav["preambles_valid"], nav["tow_continuity_valid_prns"]) == (200, 20, 10)


def test_common_intervals_are_exact_and_scoped():
    value = json.loads((ART / "common_interval_validation.json").read_text())
    verifier().validate_common(value)
    assert value["injection_executed"] is False
    assert value["outside_interval_authorized"] is False


def test_all_requested_tamper_cases_are_rejected():
    value = json.loads((ART / "tamper_test_results.json").read_text())
    names = {item["name"] for item in value["tests"]}
    assert value["all_expected_failures_observed"]
    assert len(value["tests"]) == 12
    assert {"phase_plus_one", "phase_minus_one", "prn_phase_swap", "trace_flag_row_deleted",
            "trace_flag_row_duplicated", "raw_endpoint_changed", "unexplained_sample_gap",
            "tracking_reset", "frozen_bit_flip", "original_mapping_restored",
            "interval_start_expanded_one_sample", "interval_end_expanded_one_sample"} == names


def test_verdict_and_no_injection_or_attack_execution():
    verdict = json.loads((ART / "final_verdict.json").read_text())
    config = json.loads((ART / "config.json").read_text())
    assert verdict["verdict"] == "BOUNDARY_PHASE_EXTRAPOLATION_PASS_WITH_SCOPE_LIMITATION"
    assert verdict["stage0b_injection_authorized_within_validated_intervals"] is True
    assert verdict["injection_executed"] is False
    assert config["attack_data_accessed"] is False
    assert config["synthetic_injection_performed"] is False


def test_artifact_manifest_and_fresh_clone_verifier_contract():
    verifier().main()
