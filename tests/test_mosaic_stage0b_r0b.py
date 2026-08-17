import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/mosaic_stage0b_r0b_corrected_navbit_mapping"
R0 = ROOT / "artifacts/mosaic_stage0b_r0_navbit_provenance"


def verifier():
    path = ROOT / "scripts/verify_mosaic_stage0b_r0b.py"
    spec = importlib.util.spec_from_file_location("r0b_verify_test", path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
    return module


def test_corrected_evidence_exact_windows_endpoints_and_prompt_agreement():
    v = verifier()
    mapping = v.read_csv(ART / "corrected_navbit_sample_mapping.csv.gz", compressed=True)
    evidence = v.read_csv(ART / "corrected_epoch_evidence.csv.gz", compressed=True)
    report = v.validate_corrected(mapping, evidence, allow_boundary_flag_gaps=True)
    assert len(mapping) == 6000 and len(evidence) == 126000
    assert report["prompt_agreements"] == 6000
    assert all(item["bits"] == 600 for item in report["per_prn"])
    assert report["start_nonboundary_passes"] == 6000
    assert report["internal_boundary_failures"] == 0


def test_direct_receiver_boundary_contract_fails_closed_for_presync_bits():
    boundary = json.loads((ART / "corrected_boundary_validation.json").read_text())
    assert boundary["previous_boundary_flag_passes"] == 854
    assert boundary["end_boundary_flag_passes"] == 861
    assert boundary["boundary_flag_failure_bits"] == 5146
    assert boundary["status"] == "FAIL"
    verdict = json.loads((ART / "final_verdict.json").read_text())
    assert verdict["verdict"] == "CORRECTED_BOUNDARY_STRUCTURE_FAIL"
    assert verdict["stage0b_injection_mapping_ready"] is False


def test_frozen_sequence_and_independent_gps_structure_unchanged():
    science = json.loads((ART / "independent_bit_recomputation.json").read_text())["recomputation"]
    assert science["total_validated_bits"] == 6000
    assert science["parity_valid_words"] == 200
    assert science["preambles_valid"] == 20
    assert science["tow_continuity_valid_prns"] == 10
    assert science["d29_d30_chain_errors"] == 0
    assert science["transmitted_bit_distribution"] == {"0": 3014, "1": 2986}
    assert science["unique_prn_sequence_hashes"] == 10


def test_corrected_common_intersection_is_recomputed():
    common = json.loads((ART / "corrected_common_injection_intervals.json").read_text())["recomputed"]["datasets"]
    oak = common["OAKBAT.cleanStatic"]; tex = common["TEXBAT.cleanStatic"]
    assert [oak["corrected_common_raw_start_sample"], oak["corrected_common_raw_end_sample_exclusive"]] == [150275296, 210202273]
    assert [tex["corrected_common_raw_start_sample"], tex["corrected_common_raw_end_sample_exclusive"]] == [817815304, 1117517038]
    assert oak["all_five_prns_simultaneous"] and tex["all_five_prns_simultaneous"]


def test_all_requested_tamper_negatives_are_rejected():
    tamper = json.loads((ART / "tamper_negative_tests.json").read_text())
    assert tamper["all_expected_failures_observed"]
    assert len(tamper["tests"]) == 11
    assert all(item["passed"] for item in tamper["tests"])


def test_no_attack_no_injection_scope_and_frozen_inputs_untouched():
    config = json.loads((ART / "config.json").read_text())
    assert config["attack_data_accessed"] is False
    assert config["synthetic_injection_performed"] is False
    assert config["raw_prompt_bit_search_performed"] is False
    scope = json.loads((ART / "scope_limitations.json").read_text())
    assert scope["two_consecutive_subframes"] is True
    assert scope["two_separated_intervals"] is False
    assert scope["distant_interval_validation"] == "NOT_PERFORMED"
    assert (R0 / "artifact_manifest_sha256.json").is_file()


def test_artifact_checksum_and_fresh_clone_verifier_contract():
    verifier().main()
