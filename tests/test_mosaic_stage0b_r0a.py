import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/mosaic_stage0b_r0_navbit_provenance"
AUDIT = ROOT / "artifacts/mosaic_stage0b_r0a_provenance_hardening"
VERIFY_PATH = ROOT / "scripts/verify_mosaic_stage0b_r0a.py"


def load_verifier():
    spec = importlib.util.spec_from_file_location("mosaic_stage0b_r0a_verifier", VERIFY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_independent_gps_parity_known_vector():
    verifier = load_verifier()
    transmitted = 0x22C00012  # TLM data 0x8b0000 with D29*=D30*=0.
    bits = [(transmitted >> shift) & 1 for shift in range(29, -1, -1)]
    assert tuple(bits[:8]) == verifier.PREAMBLE
    assert verifier.expected_parity(bits[:24], 0, 0) == bits[24:]
    corrupted = bits.copy()
    corrupted[12] ^= 1
    assert verifier.expected_parity(corrupted[:24], 0, 0) != corrupted[24:]


def test_compressed_bits_recompute_chain_preamble_how_and_determinism():
    verifier = load_verifier()
    rows = verifier.read_csv(FROZEN / "decoded_nav_bits.csv.gz", compressed=True)
    first, first_words = verifier.recompute_from_rows(rows)
    second, second_words = verifier.recompute_from_rows(rows)
    assert first == second and first_words == second_words
    assert first["total_validated_bits"] == 6000
    assert first["parity_valid_words"] == 200
    assert first["preambles_valid"] == 20
    assert first["tow_continuity_valid_prns"] == 10
    assert first["d29_d30_chain_errors"] == 0
    assert first["transmitted_bit_distribution"] == {"0": 3014, "1": 2986}
    assert {tuple(row["tow_seconds"]) for row in first["per_prn"] if row["dataset"].startswith("OAK")} == {(381636, 381642)}
    assert {tuple(row["tow_seconds"]) for row in first["per_prn"] if row["dataset"].startswith("TEX")} == {(477918, 477924)}


def test_tamper_negative_contract():
    tamper = json.loads((AUDIT / "tamper_negative_tests.json").read_text())
    assert tamper["all_expected_failures_observed"] is True
    assert len(tamper["tests"]) == 7
    assert all(item["passed"] for item in tamper["tests"])
    assert {item["name"] for item in tamper["tests"]} == {
        "validated_transmitted_bit_flip", "derived_transmitted_word_hex_change",
        "actual_how_bit_change_tow_csv_unchanged", "actual_bit_change_parity_csv_still_true",
        "preamble_bit_flip", "sample_start_change", "constant_plus_one_sequence",
    }


def test_receiver_flag_row_semantics_and_endpoint_distinction():
    semantics = json.loads((AUDIT / "receiver_boundary_semantics.json").read_text())
    assert semantics["nav_symbol_boundary_semantics"] == "END_OF_CURRENT_BIT"
    assert semantics["status"] == "CONFIRMED_OFF_BY_ONE_EPOCH"
    assert semantics["prompt_transition_rows"] == 754
    assert semantics["prompt_transitions_on_source_predicted_flag_to_next_edge"] == 754
    assert semantics["prompt_transitions_on_previous_to_flag_edge"] == 0
    endpoint = json.loads((AUDIT / "trace_endpoint_transcription.json").read_text())
    assert endpoint["trace_endpoint_transcription_error_samples"] == 0
    assert endpoint["semantic_mapping_status"] == "RAW_MAPPING_MISMATCH"


def test_multi_prn_common_intersection_and_scope_correction():
    common = json.loads((AUDIT / "common_injection_intervals.json").read_text())["datasets"]
    assert [common["OAKBAT.cleanStatic"]["common_raw_start_sample"], common["OAKBAT.cleanStatic"]["common_raw_end_sample_exclusive"]] == [150270296, 210197273]
    assert [common["TEXBAT.cleanStatic"]["common_raw_start_sample"], common["TEXBAT.cleanStatic"]["common_raw_end_sample_exclusive"]] == [817790304, 1117492038]
    assert len(common["OAKBAT.cleanStatic"]["included_prns"]) == 5
    assert len(common["TEXBAT.cleanStatic"]["included_prns"]) == 5
    scope = json.loads((AUDIT / "scope_limitations.json").read_text())
    assert scope["two_consecutive_subframes"] is True
    assert scope["two_separated_intervals"] is False
    assert scope["distant_interval_validation"] == "NOT_PERFORMED"


def test_no_attack_no_injection_and_artifact_checksum_fresh_clone_verifier():
    config = json.loads((AUDIT / "config.json").read_text())
    assert config["raw_prompt_redecoded"] is False
    assert config["attack_data_accessed"] is False
    assert config["synthetic_injection_performed"] is False
    assert config["receiver_executed"] is False
    verifier = load_verifier()
    verifier.main()
    verdict = json.loads((AUDIT / "final_verdict.json").read_text())
    assert verdict["verdict"] == "RAW_MAPPING_MISMATCH"
    assert verdict["stage0b_injection_authorized"] is False
