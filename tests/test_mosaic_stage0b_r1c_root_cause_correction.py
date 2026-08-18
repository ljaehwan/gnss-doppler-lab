import ast
import csv
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from gnss_doppler_lab.mosaic_stage0b_r1c_correction import (
    decide_recommendation, deterministic_bootstrap_median_difference,
    discriminative_verdict, exact_permutation_mean_test,
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/mosaic_stage0b_r1c_root_cause_correction"
PROTECTED = (
    "artifacts/mosaic_stage0b_r1_execution",
    "artifacts/mosaic_stage0b_r1a_frozen_analysis",
    "artifacts/mosaic_stage0b_r1b_multiprn_root_cause",
)
BASE_SHA = "664a37ad02eb536673e3dd2ec32668df6218d53e"


def artifact_rows():
    with (ART / "failed_vs_successful_metrics.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_r1_r1a_r1b_artifacts_byte_preserved():
    changed = subprocess.run(["git", "diff", "--name-only", BASE_SHA, "--", *PROTECTED],
                             cwd=ROOT, check=True, text=True, capture_output=True).stdout
    assert changed == ""


def test_no_iq_injection_receiver_replay_or_case_regeneration():
    paths = [ROOT / "src/gnss_doppler_lab/mosaic_stage0b_r1c_correction.py",
             ROOT / "scripts/run_mosaic_stage0b_r1c_root_cause_correction.py",
             ROOT / "scripts/verify_mosaic_stage0b_r1c_root_cause_correction.py"]
    forbidden = {"generate_injected_prefix", "run_receiver", "inject_payload",
                 "decode_interleaved_int16", "generate_cases", "regenerate_cases"}
    for path in paths:
        tree = ast.parse(path.read_text())
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert forbidden.isdisjoint(names | attrs)
    config = json.loads((ART / "config.json").read_text())
    assert not config["iq_injection_rerun"] and not config["receiver_replay_rerun"]
    assert not config["case_regeneration"] and not config["new_cases"]


def test_original_no_go_and_r1b_reproduction_preserved():
    reproduction = json.loads((ART / "reproduction_check.json").read_text())
    assert reproduction["status"] == "PASS"
    assert reproduction["r1a_final_verdict"] == "NO_GO_MOSAIC_MULTI_PRN_RECOVERY"
    assert reproduction["r1b_primary_verdict"] == "MIXED_OR_UNIDENTIFIED_ROOT_CAUSE"


def test_failure_success_accounting_and_target_prn_separation():
    rows = artifact_rows()
    assert len(rows) == 28
    assert sum(row["comparator_role"] == "failure_target" for row in rows) == 8
    assert sum(row["comparator_role"] == "success_comparator" for row in rows) == 20
    assignments = json.loads((ROOT / "artifacts/mosaic_stage0b_r1_receiver_in_loop/case_target_assignment.json").read_text())
    targets = {row["case_id"]: set(map(int, row["target_prns"])) for row in assignments["assignments"]}
    assert all(int(row["target_prn"]) in targets[row["case_id"]] for row in rows)


def test_h3_requires_comparator_discrimination():
    assert discriminative_verdict(failure_presence=True, comparator_presence=True,
                                  complete_separation=False) == "PRESENT_BUT_NOT_DISCRIMINATIVE"
    verdict = json.loads((ART / "h3_template_discrimination.json").read_text())["verdict"]
    assert verdict in {"SUPPORTED_AND_DISCRIMINATIVE", "PRESENT_BUT_NOT_DISCRIMINATIVE",
                       "INCONCLUSIVE", "UNSUPPORTED"}
    assert verdict != "SUPPORTED"


def test_h4_cannot_be_supported_by_one_or_non_target_lock_loss():
    assert discriminative_verdict(failure_presence=True, comparator_presence=True,
                                  complete_separation=False) == "PRESENT_BUT_NOT_DISCRIMINATIVE"
    h4 = json.loads((ART / "h4_lock_discrimination.json").read_text())
    assert h4["target_prns_only"]
    assert h4["verdict"] != "SUPPORTED"
    assert not h4["matched_subset"]["complete_matching_possible"]


def test_h6_is_evidence_driven_not_hard_coded():
    source = (ROOT / "scripts/run_mosaic_stage0b_r1c_root_cause_correction.py").read_text()
    assert "dilution=False" not in source and "dilution = False" not in source
    h6 = json.loads((ART / "h6_temporal_dilution.json").read_text())
    assert h6["hard_coded_dilution_removed"]
    assert h6["verdict"] == "NOT_TESTABLE_FROM_RETAINED_EVIDENCE"
    assert h6["paired_window_lengths"] == [.001, .1, .5, 1.0, 6.0]


def test_recommendation_truth_table_is_not_hard_coded():
    base = dict(reproduction_pass=True, r1a_no_go=True, stage1_same_cases_prohibited=True,
                independent_implementation_defect=False, oracle_consistently_recovers=False,
                successful_comparators_not_degraded=False, negative_result_preserved=True)
    assert decide_recommendation(**base) == "TERMINATE_MOSAIC_AS_STAGE1_PATH"
    corrected = {**base, "independent_implementation_defect": True,
                 "oracle_consistently_recovers": True, "successful_comparators_not_degraded": True}
    assert decide_recommendation(**corrected) == "CORRECTED_OBSERVER_REQUIRES_INDEPENDENT_DATA"
    unresolved = {**base, "reproduction_pass": False}
    assert decide_recommendation(**unresolved) == "ROOT_CAUSE_REMAINS_UNRESOLVED"


def test_small_sample_exact_permutation_test():
    result = exact_permutation_mean_test([1, 2], [3, 4])
    assert result["permutations"] == 6 and result["exact"]
    assert result["p_value_two_sided"] == pytest.approx(2/6)


def test_deterministic_target_bootstrap_reproduction():
    first = deterministic_bootstrap_median_difference([1, 2, 9], [3, 4, 5], replicates=1000)
    second = deterministic_bootstrap_median_difference([1, 2, 9], [3, 4, 5], replicates=1000)
    assert first == second


def test_row_level_labels_are_separate_from_global_verdicts():
    rows = artifact_rows()
    assert all("H3" not in row["row_level_labels"] and "H4" not in row["row_level_labels"] for row in rows)
    global_verdicts = json.loads((ART / "corrected_hypothesis_verdicts.json").read_text())["verdicts"]
    assert global_verdicts["H3"]["verdict"] == "PRESENT_BUT_NOT_DISCRIMINATIVE"
    assert global_verdicts["H4"]["verdict"] == "PRESENT_BUT_NOT_DISCRIMINATIVE"


def test_artifact_checksum_and_fresh_clone_verifier():
    path = ROOT / "scripts/verify_mosaic_stage0b_r1c_root_cause_correction.py"
    spec = importlib.util.spec_from_file_location("r1c_verify", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    result = module.verify()
    assert result["status"] == "PASS" and result["raw_science_regenerated"] is False
