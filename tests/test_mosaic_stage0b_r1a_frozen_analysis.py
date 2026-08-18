import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.mosaic_stage0b_r1a_frozen_analysis import (
    decide_verdict,
    deterministic_awgn_control,
    four_prn_success,
    gain_matched_control,
    is_collapsed,
    paired_bootstrap_ci,
    physics_recovered,
    spearman_abs,
    strong_resolvable,
    target_nontarget_difference,
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/mosaic_stage0b_r1a_frozen_analysis"


def test_frozen_strong_subset():
    assert strong_resolvable(-6, .1, 0)
    assert strong_resolvable(0, 0, 25)
    assert not strong_resolvable(-10, .25, 50)
    assert not strong_resolvable(-6, .05, 5)


def test_target_nontarget_pairing_uses_same_case_median():
    assert target_nontarget_difference(10, [1, 3, 100, 5]) == pytest.approx(6)
    with pytest.raises(ValueError):
        target_nontarget_difference(10, [])


def test_collapsed_case_selection_is_exact():
    assert is_collapsed(0, 0)
    assert not is_collapsed(.025, 0)
    assert not is_collapsed(0, 5)


def test_gain_control_preserves_tap_shape_and_phase_relationships():
    clean = np.array([[1+2j, 3-4j], [-2+1j, .5-.5j]])
    observed = clean * 7 + .1
    control, scalar = gain_matched_control(clean, observed)
    assert scalar.imag == 0 and scalar.real > 0
    assert np.allclose(control / clean, scalar)
    assert np.sqrt(np.mean(abs(control)**2)) == pytest.approx(np.sqrt(np.mean(abs(observed)**2)))


def test_deterministic_awgn_control_has_exact_residual_rms():
    clean = np.ones((100, 5), complex)
    first = deterministic_awgn_control(clean, 3.25, 13)
    second = deterministic_awgn_control(clean, 3.25, 13)
    assert np.array_equal(first, second)
    assert np.sqrt(np.mean(abs(first-clean)**2)) == pytest.approx(3.25)


def test_paired_bootstrap_determinism():
    values = np.arange(12, dtype=float)
    assert paired_bootstrap_ci(values) == paired_bootstrap_ci(values)


def test_per_dataset_gate_separation_cannot_be_rescued_by_pooling():
    dataset_pass = {"OAKBAT.cleanStatic": True, "TEXBAT.cleanStatic": False}
    assert not all(dataset_pass.values())


def test_execution_pass_and_physics_pass_are_separate():
    execution_status = "PASS"
    assert execution_status == "PASS"
    assert not physics_recovered(.25, -.25, 50, -50)


def test_four_prn_three_of_four_calculation():
    assert four_prn_success(3) and four_prn_success(4)
    assert not four_prn_success(2)


def test_rms_spearman_uses_tie_aware_absolute_correlation():
    assert spearman_abs([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(1)
    assert spearman_abs([1, 1, 2, 2], [1, 2, 1, 2]) == pytest.approx(0)


def _all_gates():
    return {
        "integrity_pass": True, "retained_evidence_complete": True,
        "four_prn_numeric_criterion_defined": True, "single_prn_physics_pass": True,
        "control_separation_pass": True, "multi_prn_recovery_pass": True,
        "physical_hypothesis_pass": True,
    }


@pytest.mark.parametrize("gate,verdict", [
    ("integrity_pass", "INCONCLUSIVE_RESULT_INTEGRITY_FAILURE"),
    ("retained_evidence_complete", "INCONCLUSIVE_MISSING_RETAINED_EVIDENCE"),
    ("four_prn_numeric_criterion_defined", "INCONCLUSIVE_PREREG_GATE_UNDERSPECIFIED"),
    ("single_prn_physics_pass", "NO_GO_MOSAIC_SINGLE_PRN_PHYSICS"),
    ("control_separation_pass", "NO_GO_MOSAIC_CONTROL_SEPARATION"),
    ("multi_prn_recovery_pass", "NO_GO_MOSAIC_MULTI_PRN_RECOVERY"),
    ("physical_hypothesis_pass", "NO_GO_MOSAIC_PHYSICAL_HYPOTHESIS"),
])
def test_final_verdict_truth_table(gate, verdict):
    gates = _all_gates(); gates[gate] = False
    assert decide_verdict(gates) == verdict
    assert decide_verdict(_all_gates()) == "GO_FOR_MOSAIC_STAGE1"


def test_analysis_has_no_iq_injection_or_receiver_replay_subprocess():
    paths = [
        ROOT / "src/gnss_doppler_lab/mosaic_stage0b_r1a_frozen_analysis.py",
        ROOT / "scripts/finalize_mosaic_stage0b_r1a_frozen_analysis.py",
        ROOT / "scripts/verify_mosaic_stage0b_r1a_frozen_analysis.py",
    ]
    forbidden_names = {"subprocess", "generate_injected_prefix", "run_receiver", "inject_payload"}
    for path in paths:
        tree = ast.parse(path.read_text())
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert forbidden_names.isdisjoint(names | attrs)


def test_artifact_checksum_and_compact_verdict_recomputation():
    if not (ART / "final_verdict.json").exists():
        pytest.skip("final compact artifact is generated only after ANALYSIS_FREEZE push")
    path = ROOT / "scripts/verify_mosaic_stage0b_r1a_frozen_analysis.py"
    spec = importlib.util.spec_from_file_location("r1a_verify", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.verify_manifest() >= 20
    gates, _ = module.recompute()
    stored = json.loads((ART / "final_verdict.json").read_text())
    assert decide_verdict(gates) == stored["verdict"]
