from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gnss_doppler_lab.acaf_nf_stage1_r3 import (
    ATTACK_PHASES, CENTER, GRID, NFConfig, PRIMARY_FAMILY, SetNeuralField,
    aggregate_l20, assert_no_byte_overlap, assert_no_clean_attack_time_overlap,
    attack_phase, fixed_policy_orders, gaussian_nll, pool_scores,
    role_for_support, sequential_trace, verify_freeze_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def test_exact_clean_roles_guards_and_causal_l20_boundary_exclusion():
    fs = 25_000_000
    assert role_for_support(10 * fs, 10 * fs + 500_000) == "train"
    assert role_for_support(45 * fs - 500_000, 45 * fs) == "train"
    assert role_for_support(45 * fs - 250_000, 45 * fs + 250_000) is None
    assert role_for_support(47 * fs, 47 * fs + 500_000) == "selection"
    assert role_for_support(64 * fs, 64 * fs + 500_000) == "calibration"
    assert role_for_support(84 * fs, 84 * fs + 500_000) == "holdout"
    assert role_for_support(100 * fs, 100 * fs + 500_000) is None


def test_role_byte_overlap_and_same_time_underlay_audit():
    a = {"recording_sha256": "a" * 64, "raw_start_sample": 10, "raw_end_sample": 30}
    b = {"recording_sha256": "a" * 64, "raw_start_sample": 20, "raw_end_sample": 40}
    with pytest.raises(ValueError):
        assert_no_byte_overlap({"train": [a], "selection": [b]})
    assert_no_byte_overlap({"train": [a], "selection": [{**b, "recording_sha256": "b" * 64}]})
    with pytest.raises(ValueError):
        assert_no_clean_attack_time_overlap([a], [{"raw_start_sample": 29, "raw_end_sample": 50}])
    assert_no_clean_attack_time_overlap([a], [{"raw_start_sample": 30, "raw_end_sample": 50}])


def test_per_ms_gain_and_global_phase_normalization_is_invariant():
    rng = np.random.default_rng(4)
    surfaces = rng.normal(size=(20, 11, 17)) + 1j * rng.normal(size=(20, 11, 17))
    surfaces[:, 5, 8] += 8 + 3j
    base, _ = aggregate_l20(surfaces)
    for gain in (.5, .8, 1.2, 2.0):
        for phase in (0, np.pi / 4, np.pi / 2, np.pi):
            transformed, _ = aggregate_l20(surfaces * gain * np.exp(1j * phase))
            assert np.allclose(base, transformed, atol=2e-9, rtol=2e-9)


def test_complex_query_count_center_first_and_policy_determinism():
    orders_a, orders_b = fixed_policy_orders(20260808), fixed_policy_orders(20260808)
    assert orders_a == orders_b
    assert all(order[0] == CENTER for order in orders_a.values())
    assert len(orders_a["epl_3"]) == 3
    assert len(orders_a["fixed_delay_9"]) == 9
    assert len(orders_a["random_fixed"]) == len(GRID) == 187
    assert len(set(orders_a["uniform_fixed"])) == 187


def test_neural_field_prn_free_and_preobservation_sequential_trace():
    torch.manual_seed(9)
    model = SetNeuralField(NFConfig(context_features=False)).eval()
    values = np.ones(187, dtype=np.complex64)
    first_score, first = sequential_trace(model, values, 5, [0, 0], torch.device("cpu"), policy="active_adaptive")
    second_score, second = sequential_trace(model, values, 5, [0, 0], torch.device("cpu"), policy="active_adaptive")
    assert first_score == second_score and first == second
    assert all(row["pre_observation"] for row in first)
    assert len({row["index"] for row in first}) == 5
    names = {name for name, _ in model.named_parameters()}
    assert not any("prn" in name or "identity" in name for name in names)


def test_gaussian_variance_stability_and_variable_prn_permutation_pooling():
    target = torch.tensor([[[1.0, -2.0]]]); mean = torch.zeros_like(target)
    variance = torch.full_like(target, 1e-4)
    assert torch.isfinite(gaussian_nll(target, mean, variance))
    values = [1.0, 7.0, 3.0, 2.0, 5.0]
    for method in ("median", "topk_mean", "trimmed_mean", "soft_topk"):
        assert pool_scores(values, method) == pytest.approx(pool_scores(list(reversed(values)), method))
        assert np.isfinite(pool_scores(values[:4], method))
    with pytest.raises(ValueError): pool_scores(values[:3], "median")


def test_timeline_onsets_and_ds7_ds8_family_nonindependence():
    fs = 25_000_000
    assert attack_phase("ds3", 100 * fs, 101 * fs) == "strict_pre"
    assert attack_phase("ds3", int(118.9 * fs), int(119 * fs)) == "injection_takeover"
    assert attack_phase("ds7", 150 * fs, 151 * fs) == "established_pull_off"
    assert ATTACK_PHASES["ds4"]["strict_pre"][1] == 113.8
    assert PRIMARY_FAMILY["ds7"] == PRIMARY_FAMILY["ds8"] == "ds7_ds8"


def test_freeze_manifest_rejects_policy_or_threshold_drift(tmp_path):
    names = {"model.pt", "model_context.pt", "model_no_context.pt", "normal_field_reference.npz",
             "query_policy.json", "thresholds.json", "pooling.json", "calibration.json"}
    import hashlib
    files = {}
    for name in names:
        path = tmp_path / name; path.write_bytes(name.encode()); files[name] = hashlib.sha256(name.encode()).hexdigest()
    (tmp_path / "freeze_manifest.json").write_text(json.dumps({"files": files}))
    assert verify_freeze_manifest(tmp_path) == files
    (tmp_path / "query_policy.json").write_text("attack tuned")
    with pytest.raises(RuntimeError): verify_freeze_manifest(tmp_path)


def test_runner_has_explicit_freeze_before_attack_open_and_no_attack_label_policy_input():
    path = ROOT / "scripts/run_acaf_nf_stage1_r3_static_detection.py"
    tree = ast.parse(path.read_text())
    attack = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "attack_phase_run")
    calls = [ast.unparse(node) for node in ast.walk(attack) if isinstance(node, ast.Call)]
    assert calls.index("verify_freeze_manifest(output)") < min(i for i, call in enumerate(calls) if call.startswith("authenticate(config, scenario"))
    source = (ROOT / "src/gnss_doppler_lab/acaf_nf_stage1_r3.py").read_text()
    policy_tree = ast.parse(source)
    policy = next(node for node in policy_tree.body if isinstance(node, ast.FunctionDef) and node.name == "fixed_policy_orders")
    assert [arg.arg for arg in policy.args.args] == ["seed"]
    runner = path.read_text()
    assert '"model_context.pt"' in runner and '"model_no_context.pt"' in runner
    assert '"normal_field_reference.npz"' in runner


def test_config_preregisters_normal_only_threshold_and_go_gates():
    config = json.loads((ROOT / "configs/acaf_nf_stage1_r3_static_detection.json").read_text())
    assert config["threshold"]["source"] == "cleanStatic calibration only"
    assert config["clean_roles_seconds"]["prohibited_for_fit_selection_calibration"] == [100, None]
    assert config["scenario_family"]["ds7"] == config["scenario_family"]["ds8"]
    assert config["ds7_ds8_independent_confirmations"] is False
    assert set(config["core_exclusions"]) == {"cleanDynamic", "ds5", "ds6"}


def test_independent_verifier_does_not_import_producer():
    path = ROOT / "scripts/verify_acaf_nf_stage1_r3_static_detection.py"
    if not path.exists():
        pytest.skip("verifier added after core test during implementation")
    imports = [ast.unparse(node) for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any("run_acaf_nf_stage1_r3" in item for item in imports)
