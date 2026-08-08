from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from gnss_doppler_lab.pg_scc import exact_binomial_ci, pool, select_pooling, verify_manifest
from gnss_doppler_lab.pg_scc_physics import CENTER
from gnss_doppler_lab.pg_scc_selector import _combo_split, symmetric_mask_from_logits, train_global_topk_mask

ROOT = Path(__file__).resolve().parents[1]


def test_exact_topk_global_fixed_mask_and_no_prn_identity():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(64, 187))
    teacher = np.linspace(0, 1, 64)
    labels = np.r_[np.zeros(32), np.ones(32)]
    for budget in (3, 5, 9):
        first, summary = train_global_topk_mask(features, teacher, labels, budget, seed=8, epochs=5)
        second, _ = train_global_topk_mask(features, teacher, labels, budget, seed=8, epochs=5)
        assert first == second and len(first) == len(set(first)) == budget and first[0] == CENTER
        assert summary["objective"]["exact_k"].startswith("K*softmax")
    source = (ROOT / "src/gnss_doppler_lab/pg_scc_selector.py").read_text()
    tree = ast.parse(source)
    parameters = [arg.arg for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) for arg in node.args.args]
    assert "prn" not in parameters and "prn_identity" not in source


def test_symmetric_mask_and_synthetic_parameter_split_are_exact():
    logits = np.linspace(-1, 1, 187)
    for budget in (3, 5, 9):
        mask = symmetric_mask_from_logits(logits, budget)
        assert len(mask) == len(set(mask)) == budget and CENTER in mask
    combos = [(-.75, -150, 0, .25, 0), (.75, 150, 1, 1.5, .05)]
    assert all(_combo_split(combo) in {"train", "validation"} for combo in combos)


def test_permutation_invariant_pooling_and_exact_fpr_ci():
    values = [8.0, 1.0, 5.0, 2.0, 3.0]
    for method in ("median", "robust_mean", "topk_mean"):
        assert pool(values, method) == pool(list(reversed(values)), method)
    method, diagnostics = select_pooling([values, [1, 2, 3, 4]], [[8, 9, 10, 11], [7, 8, 9, 12]])
    assert method in diagnostics
    assert exact_binomial_ci(0, 20)[0] == 0.0


def test_freeze_training_script_is_clean_only_and_orders_evaluation_guard():
    train = (ROOT / "scripts/train_pg_scc_selector.py").read_text()
    assert "attack_features.npz" not in train and "attack_features.json" not in train
    assert '"attack_iq_bytes_read_before_freeze": 0' in train
    evaluation = ROOT / "scripts/eval_pg_scc_static.py"
    if evaluation.exists():
        tree = ast.parse(evaluation.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [ast.unparse(node) for node in ast.walk(main) if isinstance(node, ast.Call)]
        verify = min(index for index, call in enumerate(calls) if "verify_freeze" in call)
        load_attack = min(index for index, call in enumerate(calls) if "load_feature_cache" in call)
        assert verify < load_attack


def test_artifact_checksum_verifier(tmp_path):
    payload = tmp_path / "x.txt"; payload.write_text("physical")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (tmp_path / "artifact_manifest_sha256.json").write_text(json.dumps({"x.txt": digest}))
    assert verify_manifest(tmp_path) == []
    payload.write_text("drift")
    assert verify_manifest(tmp_path) == ["x.txt"]


def test_normal_only_threshold_and_timeline_preregistration():
    config = json.loads((ROOT / "configs/pg_scc_stage0_static_k9.json").read_text())
    assert config["threshold"]["source"] == "cleanStatic calibration event pooling only"
    assert config["timeline_seconds"]["ds3"]["onset"] == 118.9
    assert config["timeline_seconds"]["ds4"]["truncated_at_approximately"] < 130
    assert config["scenario_family"]["ds7"] == config["scenario_family"]["ds8"]
    assert config["gates"]["real_attack_label_used_for_design"] is False
