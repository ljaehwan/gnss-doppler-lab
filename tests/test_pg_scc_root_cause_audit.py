from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "pg_scc_root_cause_audit", ROOT / "scripts/run_pg_scc_root_cause_audit.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_pg_scc_root_cause_audit",
        ROOT / "scripts/verify_pg_scc_root_cause_audit.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnostic_formulae_and_unavailable_zero_denominator():
    runner = load_runner()
    result = runner.diagnostic_scores(10.0, 6.0, 9)
    assert result["raw_rss_improvement"] == pytest.approx(4.0)
    assert result["improvement_per_k"] == pytest.approx(4.0 / 9.0)
    assert result["delta_bic"] == pytest.approx(8.0 - 4.0 * np.log(18.0))
    assert result["dof_corrected_likelihood_improvement"]["status"] == "AVAILABLE"
    assert result["dof_corrected_likelihood_improvement"]["value"] == pytest.approx(
        (4.0 / 4.0) / (6.0 / 12.0)
    )

    unavailable = runner.diagnostic_scores(5.0, 0.0, 3)
    dof = unavailable["dof_corrected_likelihood_improvement"]
    assert dof["status"] == "UNAVAILABLE"
    assert dof["value"] is None
    assert "denominator" in dof["reason"]


def test_nested_masks_exact_order_bounds_prompt_and_additions():
    runner = load_runner()
    masks = {
        "pg_scc_k3": [93, 61, 125],
        "pg_scc_k5": [93, 61, 125, 5, 181],
        "pg_scc_k9": [93, 61, 125, 5, 181, 40, 146, 44, 142],
    }
    result = runner.validate_nested_masks(masks)
    assert result["status"] == "PASS"
    assert result["ordered_additions"] == [93, 61, 125, 5, 181, 40, 146, 44, 142]
    with pytest.raises(RuntimeError, match="nested mask"):
        runner.validate_nested_masks({**masks, "pg_scc_k5": [93, 61, 61, 5, 181]})
    with pytest.raises(RuntimeError, match="nested mask"):
        runner.validate_nested_masks({**masks, "pg_scc_k3": [92, 61, 125]})


def _support_rows(method: str):
    return [
        {
            "scenario": "cleanStatic",
            "phase": "holdout",
            "second": 1,
            "time_s": 1.0,
            "prn": prn,
            "method": method,
            "score": float(prn),
        }
        for prn in (3, 7, 11, 19)
    ]


def test_common_exact_support_requires_unique_identical_keys_and_prns():
    runner = load_runner()
    rows = _support_rows("a") + _support_rows("b")
    report = runner.validate_common_support(rows, ["a", "b"], minimum_prns=4)
    assert report["status"] == "PASS"
    assert report["common_rows_per_detector"] == 4
    with pytest.raises(RuntimeError, match="duplicate"):
        runner.validate_common_support(rows + [dict(rows[0])], ["a", "b"], minimum_prns=4)
    changed = [dict(row) for row in rows]
    changed[-1]["time_s"] = 1.25
    with pytest.raises(RuntimeError, match="identical"):
        runner.validate_common_support(changed, ["a", "b"], minimum_prns=4)
    with pytest.raises(RuntimeError, match="PRN"):
        runner.validate_common_support(
            _support_rows("a")[:3] + _support_rows("b")[:3], ["a", "b"], minimum_prns=4
        )


def test_clean_synthetic_roles_are_separate_from_attack_report_only():
    runner = load_runner()
    assert runner.assert_allowed_role("selector", "synthetic_validation") == "PASS"
    assert runner.assert_allowed_role("covariance", "clean_train") == "PASS"
    assert runner.assert_allowed_role("threshold", "clean_calibration") == "PASS"
    assert runner.assert_allowed_role("attack_diagnostic", "attack_report_only") == "PASS"
    for path in ("selector", "covariance", "threshold"):
        with pytest.raises(RuntimeError, match="leakage"):
            runner.assert_allowed_role(path, "attack_report_only")
    assert runner.AUDIT_LABELS["attack"] == "POST_HOC_DIAGNOSTIC"
    assert runner.AUDIT_LABELS["k3"] == "EXPLORATORY_ONLY"


def test_random_masks_reproducible_exact_counts_and_prompt():
    runner = load_runner()
    first = runner.generate_random_masks(seed=2026080911, counts={3: 200, 5: 200, 9: 200})
    second = runner.generate_random_masks(seed=2026080911, counts={3: 200, 5: 200, 9: 200})
    assert first == second
    for budget in (3, 5, 9):
        assert len(first[budget]) == 200
        assert len({tuple(mask) for mask in first[budget]}) == 200
        assert all(len(mask) == len(set(mask)) == budget and 93 in mask for mask in first[budget])


def test_twenty_seed_stability_reproducible():
    runner = load_runner()
    seeds = list(range(20))

    def deterministic_trainer(seed: int, budget: int):
        return [93, *[((seed * 13 + offset) % 92) for offset in range(1, budget)]]

    first = runner.compute_seed_stability(seeds, (3, 5, 9), deterministic_trainer)
    second = runner.compute_seed_stability(seeds, (3, 5, 9), deterministic_trainer)
    assert first == second
    assert len(first["rows"]) == 60
    assert set(first["median_pairwise_jaccard"]) == {"3", "5", "9"}


def test_block_bootstrap_is_reproducible_and_resamples_whole_blocks():
    runner = load_runner()
    rows = [
        {"family": "ds3", "second": second, "value": float(second)} for second in range(30)
    ]
    first = runner.block_bootstrap(
        rows, value_key="value", block_seconds=10.0, iterations=25, seed=2026080921
    )
    second = runner.block_bootstrap(
        rows, value_key="value", block_seconds=10.0, iterations=25, seed=2026080921
    )
    assert first == second
    assert first["status"] == "PASS"
    assert first["block_count"] == 3
    assert first["iterations"] == 25
    assert len(first["replicate_means"]) == 25


def test_required_artifact_schema_and_manifest_verifier(tmp_path):
    runner = load_runner()
    verifier = load_verifier()
    assert set(runner.REQUIRED_ROOT_CAUSES) == {
        "SCORE_DILUTION",
        "NOISY_COORDINATE_ADDITION",
        "H1_NULL_OVERFIT",
        "DENSE_COVARIANCE_FAILURE",
        "SYNTHETIC_REAL_PHYSICS_MISMATCH",
        "SELECTOR_PROXY_OBJECTIVE_MISMATCH",
        "MASK_NON_IDENTIFIABILITY",
        "AWGN_CONTROL_MISSCALE",
        "CALIBRATION_TAIL_INSUFFICIENCY",
        "GENUINE_LACK_OF_SPARSE_GAIN",
    }
    for relative in runner.REQUIRED_ARTIFACTS:
        path = tmp_path / relative
        if relative == "plots":
            path.mkdir()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n" if path.suffix == ".json" else "x\n", encoding="utf-8")
    for name in ("config.json", "source_commit.json"):
        (tmp_path / name).write_bytes(
            (
                ROOT
                / "artifacts/pg_scc_stage0_r1_root_cause_audit"
                / name
            ).read_bytes()
        )
    runner.finalize_manifest(tmp_path)
    report = verifier.verify_tree(tmp_path, require_git=False)
    assert report["status"] == "PASS"
    (tmp_path / "README.md").write_text("drift\n", encoding="utf-8")
    assert verifier.verify_tree(tmp_path, require_git=False)["status"] == "FAIL"


def test_config_and_source_are_frozen_and_phase2_entrypoint_is_guarded():
    runner = load_runner()
    config = ROOT / "artifacts/pg_scc_stage0_r1_root_cause_audit/config.json"
    assert runner.sha256(config) == runner.CONFIG_SHA256
    source = json.loads(
        (ROOT / "artifacts/pg_scc_stage0_r1_root_cause_audit/source_commit.json").read_text()
    )
    assert source["preregistration_phase"] == "BEFORE_POST_HOC_ATTACK_ANALYSIS"
    text = (ROOT / "scripts/run_pg_scc_root_cause_audit.py").read_text()
    assert text.index("verify_implementation_freeze(") < text.index("load_protected_inputs(")
