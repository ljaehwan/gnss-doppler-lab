from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "pg_scc_root_cause_audit_r2_repair",
        ROOT / "scripts/run_pg_scc_root_cause_audit.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mixed_dilution_schema_generates_all_plots_without_value_mutation(tmp_path):
    runner = load_runner()
    runner.OUTPUT = tmp_path
    nested = [
        {"stage": stage, "delay_chips": float(index), "doppler_hz": float(index * 10)}
        for index, stage in enumerate(("K3", "K5_new", "K9_new"), 1)
    ]
    contributions = [{
        "group": "ds3", "clean_residual_variance": 1.0,
        "mean_glrt_improvement_contribution": 2.0,
    }]
    dilution = []
    for group in ("clean_train", "clean_holdout", "synthetic_h1"):
        for budget in (3, 5, 9):
            dilution.append({
                "budget": budget, "group": group,
                "mean_improvement_per_k": budget / 10.0,
            })
    dilution.extend([
        {
            "budget": 5, "group": "clean_train", "row_type": "PAIRED_CHANGE",
            "mean_improvement_per_k_change": 0.125,
        },
        {
            "budget": 9, "group": "clean_holdout", "row_type": "PAIRED_CHANGE",
            "mean_improvement_per_k_change": -0.25,
        },
    ])
    before = json.dumps(dilution, sort_keys=True)
    mismatch = [{
        "label": "POST_HOC_DIAGNOSTIC", "best_delay_chips": 0.25,
        "best_doppler_hz": 50.0,
    }]
    selector = {"learned_evidence": {str(k): {"random_percentile": 0.9} for k in (3, 5, 9)}}
    seed_rows = [
        {"budget": budget, "seed": 1, "median_pairwise_jaccard": 0.75}
        for budget in (3, 5, 9)
    ]
    bundles = {
        name: {"macro_pauc": value}
        for name, value in zip(
            ("pg_scc_k3", "pg_scc_k5", "pg_scc_k9", "fixed9", "dense_two_source_glrt"),
            (0.1, 0.2, 0.3, 0.4, 0.5),
        )
    }
    awgn = {"empirical_controls": [{"multiplier": 1.0, "false_response": 0.01}]}
    calibration = {"methods": {
        "pg_scc_k9": {"bootstrap_threshold_interval": [0.1, 0.2]},
        "fixed9": {"bootstrap_threshold_interval": [0.2, 0.3]},
    }}
    runner._save_plots(
        nested, contributions, dilution, mismatch, selector, seed_rows, [],
        bundles, awgn, calibration,
    )
    assert json.dumps(dilution, sort_keys=True) == before
    assert dilution == copy.deepcopy(dilution)
    assert {path.name for path in (tmp_path / "plots").glob("*.png")} == set(runner.REQUIRED_PLOTS)
    assert all((tmp_path / "plots" / name).stat().st_size > 0 for name in runner.REQUIRED_PLOTS)


def test_cycle_namespace_preserves_exact_plot_repair():
    runner = load_runner()
    assert runner.OUTPUT == ROOT / "artifacts/pg_scc_stage0_r2_r1_identity_repair_cycle2"
    assert runner.PREREGISTRATION_SHA == "a4603a801328666cdf70784154132e213d6d25f6"
    source = (ROOT / "scripts/run_pg_scc_root_cause_audit.py").read_text(encoding="utf-8")
    assert 'if row["group"] == group and "mean_improvement_per_k" in row' in source
