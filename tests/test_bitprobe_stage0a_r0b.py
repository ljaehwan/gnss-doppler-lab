from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gnss_doppler_lab.bitprobe_stage0a_r0b import (
    ARTIFACT_REL, DATASETS, SOURCE_FILES, analysis_bytes, analyze,
    load_contract, load_observations, minimum_relaxation, threshold_grid,
    verify_sources,
)


REPO = Path(__file__).resolve().parents[1]


def test_source_metric_hashes_match_frozen_contract() -> None:
    observed = verify_sources(REPO)
    assert set(observed) == set(SOURCE_FILES)
    assert all(observed[name]["sha256"] == SOURCE_FILES[name][1] for name in SOURCE_FILES)


def test_observations_are_exact_committed_values() -> None:
    values = load_observations(REPO)
    assert values["TEXBAT.cleanStatic"]["reproducibility_median"] == pytest.approx(0.4422987386946777)
    assert values["TEXBAT.cleanStatic"]["permutation_p"] == pytest.approx(0.4666666666666667)
    assert values["TEXBAT.cleanStatic"]["moderate_nuisance_direction_fraction"] == pytest.approx(5 / 8)
    assert values["OAKBAT.cleanStatic"]["reproducibility_median"] == pytest.approx(0.7246080377972713)
    assert values["OAKBAT.cleanStatic"]["permutation_p"] == pytest.approx(0.3)
    assert values["OAKBAT.cleanStatic"]["moderate_nuisance_direction_fraction"] == pytest.approx(7 / 8)


def test_all_frozen_tiers_fail_for_both_datasets() -> None:
    result = analyze(REPO)
    for tier in ("ORIGINAL", "MODERATE_RELAXATION", "LENIENT_RELAXATION"):
        assert all(result["tier_results"][tier][dataset]["pass"] is False for dataset in DATASETS)
    assert result["verdict"] == "NO_ROBUST_SIGNAL_UNDER_RELAXED_GATES"
    assert result["moderate_passing_datasets"] == []
    assert result["lenient_only"] == []


def test_flip_specificity_is_advisory_only() -> None:
    result = analyze(REPO)
    for tier in result["tier_results"].values():
        for dataset in DATASETS:
            assert tier[dataset]["flip_specificity_excluded"] is True
            assert all(row["criterion"] != "flip_specificity" for row in tier[dataset]["criteria"])


def test_threshold_grid_has_exact_cartesian_cardinality() -> None:
    contract = load_contract(REPO)
    rows = threshold_grid(load_observations(REPO), contract)
    assert len(rows) == 2 * 4 * 4 * 6 * 6 * 4 == 4608
    assert all(row["full_grid_pass"] is False for row in rows)


def test_minimum_relaxation_is_inverted_from_observations() -> None:
    value = minimum_relaxation(load_observations(REPO), load_contract(REPO))
    tex, oak = value["TEXBAT.cleanStatic"], value["OAKBAT.cleanStatic"]
    assert tex["required_max_reproducibility_threshold"] == 0.4
    assert tex["required_max_lower_bound_threshold"] == 0.2
    assert tex["required_max_effect_threshold"] == 0.0
    assert tex["required_min_p_threshold"] == 0.5
    assert tex["required_min_nuisance_direction_fraction"] == 0.6
    assert oak["required_max_reproducibility_threshold"] == 0.7
    assert oak["required_max_lower_bound_threshold"] == 0.4
    assert oak["required_max_effect_threshold"] == 0.03
    assert oak["required_min_p_threshold"] == 0.3
    assert oak["required_min_nuisance_direction_fraction"] == 0.75
    assert tex["fixed_separability_lower_gt_zero"] is False
    assert oak["fixed_separability_lower_gt_zero"] is False


def test_analysis_is_byte_deterministic_and_lopo_is_unavailable() -> None:
    first, second = analyze(REPO), analyze(REPO)
    assert analysis_bytes(first) == analysis_bytes(second)
    assert first["leave_one_prn_out"]["status"] == "NOT_AVAILABLE_FROM_COMMITTED_METRICS"
    assert first["leave_one_prn_out"]["tensor_access_attempted"] is False


def test_verifier_help_needs_no_pythonpath() -> None:
    environment = dict(os.environ); environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "scripts/verify_bitprobe_stage0a_r0b.py", "--help"],
        cwd=REPO, env=environment, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0
    assert "--self-test" in completed.stdout
