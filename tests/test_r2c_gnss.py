from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.r2c_gnss import (
    C_M_S, AnalyticResidualWhitener, ComplexTapProvenance, SmallNeuralNuisanceModel,
    SourceSupport, aggregate_a2, assign_attack_phase, assign_normal_split, availability_time,
    fit_second_source, fit_shared_constellation, full_score, inject_second_source,
    quantile_threshold, strict_alarm, sustained_alarms, validate_complex_taps,
)

ROOT = Path(__file__).resolve().parents[1]
TAPS = np.linspace(-1, 1, 9)
GRID = np.linspace(-1, 1, 41)


def provenance(complex_value=True):
    return ComplexTapProvenance("source.bin", "a" * 64, "receiver-v1", "extractor-v1", 25e6,
                                .25, "cleanStatic", complex_value, "none")


def test_complex_provenance_accepts_complex_and_rejects_magnitude():
    values = np.ones((3, 9), dtype=complex) + 1j
    assert validate_complex_taps(values, provenance()).shape == (3, 9)
    with pytest.raises(ValueError, match="genuine complex"):
        validate_complex_taps(np.abs(values), provenance(False))
    with pytest.raises(ValueError, match="provenance"):
        validate_complex_taps(values, ComplexTapProvenance("", "a" * 64, "r", "e", 1, .25, "x", True, "none"))


@pytest.mark.parametrize(("start", "end", "expected"), [
    (0, 300, "normal_train"), (299, 301, "excluded_guard_or_boundary"),
    (320, 400, "normal_calibration"), (319, 350, "excluded_guard_or_boundary"),
    (420, 421, "normal_holdout"), (399, 421, "excluded_guard_or_boundary"),
])
def test_chronological_source_support_split(start, end, expected):
    assert assign_normal_split(SourceSupport(start, end, "cleanStatic")) == expected


def test_attack_phase_and_availability_are_causal():
    assert assign_attack_phase(SourceSupport(30, 80, "DS3"), 100) == "stable_pre"
    assert assign_attack_phase(SourceSupport(79, 81, "DS3"), 100) == "transition_excluded"
    assert assign_attack_phase(SourceSupport(100, 101, "DS3"), 100) == "post"
    assert assign_attack_phase(SourceSupport(140, 141, "DS3"), 100) == "persistent"
    assert availability_time([SourceSupport(10, 11, "x"), SourceSupport(10.2, 11.4, "x")]) == 11.4


def synthetic_two_source(delay=-.4):
    authentic = np.exp(.3j) * np.maximum(1 - np.abs(TAPS - .1), 0)
    return inject_second_source(authentic, TAPS, delay, .4, -1.1)


def test_h0_h1_glrt_and_signed_delay_support():
    for delay in (-.45, .45):
        fit = fit_second_source(synthetic_two_source(delay), TAPS, GRID)
        assert fit.score > 20
        assert fit.h1.weighted_rss < fit.h0.weighted_rss
        assert fit.h1.identifiable
        assert min(abs(np.asarray(fit.h1.delays_chips) - delay)) <= .06
        assert fit.h0.residual.shape == (9,)


def test_delay_grid_rejects_extrapolation_and_degenerate_pairs():
    with pytest.raises(ValueError, match="inside measured"):
        fit_second_source(synthetic_two_source(), TAPS, np.linspace(-1.1, 1, 20))
    with pytest.raises(ValueError, match="no identifiable"):
        fit_second_source(synthetic_two_source(), TAPS, [-.01, .01], minimum_separation_chips=.1)


def test_global_phase_and_positive_gain_invariance():
    y = synthetic_two_source(); reference = fit_second_source(y, TAPS, GRID).score
    for phase in (0, .5, 1.7, np.pi):
        assert fit_second_source(y * np.exp(1j * phase), TAPS, GRID).score == pytest.approx(reference, rel=1e-10, abs=1e-8)
    for gain in (.5, .75, 1, 1.5, 2):
        assert fit_second_source(y * gain, TAPS, GRID).score == pytest.approx(reference, rel=1e-10, abs=1e-8)


def geometry_case():
    los = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-.6, -.5, -.6245], [.5, -.7, .5099]])
    los /= np.linalg.norm(los, axis=1, keepdims=True)
    beta = np.asarray([20, -15, 7, 80.])
    ranges = np.column_stack([-los, np.ones(len(los))]) @ beta
    return ranges / C_M_S, los, beta


def test_shared_geometry_consistency_permutation_and_variable_count():
    delays, los, beta = geometry_case()
    fit = fit_shared_constellation(delays, los)
    assert fit.valid and fit.rank == 4 and np.max(np.abs(fit.residual_m)) < 1e-8
    assert fit.beta_m == pytest.approx(beta)
    order = [3, 0, 4, 1, 2]
    permuted = fit_shared_constellation(delays[order], los[order])
    assert permuted.beta_m == pytest.approx(fit.beta_m)
    assert full_score(np.arange(1, 6), permuted) == pytest.approx(full_score(np.arange(1, 6)[order], fit))
    assert fit_shared_constellation(delays[:4], los[:4]).valid


def test_geometry_rank_and_minimum_prn_fail_closed():
    delays = np.arange(3) * 1e-8; los = np.eye(3)
    fit = fit_shared_constellation(delays, los)
    assert not fit.valid and fit.reason == "insufficient_prns" and full_score([1, 2, 3], fit) == 0
    bad = fit_shared_constellation(np.arange(4) * 1e-8, np.tile([1., 0, 0], (4, 1)))
    assert not bad.valid and bad.reason == "rank_deficient"


def test_relation_destruction_reduces_shared_score():
    delays, los, _ = geometry_case(); scores = np.full(5, 10.)
    coherent = fit_shared_constellation(delays, los)
    destroyed = fit_shared_constellation(delays[[2, 4, 0, 3, 1]], los)
    assert full_score(scores, destroyed) < full_score(scores, coherent)


def test_nuisance_and_threshold_roles_are_normal_only():
    rng = np.random.default_rng(2); residuals = rng.normal(size=(12, 9)) + 1j * rng.normal(size=(12, 9))
    analytic = AnalyticResidualWhitener().fit(residuals, ["normal_train"] * 12)
    assert analytic.covariance_.shape == (9, 9)
    neural = SmallNeuralNuisanceModel(hidden=3).fit(rng.normal(size=(12, 2)), residuals,
                                                        ["normal_train"] * 12, epochs=2)
    mean, variance = neural.predict(rng.normal(size=(2, 2)))
    assert mean.shape == variance.shape == (2, 9)
    for role in ("post", "external_normal", "normal_calibration"):
        with pytest.raises(ValueError, match="normal_train"):
            AnalyticResidualWhitener().fit(residuals, [role] * 12)
    assert quantile_threshold([1, 2, 3], .99, ["normal_calibration"] * 3) == 3
    with pytest.raises(ValueError, match="normal_calibration"):
        quantile_threshold([1], .99, ["post"])
    assert not strict_alarm(3, 3) and strict_alarm(3.01, 3)


def test_aggregation_permutation_and_sustained_boundary_reset():
    assert aggregate_a2([1, 5, 2, 4], 2) == aggregate_a2([4, 2, 5, 1], 2) == 4.5
    alarms = sustained_alarms([1, 1, 1, 1, 1, 1], ["a", "a", "a", "a", "b", "b"],
                               [0, .5, 1, 2, 2.5, 3], ["pre"] * 6)
    assert alarms.tolist() == [False, False, True, False, False, False]


def test_synthetic_noise_multipath_and_injection_controls():
    rng = np.random.default_rng(4); authentic = np.maximum(1 - np.abs(TAPS), 0).astype(complex)
    noisy = authentic + .01 * (rng.normal(size=9) + 1j * rng.normal(size=9))
    injected = inject_second_source(noisy, TAPS, -.5, .5, .9)
    assert fit_second_source(injected, TAPS, GRID).score > fit_second_source(noisy, TAPS, GRID).score
    delays, los, _ = geometry_case()
    independent = np.asarray([-.7, .55, -.25, .8, .32]) / 1_023_000
    assert full_score(np.ones(5), fit_shared_constellation(independent, los)) < full_score(np.ones(5), fit_shared_constellation(delays, los))


def test_config_has_no_absolute_paths_and_portable_entrypoints():
    config = json.loads((ROOT / "configs/r2c_gnss_stage0.json").read_text())
    assert not any(str(value).startswith("/") for value in config.get("input_paths", {}).values())
    for script in ("run_r2c_gnss_stage0.py", "verify_r2c_gnss_stage0.py"):
        text = (ROOT / "scripts" / script).read_text()
        assert text.startswith("#!/usr/bin/env python3")
        assert "/home/" not in text


def test_artifact_manifest_shape_after_generation():
    root = ROOT / "artifacts/r2c_gnss_stage0"
    required = {"README.md", "config.json", "provenance.json", "input_validity.json", "training_summary.json",
                "thresholds.json", "scenario_metrics.csv", "ablation_metrics.csv", "per_epoch_scores.csv",
                "gain_invariance.json", "phase_invariance.json", "noise_control.json", "multipath_control.json",
                "second_source_injection.json", "relation_destruction.json", "decision.json", "verification.json", "hashes.json"}
    assert required <= {path.name for path in root.iterdir() if path.is_file()}
    assert (root / "plots/relation_control_source.csv").is_file()
    assert json.loads((root / "decision.json").read_text())["verdict"] == "DATA_INVALID"
