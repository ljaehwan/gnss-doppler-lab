import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
import gnss_doppler_lab.nc_topi as n


def W():
    return np.diag(np.linspace(1.0, 2.0, 9))


def peak():
    return np.exp(-4 * n.CANONICAL_TAP_COORDS**2)


def pair(identity=("rec", "G01", 7), delta=None):
    predicted = peak()
    residual = np.linspace(-0.2, 0.2, 9) if delta is None else np.asarray(delta, float)
    std = np.linspace(0.5, 1.3, 9)
    return n.PeakPredictionPair(
        actual_raw=predicted + residual,
        predicted_raw=predicted,
        residual_standardized=residual / std,
        standardizer_std=std,
        identity=identity,
        actual_space="prompt_relative_ratio_raw",
        predicted_space="prompt_relative_ratio_raw",
        residual_space="b0_standardized",
    )


# Blocker 1: frozen score coordinate and energy/conditioner contract.
def test_peak_prediction_pair_requires_both_raw_peaks_and_validates_coordinates():
    p = pair()
    assert np.allclose(p.residual_raw, p.actual_raw - p.predicted_raw)
    with pytest.raises((TypeError, ValueError), match="actual|predicted|residual-only"):
        n.PeakPredictionPair(
            residual_standardized=np.zeros(9), standardizer_std=np.ones(9),
            identity=("r", "G1", 1), actual_space="prompt_relative_ratio_raw",
            predicted_space="prompt_relative_ratio_raw", residual_space="b0_standardized")
    with pytest.raises(ValueError, match="standardized"):
        n.PeakPredictionPair(
            actual_raw=np.ones(9), predicted_raw=np.zeros(9),
            residual_standardized=np.zeros(9), standardizer_std=np.ones(9),
            identity=("r", "G1", 1), actual_space="prompt_relative_ratio_raw",
            predicted_space="prompt_relative_ratio_raw", residual_space="b0_standardized")
    with pytest.raises(ValueError, match="space"):
        n.PeakPredictionPair(
            actual_raw=np.ones(9), predicted_raw=np.zeros(9),
            residual_standardized=np.ones(9), standardizer_std=np.ones(9),
            identity=("r", "G1", 1), actual_space="standardized",
            predicted_space="prompt_relative_ratio_raw", residual_space="b0_standardized")


def test_score_bundle_uses_raw_quadratic_energy_b0_and_scale_not_scale_squared():
    p = pair(delta=np.arange(9) / 10)
    basis = n.primary_tangent_basis(p.predicted_raw, n.CANONICAL_TAP_COORDS, W=W())
    class Fixed:
        def conditioner_transform(self, X): return np.asarray(X, float) + 10
        def predict_scale(self, X): return np.full(len(X), 4.0)
    out = n.produce_nc_topi_scores(p, basis.matrix, W(), conditioner=Fixed(), iq_features=[[1, 2]])
    expected_b0 = np.sqrt(np.mean(p.residual_standardized**2))
    assert out.b0 == pytest.approx(expected_b0)
    assert out.topi == pytest.approx(out.projection.r_perp @ W() @ out.projection.r_perp)
    assert out.nc_topi == pytest.approx(out.topi / 4.0)
    assert out.conditioner_transform.tolist() == [[11.0, 12.0]]
    assert out.total == pytest.approx(out.tangent + out.perp + out.cross)


def test_conditioner_fits_log_energy_and_caps_exp_scale_at_clean_q995():
    X = np.arange(80.0).reshape(40, 2)
    energy = np.exp(0.02 * X[:, 0] + 0.2)
    c = n.RobustConditioner().fit(
        X[:30], energy[:30], roles=["normal_train"] * 30,
        feature_names=["log_power", "flatness"])
    assert c.fit_manifest_["target"] == "log(max(S_perp, energy_epsilon))"
    uncapped = np.exp(c.predict_log_energy(X[30:]))
    cap = c.calibrate_cap(X[30:], roles=["normal_calibration"] * 10)
    assert cap == pytest.approx(np.quantile(uncapped, .995, method="higher"))
    assert np.all(c.predict_scale(X) <= cap)
    with pytest.raises(ValueError, match="frozen at q995"):
        c.calibrate_cap(X[30:], roles=["normal_calibration"] * 10, q=.99)


# Blocker 2: width cannot leak into the primary tangent basis.
def test_primary_basis_defaults_to_amplitude_shift_and_rejects_width():
    b = n.primary_tangent_basis(peak(), n.CANONICAL_TAP_COORDS, W=W())
    assert b.names == ("amplitude", "shift")
    assert "width" not in b.names
    with pytest.raises(ValueError, match="width.*diagnostic"):
        n.normalize_tangents(peak(), n.CANONICAL_TAP_COORDS, W=W(), include_width=True)
    width = n.build_width_ablation_basis(peak(), n.CANONICAL_TAP_COORDS, W=W())
    assert width.names == ("amplitude", "shift", "width")


# Blocker 3: covariance provenance is complete and fit is train-only.
def test_covariance_requires_exact_role_scenario_identity_provenance_and_audits_digest():
    r = np.arange(270.0).reshape(30, 9)
    ids = [("cleanStatic", "G01", i) for i in range(30)]
    c = n.fit_shrinkage_covariance(
        r, fit_roles=["normal_train"] * 30,
        scenarios=["cleanStatic"] * 30, identities=ids)
    assert c.audit["identity_count"] == 30
    assert len(c.audit["identity_digest_sha256"]) == 64
    for kw in ({}, {"fit_roles": ["normal_train"] * 30},
               {"fit_roles": ["normal_train"] * 30, "scenarios": ["cleanStatic"] * 30}):
        with pytest.raises((TypeError, ValueError), match="mandatory|identit|scenario|role"):
            n.fit_shrinkage_covariance(r, **kw)
    with pytest.raises(ValueError, match="normal_train"):
        n.fit_shrinkage_covariance(r, fit_roles=["normal_calibration"] * 30,
                                   scenarios=["cleanStatic"] * 30, identities=ids)
    with pytest.raises(ValueError, match="cleanStatic"):
        n.fit_shrinkage_covariance(r, fit_roles=["normal_train"] * 29 + ["normal_holdout"],
                                   scenarios=["cleanStatic"] * 29 + ["DS1"], identities=ids)
    with pytest.raises(ValueError, match="unique"):
        n.fit_shrinkage_covariance(r, fit_roles=["normal_train"] * 30,
                                   scenarios=["cleanStatic"] * 30, identities=[ids[0]] * 30)


# Blocker 4: exact W geometry, decomposition and diagnostics.
def test_primary_projection_is_unregularized_w_orthogonal_and_decomposes():
    w = W(); J = n.primary_tangent_basis(peak(), n.CANONICAL_TAP_COORDS, W=w).matrix
    r = np.linspace(-1, 1, 9)
    q = n.weighted_project(r, J, w)
    assert q.projection_kind == "orthogonal_pinv"
    assert q.ridge == 0
    assert np.linalg.norm(J.T @ w @ q.r_perp) <= 1e-9
    assert q.orthogonality_defect <= 1e-9
    assert q.total_energy == pytest.approx(q.tangent_energy + q.perp_energy + q.cross_energy)


def test_projection_rank_deficiency_bad_w_and_explicit_ridge_diagnostic():
    J = np.column_stack(([1., 0, 0, 0], [1., 0, 0, 0], [0., 1, 0, 0]))
    q = n.weighted_project([1, 2, 3, 4], J, np.eye(4))
    assert q.rank == 2 and np.isfinite(q.coefficients).all()
    ridge = n.weighted_project_ridge_diagnostic([1, 2, 3, 4], J, np.eye(4), lambda_relative=1e-3)
    assert ridge.projection_kind == "ridge_diagnostic_not_orthogonal" and ridge.ridge > 0
    bad_asym = np.eye(4); bad_asym[0, 1] = .1
    with pytest.raises(ValueError, match="symmetric"):
        n.weighted_project(np.ones(4), J, bad_asym)
    bad_psd = np.eye(4); bad_psd[0, 0] = -1
    with pytest.raises(ValueError, match="positive semidefinite"):
        n.weighted_project(np.ones(4), J, bad_psd)


# Blocker 5: primary uncertainty is paired stratified pAUC-delta block bootstrap.
def test_paired_pauc_delta_bootstrap_is_gap_recording_safe_and_uses_all_rows_point():
    # Two 10 s blocks per class, spread over recordings. One extra row is point-only.
    labels = np.r_[np.zeros(40, int), np.ones(40, int), [1]]
    times = np.r_[np.arange(20) * .5, np.arange(20) * .5,
                  20 + np.arange(20) * .5, 20 + np.arange(20) * .5, [99.0]]
    rec = np.array(["n1"] * 20 + ["n2"] * 20 + ["p1"] * 20 + ["p2"] * 20 + ["p3"])
    a = labels + np.linspace(0, .01, len(labels)); b = .5 * labels
    out = n.paired_pauc_delta_block_bootstrap(
        labels, a, b, rec, times, max_fpr=.05, reps=30, seed=3)
    assert out.available and out.valid_reps == 30
    assert out.complete_block_count == 4 and out.audit["iid_fallback"] is False
    assert out.audit["point_estimate_rows"] == 81
    assert out.point_estimate == pytest.approx(
        n.standardized_pauc(labels, a) - n.standardized_pauc(labels, b))


def test_pauc_bootstrap_unavailable_for_class_deficiency_or_too_few_blocks():
    base = dict(score_a=np.ones(20), score_b=np.zeros(20),
                recording_ids=["r"] * 20, times=np.arange(20) * .5, reps=10)
    one = n.paired_pauc_delta_block_bootstrap(labels=np.zeros(20), **base)
    assert not one.available and "class" in one.reason
    # Both classes exist but only one complete block per stratum.
    two = n.paired_pauc_delta_block_bootstrap(
        np.r_[np.zeros(20), np.ones(20)], np.r_[np.zeros(20), np.ones(20)], np.zeros(40),
        ["n"] * 20 + ["p"] * 20, np.r_[np.arange(20) * .5, np.arange(20) * .5], reps=10)
    assert not two.available and "too few" in two.reason and two.valid_reps == 0


# Blocker 6: alarm state is scoped by recording, gap and post eligibility.
def test_sustained_alarm_requires_scope_and_resets_recording_gap_transition():
    with pytest.raises(TypeError):
        n.sustained_alarm_delay([100, 100.5, 101], [1, 1, 1], onset=100)
    r = n.sustained_alarm_delay(
        [99, 99.5, 100, 100.5, 101, 101.5, 102, 102.5], [1] * 8,
        recording_ids=["a", "a", "a", "a", "b", "b", "b", "b"],
        post_eligible_mask=[0, 0, 0, 1, 1, 1, 1, 1], onset=100,
        stable_pre_mask=[1, 1, 0, 0, 0, 0, 0, 0])
    assert r.already_alarming_stable_pre and r.alarm_time == 102 and r.delay == 2
    gap = n.sustained_alarm_delay(
        [100, 100.5, 101.5, 102, 102.5], [1] * 5, recording_ids=["a"] * 5,
        post_eligible_mask=[1] * 5, onset=100)
    assert gap.alarm_time == 102.5


# Blocker 7: IQ as-of grouping is mandatory and audited.
def test_causal_iq_groups_exact_unique_sorted_and_cadence_audited():
    with pytest.raises(TypeError):
        n.build_causal_iq_context([2], [.5, 1, 1.5, 2], np.ones((4, 2)), history=4)
    out = n.build_causal_iq_context(
        [2, 2], [.5, 1, 1.5, 2, .5, 1, 1.5, 2], np.arange(16.).reshape(8, 2),
        history=4, target_groups=["a", "b"], block_groups=["a"] * 4 + ["b"] * 4,
        cadence=.5)
    assert out.valid.tolist() == [1, 1] and out.audit["cadence_ok"]
    with pytest.raises(ValueError, match="exactly match"):
        n.build_causal_iq_context([2], [.5, 1], np.ones((2, 1)), history=1,
                                  target_groups=["a"], block_groups=["a", "b"])
    with pytest.raises(ValueError, match="duplicate"):
        n.build_causal_iq_context([2], [1, 1], np.ones((2, 1)), history=1,
                                  target_groups=["a"], block_groups=["a", "a"])
    with pytest.raises(ValueError, match="sorted"):
        n.build_causal_iq_context([2], [1.5, 1], np.ones((2, 1)), history=1,
                                  target_groups=["a"], block_groups=["a", "a"])


# Blocker 8: primary joins cannot silently intersect or mismatch epoch metadata.
def test_primary_epoch_join_requires_full_identity_and_metadata_equality():
    ids = [("r", "G1", 1), ("r", "G2", 1)]
    got, a, b = n.exact_primary_epoch_join(
        ids, [1, 2], ids[::-1], [20, 10],
        source_intervals_a=[(0, 1), (0, 1)], source_intervals_b=[(0, 1), (0, 1)],
        labels_a=[0, 1], labels_b=[1, 0], valid_mask_a=[1, 1], valid_mask_b=[1, 1])
    assert got == ids and a.tolist() == [1, 2] and b.tolist() == [10, 20]
    with pytest.raises(ValueError, match="full identity set"):
        n.exact_primary_epoch_join(ids, [1, 2], ids[:1], [1])
    with pytest.raises(ValueError, match="duplicate"):
        n.exact_primary_epoch_join([ids[0], ids[0]], [1, 2], ids, [1, 2])
    with pytest.raises(ValueError, match="label"):
        n.exact_primary_epoch_join(ids, [1, 2], ids, [1, 2], labels_a=[0, 1], labels_b=[0, 0])
    diag = n.common_epoch_intersection_diagnostic(ids, [1, 2], ids[:1], [3])
    assert diag.audit["excluded_from_a"] == 1 and diag.identities == (ids[0],)


# Blocker 9: executable frozen decision grammar and boundaries.
def decision_inputs(**overrides):
    values = dict(
        clean_nc_fpr=.02, clean_b0_fpr=.01,
        stable_pre_fpr={s: .049 for s in n.ATTACK_SCENARIOS},
        pauc_delta={s: .1 for s in n.ATTACK_SCENARIOS},
        nc_delay={s: 1.0 for s in n.ATTACK_SCENARIOS},
        b0_delay={s: 2.0 for s in n.ATTACK_SCENARIOS},
        pauc_ci_lower={s: .01 for s in n.ATTACK_SCENARIOS},
        equal_rmse_pass=True, second_peak_pass=True,
        actual_nc_mean_pauc_gain=.01, shuffled_nc_mean_pauc_gain=0.0)
    values.update(overrides)
    return values


def test_decision_go_boundaries_and_strict_operators():
    d = n.evaluate_stage0_decision(**decision_inputs())
    assert d.status == "GO" and all(d.criteria.values())
    # Stable pre is strict < .05; two failures do not trigger NO-GO, but block GO.
    f = {s: .049 for s in n.ATTACK_SCENARIOS}; f["DS1"] = f["DS2"] = .05
    assert n.evaluate_stage0_decision(**decision_inputs(stable_pre_fpr=f)).status == "INCONCLUSIVE"
    f["DS3"] = .05
    assert n.evaluate_stage0_decision(**decision_inputs(stable_pre_fpr=f)).status == "NO-GO"
    assert n.evaluate_stage0_decision(**decision_inputs(clean_nc_fpr=np.nextafter(.05, 1))).status == "NO-GO"
    assert n.evaluate_stage0_decision(**decision_inputs(clean_nc_fpr=.05)).status == "INCONCLUSIVE"


def test_decision_counts_delay_finiteness_missing_and_no_go_truth_table():
    zero = {s: 0.0 for s in n.ATTACK_SCENARIOS}
    nc = {s: 1.5 for s in n.ATTACK_SCENARIOS}; b0 = {s: 2.0 for s in n.ATTACK_SCENARIOS}
    assert n.evaluate_stage0_decision(**decision_inputs(pauc_delta=zero, nc_delay=nc, b0_delay=b0)).status == "GO"
    low = dict(zero); low["DS1"] = low["DS2"] = .1
    inf = {s: np.inf for s in n.ATTACK_SCENARIOS}
    assert n.evaluate_stage0_decision(**decision_inputs(pauc_delta=low, nc_delay=inf, b0_delay=inf)).status == "INCONCLUSIVE"
    low["DS2"] = 0
    assert n.evaluate_stage0_decision(**decision_inputs(pauc_delta=low, nc_delay=inf, b0_delay=inf)).status == "NO-GO"
    missing_ci = {s: None for s in n.ATTACK_SCENARIOS}
    assert n.evaluate_stage0_decision(**decision_inputs(pauc_ci_lower=missing_ci)).status == "INCONCLUSIVE"
    no_ci = {s: 0.0 for s in n.ATTACK_SCENARIOS}
    assert n.evaluate_stage0_decision(**decision_inputs(pauc_ci_lower=no_ci)).status == "NO-GO"
    assert n.evaluate_stage0_decision(**decision_inputs(equal_rmse_pass=False)).status == "NO-GO"
    assert n.evaluate_stage0_decision(**decision_inputs(second_peak_pass=False)).status == "NO-GO"



def test_decision_remaining_operator_boundaries_and_missing_evidence():
    # c1 and c2 are inclusive, then become inconclusive immediately above.
    assert n.evaluate_stage0_decision(**decision_inputs(clean_nc_fpr=.02, clean_b0_fpr=.01)).status == "GO"
    assert n.evaluate_stage0_decision(**decision_inputs(clean_nc_fpr=np.nextafter(.02, 1), clean_b0_fpr=.02)).status == "INCONCLUSIVE"
    assert n.evaluate_stage0_decision(**decision_inputs(clean_nc_fpr=.02, clean_b0_fpr=np.nextafter(.01, 0))).status == "INCONCLUSIVE"
    one_ci = {scenario: 0.0 for scenario in n.ATTACK_SCENARIOS}; one_ci["DS1"] = np.nextafter(0.0, 1)
    assert n.evaluate_stage0_decision(**decision_inputs(pauc_ci_lower=one_ci)).status == "INCONCLUSIVE"
    two_ci = dict(one_ci); two_ci["DS2"] = np.nextafter(0.0, 1)
    assert n.evaluate_stage0_decision(**decision_inputs(pauc_ci_lower=two_ci)).status == "GO"
    assert n.evaluate_stage0_decision(**decision_inputs(actual_nc_mean_pauc_gain=.01, shuffled_nc_mean_pauc_gain=.01)).status == "INCONCLUSIVE"
    assert n.evaluate_stage0_decision(**decision_inputs(actual_nc_mean_pauc_gain=0.0, shuffled_nc_mean_pauc_gain=-1.0)).status == "INCONCLUSIVE"
    assert n.evaluate_stage0_decision(**decision_inputs(clean_b0_fpr=None)).status == "INCONCLUSIVE"
    assert n.evaluate_stage0_decision(**decision_inputs(equal_rmse_pass="true")).status == "INCONCLUSIVE"

def test_config_and_docs_encode_machine_boolean_grammar_and_physical_caveat():
    c = n.load_config()
    assert c["geometry"]["primary_include_width"] is False
    assert c["geometry"]["primary_tangents"] == ["amplitude", "shift"]
    assert c["decision"]["boolean_grammar"]["GO"] == "c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8"
    assert c["decision"]["machine_grammar"]["criteria"]["c6"]["rhs"] is True
    bad = json.loads(json.dumps(c)); bad["geometry"]["primary_include_width"] = "false"
    with pytest.raises(ValueError, match="JSON boolean false"):
        n.validate_config(bad)
    text = (Path(__file__).parents[1] / "docs" / "NC_TOPI_STAGE0.md").read_text()
    for phrase in ["normalized-shape scale direction", "not physical receiver global gain",
                   "S_perp = r_perp.T W r_perp", "scale, not scale squared",
                   "full identity set equality", "q99 NC-TOPI median only"]:
        assert phrase in text


# Retained numerical/split/synthetic contracts.
def test_aggregation_quantile_split_and_synthetic_contracts():
    ids = np.array(["G04", "G01", "G03", "G02", "G05"]); s = np.array([4., 1, 3, 2, 100])
    assert n.aggregate_prn_scores(ids, s).score == 3
    assert n.aggregate_prn_scores(ids, s, "top25_mean").score == 52
    threshold = n.higher_quantile([1, 2, 3, 4], .5, fit_roles=["normal_calibration"] * 4)
    assert threshold == 3 and n.strict_alarms([3, np.nextafter(3., 4.)], threshold).tolist() == [False, True]
    m = n.source_support_split([0, 320, 420], [300, 400, 421], scenario="cleanStatic")
    assert m.train.tolist() == [1, 0, 0] and m.calibration.tolist() == [0, 1, 0] and m.holdout.tolist() == [0, 0, 1]
    x = n.CANONICAL_TAP_COORDS; p = peak()
    assert np.allclose(n.second_peak_perturbation(p, x, .2, .25) - p, np.sqrt(.2) * n.shift_peak(p, x, .25))
    assert n.standardized_pauc([0, 0, 1, 1], [.1, .2, .8, .9]) == pytest.approx(
        roc_auc_score([0, 0, 1, 1], [.1, .2, .8, .9], max_fpr=.05))
