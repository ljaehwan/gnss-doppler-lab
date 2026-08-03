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


def pair(identity=None, delta=None):
    if identity is None:
        identity = n.EpochIdentity("rec", "cleanStatic", "G01", 7, 107.)
    elif not isinstance(identity, n.EpochIdentity) and len(identity) == 3:
        recording, prn, index = identity
        identity = n.EpochIdentity(str(recording), "cleanStatic", str(prn), int(index), 100. + int(index))
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
        coordinates=n.CANONICAL_TAP_COORDS,
    )


def provenance(role, identities, scenario="cleanStatic"):
    converted = []
    for pos, identity in enumerate(identities):
        if isinstance(identity, n.EpochIdentity):
            converted.append(identity)
        else:
            recording = str(identity[0]) if isinstance(identity, (tuple, list)) else str(identity)
            index = int(identity[-1]) if isinstance(identity, (tuple, list)) else pos
            converted.append(n.EpochIdentity(recording, scenario, "G01", index, 1000. + index))
    return n.FitProvenance(scenario=scenario, role=role, identities=tuple(converted))


def pairs(count=4):
    return [pair(("clean", "G01", i), np.linspace(-.2, .2, 9) + i / 100)
            for i in range(count)]


def covariance():
    ps = pairs(30)
    return n.fit_shrinkage_covariance(
        ps, provenance=provenance("normal_train", [p.identity for p in ps]))


def records(order=(0, 1)):
    return [n.EpochRecord("rec", "DS1", f"event-{i}", i, 100.5 + i,
                          99.5 + i, 100.5 + i, True, i % 2) for i in order]


# Blocker 1: frozen score coordinate and energy/conditioner contract.
def test_peak_prediction_pair_requires_both_raw_peaks_and_validates_coordinates():
    p = pair(); identity = n.EpochIdentity("r", "cleanStatic", "G1", 1, 101.)
    assert np.allclose(p.residual_raw, p.actual_raw - p.predicted_raw)
    with pytest.raises((TypeError, ValueError), match="actual|predicted|residual-only"):
        n.PeakPredictionPair(residual_standardized=np.zeros(9), standardizer_std=np.ones(9),
            identity=identity, actual_space=n.RAW_SPACE, predicted_space=n.RAW_SPACE,
            residual_space=n.STANDARDIZED_SPACE, coordinates=n.CANONICAL_TAP_COORDS)
    with pytest.raises(ValueError, match="standardized"):
        n.PeakPredictionPair(actual_raw=np.ones(9), predicted_raw=np.zeros(9),
            residual_standardized=np.zeros(9), standardizer_std=np.ones(9), identity=identity,
            actual_space=n.RAW_SPACE, predicted_space=n.RAW_SPACE,
            residual_space=n.STANDARDIZED_SPACE, coordinates=n.CANONICAL_TAP_COORDS)
    with pytest.raises(ValueError, match="space"):
        n.PeakPredictionPair(actual_raw=np.ones(9), predicted_raw=np.zeros(9),
            residual_standardized=np.ones(9), standardizer_std=np.ones(9), identity=identity,
            actual_space="standardized", predicted_space=n.RAW_SPACE,
            residual_space=n.STANDARDIZED_SPACE, coordinates=n.CANONICAL_TAP_COORDS)


def test_score_bundle_uses_raw_quadratic_energy_b0_and_scale_not_scale_squared():
    p = pair(delta=np.arange(9) / 10); cov = covariance()
    basis = n.primary_tangent_basis(p, n.CANONICAL_TAP_COORDS, cov)
    conditioner, _ = fitted_conditioner()
    out = n.produce_nc_topi_scores(p, basis, cov, conditioner=conditioner,
                                    iq_features=np.ones((1, 4)))
    assert out.b0 == pytest.approx(np.sqrt(np.mean(p.residual_standardized**2)))
    assert out.topi == pytest.approx(out.projection.r_perp @ cov.W @ out.projection.r_perp)
    assert out.nc_topi == pytest.approx(out.topi / out.predicted_scale)
    assert out.conditioner_transform.shape == (1, 4)
    assert out.total == pytest.approx(out.tangent + out.perp + out.cross)


def test_conditioner_fits_log_energy_and_caps_exp_scale_at_clean_q995():
    X = np.arange(160.0).reshape(40, 4)
    energy = np.exp(0.02 * X[:, 0] + 0.2)
    train_ids = [("train", i) for i in range(30)]
    cal_ids = [("cal", i) for i in range(10)]
    c = n.RobustConditioner().fit(
        X[:30], energy[:30], provenance=provenance("normal_train", train_ids),
        feature_names=n.CONDITIONER_FEATURE_SCHEMA)
    assert c.fit_manifest_["target"] == "log(max(S_perp, energy_epsilon))"
    uncapped = np.exp(c.predict_log_energy(X[30:]))
    cap = c.calibrate_cap(X[30:], provenance=provenance("normal_calibration", cal_ids))
    assert cap == pytest.approx(np.quantile(uncapped, .995, method="higher"))
    assert np.all(c.predict_scale(X) <= cap)
    with pytest.raises(ValueError, match="frozen at q995"):
        c.calibrate_cap(X[30:], provenance=provenance("normal_calibration", cal_ids), q=.99)


# Blocker 2: width cannot leak into the primary tangent basis.
def test_primary_basis_defaults_to_amplitude_shift_and_rejects_width():
    p = pair(); cov = covariance(); b = n.primary_tangent_basis(p, n.CANONICAL_TAP_COORDS, cov)
    assert b.names == ("amplitude", "shift") and "width" not in b.names
    with pytest.raises(ValueError, match="width.*diagnostic"):
        n.normalize_tangents(p, n.CANONICAL_TAP_COORDS, cov, include_width=True)
    width = n.build_width_ablation_basis(p, n.CANONICAL_TAP_COORDS, cov)
    assert width.names == ("amplitude", "shift", "width")



# Blocker 3: covariance provenance is complete and fit is train-only.
def test_covariance_requires_exact_role_scenario_identity_provenance_and_audits_digest():
    ps = pairs(30); ids = [p.identity for p in ps]
    c = n.fit_shrinkage_covariance(ps, provenance=provenance("normal_train", ids))
    assert c.audit["identity_count"] == 30
    assert len(c.audit["identity_digest_sha256"]) == 64
    with pytest.raises(TypeError, match="provenance"):
        n.fit_shrinkage_covariance(ps)
    with pytest.raises(ValueError, match="normal_train"):
        n.fit_shrinkage_covariance(ps, provenance=provenance("normal_calibration", ids))
    with pytest.raises(ValueError, match="cleanStatic"):
        n.fit_shrinkage_covariance(ps, provenance=provenance("normal_train", ids, "DS1"))
    with pytest.raises(ValueError, match="exactly match"):
        n.fit_shrinkage_covariance(ps, provenance=provenance("normal_train", ids[::-1]))


# Blocker 4: exact W geometry, decomposition and diagnostics.
def test_primary_projection_is_unregularized_w_orthogonal_and_decomposes():
    cov = covariance(); w = cov.W; J = n.primary_tangent_basis(pair(), n.CANONICAL_TAP_COORDS, cov).matrix
    r = np.linspace(-1, 1, 9)
    q = n.weighted_project(r, J, w)
    assert q.projection_kind == "orthogonal_whitened_svd"
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
    left = records(); right = records((1, 0))
    a_map = {record.identity_key: score for record, score in zip(left, [1., 2.])}
    b_map = {record.identity_key: score for record, score in zip(right, [20., 10.])}
    got, a, b = n.exact_primary_epoch_join(left, a_map, right, b_map)
    assert got == tuple(record.identity_key for record in left)
    assert a.tolist() == [1, 2] and b.tolist() == [10, 20]
    with pytest.raises(ValueError, match="full identity set"):
        n.exact_primary_epoch_join(left, a_map, right[:1], {right[0].identity_key: 20.})
    with pytest.raises(ValueError, match="duplicate"):
        n.exact_primary_epoch_join([left[0], left[0]], a_map, right, b_map)
    diag_ids = [record.identity_key for record in left]
    diag = n.common_epoch_intersection_diagnostic(diag_ids, [1, 2], diag_ids[:1], [3])
    assert diag.audit["excluded_from_a"] == 1 and diag.identities == (diag_ids[0],)


# Blocker 9: executable frozen decision grammar and boundaries.
def decision_inputs(**overrides):
    values = dict(
        clean_nc_fpr=.02, clean_b0_fpr=.01,
        stable_pre_fpr={s: .049 for s in n.ATTACK_SCENARIOS},
        nc_pauc={s: .7 for s in n.ATTACK_SCENARIOS},
        b0_pauc={s: .6 for s in n.ATTACK_SCENARIOS},
        pauc_delta={s: .1 for s in n.ATTACK_SCENARIOS},
        nc_delay={s: 1.0 for s in n.ATTACK_SCENARIOS},
        b0_delay={s: 2.0 for s in n.ATTACK_SCENARIOS},
        pauc_ci_lower={s: .01 for s in n.ATTACK_SCENARIOS},
        pauc_ci_upper={s: .2 for s in n.ATTACK_SCENARIOS},
        equal_rmse_pass=True, second_peak_pass=True,
        actual_nc_mean_pauc=.7, topi_mean_pauc=.6, shuffled_nc_mean_pauc=.61)
    values.update(overrides)
    if "pauc_delta" in overrides and "nc_pauc" not in overrides and "b0_pauc" not in overrides:
        values["nc_pauc"] = {scenario: .6 + delta for scenario, delta in values["pauc_delta"].items()}
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
    invalid = n.evaluate_stage0_decision(**decision_inputs(pauc_delta=low, nc_delay=inf, b0_delay=inf))
    assert invalid.status == "INCONCLUSIVE" and invalid.validation_errors
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
    assert n.evaluate_stage0_decision(**decision_inputs(actual_nc_mean_pauc=.61, topi_mean_pauc=.6, shuffled_nc_mean_pauc=.61)).status == "INCONCLUSIVE"
    assert n.evaluate_stage0_decision(**decision_inputs(actual_nc_mean_pauc=.6, topi_mean_pauc=.6, shuffled_nc_mean_pauc=.5)).status == "INCONCLUSIVE"
    assert n.evaluate_stage0_decision(**decision_inputs(clean_b0_fpr=None)).status == "INCONCLUSIVE"
    assert n.evaluate_stage0_decision(**decision_inputs(equal_rmse_pass="true")).status == "INCONCLUSIVE"

def test_config_and_docs_encode_machine_boolean_grammar_and_physical_caveat():
    c = n.load_config()
    assert c["geometry"]["primary_include_width"] is False
    assert c["geometry"]["primary_tangents"] == ["amplitude", "shift"]
    assert c["geometry"]["basis_provenance"]["primary_kind"] == "primary_amp_shift"
    assert c["fit_policy"]["typed_provenance"] == "FitProvenance(scenario, role, tuple[EpochIdentity])"
    assert c["decision"]["evidence_domains"]["fpr"] == "finite [0,1]"
    assert "real scalar only" in c["decision"]["evidence_domains"]["scalar_type"]
    assert tuple(c["iq_conditioner"]["features"]) == n.CONDITIONER_FEATURE_SCHEMA
    assert c["geometry"]["basis_provenance"]["raw_weight_primary_input"] is False
    assert c["epoch_schema"]["primary_join"] == "exact full identity and metadata equality"
    assert c["decision"]["boolean_grammar"]["GO"] == "c1 && c2 && c3 && c4 && c5 && c6 && c7 && c8"
    assert c["decision"]["machine_grammar"]["criteria"]["c6"]["rhs"] is True
    bad = json.loads(json.dumps(c)); bad["geometry"]["primary_include_width"] = "false"
    with pytest.raises(ValueError, match="JSON boolean false"):
        n.validate_config(bad)
    text = (Path(__file__).parents[1] / "docs" / "NC_TOPI_STAGE0.md").read_text()
    for phrase in ["normalized-shape scale direction", "not physical receiver global gain",
                   "S_perp = r_perp.T W r_perp", "scale, not scale squared",
                   "full identity set equality", "q99 NC-TOPI median only",
                   "FitProvenance(scenario, role, tuple[EpochIdentity])", "primary_amp_shift",
                   "produce_topi_scores", "`ThresholdCalibration` is factory-only",
                   "log_noise_floor_scale", "validation_errors"]:
        assert phrase in text


# Retained numerical/split/synthetic contracts.
def test_aggregation_quantile_split_and_synthetic_contracts():
    ids = np.array(["G04", "G01", "G03", "G02", "G05"]); s = np.array([4., 1, 3, 2, 100])
    assert n.aggregate_prn_scores(ids, s).score == 3
    assert n.aggregate_prn_scores(ids, s, "top25_mean").score == 52
    threshold = n.calibrate_threshold([1, 2, 3, 4], .99, provenance=provenance(
        "normal_calibration", [("cal", i) for i in range(4)]),
        detector="NC-TOPI", aggregator="median")
    assert threshold.value == 4 and n.strict_alarms(
        [4, np.nextafter(4., 5.)], threshold).tolist() == [False, True]
    m = n.source_support_split([0, 320, 420], [300, 400, 421], scenario="cleanStatic")
    assert m.train.tolist() == [1, 0, 0] and m.calibration.tolist() == [0, 1, 0] and m.holdout.tolist() == [0, 0, 1]
    x = n.CANONICAL_TAP_COORDS; p = peak()
    assert np.allclose(n.second_peak_perturbation(p, x, .2, .25) - p, np.sqrt(.2) * n.shift_peak(p, x, .25))
    assert n.standardized_pauc([0, 0, 1, 1], [.1, .2, .8, .9]) == pytest.approx(
        roc_auc_score([0, 0, 1, 1], [.1, .2, .8, .9], max_fpr=.05))


# Release-blocker regression: every primary and fit path is typed and fail-closed.
def test_primary_basis_is_immutable_pair_bound_and_width_has_diagnostic_score_only():
    p = pair(); cov = covariance()
    basis = n.primary_tangent_basis(p, n.CANONICAL_TAP_COORDS, cov)
    assert basis.basis_kind == "primary_amp_shift"
    assert basis.names == ("amplitude", "shift")
    assert not basis.matrix.flags.writeable
    with pytest.raises(ValueError):
        basis.matrix[0, 0] = 123
    with pytest.raises(TypeError, match="factory-only"):
        n.TangentBasis()
    with pytest.raises(ValueError):
        basis.matrix.flags.writeable = True
    assert not p.predicted_raw.flags.writeable
    out = n.produce_topi_scores(p, basis, cov)
    assert out.identity == p.identity
    with pytest.raises((TypeError, ValueError), match="TangentBasis|basis"):
        n.produce_topi_scores(p, basis.matrix, cov)
    with pytest.raises((TypeError, ValueError), match="predicted|identity|coordinate"):
        n.produce_topi_scores(pair(("other", "G01", 7)), basis, cov)
    with pytest.raises((TypeError, ValueError), match="PeakPredictionPair"):
        n.primary_tangent_basis(p.residual_raw, n.CANONICAL_TAP_COORDS, cov)
    width = n.build_width_ablation_basis(p, n.CANONICAL_TAP_COORDS, cov)
    assert width.basis_kind == "width_diagnostic"
    with pytest.raises(ValueError, match="diagnostic|primary"):
        n.produce_topi_scores(p, width, cov)
    diagnostic = n.produce_width_ablation_scores(p, width, cov)
    assert diagnostic.label == "width_diagnostic" and diagnostic.primary is False
    fake = n.TangentBasis._create(
        p, cov, np.ones_like(basis.matrix), np.ones_like(basis.raw),
        ("amplitude", "shift"), "primary_amp_shift", {})
    with pytest.raises(ValueError, match="arbitrary basis"):
        n.produce_topi_scores(p, fake, cov)


def test_covariance_derives_raw_residuals_from_typed_pairs_and_fit_provenance():
    ps = pairs(30); ids = [p.identity for p in ps]
    fit = n.fit_shrinkage_covariance(ps, provenance=provenance("normal_train", ids))
    assert fit.audit["residual_space"] == n.RAW_SPACE
    assert fit.audit["pair_identity_digest_sha256"]
    with pytest.raises((TypeError, ValueError), match="PeakPredictionPair|ResidualBatch"):
        n.fit_shrinkage_covariance(np.stack([p.residual_standardized for p in ps]),
                                   provenance=provenance("normal_train", ids))
    batch = n.ResidualBatch.from_pairs(ps)
    assert not batch.residual_raw.flags.writeable
    with pytest.raises(TypeError, match="factory-only"):
        n.ResidualBatch()
    n.fit_shrinkage_covariance(batch, provenance=provenance("normal_train", ids))


def test_primary_whitened_svd_projection_reports_effective_rank_and_tolerance():
    J = np.diag([1., 1e-6]); r = np.array([1., 1.]); w = np.eye(2)
    q = n.weighted_project(r, J, w, pinv_rcond=1e-10)
    assert q.effective_rank == 2 and q.rank == 2
    assert q.rank_tolerance == pytest.approx(1e-10)
    assert q.orthogonality_defect <= q.orthogonality_tolerance
    assert q.orthogonality_verified_full_span
    dropped = n.weighted_project(r, J, w, pinv_rcond=1e-5)
    assert dropped.effective_rank == 1
    assert not dropped.orthogonality_verified_full_span
    assert dropped.orthogonality_scope == "retained_effective_tangent_span"


def test_sustained_alarm_sorts_per_recording_rejects_duplicates_and_audits_each_recording():
    out = n.sustained_alarm_delay(
        [11., 10., 10.5, 21., 20., 20.5], [1] * 6,
        recording_ids=["a", "a", "a", "b", "b", "b"],
        post_eligible_mask=[1] * 6, onset=10.,
        stable_pre_mask=[0, 0, 0, 1, 0, 0])
    assert out.alarm_time == 11. and out.delay == 1.
    assert out.stable_pre_alarm_by_recording == {"a": False, "b": True}
    with pytest.raises(ValueError, match="duplicate"):
        n.sustained_alarm_delay([10, 10], [1, 1], recording_ids=["a", "a"],
                                post_eligible_mask=[1, 1], onset=10)


@pytest.mark.parametrize("bad", ["", "   ", None, 3])
def test_iq_group_ids_are_nonempty_strings_and_cannot_cross_groups(bad):
    with pytest.raises(ValueError, match="group.*nonempty string"):
        n.build_causal_iq_context([2], [1], np.ones((1, 1)), history=1,
                                  target_groups=[bad], block_groups=[bad])
    with pytest.raises(ValueError, match="exactly match"):
        n.build_causal_iq_context([2], [1], np.ones((1, 1)), history=1,
                                  target_groups=["recording-a"], block_groups=["recording-b"])


def test_epoch_record_primary_join_has_fixed_identity_and_mandatory_equal_metadata():
    left = records(); right = records((1, 0))
    score_a = {record.identity_key: float(i + 1) for i, record in enumerate(left)}
    score_b = {right[0].identity_key: 20., right[1].identity_key: 10.}
    identity, a, b = n.exact_primary_epoch_join(left, score_a, right, score_b)
    assert identity == tuple(record.identity_key for record in left)
    assert a.tolist() == [1, 2] and b.tolist() == [10, 20]
    changed = list(right)
    changed[0] = n.EpochRecord("rec", "DS1", "event-1", 1, 101.5, 100.5, 101.5, True, 0)
    with pytest.raises(ValueError, match="label"):
        n.exact_primary_epoch_join(left, score_a, changed, score_b)
    with pytest.raises(ValueError, match="score map"):
        n.exact_primary_epoch_join(left, {left[0].identity_key: 1.}, right, score_b)


def test_decision_invalid_domains_and_scenario_sets_are_always_inconclusive():
    base = decision_inputs()
    base.update(
        nc_pauc={s: .7 for s in n.ATTACK_SCENARIOS},
        b0_pauc={s: .6 for s in n.ATTACK_SCENARIOS},
        pauc_ci_upper={s: .2 for s in n.ATTACK_SCENARIOS},
        actual_nc_mean_pauc=.7, topi_mean_pauc=.6, shuffled_nc_mean_pauc=.61)
    assert n.evaluate_stage0_decision(**base).status == "GO"
    malformed = [
        ("clean_nc_fpr", -1), ("clean_b0_fpr", 1.1),
        ("actual_nc_mean_pauc", np.nan), ("topi_mean_pauc", np.inf),
        ("equal_rmse_pass", 1),
    ]
    for key, value in malformed:
        bad = dict(base); bad[key] = value
        out = n.evaluate_stage0_decision(**bad)
        assert out.status == "INCONCLUSIVE" and out.validation_errors
    bad = dict(base); bad["pauc_ci_lower"] = {s: .3 for s in n.ATTACK_SCENARIOS}
    assert n.evaluate_stage0_decision(**bad).validation_errors
    for mapping_name in ("stable_pre_fpr", "pauc_delta", "nc_delay", "b0_delay",
                         "pauc_ci_lower", "pauc_ci_upper", "nc_pauc", "b0_pauc"):
        bad = dict(base); values = dict(bad[mapping_name]); values.pop("DS8"); values["DS9"] = 0
        bad[mapping_name] = values
        out = n.evaluate_stage0_decision(**bad)
        assert out.status == "INCONCLUSIVE" and out.validation_errors
    bad = dict(base); bad["nc_delay"] = {s: (-1 if s == "DS1" else None) for s in n.ATTACK_SCENARIOS}
    assert n.evaluate_stage0_decision(**bad).validation_errors


def test_all_fit_and_calibration_apis_require_disjoint_typed_clean_provenance():
    X = np.arange(160.).reshape(40, 4); y = np.linspace(1, 2, 40)
    train_ids = [("train", i) for i in range(30)]
    cal_ids = [("cal", i) for i in range(10)]
    c = n.RobustConditioner().fit(X[:30], y[:30],
                                  provenance=provenance("normal_train", train_ids))
    assert c.fit_manifest_["identity_digest_sha256"]
    assert len(c.fit_manifest_["fit_digest_sha256"]) == 64
    c.calibrate_cap(X[30:], provenance=provenance("normal_calibration", cal_ids))
    with pytest.raises((TypeError, ValueError), match="provenance"):
        n.RobustConditioner().fit(X[:30], y[:30])
    for scenario, role in (("DS1", "normal_train"), ("cleanStatic", "normal_calibration")):
        with pytest.raises(ValueError, match="cleanStatic|normal_train"):
            n.RobustConditioner().fit(X[:30], y[:30],
                                      provenance=provenance(role, train_ids, scenario))
    with pytest.raises(ValueError, match="disjoint"):
        c.calibrate_cap(X[:10], provenance=provenance("normal_calibration", train_ids[:10]))
    with pytest.raises(ValueError, match="normal_calibration"):
        c.calibrate_cap(X[30:], provenance=provenance("normal_train", cal_ids))
    with pytest.raises(ValueError, match="cleanStatic"):
        c.calibrate_cap(X[30:], provenance=provenance("normal_calibration", cal_ids, "DS1"))
    with pytest.raises(ValueError, match="length"):
        c.calibrate_cap(X[30:], provenance=provenance("normal_calibration", cal_ids[:-1]))
    with pytest.raises(TypeError, match="provenance"):
        n.calibrate_threshold(y[30:], .99, provenance=None)
    with pytest.raises(ValueError, match="normal_calibration"):
        n.calibrate_threshold(y[30:], .99, provenance=provenance("normal_train", cal_ids))
    threshold = n.calibrate_threshold(y[30:], .99,
                                      provenance=provenance("normal_calibration", cal_ids))
    assert threshold.identity_digest_sha256 and threshold.value == pytest.approx(y[-1])
    shuffled = n.shuffled_control_target(y[:30], provenance=provenance("normal_train", train_ids))
    assert np.array_equal(shuffled, n.shuffled_control_target(
        y[:30], provenance=provenance("normal_train", train_ids)))
    with pytest.raises(ValueError, match="cleanStatic"):
        n.shuffled_control_target(y[:30], provenance=provenance("normal_train", train_ids, "DS1"))
    with pytest.raises(ValueError, match="normal_train"):
        n.shuffled_control_target(y[:30], provenance=provenance("normal_calibration", train_ids))
    with pytest.raises(TypeError, match="provenance"):
        n.shuffled_control_target(y[:30], provenance=None)


# Final Stage-0 primary-boundary adversarial probes.
def canonical_identity(index=0, *, prn="G01"):
    return n.EpochIdentity("rec", "cleanStatic", prn, index, 100.0 + index)


def sealed_geometry(count=30):
    ps = [pair(canonical_identity(i), np.linspace(-.2, .2, 9) + i / 100)
          for i in range(count)]
    fit = n.fit_shrinkage_covariance(
        ps, provenance=provenance("normal_train", [p.identity for p in ps]))
    return ps, fit


def fitted_conditioner():
    X = np.arange(160., dtype=float).reshape(40, 4)
    y = np.linspace(1., 2., 40)
    train = [canonical_identity(i) for i in range(30)]
    cal = [n.EpochIdentity("rec", "cleanStatic", "G01", 100 + i, 500. + i)
           for i in range(10)]
    c = n.RobustConditioner().fit(
        X[:30], y[:30], provenance=provenance("normal_train", train),
        feature_names=n.CONDITIONER_FEATURE_SCHEMA)
    c.calibrate_cap(X[30:], provenance=provenance("normal_calibration", cal))
    return c, X


def test_primary_score_and_alarm_boundaries_require_sealed_typed_state():
    ps, covariance = sealed_geometry()
    p = ps[0]
    basis = n.primary_tangent_basis(p, n.CANONICAL_TAP_COORDS, covariance)
    topi = n.produce_topi_scores(p, basis, covariance)
    assert topi.topi == pytest.approx(topi.projection.perp_energy)
    with pytest.raises(TypeError, match="CovarianceFit"):
        n.produce_topi_scores(p, basis, covariance.W)
    with pytest.raises((TypeError, ValueError), match="RobustConditioner|calibrated|sealed"):
        n.produce_nc_topi_scores(p, basis, covariance, conditioner=None,
                                 iq_features=np.ones((1, 4)))
    class Duck:
        def conditioner_transform(self, X): return X
        def predict_scale(self, X): return np.ones(len(X))
    with pytest.raises(TypeError, match="RobustConditioner"):
        n.produce_nc_topi_scores(p, basis, covariance, conditioner=Duck(),
                                 iq_features=np.ones((1, 4)))
    conditioner, _ = fitted_conditioner()
    nc = n.produce_nc_topi_scores(
        p, basis, covariance, conditioner=conditioner, iq_features=np.ones((1, 4)))
    assert nc.nc_topi == pytest.approx(nc.topi / nc.predicted_scale)

    ids = [canonical_identity(200 + i) for i in range(4)]
    threshold = n.calibrate_threshold(
        [1, 2, 3, 4], .99, provenance=provenance("normal_calibration", ids),
        detector="NC-TOPI", aggregator="median")
    assert n.strict_alarms([threshold.value, np.nextafter(threshold.value, np.inf)],
                           threshold).tolist() == [False, True]
    with pytest.raises(TypeError, match="ThresholdCalibration"):
        n.strict_alarms([3., 4.], threshold.value)


def test_every_sealed_boundary_revalidates_content_after_object_setattr_tamper():
    ps, covariance = sealed_geometry()
    batch = n.ResidualBatch.from_pairs(ps)
    object.__setattr__(batch, "residual_space", "b0_standardized")
    with pytest.raises(ValueError, match="ResidualBatch|seal|raw"):
        n.fit_shrinkage_covariance(
            batch, provenance=provenance("normal_train", [p.identity for p in ps]))

    p = ps[0]
    basis = n.primary_tangent_basis(p, n.CANONICAL_TAP_COORDS, covariance)
    object.__setattr__(covariance, "W", np.eye(9))
    with pytest.raises(ValueError, match="CovarianceFit|seal"):
        n.produce_topi_scores(p, basis, covariance)

    conditioner, _ = fitted_conditioner()
    object.__setattr__(conditioner, "cap_", conditioner.cap_ + 1.)
    with pytest.raises(ValueError, match="conditioner.*seal|sealed"):
        n.produce_nc_topi_scores(p, basis, sealed_geometry()[1], conditioner=conditioner,
                                 iq_features=np.ones((1, 4)))

    ids = [canonical_identity(300 + i) for i in range(4)]
    threshold = n.calibrate_threshold(
        [1, 2, 3, 4], .99, provenance=provenance("normal_calibration", ids),
        detector="NC-TOPI", aggregator="median")
    object.__setattr__(threshold, "quantile", .995)
    with pytest.raises(ValueError, match="ThresholdCalibration|seal|quantile"):
        n.strict_alarms([4.], threshold)


def test_identity_digest_and_primary_coordinates_are_canonical_and_exact():
    identity = canonical_identity()
    p = pair(identity)
    with pytest.raises((TypeError, ValueError), match="EpochIdentity|identity"):
        pair(("rec", "cleanStatic", "G01", 0, 100.))
    for args in [(" rec", "cleanStatic", "G01", 0, 100.),
                 ("rec", "cleanStatic", "G01 ", 0, 100.)]:
        with pytest.raises(ValueError, match="nonempty|whitespace|canonical"):
            n.EpochIdentity(*args)
    with pytest.raises(ValueError, match="coordinates.*canonical"):
        n.PeakPredictionPair(
            actual_raw=p.actual_raw, predicted_raw=p.predicted_raw,
            residual_standardized=p.residual_standardized,
            standardizer_std=p.standardizer_std, identity=identity,
            actual_space=n.RAW_SPACE, predicted_space=n.RAW_SPACE,
            residual_space=n.STANDARDIZED_SPACE,
            coordinates=np.nextafter(n.CANONICAL_TAP_COORDS, np.inf))
    with pytest.raises(ValueError, match="availability"):
        n.EpochIdentity("rec", "cleanStatic", "G01", 0, np.inf)
    with pytest.raises(ValueError, match="target_index"):
        n.EpochIdentity("rec", "cleanStatic", "G01", True, 100.)


def test_decision_evidence_rejects_strings_arrays_objects_and_accepts_numpy_bool():
    for bad in ["0.02", [0.02], np.array([0.02]), object()]:
        out = n.evaluate_stage0_decision(**decision_inputs(clean_nc_fpr=bad))
        assert out.status == "INCONCLUSIVE" and out.validation_errors
    for bad in ["true", [True], np.array([True]), 1]:
        out = n.evaluate_stage0_decision(**decision_inputs(equal_rmse_pass=bad))
        assert out.status == "INCONCLUSIVE" and out.validation_errors
    assert n.evaluate_stage0_decision(**decision_inputs(
        equal_rmse_pass=np.bool_(True), second_peak_pass=np.bool_(True))).status == "GO"


def test_conditioner_schema_is_exact_frozen_width_and_tamper_checked_on_transform():
    X = np.arange(160., dtype=float).reshape(40, 4)
    y = np.linspace(1., 2., 40)
    ids = [canonical_identity(i) for i in range(30)]
    good = n.CONDITIONER_FEATURE_SCHEMA
    for names in [good[::-1], good[:-1], ("log_power ",) + good[1:],
                  ("LOG_POWER",) + good[1:], ("PRN ",) + good[1:]]:
        with pytest.raises(ValueError, match="feature|schema|forbidden"):
            n.RobustConditioner().fit(
                X[:30], y[:30], provenance=provenance("normal_train", ids),
                feature_names=names)
    with pytest.raises(ValueError, match="width|four|4"):
        n.RobustConditioner().fit(
            X[:30, :3], y[:30], provenance=provenance("normal_train", ids),
            feature_names=good)
    c = n.RobustConditioner().fit(
        X[:30], y[:30], provenance=provenance("normal_train", ids), feature_names=good)
    with pytest.raises(ValueError, match="dimension|width|four|4"):
        c.conditioner_transform(X[:2, :3])
    shuffled = n.shuffled_control_target(
        y[:30], provenance=provenance("normal_train", ids), feature_names=good)
    assert shuffled.shape == (30,)
    with pytest.raises(ValueError, match="feature|schema"):
        n.shuffled_control_target(
            y[:30], provenance=provenance("normal_train", ids),
            feature_names=good[::-1])



def test_pair_basis_identity_provenance_and_conditioner_valid_value_tamper_fail_closed():
    ps, cov = sealed_geometry(); p = ps[0]
    basis = n.primary_tangent_basis(p, n.CANONICAL_TAP_COORDS, cov)
    object.__setattr__(p.identity, "prn", "G02")
    with pytest.raises(ValueError, match="EpochIdentity|seal"):
        n.produce_topi_scores(p, basis, cov)

    p = ps[1]; basis = n.primary_tangent_basis(p, n.CANONICAL_TAP_COORDS, cov)
    object.__setattr__(basis, "names", ("shift", "amplitude"))
    with pytest.raises(ValueError, match="basis names|seal"):
        n.produce_topi_scores(p, basis, cov)

    ids = [canonical_identity(400 + i) for i in range(4)]
    fit = provenance("normal_calibration", ids)
    object.__setattr__(fit, "role", "normal_train")
    with pytest.raises(ValueError, match="normal_calibration|FitProvenance|provenance"):
        n.calibrate_threshold([1, 2, 3, 4], .99, provenance=fit)

    conditioner, _ = fitted_conditioner()
    replacement = tuple(reversed(conditioner._fit_identities_))
    object.__setattr__(conditioner, "_fit_identities_", replacement)
    with pytest.raises(ValueError, match="conditioner.*seal|provenance"):
        conditioner.predict_scale(np.ones((1, 4)))


def test_primary_alarm_checks_detector_aggregator_quantile_and_factory_seal():
    ids = [canonical_identity(500 + i) for i in range(4)]
    fit = provenance("normal_calibration", ids)
    bad_calibrations = [
        n.calibrate_threshold([1, 2, 3, 4], .99, provenance=fit,
                              detector="TOPI", aggregator="median"),
        n.calibrate_threshold([1, 2, 3, 4], .99, provenance=fit,
                              detector="NC-TOPI", aggregator="top25_mean"),
        n.calibrate_threshold([1, 2, 3, 4], .995, provenance=fit,
                              detector="NC-TOPI", aggregator="median"),
    ]
    for threshold in bad_calibrations:
        with pytest.raises(ValueError, match="q99 NC-TOPI median"):
            n.strict_alarms([4.], threshold)
    with pytest.raises(TypeError, match="factory-only"):
        n.ThresholdCalibration()
    uncalibrated = n.RobustConditioner().fit(
        np.arange(120.).reshape(30, 4), np.linspace(1, 2, 30),
        provenance=provenance("normal_train", [canonical_identity(600 + i) for i in range(30)]),
        feature_names=n.CONDITIONER_FEATURE_SCHEMA)
    ps, cov = sealed_geometry(); p = ps[0]
    basis = n.primary_tangent_basis(p, n.CANONICAL_TAP_COORDS, cov)
    with pytest.raises(ValueError, match="calibrated"):
        n.produce_nc_topi_scores(p, basis, cov, conditioner=uncalibrated,
                                 iq_features=np.ones((1, 4)))


# Projection workspace/performance contracts.
def test_projection_workspace_reuses_whitening_preserves_scores_and_tamper_fails():
    p = pair(delta=np.arange(9) / 13); cov = covariance()
    basis = n.primary_tangent_basis(p, n.CANONICAL_TAP_COORDS, cov)
    legacy = n.produce_topi_scores(p, basis, cov)
    workspace = n.ProjectionWorkspace.from_covariance(cov)
    cached = n.produce_topi_scores(p, basis, cov, workspace=workspace)
    assert cached.topi == pytest.approx(legacy.topi, rel=1e-13, abs=1e-14)
    assert np.allclose(cached.projection.r_perp, legacy.projection.r_perp, rtol=1e-13, atol=1e-14)
    assert cached.projection.orthogonality_verified_full_span == legacy.projection.orthogonality_verified_full_span
    object.__setattr__(workspace, "covariance_fit_digest_sha256", "0" * 64)
    with pytest.raises(ValueError, match="workspace|covariance"):
        n.produce_topi_scores(p, basis, cov, workspace=workspace)


def test_condition_precomputed_geometry_matches_old_api_and_rejects_wrong_pair_basis():
    p = pair(delta=np.arange(9) / 17); cov = covariance()
    basis = n.primary_tangent_basis(p, n.CANONICAL_TAP_COORDS, cov)
    workspace = n.ProjectionWorkspace.from_covariance(cov)
    top = n.produce_topi_scores(p, basis, cov, workspace=workspace)
    conditioner, _ = fitted_conditioner()
    features = np.ones((1, 4))
    old = n.produce_nc_topi_scores(p, basis, cov, conditioner=conditioner,
                                    iq_features=features, workspace=workspace)
    reused = n.condition_topi_scores(top, p, basis, cov, conditioner=conditioner,
                                     iq_features=features, workspace=workspace)
    assert reused.nc_topi == pytest.approx(old.nc_topi, rel=0, abs=0)
    assert reused.topi == pytest.approx(top.topi, rel=0, abs=0)
    other = pair(("other", "G01", 88), delta=np.arange(9) / 17)
    other_basis = n.primary_tangent_basis(other, n.CANONICAL_TAP_COORDS, cov)
    with pytest.raises(ValueError, match="identity|pair|geometry"):
        n.condition_topi_scores(top, other, other_basis, cov, conditioner=conditioner,
                                iq_features=features, workspace=workspace)


def test_workspace_projection_inventory_1000_pairs_under_15_seconds():
    import time
    rng = np.random.default_rng(20260803); predicted = peak(); std = np.linspace(.5, 1.3, 9)
    ps = []
    for i in range(1030):
        residual = rng.normal(0, .08, 9)
        ident = n.EpochIdentity("perf", "cleanStatic", f"G{i % 24 + 1:02d}", i, 1000 + i * .5)
        ps.append(n.PeakPredictionPair(predicted + residual, predicted, residual / std, std, ident,
            n.RAW_SPACE, n.RAW_SPACE, n.STANDARDIZED_SPACE, n.CANONICAL_TAP_COORDS))
    train = ps[:30]; cov = n.fit_shrinkage_covariance(train,
        provenance=n.FitProvenance("cleanStatic", "normal_train", tuple(x.identity for x in train)))
    workspace = n.ProjectionWorkspace.from_covariance(cov)
    X=rng.normal(size=(1030,4)); target=np.linspace(.1,2.,30)
    actual=n.RobustConditioner().fit(X[:30],target,
        provenance=n.FitProvenance("cleanStatic","normal_train",tuple(x.identity for x in train)))
    shuffled=n.RobustConditioner().fit(X[:30],target[::-1],
        provenance=n.FitProvenance("cleanStatic","normal_train",tuple(x.identity for x in train)))
    calibration=n.FitProvenance("cleanStatic","normal_calibration",tuple(x.identity for x in ps[30:60]))
    actual.calibrate_cap(X[30:60],provenance=calibration);shuffled.calibrate_cap(X[30:60],provenance=calibration)
    start = time.perf_counter()
    for i,p in enumerate(ps[30:]):
        basis = n.primary_tangent_basis(p, n.CANONICAL_TAP_COORDS, cov)
        top = n.produce_topi_scores(p, basis, cov, workspace=workspace)
        n.condition_topi_scores(top,p,basis,cov,conditioner=actual,iq_features=X[i:i+1],workspace=workspace)
        n.condition_topi_scores(top,p,basis,cov,conditioner=shuffled,iq_features=X[i:i+1],workspace=workspace)
        width_basis = n.build_width_ablation_basis(p, n.CANONICAL_TAP_COORDS, cov)
        n.produce_width_ablation_scores(p, width_basis, cov, workspace=workspace)
        n.weighted_project(p.residual_raw, basis.matrix[:, [0]], cov.W,
                           workspace=workspace, covariance=cov)
        n.weighted_project(p.residual_raw, basis.matrix[:, [1]], cov.W,
                           workspace=workspace, covariance=cov)
        assert np.isfinite(top.topi)
    assert time.perf_counter() - start < 15
