"""Executable tests for the frozen GCSPO Stage-0 contract.

These tests deliberately exercise scientific equations and access boundaries,
not just artifact shape.  Protected receiver rows are never opened here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.gcspo import (
    AccessGate,
    Role,
    SharedVAR,
    aggregate_20ms,
    apply_var_transfer,
    block_index,
    build_physical_loading,
    build_state_prior_precision,
    common_epoch_covariance,
    common_support,
    content_seed,
    empirical_threshold,
    fit_common_gamma,
    fit_whitener,
    geometry_observability,
    los_derangement,
    map_edf_score,
    nearest_rank_percentile,
    persistent_three_of_five,
    pooled_signed_innovation_score,
    role_for_interval,
    temporal_desynchronization,
    validate_role_disjointness,
    weighted_low_fpr_pauc,
)


def test_causal_20ms_aggregation_and_right_endpoint():
    rows = [
        {"time_s": 0.000, "sample_count": 0, "prn": 2, "q": [1.0, 5.0]},
        {"time_s": 0.019, "sample_count": 19, "prn": 2, "q": [3.0, 1.0]},
        {"time_s": 0.020, "sample_count": 20, "prn": 2, "q": [9.0, 7.0]},
        {"time_s": 0.039, "sample_count": 39, "prn": 2, "q": [7.0, 3.0]},
    ]
    epochs = aggregate_20ms(rows)
    assert [e["epoch"] for e in epochs] == [0, 1]
    assert [e["availability_s"] for e in epochs] == pytest.approx([0.02, 0.04])
    assert epochs[0]["q"].tolist() == pytest.approx([2.0, 3.0])
    assert epochs[1]["q"].tolist() == pytest.approx([8.0, 5.0])
    # Adding a future row cannot alter an already emitted epoch.
    assert aggregate_20ms(rows[:2])[0]["q"].tolist() == epochs[0]["q"].tolist()


def test_exact_duplicate_scientific_rows_are_fatal():
    row = {"time_s": 0.001, "sample_count": 25, "prn": 7, "q": [1.0]}
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_20ms([row, dict(row)])


def test_roles_are_half_open_disjoint_and_windows_cannot_cross():
    roles = [Role("train", 30.0, 210.0), Role("calibration", 220.0, 340.0), Role("holdout", 350.0, 470.0)]
    validate_role_disjointness(roles)
    assert role_for_interval(30.0, 31.0, roles).name == "train"
    assert role_for_interval(209.5, 210.0, roles).name == "train"
    assert role_for_interval(209.5, 210.5, roles) is None
    assert role_for_interval(210.0, 211.0, roles) is None
    with pytest.raises(ValueError, match="overlap"):
        validate_role_disjointness([Role("a", 0, 1), Role("b", 0.5, 2)])


def test_shared_var_is_causal_shared_and_resets_history():
    # q_t = [q_(t-1,1)+1, q_(t-1,2)-2] for every PRN.
    histories, targets = [], []
    for offset in (0.0, 10.0, -3.0):
        q = np.asarray([[offset + k, offset - 2 * k] for k in range(8)], dtype=float)
        for k in range(1, len(q)):
            histories.append(q[k - 1 : k])
            targets.append(q[k])
    model = SharedVAR.fit(np.asarray(histories), np.asarray(targets), ridge=1e-8)
    assert model.lags == 1
    pred = model.predict(np.asarray([[100.0, -50.0]]))
    assert pred == pytest.approx([101.0, -52.0], abs=1e-5)
    with pytest.raises(ValueError, match="history"):
        model.predict(np.empty((0, 2)))


def test_whitener_is_train_only_symmetric_and_sign_preserving():
    rng = np.random.default_rng(23)
    residuals = rng.normal(size=(300, 3)) @ np.asarray([[2.0, 0.1, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 0.5]])
    w = fit_whitener(residuals)
    assert np.array_equal(w.inverse_sqrt, w.inverse_sqrt.T)
    z = w.transform(residuals)
    assert np.all(np.isfinite(z))
    assert np.sign(w.transform(w.location + np.asarray([1.0, 0.0, 0.0]))[0]) == 1


def test_gamma_is_fit_in_z_coordinates_and_common_covariance_is_shared():
    z_by_epoch = [np.asarray([[1.0, 0.0], [1.2, 0.1], [0.8, -0.1], [1.1, 0.0]]) + k for k in range(8)]
    gamma = fit_common_gamma(z_by_epoch)
    assert gamma.shape == (2, 2)
    assert np.linalg.eigvalsh(gamma).min() >= 0
    covariance = common_epoch_covariance(gamma, prn_count=4)
    assert covariance.shape == (8, 8)
    assert np.array_equal(covariance, covariance.T)
    assert np.allclose(covariance[:2, :2], np.eye(2) + gamma)
    assert np.allclose(covariance[:2, 2:4], gamma)


def tetrahedron_los():
    x = 1.0 / math.sqrt(3.0)
    return np.asarray([[x, x, x], [x, -x, -x], [-x, x, -x], [-x, -x, x]])


def test_geometry_rank_condition_and_variable_prn_count():
    ok = geometry_observability(tetrahedron_los())
    assert ok["available"] and ok["rank"] == 4 and ok["condition_number"] <= 10_000
    unavailable = geometry_observability(tetrahedron_los()[:3])
    assert unavailable["available"] is False
    collinear = np.tile([1.0, 0.0, 0.0], (4, 1))
    assert geometry_observability(collinear)["available"] is False


def test_physical_loading_uses_frozen_units_signs_and_zero_epl_rows():
    loading = build_physical_loading(
        np.asarray([1.0, 0.0, 0.0]),
        validated_rows={"code_error_chips", "pll_phase_error_cycles", "carrier_doppler_hz", "code_frequency_offset_chips_s"},
    )
    c = 299_792_458.0
    chip_m = c / 1_023_000.0
    wavelength = c / 1_575_420_000.0
    # For p_x=1, rho=-1 and the frozen q shift -rho/unit is positive.
    assert loading[6, 0] == pytest.approx(1.0 / chip_m)
    assert loading[7, 0] == pytest.approx(1.0 / wavelength)
    assert loading[8, 4] == pytest.approx(1.0 / wavelength)
    assert loading[9, 4] == pytest.approx(1.0 / chip_m)
    assert not loading[:6].any()


def test_unverified_physical_loading_is_exactly_zero():
    loading = build_physical_loading(tetrahedron_los()[0], validated_rows=set())
    assert np.array_equal(loading, np.zeros((10, 8)))


def test_var_transfer_is_applied_exactly_once_with_zero_prehistory():
    # One q coordinate, one state, VAR(1) coefficient .25.
    direct = np.asarray([[[2.0]], [[3.0]], [[5.0]]])
    out = apply_var_transfer(direct, np.asarray([[[0.25]]]))
    assert out[:, 0, 0] == pytest.approx([2.0, 2.5, 4.25])
    assert not np.allclose(out, apply_var_transfer(out, np.asarray([[[0.25]]])))


def test_normalized_constant_velocity_prior_has_no_cross_window_edge():
    precision = build_state_prior_precision(epoch_count=3, smoothness=2.0, dt_s=0.02)
    assert precision.shape == (24, 24)
    assert np.array_equal(precision, precision.T)
    assert np.linalg.eigvalsh(precision).min() > 0
    one = build_state_prior_precision(epoch_count=1, smoothness=2.0, dt_s=0.02)
    assert one.shape == (8, 8)


def test_map_score_is_observation_gain_minus_exactly_one_edf_bic_penalty():
    y = np.asarray([2.0, -1.0, 0.5])
    g = np.asarray([[1.0], [0.5], [-0.25]])
    r = np.asarray([[2.0]])
    result = map_edf_score(y, g, r)
    x = np.linalg.solve(g.T @ g + r, g.T @ y)
    gain = float(y @ y - (y - g @ x) @ (y - g @ x))
    influence = g @ np.linalg.solve(g.T @ g + r, g.T)
    edf = float(np.trace(influence))
    assert result["state"] == pytest.approx(x)
    assert result["likelihood_improvement_twice"] == pytest.approx(gain)
    assert result["effective_dof"] == pytest.approx(edf)
    assert result["penalty"] == pytest.approx(edf * math.log(len(y)))
    assert result["score"] == pytest.approx(gain - edf * math.log(len(y)))
    assert 0 <= edf <= np.linalg.matrix_rank(g) <= len(y)


def test_a1_pooling_is_prn_permutation_invariant():
    z = np.arange(4 * 50 * 3, dtype=float).reshape(4, 50, 3) / 100.0
    a = pooled_signed_innovation_score(z)
    b = pooled_signed_innovation_score(z[[2, 0, 3, 1]])
    assert a == pytest.approx(b)


def test_empirical_threshold_is_nearest_rank_and_alarm_is_strict():
    values = np.arange(1.0, 101.0)
    threshold = empirical_threshold(values, 0.99)
    assert threshold == 99.0
    assert not (threshold > threshold)
    assert 100.0 > threshold


def test_persistence_is_causal_three_of_five_and_missing_breaks_run():
    alarms = [True, False, True, False, True, True, None, True, True, True, True, True]
    result = persistent_three_of_five(alarms)
    assert result[:4] == [False] * 4
    assert result[4] is True and result[5] is True
    assert result[6:9] == [False, False, False]
    assert result[9] is False and result[11] is True


def test_common_support_removes_unavailable_rows_from_every_method():
    support = common_support({"Full": [1.0, np.nan, 3.0], "A1": [4.0, 5.0, 6.0], "A2": [7.0, 8.0, np.nan]})
    assert support["mask"].tolist() == [True, False, False]
    assert all(values.tolist() == [original] for values, original in zip((support["scores"][k] for k in ("Full", "A1", "A2")), (1.0, 4.0, 7.0)))


def test_los_shuffle_preserves_los_rows_and_deranges_identity():
    los = tetrahedron_los()
    shuffled, permutation, record = los_derangement(los, seed=content_seed("LOS_SHUFFLE", "DS3", "transition", 0, "NA", "segment-0"))
    assert sorted(map(tuple, shuffled)) == sorted(map(tuple, los))
    assert not np.any(permutation == np.arange(4))
    assert record["control_id"] == "LOS_SHUFFLE"


def test_temporal_desynchronization_preserves_each_prn_vector_norm_and_energy():
    r = np.arange(4 * 80 * 3, dtype=float).reshape(4, 80, 3)
    shifted, shifts = temporal_desynchronization(r, seed=content_seed("PER_PRN_TEMPORAL_SHIFT", "DS3", "transition", 0, "NA", "segment-0"))
    assert len(set(shifts)) == 4
    for before, after in zip(r, shifted):
        assert sorted(map(tuple, before)) == sorted(map(tuple, after))
        assert np.linalg.norm(before) == pytest.approx(np.linalg.norm(after))
    assert np.sum(r * r) == pytest.approx(np.sum(shifted * shifted))


def test_content_rng_seed_matches_frozen_sha256_encoding():
    material = "23|CONTROL|DS3|transition|2|0.5|object"
    expected = int.from_bytes(hashlib.sha256(material.encode()).digest()[:16], "big")
    assert content_seed("CONTROL", "DS3", "transition", 2, 0.5, "object") == expected


def test_weighted_low_fpr_pauc_groups_ties_and_balances_cells():
    scores = np.asarray([0.9, 0.8, 0.8, 0.1, 0.7, 0.6, 0.2, 0.0])
    labels = np.asarray([1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)
    cells = np.asarray(["p1", "p1", "n1", "n1", "p2", "n2", "n2", "n2"])
    value = weighted_low_fpr_pauc(scores, labels, cells, alpha=0.05)
    assert 0.0 <= value <= 1.0
    # A perfect ranking has normalized pAUC one.
    perfect = weighted_low_fpr_pauc(np.asarray([4, 3, 2, 1.0]), np.asarray([1, 1, 0, 0], bool), np.asarray(["p", "p", "n", "n"]), alpha=0.05)
    assert perfect == pytest.approx(1.0)


def test_bootstrap_endpoint_assignment_and_nearest_rank_percentiles():
    assert block_index(10.0, phase_start=0.0) == 0
    assert block_index(10.0000001, phase_start=0.0) == 1
    assert nearest_rank_percentile(np.arange(2000), 0.05) == 99
    assert nearest_rank_percentile(np.arange(2000), 0.025) == 49
    assert nearest_rank_percentile(np.arange(2000), 0.975) == 1949


def test_ds7_ds8_are_one_family_in_frozen_preregistration():
    path = Path(__file__).parents[1] / "artifacts/gcspo_stage0_static/preregistration.json"
    frozen = json.loads(path.read_text())
    assert frozen["timelines"]["DS7"]["post_110_family"] == frozen["timelines"]["DS8"]["post_110_family"]
    assert "DS7 pre-110 TRAINING_REPLAY_DESCRIPTIVE" in frozen["primary_pooled_static_contrast"]["excluded_negative"]


def test_protected_gate_rejects_before_complete_freeze_and_exact_remote_sync(tmp_path):
    gate = AccessGate(tmp_path / "access_ledger.jsonl")
    with pytest.raises(PermissionError, match="VALID_FOR_PROTECTED_ACCESS"):
        gate.authorize(Path("/protected/ds3.mat"), scenario="DS3", phase="transition", expected_sha256="0" * 64, expected_size=1)
    gate.set_preflight(clean_only_pass=True, reviews_pass=True, freeze_sha="a" * 40, frozen_hashes={"science": "b" * 64})
    with pytest.raises(PermissionError, match="remote"):
        gate.authorize(Path("/protected/ds3.mat"), scenario="DS3", phase="transition", expected_sha256="0" * 64, expected_size=1)
    gate.set_remote_sync(local_sha="a" * 40, remote_sha="a" * 40, ahead=0, behind=0, clean=True)
    assert gate.state == "VALID_FOR_PROTECTED_ACCESS"


def test_access_gate_rejects_directories_globs_and_prior_results(tmp_path):
    gate = AccessGate(tmp_path / "ledger.jsonl")
    gate.set_preflight(clean_only_pass=True, reviews_pass=True, freeze_sha="a" * 40, frozen_hashes={"science": "b" * 64})
    gate.set_remote_sync(local_sha="a" * 40, remote_sha="a" * 40, ahead=0, behind=0, clean=True)
    for bad in (tmp_path, Path("/x/*.mat"), Path("/x/scenario_metrics.csv")):
        with pytest.raises((PermissionError, ValueError)):
            gate.authorize(bad, scenario="DS3", phase="transition", expected_sha256="0" * 64, expected_size=1)


def test_frozen_files_have_contract_sha256():
    root = Path(__file__).parents[1]
    expected = {
        "artifacts/gcspo_stage0_static/README.md": "cb24096d8160e0ebea1e78aec416ba3abdd678940d2d829ebf2c4ad351bda8df",
        "artifacts/gcspo_stage0_static/config.json": "919353cbf66230df506a9eb672d366dc61450b6637003f470939c0d3c91ee30e",
        "artifacts/gcspo_stage0_static/data_inventory.json": "4faffaede28119f7655da25b44129b09e76f1bb49ec5169861b6336abaea3631",
        "artifacts/gcspo_stage0_static/preregistration.json": "2390fddb2048db9c333dbb9d7a7bae1c3a174fa59a144902cf9743ad21501a03",
        "artifacts/gcspo_stage0_static/source_commit.json": "38215e854859dd18816625d089bac5ff8d1e7378882abdd42d70ce19c8e895d3",
        "docs/GCSPO_STAGE0.md": "d5cb40d436b55cd58cff3063e018b19b4f8296f5af5e96331b77af663324314f",
    }
    assert {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in expected} == expected


def test_receiver_semantic_preflight_is_source_hash_bound_and_proves_requested_rows(tmp_path):
    from gnss_doppler_lab.gcspo_artifacts import preflight_receiver_semantics
    source_root = Path("/home/ubuntu/build-gnss-sdr-complex9")
    if not source_root.is_dir():
        pytest.skip("pinned receiver source is not installed")
    report = preflight_receiver_semantics(source_root)
    assert report["overall_status"] == "PASS"
    assert set(report["validated_rows"]) == {
        "code_error_chips", "pll_phase_error_cycles",
        "carrier_doppler_hz", "code_frequency_offset_chips_s",
    }
    assert report["synthetic_vectors"]["range_impulse"]["sign"] == "negative"
    assert report["synthetic_vectors"]["rate_ramp"]["carrier_aiding_ratio"] == pytest.approx(1 / 1540)


def test_manifest_is_sorted_recursive_and_excludes_only_contract_cycle(tmp_path):
    from gnss_doppler_lab.gcspo_artifacts import build_artifact_manifest, verify_artifact_manifest
    (tmp_path / "plots").mkdir()
    (tmp_path / "a.json").write_text("{}\n")
    (tmp_path / "plots/data.csv").write_text("x\n1\n")
    for excluded in ("artifact_manifest_sha256.json", "verifier_report.json", "fresh_clone_verifier_report.json"):
        (tmp_path / excluded).write_text("excluded\n")
    manifest = build_artifact_manifest(tmp_path)
    assert [row["path"] for row in manifest["files"]] == ["a.json", "plots/data.csv"]
    verify_artifact_manifest(tmp_path, manifest)
    (tmp_path / "a.json").write_text("changed\n")
    with pytest.raises(ValueError, match="checksum"):
        verify_artifact_manifest(tmp_path, manifest)


def test_valid_manifest_packaging_quarantines_clean_intermediates(tmp_path):
    from gnss_doppler_lab.gcspo_artifacts import prepare_valid_artifact_manifest
    from gnss_doppler_lab.gcspo_verify import FINAL_REQUIRED

    reports = {"verifier_report.json", "fresh_clone_verifier_report.json"}
    for name in FINAL_REQUIRED - reports - {"artifact_manifest_sha256.json"}:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("contract artifact\n")
    plots = tmp_path / "plots"
    plots.mkdir()
    (plots / "numeric_sidecar.csv").write_text("score\n1\n")
    (tmp_path / "implementation_manifest.json").write_text("{}\n")
    (tmp_path / "clean_only_report.json").write_text("clean intermediate\n")
    (tmp_path / "b0_clean_recomputed").mkdir()
    (tmp_path / "b0_clean_recomputed/scratch.csv").write_text("scratch\n")

    manifest = prepare_valid_artifact_manifest(tmp_path)
    paths = {row["path"] for row in manifest["files"]}
    assert "implementation_manifest.json" in paths and "plots/numeric_sidecar.csv" in paths
    assert not (tmp_path / "clean_only_report.json").exists()
    assert not (tmp_path / "b0_clean_recomputed").exists()
    runner = (Path(__file__).parents[1] / "scripts/run_gcspo_stage0.py").read_text()
    assert "prepare_valid_artifact_manifest(args.artifact_dir)" in runner


def test_invalid_run_exact_set_has_zero_attack_access_and_no_verdict(tmp_path):
    from gnss_doppler_lab.gcspo_artifacts import write_fail_closed_invalid, verify_invalid_artifacts
    frozen_root = Path(__file__).parents[1] / "artifacts/gcspo_stage0_static"
    for name in ("README.md", "config.json", "data_inventory.json", "preregistration.json", "source_commit.json"):
        (tmp_path / name).write_bytes((frozen_root / name).read_bytes())
    (tmp_path / "clean_only_report.json").write_text("clean intermediate\n")
    (tmp_path / "b0_clean_recomputed").mkdir()
    (tmp_path / "b0_clean_recomputed/scratch.csv").write_text("scratch\n")
    write_fail_closed_invalid(tmp_path, reason_codes=["MANDATORY_FULL_UNAVAILABLE"], failed_checks=[{"id": "closed_loop_transfer_jacobian", "status": "FAIL"}], target_commit="a" * 40)
    report = verify_invalid_artifacts(tmp_path, allow_missing_reports=True)
    assert report["run_status"] == "INVALID_EXPERIMENT_NO_ATTACK_ACCESS"
    assert json.loads((tmp_path / "invalid_run.json").read_text())["attack_access_count"] == 0
    assert not (tmp_path / "final_verdict.json").exists()


def test_preaccess_invalid_cannot_erase_a_started_access_ledger(tmp_path):
    from gnss_doppler_lab.gcspo_artifacts import write_fail_closed_invalid
    frozen_root = Path(__file__).parents[1] / "artifacts/gcspo_stage0_static"
    for name in ("README.md", "config.json", "data_inventory.json", "preregistration.json", "source_commit.json"):
        (tmp_path / name).write_bytes((frozen_root / name).read_bytes())
    ledger = tmp_path / "access_ledger.jsonl"
    ledger.write_text('{"record_type":"PRE"}\n')
    with pytest.raises(RuntimeError, match="protected access"):
        write_fail_closed_invalid(tmp_path, reason_codes=["CRASH"], failed_checks=[], target_commit="a" * 40)
    assert ledger.read_text() == '{"record_type":"PRE"}\n'


def test_preaccess_failure_handler_writes_exact_invalid_without_running_protected(tmp_path):
    import importlib.util
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("gcspo_stage0_runner", root / "scripts/run_gcspo_stage0.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    frozen_root = root / "artifacts/gcspo_stage0_static"
    for name in ("README.md", "config.json", "data_inventory.json", "preregistration.json", "source_commit.json"):
        (tmp_path / name).write_bytes((frozen_root / name).read_bytes())

    module._record_preaccess_invalid(tmp_path, PermissionError("freeze absent"))
    from gnss_doppler_lab.gcspo_artifacts import verify_invalid_artifacts
    invalid = verify_invalid_artifacts(tmp_path, allow_missing_reports=True)
    assert invalid["attack_access_count"] == 0 and invalid["reason_codes"] == ["PREACCESS_PREFLIGHT_FAILED"]
    assert not (tmp_path / "final_verdict.json").exists()



def test_poststart_failure_quarantines_verdict_and_preserves_access_ledger(tmp_path):
    from gnss_doppler_lab.gcspo_artifacts import quarantine_failed_final_verdict
    ledger = tmp_path / "access_ledger.jsonl"
    ledger.write_text('{"record_type":"PRE"}\n')
    verdict = tmp_path / "final_verdict.json"
    verdict.write_text('{"scientific_status":"VALID_SCIENCE"}\n')
    quarantine_failed_final_verdict(tmp_path)
    assert not verdict.exists()
    assert ledger.read_text() == '{"record_type":"PRE"}\n'
    runner = (Path(__file__).parents[1] / "scripts/run_gcspo_stage0.py").read_text()
    assert "quarantine_failed_final_verdict(args.artifact_dir)" in runner


def test_signed_q_normalization_uses_train_epsilon_and_excludes_duplicate_fields():
    from gnss_doppler_lab.gcspo_clean import signed_q
    columns = {name: np.asarray(value, float) for name, value in {
        "I_E": [3], "Q_E": [4], "I_P": [0], "Q_P": [0], "I_L": [0], "Q_L": [0],
        "code_error_chips": [.25], "carr_error_hz": [-.1], "carrier_doppler_hz": [12], "code_freq_chips": [1023001],
    }.items()}
    q = signed_q(columns, epsilon=np.asarray([5.0]))
    assert q.shape == (1, 10)
    assert q[0, :6] == pytest.approx([.3, .4, 0, 0, 0, 0])
    assert q[0, 6:] == pytest.approx([.25, -.1, 12, 1])


def test_causal_history_builder_resets_on_gap_and_role_boundary():
    from gnss_doppler_lab.gcspo_clean import causal_histories
    epochs = np.asarray([0, 1, 2, 4, 5, 6, 10, 11, 12])
    values = np.column_stack([epochs, -epochs]).astype(float)
    histories, targets, target_epochs = causal_histories(epochs, values, lags=2)
    assert target_epochs.tolist() == [2, 6, 12]
    assert histories[:, :, 0].tolist() == [[0, 1], [4, 5], [10, 11]]
    assert targets[:, 0].tolist() == [2, 6, 12]


def test_clean_ridge_selection_uses_validation_and_larger_relative_tie():
    from gnss_doppler_lab.gcspo_clean import select_shared_var_ridge
    histories = np.zeros((12, 1, 1)); targets = np.ones((12, 1))
    model, report = select_shared_var_ridge(histories[:8], targets[:8], histories[8:], targets[8:], [0.1, 1.0])
    assert report["selected_ridge"] == 1.0
    assert model.lags == 1


def test_frozen_window_endpoints_are_right_end_anchored_and_contained():
    from gnss_doppler_lab.gcspo_clean import window_endpoints
    assert window_endpoints(220, 222).tolist() == pytest.approx([221, 221.5, 222])
    assert window_endpoints(220.1, 222.1).tolist() == pytest.approx([221.5, 222])


def test_a2_loading_has_frozen_sign_and_rejects_no_eigengap():
    from gnss_doppler_lab.gcspo_ablations import fit_a2_loading
    x = [np.asarray([[k, 0.1], [k + .1, -.1], [k - .1, 0], [k + .2, .05]]) for k in range(10)]
    loading, report = fit_a2_loading(x)
    assert np.linalg.norm(loading) == pytest.approx(1)
    assert loading[np.argmax(np.abs(loading))] > 0
    assert report["status"] == "PASS"
    isotropic = [np.asarray([[1, 0], [-1, 0], [0, 1], [0, -1]], float) for _ in range(4)]
    with pytest.raises(ValueError, match="eigengap"):
        fit_a2_loading(isotropic)


def test_a2_scalar_prior_is_exact_first_difference_precision():
    from gnss_doppler_lab.gcspo_ablations import scalar_random_walk_precision
    expected = np.asarray([[2, -1, 0], [-1, 2, -1], [0, -1, 1]], float) * 2
    assert scalar_random_walk_precision(3, 2) == pytest.approx(expected)


def test_a5_prior_is_block_independent_and_resets_each_prn_segment():
    from gnss_doppler_lab.gcspo_a5 import a5_prior_precision
    prior = a5_prior_precision(prn_count=2, epoch_count=3, smoothness=1.0)
    assert prior.shape == (12, 12)
    assert prior == pytest.approx(prior.T)
    assert np.linalg.eigvalsh(prior).min() > 0
    # State ordering is epoch, PRN, [normalized range, normalized rate].
    assert prior[0, 2] == 0 and prior[1, 3] == 0


def test_a5_loading_uses_normalized_range_and_rate_scales():
    from gnss_doppler_lab.gcspo_a5 import a5_direct_loading
    loading = a5_direct_loading({"code_error_chips", "pll_phase_error_cycles", "carrier_doppler_hz", "code_frequency_offset_chips_s"})
    assert loading.shape == (10, 2)
    assert loading[6, 0] == pytest.approx(-10 / (299792458 / 1023000))
    assert loading[7, 0] == pytest.approx(-10 / (299792458 / 1575420000))
    assert loading[8, 1] == pytest.approx(-1 / (299792458 / 1575420000))
    assert loading[9, 1] == pytest.approx(-1 / (299792458 / 1023000))
