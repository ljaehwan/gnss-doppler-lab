from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest
from gnss_doppler_lab.gcmr_geometry import GpsEphemeris

from gnss_doppler_lab.r2c_gnss import (
    C_M_S, AnalyticResidualWhitener, ComplexTapProvenance, SmallNeuralNuisanceModel,
    SourceSupport, aggregate_a2, assign_attack_phase, assign_normal_split, availability_time,
    artifact_hashes, build_empirical_template, fit_second_source as _fit_second_source,
    fit_shared_constellation, full_score, inject_second_source, derive_stage0_verdict,
    quantile_threshold, strict_alarm, sustained_alarms, validate_complex_taps,
)

ROOT = Path(__file__).resolve().parents[1]
TAPS = np.linspace(-1, 1, 9)
GRID = np.linspace(-1, 1, 41)


def fit_second_source(*args, **kwargs):
    """Synthetic tests must opt in to the ideal template explicitly."""
    return _fit_second_source(*args, template_kind="synthetic_ideal", **kwargs)


def provenance(complex_value=True):
    return ComplexTapProvenance("source.bin", "a" * 64, "receiver-v1", "extractor-v1", 25e6,
                                .25, "cleanStatic", complex_value, "none")


def _geometry_ephemeris(*, week=49, toe=100.0):
    return GpsEphemeris(7, 1.0, 4.5e-9, .01, 5153.7955, .7, .94, -.3, -8e-9,
                        1e-10, 1e-6, 2e-6, 200., -80., 3e-8, -2e-8, toe, week,
                        toc=toe, decoded_tow=toe + 1, SV_health=0)


def _geometry_directory(tmp_path, *, week=2097, tow=100.0):
    directory = tmp_path / "receiver"; (directory / "raw").mkdir(parents=True)
    (directory / "geometry_time_binding.json").write_text(json.dumps({
        "schema": "gnss-doppler-lab.geometry-time-binding.v1", "scenario_id": "DS3",
        "full_gps_week": week, "recording_start_tow_s": tow}))
    (directory / "gps_ephemeris.xml").write_text("fixture")
    (directory / "nmea_pvt.nmea").write_text("fixture")
    with h5py.File(directory / "raw/observables.mat", "w") as handle:
        handle["RX_time"] = np.asarray([tow, tow + .1])
    return directory


def test_build_geometry_calls_alignment_and_binds_causal_pvt(tmp_path, monkeypatch):
    runner = load_runner(); directory = _geometry_directory(tmp_path)
    monkeypatch.setattr(runner, "parse_gnss_sdr_gps_ephemeris_xml",
                        lambda _: {7: _geometry_ephemeris()})
    early = np.asarray([6378137., 0., 0.]); late = np.asarray([0., 6378137., 0.])
    monkeypatch.setattr(runner, "_nmea_positions", lambda *_: [(0., early), (.5, late)])
    dataset = {"prns": [7], "time": np.asarray([.25]), "prn": np.asarray([7])}
    first_los, report = runner.build_geometry(directory, dataset, "DS3")
    assert report["time_alignment"]["full_gps_week"] == 2097
    selection = report["event_receiver_selections"]["0.250000000/G07"]
    assert selection["selected_pvt_time_s"] == 0.0 and selection["age_s"] == .25
    assert report["time_binding_source"]["sha256"] == runner.sha256_file(directory / "geometry_time_binding.json")
    monkeypatch.setattr(runner, "_nmea_positions", lambda *_: [(0., early), (.5, late * 2)])
    second_los, _ = runner.build_geometry(directory, dataset, "DS3")
    assert second_los[(.25, 7)] == pytest.approx(first_los[(.25, 7)])


def test_build_geometry_rejects_missing_week_binding_week_mismatch_and_stale_toe(tmp_path, monkeypatch):
    runner = load_runner(); dataset = {"prns": [7], "time": np.asarray([.25]), "prn": np.asarray([7])}
    directory = tmp_path / "missing"; directory.mkdir()
    with pytest.raises(ValueError, match="binding absent"):
        runner.build_geometry(directory, dataset, "DS3")
    directory = _geometry_directory(tmp_path / "week", week=2098)
    monkeypatch.setattr(runner, "parse_gnss_sdr_gps_ephemeris_xml", lambda _: {7: _geometry_ephemeris()})
    with pytest.raises(ValueError, match="week"):
        runner.build_geometry(directory, dataset, "DS3")
    directory = _geometry_directory(tmp_path / "stale", tow=1000.)
    with pytest.raises(ValueError, match="toe age"):
        runner.build_geometry(directory, dataset, "DS3", max_toe_age_s=10.)


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
    four = fit_shared_constellation(delays[:4], los[:4], minimum_prns=4)
    assert not four.valid and four.rank == 4 and four.residual_dof == 0
    assert four.reason == "zero_residual_degrees_of_freedom"


def test_geometry_rank_and_minimum_prn_fail_closed():
    delays = np.arange(3) * 1e-8; los = np.eye(3)
    fit = fit_shared_constellation(delays, los)
    assert not fit.valid and fit.reason == "insufficient_prns" and full_score([1, 2, 3], fit) == 0
    bad = fit_shared_constellation(np.arange(5) * 1e-8, np.tile([1., 0, 0], (5, 1)))
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


def test_proper_complex_covariance_orientation_and_whitening():
    rng = np.random.default_rng(23)
    # Deliberately non-real Hermitian covariance with imaginary cross terms.
    mixing = np.eye(9, dtype=complex)
    mixing[0, 1] = 0.7j
    mixing[2, 0] = 0.35 - 0.2j
    z = (rng.normal(size=(20000, 9)) + 1j * rng.normal(size=(20000, 9))) / np.sqrt(2)
    samples = z @ mixing.T
    model = AnalyticResidualWhitener(shrinkage=0, epsilon=1e-12).fit(
        samples, ["normal_train"] * len(samples))
    expected = (samples - samples.mean(0)).T @ (samples - samples.mean(0)).conj() / (len(samples)-1)
    assert model.covariance_ == pytest.approx(expected, abs=1e-10)
    assert model.covariance_ == pytest.approx(model.covariance_.conj().T, abs=1e-12)
    assert np.linalg.eigvalsh(model.covariance_).min() >= 0


def test_aggregation_permutation_and_sustained_boundary_reset():
    assert aggregate_a2([1, 5, 2, 4], 2) == aggregate_a2([4, 2, 5, 1], 2) == 4.5
    alarms = sustained_alarms([1, 1, 1, 1, 1, 1], ["a", "a", "a", "a", "b", "b"],
                               [0, .5, 1, 2, 2.5, 3], ["pre"] * 6)
    assert alarms.tolist() == [False, False, True, False, False, False]


def test_synthetic_noise_multipath_and_injection_controls():
    rng = np.random.default_rng(4); authentic = np.maximum(1 - np.abs(TAPS), 0).astype(complex)
    noisy = authentic + .01 * (rng.normal(size=9) + 1j * rng.normal(size=9))
    injected = inject_second_source(noisy, TAPS, -.5, .5, .9)
    component = injected - noisy
    assert np.vdot(component, component).real / np.vdot(noisy, noisy).real == pytest.approx(.5, abs=1e-12)
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
    assert json.loads((root / "decision.json").read_text())["verdict"] in {
        "PHYSICS_SUPPORTED", "NOT_SUPPORTED", "DATA_INVALID", "INCONCLUSIVE"
    }
    if (root / "freeze.json").exists():
        assert json.loads((root / "freeze.json").read_text())[
            "written_before_attack_score_computation"
        ]


def load_runner():
    spec = importlib.util.spec_from_file_location("r2c_runner", ROOT / "scripts/run_r2c_gnss_stage0.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_selected_real_source_time_controls_bin_boundaries():
    runner = load_runner()
    common = {"y": np.ones((2, 9), complex), "prn": np.array([1, 2]),
              "cn0": np.array([40., 40.]), "a1": np.array([1., 2.]),
              "power": np.ones(2), "bin": np.array([639, 639])}
    normal = {**common, "time": np.array([319.999, 320.001])}
    row = runner.make_epochs("cleanStatic", normal)[0]
    assert row["source_start_s"] == 319.999
    assert row["phase"] == "excluded_guard_or_boundary"
    attack = {**common, "time": np.array([99.999, 100.001]), "bin": np.array([199, 199])}
    assert runner.make_epochs("DS3", attack)[0]["phase"] == "transition_excluded"
    exact = {**common, "time": np.array([100., 100.001]), "bin": np.array([200, 200])}
    assert runner.make_epochs("DS3", exact)[0]["phase"] == "post"


def test_condition_scaler_is_train_only_and_future_attack_cannot_change_score(monkeypatch):
    runner = load_runner()
    base = {"y": np.ones((3, 9), complex), "cn0": np.array([40., 41., np.nan])}
    train = np.array([True, True, False])
    scaler = runner.fit_condition_scaler(base, train)
    before = runner._conditions(base, scaler)[0].copy()
    changed = {"y": base["y"].copy(), "cn0": base["cn0"].copy()}
    changed["y"][2] *= 1e9; changed["cn0"][2] = -1e9
    after = runner._conditions(changed, scaler)[0]
    assert after == pytest.approx(before)
    assert scaler["fit_role"] == "cleanStatic normal_train"
    assert len(scaler["sha256"]) == 64


def test_npz_validation_rejects_degenerate_q_and_bad_prn(tmp_path):
    runner = load_runner()
    common = {"time_s": np.array([0., .1]), "channel": np.array([0, 0]),
              "segment_index": np.array([0, 0]), "sample_count": np.array([0, 1])}
    for name, prn, q in (("bad_q", [1, 1], 0.), ("bad_prn", [0, 1], 1.)):
        iq = np.ones((2, 9, 2), np.float32); iq[..., 1] *= q
        path = tmp_path / f"{name}.npz"
        np.savez(path, complex_iq=iq, prn=np.array(prn), **common)
        with pytest.raises(ValueError):
            runner.load_sample(path)


def test_decision_is_derived_fail_closed_from_required_interface():
    gates = {name: {"status": "PASS"} for name in
             ("complex_provenance", "time_alignment", "los_geometry", "b0_interface")}
    gates["b0_interface"] = {"status": "FAIL"}
    result = derive_stage0_verdict(gates)
    assert result["verdict"] == "DATA_INVALID"
    assert "time_alignment=PASS" in result["reason"]
    assert "b0_interface=FAIL" in result["reason"]


def test_decision_synthetic_evaluated_combinations_have_no_hard_coded_branch():
    names = ("complex_provenance", "time_alignment", "los_geometry", "b0_interface",
             "clean_dynamic_fpr", "gain_invariance", "phase_invariance", "full_exceeds_b0",
             "full_b0_ci", "geometry_improvement", "relation_destruction", "shortcut_controls")
    passing = {name: {"status": "PASS"} for name in names}
    assert derive_stage0_verdict(passing)["verdict"] == "PHYSICS_SUPPORTED"
    failing = {name: dict(value) for name, value in passing.items()}
    failing["geometry_improvement"] = {"status": "FAIL"}
    result = derive_stage0_verdict(failing)
    assert result["verdict"] == "NOT_SUPPORTED"
    assert "geometry_improvement=FAIL" in result["reason"]
    absent = {name: dict(value) for name, value in passing.items()}
    absent["full_b0_ci"] = {"status": "NOT_EVALUATED"}
    assert derive_stage0_verdict(absent)["verdict"] == "NOT_SUPPORTED"


def test_empirical_template_is_normal_train_only_and_real_fit_requires_it():
    rng = np.random.default_rng(9)
    base = np.asarray([.5, .62, .75, .88, 1., .87, .74, .61, .49], complex)
    rows = np.asarray([base * (1 + .1 * rng.random()) * np.exp(1j * rng.uniform(-np.pi, np.pi))
                       for _ in range(30)])
    template, metadata = build_empirical_template(rows, ["normal_train"] * len(rows))
    assert metadata["fit_role"] == "cleanStatic normal_train"
    assert template[4] == pytest.approx(1)
    with pytest.raises(ValueError, match="normal_train"):
        build_empirical_template(np.r_[rows, rows[:1]], ["normal_train"] * len(rows) + ["post"])
    with pytest.raises(ValueError, match="requires frozen"):
        _fit_second_source(rows[0], TAPS, GRID, template_kind="empirical_receiver")
    real = _fit_second_source(rows[0], TAPS, GRID, template_kind="empirical_receiver",
                              template_values=template)
    changed = _fit_second_source(rows[0], TAPS, GRID, template_kind="empirical_receiver",
                                 template_values=template + np.linspace(0, .2, 9))
    assert real.score != pytest.approx(changed.score)


def test_gain_sweep_recomputes_scores_not_constant(monkeypatch):
    runner = load_runner()
    class Fit:
        score = 0.0
    def fake(value, *args, **kwargs):
        result = Fit()
        result.score = float(np.abs(value[0]))
        return result
    monkeypatch.setattr(runner, "fit_second_source", fake)
    dataset = {"y": np.ones((2, 9), complex), "time": np.array([320.1, 320.2]),
               "bin": np.array([640, 640]), "prn": np.array([1, 2]),
               "cn0": np.array([40., 40.]), "a1": np.ones(2), "power": np.ones(2)}
    reference = runner.make_epochs("cleanStatic", dataset)[0]
    gained = runner.make_epochs("cleanStatic", dataset, gain=2., template=np.ones(9))[0]
    assert gained["A1"] == 2 * reference["A1"]


def test_verifier_tamper_negative_semantics(tmp_path):
    source = ROOT / "artifacts/r2c_gnss_stage0"
    copied = tmp_path / "artifact"
    shutil.copytree(source, copied)
    rows = (copied / "per_epoch_scores.csv").read_text().splitlines()
    fields = rows[0].split(",")
    values = rows[1].split(",")
    if "A1" in fields:
        values[fields.index("A1")] = str(float(values[fields.index("A1")]) + 123.0)
    else:
        values[fields.index("status")] = "TAMPERED"
    rows[1] = ",".join(values)
    (copied / "per_epoch_scores.csv").write_text("\n".join(rows) + "\n")
    (copied / "hashes.json").write_text(json.dumps(
        {"algorithm": "sha256", "files": artifact_hashes(copied)}) + "\n")
    spec = importlib.util.spec_from_file_location("r2c_verifier", ROOT / "scripts/verify_r2c_gnss_stage0.py")
    verifier = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(verifier)
    errors = verifier.verify(copied, check_external=False)
    assert errors


@pytest.mark.parametrize("relative", [
    "per_epoch_scores.csv", "gain_invariance.json", "bootstrap_comparisons.json",
    "decision.json", "thresholds.json", "input_validity.json", "freeze.json",
])
def test_independent_recomputation_rejects_regenerated_hash_tampering(tmp_path, relative):
    source = ROOT / "artifacts/r2c_gnss_stage0"
    authentic, changed = tmp_path / "authentic", tmp_path / "changed"
    shutil.copytree(source, authentic); shutil.copytree(source, changed)
    target = changed / relative
    if target.suffix == ".csv":
        target.write_text(target.read_text() + "# tampered\n")
    else:
        value = json.loads(target.read_text()); value["tampered_after_generation"] = True
        target.write_text(json.dumps(value, sort_keys=True) + "\n")
    (changed / "hashes.json").write_text(json.dumps(
        {"algorithm": "sha256", "files": artifact_hashes(changed)}) + "\n")
    spec = importlib.util.spec_from_file_location("r2c_verifier_compare", ROOT / "scripts/verify_r2c_gnss_stage0.py")
    verifier = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(verifier)
    assert verifier.compare_recomputed_artifact(changed, authentic)
