import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.gcmr_experiment import (
    implementation_manifest, load_checkpoint, score_events, train_clean_model,
)
from gnss_doppler_lab.gcmr_model import CleanReferenceScoreCalibrator

from gnss_doppler_lab.gcmr_relations import GcmrPairRelationEvent

SCRIPT = Path(__file__).parents[1] / "scripts/run_gcmr_texbat_loso.py"
spec = importlib.util.spec_from_file_location("gcmr_texbat_loso", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def event(start):
    return GcmrPairRelationEvent(
        float(start), float(start + 1), np.array([[1, 2]]),
        np.ones((1, 10)), np.ones((1, 10), dtype=bool), np.ones((1, 8)),
    )


def regular_events(last_start=460.5):
    return [event(x) for x in np.arange(0, last_start + .01, .5)]


def test_fixed_fold_order_donors_and_hyperparameters():
    assert module.SCENARIOS == ("DS1", "DS2", "DS3", "DS4")
    assert [module.donors_for(x) for x in module.SCENARIOS] == [
        ("DS2", "DS3", "DS4"), ("DS1", "DS3", "DS4"),
        ("DS1", "DS2", "DS4"), ("DS1", "DS2", "DS3"),
    ]
    assert module.TRAINING_CONFIG == {
        "seed": 7, "max_epochs": 40, "patience": 6,
        "learning_rate": 1e-3, "compactness_weight": .01,
        "warmup_epochs": 5, "pair_hidden": 32,
        "event_hidden": 64, "latent_dim": 32,
    }
    assert module.THRESHOLD_QUANTILE == .99


def test_roles_whole_window_counts_order_and_support_purges():
    roles = module.split_donor_roles(regular_events())
    assert list(roles) == ["train", "validation", "clean_reference", "calibration"]
    assert [len(x) for x in roles.values()] == [47, 21, 21, 21]
    assert [(x[0].window_start_s, x[-1].window_end_s) for x in roles.values()] == [
        (30., 54.), (55., 66.), (67., 78.), (79., 90.)]
    module.validate_role_support(roles)
    contaminated = copy.deepcopy(roles)
    contaminated["validation"] = [event(53.5), *contaminated["validation"]]
    with pytest.raises(ValueError, match="support overlap|contain"):
        module.validate_role_support(contaminated)


def test_prepare_fold_exact_exclusion_counts_and_order():
    all_events = {x: regular_events() for x in module.SCENARIOS}
    roles, sources = module.prepare_fold_roles(all_events, "DS2")
    assert sources == {name: ["DS1", "DS3", "DS4"] for name in module.ROLE_INTERVALS}
    assert [len(roles[x]) for x in module.ROLE_INTERVALS] == [141, 63, 63, 63]
    module.assert_no_held_contamination("DS2", sources)
    bad = copy.deepcopy(sources); bad["calibration"].append("DS2")
    with pytest.raises(ValueError, match="held.*contamination"):
        module.assert_no_held_contamination("DS2", bad)


def test_held_regions_use_whole_windows_and_expected_counts():
    expected = {
        "DS1": (702, 460.5), "DS2": (693, 456.),
        "DS3": (694, 456.5), "DS4": (35, 127.),
    }
    for name, (post_count, last_start) in expected.items():
        regions = module.split_held_regions(regular_events(last_start))
        assert [len(regions[x]) for x in ("stable_pre", "transition", "stable_post")] == [119, 39, post_count]
        module.validate_held_counts(name, regions)
    boundary = [event(x) for x in (29.5, 30., 89., 89.5, 90., 109., 109.5, 110.)]
    regions = module.split_held_regions(boundary)
    assert [x.window_start_s for x in regions["stable_pre"]] == [30., 89.]
    assert [x.window_start_s for x in regions["transition"]] == [90., 109.]
    assert [x.window_start_s for x in regions["stable_post"]] == [110.]


def test_strict_alarm_metrics_and_earliest_causal_latency():
    scored = {
        "window_start_s": np.array([99., 99.5, 100., 100.5, 101.]),
        "window_end_s": np.array([100., 100.5, 101., 101.5, 102.]),
        "availability_s": np.array([100., 100.5, 101., 101.5, 102.]),
        "combined_score": np.array([9., 9., 2., 3., 9.]),
    }
    m = module.score_metrics(scored, threshold=2.)
    assert m["alarm_count"] == 4  # equality is not an alarm
    latency = module.onset_latency(scored, threshold=2.)
    assert latency == {"first_alarm_available_s": 101.5, "first_alarm_delay_s": 1.5}
    scored["combined_score"] = np.array([9., 9., 3., 3., 9.])
    assert module.onset_latency(scored, 2.)["first_alarm_delay_s"] == 1.


def test_train_all_before_evaluate_state_machine():
    gate = module.LosoCampaignGate()
    gate.mark_trained("DS1")
    with pytest.raises(RuntimeError, match="all four"):
        gate.freeze_all()
    with pytest.raises(RuntimeError, match="frozen"):
        gate.begin_evaluation("DS1")
    for held in ("DS2", "DS3", "DS4"): gate.mark_trained(held)
    gate.freeze_all()
    assert gate.training_order == list(module.SCENARIOS)
    for held in module.SCENARIOS: gate.begin_evaluation(held)
    with pytest.raises(RuntimeError, match="duplicate"):
        gate.begin_evaluation("DS1")


def test_cache_hash_and_metadata_stale_rejection(tmp_path):
    path = tmp_path / "ds1.relations.npz"
    metadata = {
        "scenario": "DS1", "schema_version": 4, "relation_contract_version": 4,
        "observation_features": ["fixture"], "condition_features": ["fixture"],
        "source_sha256": {str(tmp_path / "source"): "a" * 64},
        "external_evaluation_implementation": {"external_adapter": {"path": "x", "sha256": "b" * 64}},
    }
    np.savez(path, metadata_json=np.asarray(json.dumps(metadata)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    seen = []
    def loader(p, *, source_paths, expected_metadata):
        seen.append((list(source_paths), expected_metadata)); return [event(0)], metadata
    got = module.load_pinned_event_cache(path, "DS1", expected_sha256=digest, loader=loader, adapter_validator=lambda m: None)
    assert len(got.events) == 1 and seen[0][0] == [tmp_path / "source"]
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA256"):
        module.load_pinned_event_cache(path, "DS1", expected_sha256=digest, loader=loader, adapter_validator=lambda m: None)
    np.savez(path, metadata_json=np.asarray(json.dumps({**metadata, "scenario": "DS2"})))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="metadata.*scenario"):
        module.load_pinned_event_cache(path, "DS1", expected_sha256=digest, loader=loader, adapter_validator=lambda m: None)
    def stale(*a, **k): raise ValueError("stale event cache: source hash mismatch")
    np.savez(path, metadata_json=np.asarray(json.dumps(metadata)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="stale"):
        module.load_pinned_event_cache(path, "DS1", expected_sha256=digest, loader=stale, adapter_validator=lambda m: None)


def test_pinned_real_cache_hashes_are_complete():
    assert module.CACHE_SHA256 == {
        "DS1": "8ddf6d7a6b70d4c7497ccb75ca0844a50fd6eeebd733ef030dbb11ce1bbbcfef",
        "DS2": "c18201693e4240247b06647e5fae5eb7e51cf82a703018b838c03e678880a24d",
        "DS3": "bc1ded77e5ce3d74262338980ea3668a4f2ba8eaf5c8bcbb25a4ea80821a35ed",
        "DS4": "518be3f9154b52f58a777351f97f81d3a3e3998d6a3343f9cae6791d129c576f",
    }


def test_threshold_role_separation(monkeypatch):
    calls = []
    class Model: pass
    class Training:
        model = Model(); best_epoch = 1; history = []; config = module.TRAINING_CONFIG
    monkeypatch.setattr(module, "train_clean_model", lambda tr, va, **kw: calls.append(("train", tr, va, kw)) or Training())
    def score(model, events, calibrator=None, **kw):
        calls.append(("score", events, calibrator))
        n=len(events); return {"reconstruction": np.arange(n, dtype=float), "latent": np.ones((n, 2)), "combined_score": np.arange(n, dtype=float)}
    monkeypatch.setattr(module, "score_events", score)
    class Cal:
        def fit(self, r, z): calls.append(("fit_calibrator", r.copy(), z.copy())); return self
    monkeypatch.setattr(module, "CleanReferenceScoreCalibrator", Cal)
    monkeypatch.setattr(module, "calibration_threshold", lambda x, quantile: calls.append(("threshold", x.copy(), quantile)) or 42.)
    roles = {name: [event(i)] * (2 if name == "clean_reference" else 1) for i, name in enumerate(module.ROLE_INTERVALS)}
    fitted = module.fit_fold(roles, device="cpu", max_epochs=3)
    assert fitted.threshold == 42.
    assert calls[0][0] == "train" and calls[0][1] is roles["train"] and calls[0][2] is roles["validation"]
    assert calls[1][1] is roles["clean_reference"]
    assert calls[-2][1] is roles["calibration"]
    assert calls[-1][0] == "threshold" and calls[-1][2] == .99
    assert calls[0][3]["seed"] == 7 and calls[0][3]["patience"] == 6


def test_real_checkpoint_preflight_has_required_manifest_and_roundtrips(tmp_path):
    records = {}
    for name in module.SCENARIOS:
        cache = tmp_path / f"{name}.cache"
        cache.write_bytes(name.encode())
        records[name] = module.CacheRecord([], {"scenario": name}, cache, module._sha(cache))
    provenance = module.campaign_provenance(records)
    assert provenance["implementation"] == implementation_manifest()
    assert provenance["core_implementation_manifest"] == implementation_manifest()

    samples = regular_events(7.)
    training = train_clean_model(samples[:10], samples[10:], seed=7, max_epochs=1, device="cpu")
    raw = score_events(training.model, samples[:6], device="cpu")
    calibrator = CleanReferenceScoreCalibrator().fit(raw["reconstruction"], raw["latent"])
    expected = {**provenance, "held_scenario": "DS1", "donors": ["DS2", "DS3", "DS4"]}
    path = tmp_path / "tiny.pt"
    module.save_checkpoint(path, training, calibrator, 1.25, provenance=expected)
    loaded = load_checkpoint(path, expected_provenance=expected, device="cpu")
    before = score_events(training.model, samples[6:8], calibrator, device="cpu")
    after = score_events(loaded.model, samples[6:8], loaded.calibrator, device="cpu")
    assert loaded.provenance == expected
    assert loaded.threshold == 1.25
    assert np.array_equal(before["combined_score"], after["combined_score"])


def test_summary_classification_shared_context_and_no_event_ci():
    summary = module.build_campaign_summary({}, provenance={"fixture": True})
    assert summary["evaluation_classification"] == "within_texbat_loso_experiment"
    assert summary["external_validation"] is False
    limitation = summary["shared_context_limitation"]
    assert "distinct bytes/events" in limitation
    assert "same TEXBAT static-reference cohort" in limitation
    assert "dependent" in limitation and "independent-environment" in limitation
    assert summary["event_level_confidence_intervals"] == "not reported"


def test_ds4_conflict_and_post120_secondary_sensitivity():
    scored = {
        "window_start_s": np.array([110., 119.5, 120., 121.]),
        "window_end_s": np.array([111., 120.5, 121., 122.]),
        "availability_s": np.array([111., 120.5, 121., 122.]),
        "combined_score": np.array([0., 9., 3., 0.]),
    }
    out = module.fold_metrics("DS4", scored, threshold=2.)
    assert out["onset_conflict"]["primary_onset_s"] == 100.
    assert out["onset_conflict"]["auxiliary_onset_s"] == 110.
    assert out["post120_sensitivity"]["event_count"] == 2
    assert out["post120_sensitivity"]["alarm_count"] == 1
    assert out["post120_sensitivity"]["classification"] == "secondary_sensitivity_only"


def test_synthetic_campaign_smoke_trains_and_checkpoints_before_scoring(monkeypatch, tmp_path):
    records = {}
    last = {"DS1": 460.5, "DS2": 456., "DS3": 456.5, "DS4": 127.}
    for name in module.SCENARIOS:
        cache = tmp_path / f"{name}.cache"; cache.write_bytes(name.encode())
        records[name] = module.CacheRecord(regular_events(last[name]), {"scenario": name}, cache, module._sha(cache))
    actions = []
    monkeypatch.setattr(module, "load_all_caches", lambda cache_dir: records)
    class Training:
        model = object(); best_epoch = 1; history = [{"epoch": 1}]
        config = module.TRAINING_CONFIG
    monkeypatch.setattr(module, "fit_fold", lambda roles, **kw: actions.append("fit") or module.FittedFold(Training(), object(), 2.))
    saved = {}
    def fake_save(path, training, calibrator, threshold, *, provenance):
        actions.append("checkpoint"); Path(path).write_bytes(b"model")
        saved[str(Path(path))] = (training, calibrator, threshold, provenance)
    monkeypatch.setattr(module, "save_checkpoint", fake_save)
    class Loaded:
        def __init__(self, path, expected):
            training, calibrator, threshold, provenance = saved[str(Path(path))]
            assert provenance == expected
            self.model = ("restored", expected["held_scenario"])
            self.calibrator = ("restored-calibrator", expected["held_scenario"])
            self.threshold = threshold
            self.best_epoch = training.best_epoch
            self.provenance = provenance
    def fake_load(path, *, expected_provenance, device):
        actions.append("load")
        return Loaded(path, expected_provenance)
    monkeypatch.setattr(module, "load_checkpoint", fake_load)
    def fake_score(model, events, calibrator, **kwargs):
        assert model[0] == "restored" and calibrator[0] == "restored-calibrator"
        actions.append("score"); starts=np.asarray([e.window_start_s for e in events]); ends=starts+1
        return {"window_start_s": starts, "window_end_s": ends, "availability_s": ends,
                "reconstruction": np.zeros(len(starts)), "latent_score": np.zeros(len(starts)),
                "combined_score": np.where(starts >= 100., 3., 0.)}
    monkeypatch.setattr(module, "score_events", fake_score)
    output = tmp_path / "released"
    summary = module.run_campaign(output_dir=output, cache_dir=tmp_path / "unused", max_epochs=1, device="cpu")
    assert actions[:8] == ["fit"] * 4 + ["checkpoint"] * 4
    assert actions[8:] == [item for _ in module.SCENARIOS for item in ("load", "score")]
    assert list(summary["folds"]) == list(module.SCENARIOS)
    assert (output / "summary.json").is_file()
    for name in module.SCENARIOS:
        fold = output / name.lower()
        assert (fold / "model.pt").is_file() and (fold / "scores.csv").is_file() and (fold / "fold.json").is_file()
