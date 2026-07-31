#!/usr/bin/env python3
"""Leakage-safe four-fold TEXBAT DS1--DS4 LOSO GCMR experiment.

This is a within-TEXBAT experiment, not external validation.  Every fold uses
only the other three scenarios' clean prefixes for all fitted artifacts.  All
four fold models are trained and checkpointed before any held scenario is
scored, and results are atomically released only when the campaign completes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from gnss_doppler_lab.gcmr_experiment import (
    calibration_threshold, implementation_manifest, load_checkpoint, load_event_cache,
    save_checkpoint, save_score_csv, score_events, train_clean_model,
)
from gnss_doppler_lab.gcmr_model import CleanReferenceScoreCalibrator

SCENARIOS = ("DS1", "DS2", "DS3", "DS4")
CACHE_DIR = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/gcmr-texbat-event-cache-v1")
DEFAULT_OUTPUT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/gcmr-texbat-loso-v1")
CACHE_SHA256 = {
    "DS1": "8ddf6d7a6b70d4c7497ccb75ca0844a50fd6eeebd733ef030dbb11ce1bbbcfef",
    "DS2": "c18201693e4240247b06647e5fae5eb7e51cf82a703018b838c03e678880a24d",
    "DS3": "bc1ded77e5ce3d74262338980ea3668a4f2ba8eaf5c8bcbb25a4ea80821a35ed",
    "DS4": "518be3f9154b52f58a777351f97f81d3a3e3998d6a3343f9cae6791d129c576f",
}
ROLE_INTERVALS = OrderedDict((
    ("train", (30., 54.)),
    ("validation", (55., 66.)),
    ("clean_reference", (67., 78.)),
    ("calibration", (79., 90.)),
))
EXPECTED_ROLE_COUNTS = {name: count for name, count in zip(ROLE_INTERVALS, (47, 21, 21, 21))}
EXPECTED_HELD_COUNTS = {
    "DS1": {"stable_pre": 119, "transition": 39, "stable_post": 702},
    "DS2": {"stable_pre": 119, "transition": 39, "stable_post": 693},
    "DS3": {"stable_pre": 119, "transition": 39, "stable_post": 694},
    "DS4": {"stable_pre": 119, "transition": 39, "stable_post": 35},
}
TRAINING_CONFIG = {
    "seed": 7, "max_epochs": 40, "patience": 6,
    "learning_rate": 1e-3, "compactness_weight": .01,
    "warmup_epochs": 5, "pair_hidden": 32, "event_hidden": 64,
    "latent_dim": 32,
}
THRESHOLD_QUANTILE = .99
PRIMARY_ONSET_S = 100.
SHARED_CONTEXT_LIMITATION = (
    "Donor scenarios are distinct bytes/events but belong to the same TEXBAT "
    "static-reference cohort with correlated derived features; folds are dependent "
    "and do not prove independent-environment generalization."
)


class CacheRecord(NamedTuple):
    events: list
    metadata: dict
    path: Path
    sha256: str


class FittedFold(NamedTuple):
    training: object
    calibrator: object
    threshold: float


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def _external_adapter():
    path = Path(__file__).with_name("run_gcmr_texbat_external.py")
    spec = importlib.util.spec_from_file_location("gcmr_texbat_external_for_loso", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def donors_for(held):
    if held not in SCENARIOS: raise ValueError(f"unknown held scenario: {held}")
    return tuple(x for x in SCENARIOS if x != held)


def split_donor_roles(events):
    events = list(events)
    roles = OrderedDict()
    for name, (start, end) in ROLE_INTERVALS.items():
        roles[name] = [e for e in events if e.window_start_s >= start and e.window_end_s <= end]
    validate_role_support(roles)
    return roles


def validate_role_support(roles, *, enforce_counts=True):
    if list(roles) != list(ROLE_INTERVALS):
        raise ValueError("role order/names differ from the frozen role contract")
    for name, events in roles.items():
        lo, hi = ROLE_INTERVALS[name]
        if any(e.window_start_s < lo or e.window_end_s > hi for e in events):
            raise ValueError(f"{name} event is not wholly contained")
        if enforce_counts and len(events) != EXPECTED_ROLE_COUNTS[name]:
            raise ValueError(f"{name} expected {EXPECTED_ROLE_COUNTS[name]} events, got {len(events)}")
    names = list(roles)
    for left, right in zip(names, names[1:]):
        if roles[left] and roles[right]:
            left_end = max(e.window_end_s for e in roles[left])
            right_start = min(e.window_start_s for e in roles[right])
            if left_end > right_start:
                raise ValueError(f"raw support overlap between {left} and {right}")
    return roles


def assert_no_held_contamination(held, role_sources):
    expected = list(donors_for(held))
    for role in ROLE_INTERVALS:
        sources = list(role_sources.get(role, ()))
        if sources != expected or held in sources:
            raise ValueError(f"held scenario contamination in {role}: {sources}")
    return True


def prepare_fold_roles(events_by_scenario, held):
    donors = donors_for(held)
    if set(events_by_scenario) != set(SCENARIOS):
        raise ValueError("exactly DS1--DS4 event collections are required")
    by_donor = {name: split_donor_roles(events_by_scenario[name]) for name in donors}
    roles = OrderedDict((role, [e for donor in donors for e in by_donor[donor][role]]) for role in ROLE_INTERVALS)
    sources = {role: list(donors) for role in ROLE_INTERVALS}
    assert_no_held_contamination(held, sources)
    for role, events in roles.items():
        expected = EXPECTED_ROLE_COUNTS[role] * 3
        if len(events) != expected: raise ValueError(f"fold {held} {role}: expected {expected}, got {len(events)}")
    return roles, sources


def split_held_regions(events):
    events = list(events)
    return OrderedDict((
        ("stable_pre", [e for e in events if e.window_start_s >= 30. and e.window_end_s <= 90.]),
        ("transition", [e for e in events if e.window_start_s >= 90. and e.window_end_s <= 110.]),
        ("stable_post", [e for e in events if e.window_start_s >= 110.]),
    ))


def validate_held_counts(scenario, regions):
    if scenario not in EXPECTED_HELD_COUNTS: raise ValueError(f"unknown scenario: {scenario}")
    actual = {name: len(regions[name]) for name in EXPECTED_HELD_COUNTS[scenario]}
    if actual != EXPECTED_HELD_COUNTS[scenario]:
        raise ValueError(f"{scenario} held region counts differ: {actual}")
    return actual


def _read_cache_metadata(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            return json.loads(str(data["metadata_json"]))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"incompatible event cache metadata: {exc}") from exc


def load_pinned_event_cache(path, scenario, *, expected_sha256, loader=load_event_cache,
                            adapter_validator=None):
    path = Path(path)
    actual_sha = _sha(path)
    if actual_sha != expected_sha256:
        raise ValueError(f"{scenario} cache SHA256 mismatch: {actual_sha}")
    metadata = _read_cache_metadata(path)
    if metadata.get("scenario") != scenario:
        raise ValueError(f"cache metadata scenario mismatch for {scenario}")
    source_map = metadata.get("source_sha256")
    if not isinstance(source_map, dict) or not source_map:
        raise ValueError("cache metadata source hashes missing")
    validator = adapter_validator
    if validator is None: validator = _external_adapter()._validate_adapter_cache_provenance
    validator(metadata)
    generated = {"schema_version", "relation_contract_version", "observation_features", "condition_features", "source_sha256"}
    original_metadata = {key: value for key, value in metadata.items() if key not in generated}
    source_paths = [Path(x) for x in sorted(source_map)]
    events, validated_metadata = loader(path, source_paths=source_paths, expected_metadata=original_metadata)
    if validated_metadata != metadata:
        raise ValueError("validated cache metadata differs from original metadata")
    if _sha(path) != actual_sha:
        raise ValueError(f"{scenario} immutable cache changed during load")
    return CacheRecord(list(events), metadata, path.resolve(), actual_sha)


def load_all_caches(cache_dir=CACHE_DIR):
    root = Path(cache_dir)
    return {name: load_pinned_event_cache(root / f"{name.lower()}.relations.npz", name,
            expected_sha256=CACHE_SHA256[name]) for name in SCENARIOS}


def fit_fold(roles, *, device=None, max_epochs=40):
    if not 1 <= int(max_epochs) <= TRAINING_CONFIG["max_epochs"]:
        raise ValueError("max_epochs must be in [1, 40]")
    kwargs = {**TRAINING_CONFIG, "max_epochs": int(max_epochs), "device": device}
    training = train_clean_model(roles["train"], roles["validation"], **kwargs)
    reference = score_events(training.model, roles["clean_reference"], device=device)
    calibrator = CleanReferenceScoreCalibrator().fit(reference["reconstruction"], reference["latent"])
    calibration = score_events(training.model, roles["calibration"], calibrator, device=device)
    threshold = calibration_threshold(calibration["combined_score"], quantile=THRESHOLD_QUANTILE)
    return FittedFold(training, calibrator, threshold)


class LosoCampaignGate:
    def __init__(self):
        self.training_order = []
        self.frozen = False
        self.evaluation_order = []

    def mark_trained(self, held):
        if self.frozen: raise RuntimeError("models are already frozen")
        expected = SCENARIOS[len(self.training_order)] if len(self.training_order) < len(SCENARIOS) else None
        if held != expected: raise RuntimeError(f"training must follow frozen fold order; expected {expected}")
        self.training_order.append(held)

    def freeze_all(self):
        if tuple(self.training_order) != SCENARIOS:
            raise RuntimeError("all four models must be trained before freeze")
        self.frozen = True

    def begin_evaluation(self, held):
        if not self.frozen: raise RuntimeError("all models must be frozen before held evaluation")
        if held in self.evaluation_order: raise RuntimeError("duplicate held evaluation")
        expected = SCENARIOS[len(self.evaluation_order)]
        if held != expected: raise RuntimeError(f"evaluation must follow frozen fold order; expected {expected}")
        self.evaluation_order.append(held)


def score_metrics(scored, threshold, mask=None):
    scores = np.asarray(scored["combined_score"], float)
    times = np.asarray(scored["availability_s"], float)
    if mask is not None: scores, times = scores[mask], times[mask]
    alarm = scores > threshold
    first = float(times[np.flatnonzero(alarm)[0]]) if alarm.any() else None
    return {
        "event_count": int(len(scores)), "alarm_count": int(alarm.sum()),
        "alarm_rate": float(alarm.mean()) if len(scores) else None,
        "score_median": float(np.median(scores)) if len(scores) else None,
        "score_q99": float(np.quantile(scores, .99)) if len(scores) else None,
        "first_alarm_available_s": first,
    }


def onset_latency(scored, threshold):
    starts = np.asarray(scored["window_start_s"], float)
    ends = np.asarray(scored["window_end_s"], float)
    scores = np.asarray(scored["combined_score"], float)
    eligible_alarm = (starts >= PRIMARY_ONSET_S) & (scores > threshold)
    if not eligible_alarm.any():
        return {"first_alarm_available_s": None, "first_alarm_delay_s": None}
    first = float(ends[np.flatnonzero(eligible_alarm)[0]])
    return {"first_alarm_available_s": first, "first_alarm_delay_s": first - PRIMARY_ONSET_S}


def _score_region_masks(scored):
    start = np.asarray(scored["window_start_s"], float)
    end = np.asarray(scored["window_end_s"], float)
    return OrderedDict((
        ("stable_pre", (start >= 30.) & (end <= 90.)),
        ("transition", (start >= 90.) & (end <= 110.)),
        ("stable_post", start >= 110.),
    ))


def fold_metrics(scenario, scored, threshold):
    result = {name: score_metrics(scored, threshold, mask) for name, mask in _score_region_masks(scored).items()}
    result.update(onset_latency(scored, threshold))
    start = np.asarray(scored["window_start_s"], float)
    post120 = score_metrics(scored, threshold, start >= 120.)
    post120["classification"] = "secondary_sensitivity_only"
    result["post120_sensitivity"] = post120
    if scenario == "DS4":
        result["onset_conflict"] = {
            "primary_onset_s": 100., "auxiliary_onset_s": 110.,
            "resolution": "primary TEXBAT contract retained; post>=120 is secondary sensitivity only",
        }
    return result


def _git(args):
    try:
        return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    except Exception:
        return "unavailable"


def campaign_provenance(records):
    return {
        "runner": "scripts/run_gcmr_texbat_loso.py",
        "runner_sha256": _sha(Path(__file__).resolve()),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_status": _git(["status", "--short"]),
        "implementation": implementation_manifest(),
        # Retain the runner-facing manifest name for external audit consumers.
        "core_implementation_manifest": implementation_manifest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                        "platform": platform.platform()},
        "cache_sha256": {name: records[name].sha256 for name in SCENARIOS},
        "original_cache_metadata": {name: records[name].metadata for name in SCENARIOS},
        "fixed_training_config": TRAINING_CONFIG,
        "threshold": {"quantile": THRESHOLD_QUANTILE, "comparison": "score > threshold"},
    }


def build_campaign_summary(folds, *, provenance):
    return {
        "evaluation_classification": "within_texbat_loso_experiment",
        "external_validation": False,
        "shared_context_limitation": SHARED_CONTEXT_LIMITATION,
        "event_level_confidence_intervals": "not reported",
        "fold_order": list(SCENARIOS), "folds": folds, "provenance": provenance,
    }


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=_jsonable) + "\n")


def run_campaign(*, output_dir=DEFAULT_OUTPUT, cache_dir=CACHE_DIR, max_epochs=40, device=None):
    output = Path(output_dir).resolve()
    if output.exists(): raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        records = load_all_caches(cache_dir)
        events = {name: records[name].events for name in SCENARIOS}
        provenance = campaign_provenance(records)
        gate = LosoCampaignGate(); fitted = {}; role_info = {}; training_info = {}
        # No held-region selection, scoring, metrics, or result files occur in this loop.
        for held in SCENARIOS:
            roles, sources = prepare_fold_roles(events, held)
            fitted[held] = fit_fold(roles, device=device, max_epochs=max_epochs)
            role_info[held] = {"sources": sources, "counts": {k: len(v) for k, v in roles.items()}}
            training_info[held] = {
                "best_epoch": fitted[held].training.best_epoch,
                "history": fitted[held].training.history,
            }
            gate.mark_trained(held)
        # Persist every model before any held scenario is evaluated.
        fold_provenance = {}
        for held in SCENARIOS:
            fold_dir = staging / held.lower(); fold_dir.mkdir(parents=True)
            expected_provenance = {
                **provenance, "held_scenario": held, "donors": list(donors_for(held)),
                "role_contract": {k: list(v) for k, v in ROLE_INTERVALS.items()},
                "role_sources_and_counts": role_info[held],
                "leakage_contract": {
                    "held_excluded_from": ["scaler", "training", "validation_epoch_selection", "latent_clean_reference", "calibration", "threshold", "seed_and_hyperparameter_choice"],
                    "scaler_model_fit": "donor train only", "epoch_selection": "donor validation only",
                    "score_calibrator_fit": "donor clean_reference only", "threshold_q99": "donor calibration only",
                },
            }
            fit = fitted[held]
            save_checkpoint(fold_dir / "model.pt", fit.training, fit.calibrator, fit.threshold,
                            provenance=expected_provenance)
            fold_provenance[held] = expected_provenance
            del fitted[held]
        if fitted:
            raise RuntimeError("in-memory folds survived checkpoint persistence")
        gate.freeze_all()
        fold_results = {}
        for held in SCENARIOS:
            gate.begin_evaluation(held)
            regions = split_held_regions(events[held]); counts = validate_held_counts(held, regions)
            held_events = [e for name in ("stable_pre", "transition", "stable_post") for e in regions[name]]
            fold_dir = staging / held.lower()
            expected_provenance = fold_provenance[held]
            restored = load_checkpoint(fold_dir / "model.pt",
                                       expected_provenance=expected_provenance, device=device or "cpu")
            if restored.provenance != expected_provenance:
                raise ValueError(f"{held} loaded checkpoint provenance differs from saved provenance")
            if (restored.provenance.get("held_scenario") != held
                    or restored.provenance.get("donors") != list(donors_for(held))):
                raise ValueError(f"{held} loaded checkpoint fold identity mismatch")
            scored = score_events(restored.model, held_events, restored.calibrator, device=device)
            save_score_csv(fold_dir / "scores.csv", scored, restored.threshold)
            metrics = fold_metrics(held, scored, restored.threshold)
            checkpoint_sha = _sha(fold_dir / "model.pt")
            result = {
                "held_scenario": held, "donors": list(donors_for(held)),
                "held_region_event_counts": counts, "threshold": restored.threshold,
                "threshold_source": "donor calibration role only q99",
                "best_epoch": restored.best_epoch, "training_history": training_info[held]["history"],
                "loaded_checkpoint_provenance": restored.provenance,
                "model_checkpoint_sha256": checkpoint_sha, "metrics": metrics,
            }
            _write_json(fold_dir / "fold.json", result); fold_results[held] = result
        # Fail closed if any supposedly immutable input changed during the campaign.
        for name in SCENARIOS:
            if _sha(records[name].path) != records[name].sha256:
                raise ValueError(f"{name} immutable cache changed during campaign")
        summary = build_campaign_summary(fold_results, provenance=provenance)
        _write_json(staging / "summary.json", summary)
        staging.rename(output)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    summary = run_campaign(output_dir=args.output_dir, cache_dir=args.cache_dir,
                           max_epochs=args.max_epochs, device=args.device)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "folds": list(summary["folds"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
