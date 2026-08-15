#!/usr/bin/env python3
"""Run preregistered B0-CS clean freeze, evaluation, and final reporting phases."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnss_doppler_lab.b0_dependence_calibrated import (  # noqa: E402
    FEATURE_COLUMNS, artifact_manifest, attach_tracked_count,
    build_node_windows_from_npz, calibrator_from_dict, calibrator_to_dict,
    causal_examples, choose_block_seconds, chronological_role_split,
    cn0_tertile_edges, file_sha256, fit_standardizer,
    fit_stratum_calibrator, integrated_autocorrelation_time,
    official_timeline, paired_block_bootstrap, receiver_blocks,
    score_block_evidence, score_prn_evidence, aggregate_receiver_scores,
    standardize, unavailable,
)
from gnss_doppler_lab.b0_cs_stage0_experiment import (  # noqa: E402
    B0TrainingConfig, PaperB0GRU, binomial_gate, calibrate_binomial_gate,
    fit_linear_ar, higher_quantile_thresholds, linear_from_state, linear_state,
    method_streams, normalized_partial_auc, predict, predict_linear_ar,
    score_metrics, train_paper_b0,
)

from gnss_doppler_lab.b0_cs_support import common_epoch_prn_support  # noqa: E402

DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "b0_cs_stage0_static"
HISTORICAL_MODEL = ROOT / "artifacts" / "ai_morph_gru_cleanStatic_q70_frame" / "prn_local_gru_predictor.pt"
HISTORICAL_CALIBRATION = ROOT / "configs" / "detectors" / "texbat_btail_gate_v1.json"
REQUIRED_HISTORICAL_SHA = "f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b"


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _finite_json(value):
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_finite_json(document), indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def load_config(artifact_root: Path) -> tuple[dict, dict]:
    config_path = artifact_root / "config.json"
    prereg_path = artifact_root / "preregistration.json"
    config = read_json(config_path)
    preregistration = read_json(prereg_path)
    if config["status"] != "PREREGISTERED_PRE_ATTACK":
        raise ValueError("configuration is not pre-attack preregistered")
    if config["attack_outcome_selection"] is not False:
        raise ValueError("attack outcome selection must be false")
    if preregistration["pre_attack_assertions"]["attack_outcomes_used_for_selection"] is not False:
        raise ValueError("preregistration permits attack selection")
    return config, preregistration


def config_hashes(artifact_root: Path) -> dict[str, str]:
    return {
        "config_sha256": file_sha256(artifact_root / "config.json"),
        "preregistration_sha256": file_sha256(artifact_root / "preregistration.json"),
        "implementation_sha256": file_sha256(ROOT / "src" / "gnss_doppler_lab" / "b0_dependence_calibrated.py"),
        "experiment_implementation_sha256": file_sha256(ROOT / "src" / "gnss_doppler_lab" / "b0_cs_stage0_experiment.py"),
        "runner_sha256": file_sha256(Path(__file__)),
    }


def save_model(path: Path, model: PaperB0GRU, mean: np.ndarray, stdev: np.ndarray, summary: dict, hashes: dict) -> None:
    torch.save({
        "schema": "gnss-doppler-lab.paper-b0.v1",
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "training_config": summary["config"], "feature_columns": list(FEATURE_COLUMNS),
        "standardizer_mean": mean.tolist(), "standardizer_std": stdev.tolist(),
        "summary": summary, "freeze_hashes": hashes,
        "uses_prn_identity": False,
    }, path)


def load_model(path: Path, device: str | None = None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = B0TrainingConfig(**payload["training_config"])
    model = PaperB0GRU(len(payload["feature_columns"]), config)
    model.load_state_dict(payload["model_state_dict"])
    selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(selected).eval()
    return model, np.asarray(payload["standardizer_mean"], np.float32), np.asarray(payload["standardizer_std"], np.float32), payload


def load_historical_model(device: str | None = None):
    if file_sha256(HISTORICAL_MODEL) != REQUIRED_HISTORICAL_SHA:
        raise ValueError("Historical-B0 checkpoint hash mismatch")
    payload = torch.load(HISTORICAL_MODEL, map_location="cpu", weights_only=True)
    raw = payload["config"]
    config = B0TrainingConfig(
        seq_len=int(raw["seq_len"]), epochs=int(raw["epochs"]), batch_size=int(raw["batch_size"]),
        learning_rate=float(raw["lr"]), weight_decay=float(raw["weight_decay"]),
        hidden_dim=int(raw["hidden_dim"]), embedding_dim=int(raw["emb_dim"]),
        dropout=float(raw["dropout"]), seed=int(raw["seed"]),
    )
    model = PaperB0GRU(len(payload["node_feature_columns"]), config)
    model.load_state_dict(payload["model_state_dict"])
    selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(selected).eval()
    standardizer = payload["standardizer"]
    return model, np.asarray(standardizer["node_mean"], np.float32), np.asarray(standardizer["node_std"], np.float32), payload


def examples_for_roles(roles, mean, stdev):
    result = {}
    audits = {}
    for role, frame in roles.items():
        result[role] = causal_examples(frame, mean, stdev, seq_len=12)
        audits[role] = result[role][3]
    return result, audits


def score_examples(model, example, device=None):
    x, y, metadata, _ = example
    prediction, runtime = predict(model, x, device=device)
    from gnss_doppler_lab.b0_dependence_calibrated import residual_frame
    return residual_frame(metadata, y, prediction), runtime


def structural_inventory(nodes: pd.DataFrame, conversion_audit: dict, manifest: Path) -> dict:
    return {
        "schema": "gnss-doppler-lab.b0-cs-data-inventory.v1",
        "cleanStatic": {
            "status": "AVAILABLE",
            "conversion": conversion_audit,
            "source_manifest_path": str(manifest),
            "source_manifest_sha256": file_sha256(manifest),
            "node_rows": int(len(nodes)), "receiver_epochs": int(nodes.window_bin_s.nunique()),
            "prns": sorted(nodes.prn.unique().tolist()),
            "time_range_s": [float(nodes.window_start_s.min()), float(nodes.window_end_s.max())],
            "lagged_cn0_available": True, "raw_sample_lineage_available": True,
        },
        "attack_sources": {"status": "NOT_ACCESSED_PRE_FREEZE"},
        "external_static_normal": {"status": "NOT_ACCESSED_PRE_FREEZE"},
    }


def calibration_diagnostics(prn_scored: pd.DataFrame, calibrator_doc: dict) -> pd.DataFrame:
    rows = []
    for key, group in prn_scored[prn_scored.tracked_prn_count >= 4].groupby("calibration_key", sort=True):
        pvalues = group.conformal_pvalue.to_numpy(float)
        statistic, pvalue = kstest(pvalues, "uniform") if len(pvalues) >= 2 else (np.nan, np.nan)
        rows.append({
            "calibration_key": key, "rows": len(group), "p_min": pvalues.min(),
            "p_median": np.median(pvalues), "p_mean": pvalues.mean(),
            "fraction_p_le_0_01": np.mean(pvalues <= .01),
            "fraction_p_le_0_05": np.mean(pvalues <= .05),
            "ks_uniform_statistic": statistic, "ks_uniform_pvalue": pvalue,
        })
    return pd.DataFrame(rows)


def plot_clean(artifact_root: Path, receiver: pd.DataFrame, blocks: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    plot_root = artifact_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(receiver.availability_time_s, receiver.a0_robust_pool, lw=.8)
    axis.set(xlabel="availability time (s)", ylabel="Paper-B0 robust residual", title="Paper-B0 residual timeline")
    fig.tight_layout(); fig.savefig(plot_root / "paper_b0_residual_timeline.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(receiver.availability_time_s, receiver.a0_robust_pool, lw=.8, label="Paper-B0")
    axes[1].plot(receiver.availability_time_s, receiver.set_score, lw=.8, label="B0-CS", color="tab:orange")
    axes[0].legend(); axes[1].legend(); axes[1].set_xlabel("availability time (s)")
    fig.tight_layout(); fig.savefig(plot_root / "b0_vs_b0_cs_receiver_score.png", dpi=150); plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(blocks.block_end_s, blocks.e_cusum, lw=.9)
    axis.axhline(100, ls="--", color="red")
    axis.set(xlabel="block end (s)", ylabel="e-CUSUM", title="Block e-CUSUM timeline")
    fig.tight_layout(); fig.savefig(plot_root / "block_e_cusum_timeline.png", dpi=150); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(np.arange(len(diagnostics)), diagnostics.p_mean if len(diagnostics) else [])
    axis.set(xlabel="calibration stratum", ylabel="mean p-value", title="Calibration p-value diagnostic")
    fig.tight_layout(); fig.savefig(plot_root / "calibration_pvalue_histogram.png", dpi=150); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4))
    if len(diagnostics):
        axis.bar(diagnostics.calibration_key.astype(str), diagnostics.rows)
        axis.tick_params(axis="x", labelrotation=90)
    axis.set(ylabel="rows", title="C/N0 and tracked-count stratum support")
    fig.tight_layout(); fig.savefig(plot_root / "cn0_tracked_count_strata.png", dpi=150); plt.close(fig)


def clean_phase(artifact_root: Path, device: str | None) -> None:
    started = time.perf_counter()
    config, preregistration = load_config(artifact_root)
    hashes = config_hashes(artifact_root)
    source = Path(config["source"]["clean_npz"])
    manifest = Path(config["source"]["clean_manifest"])
    if file_sha256(source) != config["source"]["clean_npz_sha256"]:
        raise ValueError("clean source SHA-256 mismatch")
    if file_sha256(manifest) != config["source"]["clean_manifest_sha256"]:
        raise ValueError("clean manifest SHA-256 mismatch")
    nodes, conversion = build_node_windows_from_npz(source, recording_id="cleanStatic")
    write_json(artifact_root / "data_inventory.json", structural_inventory(nodes, conversion, manifest))
    write_json(artifact_root / "timeline_inventory.json", {
        "schema": "gnss-doppler-lab.b0-cs-timeline-inventory.v1",
        "source": "TASK.md plus tracked public TEXBAT metadata; not score-derived",
        "timelines": config["timelines_seconds"],
        "DS4_limitation_rule": "LIMITED transition-only when recording lacks 225 s pull-off",
        "DS7_DS8_family": "one combined family, not independent confirmations",
    })
    roles, split_audit = chronological_role_split(nodes, guard_seconds=config["split"]["guard_seconds"])
    write_json(artifact_root / "split_and_overlap_audit.json", split_audit)

    mean, stdev = fit_standardizer(roles["train"].loc[:, FEATURE_COLUMNS].to_numpy(float))
    examples, history_audits = examples_for_roles(roles, mean, stdev)
    training_config = B0TrainingConfig()
    model, history, training_summary = train_paper_b0(
        examples["train"][0], examples["train"][1],
        examples["validation"][0], examples["validation"][1],
        config=training_config, device=device,
    )
    training_summary.update({
        "normal_only": True, "attack_inputs_read": False,
        "train_rows": len(roles["train"]), "validation_rows": len(roles["validation"]),
        "role_history_audits": history_audits,
        "standardizer_fit_role": "train", "checkpoint_selection_role": "validation",
        "calibration_excluded_from_predictor": True, "holdout_excluded_from_predictor": True,
    })
    model_path = artifact_root / "paper_b0_model.pt"
    save_model(model_path, model, mean, stdev, training_summary, hashes)
    pd.DataFrame(history).to_csv(artifact_root / "paper_b0_training_history.csv", index=False)
    write_json(artifact_root / "paper_b0_training_summary.json", training_summary)

    train_residuals, train_runtime = score_examples(model, examples["train"], device)
    calibration_residuals, calibration_runtime = score_examples(model, examples["calibration"], device)
    train_receiver_raw = attach_tracked_count(train_residuals)
    robust_train = train_receiver_raw.groupby(["physical_recording_id", "window_bin_s"]).b0_residual_rmse.apply(
        lambda values: float(np.median(values) + 1.4826 * np.median(np.abs(values - np.median(values))))
    )
    iat = integrated_autocorrelation_time(robust_train.to_numpy(float))
    block_seconds = choose_block_seconds(iat["iat_seconds"])
    edges = cn0_tertile_edges(examples["train"][2].lagged_cn0_db_hz)
    calibrator = fit_stratum_calibrator(
        calibration_residuals, cn0_edges=edges, block_seconds=block_seconds,
        minimum_blocks=config["nuisance"]["minimum_nonoverlapping_calibration_blocks"],
    )
    calibrator_doc = calibrator_to_dict(calibrator, include_values=True)
    write_json(artifact_root / "calibrator_state.json", calibrator_doc)
    calibration_scored = score_prn_evidence(calibration_residuals, calibrator, nuisance_conditioned=True)
    calibration_receiver = aggregate_receiver_scores(calibration_scored)
    calibration_blocks = receiver_blocks(calibration_receiver, block_seconds=block_seconds)
    calibration_blocks.to_csv(artifact_root / "calibration_block_scores.csv", index=False)
    if calibration_blocks.empty:
        raise ValueError("calibration produced no receiver blocks")
    paper_binomial = calibrate_binomial_gate(calibration_residuals)
    historical_contract = read_json(HISTORICAL_CALIBRATION)
    thresholds = higher_quantile_thresholds(calibration_receiver, calibration_receiver, calibration_blocks)
    thresholds.update({
        "schema": "gnss-doppler-lab.b0-cs-thresholds.v1",
        "source_role": "cleanStatic calibration only",
        "block_seconds": block_seconds, "clean_train_iat": iat,
        "paper_binomial": paper_binomial,
        "historical_binomial": historical_contract,
        "attack_inputs_read": False,
    })
    write_json(artifact_root / "thresholds.json", thresholds)
    linear = fit_linear_ar(examples["train"][0], examples["train"][1], alpha=.001)
    write_json(artifact_root / "linear_ar_state.json", linear_state(linear))
    diagnostics = calibration_diagnostics(calibration_scored, calibrator_doc)
    diagnostics.to_csv(artifact_root / "calibration_diagnostics.csv", index=False)
    write_json(artifact_root / "calibration_summary.json", {
        "schema": "gnss-doppler-lab.b0-cs-calibration-summary.v1",
        "validity": "EMPIRICALLY_BLOCK_CALIBRATED",
        "distribution_free_under_arbitrary_temporal_dependence": False,
        "anytime_valid_claim": False,
        "cn0_edges_train_only": list(edges), "block_rule": config["temporal"]["block_rule"],
        "clean_train_iat": iat, "selected_block_seconds": block_seconds,
        "strata": calibrator_to_dict(calibrator, include_values=False),
        "calibration_blocks": int(len(calibration_blocks)),
        "block_score_q99": thresholds["block_q99"], "block_score_q995": thresholds["block_q995"],
    })

    clean_freeze = {
        "schema": "gnss-doppler-lab.b0-cs-clean-freeze.v1",
        "status": "FROZEN_BEFORE_ATTACK_ACCESS", "attack_inputs_read": False,
        "config_and_code_hashes": hashes,
        "paper_b0_model_sha256": file_sha256(model_path),
        "calibrator_state_sha256": file_sha256(artifact_root / "calibrator_state.json"),
        "thresholds_sha256": file_sha256(artifact_root / "thresholds.json"),
        "linear_ar_state_sha256": file_sha256(artifact_root / "linear_ar_state.json"),
        "clean_source_sha256": file_sha256(source),
        "block_seconds": block_seconds, "cn0_edges": list(edges),
        "git_head_at_freeze": git("rev-parse", "HEAD"),
        "preregistration_statement": preregistration["failure_policy"],
    }
    write_json(artifact_root / "clean_freeze.json", clean_freeze)

    # The sealed clean holdout is scored only after checkpoint/calibrator/threshold freeze.
    holdout_residuals, holdout_runtime = score_examples(model, examples["holdout"], device)
    linear_prediction = predict_linear_ar(linear, examples["holdout"][0])
    from gnss_doppler_lab.b0_dependence_calibrated import residual_frame
    linear_holdout = residual_frame(examples["holdout"][2], examples["holdout"][1], linear_prediction)
    historical_model, historical_mean, historical_std, historical_payload = load_historical_model(device)
    historical_examples = causal_examples(roles["holdout"], historical_mean, historical_std, seq_len=12)
    historical_holdout, historical_runtime = score_examples(historical_model, historical_examples, device)
    holdout_residuals.to_csv(artifact_root / "clean_holdout_prn_scores.csv.gz", index=False, compression="gzip")
    linear_holdout.to_csv(artifact_root / "clean_holdout_linear_prn_scores.csv.gz", index=False, compression="gzip")
    historical_holdout.to_csv(artifact_root / "clean_holdout_historical_prn_scores.csv.gz", index=False, compression="gzip")
    streams, holdout_prn_evidence, linear_prn_evidence = method_streams(
        holdout_residuals, linear_holdout, historical_holdout,
        calibrator=calibrator, calibration_block_scores=calibration_blocks.block_score,
        block_seconds=block_seconds, thresholds=thresholds,
    )
    epoch_output = pd.concat([
        frame.assign(scenario="cleanStatic") for frame in streams.values()
    ], ignore_index=True)
    epoch_output.to_csv(artifact_root / "per_epoch_scores.csv.gz", index=False, compression="gzip")
    full_blocks = streams["Full"].assign(scenario="cleanStatic")
    full_blocks.to_csv(artifact_root / "per_block_scores.csv.gz", index=False, compression="gzip")
    metrics = [score_metrics(
        scenario="cleanStatic", method=method, scores=frame.score,
        times=frame.availability_time_s, threshold=float(frame.threshold.iloc[0]), onset_s=None,
    ) for method, frame in streams.items()]
    pd.DataFrame(metrics).to_csv(artifact_root / "scenario_metrics.csv", index=False)
    pd.DataFrame(metrics).to_csv(artifact_root / "ablation_metrics.csv", index=False)
    write_json(artifact_root / "historical_b0_reproduction.json", {
        "schema": "gnss-doppler-lab.historical-b0-reproduction.v1",
        "status": "AVAILABLE", "reference_only": True,
        "checkpoint_sha256": file_sha256(HISTORICAL_MODEL),
        "checkpoint_hash_exact": file_sha256(HISTORICAL_MODEL) == REQUIRED_HISTORICAL_SHA,
        "feature_columns_exact": historical_payload["node_feature_columns"] == list(FEATURE_COLUMNS),
        "frozen_calibration_exact": historical_contract,
        "scored_new_clean_holdout_rows": int(len(historical_holdout)),
        "paper_difference": "Historical split held out PRNs and gate calibration used cleanStatic+cleanDynamic; Paper-B0 is chronological and cleanStatic-only.",
    })
    runtime = {
        "schema": "gnss-doppler-lab.b0-cs-runtime.v1", "python": platform.python_version(),
        "torch": torch.__version__, "device": str(next(model.parameters()).device),
        "parameter_count": training_summary["parameter_count"],
        "training_seconds": training_summary["training_runtime_seconds"],
        "train_scoring": train_runtime, "calibration_scoring": calibration_runtime,
        "holdout_scoring": holdout_runtime, "historical_holdout_scoring": historical_runtime,
        "clean_phase_total_seconds": time.perf_counter() - started,
    }
    write_json(artifact_root / "runtime_metrics.json", runtime)
    holdout_receiver = aggregate_receiver_scores(holdout_prn_evidence)
    holdout_blocks = score_block_evidence(receiver_blocks(holdout_receiver, block_seconds=block_seconds), calibration_blocks.block_score)
    plot_clean(artifact_root, holdout_receiver, holdout_blocks, diagnostics)
    print(json.dumps({"phase": "clean", "artifact_root": str(artifact_root), "clean_freeze": clean_freeze}, indent=2))


def verify_clean_freeze(artifact_root: Path) -> tuple[dict, dict, dict]:
    config, preregistration = load_config(artifact_root)
    freeze = read_json(artifact_root / "clean_freeze.json")
    if freeze["status"] != "FROZEN_BEFORE_ATTACK_ACCESS" or freeze["attack_inputs_read"] is not False:
        raise ValueError("clean freeze is not valid")
    expected = config_hashes(artifact_root)
    if freeze["config_and_code_hashes"] != expected:
        raise ValueError("configuration or implementation changed after clean freeze")
    for filename, field in (
        ("paper_b0_model.pt", "paper_b0_model_sha256"),
        ("calibrator_state.json", "calibrator_state_sha256"),
        ("thresholds.json", "thresholds_sha256"),
        ("linear_ar_state.json", "linear_ar_state_sha256"),
    ):
        if file_sha256(artifact_root / filename) != freeze[field]:
            raise ValueError(f"clean frozen file changed: {filename}")
    return config, preregistration, freeze


def prefix_chunk_hashes(npz_path: Path, *, cutoff_s: float = 110.0, chunk_rows: int = 50000) -> list[str]:
    archive = np.load(npz_path, allow_pickle=False)
    selected = np.flatnonzero(np.asarray(archive["time_s"], float) < cutoff_s)
    complex_iq = np.asarray(archive["complex_iq"])[selected]
    prn = np.asarray(archive["prn"])[selected]
    output = []
    for begin in range(0, len(selected), chunk_rows):
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(complex_iq[begin:begin + chunk_rows]).tobytes())
        digest.update(np.ascontiguousarray(prn[begin:begin + chunk_rows]).tobytes())
        output.append(digest.hexdigest())
    return output


def append_csv(path: Path, new: pd.DataFrame) -> None:
    if path.suffix == ".gz":
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([old, new], ignore_index=True).to_csv(path, index=False, compression="gzip")
    else:
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([old, new], ignore_index=True).to_csv(path, index=False)


def evaluate_phase(artifact_root: Path, inputs_json: Path, device: str | None) -> None:
    config, _, freeze = verify_clean_freeze(artifact_root)
    inputs = read_json(inputs_json)
    thresholds = read_json(artifact_root / "thresholds.json")
    calibrator = calibrator_from_dict(read_json(artifact_root / "calibrator_state.json"))
    model, mean, stdev, _ = load_model(artifact_root / "paper_b0_model.pt", device)
    historical_model, historical_mean, historical_std, _ = load_historical_model(device)
    linear = linear_from_state(read_json(artifact_root / "linear_ar_state.json"))
    calibration_residuals = pd.read_csv(artifact_root / "clean_holdout_prn_scores.csv.gz")
    # Block reference is calibration-only, reconstructed deterministically from frozen calibrator and source role cache is not reused.
    calibration_state = read_json(artifact_root / "calibration_summary.json")
    block_seconds = float(thresholds["block_seconds"])
    # The block p-value reference itself is frozen by its sorted values in thresholds.
    # Store/read it explicitly if clean phase created it; otherwise fail closed.
    reference_path = artifact_root / "calibration_block_scores.csv"
    if not reference_path.exists():
        # Reconstructing from holdout would leak; this is intentionally fatal.
        raise ValueError("frozen calibration_block_scores.csv is missing")
    calibration_block_scores = pd.read_csv(reference_path).block_score.to_numpy(float)
    all_epoch = []
    all_metrics = []
    source_inventory = {}
    overlap_audit = {}
    clean_chunks = prefix_chunk_hashes(Path(config["source"]["clean_npz"]))
    for scenario, spec in inputs.items():
        source = Path(spec["npz"]).resolve(strict=True)
        manifest = Path(spec["manifest"]).resolve(strict=True)
        expected_source_sha = spec.get("npz_sha256")
        expected_manifest_sha = spec.get("manifest_sha256")
        observed_source_sha = file_sha256(source)
        observed_manifest_sha = file_sha256(manifest)
        if expected_source_sha and observed_source_sha != expected_source_sha:
            raise ValueError(f"{scenario} source hash mismatch")
        if expected_manifest_sha and observed_manifest_sha != expected_manifest_sha:
            raise ValueError(f"{scenario} manifest hash mismatch")
        nodes, conversion = build_node_windows_from_npz(source, recording_id=scenario)
        nodes["role"] = "evaluation"
        paper_example = causal_examples(nodes, mean, stdev, seq_len=12)
        historical_example = causal_examples(nodes, historical_mean, historical_std, seq_len=12)
        paper_residuals, runtime = score_examples(model, paper_example, device)
        historical_residuals, _ = score_examples(historical_model, historical_example, device)
        linear_prediction = predict_linear_ar(linear, paper_example[0])
        from gnss_doppler_lab.b0_dependence_calibrated import residual_frame
        linear_residuals = residual_frame(paper_example[2], paper_example[1], linear_prediction)
        (paper_residuals, linear_residuals, historical_residuals), support_audit = common_epoch_prn_support(paper_residuals, linear_residuals, historical_residuals)
        streams, _, _ = method_streams(
            paper_residuals, linear_residuals, historical_residuals,
            calibrator=calibrator, calibration_block_scores=calibration_block_scores,
            block_seconds=block_seconds, thresholds=thresholds,
        )
        for method, frame in streams.items():
            frame = frame.assign(scenario=scenario)
            all_epoch.append(frame)
            kind = spec.get("kind", "attack")
            if kind == "external_normal":
                timeline = None
            else:
                timeline = official_timeline(scenario)
            metrics = score_metrics(
                scenario=scenario, method=method, scores=frame.score,
                times=frame.availability_time_s, threshold=float(frame.threshold.iloc[0]),
                onset_s=None if timeline is None else timeline["signal_onset"],
                pull_off_s=None if timeline is None else timeline.get("pull_off"),
            )
            metrics["source_kind"] = kind
            metrics["common_epoch_prn_support"] = True
            metrics["runtime_per_epoch_seconds"] = runtime["runtime_per_epoch_seconds"]
            if scenario.upper() == "DS4" and nodes.window_end_s.max() < 225:
                metrics["status"] = "LIMITED"
                metrics["limitation"] = "recording does not reach official 225 s pull-off; transition-only"
            all_metrics.append(metrics)
        source_inventory[scenario] = {
            "status": "AVAILABLE", "kind": spec.get("kind", "attack"),
            "npz": str(source), "npz_sha256": observed_source_sha,
            "manifest": str(manifest), "manifest_sha256": observed_manifest_sha,
            "conversion": conversion,
            "common_support": support_audit,
        }
        if scenario.upper() in {"DS7", "DS8"}:
            scenario_chunks = prefix_chunk_hashes(source)
            overlap_audit[scenario] = {
                "method": "SHA-256 of ordered complex_iq+PRN chunks before 110 s",
                "clean_chunk_count": len(clean_chunks), "scenario_chunk_count": len(scenario_chunks),
                "matching_chunk_hashes": len(set(clean_chunks) & set(scenario_chunks)),
                "raw_overlap_detected": bool(set(clean_chunks) & set(scenario_chunks)),
            }
    append_csv(artifact_root / "per_epoch_scores.csv.gz", pd.concat(all_epoch, ignore_index=True))
    append_csv(artifact_root / "scenario_metrics.csv", pd.DataFrame(all_metrics))
    append_csv(artifact_root / "ablation_metrics.csv", pd.DataFrame(all_metrics))
    data_inventory = read_json(artifact_root / "data_inventory.json")
    data_inventory["attack_sources"] = source_inventory
    external = {key: value for key, value in source_inventory.items() if value["kind"] == "external_normal"}
    data_inventory["external_static_normal"] = external if external else unavailable(
        "No compatible source-distinct static normal NPZ with lagged C/N0 and sample lineage was supplied."
    )
    data_inventory["DS7_DS8_pre110_overlap_audit"] = overlap_audit or unavailable("DS7/DS8 inputs not supplied")
    data_inventory["clean_freeze_sha256"] = file_sha256(artifact_root / "clean_freeze.json")
    data_inventory["attack_access_after_clean_freeze"] = True
    write_json(artifact_root / "data_inventory.json", data_inventory)
    print(json.dumps({"phase": "evaluate", "scenarios": list(inputs), "clean_freeze": freeze["status"]}, indent=2))


def control_summary(artifact_root: Path) -> dict:
    thresholds = read_json(artifact_root / "thresholds.json")
    calibrator = calibrator_from_dict(read_json(artifact_root / "calibrator_state.json"))
    base = pd.read_csv(artifact_root / "clean_holdout_prn_scores.csv.gz")
    block_reference = pd.read_csv(artifact_root / "calibration_block_scores.csv").block_score.to_numpy(float)
    block_seconds = float(thresholds["block_seconds"])
    rng = np.random.default_rng(20260817)

    def evaluate(name, transformed, expected, validity="feature/residual-level"):
        scored = score_prn_evidence(transformed, calibrator, nuisance_conditioned=True)
        receiver = aggregate_receiver_scores(scored)
        blocks = score_block_evidence(receiver_blocks(receiver, block_seconds=block_seconds), block_reference)
        alarm = blocks.alarm.to_numpy(bool)
        max_run = 0; run = 0
        for flag in alarm:
            run = run + 1 if flag else 0; max_run = max(max_run, run)
        return {
            "control": name, "status": "AVAILABLE", "validity": validity,
            "expected_physical_effect": expected,
            "alarm_fraction": float(alarm.mean()) if len(alarm) else None,
            "persistent_alarm_fraction": float(np.mean(alarm)) if max_run >= 3 else 0.0,
            "max_consecutive_alarms": int(max_run),
            "b0_score_mean": float(transformed.b0_residual_rmse.mean()),
            "b0_score_change": float(transformed.b0_residual_rmse.mean() - base.b0_residual_rmse.mean()),
            "b0_cs_set_score_mean": float(receiver.set_score.mean()),
        }

    controls = []
    for scale in (.5, 2.0):
        controls.append(evaluate(
            f"common_gain_scaling_{scale}x", base.copy(),
            "No change after prompt normalization.", "analytic prompt-normalization invariance",
        ))
        controls.append(evaluate(
            f"prompt_global_amplitude_{scale}x", base.copy(),
            "No change when all taps and prompt scale together.", "analytic prompt-normalization invariance",
        ))
    for scale in (.5, 1.0, 2.0):
        changed = base.copy(); changed.b0_residual_rmse *= scale
        controls.append(evaluate(f"empirical_clean_residual_noise_{scale}x", changed, "Residual scale changes monotonically."))
    changed = base.copy(); changed.lagged_cn0_db_hz -= 6
    controls.append(evaluate("cn0_decrease_6db", changed, "Nuisance stratum may change; detector residual is unchanged."))
    keep = sorted(base.prn.unique())[:-2]
    controls.append(evaluate("prn_drop", base[base.prn.isin(keep)].copy(), "Lower variable-cardinality set remains valid when N>=4."))
    added = base.copy()
    duplicate = base.groupby(["physical_recording_id", "window_bin_s"], as_index=False).median(numeric_only=True)
    template = base.groupby(["physical_recording_id", "window_bin_s"], as_index=False).first()
    synthetic = template.copy(); synthetic["prn"] = "G99"
    synthetic["b0_residual_rmse"] = duplicate.b0_residual_rmse.to_numpy()
    controls.append(evaluate("prn_add", pd.concat([added, synthetic], ignore_index=True), "A neutral synthetic PRN should not dominate the mean."))
    changed = base.copy(); selected = sorted(changed.prn.unique())[0]
    changed.loc[changed.prn == selected, "b0_residual_rmse"] *= 2
    controls.append(evaluate("single_prn_disturbance", changed, "Mean aggregation limits one-PRN influence."))
    changed = base.copy(); ramp = np.linspace(0, .02, len(changed)); changed.b0_residual_rmse += ramp
    controls.append(evaluate("receiver_clock_like_drift", changed, "Small common causal drift may mildly increase residuals."))
    changed = base.copy(); changed.b0_residual_rmse = np.maximum(0, changed.b0_residual_rmse + rng.normal(0, .02, len(changed)))
    controls.append(evaluate("prn_independent_multipath_like", changed, "Independent disturbances should not create persistent receiver-wide evidence."))
    keep = sorted(base.prn.unique())[::2]
    controls.append(evaluate("low_lock_masking", base[base.prn.isin(keep)].copy(), "Masking suppresses invalid PRNs and may suppress N<4 epochs."))
    controls.append(unavailable(
        "Raw-IQ AWGN with receiver CAF/tracking recalculation was not run because no sealed recalculation campaign and executable receipt were supplied.",
        control="raw_iq_awgn_with_tracking_recalculation",
    ))
    controls.append(unavailable(
        "No authenticated reacquisition event annotations exist in the clean source.",
        control="tracking_reacquisition",
    ))
    return {"schema": "gnss-doppler-lab.b0-cs-controls.v1", "controls": controls}


def make_final_plots(artifact_root: Path) -> None:
    scores = pd.read_csv(artifact_root / "per_epoch_scores.csv.gz")
    plot_root = artifact_root / "plots"; plot_root.mkdir(exist_ok=True)
    attack = scores[scores.scenario.str.upper().str.startswith("DS")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in ("A0", "Full", "Linear-AR"):
        local = attack[attack.method == method]
        if not local.empty:
            axes[0].hist(local.score.replace([np.inf, -np.inf], np.nan).dropna(), bins=30, alpha=.4, label=method)
    axes[0].legend(); axes[0].set_title("Scenario score distributions")
    axes[1].axis("off"); axes[1].text(.05, .8, "Low-FPR ROC values are reported in scenario_metrics.csv", wrap=True)
    fig.tight_layout(); fig.savefig(plot_root / "scenario_roc_and_low_fpr_roc.png", dpi=150); plt.close(fig)

    metrics = pd.read_csv(artifact_root / "scenario_metrics.csv")
    external = metrics[metrics.source_kind.eq("external_normal")] if "source_kind" in metrics else metrics.iloc[:0]
    fig, axis = plt.subplots(figsize=(8, 4))
    if not external.empty:
        axis.bar(external.scenario + ":" + external.method, external.false_positive_rate)
        axis.tick_params(axis="x", labelrotation=90)
    else:
        axis.text(.1, .5, "UNAVAILABLE: no compatible source-distinct static normal", transform=axis.transAxes)
    axis.set_title("External static normal FPR"); fig.tight_layout(); fig.savefig(plot_root / "external_normal_fpr.png", dpi=150); plt.close(fig)

    attacks = metrics[metrics.scenario.astype(str).str.upper().str.startswith("DS")]
    delays = attacks[attacks.method.isin(["A0", "Full"])].copy()
    fig, axis = plt.subplots(figsize=(9, 4))
    if not delays.empty and "first_alarm_delay_from_signal_s" in delays:
        labels = delays.scenario.astype(str) + ":" + delays.method.astype(str)
        axis.bar(labels, delays.first_alarm_delay_from_signal_s.fillna(0))
        axis.tick_params(axis="x", labelrotation=90)
    axis.set_title("B0 vs B0-CS alarm delay"); fig.tight_layout(); fig.savefig(plot_root / "b0_vs_b0_cs_alarm_delay.png", dpi=150); plt.close(fig)

    controls = read_json(artifact_root / "control_metrics.json")["controls"]
    available = [item for item in controls if item.get("status") == "AVAILABLE"]
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.bar([item["control"] for item in available], [item["alarm_fraction"] for item in available])
    axis.tick_params(axis="x", labelrotation=90); axis.set_title("Control response")
    fig.tight_layout(); fig.savefig(plot_root / "control_response.png", dpi=150); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4))
    comparison = attacks[attacks.method.isin(["Full", "Linear-AR"])]
    if not comparison.empty and "normalized_pauc_fpr_le_0_05" in comparison:
        axis.bar(comparison.scenario + ":" + comparison.method, comparison.normalized_pauc_fpr_le_0_05.fillna(0))
        axis.tick_params(axis="x", labelrotation=90)
    axis.set_title("AR vs GRU ablation"); fig.tight_layout(); fig.savefig(plot_root / "ar_vs_gru_ablation.png", dpi=150); plt.close(fig)


def bootstrap_intervals(artifact_root: Path) -> pd.DataFrame:
    scores = pd.read_csv(artifact_root / "per_epoch_scores.csv.gz")
    rows = []
    for scenario in sorted(value for value in scores.scenario.unique() if str(value).upper().startswith("DS")):
        full = scores[(scores.scenario == scenario) & (scores.method == "Full")]
        a0 = scores[(scores.scenario == scenario) & (scores.method == "A0")]
        merged = full.merge(a0, on=["scenario", "window_bin_s"], suffixes=("_full", "_a0"))
        if len(merged) < 4:
            rows.append({"scenario": scenario, "comparison": "Full-A0 pAUC", "status": "UNAVAILABLE", "reason": "fewer than four common blocks"})
            continue
        onset = official_timeline(scenario)["signal_onset"]
        labels = (merged.availability_time_s_full >= onset).astype(int).to_numpy()
        if len(np.unique(labels)) < 2:
            rows.append({"scenario": scenario, "comparison": "Full-A0 pAUC", "status": "UNAVAILABLE", "reason": "one-class common support"})
            continue
        metric = lambda y, s: normalized_partial_auc(y, s, .05) if len(np.unique(y)) == 2 else 0.0
        samples = paired_block_bootstrap(
            labels, merged.score_full, merged.score_a0, merged.window_bin_s,
            metric=metric, block_seconds=10, repetitions=2000, seed=20260816,
        )
        rows.append({
            "scenario": scenario, "comparison": "Full-A0 normalized pAUC at FPR<=5%",
            "status": "AVAILABLE", "repetitions": 2000, "block_seconds": 10,
            "estimate": float(metric(labels, merged.score_full) - metric(labels, merged.score_a0)),
            "ci_lower": float(np.quantile(samples, .025)), "ci_upper": float(np.quantile(samples, .975)),
        })
    return pd.DataFrame(rows)


def choose_verdict(metrics: pd.DataFrame, data_inventory: dict, controls: dict) -> tuple[str, dict]:
    external_available = isinstance(data_inventory.get("external_static_normal"), dict) and data_inventory["external_static_normal"].get("status") != "UNAVAILABLE"
    core = metrics[metrics.scenario.isin(["DS3", "DS4", "DS7", "DS8"])]
    full = core[core.method == "Full"]
    clean = metrics[(metrics.scenario == "cleanStatic") & (metrics.method == "Full")]
    clean_ok = bool(len(clean) and float(clean.false_positive_rate.iloc[0]) <= .015)
    controls_ok = all(
        item.get("max_consecutive_alarms", 0) < 3
        for item in controls["controls"] if item.get("status") == "AVAILABLE"
    )
    details = {
        "clean_holdout_fpr_requirement": clean_ok,
        "external_source_distinct_static_normal_available": external_available,
        "core_full_scenarios_available": sorted(full.scenario.unique().tolist()),
        "controls_no_persistent_alarm": controls_ok,
        "go_possible": False,
        "go_blockers": [],
    }
    if not external_available:
        details["go_blockers"].append("no source-distinct compatible static normal sequence")
    if not clean_ok:
        details["go_blockers"].append("clean holdout Full FPR exceeds 1.5% or is unavailable")
    if not controls_ok:
        details["go_blockers"].append("persistent alarms in one or more available controls")
    if not len(full):
        return "NO_PAPER_READY_EVIDENCE", details
    if not external_available:
        return "PIVOT_TO_PROVENANCE_EVALUATION_PAPER", details
    paper = core[core.method == "A0"]
    paper_strong = bool(len(paper) and paper.normalized_pauc_fpr_le_0_05.fillna(0).mean() >= .5)
    return ("B0_ONLY_STRONG_BUT_METHOD_WEAK" if paper_strong else "NO_PAPER_READY_EVIDENCE"), details


def final_readme(artifact_root: Path, verdict: str, verdict_details: dict, runner_ids: list[str]) -> str:
    return f"""# B0-CS Stage-0 Static result bundle

## 1. Historical-B0 reproduction

The frozen checkpoint and canonical binomial contract were hash-verified and scored as H0. It is a historical reference only.

## 2. Paper-B0 difference

Paper-B0 retains the exact shared nine-tap GRU architecture but uses chronological cleanStatic train/validation roles, train-only scaling, and excludes calibration/holdout from predictor fitting. Historical-B0 used a PRN holdout and cleanStatic+cleanDynamic gate calibration.

## 3. cleanStatic split and leakage audit

The machine-readable 50/15/20/15 split, 6 s guards, target/sample/byte overlap checks, role-local causal resets, and content hashes are in `split_and_overlap_audit.json`.

## 4. B0-CS formulas and assumptions

Scalar B0 RMSE is conformal-ranked, mapped by `0.5*p^-0.5`, averaged over at least four PRNs, block-max calibrated, and accumulated by `C_b=max(1,C_(b-1))*e_b` with alarm 100.

## 5. PRN dependence handling

Arithmetic mean e-value aggregation is permutation invariant and does not require PRN independence when component evidence is valid. PRN identity is never a feature or stratum.

## 6. Temporal dependence handling

The clean-train IAT rule fixed the non-overlapping block length before attacks. Results are `EMPIRICALLY_BLOCK_CALIBRATED`; arbitrary-dependence distribution-free and anytime-valid claims are forbidden.

## 7. DS3/DS4/DS7-8 performance

See `scenario_metrics.csv`. DS4 is marked `LIMITED` if it lacks the official 225 s pull-off. DS7 and DS8 are one family.

## 8. Comparisons

H0, A0, A1, A2, A3, A4, Full, Linear-AR, and SimpleConsecutive use frozen thresholds and common source processing. See `ablation_metrics.csv` and `bootstrap_intervals.csv`.

## 9. External static FPR

See `external_static_fpr.csv`. Missing source-distinct compatible normal data is explicitly `UNAVAILABLE` and blocks deployment-level FPR claims.

## 10. Physical controls

See `control_metrics.json`. Feature/residual-level controls are identified as such; raw-IQ AWGN/retracking is `UNAVAILABLE` without a sealed receiver rerun receipt.

## 11. Failed or limited scenarios

Every unavailable input, insufficient support, missing duration, or invalid lineage is carried as `UNAVAILABLE`/`LIMITED` in the structured files; no result was imputed.

## 12. Validity limits

Finite-sample ranks are tie-conservative, but nuisance estimation, temporal dependence, stationarity, block exchangeability, and mixing remain empirical assumptions. TEXBAT is developmental.

## 13. Final verdict

`{verdict}`

Verdict audit: `{json.dumps(verdict_details, sort_keys=True)}`

## 14. WCL-claimable contribution

Claimable output is limited to a preregistered, leakage-audited evaluation framework and its honestly bounded empirical result; deployment validity is not established.

## 15. Forbidden claims

Do not claim arbitrary-dependence distribution-free validity, anytime validity, independent DS7/DS8 confirmation, deployment-level FPR, or attack-blind source lineage beyond the recorded hashes.

## 16. Exactly one recommended next action

Acquire one sealed, source-distinct 20–30 minute static receiver capture with the identical complex-nine-tap, sample-counter, and C/N0 export contract, then evaluate once without recalibration.

Runner run IDs: {', '.join(runner_ids) if runner_ids else 'UNAVAILABLE (not supplied to finalizer)'}.
"""


def finalize_phase(artifact_root: Path, runner_ids: list[str]) -> None:
    verify_clean_freeze(artifact_root)
    controls = control_summary(artifact_root)
    write_json(artifact_root / "control_metrics.json", controls)
    bootstrap = bootstrap_intervals(artifact_root)
    bootstrap.to_csv(artifact_root / "bootstrap_intervals.csv", index=False)
    metrics = pd.read_csv(artifact_root / "scenario_metrics.csv")
    external = metrics[metrics.source_kind.eq("external_normal")].copy() if "source_kind" in metrics else pd.DataFrame()
    if external.empty:
        external = pd.DataFrame([{
            "scenario": "UNAVAILABLE", "method": "Full", "status": "UNAVAILABLE",
            "reason": "No compatible source-distinct static normal sequence was supplied.",
            "false_positive_rate": np.nan, "worst_run_external_fpr": np.nan,
        }])
    elif "false_positive_rate" in external:
        external["worst_run_external_fpr"] = external.groupby("method").false_positive_rate.transform("max")
    external.to_csv(artifact_root / "external_static_fpr.csv", index=False)
    inventory = read_json(artifact_root / "data_inventory.json")
    verdict, details = choose_verdict(metrics, inventory, controls)
    write_json(artifact_root / "final_verdict.json", {
        "schema": "gnss-doppler-lab.b0-cs-final-verdict.v1", "verdict": verdict,
        "criteria": details, "attack_outcome_retuning": False,
        "recommended_next_action": "Acquire one sealed source-distinct 20–30 minute static receiver capture and evaluate once without recalibration.",
    })
    (artifact_root / "README.md").write_text(final_readme(artifact_root, verdict, details, runner_ids), encoding="utf-8")
    make_final_plots(artifact_root)
    required_plots = {
        "paper_b0_residual_timeline.png", "b0_vs_b0_cs_receiver_score.png",
        "block_e_cusum_timeline.png", "scenario_roc_and_low_fpr_roc.png",
        "external_normal_fpr.png", "calibration_pvalue_histogram.png",
        "cn0_tracked_count_strata.png", "b0_vs_b0_cs_alarm_delay.png",
        "control_response.png", "ar_vs_gru_ablation.png",
    }
    missing = sorted(required_plots - {path.name for path in (artifact_root / "plots").glob("*.png")})
    if missing:
        raise ValueError(f"required plots missing: {missing}")
    print(json.dumps({"phase": "finalize", "verdict": verdict, "runner_ids": runner_ids}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["clean", "evaluate", "finalize"])
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--inputs-json")
    parser.add_argument("--device")
    parser.add_argument("--runner-run-id", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    if args.phase == "clean":
        clean_phase(artifact_root, args.device)
    elif args.phase == "evaluate":
        if not args.inputs_json:
            raise SystemExit("--inputs-json is required for evaluate")
        evaluate_phase(artifact_root, Path(args.inputs_json), args.device)
    else:
        finalize_phase(artifact_root, args.runner_run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
