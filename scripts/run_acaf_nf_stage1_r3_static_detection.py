#!/usr/bin/env python3
"""Run the fail-closed ACAF-NF Stage-1 R3 static campaign.

``clean`` cannot open attack paths.  It produces and hashes every learned or
selected object.  ``attack`` first verifies that freeze manifest and only then
authenticates and opens DS3/DS4/DS7/DS8.  ``all`` executes both in order.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.acaf_nf_stage1_r2a_l20_foundation_audit import (  # noqa: E402
    State, complex_caf_surface,
)
from gnss_doppler_lab.acaf_nf_stage0_r13_reconstruction import code_replica, carrier_wipeoff  # noqa: E402
from gnss_doppler_lab.acaf_nf_stage1_r3 import (  # noqa: E402
    ATTACK_PHASES, BUDGETS, CENTER, CLEAN_ROLES, DELAYS, DOPPLERS, FS_HZ, GRID,
    L20, NFConfig, PRIMARY_FAMILY, SUPPORT_SAMPLES, SetNeuralField,
    aggregate_l20, alarm_metrics, assert_no_byte_overlap, assert_no_clean_attack_time_overlap, attack_phase,
    binary_metrics, choose_pooling, fixed_policy_orders, gaussian_nll,
    paired_block_bootstrap, pool_scores, predict_distribution, role_for_support,
    sequential_trace, sha256, verify_freeze_manifest,
)

DEFAULT_OUTPUT = ROOT / "artifacts/acaf_nf_stage1_r3_static_detection"
DEFAULT_CONFIG = ROOT / "configs/acaf_nf_stage1_r3_static_detection.json"
REQUIRED_MAT = (
    "PRN_start_sample_count", "PRN", "carrier_doppler_hz", "code_freq_chips", "aux1",
    "Prompt_I", "Prompt_Q", "CN0_SNV_dB_Hz", "carrier_lock_test",
)
PROTECTED = (
    "model.pt", "model_context.pt", "model_no_context.pt", "normal_field_reference.npz",
    "query_policy.json", "thresholds.json", "pooling.json", "calibration.json",
)
SCIENCE_METHODS = (
    "active_adaptive", "learned_fixed", "uniform_fixed", "random_fixed",
    "epl_3", "fixed_delay_9", "active_magnitude", "active_no_context", "dense_nf",
)

_RAW: np.memmap | None = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    values = list(rows)
    names = list(fields or dict.fromkeys(key for row in values for key in row))
    if not names:
        names = ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(values)


def _channel(path: Path) -> int:
    digits = "".join(c for c in path.stem if c.isdigit())
    if not digits:
        raise ValueError(f"channel missing in {path}")
    return int(digits)


def _mat_paths(config: dict[str, Any], scenario: str) -> list[Path]:
    directory = Path(config["tracker_root"]) / scenario / "raw"
    inventory = load(_manifest_path(config, scenario))["tracking"]["mat_inventory"]
    paths = sorted((directory / name for name in inventory), key=_channel)
    if len(paths) < 8:
        raise RuntimeError(f"insufficient tracker MAT files for {scenario}: {len(paths)}")
    for path in paths:
        if not path.is_file() or sha256(path) != inventory[path.name]:
            raise RuntimeError(f"tracker MAT binding failed: {path}")
    return paths


def _manifest_path(config: dict[str, Any], scenario: str) -> Path:
    return Path(config["tracker_root"]) / scenario / "manifest.json"


def _select_even(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    return [values[int(i)] for i in np.linspace(0, len(values) - 1, count, dtype=int)]


def collect_candidate_events(config: dict[str, Any], scenario: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """CPU-only tracker scan; returns one causal L20 per channel/receiver second."""
    by_phase_second: dict[str, dict[int, dict[tuple[int, int], list[State]]]] = defaultdict(lambda: defaultdict(dict))
    census = []
    for path in _mat_paths(config, scenario):
        with h5py.File(path, "r") as handle:
            arrays = {name: np.asarray(handle[name]).reshape(-1) for name in REQUIRED_MAT}
        if len({len(x) for x in arrays.values()}) != 1:
            raise RuntimeError(f"MAT length mismatch: {path}")
        channel = _channel(path)
        stamps = arrays["PRN_start_sample_count"].astype(np.int64)
        prns = arrays["PRN"].astype(np.int64)
        n = len(stamps)
        def rolling_all(values: np.ndarray, width: int) -> np.ndarray:
            flags = np.asarray(values, dtype=np.int8)
            cumulative = np.r_[0, np.cumsum(flags, dtype=np.int64)]
            result = np.zeros(len(flags), dtype=bool)
            result[width - 1:] = cumulative[width:] - cumulative[:-width] == width
            return result
        same_prn = (prns[1:] == prns[:-1]) & (prns[1:] >= 1) & (prns[1:] <= 32)
        exact_support = np.diff(stamps) == SUPPORT_SAMPLES
        tracker_finite = np.logical_and.reduce([np.isfinite(arrays[name]) for name in ("Prompt_I", "Prompt_Q", "CN0_SNV_dB_Hz", "carrier_lock_test")])
        tracker_good = tracker_finite & (arrays["CN0_SNV_dB_Hz"] >= config["quality"]["cn0_min_db_hz"]) & (arrays["carrier_lock_test"] >= config["quality"]["carrier_lock_min"])
        state_good = np.logical_and.reduce([np.isfinite(arrays[name]) for name in ("carrier_doppler_hz", "code_freq_chips", "aux1")])
        valid = np.zeros(n, dtype=bool)
        candidates = np.arange(20, n)
        valid[candidates] = (
            rolling_all(same_prn, 20)[candidates - 1]
            & rolling_all(exact_support, 19)[candidates - 2]
            & rolling_all(tracker_good, 20)[candidates]
            & rolling_all(state_good, 20)[candidates - 1]
        )
        accepted = 0
        seen: set[tuple[int, int]] = set()
        for cur in np.flatnonzero(valid):
            tracker_rows = np.arange(cur - 19, cur + 1)
            state_rows = tracker_rows - 1
            prn = int(prns[cur])
            starts = stamps[state_rows]
            start, end = int(starts[0]), int(starts[-1] + SUPPORT_SAMPLES)
            phase = role_for_support(start, end) if scenario == "cleanStatic" else attack_phase(scenario, start, end)
            if phase is None:
                continue
            second = int((end - 1) // FS_HZ)
            key = (phase, second)
            if key in seen:
                continue
            states = [State(
                channel=channel, prn=prn, tracker_row=int(row), state_row=int(state),
                raw_start_sample=int(stamps[state]), code_freq_chips=float(arrays["code_freq_chips"][state]),
                carrier_doppler_hz=float(arrays["carrier_doppler_hz"][state]), aux1=float(arrays["aux1"][state]),
                prompt_i=float(arrays["Prompt_I"][row]), prompt_q=float(arrays["Prompt_Q"][row]),
                cn0_db_hz=float(arrays["CN0_SNV_dB_Hz"][row]), carrier_lock=float(arrays["carrier_lock_test"][row]),
            ) for row, state in zip(tracker_rows, state_rows)]
            by_phase_second[phase][second][(channel, prn)] = states
            seen.add(key); accepted += 1
        census.append({"channel": channel, "path": str(path), "rows": len(stamps), "candidate_seconds": accepted})
    result: dict[str, list[dict[str, Any]]] = {}
    targets = config["clean_event_targets"] if scenario == "cleanStatic" else config["attack_event_targets_per_phase"]
    for phase, seconds in by_phase_second.items():
        valid = sorted(second for second, pairs in seconds.items() if len(pairs) >= config["quality"]["minimum_prns"])
        chosen = _select_even(valid, int(targets.get(phase, 24)))
        result[phase] = [{"second": second, "pairs": seconds[second]} for second in chosen]
    required = set(CLEAN_ROLES) if scenario == "cleanStatic" else set(ATTACK_PHASES[scenario])
    missing = required - set(result)
    if missing:
        raise RuntimeError(f"{scenario} missing causal phases: {sorted(missing)}")
    return result, {"scenario": scenario, "channels": census, "selected": {k: len(v) for k, v in result.items()}}


def _init_raw(path: str) -> None:
    global _RAW
    _RAW = np.memmap(path, dtype="<i2", mode="r")


def _evaluate_window(payload: tuple[str, str, int, list[State]]) -> dict[str, Any]:
    scenario, phase, second, states = payload
    if _RAW is None:
        raise RuntimeError("raw map unavailable")
    surfaces = []
    powers = []
    prompts = []
    for state in states:
        start = state.raw_start_sample
        packed = np.asarray(_RAW[2 * start:2 * (start + SUPPORT_SAMPLES)]).reshape(-1, 2)
        iq = packed[:, 0].astype(np.float64) + 1j * packed[:, 1].astype(np.float64)
        powers.append(float(np.mean(np.abs(iq) ** 2)))
        prompts.append(math.hypot(state.prompt_i, state.prompt_q))
        surfaces.append(complex_caf_surface(iq, state, carrier_sign=-1))
    center, variance = aggregate_l20(surfaces)
    anchor = states[-1]
    return {
        "scenario": scenario, "phase": phase, "second": second, "time_s": float((states[0].raw_start_sample + states[-1].raw_start_sample + SUPPORT_SAMPLES) / 2 / FS_HZ),
        "channel": anchor.channel, "prn": anchor.prn, "raw_start_sample": states[0].raw_start_sample,
        "raw_end_sample": states[-1].raw_start_sample + SUPPORT_SAMPLES,
        "cn0_db_hz": float(np.mean([x.cn0_db_hz for x in states])), "carrier_lock": float(np.mean([x.carrier_lock for x in states])),
        "raw_power": float(np.mean(powers)), "prompt_magnitude": float(np.mean(prompts)),
        "surface": center.reshape(-1).astype(np.complex64), "l20_variance": variance.reshape(-1).astype(np.float32),
        "constituent_center_sha256": hashlib.sha256(np.asarray([x[5, 8] for x in surfaces], np.complex128).view(np.uint8)).hexdigest(),
        "states": [asdict(x) for x in states],
    }


def evaluate_events(config: dict[str, Any], scenario: str, events: dict[str, list[dict[str, Any]]], workers: int) -> list[dict[str, Any]]:
    tasks = [(scenario, phase, event["second"], states) for phase, values in events.items() for event in values for states in event["pairs"].values()]
    print(json.dumps({"stage": "raw_caf", "scenario": scenario, "l20_prn_windows": len(tasks), "one_ms_surfaces": len(tasks) * 20, "workers": workers}), flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_raw, initargs=(config["raw_paths"][scenario],)) as pool:
        result = list(pool.map(_evaluate_window, tasks, chunksize=1))
    print(json.dumps({"stage": "raw_caf_complete", "scenario": scenario, "l20_prn_windows": len(result)}), flush=True)
    return result


def save_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    metadata = []
    for row in rows:
        metadata.append({k: v for k, v in row.items() if k not in {"surface", "l20_variance", "states"}} | {"states": row["states"]})
    np.savez_compressed(path, surfaces=np.stack([x["surface"] for x in rows]), variances=np.stack([x["l20_variance"] for x in rows]))
    dump(path.with_suffix(".json"), metadata)


def load_cache(path: Path) -> list[dict[str, Any]]:
    archive = np.load(path)
    rows = load(path.with_suffix(".json"))
    for i, row in enumerate(rows):
        row["surface"] = archive["surfaces"][i]
        row["l20_variance"] = archive["variances"][i]
    return rows


def receiver_context_stats(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    train = np.asarray([[x["cn0_db_hz"], x["carrier_lock"]] for x in rows if x["phase"] == "train"], np.float32)
    mean, std = train.mean(0), train.std(0)
    std[std < 1e-6] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist()}


def context_value(row: dict[str, Any], stats: dict[str, list[float]]) -> np.ndarray:
    return (np.asarray([row["cn0_db_hz"], row["carrier_lock"]], np.float32) - stats["mean"]) / stats["std"]


def _model_selection_nll(model: SetNeuralField, rows: list[dict[str, Any]], stats: dict[str, list[float]], device: torch.device) -> float:
    model.eval(); order = fixed_policy_orders()["uniform_fixed"][:9]
    losses = []
    with torch.no_grad():
        for row in rows:
            values = row["surface"]
            mean, variance = predict_distribution(model, values, order, [i for i in range(187) if i not in order], context_value(row, stats), device)
            target = np.column_stack((values.real, values.imag))[[i for i in range(187) if i not in order]]
            losses.append(float(np.mean(.5 * np.sum((target - mean) ** 2 / variance + np.log(variance), axis=1))))
    return float(np.mean(losses))


def train_model(rows: list[dict[str, Any]], stats: dict[str, list[float]], context_features: bool, device: torch.device, seed: int) -> tuple[SetNeuralField, dict[str, Any]]:
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = NFConfig(context_features=context_features)
    model = SetNeuralField(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    train = [x for x in rows if x["phase"] == "train"]
    selection = [x for x in rows if x["phase"] == "selection"]
    y = np.stack([np.column_stack((x["surface"].real, x["surface"].imag)) for x in train]).astype(np.float32)
    rc = np.stack([context_value(x, stats) for x in train]).astype(np.float32)
    coordinates = torch.as_tensor(np.broadcast_to(GRID, (len(train), 187, 2)).copy(), device=device)
    values = torch.as_tensor(y, device=device); receiver = torch.as_tensor(rc, device=device)
    rng = np.random.default_rng(seed)
    best, best_state, best_epoch, history, patience = math.inf, None, 0, [], 12
    for epoch in range(1, 101):
        model.train(); permutation = rng.permutation(len(train)); epoch_loss = []
        for offset in range(0, len(train), 64):
            idx = permutation[offset:offset + 64]
            mask = np.zeros((len(idx), 187), np.float32)
            for j in range(len(idx)):
                count = int(rng.choice([3, 5, 9, 16])); chosen = [CENTER, *rng.choice([i for i in range(187) if i != CENTER], count - 1, replace=False).tolist()]
                mask[j, chosen] = 1
            tm = torch.as_tensor(mask, device=device)
            mean, variance = model(coordinates[idx], values[idx], tm, coordinates[idx], receiver[idx] if context_features else None)
            point = .5 * (((values[idx] - mean) ** 2) / variance + torch.log(variance)).sum(-1)
            loss = (point * (1 - tm)).sum() / (1 - tm).sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step(); epoch_loss.append(float(loss.detach().cpu()))
        selection_nll = _model_selection_nll(model, selection, stats, device)
        history.append({"epoch": epoch, "train_nll": float(np.mean(epoch_loss)), "selection_nll": selection_nll, "gradient_norm_last": float(nn_clip)})
        if selection_nll < best - 1e-5:
            best, best_epoch, patience = selection_nll, epoch, 12
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience == 0:
                break
    if best_state is None:
        raise RuntimeError("model did not train")
    model.load_state_dict(best_state); model.to(device).eval()
    return model, {"config": asdict(cfg), "best_epoch": best_epoch, "selection_nll": best, "history": history,
                   "train_samples": len(train), "selection_samples": len(selection), "device": str(device)}


def _trace_prefix(trace: list[dict[str, Any]], budget: int) -> float:
    return float(np.mean([x["sequential_surprise"] for x in trace[:budget]]))


def learned_order(model: SetNeuralField, selection: list[dict[str, Any]], stats: dict[str, list[float]], device: torch.device) -> list[int]:
    rank = defaultdict(list)
    for row in selection:
        _, trace = sequential_trace(model, row["surface"], 16, context_value(row, stats), device, policy="active_adaptive")
        for item in trace:
            rank[int(item["index"])].append(int(item["step"]))
    first = [CENTER, *sorted((i for i in rank if i != CENTER), key=lambda i: (np.mean(rank[i]), i))]
    remaining = [i for i in range(187) if i not in first]
    # Normal-only prior uncertainty finishes the dense order.
    representative = selection[0]
    _, variance = predict_distribution(model, representative["surface"], first[:16], remaining, context_value(representative, stats), device)
    return [*first, *[remaining[i] for i in np.argsort(-np.sum(np.log(variance), axis=1))]]


def dense_nf_score(model: SetNeuralField, row: dict[str, Any], stats: dict[str, list[float]], device: torch.device) -> float:
    targets = [i for i in range(187) if i != CENTER]
    mean, variance = predict_distribution(model, row["surface"], [CENTER], targets, context_value(row, stats), device)
    actual = np.column_stack((row["surface"].real, row["surface"].imag))[targets]
    return float(np.mean(np.sum((actual - mean) ** 2 / variance + np.log(variance), axis=1)))


def analytic_scores(row: dict[str, Any], template: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    y = row["surface"].astype(np.complex128); w = 1 / np.maximum(variance, 1e-5)
    denom = np.sum(w * np.abs(template) ** 2)
    alpha = np.sum(w * np.conj(template) * y) / max(denom, 1e-12)
    residual = y - alpha * template
    one = float(np.mean(w * np.abs(residual) ** 2))
    best = one
    field = template.reshape(11, 17)
    for di, dj in ((0, -4), (0, -2), (0, 2), (0, 4), (-2, 0), (-1, 0), (1, 0), (2, 0), (-1, -2), (1, 2)):
        shifted = np.roll(field, (di, dj), axis=(0, 1)).reshape(-1)
        design = np.column_stack((template, shifted))
        gram = design.conj().T @ (w[:, None] * design) + np.eye(2) * 1e-6
        beta = np.linalg.solve(gram, design.conj().T @ (w * y))
        best = min(best, float(np.mean(w * np.abs(y - design @ beta) ** 2)))
    flat = row["surface"]
    epl_indices = fixed_policy_orders()["epl_3"]
    delay_indices = fixed_policy_orders()["fixed_delay_9"]
    return {
        "raw_power_only": float(row["raw_power"]), "prompt_magnitude_only": float(row["prompt_magnitude"]),
        "epl_3_complex": float(np.mean(np.abs(residual[epl_indices]) ** 2)),
        "fixed_9_delay_tap_complex": float(np.mean(np.abs(residual[delay_indices]) ** 2)),
        "dense_one_source_residual": one, "analytic_two_source_glrt": one - best,
        "previous_acaf_r1_proxy": float(np.mean(np.abs(flat - template) ** 2)),
    }


def score_rows(
    rows: list[dict[str, Any]], model: SetNeuralField, context_model: SetNeuralField, no_context_model: SetNeuralField,
    stats: dict[str, list[float]], policy: dict[str, Any], template: np.ndarray, variance: np.ndarray,
    device: torch.device, *, traces: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fixed = {**fixed_policy_orders(), "learned_fixed": policy["learned_fixed_order"]}
    node_scores, query_rows, baseline_rows = [], [], []
    for number, row in enumerate(rows):
        receiver = context_value(row, stats)
        active_score, active_trace = sequential_trace(model, row["surface"], 16, receiver, device, policy="active_adaptive")
        for budget in (3, 5, 9, 16):
            node_scores.append({**{k: row[k] for k in ("scenario", "phase", "second", "time_s", "channel", "prn", "raw_start_sample", "raw_end_sample", "cn0_db_hz", "carrier_lock", "raw_power", "prompt_magnitude")},
                                "method": "active_adaptive", "budget": budget, "score": _trace_prefix(active_trace, budget)})
        magnitude_trace = [dict(x, sequential_surprise=float((x["actual_real"] ** 2 + x["actual_imag"] ** 2) ** .5 - (x["mu_real"] ** 2 + x["mu_imag"] ** 2) ** .5) ** 2 / max(x["var_real"] + x["var_imag"], 1e-9)) for x in active_trace]
        for budget in (3, 5, 9):
            node_scores.append({**{k: row[k] for k in ("scenario", "phase", "second", "time_s", "channel", "prn")}, "method": "active_magnitude", "budget": budget, "score": _trace_prefix(magnitude_trace, budget)})
        no_score, no_trace = sequential_trace(no_context_model, row["surface"], 9, receiver, device, policy="active_adaptive")
        node_scores.append({**{k: row[k] for k in ("scenario", "phase", "second", "time_s", "channel", "prn")}, "method": "active_no_context", "budget": 9, "score": no_score})
        context_score, _ = sequential_trace(context_model, row["surface"], 9, receiver, device, policy="active_adaptive")
        node_scores.append({**{k: row[k] for k in ("scenario", "phase", "second", "time_s", "channel", "prn")}, "method": "active_context", "budget": 9, "score": context_score})
        for method in ("learned_fixed", "uniform_fixed", "random_fixed"):
            _, trace = sequential_trace(model, row["surface"], 16, receiver, device, policy=method, fixed_order=fixed[method])
            for budget in (3, 5, 9, 16):
                node_scores.append({**{k: row[k] for k in ("scenario", "phase", "second", "time_s", "channel", "prn")}, "method": method, "budget": budget, "score": _trace_prefix(trace, budget)})
        for method, budget in (("epl_3", 3), ("fixed_delay_9", 9)):
            _, trace = sequential_trace(model, row["surface"], budget, receiver, device, policy=method, fixed_order=fixed[method])
            node_scores.append({**{k: row[k] for k in ("scenario", "phase", "second", "time_s", "channel", "prn")}, "method": method, "budget": budget, "score": _trace_prefix(trace, budget)})
        node_scores.append({**{k: row[k] for k in ("scenario", "phase", "second", "time_s", "channel", "prn")}, "method": "dense_nf", "budget": 187, "score": dense_nf_score(model, row, stats, device)})
        analytic = analytic_scores(row, template, variance)
        for method, score in analytic.items():
            baseline_rows.append({**{k: row[k] for k in ("scenario", "phase", "second", "time_s", "channel", "prn")}, "method": method, "budget": {"epl_3_complex": 3, "fixed_9_delay_tap_complex": 9}.get(method, 187 if "dense" in method or "glrt" in method or "acaf" in method else 0), "score": score})
        if traces:
            for item in active_trace:
                query_rows.append({"scenario": row["scenario"], "phase": row["phase"], "time_s": row["time_s"], "channel": row["channel"], "prn": row["prn"], **item})
        if number % 50 == 0:
            print(json.dumps({"scored_nodes": number + 1, "total": len(rows), "scenario": row["scenario"]}), flush=True)
    return node_scores, query_rows, baseline_rows


def pooled_events(node_rows: list[dict[str, Any]], pooling: str) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in node_rows:
        groups[(row["scenario"], row["phase"], row["second"], row["method"], int(row["budget"]))].append(row)
    out = []
    for (scenario, phase, second, method, budget), rows in sorted(groups.items()):
        scores = [float(x["score"]) for x in rows]
        if len(scores) < 4:
            continue
        out.append({"scenario": scenario, "family": PRIMARY_FAMILY.get(scenario, "cleanStatic"), "phase": phase, "time_s": float(np.mean([x["time_s"] for x in rows])),
                    "method": method, "budget": budget, "tracked_prn_count": len(rows), "used_prn_count": len(rows),
                    "prn_scores_json": json.dumps({str(x["prn"]): x["score"] for x in rows}, sort_keys=True), "pooled_score": pool_scores(scores, pooling)})
    return out


def calibrations(clean_pooled: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    cal, thresholds = {}, {}
    keys = sorted({(x["method"], x["budget"]) for x in clean_pooled})
    for method, budget in keys:
        values = np.asarray([x["pooled_score"] for x in clean_pooled if x["phase"] == "calibration" and x["method"] == method and x["budget"] == budget])
        center, scale = float(np.median(values)), max(float(np.quantile(values, .75) - np.quantile(values, .25)), 1e-9)
        name = f"{method}:K{budget}"; cal[name] = {"center": center, "scale": scale, "n": len(values), "source": "cleanStatic calibration only"}
        z = (values - center) / scale
        thresholds[name] = {"q99": float(np.quantile(z, .99, method="higher")), "q99.5": float(np.quantile(z, .995, method="higher")), "target_fpr": 0.01, "n": len(z)}
    return cal, thresholds


def apply_calibration(rows: list[dict[str, Any]], calibration: dict[str, Any]) -> None:
    for row in rows:
        key = f"{row['method']}:K{row['budget']}"; values = calibration[key]
        row["calibrated_score"] = (row["pooled_score"] - values["center"]) / values["scale"]


def authenticate(config: dict[str, Any], scenario: str, full_hash: bool) -> dict[str, Any]:
    raw = Path(config["raw_paths"][scenario]); manifest_path = _manifest_path(config, scenario); manifest = load(manifest_path)
    source = manifest["source"]
    report = {"scenario": scenario, "raw_path": str(raw), "raw_size_bytes": raw.stat().st_size,
              "expected_sha256": config["raw_sha256"][scenario], "manifest_path": str(manifest_path),
              "manifest_sha256": sha256(manifest_path), "manifest_raw_sha256": source["sha256"],
              "sample_rate_hz": source["sample_rate_hz"], "sample_format": source["sample_format"], "full_hash_performed": full_hash}
    if full_hash:
        report["actual_sha256"] = sha256(raw)
    report["status"] = "PASS" if (raw.stat().st_size == source["size_bytes"] and source["sha256"] == config["raw_sha256"][scenario]
        and (not full_hash or report["actual_sha256"] == config["raw_sha256"][scenario])) else "FAIL"
    if report["status"] != "PASS":
        raise RuntimeError(f"source authentication failed: {scenario}")
    return report


def tracker_binding_report(config: dict[str, Any], scenario: str) -> dict[str, Any]:
    manifest_path = _manifest_path(config, scenario); manifest = load(manifest_path)
    directory = Path(config["tracker_root"]) / scenario / "raw"
    files = {}
    for name, expected in sorted(manifest["tracking"]["mat_inventory"].items()):
        path = directory / name; actual = sha256(path) if path.is_file() else None
        files[name] = {"path": str(path), "expected_sha256": expected, "actual_sha256": actual, "match": actual == expected}
    return {"manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path), "files": files,
            "status": "PASS" if len(files) >= 8 and all(x["match"] for x in files.values()) else "FAIL"}


def clean_phase(output: Path, config_path: Path, workers: int) -> None:
    resume = output.is_dir() and (output / "clean_features.npz").is_file() and (output / "normal_split.json").is_file()
    if output.exists() and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True); (output / "plots").mkdir(exist_ok=True)
    config = load(config_path)
    if resume and load(output / "config.json") != config:
        raise RuntimeError("resume config drift")
    dump(output / "config.json", config)
    foundation = load(ROOT / config["foundation_artifact"] / "foundation_verdict.json")
    if foundation["verdict"] != "FOUNDATION_VALID":
        raise RuntimeError("R2a foundation is not valid")
    if (output / "clean_authentication.json").is_file():
        clean_auth = load(output / "clean_authentication.json")
        if clean_auth.get("status") != "PASS" or clean_auth.get("actual_sha256") != config["raw_sha256"]["cleanStatic"]:
            raise RuntimeError("persisted clean authentication is invalid")
    else:
        clean_auth = authenticate(config, "cleanStatic", True)
        dump(output / "clean_authentication.json", clean_auth)
    source = {"cleanStatic": clean_auth, "attacks": {x: {"status": "NOT_ACCESSED", "bytes_read": 0} for x in ("ds3", "ds4", "ds7", "ds8")}}
    print(json.dumps({"stage": "source_authenticated", "scenario": "cleanStatic"}), flush=True)
    if resume:
        rows = load_cache(output / "clean_features.npz")
        print(json.dumps({"stage": "resume_clean_cache", "l20_prn_windows": len(rows)}), flush=True)
    else:
        events, census = collect_candidate_events(config, "cleanStatic")
        print(json.dumps({"stage": "tracker_candidates", "scenario": "cleanStatic", "selected_events": {k: len(v) for k, v in events.items()}}), flush=True)
        rows = evaluate_events(config, "cleanStatic", events, workers)
        if any(x["raw_end_sample"] > 100 * FS_HZ for x in rows):
            raise RuntimeError("clean fit/selection/calibration opened >=100 seconds")
        by_role = defaultdict(list)
        for row in rows:
            by_role[row["phase"]].append({**row, "recording_sha256": config["raw_sha256"]["cleanStatic"]})
        assert_no_byte_overlap(by_role)
        save_cache(output / "clean_features.npz", rows)
        dump(output / "normal_split.json", {"roles": CLEAN_ROLES, "guards_excluded": [[0, 10], [45, 47], [62, 64], [82, 84]],
            "windows": {k: len(v) for k, v in by_role.items()}, "prn_windows": {k: sum(1 for x in rows if x["phase"] == k) for k in CLEAN_ROLES},
            "boundary_crossing_windows_excluded": True, "byte_overlap_status": "PASS", "clean_after_100s_accessed": False, "census": census})
    stats = receiver_context_stats(rows); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if all((output / name).is_file() for name in ("training_checkpoint.json", "model.pt", "model_context.pt", "model_no_context.pt")):
        training = load(output / "training_checkpoint.json"); context_report = training["NF-context"]; no_context_report = training["NF-no-context"]
        selected_context = training["selected"] == "NF-context"
        selected_model, context_model, no_context_model, _ = _load_models(output, device)
        print(json.dumps({"stage": "resume_trained_models", "selected": training["selected"]}), flush=True)
    else:
        context_model, context_report = train_model(rows, stats, True, device, config["seed"])
        print(json.dumps({"stage": "model_trained", "model": "NF-context", "best_epoch": context_report["best_epoch"], "selection_nll": context_report["selection_nll"]}), flush=True)
        no_context_model, no_context_report = train_model(rows, stats, False, device, config["seed"] + 1)
        print(json.dumps({"stage": "model_trained", "model": "NF-no-context", "best_epoch": no_context_report["best_epoch"], "selection_nll": no_context_report["selection_nll"]}), flush=True)
        selected_context = context_report["selection_nll"] <= no_context_report["selection_nll"]
        selected_model = context_model if selected_context else no_context_model
        chosen_report = context_report if selected_context else no_context_report
        shared = {"context_stats": stats, "grid": GRID.tolist(), "prn_identity_feature": False, "absolute_time_feature": False}
        torch.save({"state_dict": context_model.state_dict(), "config": context_report["config"], "selection": "NF-context",
                    "seed": config["seed"], **shared}, output / "model_context.pt")
        torch.save({"state_dict": no_context_model.state_dict(), "config": no_context_report["config"], "selection": "NF-no-context",
                    "seed": config["seed"] + 1, **shared}, output / "model_no_context.pt")
        torch.save({"state_dict": selected_model.state_dict(), "config": chosen_report["config"], "context_stats": stats,
                    "selection": "NF-context" if selected_context else "NF-no-context", "seed": config["seed"] if selected_context else config["seed"] + 1,
                    "grid": GRID.tolist(), "prn_identity_feature": False, "absolute_time_feature": False}, output / "model.pt")
        dump(output / "training_checkpoint.json", {"selected": "NF-context" if selected_context else "NF-no-context", "NF-context": context_report, "NF-no-context": no_context_report})
    selected_model = selected_model if 'selected_model' in locals() else (context_model if selected_context else no_context_model)
    selection_rows = [x for x in rows if x["phase"] == "selection"]
    order = learned_order(selected_model, selection_rows, stats, device)
    query_policy = {"active": "adaptive maximum predicted log-variance; clean-normal model only", "learned_fixed_order": order,
                    "orders": fixed_policy_orders(), "selection_source": "cleanStatic [47,62) only", "attack_labels_used": False,
                    "center_first": CENTER, "budget_unit": "complex coordinate", "seed": config["seed"]}
    dump(output / "query_policy.json", query_policy)
    template = np.mean(np.stack([x["surface"] for x in rows if x["phase"] == "train"]), axis=0)
    variance = np.var(np.stack([x["surface"] for x in rows if x["phase"] == "train"]), axis=0).real + 1e-4
    np.savez(output / "normal_field_reference.npz", template=template, variance=variance)
    clean_node, clean_traces, clean_baselines = score_rows(rows, selected_model, context_model, no_context_model, stats, query_policy, template, variance, device, traces=True)
    selection_node = [x for x in clean_node if x["phase"] == "selection" and x["method"] == "active_adaptive" and x["budget"] == 9]
    selection_events = defaultdict(list)
    for x in selection_node: selection_events[x["second"]].append(x["score"])
    pooling, pooling_diag = choose_pooling(list(selection_events.values()))
    dump(output / "pooling.json", {"method": pooling, "candidates": pooling_diag, "source": "cleanStatic selection only", "attack_results_used": False})
    clean_pooled = pooled_events(clean_node + clean_baselines, pooling)
    calibration, thresholds = calibrations(clean_pooled); dump(output / "calibration.json", calibration); dump(output / "thresholds.json", thresholds)
    apply_calibration(clean_pooled, calibration)
    write_csv(output / "clean_pooled_scores.csv", clean_pooled)
    write_csv(output / "query_traces_clean.csv", clean_traces)
    write_csv(output / "clean_node_scores.csv", clean_node + clean_baselines)
    dump(output / "model_manifest.json", {"selected": "NF-context" if selected_context else "NF-no-context", "selection_rule": "minimum cleanStatic selection NLL",
        "NF-context": context_report, "NF-no-context": no_context_report, "receiver_context_standardization": stats,
        "checkpoint_roles": {"model.pt": "clean-selection winner used by Full", "model_context.pt": "distinct NF-context checkpoint",
                             "model_no_context.pt": "distinct NF-no-context checkpoint"},
        "checkpoint_sha256": {name: sha256(output / name) for name in ("model.pt", "model_context.pt", "model_no_context.pt")},
        "device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "shared_all_prns": True, "prn_embedding": False, "raw_power_input": False, "prompt_magnitude_input": False})
    dump(output / "source_binding.json", source)
    freeze = {"schema": "acaf_nf_stage1_r3_freeze.v1", "generated_at_utc": now(), "attack_iq_opened_before_freeze": False,
              "files": {name: sha256(output / name) for name in PROTECTED}}
    dump(output / "freeze_manifest.json", freeze)
    dump(output / "execution_validity.json", {"status": "CLEAN_CHECKPOINT_FROZEN", "attack_iq_files_opened": [], "attack_iq_bytes_read": 0,
        "attack_rows_in_fit_selection_calibration_threshold": 0, "clean_after_100s_used": False, "freeze_manifest_sha256": sha256(output / "freeze_manifest.json")})
    print(json.dumps({"checkpoint": "clean", "status": "FROZEN", "freeze": freeze["files"]}, indent=2), flush=True)


def _load_checkpoint(path: Path, device: torch.device) -> tuple[SetNeuralField, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SetNeuralField(NFConfig(**checkpoint["config"])); model.load_state_dict(checkpoint["state_dict"]); model.to(device).eval()
    return model, checkpoint


def _load_models(output: Path, device: torch.device) -> tuple[SetNeuralField, SetNeuralField, SetNeuralField, dict[str, Any]]:
    model, checkpoint = _load_checkpoint(output / "model.pt", device)
    context_model, _ = _load_checkpoint(output / "model_context.pt", device)
    no_context_model, _ = _load_checkpoint(output / "model_no_context.pt", device)
    return model, context_model, no_context_model, checkpoint["context_stats"]


def scenario_metric_rows(all_rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    clean_hold = [x for x in all_rows if x["scenario"] == "cleanStatic" and x["phase"] == "holdout"]
    for scenario in ("ds3", "ds4", "ds7", "ds8"):
        for method, budget in sorted({(x["method"], x["budget"]) for x in all_rows}):
            attack = [x for x in all_rows if x["scenario"] == scenario and x["method"] == method and x["budget"] == budget]
            clean = [x for x in clean_hold if x["method"] == method and x["budget"] == budget]
            if not attack or not clean: continue
            pre = [x for x in attack if x["phase"] == "strict_pre"]
            post = [x for x in attack if x["phase"] != "strict_pre"]
            if not pre or not post: continue
            metric = binary_metrics([0] * len(pre) + [1] * len(post), [x["calibrated_score"] for x in pre + post])
            threshold = thresholds[f"{method}:K{budget}"]["q99"]
            onset = ATTACK_PHASES[scenario]["strict_pre"][1]
            alarm = alarm_metrics([x["time_s"] for x in post], [x["calibrated_score"] for x in post], threshold, onset)
            output.append({"scenario": scenario, "family": PRIMARY_FAMILY[scenario], "method": method, "budget": budget, **metric,
                "clean_holdout_fpr": float(np.mean([x["calibrated_score"] >= threshold for x in clean])),
                "strict_external_pre_onset_fpr": float(np.mean([x["calibrated_score"] >= threshold for x in pre])), **alarm,
                "tracked_prn_count_min": min(x["tracked_prn_count"] for x in attack), "evaluation_epochs": len(attack),
                "correlator_count": budget, "ds4_full_recording": scenario != "ds4", "status": "TRANSITION_ONLY" if scenario == "ds4" else "PASS"})
    return output


def physical_controls(
    clean_rows: list[dict[str, Any]], model: SetNeuralField, stats: dict[str, list[float]],
    threshold: float, calibration: dict[str, float], pooling: str, device: torch.device,
) -> dict[str, Any]:
    selected = [x for x in clean_rows if x["phase"] == "holdout"]

    def score_nodes(surfaces: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        nodes = []
        for row, surface in zip(selected, surfaces):
            score, _ = sequential_trace(model, surface, 9, context_value(row, stats), device, policy="active_adaptive")
            nodes.append({"scenario": "cleanStatic", "phase": "holdout", "second": row["second"], "time_s": row["time_s"],
                          "method": "active_adaptive", "budget": 9, "score": score})
        grouped = defaultdict(list)
        for node in nodes: grouped[node["second"]].append(node["score"])
        pooled = np.asarray([pool_scores(values, pooling) for _, values in sorted(grouped.items()) if len(values) >= 4])
        calibrated = (pooled - calibration["center"]) / calibration["scale"]
        return np.asarray([x["score"] for x in nodes]), calibrated

    base, base_calibrated = score_nodes([x["surface"] for x in selected])
    gain = {}; phase = {}
    # The control is applied before the exact prompt rotation/division equation.
    for value in (.5, .8, 1.2, 2.0):
        transformed = [aggregate_l20([x["surface"].reshape(11, 17) * value] * 20)[0].reshape(-1) for x in selected]
        scores, calibrated = score_nodes(transformed)
        gain[str(value)] = {"max_abs_score_delta": float(np.max(np.abs(scores - base))),
                            "pooled_calibrated_fpr": float(np.mean(calibrated >= threshold)),
                            "alarm_delta": float(np.mean(calibrated >= threshold) - np.mean(base_calibrated >= threshold))}
    phases = [0, math.pi / 4, math.pi / 2, math.pi]
    rng = np.random.default_rng(20260808); phases.append(float(rng.uniform(-math.pi, math.pi)))
    for value in phases:
        transformed = [aggregate_l20([x["surface"].reshape(11, 17) * np.exp(1j * value)] * 20)[0].reshape(-1) for x in selected]
        scores, calibrated = score_nodes(transformed)
        phase[str(value)] = {"max_abs_score_delta": float(np.max(np.abs(scores - base))),
                             "pooled_calibrated_fpr": float(np.mean(calibrated >= threshold)),
                             "alarm_delta": float(np.mean(calibrated >= threshold) - np.mean(base_calibrated >= threshold))}
    awgn = {}
    for target in (35, 40, 45):
        noisy_scores = []
        for i, row in enumerate(selected):
            # C/N0*T is the coherent carrier-to-noise ratio for T=1 ms.
            sigma = 1 / math.sqrt(max(10 ** (target / 10) * .001, 1e-9))
            noise = np.random.default_rng(1000 + target + i).normal(0, sigma / math.sqrt(2), (187, 2))
            noisy = row["surface"] + noise[:, 0] + 1j * noise[:, 1]
            noisy_scores.append(sequential_trace(model, noisy, 9, context_value(row, stats), device, policy="active_adaptive")[0])
        awgn[str(target)] = {"target_cn0_db_hz": target, "caf_noise_sigma_from_cn0_times_1ms": sigma, "mean_score": float(np.mean(noisy_scores))}
    return {"gain": gain, "global_phase": phase, "awgn_cn0": awgn, "base_prn_n": len(base), "base_pooled_event_n": len(base_calibrated),
            "base_pooled_calibrated_fpr": float(np.mean(base_calibrated >= threshold)), "fpr_unit": "pooled variable-PRN event after frozen calibration",
            "normalization": "per-ms prompt phase and magnitude",
            "raw_rms_diagnostic_used_for_scale_audit": True, "gain_phase_fpr_increase_max_percentage_points": 100 * max([abs(x["alarm_delta"]) for x in gain.values()] + [abs(x["alarm_delta"]) for x in phase.values()])}


def positive_control(clean_rows: list[dict[str, Any]], template: np.ndarray, variance: np.ndarray) -> dict[str, Any]:
    base = clean_rows[0]["surface"].reshape(11, 17); results = []
    for ratio in (.25, .5, 1.0):
        for phase in (0, math.pi / 2, math.pi):
            for di, dj in ((-1, -2), (0, -2), (0, 2), (1, 2)):
                second = np.roll(base, (di, dj), axis=(0, 1)) * ratio * np.exp(1j * phase)
                mixed = (base + second).reshape(-1); mixed /= max(abs(mixed[CENTER]), 1e-9)
                fake = {**clean_rows[0], "surface": mixed}
                score = analytic_scores(fake, template, variance)["analytic_two_source_glrt"]
                results.append({"power_ratio": ratio, "relative_phase_rad": phase, "delay_chips": float(DELAYS[8 + dj]), "residual_doppler_hz": float(DOPPLERS[5 + di]), "glrt_score": score})
    return {"status": "DIAGNOSTIC_ONLY", "construction": "same-PRN complex CAF second component with signed delay/Doppler/phase and prompt renormalization", "different_prn_used": False,
            "zero_padding_shift_used": False, "results": results, "sensitivity": float(np.corrcoef([x["power_ratio"] for x in results], [x["glrt_score"] for x in results])[0, 1])}


def plots(output: Path, pooled: list[dict[str, Any]], metrics: list[dict[str, Any]], controls: dict[str, Any]) -> None:
    plot_dir = output / "plots"; plot_dir.mkdir(exist_ok=True)
    for scenario in ("ds3", "ds4", "ds7", "ds8"):
        rows = [x for x in pooled if x["scenario"] == scenario and x["method"] == "active_adaptive" and x["budget"] == 9]
        plt.figure(figsize=(9, 3)); plt.plot([x["time_s"] for x in rows], [x["calibrated_score"] for x in rows], marker="o", ms=3)
        onset = ATTACK_PHASES[scenario]["strict_pre"][1]; plt.axvline(onset, color="orange", label="signal onset")
        if scenario in ("ds3", "ds7", "ds8"): plt.axvline(195 if scenario == "ds3" else 150, color="red", label="pull-off/time-push")
        plt.xlabel("receiver/raw time (s)"); plt.ylabel("calibrated score"); plt.legend(); plt.tight_layout(); plt.savefig(plot_dir / f"{scenario}_score_time.png", dpi=140); plt.close()
    active = [x for x in metrics if x["method"] == "active_adaptive"]
    plt.figure(figsize=(7, 4))
    for scenario in ("ds3", "ds4", "ds7", "ds8"):
        rows = sorted([x for x in active if x["scenario"] == scenario], key=lambda x: x["budget"])
        plt.plot([x["budget"] for x in rows], [x["normalized_partial_auc_fpr_0_05"] for x in rows], marker="o", label=scenario)
    plt.xscale("log"); plt.xlabel("complex correlator queries K"); plt.ylabel("normalized pAUC @ FPR≤5%")
    plt.legend(); plt.tight_layout(); plt.savefig(plot_dir / "budget_pauc.png", dpi=140); plt.savefig(plot_dir / "detection_correlator_tradeoff.png", dpi=140); plt.close()
    heat = np.zeros((11, 17)); order = load(output / "query_policy.json")["learned_fixed_order"][:16]
    for rank, index in enumerate(order, 1): heat[np.unravel_index(index, (11, 17))] = 17 - rank
    plt.figure(figsize=(7, 4)); plt.imshow(heat, aspect="auto", origin="lower"); plt.colorbar(label="query priority"); plt.tight_layout(); plt.savefig(plot_dir / "active_query_coordinate_heatmap.png", dpi=140); plt.savefig(plot_dir / "query_order.png", dpi=140); plt.close()
    rows = [x for x in pooled if x["method"] == "active_adaptive" and x["budget"] == 9]
    prns = sorted({int(k) for x in rows for k in json.loads(x["prn_scores_json"])})
    matrix = np.full((len(prns), len(rows)), np.nan)
    for j, row in enumerate(rows):
        for prn, score in json.loads(row["prn_scores_json"]).items(): matrix[prns.index(int(prn)), j] = score
    plt.figure(figsize=(10, 4)); plt.imshow(matrix, aspect="auto"); plt.yticks(range(len(prns)), prns); plt.colorbar(); plt.tight_layout(); plt.savefig(plot_dir / "prn_score_heatmap.png", dpi=140); plt.close()
    for name, methods in (("dense_vs_active", ("active_adaptive", "dense_nf")), ("active_vs_b0", ("active_adaptive", "B0_native"))):
        plt.figure(figsize=(6, 4)); subset = [x for x in metrics if x["method"] in methods]
        if subset:
            for method in methods:
                vals = [x["normalized_partial_auc_fpr_0_05"] for x in subset if x["method"] == method]
                plt.bar(method, np.mean(vals) if vals else 0)
        else: plt.text(.5, .5, "B0 exact lineage unavailable", ha="center")
        plt.ylabel("macro pAUC"); plt.tight_layout(); plt.savefig(plot_dir / f"{name}.png", dpi=140); plt.close()
    labels = list(controls["gain"]); vals = [controls["gain"][x]["max_abs_score_delta"] for x in labels]
    plt.figure(figsize=(7, 4)); plt.bar(labels, vals); plt.ylabel("max |score delta|"); plt.xlabel("global gain"); plt.tight_layout(); plt.savefig(plot_dir / "gain_phase_awgn_control.png", dpi=140); plt.close()
    pre = [x for x in metrics if x["method"] == "active_adaptive" and x["budget"] == 9]
    plt.figure(figsize=(7, 4)); plt.bar([x["scenario"] for x in pre], [x["strict_external_pre_onset_fpr"] for x in pre]); plt.axhline(.05, color="red"); plt.ylabel("strict pre-onset FPR"); plt.tight_layout(); plt.savefig(plot_dir / "external_pre_onset_fpr.png", dpi=140); plt.close()


def family_metric_rows(all_rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute three equally weighted primary-family results, pooling DS7 and DS8."""
    output = []
    clean_hold = [x for x in all_rows if x["scenario"] == "cleanStatic" and x["phase"] == "holdout"]
    families = {"ds3": ("ds3",), "ds4": ("ds4",), "ds7_ds8": ("ds7", "ds8")}
    for family, scenarios in families.items():
        for method, budget in sorted({(x["method"], x["budget"]) for x in all_rows}):
            attack = [x for x in all_rows if x["scenario"] in scenarios and x["method"] == method and x["budget"] == budget]
            clean = [x for x in clean_hold if x["method"] == method and x["budget"] == budget]
            pre = [x for x in attack if x["phase"] == "strict_pre"]
            post = [x for x in attack if x["phase"] != "strict_pre"]
            if not pre or not post or not clean:
                continue
            metric = binary_metrics([0] * len(pre) + [1] * len(post), [x["calibrated_score"] for x in pre + post])
            threshold = thresholds[f"{method}:K{budget}"]["q99"]
            per_scenario_alarm = []
            for scenario in scenarios:
                scenario_post = [x for x in post if x["scenario"] == scenario]
                per_scenario_alarm.append(alarm_metrics([x["time_s"] for x in scenario_post], [x["calibrated_score"] for x in scenario_post],
                                                        threshold, ATTACK_PHASES[scenario]["strict_pre"][1]))
            delays = [x["first_alarm_delay_s"] for x in per_scenario_alarm if x["first_alarm_delay_s"] is not None]
            output.append({"family": family, "scenario_members": "+".join(scenarios), "method": method, "budget": budget, **metric,
                "clean_holdout_fpr": float(np.mean([x["calibrated_score"] >= threshold for x in clean])),
                "strict_external_pre_onset_fpr": float(np.mean([x["calibrated_score"] >= threshold for x in pre])),
                "attack_detection_rate": float(np.mean([x["calibrated_score"] >= threshold for x in post])),
                "first_alarm_delay_s": min(delays) if delays else None,
                "sustained_alarm_fraction": float(np.mean([x["calibrated_score"] >= threshold for x in post])),
                "tracked_prn_count_min": min(x["tracked_prn_count"] for x in attack), "evaluation_epochs": len(attack),
                "correlator_count": budget, "status": "TRANSITION_ONLY" if family == "ds4" else "PASS"})
    return output


def low_fpr_block_bootstrap(
    rows: list[dict[str, Any]], left: str, left_budget: int, right: str, right_budget: int,
    scenarios: tuple[str, ...], *, seed: int, replicates: int = 10_000,
) -> dict[str, Any]:
    """Paired 10-second block bootstrap of normalized pAUC differences."""
    left_rows = [x for x in rows if x["scenario"] in scenarios and x["method"] == left and x["budget"] == left_budget]
    right_map = {(x["scenario"], x["phase"], round(x["time_s"], 3)): x for x in rows
                 if x["scenario"] in scenarios and x["method"] == right and x["budget"] == right_budget}
    pairs = [(x, right_map.get((x["scenario"], x["phase"], round(x["time_s"], 3)))) for x in left_rows]
    pairs = [(a, b) for a, b in pairs if b is not None]
    if not pairs:
        return {"status": "INCONCLUSIVE", "reason": "no_paired_epochs", "metric": "normalized_partial_auc_fpr_0_05"}
    labels = np.asarray([0 if a["phase"] == "strict_pre" else 1 for a, _ in pairs], int)
    left_scores = np.asarray([a["calibrated_score"] for a, _ in pairs], float)
    right_scores = np.asarray([b["calibrated_score"] for _, b in pairs], float)
    block_keys = [(a["scenario"], int(a["time_s"] // 10)) for a, _ in pairs]
    unique_blocks = sorted(set(block_keys)); indices = {key: np.flatnonzero(np.asarray([value == key for value in block_keys])) for key in unique_blocks}
    if len(unique_blocks) < 2 or set(labels.tolist()) != {0, 1}:
        return {"status": "INCONCLUSIVE", "reason": "insufficient_two_class_10s_blocks", "blocks": len(unique_blocks),
                "metric": "normalized_partial_auc_fpr_0_05"}
    actual = binary_metrics(labels, left_scores)["normalized_partial_auc_fpr_0_05"] - binary_metrics(labels, right_scores)["normalized_partial_auc_fpr_0_05"]
    rng = np.random.default_rng(seed); effects = []
    for _ in range(replicates):
        sampled_blocks = rng.choice(len(unique_blocks), size=len(unique_blocks), replace=True)
        sampled = np.concatenate([indices[unique_blocks[i]] for i in sampled_blocks])
        if set(labels[sampled].tolist()) != {0, 1}:
            continue
        effects.append(binary_metrics(labels[sampled], left_scores[sampled])["normalized_partial_auc_fpr_0_05"]
                       - binary_metrics(labels[sampled], right_scores[sampled])["normalized_partial_auc_fpr_0_05"])
    if len(effects) < replicates // 2:
        return {"status": "INCONCLUSIVE", "reason": "too_few_valid_two_class_resamples", "valid_replicates": len(effects),
                "blocks": len(unique_blocks), "metric": "normalized_partial_auc_fpr_0_05"}
    return {"status": "PASS", "effect": float(actual), "ci95": [float(np.quantile(effects, .025)), float(np.quantile(effects, .975))],
            "metric": "normalized_partial_auc_fpr_0_05", "block_seconds": 10, "blocks": len(unique_blocks),
            "replicates_requested": replicates, "valid_replicates": len(effects), "seed": seed}


def finalize(output: Path, config: dict[str, Any], pooled: list[dict[str, Any]], node: list[dict[str, Any]], traces: list[dict[str, Any]], baselines: list[dict[str, Any]]) -> None:
    thresholds, calibration = load(output / "thresholds.json"), load(output / "calibration.json")
    metrics = scenario_metric_rows(pooled, thresholds); write_csv(output / "scenario_metrics.csv", metrics)
    family_metrics = family_metric_rows(pooled, thresholds); write_csv(output / "family_metrics.csv", family_metrics)
    phases = []
    for key in sorted({(x["scenario"], x["phase"], x["method"], x["budget"]) for x in pooled}):
        rows = [x for x in pooled if (x["scenario"], x["phase"], x["method"], x["budget"]) == key]
        threshold = thresholds[f"{key[2]}:K{key[3]}"]["q99"]
        phases.append({"scenario": key[0], "phase": key[1], "method": key[2], "budget": key[3], "n": len(rows), "mean_score": float(np.mean([x["calibrated_score"] for x in rows])), "alarm_fraction": float(np.mean([x["calibrated_score"] >= threshold for x in rows])), "tracked_prn_min": min(x["tracked_prn_count"] for x in rows)})
    write_csv(output / "phase_metrics.csv", phases)
    write_csv(output / "per_epoch_scores.csv", pooled); write_csv(output / "query_traces.csv", traces); write_csv(output / "per_prn_scores.csv", node + baselines)
    budget = []
    for method, k in sorted({(x["method"], x["budget"]) for x in family_metrics}):
        selected = [x for x in family_metrics if x["method"] == method and x["budget"] == k]
        budget.append({"method": method, "budget": k, "correlator_count": k, "dense_fraction": k / 187,
                       "primary_family_macro_pauc": float(np.mean([x["normalized_partial_auc_fpr_0_05"] for x in selected])),
                       "macro_unit": "three equally weighted primary families: DS3, DS4, pooled DS7/DS8",
                       "wall_clock_inference_ms_per_prn": None, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0})
    write_csv(output / "budget_metrics.csv", budget)
    baseline_methods = {x["method"] for x in baselines}
    write_csv(output / "baseline_metrics.csv", [x for x in metrics if x["method"] in baseline_methods])
    mapping = {"A0": ("raw_power_only", 0), "A1": ("epl_3", 3), "A2": ("fixed_delay_9", 9), "A3": ("dense_one_source_residual", 187),
               "A4": ("analytic_two_source_glrt", 187), "A5": ("learned_fixed", 9), "A6": ("active_magnitude", 9), "A7": ("active_adaptive", 9),
               "A8": ("active_no_context", 9), "Full": ("active_adaptive", 9)}
    ablations = []
    for name, (method, k) in mapping.items():
        vals = [x for x in family_metrics if x["method"] == method and x["budget"] == k]
        ablations.append({"ablation": name, "method": method, "budget": k,
                          "primary_family_macro_pauc": float(np.mean([x["normalized_partial_auc_fpr_0_05"] for x in vals])) if vals else None,
                          "status": "PASS" if vals else "NOT_AVAILABLE",
                          "note": "Full selected by clean selection NLL; A8 can coincide with Full when NF-no-context wins." if name in {"A8", "Full"} else ""})
    write_csv(output / "ablation_metrics.csv", ablations)
    comparisons = (
        ("active_adaptive", 3, "epl_3", 3),
        ("active_adaptive", 9, "fixed_delay_9", 9),
        ("active_adaptive", 9, "learned_fixed", 9),
        ("active_adaptive", 9, "uniform_fixed", 9),
        ("active_adaptive", 9, "random_fixed", 9),
        ("active_adaptive", 9, "dense_nf", 187),
        ("analytic_two_source_glrt", 187, "dense_one_source_residual", 187),
        ("active_context", 9, "active_no_context", 9),
    )
    boots = {"schema": "acaf_nf_stage1_r3_bootstrap.v2", "replicates": 10000, "block_seconds": 10,
             "metric": "normalized_partial_auc_fpr_0_05", "scenario_comparisons": {}, "family_comparisons": {},
             "active_vs_b0": {"status": "INCONCLUSIVE", "reason": "B0 exact native lineage unavailable; no historic scores reused"}}
    for scenario in ("ds3", "ds4", "ds7", "ds8"):
        for left, lk, right, rk in comparisons:
            key = f"{scenario}:{left}_K{lk}_vs_{right}_K{rk}"
            boots["scenario_comparisons"][key] = low_fpr_block_bootstrap(pooled, left, lk, right, rk, (scenario,), seed=config["seed"])
    for family, scenarios in {"ds3": ("ds3",), "ds4": ("ds4",), "ds7_ds8": ("ds7", "ds8")}.items():
        for left, lk, right, rk in comparisons:
            key = f"{family}:{left}_K{lk}_vs_{right}_K{rk}"
            boots["family_comparisons"][key] = low_fpr_block_bootstrap(pooled, left, lk, right, rk, scenarios, seed=config["seed"])
    boots["family_note"] = "DS7 and DS8 are jointly resampled and evaluated as one non-independent scenario family"
    dump(output / "bootstrap_results.json", boots)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model, _, _, stats = _load_models(output, device)
    clean = load_cache(output / "clean_features.npz"); control_threshold = thresholds["active_adaptive:K9"]["q99"]
    controls = physical_controls(clean, model, stats, control_threshold, calibration["active_adaptive:K9"], load(output / "pooling.json")["method"], device)
    dump(output / "physical_controls.json", controls)
    ref = np.load(output / "normal_field_reference.npz"); positive = positive_control(clean, ref["template"], ref["variance"]); dump(output / "positive_control_results.json", positive)
    b0 = {"status": "UNAVAILABLE_EXACT_LINEAGE", "native_evaluator_invoked": False, "historic_scores_reused": False,
          "reason": "No frozen B0 TEXBAT node-feature and threshold lineage exists on the exact R3 epochs; OAKBAT/CMTE checkpoints are different input campaigns.",
          "epoch_alignment_tolerance_s": 1.0, "same_epoch_scores_computed": False, "comparison_valid": False}
    dump(output / "b0_comparison.json", b0)
    dump(output / "diagnostic_scenarios.json", {
        "ds1": {"status": "NOT_EVALUATED_MISSING_R2A_TRACKER_BINDING", "core_gate_used": False},
        "ds2": {"status": "NOT_EVALUATED_MISSING_R2A_TRACKER_BINDING", "core_gate_used": False},
        "reason": "No authenticated R2a-aligned tracker replay/manifest exists for DS1 or DS2; raw IQ was not opened without that binding.",
        "core_exclusions_not_accessed": ["cleanDynamic", "ds5", "ds6"],
    })
    power = [x for x in family_metrics if x["method"] == "raw_power_only"]
    full = [x for x in family_metrics if x["method"] == "active_adaptive" and x["budget"] == 9]
    holdout_fpr = max((x["clean_holdout_fpr"] for x in full), default=1.0); pre_fpr = max((x["strict_external_pre_onset_fpr"] for x in full), default=1.0)
    gates = {"clean_holdout_q99_fpr_le_2pct": holdout_fpr <= .02, "strict_external_pre_worst_fpr_le_5pct": pre_fpr <= .05,
             "gain_phase_fpr_increase_le_2pp": controls["gain_phase_fpr_increase_max_percentage_points"] <= 2,
             "minimum_4_prns": min((x["tracked_prn_count_min"] for x in full), default=0) >= 4,
             "raw_provenance_timeline_pass": True, "ds4_full_available": False, "b0_lineage_available": False}
    verdict = "ACAF_NF_INCONCLUSIVE" if not gates["ds4_full_available"] or not gates["b0_lineage_available"] else "ACAF_NF_NO_GO"
    dump(output / "go_no_go.json", {"verdict": verdict, "gates": gates, "primary_family_definition": ["ds3", "ds4", "ds7_ds8"],
        "ds7_ds8_independent_confirmations": False, "threshold_or_policy_changed_after_attack": False,
        "reason": "FULL_DS4_UNAVAILABLE and native exact B0 lineage unavailable; preregistered rules prohibit forced GO/NO-GO.",
        "detection_gate_descriptive_only": {"family_weighting": "DS3, DS4, pooled DS7/DS8 each weight 1/3",
            "active_k9_macro_pauc": float(np.mean([x["normalized_partial_auc_fpr_0_05"] for x in full])),
            "power_only_macro_pauc": float(np.mean([x["normalized_partial_auc_fpr_0_05"] for x in power])) if power else None}})
    plots(output, pooled, metrics, controls)


def attack_phase_run(output: Path, config_path: Path, workers: int) -> None:
    config = load(config_path); frozen = verify_freeze_manifest(output)
    source = load(output / "source_binding.json")
    all_attack = []; timeline = {"schema": "acaf_nf_stage1_r3_timeline.v1", "raw_time_origin": "first complex sample is t=0; receiver PRN_start_sample_count is absolute from first_file_sample=0", "scenarios": {}}
    for scenario in ("ds3", "ds4", "ds7", "ds8"):
        source["attacks"][scenario] = authenticate(config, scenario, True)
        print(json.dumps({"stage": "source_authenticated", "scenario": scenario}), flush=True)
        events, census = collect_candidate_events(config, scenario); rows = evaluate_events(config, scenario, events, workers); all_attack.extend(rows)
        duration = Path(config["raw_paths"][scenario]).stat().st_size / 4 / FS_HZ
        finite_phases = {name: [left, duration if math.isinf(right) else right] for name, (left, right) in ATTACK_PHASES[scenario].items()}
        timeline["scenarios"][scenario] = {"phases": finite_phases, "duration_s": duration, "tracker_census": census,
            "status": "FULL_DS4_UNAVAILABLE" if scenario == "ds4" and duration < 225 else "PASS", "official_onset_s": ATTACK_PHASES[scenario]["strict_pre"][1]}
    dump(output / "source_binding.json", source); dump(output / "timeline_validation.json", timeline); save_cache(output / "attack_features.npz", all_attack)
    assert_no_clean_attack_time_overlap(load_cache(output / "clean_features.npz"), all_attack)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model, context_model, no_model, stats = _load_models(output, device)
    policy = load(output / "query_policy.json"); pooling = load(output / "pooling.json")["method"]
    ref = np.load(output / "normal_field_reference.npz"); start = time.perf_counter()
    attack_node, traces, attack_baselines = score_rows(all_attack, model, context_model, no_model, stats, policy, ref["template"], ref["variance"], device, traces=True)
    clean_node = list(csv.DictReader((output / "clean_node_scores.csv").open(newline="", encoding="utf-8")))
    for row in clean_node:
        for key in ("second", "channel", "prn", "budget"): row[key] = int(float(row[key]))
        for key in ("time_s", "score"): row[key] = float(row[key])
    clean_pooled = list(csv.DictReader((output / "clean_pooled_scores.csv").open(newline="", encoding="utf-8")))
    for row in clean_pooled:
        for key in ("budget", "tracked_prn_count", "used_prn_count"): row[key] = int(float(row[key]))
        for key in ("time_s", "pooled_score", "calibrated_score"): row[key] = float(row[key])
    attack_pooled = pooled_events(attack_node + attack_baselines, pooling); calibration = load(output / "calibration.json"); apply_calibration(attack_pooled, calibration)
    all_pooled = clean_pooled + attack_pooled
    finalize(output, config, all_pooled, clean_node + attack_node, traces, attack_baselines)
    execution = load(output / "execution_validity.json"); execution.update({"status": "COMPLETE_INCONCLUSIVE", "attack_iq_files_opened": [config["raw_paths"][x] for x in ("ds3", "ds4", "ds7", "ds8")],
        "attack_iq_bytes_read": sum(Path(config["raw_paths"][x]).stat().st_size for x in ("ds3", "ds4", "ds7", "ds8")), "frozen_hashes_verified_before_attack": frozen,
        "attack_rows_in_fit_selection_calibration_threshold": 0, "scoring_wall_seconds": time.perf_counter() - start, "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "science_status": "INCONCLUSIVE", "reason": "FULL_DS4_UNAVAILABLE; B0 exact lineage unavailable"})
    dump(output / "execution_validity.json", execution)
    readme = """# ACAF-NF Stage-1 R3 static detection\n\nThis is the first formal normal-only static ACAF-NF campaign. The clean roles are exactly train `[10,45)`, selection `[47,62)`, calibration `[64,82)`, and holdout `[84,100)` seconds; crossing L20 windows and all cleanStatic data at or after 100 s are excluded from fitting, selection, and calibration. Each 1-ms complex CAF is prompt-phase rotated and prompt-amplitude normalized before robust L20 aggregation.\n\nThe clean-selection winner, distinct context and no-context models, normal-field analytic reference, query policy, pooling, calibration, and thresholds were SHA-256 frozen before attack evaluation. The result is `ACAF_NF_INCONCLUSIVE`, not a forced no-go: DS4 ends at 128.21987328 s, far before its approximately 225 s established pull-off, and an exact native B0 input/threshold lineage on these epochs is unavailable. DS7 and DS8 are jointly evaluated as one non-independent family in `family_metrics.csv` and family-level bootstrap results, so they receive only one-third of the primary macro weight. DS1/DS2 were not opened because no authenticated R2a-aligned tracker binding exists; `diagnostic_scenarios.json` records that fail-closed limitation. No positive detection or active-query contribution claim is made unless its preregistered descriptive controls pass. See `go_no_go.json`, CSV metrics, physical controls, and verifier report for the bounded claims.\n"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def _raw_iq(raw: np.memmap, start: int) -> np.ndarray:
    packed = np.asarray(raw[2 * start:2 * (start + SUPPORT_SAMPLES)]).reshape(-1, 2)
    return packed[:, 0].astype(np.float64) + 1j * packed[:, 1].astype(np.float64)


def post_attack_controls(output: Path, config_path: Path) -> None:
    """Outcome-blind diagnostics derived from frozen seed/policies only."""
    verify_freeze_manifest(output); config = load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, context_model, no_context_model, stats = _load_models(output, device); policy = load(output / "query_policy.json")
    clean, attack = load_cache(output / "clean_features.npz"), load_cache(output / "attack_features.npz")
    pooling = load(output / "pooling.json")["method"]
    rng = np.random.default_rng(config["seed"] + 77)
    learned = list(policy["learned_fixed_order"]); shuffled = [CENTER, *rng.permutation(learned[1:]).tolist()]
    shuffled_nodes = []
    for row in clean + attack:
        score, _ = sequential_trace(model, row["surface"], 9, context_value(row, stats), device, policy="query_shuffle", fixed_order=shuffled)
        shuffled_nodes.append({**{k: row[k] for k in ("scenario", "phase", "second", "time_s", "channel", "prn")}, "method": "learned_order_shuffle", "budget": 9, "score": score})
    shuffled_pooled = pooled_events(shuffled_nodes, pooling)
    cal_values = np.asarray([x["pooled_score"] for x in shuffled_pooled if x["scenario"] == "cleanStatic" and x["phase"] == "calibration"])
    center, scale = float(np.median(cal_values)), max(float(np.quantile(cal_values, .75) - np.quantile(cal_values, .25)), 1e-9)
    for row in shuffled_pooled: row["calibrated_score"] = (row["pooled_score"] - center) / scale
    shuffle_metrics = []
    scenario_rows = list(csv.DictReader((output / "scenario_metrics.csv").open(newline="", encoding="utf-8")))
    for scenario in ("ds3", "ds4", "ds7", "ds8"):
        values = [x for x in shuffled_pooled if x["scenario"] == scenario]; pre = [x for x in values if x["phase"] == "strict_pre"]; post = [x for x in values if x["phase"] != "strict_pre"]
        metric = binary_metrics([0] * len(pre) + [1] * len(post), [x["calibrated_score"] for x in pre + post])
        active = next(x for x in scenario_rows if x["scenario"] == scenario and x["method"] == "active_adaptive" and x["budget"] == "9")
        random_fixed = next(x for x in scenario_rows if x["scenario"] == scenario and x["method"] == "random_fixed" and x["budget"] == "9")
        uniform = next(x for x in scenario_rows if x["scenario"] == scenario and x["method"] == "uniform_fixed" and x["budget"] == "9")
        shuffle_metrics.append({"scenario": scenario, **metric, "active_k9_pauc": float(active["normalized_partial_auc_fpr_0_05"]),
            "random_coordinate_k9_pauc": float(random_fixed["normalized_partial_auc_fpr_0_05"]), "uniform_k9_pauc": float(uniform["normalized_partial_auc_fpr_0_05"]),
            "shuffle_decreased_performance": metric["normalized_partial_auc_fpr_0_05"] < float(active["normalized_partial_auc_fpr_0_05"])})
    family_control_metrics = []
    family_rows = list(csv.DictReader((output / "family_metrics.csv").open(newline="", encoding="utf-8")))
    for family, scenarios in {"ds3": ("ds3",), "ds4": ("ds4",), "ds7_ds8": ("ds7", "ds8")}.items():
        values = [x for x in shuffled_pooled if x["scenario"] in scenarios]
        pre, post = [x for x in values if x["phase"] == "strict_pre"], [x for x in values if x["phase"] != "strict_pre"]
        metric = binary_metrics([0] * len(pre) + [1] * len(post), [x["calibrated_score"] for x in pre + post])
        active = next(x for x in family_rows if x["family"] == family and x["method"] == "active_adaptive" and x["budget"] == "9")
        family_control_metrics.append({"family": family, **metric, "active_k9_pauc": float(active["normalized_partial_auc_fpr_0_05"]),
                                       "shuffle_decreased_performance": metric["normalized_partial_auc_fpr_0_05"] < float(active["normalized_partial_auc_fpr_0_05"])})
    dump(output / "query_policy_controls.json", {"schema": "acaf_nf_stage1_r3_query_controls.v1", "seed": config["seed"] + 77,
        "learned_order_shuffle": shuffled, "calibration_source": "cleanStatic calibration only; diagnostic and not added to frozen thresholds",
        "attack_labels_used_to_construct_order": False, "metrics": shuffle_metrics,
        "family_metrics": family_control_metrics,
        "control_pass": all(x["shuffle_decreased_performance"] for x in family_control_metrics),
        "interpretation": "FAIL means the claimed active-coordinate information contribution is unsupported."})

    # Actual raw-IQ AWGN re-correlation using each support's measured raw RMS.
    raw = np.memmap(config["raw_paths"]["cleanStatic"], dtype="<i2", mode="r")
    holdout = [x for x in clean if x["phase"] == "holdout"][:2]
    threshold_doc, calibration_doc = load(output / "thresholds.json"), load(output / "calibration.json")
    threshold = threshold_doc["active_adaptive:K9"]["q99"]; cal = calibration_doc["active_adaptive:K9"]
    awgn_rows = []
    for target in (35, 40, 45):
        for wi, row in enumerate(holdout):
            surfaces, raw_rms, added_sigma = [], [], []
            for si, state_dict in enumerate(row["states"]):
                state = State(**state_dict); iq = _raw_iq(raw, state.raw_start_sample); rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
                # Added complex variance = measured raw power / ((C/N0)*T), T=1 ms.
                sigma = rms / math.sqrt(2 * 10 ** (target / 10) * .001)
                noise_rng = np.random.default_rng(config["seed"] + target * 100 + wi * 20 + si)
                noisy = iq + noise_rng.normal(0, sigma, iq.size) + 1j * noise_rng.normal(0, sigma, iq.size)
                surfaces.append(complex_caf_surface(noisy, state, carrier_sign=-1)); raw_rms.append(rms); added_sigma.append(sigma)
            aggregate, _ = aggregate_l20(surfaces); score, _ = sequential_trace(model, aggregate.reshape(-1), 9, context_value(row, stats), device, policy="active_adaptive")
            z = (score - cal["center"]) / cal["scale"]
            awgn_rows.append({"target_cn0_db_hz": target, "window": wi, "raw_rms_mean": float(np.mean(raw_rms)), "added_awgn_per_quadrature_sigma_mean": float(np.mean(added_sigma)),
                              "calibrated_score": z, "alarm": bool(z >= threshold)})
    controls = load(output / "physical_controls.json"); controls["awgn_cn0"] = {"method": "actual raw-IQ AWGN followed by complete CAF re-correlation",
        "formula": "sigma_quadrature = measured_raw_rms / sqrt(2 * 10^(target_cn0/10) * 0.001s)", "rows": awgn_rows,
        "fpr_by_target": {str(target): float(np.mean([x["alarm"] for x in awgn_rows if x["target_cn0_db_hz"] == target])) for target in (35, 40, 45)}}
    controls["raw_rms_diagnostic_used_for_scale_audit"] = True; dump(output / "physical_controls.json", controls)

    # Same-PRN raw-IQ second source: construct code/carrier waveforms, correlate
    # the complex mixture, then reapply prompt normalization for every ms.
    base_row = next(x for x in clean if x["phase"] == "holdout")
    authentic_iq, authentic_surfaces = [], []
    for state_dict in base_row["states"]:
        state = State(**state_dict); iq = _raw_iq(raw, state.raw_start_sample); authentic_iq.append(iq); authentic_surfaces.append(complex_caf_surface(iq, state, carrier_sign=-1))
    positive_rows = []
    t = np.arange(SUPPORT_SAMPLES) / FS_HZ
    for ratio in (.25, .75):
        for relative_phase in (0.0, math.pi / 2, math.pi):
            for delay, doppler in ((-.25, -100.0), (-.125, 50.0), (.125, -50.0), (.25, 100.0)):
                mixed_surfaces = []
                for iq, auth, state_dict in zip(authentic_iq, authentic_surfaces, base_row["states"]):
                    state = State(**state_dict)
                    replica = code_replica(state.prn, SUPPORT_SAMPLES, FS_HZ, state.code_freq_chips, state.aux1, -1, delay, replica_direction=1)[0]
                    carrier = np.exp(1j * 2 * np.pi * (state.carrier_doppler_hz + doppler) * t)
                    component = replica * carrier
                    component_surface = complex_caf_surface(component, state, carrier_sign=-1)
                    rotation = np.angle(auth[5, 8]) + relative_phase - np.angle(component_surface[5, 8])
                    amplitude = ratio * abs(auth[5, 8]) / max(abs(component_surface[5, 8]), 1e-12)
                    mixed_surfaces.append(auth + component_surface * amplitude * np.exp(1j * rotation))
                aggregate, _ = aggregate_l20(mixed_surfaces); synthetic = {**base_row, "surface": aggregate.reshape(-1)}
                ref = np.load(output / "normal_field_reference.npz"); analytic = analytic_scores(synthetic, ref["template"], ref["variance"])
                active_score, _ = sequential_trace(model, synthetic["surface"], 9, context_value(base_row, stats), device, policy="active_adaptive")
                positive_rows.append({"power_amplitude_ratio": ratio, "relative_phase_rad": relative_phase, "delay_chips": delay, "residual_doppler_hz": doppler,
                    "active_k9_score": active_score, "analytic_two_source_glrt": analytic["analytic_two_source_glrt"]})
    dump(output / "positive_control_results.json", {"status": "DIAGNOSTIC_ONLY", "construction": "actual authenticated raw IQ plus a generated same-PRN complex code/carrier second source; full CAF re-correlation and per-ms prompt renormalization",
        "different_prn_used": False, "magnitude_only_addition": False, "zero_padding_or_circular_shift_used": False,
        "relative_phase_sweep": [0, math.pi / 2, math.pi], "signed_delay_and_doppler": True, "multiple_power_ratios": True, "results": positive_rows})

    # Actual method timings on clean holdout, excluding raw CAF preprocessing.
    samples = holdout * 4; timing = {}
    orders = {**fixed_policy_orders(), "learned_fixed": policy["learned_fixed_order"]}
    for method, budget in sorted({(row["method"], int(row["budget"])) for row in csv.DictReader((output / "budget_metrics.csv").open(newline="", encoding="utf-8"))}):
        start = time.perf_counter()
        for row in samples:
            receiver = context_value(row, stats)
            if method == "active_adaptive": sequential_trace(model, row["surface"], budget, receiver, device, policy="active_adaptive")
            elif method == "active_magnitude": sequential_trace(model, row["surface"], budget, receiver, device, policy="active_adaptive", magnitude_only=True)
            elif method == "active_context": sequential_trace(context_model, row["surface"], budget, receiver, device, policy="active_adaptive")
            elif method == "active_no_context": sequential_trace(no_context_model, row["surface"], budget, receiver, device, policy="active_adaptive")
            elif method in orders and budget > 0: sequential_trace(model, row["surface"], budget, receiver, device, policy=method, fixed_order=orders[method])
            elif method == "dense_nf": dense_nf_score(model, row, stats, device)
            else:
                ref = np.load(output / "normal_field_reference.npz"); analytic_scores(row, ref["template"], ref["variance"])
        if device.type == "cuda": torch.cuda.synchronize()
        timing[(method, budget)] = 1000 * (time.perf_counter() - start) / len(samples)
    budget_rows = list(csv.DictReader((output / "budget_metrics.csv").open(newline="", encoding="utf-8")))
    for row in budget_rows: row["wall_clock_inference_ms_per_prn"] = timing[(row["method"], int(row["budget"]))]
    write_csv(output / "budget_metrics.csv", budget_rows)
    binding = load(output / "source_binding.json")
    binding["tracker_mat_binding"] = {scenario: tracker_binding_report(config, scenario) for scenario in ("cleanStatic", "ds3", "ds4", "ds7", "ds8")}
    if any(x["status"] != "PASS" for x in binding["tracker_mat_binding"].values()): raise RuntimeError("tracker MAT binding failed in post-attack audit")
    dump(output / "source_binding.json", binding)
    family_rows = list(csv.DictReader((output / "family_metrics.csv").open(newline="", encoding="utf-8")))
    def family_values(method: str, budget: int) -> dict[str, float]:
        return {x["family"]: float(x["normalized_partial_auc_fpr_0_05"]) for x in family_rows if x["method"] == method and int(x["budget"]) == budget}
    active, fixed, dense = family_values("active_adaptive", 9), family_values("fixed_delay_9", 9), family_values("dense_nf", 187)
    random_values, uniform_values, power_values = family_values("random_fixed", 9), family_values("uniform_fixed", 9), family_values("raw_power_only", 0)
    families = ("ds3", "ds4", "ds7_ds8")
    active_beats_fixed = sum(active[x] > fixed[x] for x in families)
    active_within_dense = sum(active[x] >= dense[x] - config["go_no_go_gates_frozen_before_attack"]["dense_pauc_tolerance"] for x in families)
    active_beats_random_uniform = sum(active[x] > random_values[x] and active[x] > uniform_values[x] for x in families)
    query_pass = load(output / "query_policy_controls.json")["control_pass"]
    power_excluded = float(np.mean(list(active.values()))) > float(np.mean(list(power_values.values())))
    go = load(output / "go_no_go.json")
    go["detection_gate_descriptive_only"] = {
        "family_weighting": "DS3, DS4, pooled DS7/DS8 each weight 1/3",
        "active_k9_macro_pauc": float(np.mean(list(active.values()))), "power_only_macro_pauc": float(np.mean(list(power_values.values()))),
        "active_beats_same_k_fixed_primary_families": active_beats_fixed, "active_beats_same_k_fixed_required_families": 2,
        "active_k9_within_5pct_dense_primary_families": active_within_dense,
        "active_outperforms_random_and_uniform_primary_families": active_beats_random_uniform,
        "query_shuffle_decreased_all_three_primary_families": query_pass, "power_only_explanation_excluded": power_excluded,
        "active_query_gate_pass": active_beats_random_uniform >= 2 and query_pass,
        "detection_gate_pass": active_beats_fixed >= 2 and active_within_dense >= 2 and active_beats_random_uniform >= 2 and query_pass and power_excluded,
    }
    dump(output / "go_no_go.json", go)
    print(json.dumps({"stage": "post_attack_controls_complete", "shuffle_control_pass": load(output / "query_policy_controls.json")["control_pass"], "awgn_raw_windows": len(awgn_rows), "positive_sweeps": len(positive_rows)}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=("clean", "attack", "controls", "all")); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG); parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1)); args = parser.parse_args()
    if args.phase in ("clean", "all"): clean_phase(args.output.resolve(), args.config.resolve(), args.workers)
    if args.phase in ("attack", "all"): attack_phase_run(args.output.resolve(), args.config.resolve(), args.workers)
    if args.phase in ("controls", "all"): post_attack_controls(args.output.resolve(), args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
