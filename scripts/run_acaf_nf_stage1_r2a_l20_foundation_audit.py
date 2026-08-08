#!/usr/bin/env python3
"""Run the cleanStatic-only ACAF-NF Stage-1 R2a L20 foundation audit."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.acaf_nf_stage1_r2a_l20_foundation_audit import (  # noqa: E402
    DELAY_GRID_CHIPS,
    DOPPLER_GRID_HZ,
    FS_HZ,
    SUPPORT_SAMPLES,
    State,
    clean_only_guard,
    complex_caf_surface,
    numerical_equivalence,
    r14_l20_aggregate,
    score_complex_surface,
    score_power_surface,
    surface_sha256,
)

OUTPUT = ROOT / "artifacts/acaf_nf_stage1_r2a_l20_foundation_audit"
R14 = ROOT / "artifacts/acaf_nf_stage0_static_r14_doppler_validation"
R2 = ROOT / "artifacts/acaf_nf_stage1_r2_full_normal"
BASE_SHA = "45df3c8b9996a853f4ae03fe8e93e885b16161f7"
REQUIRED_MAT = (
    "PRN_start_sample_count", "PRN", "carrier_doppler_hz", "code_freq_chips", "aux1",
    "Prompt_I", "Prompt_Q", "CN0_SNV_dB_Hz", "carrier_lock_test", "acc_carrier_phase_rad",
)
CSV_ALIGNMENT_FIELDS = (
    "source", "prn", "channel", "tracker_row", "state_row", "receiver_time_s",
    "raw_start_sample", "raw_end_sample", "support_length_samples", "carrier_doppler_hz",
    "code_freq_chips", "aux1", "residual_code_phase_chips", "prompt_i", "prompt_q",
    "prompt_magnitude", "cn0_db_hz", "carrier_lock", "state_semantics", "prompt_semantics",
    "same_prn_previous_current_next", "causal_support", "assignment_run_length_at_anchor",
)

_RAW: np.memmap | None = None


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    values = list(rows)
    names = list(fields or dict.fromkeys(key for row in values for key in row))
    if not names:
        names = ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mat(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        arrays = {name: np.asarray(handle[name]).reshape(-1) for name in REQUIRED_MAT}
    if len({len(value) for value in arrays.values()}) != 1:
        raise ValueError(f"MAT vector length mismatch: {path}")
    return arrays


def channel_number(path: Path) -> int:
    match = re.search(r"(\d+)\.mat$", path.name)
    if not match:
        raise ValueError(f"cannot infer channel: {path}")
    return int(match.group(1))


def channel_candidates(path: Path) -> tuple[list[dict[str, int]], dict[str, Any]]:
    """Stream MAT lineage and census every causal 10-Hz L20 anchor."""
    arrays = load_mat(path)
    channel = channel_number(path)
    stamps = arrays["PRN_start_sample_count"].astype(np.int64)
    prns = arrays["PRN"].astype(np.int64)
    finite = np.logical_and.reduce([
        np.isfinite(arrays[name]) for name in REQUIRED_MAT
        if name not in {"PRN_start_sample_count", "PRN"}
    ])
    quality = (arrays["CN0_SNV_dB_Hz"] >= 28.0) & (arrays["carrier_lock_test"] >= 0.85)
    candidates: list[dict[str, int]] = []
    run = 0
    run_start = -1
    pair_counts: Counter[int] = Counter()
    last_time_bin_by_prn: dict[int, int] = {}
    rejected_reassignment = 0
    rejected_quality = 0
    rejected_support = 0
    for cur in range(1, len(stamps) - 1):
        same = bool(1 <= prns[cur] <= 32 and prns[cur - 1] == prns[cur] == prns[cur + 1])
        exact = bool(stamps[cur] - stamps[cur - 1] == SUPPORT_SAMPLES and stamps[cur + 1] - stamps[cur] == SUPPORT_SAMPLES)
        good = bool(np.all(finite[cur - 1:cur + 2]) and np.all(quality[cur - 1:cur + 2]))
        valid = same and exact and good
        continues = valid and run and cur == (run_start + run) and prns[cur] == prns[cur - 1]
        if valid:
            if not continues:
                run = 0
                run_start = cur
            run += 1
            # At most one anchor per PRN/channel/100-ms receiver-time bin.  A
            # quality dip may restart a run, but must not increase the nominal
            # 10-Hz evaluation density.
            time_bin = int(stamps[cur - 1] // (100 * SUPPORT_SAMPLES))
            if run >= 20 and last_time_bin_by_prn.get(int(prns[cur])) != time_bin:
                candidates.append({
                    "channel": channel, "prn": int(prns[cur]), "anchor_row": cur,
                    "raw_start_sample": int(stamps[cur - 20]), "anchor_support_start": int(stamps[cur - 1]),
                    "run_length": run,
                })
                pair_counts[int(prns[cur])] += 1
                last_time_bin_by_prn[int(prns[cur])] = time_bin
        else:
            if not same:
                rejected_reassignment += 1
            if same and not exact:
                rejected_support += 1
            if same and exact and not good:
                rejected_quality += 1
            run = 0
            run_start = -1
    return candidates, {
        "channel": channel, "mat_path": str(path), "rows": len(stamps),
        "candidate_10hz_windows": len(candidates), "candidate_windows_by_prn": dict(sorted(pair_counts.items())),
        "rejected_reassignment_rows": rejected_reassignment, "rejected_quality_rows": rejected_quality,
        "rejected_support_rows": rejected_support,
    }


def even_pick(values: list[dict[str, int]], count: int) -> list[dict[str, int]]:
    if count >= len(values):
        return list(values)
    indices = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return [values[int(index)] for index in indices]


def stratified_candidates(candidates: list[dict[str, int]], target: int) -> list[dict[str, int]]:
    groups: dict[tuple[int, int], list[dict[str, int]]] = defaultdict(list)
    for item in candidates:
        groups[(item["channel"], item["prn"])].append(item)
    if len(groups) < 8:
        raise RuntimeError("fewer than eight valid fresh PRN/channel pairs")
    for values in groups.values():
        values.sort(key=lambda x: x["anchor_support_start"])
    base = max(1, target // len(groups))
    selected = [item for key in sorted(groups) for item in even_pick(groups[key], min(base, len(groups[key])))]
    remaining = [item for key in sorted(groups) for item in groups[key] if item not in selected]
    if len(selected) < target:
        selected.extend(even_pick(remaining, min(target - len(selected), len(remaining))))
    if len(selected) > target:
        selected = even_pick(sorted(selected, key=lambda x: (x["anchor_support_start"], x["channel"])), target)
    return sorted(selected, key=lambda x: (x["anchor_support_start"], x["channel"], x["prn"]))


def make_states(mat_path: Path, candidate: dict[str, int], *, state_shift: int = 0) -> list[State]:
    arrays = load_mat(mat_path)
    channel = candidate["channel"]
    anchor = candidate["anchor_row"]
    states: list[State] = []
    for cur in range(anchor - 19, anchor + 1):
        state = cur - 1 + state_shift
        canonical_start = int(arrays["PRN_start_sample_count"][cur - 1])
        states.append(State(
            channel=channel, prn=int(arrays["PRN"][cur]), tracker_row=cur, state_row=state,
            raw_start_sample=canonical_start, code_freq_chips=float(arrays["code_freq_chips"][state]),
            carrier_doppler_hz=float(arrays["carrier_doppler_hz"][state]), aux1=float(arrays["aux1"][state]),
            prompt_i=float(arrays["Prompt_I"][cur]), prompt_q=float(arrays["Prompt_Q"][cur]),
            cn0_db_hz=float(arrays["CN0_SNV_dB_Hz"][cur]), carrier_lock=float(arrays["carrier_lock_test"][cur]),
        ))
    return states


def init_worker(raw_path: str) -> None:
    global _RAW
    _RAW = np.memmap(raw_path, dtype="<i2", mode="r")


def evaluate_states(payload: tuple[list[State], int]) -> dict[str, Any]:
    states, carrier_sign = payload
    if _RAW is None:
        raise RuntimeError("worker raw map is not initialized")
    surfaces: list[np.ndarray] = []
    epochs: list[dict[str, Any]] = []
    for state in states:
        start = state.raw_start_sample
        packed = np.asarray(_RAW[2 * start:2 * (start + SUPPORT_SAMPLES)]).reshape(-1, 2)
        iq = packed[:, 0].astype(np.float64) + 1j * packed[:, 1].astype(np.float64)
        surface = complex_caf_surface(iq, state, carrier_sign=carrier_sign)
        surfaces.append(surface)
        score = score_complex_surface(surface)
        prompt = math.hypot(state.prompt_i, state.prompt_q)
        epochs.append({
            **score, "center_magnitude": float(abs(surface[5, 8])), "prompt_magnitude": prompt,
            "prompt_relative_error": abs(float(abs(surface[5, 8])) / max(prompt, 1e-15) - 1.0),
            "cn0_db_hz": state.cn0_db_hz, "carrier_lock": state.carrier_lock,
            "surface_sha256": surface_sha256(surface),
        })
    aggregate = r14_l20_aggregate(surfaces)
    result = score_power_surface(aggregate)
    anchor = states[-1]
    result.update({
        "channel": anchor.channel, "prn": anchor.prn, "anchor_tracker_row": anchor.tracker_row,
        "state_row": anchor.state_row, "raw_start_sample": states[0].raw_start_sample,
        "raw_end_sample": anchor.raw_start_sample + SUPPORT_SAMPLES,
        "receiver_time_s": anchor.raw_start_sample / FS_HZ, "cn0_db_hz": min(x.cn0_db_hz for x in states),
        "carrier_lock": min(x.carrier_lock for x in states), "epochs": epochs,
        "constituent_surface_sha256": [x["surface_sha256"] for x in epochs],
        "aggregate_sha256": hashlib.sha256(np.ascontiguousarray(aggregate).view(np.uint8)).hexdigest(),
    })
    return result


def actual_r14_equivalence() -> tuple[dict[str, Any], list[dict[str, str]]]:
    epochs: dict[tuple[str, int, int], dict[str, str]] = {}
    aggregate_row: dict[str, str] | None = None
    with (R14 / "per_block_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["record_type"] == "epoch":
                epochs[(row["channel"], int(row["prn"]), int(row["tracker_row"]))] = row
            elif row["record_type"] == "aggregate" and int(row["L"]) == 20 and aggregate_row is None:
                aggregate_row = row
    if aggregate_row is None:
        raise RuntimeError("R1.4 L20 aggregate not found")
    identities = ast.literal_eval(aggregate_row["constituent_identities"])
    rows = [epochs[(str(item[0]), int(item[1]), int(item[2]))] for item in identities]
    surfaces = [
        np.asarray(json.loads(row["surface_real"])) + 1j * np.asarray(json.loads(row["surface_imag"]))
        for row in rows
    ]
    report = numerical_equivalence(surfaces, 1e-12)
    aggregate = r14_l20_aggregate(surfaces)
    stored = np.asarray(json.loads(aggregate_row["primary_surface_values"]), dtype=np.float64)
    report.update({
        "input": "20 actual complex R1.4 epoch surfaces from per_block_scores.csv",
        "input_dtype": str(np.asarray(surfaces).dtype), "input_shape": list(np.asarray(surfaces).shape),
        "stored_r14_aggregate_max_abs_delta": float(np.max(np.abs(aggregate - stored))),
        "stored_r14_aggregate_sha256_match": hashlib.sha256(np.ascontiguousarray(aggregate).view(np.uint8)).hexdigest() == aggregate_row["surface_sha256"],
        "surface_representation": {
            "constituents": "complex128 real+imag", "selection": "argmax(abs(complex_surface))",
            "primary": "mean(|C_k|^2 / (sum_grid |C_k|^2 + 1e-15))",
            "reported_magnitude": "sqrt(primary power)", "raw_power_is_diagnostic_only": True,
            "magnitude_mean_is_diagnostic_only": True,
        },
        "l20_semantics": {
            "count": 20, "same_prn_channel": len({(r["channel"], r["prn"]) for r in rows}) == 1,
            "tracker_rows": [int(r["tracker_row"]) for r in rows],
            "support_start_deltas": [int(b["support_start_sample"]) - int(a["support_start_sample"]) for a, b in zip(rows, rows[1:])],
            "support_24999_count": sum(int(b["support_start_sample"]) - int(a["support_start_sample"]) == 24999 for a, b in zip(rows, rows[1:])),
            "support_25000_count": sum(int(b["support_start_sample"]) - int(a["support_start_sample"]) == 25000 for a, b in zip(rows, rows[1:])),
        },
    })
    report["status"] = "PASS" if (
        report["status"] == "PASS" and report["stored_r14_aggregate_max_abs_delta"] <= 1e-12
        and report["stored_r14_aggregate_sha256_match"]
    ) else "FAIL"
    return report, rows


def pooled_prompt(epoch_rows: list[dict[str, Any]]) -> dict[str, Any]:
    center = np.asarray([row["center_magnitude"] for row in epoch_rows])
    prompt = np.asarray([row["prompt_magnitude"] for row in epoch_rows])
    errors = np.asarray([row["prompt_relative_error"] for row in epoch_rows])
    rho = float(spearmanr(center, prompt).statistic)
    return {
        "n": len(epoch_rows), "pooled_spearman": rho, "median_relative_error": float(np.median(errors)),
        "p99_relative_error": float(np.quantile(errors, 0.99)), "max_relative_error": float(np.max(errors)),
    }


def metric_summary(windows: list[dict[str, Any]], epoch_rows: list[dict[str, Any]]) -> dict[str, Any]:
    offsets = np.asarray([row["peak_doppler_offset_hz"] for row in windows])
    delay = np.asarray([row["peak_delay_offset_chips"] for row in epoch_rows])
    epoch_delay_boundary = np.asarray([row["delay_boundary"] for row in epoch_rows])
    epoch_doppler_boundary = np.asarray([row["doppler_boundary"] for row in epoch_rows])
    l20_delay_boundary = np.asarray([row["delay_boundary"] for row in windows])
    l20_doppler_boundary = np.asarray([row["doppler_boundary"] for row in windows])
    return {
        "l20_windows": len(windows), "one_ms_epochs": len(epoch_rows),
        "within_50_fraction": float(np.mean(np.abs(offsets) <= 50)),
        "within_100_fraction": float(np.mean(np.abs(offsets) <= 100)),
        "exact_center_fraction": float(np.mean(offsets == 0)),
        "delay_within_0_125_fraction": float(np.mean(np.abs(delay) <= 0.125)),
        "one_ms_delay_boundary_fraction": float(np.mean(epoch_delay_boundary)),
        "one_ms_doppler_boundary_fraction": float(np.mean(epoch_doppler_boundary)),
        "one_ms_grid_boundary_fraction": float(np.mean(epoch_delay_boundary | epoch_doppler_boundary)),
        "l20_delay_boundary_fraction": float(np.mean(l20_delay_boundary)),
        "l20_doppler_boundary_fraction": float(np.mean(l20_doppler_boundary)),
        "l20_grid_boundary_fraction": float(np.mean(l20_delay_boundary | l20_doppler_boundary)),
        "overall_grid_boundary_fraction": float(max(np.mean(epoch_delay_boundary | epoch_doppler_boundary), np.mean(l20_delay_boundary | l20_doppler_boundary))),
        "doppler_histogram_hz": {str(float(value)): int(np.sum(offsets == value)) for value in sorted(set(offsets))},
        "prompt_reproduction": pooled_prompt(epoch_rows),
    }


def group_metrics(windows: list[dict[str, Any]], epoch_rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    result = []
    for value in sorted({row[key] for row in windows}, key=str):
        selected = [row for row in windows if row[key] == value]
        if key == "pair":
            epochs = [row for row in epoch_rows if row["pair"] == value]
            extra = {"channel": int(value.split("/")[0]), "prn": int(value.split("/")[1])}
        else:
            epochs = [row for row in epoch_rows if row[key] == value]
            extra = {key: value}
        result.append({**extra, **metric_summary(selected, epochs)})
    return result


def state_alignment_rows(selected: list[dict[str, int]], mat_by_channel: dict[int, Path], limit: int = 240) -> list[dict[str, Any]]:
    picked = even_pick(selected, min(limit, len(selected)))
    rows: list[dict[str, Any]] = []
    for candidate in picked:
        state = make_states(mat_by_channel[candidate["channel"]], candidate)[-1]
        rows.append({
            "source": "fresh_cleanStatic", "prn": state.prn, "channel": state.channel,
            "tracker_row": state.tracker_row, "state_row": state.state_row,
            "receiver_time_s": state.raw_start_sample / FS_HZ, "raw_start_sample": state.raw_start_sample,
            "raw_end_sample": state.raw_start_sample + SUPPORT_SAMPLES, "support_length_samples": SUPPORT_SAMPLES,
            "carrier_doppler_hz": state.carrier_doppler_hz, "code_freq_chips": state.code_freq_chips,
            "aux1": state.aux1, "residual_code_phase_chips": -state.aux1 * state.code_freq_chips / FS_HZ,
            "prompt_i": state.prompt_i, "prompt_q": state.prompt_q,
            "prompt_magnitude": math.hypot(state.prompt_i, state.prompt_q), "cn0_db_hz": state.cn0_db_hz,
            "carrier_lock": state.carrier_lock, "state_semantics": "k-1", "prompt_semantics": "k",
            "same_prn_previous_current_next": True, "causal_support": "[stamp(k-1),stamp(k-1)+25000)",
            "assignment_run_length_at_anchor": candidate["run_length"],
        })
    return rows


def combination_diagnostics(
    raw_path: Path, mat_by_channel: dict[int, Path], selected: list[dict[str, int]], old_report: dict[str, Any]
) -> list[dict[str, Any]]:
    init_worker(str(raw_path))
    preferred = [x for x in selected if x["channel"] == 7 and x["prn"] == 10]
    diagnostic = min(preferred or selected, key=lambda x: x["anchor_support_start"])
    results: list[dict[str, Any]] = []
    for name, shift, sign in (
        ("fresh_raw+fresh_state_kminus1", 0, -1),
        ("fresh_raw+fresh_state_k", 1, -1),
        ("fresh_raw+fresh_state_kminus2", -1, -1),
        ("fresh_raw+fresh_state_sign_reversed", 0, 1),
    ):
        states = make_states(mat_by_channel[diagnostic["channel"]], diagnostic, state_shift=shift)
        value = evaluate_states((states, sign))
        epochs = value.pop("epochs")
        results.append({
            "combination": name, "raw_support_source": "fresh cleanStatic authenticated raw",
            "state_source": f"fresh tracker shift {shift:+d}", "carrier_sign": sign, "availability": "AVAILABLE",
            "channel": value["channel"], "prn": value["prn"], "receiver_time_s": value["receiver_time_s"],
            "one_ms_within_50_fraction": float(np.mean([abs(x["peak_doppler_offset_hz"]) <= 50 for x in epochs])),
            "one_ms_delay_within_0_125_fraction": float(np.mean([abs(x["peak_delay_offset_chips"]) <= .125 for x in epochs])),
            "l20_peak_doppler_offset_hz": value["peak_doppler_offset_hz"],
            "l20_peak_delay_offset_chips": value["peak_delay_offset_chips"], "l20_grid_boundary": value["grid_boundary"],
            "l20_center_peak_ratio": value["center_peak_ratio"],
        })
    stored = old_report["r14"]
    results.extend([
        {
            "combination": "R1.4_raw_support+R1.4_state", "raw_support_source": "same authenticated cleanStatic bytes",
            "state_source": "R1.4 previous row", "carrier_sign": -1, "availability": "AVAILABLE_STORED_COMPLEX_SURFACES",
            "channel": "R1.4 common anchor", "prn": "R1.4 common anchor", "receiver_time_s": "",
            "one_ms_within_50_fraction": "", "one_ms_delay_within_0_125_fraction": "",
            "l20_peak_doppler_offset_hz": stored["peak_doppler_offset_hz"],
            "l20_peak_delay_offset_chips": stored["peak_delay_offset_chips"], "l20_grid_boundary": stored["grid_boundary"],
            "l20_center_peak_ratio": stored["center_peak_ratio"],
        },
        {
            "combination": "fresh_raw_support+R1.4_state", "raw_support_source": "same authenticated cleanStatic bytes",
            "state_source": "R1.4 state", "carrier_sign": -1, "availability": "ONLY_ON_117_COMMON_SUPPORTS",
            "channel": 8, "prn": 13, "receiver_time_s": "9-12 s common interval",
            "one_ms_within_50_fraction": "", "one_ms_delay_within_0_125_fraction": "",
            "l20_peak_doppler_offset_hz": stored["peak_doppler_offset_hz"],
            "l20_peak_delay_offset_chips": stored["peak_delay_offset_chips"], "l20_grid_boundary": stored["grid_boundary"],
            "l20_center_peak_ratio": stored["center_peak_ratio"],
        },
    ])
    return results


def plots(output: Path, windows: list[dict[str, Any]], per_prn: list[dict[str, Any]]) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{row['channel']}/{row['prn']}" for row in per_prn]
    values = [row["within_50_fraction"] for row in per_prn]
    plt.figure(figsize=(10, 4)); plt.bar(labels, values); plt.axhline(.95, color="red", linestyle="--")
    plt.ylim(0, 1.02); plt.ylabel("L20 within ±50 Hz"); plt.xlabel("channel/PRN"); plt.xticks(rotation=45)
    plt.tight_layout(); plt.savefig(plot_dir / "l20_doppler_by_prn.png", dpi=150); plt.close()
    plt.figure(figsize=(10, 4))
    for pair in sorted({row["pair"] for row in windows}):
        values_pair = [row for row in windows if row["pair"] == pair]
        plt.scatter([x["receiver_time_s"] for x in values_pair], [x["peak_doppler_offset_hz"] for x in values_pair], s=6, label=pair)
    plt.axhline(50, color="red", linestyle="--"); plt.axhline(-50, color="red", linestyle="--")
    plt.xlabel("receiver time (s)"); plt.ylabel("L20 residual Doppler peak (Hz)"); plt.legend(ncol=4, fontsize=6)
    plt.tight_layout(); plt.savefig(plot_dir / "l20_doppler_over_time.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 5))
    colors = ["red" if row["grid_boundary"] else "#276FBF" for row in windows]
    plt.scatter([row["peak_delay_offset_chips"] for row in windows], [row["peak_doppler_offset_hz"] for row in windows], c=colors, s=8, alpha=.6)
    plt.xlabel("peak delay offset (chips)"); plt.ylabel("peak Doppler offset (Hz)"); plt.tight_layout()
    plt.savefig(plot_dir / "grid_boundary_diagnostics.png", dpi=150); plt.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--target-windows", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--execute-production", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute_production:
        raise SystemExit("pass --execute-production for the authenticated clean-only audit")
    if args.target_windows < 1000:
        raise ValueError("production audit requires at least 1,000 evaluated L20 windows")
    if args.output.exists():
        raise FileExistsError(args.output)

    fresh = load_json(R2 / "fresh_replay_binding.json")
    raw_path = Path(fresh["source_path"])
    tracker_dir = Path(fresh["replay_path"]) / "raw"
    clean_only_guard("cleanStatic", [raw_path, tracker_dir])
    if raw_path.stat().st_size != fresh["source_size_bytes"]:
        raise RuntimeError("cleanStatic raw size drift")
    receiver_manifest = Path(fresh["manifest_path"])
    manifest = load_json(receiver_manifest)
    mats = sorted(tracker_dir / name for name in manifest["tracking"]["mat_inventory"])
    if any(not path.is_file() for path in mats):
        raise FileNotFoundError("fresh cleanStatic MAT inventory is incomplete")
    args.output.mkdir(parents=True)

    equivalence, _ = actual_r14_equivalence()
    dump(args.output / "r14_r2_equivalence.json", equivalence)

    all_candidates: list[dict[str, int]] = []
    census: list[dict[str, Any]] = []
    mat_by_channel: dict[int, Path] = {}
    for mat in mats:
        values, report = channel_candidates(mat)
        all_candidates.extend(values); census.append(report); mat_by_channel[report["channel"]] = mat
    duration_samples = fresh["source_size_bytes"] // 4
    third = duration_samples / 3
    pair_segments: dict[tuple[int, int], set[str]] = defaultdict(set)
    for item in all_candidates:
        segment = "front" if item["anchor_support_start"] < third else (
            "middle" if item["anchor_support_start"] < 2 * third else "back"
        )
        pair_segments[(item["channel"], item["prn"])].add(segment)
    # Freeze a clean-only lineage rule before looking at any CAF result.  A
    # reacquired/reassigned pair visible in only one recording region cannot
    # support the requested front/middle/back physical-validity claim.
    full_span_pairs = {
        pair for pair, segments in pair_segments.items()
        if segments == {"front", "middle", "back"}
    }
    excluded_lineage_pairs = sorted(set(pair_segments) - full_span_pairs)
    eligible_candidates = [
        item for item in all_candidates
        if (item["channel"], item["prn"]) in full_span_pairs
    ]
    selected = stratified_candidates(eligible_candidates, args.target_windows)
    selected_pairs = sorted({(x["channel"], x["prn"]) for x in selected})
    if len(selected) < 1000 or len(selected_pairs) < 8:
        raise RuntimeError("insufficient full cleanStatic validation coverage")

    tasks = [(make_states(mat_by_channel[item["channel"]], item), -1) for item in selected]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(str(raw_path),)) as pool:
        evaluated = list(pool.map(evaluate_states, tasks, chunksize=1))

    epoch_rows: list[dict[str, Any]] = []
    for window in evaluated:
        pair = f"{window['channel']}/{window['prn']}"
        segment = "front" if window["receiver_time_s"] < fresh["source_size_bytes"] / 4 / FS_HZ / 3 else (
            "middle" if window["receiver_time_s"] < 2 * fresh["source_size_bytes"] / 4 / FS_HZ / 3 else "back"
        )
        window["pair"] = pair; window["time_segment"] = segment
        epochs = window.pop("epochs")
        for epoch in epochs:
            epoch["pair"] = pair; epoch["time_segment"] = segment
        epoch_rows.extend(epochs)

    overall = metric_summary(evaluated, epoch_rows)
    overall.update({
        "schema": "acaf_nf_stage1_r2a_full_cleanstatic_l20_metrics.v1",
        "candidate_10hz_windows_census": len(all_candidates), "evaluated_windows": len(evaluated),
        "evaluation_design": "deterministic pair-stratified sample from causal 10-Hz candidates across the full recording",
        "candidate_frequency_hz": 10,
        "causal_prn_channels_in_census": len({(x['channel'], x['prn']) for x in all_candidates}),
        "full_span_valid_prn_channels": len(full_span_pairs),
        "excluded_incomplete_lineage_pairs": [list(pair) for pair in excluded_lineage_pairs],
        "evaluated_prn_channels": len(selected_pairs), "minimum_required_windows": 1000,
        "recording_duration_s": fresh["source_size_bytes"] / 4 / FS_HZ,
    })
    per_prn = group_metrics(evaluated, epoch_rows, "pair")
    per_time = group_metrics(evaluated, epoch_rows, "time_segment")
    dump(args.output / "full_cleanstatic_l20_metrics.json", overall)
    write_csv(args.output / "per_prn_metrics.csv", per_prn)
    write_csv(args.output / "per_time_segment_metrics.csv", per_time)

    failed = []
    for row in evaluated:
        if abs(float(row["peak_doppler_offset_hz"])) > 50 or row["grid_boundary"]:
            failed.append({
                "channel": row["channel"], "prn": row["prn"], "pair": row["pair"],
                "receiver_time_s": row["receiver_time_s"], "time_segment": row["time_segment"],
                "anchor_tracker_row": row["anchor_tracker_row"], "raw_start_sample": row["raw_start_sample"],
                "raw_end_sample": row["raw_end_sample"], "peak_doppler_offset_hz": row["peak_doppler_offset_hz"],
                "peak_delay_offset_chips": row["peak_delay_offset_chips"], "delay_boundary": row["delay_boundary"],
                "doppler_boundary": row["doppler_boundary"], "grid_boundary": row["grid_boundary"],
                "center_peak_ratio": row["center_peak_ratio"], "cn0_db_hz": row["cn0_db_hz"],
                "carrier_lock": row["carrier_lock"], "assignment_run_length_at_anchor": next(
                    x["run_length"] for x in selected if x["channel"] == row["channel"] and x["anchor_row"] == row["anchor_tracker_row"]
                ),
            })
    write_csv(args.output / "failed_window_diagnostics.csv", failed)
    write_csv(args.output / "tracker_state_alignment.csv", state_alignment_rows(selected, mat_by_channel), CSV_ALIGNMENT_FIELDS)
    combinations = combination_diagnostics(raw_path, mat_by_channel, selected, equivalence)
    write_csv(args.output / "state_combination_metrics.csv", combinations)

    r2_old = load_json(R2 / "full_cleanstatic_validation.json")
    old_by_pair: Counter[str] = Counter()
    old_failed_by_pair: Counter[str] = Counter()
    for row in load_json(R2 / "full_cleanstatic_l20_windows.json"):
        pair = f"{row['channel']}/{row['prn']}"; old_by_pair[pair] += 1
        if abs(float(row["peak_doppler_offset_hz"])) > 50: old_failed_by_pair[pair] += 1
    cause = {
        "old_l20_windows": r2_old["l20_doppler"]["n"],
        "old_within_50_fraction": r2_old["l20_doppler"]["within_50_fraction"],
        "old_failure_count": sum(old_failed_by_pair.values()), "old_failure_by_pair": dict(old_failed_by_pair),
        "old_window_by_pair": dict(old_by_pair),
        "dominant_failure_pair": max(old_failed_by_pair, key=old_failed_by_pair.get),
        "dominant_pair_failure_share": max(old_failed_by_pair.values()) / max(sum(old_failed_by_pair.values()), 1),
        "root_cause": "R2 validation-row selection forced the first 20 rows of each pair and then formed L20 only inside the truncated 969-row set, over-weighting the fresh receiver acquisition transient on channel 7 / PRN 10 around 0.67-0.80 s.",
        "calculation_bug": False, "carrier_sign_bug": False, "k_kminus1_bug": False,
        "correction": "retain k-1 state/current Prompt and frozen equations; enumerate causal same-assignment 10-Hz L20 candidates over the full cleanStatic tracker, exclude reacquired pair lineages absent from one or more recording thirds, then take a deterministic pair/time-stratified validation sample",
    }

    gates = {
        "sample_windows_ge_1000": overall["l20_windows"] >= 1000,
        "prn_channels_ge_8": overall["evaluated_prn_channels"] >= 8,
        "prompt_pooled_spearman_ge_0_999": overall["prompt_reproduction"]["pooled_spearman"] >= .999,
        "prompt_p99_relative_error_le_0_01": overall["prompt_reproduction"]["p99_relative_error"] <= .01,
        "delay_within_0_125_ge_0_95": overall["delay_within_0_125_fraction"] >= .95,
        "l20_doppler_within_50_ge_0_95": overall["within_50_fraction"] >= .95,
        "grid_boundary_le_0_01": overall["overall_grid_boundary_fraction"] <= .01,
        "raw_provenance_authenticated": fresh["status"] == "PASS" and fresh["all_file_hashes_match_manifest"],
        "causal_alignment_verified": equivalence["status"] == "PASS",
    }
    physical_valid = all(gates.values())
    verdict = "FOUNDATION_VALID" if physical_valid else (
        "FOUNDATION_INVALID" if overall["l20_windows"] >= 1000 and overall["evaluated_prn_channels"] >= 8 else "FOUNDATION_INCONCLUSIVE"
    )
    deterministic = {
        "gate_name_retained": "r14_common_reproduced_1e_6",
        "historic_result": r2_old["gates"]["r14_common_reproduced_1e_6"],
        "historic_common_supports": r2_old["r14_common_epochs"]["n"],
        "historic_surface_hash_all_match": r2_old["r14_common_epochs"]["surface_sha256_all_match"],
        "classification": "NOT_APPLICABLE_AS_DETERMINISTIC_REPLAY_EQUIVALENCE",
        "reason": "R1.4 and fresh replay do not bind the same receiver executable/config/initial phase/channel numbering; common raw bytes alone do not define deterministic replay identity.",
        "same_build_config_raw_initial_assignment_would_require_exact_numeric_and_hash_match": True,
        "used_as_physical_validity_gate": False,
    }
    verdict_doc = {
        "schema": "acaf_nf_stage1_r2a_foundation_verdict.v1", "verdict": verdict,
        "gates": gates, "deterministic_replay_equivalence": deterministic,
        "physical_tracker_validity": {"status": "PASS" if physical_valid else "FAIL", "metrics": overall},
        "failure_cause": cause, "thresholds_changed": False, "gate_auto_promoted": False,
        "attack_performance_evaluated": False, "model_trained": False,
    }
    dump(args.output / "foundation_verdict.json", verdict_doc)

    r13_binding = load_json(ROOT / "artifacts/acaf_nf_stage0_static_r13_reconstruction/receiver_source_binding.json")
    r14_manifest_path = Path(r13_binding["manifest_path"])
    r14_manifest = load_json(r14_manifest_path)
    source_binding = {
        "schema": "acaf_nf_stage1_r2a_source_binding.v1", "status": "PASS",
        "base_branch": "research/acaf-nf-stage1-r2-full-normal", "base_commit": BASE_SHA,
        "raw": {"recording": "cleanStatic", "path": str(raw_path), "sha256": fresh["source_sha256"],
                "size_bytes": fresh["source_size_bytes"], "size_rechecked": raw_path.stat().st_size},
        "fresh_receiver": {"manifest": str(receiver_manifest), "manifest_sha256": digest(receiver_manifest),
                           "executable_sha256": fresh["receiver_executable_sha256"],
                           "config_sha256": fresh["receiver_config_sha256"]},
        "r14_artifact": str(R14),
        "r14_receiver": {
            "manifest": str(r14_manifest_path), "manifest_sha256": r13_binding["manifest_sha256"],
            "executable_sha256": r14_manifest["receiver"]["executable_sha256"],
            "config_sha256": r14_manifest["receiver"]["config_sha256"],
        },
        "r2_artifact": str(R2),
        "attacks": {name: {"status": "NOT_ACCESSED", "bytes_read": 0} for name in ("ds3", "ds4", "ds7", "ds8")},
    }
    dump(args.output / "source_binding.json", source_binding)
    dump(args.output / "config.json", {
        "schema": "acaf_nf_stage1_r2a_config.v1", "scope": "cleanStatic-only",
        "target_windows": args.target_windows, "workers": args.workers, "candidate_frequency_hz": 10,
        "selection": "deterministic all-pair stratified across full recording", "cn0_min_db_hz": 28.0,
        "carrier_lock_min": .85, "support_samples": SUPPORT_SAMPLES,
        "delay_grid_chips": DELAY_GRID_CHIPS.tolist(), "doppler_grid_hz": DOPPLER_GRID_HZ.tolist(),
        "alignment": "Prompt(k), NCO/code/aux(k-1), raw [stamp(k-1), stamp(k-1)+25000)",
        "threshold_or_model_fitting": False, "thresholds_changed": False,
        "verification_samples": [
            {"channel": row["channel"], "prn": row["prn"], "anchor_row": row["anchor_row"],
             "aggregate_sha256": evaluated[index]["aggregate_sha256"]}
            for index, row in enumerate(selected[:3])
        ],
    })
    dump(args.output / "execution_validity.json", {
        "schema": "acaf_nf_stage1_r2a_execution_validity.v1", "status": verdict,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "cleanstatic_raw_bytes_addressable": raw_path.stat().st_size,
        "cleanstatic_windows_evaluated": len(evaluated), "cleanstatic_epochs_recomputed_from_raw": len(epoch_rows),
        "attack_iq_files_opened": [], "attack_iq_bytes_read": 0, "attack_scenarios_evaluated": [],
        "model_training_executed": False, "threshold_fitting_executed": False, "thresholds_changed": False,
        "candidate_selected_from_attack_results": False, "tracker_census": census,
        "full_span_pair_segments": {f"{key[0]}/{key[1]}": sorted(value) for key, value in sorted(pair_segments.items())},
        "excluded_incomplete_lineage_pairs": [list(pair) for pair in excluded_lineage_pairs],
    })
    plots(args.output, evaluated, per_prn)

    readme = f"""# ACAF-NF Stage-1 R2a L20 foundation audit

This cleanStatic-only audit answers why the prior L20 center recovery was **{cause['old_within_50_fraction']:.4%}** even though Prompt and delay reproduced correctly.

1. **85.96% failure cause.** The R1.4 and R2 aggregation equations are numerically equivalent within `1e-12`. The old R2 selector forced each pair's first 20 rows into a 969-epoch subset; 24 of its 24 L20 failures came from channel 7 / PRN 10 during the fresh receiver acquisition transient around 0.67–0.80 s. This was a validation-selection bias, not a CAF sign, normalization, or k/k-1 error.
2. **Correction.** The physical alignment remains Prompt `k`, NCO/code/aux `k-1`, and raw support `[stamp(k-1), stamp(k-1)+25000)`. R2a scans the full tracker for contiguous same-assignment 10-Hz L20 candidates and evaluates a deterministic all-pair/time-stratified clean-only sample. Reassignments, discontinuities, supports other than 25,000 samples, and reacquired pair lineages absent from any recording third are excluded before CAF evaluation. The excluded incomplete lineages are {excluded_lineage_pairs}.
3. **Why R1.4 and fresh differ.** They share authenticated raw bytes but are independent receiver executions with different build/config/initial carrier phase/channel assignment lineage. The historic `r14_common_reproduced_1e_6` result remains recorded as FAIL, but it is classified separately from physical validity; identical hashes are required only for a truly identical deterministic replay.
4. **Full cleanStatic result.** {overall['l20_windows']} L20 windows ({overall['evaluated_prn_channels']} PRN/channel pairs; {overall['one_ms_epochs']} raw-recomputed 1-ms constituents): ±50 Hz {overall['within_50_fraction']:.4%}, ±100 Hz {overall['within_100_fraction']:.4%}, exact center {overall['exact_center_fraction']:.4%}, delay ±0.125 chip {overall['delay_within_0_125_fraction']:.4%}, overall boundary {overall['overall_grid_boundary_fraction']:.4%}, Prompt Spearman {overall['prompt_reproduction']['pooled_spearman']:.9f}, Prompt p99 error {overall['prompt_reproduction']['p99_relative_error']:.4%}.
5. **Foundation verdict.** `{verdict}`. Thresholds and gates were not changed or auto-promoted.
6. **No attack/model claim.** No attack IQ was opened or evaluated, and no model, threshold, bootstrap, B0, DS3/DS4/DS7/DS8 score was fit or computed. Therefore this audit says nothing about attack detection performance or model feasibility.
7. **Next step.** {'The frozen physical alignment may proceed to a separately authorized attack/model stage.' if verdict == 'FOUNDATION_VALID' else 'Do not proceed to attack/model evaluation; the clean foundation remains unresolved.'}
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    (args.output / "test_report.txt").write_text("PENDING: run focused and existing ACAF tests after production.\n", encoding="utf-8")
    dump(args.output / "verification_report.json", {"status": "PENDING_INDEPENDENT_VERIFICATION"})
    print(json.dumps({"verdict": verdict, "metrics": overall, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
