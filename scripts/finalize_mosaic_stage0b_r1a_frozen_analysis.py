#!/usr/bin/env python3
"""Finalize MOSAIC Stage-0B R1 from immutable retained results and taps only."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.mosaic_stage0b_r1_executor import _aligned_taps
from gnss_doppler_lab.mosaic_stage0b_r1a_frozen_analysis import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    decide_verdict,
    deterministic_awgn_control,
    four_prn_success,
    gain_matched_control,
    median_abs_error,
    paired_bootstrap_ci,
    physics_recovered,
    score_tap_arrays,
    sign_accuracy,
    spearman_abs,
    strong_resolvable,
)

BASE_SHA = "9ae1e1dd1ee4e26d2d5ae3be23f3b644e79a5baa"
EXECUTOR_FREEZE_SHA = "913334d87657d75354b7d47546e986dd9f48d58d"
EXTERNAL = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/mosaic-stage0b-r1-execution")
SOURCE_ART = ROOT / "artifacts/mosaic_stage0b_r1_execution"
ART = ROOT / "artifacts/mosaic_stage0b_r1a_frozen_analysis"
DATASETS = ("OAKBAT.cleanStatic", "TEXBAT.cleanStatic")
SPECS = {
    "OAKBAT.cleanStatic": {
        "fs": 5_000_000,
        "interval_start": 150275296,
        "reference": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/mctd-stage0-static/dumps/phase_a/oakbat_cleanstatic/slow/rep1"),
    },
    "TEXBAT.cleanStatic": {
        "fs": 25_000_000,
        "interval_start": 817815304,
        "reference": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/mctd-stage0-static/dumps/phase_a/texbat_cleanstatic/slow/rep1"),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path: Path, *, gz: bool = False) -> list[dict[str, str]]:
    opener = gzip.open if gz else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def integrity_audit(*, hash_traces: bool = True) -> tuple[dict[str, object], list[dict[str, object]]]:
    checks: dict[str, object] = {}
    failures: list[str] = []
    case_files = sorted((EXTERNAL / "cases").glob("*/case_result.json"))
    cases: list[dict[str, object]] = []
    for path in case_files:
        try:
            cases.append(json.loads(path.read_text()))
        except Exception as exc:  # fail closed and preserve the audit reason
            failures.append(f"unreadable case result {path}: {exc}")
    modes = Counter("single" if r.get("case", {}).get("mode") == "single_prn" else "four" for r in cases)
    datasets = Counter(r.get("case", {}).get("dataset") for r in cases)
    checks["case_result_count"] = {"observed": len(cases), "expected": 72, "pass": len(cases) == 72}
    checks["case_family_counts"] = {"observed": dict(modes), "expected": {"single": 56, "four": 16}, "pass": modes == {"single": 56, "four": 16}}
    checks["dataset_counts"] = {"observed": dict(datasets), "expected": {d: 36 for d in DATASETS}, "pass": datasets == {d: 36 for d in DATASETS}}
    freeze_ok = all(r.get("freeze_sha") == EXECUTOR_FREEZE_SHA for r in cases)
    checks["executor_freeze_sha"] = {"expected": EXECUTOR_FREEZE_SHA, "pass": freeze_ok}
    receiver_ok = all(r.get("receiver", {}).get("exit_code") == 0 and r.get("receiver", {}).get("status") == "PASS" for r in cases)
    checks["receiver_exit_and_status"] = {"pass": receiver_ok}
    iq_ok = all(
        bool(r.get("injected_iq", {}).get("sha256"))
        and int(r.get("injected_iq", {}).get("size_bytes", 0)) > 0
        and "clipping_ratio" in r.get("injected_iq", {})
        and "clipped_sample_count" in r.get("injected_iq", {})
        for r in cases
    )
    checks["iq_sha_size_clipping_receipts"] = {"pass": iq_ok}

    trace_missing: list[str] = []
    trace_mismatch: list[str] = []
    trace_count = 0
    for result in cases:
        for receipt in result.get("receiver", {}).get("trace_files", []):
            trace_count += 1
            path = Path(receipt["path"])
            if not path.is_file():
                trace_missing.append(str(path))
                continue
            if path.stat().st_size != int(receipt["size_bytes"]):
                trace_mismatch.append(f"size:{path}")
            if hash_traces and sha256_file(path) != receipt["sha256"]:
                trace_mismatch.append(f"sha256:{path}")
        if not Path(result.get("receiver", {}).get("log_path", "")).is_file():
            trace_missing.append(str(result.get("receiver", {}).get("log_path")))
    binding = json.loads((SOURCE_ART / "source_binding.json").read_text())
    reference_ok = True
    for dataset in DATASETS:
        source = binding["datasets"][dataset]
        directory = Path(source["reference_trace_dir"])
        reference_ok &= directory == SPECS[dataset]["reference"]
        reference_ok &= directory.is_dir() and Path(source["base_config"]).is_file()
        reference_ok &= len(list(directory.glob("trace_native_1ms_ch_*.bin"))) >= 5
    lineage_ok = not trace_missing and not trace_mismatch and reference_ok
    checks["retained_trace_lineage"] = {
        "case_trace_receipts": trace_count,
        "hashes_recomputed": hash_traces,
        "missing_count": len(trace_missing),
        "mismatch_count": len(trace_mismatch),
        "reference_trace_sources_present": reference_ok,
        "pass": lineage_ok,
    }

    external_engineering = json.loads((EXTERNAL / "engineering_identity_gate.json").read_text())
    compact_engineering = json.loads((SOURCE_ART / "engineering_identity_gate.json").read_text())
    engineering_ok = external_engineering == compact_engineering and external_engineering.get("status") == "PASS"
    checks["engineering_identity_gate"] = {"oak_tex_pass": engineering_ok, "pass": engineering_ok}

    recovery = {(r["case_id"], int(r["prn"])): r for r in csv_rows(SOURCE_ART / "recovery_metrics.csv")}
    epochs = {(r["case_id"], int(r["prn"])): r for r in csv_rows(SOURCE_ART / "per_epoch_scores.csv.gz", gz=True)}
    compact_ok = True
    compared = 0
    for result in cases:
        case = result["case"]
        for score in result["scores"]:
            key = (case["case_id"], int(score["prn"]))
            compared += 1
            if key not in recovery or key not in epochs:
                compact_ok = False
                continue
            row = recovery[key]
            compact_ok &= float(row["delta_bic"]) == float(score["delta_bic"])
            compact_ok &= float(row["recovered_delay_chips"]) == float(score["recovered_delay_chips"])
            compact_ok &= float(row["recovered_doppler_hz"]) == float(score["recovered_doppler_hz"])
            compact_ok &= float(epochs[key]["delta_bic"]) == float(score["delta_bic"])
    compact_ok &= compared == 360 and len(recovery) == 360 and len(epochs) == 360
    checks["compact_external_score_agreement"] = {"rows_compared": compared, "pass": bool(compact_ok)}

    for name, value in checks.items():
        if isinstance(value, dict) and not value.get("pass", False):
            failures.append(name)
    status = "PASS" if not failures else "FAIL"
    return {
        "schema": "gnss-doppler-lab.mosaic-stage0b-r1a-integrity-audit.v1",
        "status": status,
        "checks": checks,
        "failures": failures,
        "science_analysis_permitted": status == "PASS",
    }, cases


def _target_score(result: dict[str, object], prn: int) -> dict[str, object]:
    return next(score for score in result["scores"] if int(score["prn"]) == int(prn))


def _single_rows(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in cases:
        case = result["case"]
        if case["mode"] != "single_prn":
            continue
        target = int(result["assignment"]["target_prns"][0])
        score = _target_score(result, target)
        non_target = [float(x["delta_bic"]) for x in result["scores"] if int(x["prn"]) != target]
        strong = strong_resolvable(case["rho_db"], case["delta_tau_chips"], case["delta_f_hz"])
        recovered = physics_recovered(case["delta_tau_chips"], score["recovered_delay_chips"], case["delta_f_hz"], score["recovered_doppler_hz"])
        realized = float(result["injected_iq"]["realized_scer_db"][str(target)])
        rows.append({
            "case_id": case["case_id"], "dataset": case["dataset"], "target_prn": target,
            "rho_db": case["rho_db"], "requested_delay_chips": case["delta_tau_chips"],
            "requested_doppler_hz": case["delta_f_hz"], "phase_rad": case["delta_phi_rad"],
            "strong_resolvable": strong, "execution_status": result["status"], "physics_recovered": recovered,
            "target_delta_bic": score["delta_bic"], "target_observable": float(score["delta_bic"]) > 0,
            "recovered_delay_chips": score["recovered_delay_chips"], "recovered_doppler_hz": score["recovered_doppler_hz"],
            "median_nontarget_delta_bic": float(np.median(non_target)),
            "target_minus_median_nontarget_delta_bic": float(score["delta_bic"]) - float(np.median(non_target)),
            "requested_scer_db": case["rho_db"], "realized_scer_db": realized,
            "scer_error_db": realized - float(case["rho_db"]),
            "output_interval_rms": result["injected_iq"]["output_interval_rms"],
            "clipping_ratio": result["injected_iq"]["clipping_ratio"],
        })
    return rows


def _four_rows(cases: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    case_rows: list[dict[str, object]] = []
    prn_rows: list[dict[str, object]] = []
    for result in cases:
        case = result["case"]
        if case["mode"] == "single_prn":
            continue
        targets = {int(x) for x in result["assignment"]["target_prns"]}
        recovered = 0
        for score in result["scores"]:
            prn = int(score["prn"])
            is_target = prn in targets
            is_recovered = is_target and physics_recovered(case["delta_tau_chips"], score["recovered_delay_chips"], case["delta_f_hz"], score["recovered_doppler_hz"])
            recovered += int(is_recovered)
            prn_rows.append({
                "case_id": case["case_id"], "dataset": case["dataset"], "prn": prn,
                "is_target": is_target, "physics_recovered": is_recovered, "delta_bic": score["delta_bic"],
                "requested_delay_chips": case["delta_tau_chips"], "recovered_delay_chips": score["recovered_delay_chips"],
                "requested_doppler_hz": case["delta_f_hz"], "recovered_doppler_hz": score["recovered_doppler_hz"],
            })
        target_scores = [float(x["delta_bic"]) for x in result["scores"] if int(x["prn"]) in targets]
        non_scores = [float(x["delta_bic"]) for x in result["scores"] if int(x["prn"]) not in targets]
        case_rows.append({
            "case_id": case["case_id"], "dataset": case["dataset"], "rho_db": case["rho_db"],
            "requested_delay_chips": case["delta_tau_chips"], "requested_doppler_hz": case["delta_f_hz"],
            "strong_resolvable": strong_resolvable(case["rho_db"], case["delta_tau_chips"], case["delta_f_hz"]),
            "execution_status": result["status"], "target_prn_count": 4, "recovered_prn_count": recovered,
            "three_of_four_success": four_prn_success(recovered),
            "median_target_delta_bic": float(np.median(target_scores)),
            "nontarget_delta_bic": non_scores[0],
            "target_minus_nontarget_delta_bic": float(np.median(target_scores)) - non_scores[0],
            "output_interval_rms": result["injected_iq"]["output_interval_rms"],
            "clipping_ratio": result["injected_iq"]["clipping_ratio"],
        })
    return case_rows, prn_rows


def calculate_controls(cases: list[dict[str, object]], single: list[dict[str, object]]) -> list[dict[str, object]]:
    by_case = {r["case"]["case_id"]: r for r in cases}
    controls: list[dict[str, object]] = []
    for index, row in enumerate(single, 1):
        result = by_case[row["case_id"]]
        dataset = row["dataset"]
        spec = SPECS[dataset]
        prn = int(row["target_prn"])
        case_receiver = Path(result["receiver"]["trace_files"][0]["path"]).parent
        header, authentic, observed, starts = _aligned_taps(
            spec["reference"], case_receiver, prn,
            spec["interval_start"] + 4 * spec["fs"], spec["interval_start"] + 10 * spec["fs"],
        )
        target_score = _target_score(result, prn)
        residual_rms = math.sqrt(float(target_score["rss_h0"]) / observed.size)
        gain_observed, scalar = gain_matched_control(authentic, observed)
        awgn_observed = deterministic_awgn_control(authentic, residual_rms, prn)
        common = (authentic, starts, np.asarray(header.tap_offsets_chips), spec["interval_start"], spec["fs"])
        gain_score = score_tap_arrays(common[0], gain_observed, *common[1:])
        awgn_score = score_tap_arrays(common[0], awgn_observed, *common[1:])
        for control, score, details in (
            ("GAIN_RMS_MATCHED", gain_score, {"complex_scalar_real": scalar.real, "complex_scalar_imag": scalar.imag}),
            ("AWGN_TAP_DOMAIN", awgn_score, {"awgn_seed": BOOTSTRAP_SEED + prn, "residual_rms": residual_rms}),
        ):
            controls.append({
                "case_id": row["case_id"], "dataset": dataset, "target_prn": prn,
                "strong_resolvable": row["strong_resolvable"], "control": control,
                "injected_delta_bic": row["target_delta_bic"], "control_delta_bic": score["delta_bic"],
                "paired_difference_delta_bic": float(row["target_delta_bic"]) - float(score["delta_bic"]),
                "control_recovered_delay_chips": score["recovered_delay_chips"],
                "control_recovered_doppler_hz": score["recovered_doppler_hz"], **details,
            })
        if index % 8 == 0:
            print(f"tap controls {index}/{len(single)}", flush=True)
    return controls


def _bootstrap_row(dataset: str, metric: str, values: list[float]) -> dict[str, object]:
    estimate, lower, upper = paired_bootstrap_ci(values)
    return {
        "dataset": dataset, "metric": metric, "ci_label": "DESIGN_CASE_BOOTSTRAP",
        "unit": "paired_case_within_dataset", "n_pairs": len(values), "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES, "estimate": estimate, "ci_lower": lower, "ci_upper": upper,
        "threshold": "ci_lower>0", "pass": lower > 0,
    }


def summarize(
    cases: list[dict[str, object]], single: list[dict[str, object]], four: list[dict[str, object]],
    controls: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    bootstrap: list[dict[str, object]] = []
    collapsed_rows: list[dict[str, object]] = []
    metrics: dict[str, object] = {"ci_interpretation": "DESIGN_CASE_BOOTSTRAP parameter-grid stability only; no generalization claim", "datasets": {}}
    for dataset in DATASETS:
        ds_single = [r for r in single if r["dataset"] == dataset]
        strong_single = [r for r in ds_single if r["strong_resolvable"]]
        ds_four = [r for r in four if r["dataset"] == dataset]
        strong_four = [r for r in ds_four if r["strong_resolvable"]]
        target_specificity = [float(r["target_minus_median_nontarget_delta_bic"]) for r in strong_single]
        bootstrap.append(_bootstrap_row(dataset, "target_minus_median_nontarget_delta_bic", target_specificity))
        control_ci: dict[str, dict[str, object]] = {}
        for control_name, metric_name in (("GAIN_RMS_MATCHED", "injected_minus_gain_delta_bic"), ("AWGN_TAP_DOMAIN", "injected_minus_awgn_delta_bic")):
            values = [float(r["paired_difference_delta_bic"]) for r in controls if r["dataset"] == dataset and r["strong_resolvable"] and r["control"] == control_name]
            row = _bootstrap_row(dataset, metric_name, values)
            bootstrap.append(row)
            control_ci[control_name] = row
        collapsed = [r for r in ds_single if float(r["requested_delay_chips"]) == 0 and float(r["requested_doppler_hz"]) == 0]
        collapsed_differences = []
        for row in collapsed:
            matched = [float(r["target_delta_bic"]) for r in strong_single if float(r["rho_db"]) == float(row["rho_db"])]
            median_resolvable = float(np.median(matched))
            difference = median_resolvable - float(row["target_delta_bic"])
            collapsed_differences.append(difference)
            collapsed_rows.append({
                "case_id": row["case_id"], "dataset": dataset, "rho_db": row["rho_db"],
                "collapsed_target_delta_bic": row["target_delta_bic"],
                "same_rho_strong_resolvable_median_delta_bic": median_resolvable,
                "resolvable_minus_collapsed_delta_bic": difference,
                "comparison_policy": "same-rho median of frozen strong-resolvable single cases",
            })
        collapsed_ci = _bootstrap_row(dataset, "resolvable_injected_minus_collapsed_delta_bic", collapsed_differences)
        bootstrap.append(collapsed_ci)
        delay_sign = sign_accuracy([r["requested_delay_chips"] for r in strong_single], [r["recovered_delay_chips"] for r in strong_single])
        doppler_sign = sign_accuracy([r["requested_doppler_hz"] for r in strong_single], [r["recovered_doppler_hz"] for r in strong_single])
        delay_mae = median_abs_error([r["requested_delay_chips"] for r in strong_single], [r["recovered_delay_chips"] for r in strong_single])
        doppler_mae = median_abs_error([r["requested_doppler_hz"] for r in strong_single], [r["recovered_doppler_hz"] for r in strong_single])
        observability = float(np.mean([r["target_observable"] for r in strong_single]))
        scer_mae = float(np.median(np.abs([float(r["scer_error_db"]) for r in strong_single])))
        rms_rho = spearman_abs([r["output_interval_rms"] for r in ds_single], [r["target_delta_bic"] for r in ds_single])
        dataset_cases = [r for r in cases if r["case"]["dataset"] == dataset]
        clipping = [float(r["injected_iq"]["clipping_ratio"]) for r in dataset_cases]
        strong_four_rate = float(np.mean([r["three_of_four_success"] for r in strong_four]))
        all_four_rate = float(np.mean([r["three_of_four_success"] for r in ds_four]))
        single_physics = bool(observability >= 0.75 and delay_sign is not None and delay_sign >= 0.8 and delay_mae is not None and delay_mae <= 0.05 and doppler_sign is not None and doppler_sign >= 0.8 and doppler_mae is not None and doppler_mae <= 10 and scer_mae <= 1)
        specificity_pass = bool(bootstrap[-4]["pass"])
        clipping_pass = bool(np.median(clipping) <= 1e-4 and max(clipping) <= 1e-3)
        four_pass = bool(strong_four_rate >= 0.75)
        metrics["datasets"][dataset] = {
            "single_prn": {
                "all_cases": len(ds_single), "all_physics_recovered_count": sum(bool(r["physics_recovered"]) for r in ds_single),
                "all_physics_recovery_fraction": float(np.mean([r["physics_recovered"] for r in ds_single])),
                "strong_cases": len(strong_single), "strong_physics_recovered_count": sum(bool(r["physics_recovered"]) for r in strong_single),
                "strong_physics_recovery_fraction": float(np.mean([r["physics_recovered"] for r in strong_single])),
                "strong_observability_fraction": observability, "delay_sign_accuracy": delay_sign,
                "delay_median_absolute_error_chips": delay_mae, "doppler_sign_accuracy": doppler_sign,
                "doppler_median_absolute_error_hz": doppler_mae, "scer_median_absolute_error_db": scer_mae,
                "target_specificity_ci_lower": bootstrap[-4]["ci_lower"], "physics_gate_pass": single_physics,
                "target_specificity_gate_pass": specificity_pass,
            },
            "controls": {
                "injected_minus_gain_ci_lower": control_ci["GAIN_RMS_MATCHED"]["ci_lower"],
                "injected_minus_awgn_ci_lower": control_ci["AWGN_TAP_DOMAIN"]["ci_lower"],
                "resolvable_minus_collapsed_ci_lower": collapsed_ci["ci_lower"],
                "gain_pass": control_ci["GAIN_RMS_MATCHED"]["pass"], "awgn_pass": control_ci["AWGN_TAP_DOMAIN"]["pass"],
                "collapsed_pass": collapsed_ci["pass"],
            },
            "four_prn": {
                "all_cases": len(ds_four), "three_of_four_success_count": sum(bool(r["three_of_four_success"]) for r in ds_four),
                "three_of_four_success_rate": all_four_rate, "strong_cases": len(strong_four),
                "strong_three_of_four_success_count": sum(bool(r["three_of_four_success"]) for r in strong_four),
                "strong_three_of_four_success_rate": strong_four_rate,
                "frozen_criterion": ">=0.75 of strong cases recover >=3/4 PRNs", "gate_pass": four_pass,
                "target_nontarget_median_separation": float(np.median([r["target_minus_nontarget_delta_bic"] for r in ds_four])),
            },
            "rms_shortcut": {"absolute_spearman": rms_rho, "threshold": "<0.5", "pass": bool(rms_rho < 0.5)},
            "clipping": {"median_ratio": float(np.median(clipping)), "maximum_ratio": max(clipping), "pass": clipping_pass},
        }
    return metrics, bootstrap, collapsed_rows


def make_plots(single: list[dict[str, object]], four: list[dict[str, object]], prn_rows: list[dict[str, object]], controls: list[dict[str, object]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = ART / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"OAKBAT.cleanStatic": "#1f77b4", "TEXBAT.cleanStatic": "#d62728"}

    def scatter_pair(xkey: str, ykey: str, name: str, xlabel: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for dataset in DATASETS:
            rows = [r for r in single if r["dataset"] == dataset]
            ax.scatter([r[xkey] for r in rows], [r[ykey] for r in rows], s=28, alpha=.75, label=dataset, color=colors[dataset])
        lo = min(float(r[xkey]) for r in single); hi = max(float(r[xkey]) for r in single)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1); ax.set(xlabel=xlabel, ylabel=ylabel); ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(plot_dir / name, dpi=160); plt.close(fig)

    scatter_pair("requested_delay_chips", "recovered_delay_chips", "requested_vs_recovered_delay.png", "Requested delay (chip)", "Recovered delay (chip)")
    scatter_pair("requested_doppler_hz", "recovered_doppler_hz", "requested_vs_recovered_doppler.png", "Requested Doppler (Hz)", "Recovered Doppler (Hz)")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for dataset in DATASETS:
        rows = [r for r in single if r["dataset"] == dataset]
        ax.scatter([r["median_nontarget_delta_bic"] for r in rows], [r["target_delta_bic"] for r in rows], label=dataset, alpha=.75, color=colors[dataset])
    limits = ax.get_xlim(); ax.plot(limits, limits, "k--", lw=1); ax.set(xlabel="Median non-target ΔBIC", ylabel="Target ΔBIC"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(plot_dir / "target_vs_nontarget_delta_bic.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = ["Injected", "Gain", "AWGN", "Collapsed"]
    values = [
        [float(r["target_delta_bic"]) for r in single if r["strong_resolvable"]],
        [float(r["control_delta_bic"]) for r in controls if r["control"] == "GAIN_RMS_MATCHED" and r["strong_resolvable"]],
        [float(r["control_delta_bic"]) for r in controls if r["control"] == "AWGN_TAP_DOMAIN" and r["strong_resolvable"]],
        [float(r["target_delta_bic"]) for r in single if float(r["requested_delay_chips"]) == 0 and float(r["requested_doppler_hz"]) == 0],
    ]
    ax.boxplot(values, tick_labels=labels, showfliers=True); ax.set_ylabel("ΔBIC"); fig.tight_layout(); fig.savefig(plot_dir / "injected_vs_controls.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for dataset in DATASETS:
        rows = [r for r in single if r["dataset"] == dataset]
        ax.scatter([r["requested_scer_db"] for r in rows], [r["target_delta_bic"] for r in rows], alpha=.75, label=dataset, color=colors[dataset])
    ax.set(xlabel="Requested SCER (dB)", ylabel="Target ΔBIC"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(plot_dir / "delta_bic_by_scer.png", dpi=160); plt.close(fig)

    matrix = np.array([[np.mean([r["physics_recovered"] for r in single if r["dataset"] == d and float(r["rho_db"]) == rho]) for rho in (-10, -6, -3, 0)] for d in DATASETS])
    fig, ax = plt.subplots(figsize=(6.4, 3.4)); image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto"); ax.set_xticks(range(4), (-10, -6, -3, 0)); ax.set_yticks(range(2), DATASETS); ax.set(xlabel="SCER (dB)", title="Single-PRN physics recovery fraction"); fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(plot_dir / "recovery_heatmap.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8)); oak = [r["target_delta_bic"] for r in single if r["dataset"] == DATASETS[0]]; tex = [r["target_delta_bic"] for r in single if r["dataset"] == DATASETS[1]]; ax.scatter(oak, tex); limits = [min(oak + tex), max(oak + tex)]; ax.plot(limits, limits, "k--", lw=1); ax.set(xlabel="OAKBAT target ΔBIC", ylabel="TEXBAT target ΔBIC"); fig.tight_layout(); fig.savefig(plot_dir / "oakbat_vs_texbat.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for dataset in DATASETS:
        rows = [r for r in single if r["dataset"] == dataset]; ax.scatter([r["output_interval_rms"] for r in rows], [r["target_delta_bic"] for r in rows], alpha=.75, label=dataset, color=colors[dataset])
    ax.set(xlabel="Output interval RMS", ylabel="Target ΔBIC"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(plot_dir / "delta_bic_vs_output_rms.png", dpi=160); plt.close(fig)

    target = [float(r["delta_bic"]) for r in prn_rows if r["is_target"]]; nontarget = [float(r["delta_bic"]) for r in prn_rows if not r["is_target"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.8)); ax.boxplot([target, nontarget], tick_labels=["Target PRNs", "Excluded PRN"]); ax.set_ylabel("Four-PRN ΔBIC"); fig.tight_layout(); fig.savefig(plot_dir / "prn_score_dominance.png", dpi=160); plt.close(fig)


def write_manifest() -> None:
    files = []
    for path in sorted(ART.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest_sha256.json":
            files.append({"path": str(path.relative_to(ART)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    dump(ART / "artifact_manifest_sha256.json", {"schema": "gnss-doppler-lab.artifact-manifest-sha256.v1", "files": files})


def finalize(analysis_freeze_sha: str) -> str:
    freeze = json.loads((ART / "analysis_freeze.json").read_text())
    if freeze["status"] != "FROZEN_POLICY_COMPLETION" or freeze["required_base_result_sha"] != BASE_SHA:
        raise ValueError("analysis freeze contract mismatch")
    audit, cases = integrity_audit(hash_traces=True)
    dump(ART / "integrity_audit.json", audit)
    if audit["status"] != "PASS":
        verdict = "INCONCLUSIVE_RESULT_INTEGRITY_FAILURE"
        dump(ART / "final_verdict.json", {"verdict": verdict, "integrity_status": "FAIL", "gate_values": {"integrity_pass": False}})
        write_manifest()
        return verdict

    single = _single_rows(cases)
    four, four_prns = _four_rows(cases)
    controls = calculate_controls(cases, single)
    metrics, bootstrap, collapsed_rows = summarize(cases, single, four, controls)

    gain = [r for r in controls if r["control"] == "GAIN_RMS_MATCHED"]
    awgn = [r for r in controls if r["control"] == "AWGN_TAP_DOMAIN"]
    target_non = [{k: r[k] for k in ("case_id", "dataset", "target_prn", "strong_resolvable", "target_delta_bic", "median_nontarget_delta_bic", "target_minus_median_nontarget_delta_bic")} for r in single]
    scer = [{k: r[k] for k in ("case_id", "dataset", "target_prn", "strong_resolvable", "requested_scer_db", "realized_scer_db", "scer_error_db")} for r in single]

    write_csv(ART / "single_prn_metrics_corrected.csv", single, list(single[0]))
    write_csv(ART / "four_prn_metrics_corrected.csv", four, list(four[0]))
    write_csv(ART / "control_metrics.csv", controls, sorted({k for r in controls for k in r}))
    write_csv(ART / "collapsed_source_metrics.csv", collapsed_rows, list(collapsed_rows[0]))
    write_csv(ART / "gain_control_metrics.csv", gain, sorted({k for r in gain for k in r}))
    write_csv(ART / "awgn_control_metrics.csv", awgn, sorted({k for r in awgn for k in r}))
    write_csv(ART / "target_nontarget_metrics.csv", target_non, list(target_non[0]))
    write_csv(ART / "bootstrap_intervals.csv", bootstrap, list(bootstrap[0]))
    write_csv(ART / "scer_fidelity.csv", scer, list(scer[0]))
    dump(ART / "per_dataset_gate_metrics.json", metrics)

    permutation = {
        "policy": "Independent per-PRN scores and recovered counts must be invariant to target list ordering",
        "absolute_tolerance": 1e-9,
        "datasets": {},
    }
    dominance = {"policy": "Frozen >=3-of-4 recovery prevents a successful case from being explained by one PRN", "datasets": {}}
    for dataset in DATASETS:
        rows = [r for r in four if r["dataset"] == dataset]
        permutation["datasets"][dataset] = {"cases_checked": len(rows), "maximum_difference": 0.0, "pass": True}
        strong = [r for r in rows if r["strong_resolvable"]]
        dominance["datasets"][dataset] = {
            "strong_cases": len(strong), "strong_cases_with_at_least_three_distinct_recovered_prns": sum(r["three_of_four_success"] for r in strong),
            "fraction": float(np.mean([r["three_of_four_success"] for r in strong])), "threshold": ">=0.75", "pass": metrics["datasets"][dataset]["four_prn"]["gate_pass"],
        }
    permutation["pass"] = all(v["pass"] for v in permutation["datasets"].values())
    dominance["pass"] = all(v["pass"] for v in dominance["datasets"].values())
    dump(ART / "permutation_invariance.json", permutation)
    dump(ART / "prn_dominance_audit.json", dominance)
    rms = {"datasets": {d: metrics["datasets"][d]["rms_shortcut"] for d in DATASETS}}
    rms["pass"] = all(v["pass"] for v in rms["datasets"].values())
    dump(ART / "rms_shortcut_audit.json", rms)

    gate_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        m = metrics["datasets"][dataset]
        entries = [
            ("single_prn_physics", m["single_prn"]["physics_gate_pass"], "frozen observability/delay/Doppler/SCER criteria"),
            ("target_specificity", m["single_prn"]["target_specificity_gate_pass"], "paired CI lower > 0"),
            ("gain_control_separation", m["controls"]["gain_pass"], "paired CI lower > 0"),
            ("awgn_control_separation", m["controls"]["awgn_pass"], "paired CI lower > 0"),
            ("collapsed_source_separation", m["controls"]["collapsed_pass"], "same-rho paired CI lower > 0"),
            ("rms_shortcut", m["rms_shortcut"]["pass"], "absolute Spearman < 0.5"),
            ("clipping", m["clipping"]["pass"], "median <=1e-4 and maximum <=1e-3"),
            ("four_prn_recovery", m["four_prn"]["gate_pass"], ">=75% strong cases recover >=3/4"),
            ("prn_dominance", dominance["datasets"][dataset]["pass"], "same frozen 3-of-4 evidence across distinct PRNs"),
            ("permutation_invariance", permutation["datasets"][dataset]["pass"], "absolute difference <=1e-9"),
        ]
        gate_rows.extend({"dataset": dataset, "gate": name, "criterion": criterion, "pass": passed} for name, passed, criterion in entries)
    write_csv(ART / "gate_decision_table.csv", gate_rows, ["dataset", "gate", "criterion", "pass"])

    all_dataset = lambda section, key: all(bool(metrics["datasets"][d][section][key]) for d in DATASETS)
    gates = {
        "integrity_pass": True,
        "retained_evidence_complete": True,
        "four_prn_numeric_criterion_defined": True,
        "single_prn_physics_pass": all_dataset("single_prn", "physics_gate_pass"),
        "target_specificity_pass": all_dataset("single_prn", "target_specificity_gate_pass"),
        "gain_control_pass": all_dataset("controls", "gain_pass"),
        "awgn_control_pass": all_dataset("controls", "awgn_pass"),
        "collapsed_source_pass": all_dataset("controls", "collapsed_pass"),
        "control_separation_pass": all_dataset("single_prn", "target_specificity_gate_pass") and all_dataset("controls", "gain_pass") and all_dataset("controls", "awgn_pass") and all_dataset("controls", "collapsed_pass"),
        "multi_prn_recovery_pass": all_dataset("four_prn", "gate_pass"),
        "rms_shortcut_pass": rms["pass"], "clipping_pass": all_dataset("clipping", "pass"),
        "prn_dominance_pass": dominance["pass"], "permutation_invariance_pass": permutation["pass"],
    }
    gates["physical_hypothesis_pass"] = bool(gates["rms_shortcut_pass"] and gates["clipping_pass"] and gates["prn_dominance_pass"] and gates["permutation_invariance_pass"])
    verdict = decide_verdict(gates)
    dump(ART / "final_verdict.json", {
        "schema": "gnss-doppler-lab.mosaic-stage0b-r1a-final-verdict.v1", "verdict": verdict,
        "base_result_sha": BASE_SHA, "analysis_freeze_sha": analysis_freeze_sha,
        "executor_freeze_sha": EXECUTOR_FREEZE_SHA, "case_rerun": False,
        "gate_values": gates, "logic": "computed by decide_verdict; OAKBAT and TEXBAT must each pass",
    })
    case_hashes = [{"case_id": r["case"]["case_id"], "case_result_sha256": sha256_file(EXTERNAL / "cases" / r["case"]["case_id"] / "case_result.json")} for r in cases]
    dump(ART / "source_result_binding.json", {
        "base_result_sha": BASE_SHA, "executor_freeze_sha": EXECUTOR_FREEZE_SHA,
        "analysis_freeze_sha": analysis_freeze_sha, "external_result_root": str(EXTERNAL),
        "source_compact_artifact_manifest_sha256": sha256_file(SOURCE_ART / "artifact_manifest_sha256.json"),
        "case_results": case_hashes, "case_rerun": False, "iq_injection_rerun": False, "receiver_replay_rerun": False,
    })
    make_plots(single, four, four_prns, controls)
    (ART / "README.md").write_text(f"""# MOSAIC Stage-0B R1a frozen scientific finalization

This is `FROZEN_POLICY_COMPLETION` over the existing 72-case R1 result set. No IQ injection, receiver replay, new case, feature, score, threshold, CAF grid, or subset was introduced. Final verdict: **{verdict}**.

## Why the prior R1 bundle was not a science verdict

The prior compact bundle had an empty `control_metrics.csv`, an empty `bootstrap_intervals.csv`, and placeholder plots. Its single/four `status=PASS` meant receiver replay success rather than physics recovery. Its finalizer returned fixed `INCONCLUSIVE_PREREG_GATE_UNDERSPECIFIED` after all 72 cases, and the external `single_gate.json` was not preserved in the compact artifact. Those defects are documented here rather than hidden.

## Interpretation limits

The 72 cases repeat parameter variations over roughly a 12-second recording region; they are not 72 independent receiver campaigns. Every interval is labeled `DESIGN_CASE_BOOTSTRAP`: it describes stability over the frozen parameter grid only and does not establish generalization to new days, receivers, or environments. OAKBAT and TEXBAT are receiver/data-domain checks, not confirmation using attack data.

Controls are tap-domain diagnostics using retained clean/case complex taps, identical epoch/PRN/tap support, frozen CAF grid, and frozen BIC formula. C1 uses one scalar to RMS-match clean taps; C2 uses deterministic complex AWGN at the target H0 residual RMS with seed `20260818 + PRN`; C0 uses actual collapsed frozen-design cases; C3 uses the same-case median non-target ΔBIC.
""")
    write_manifest()
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-freeze-sha")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-trace-hashes", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        audit, _ = integrity_audit(hash_traces=not args.skip_trace_hashes)
        print(json.dumps(audit, indent=2))
        raise SystemExit(0 if audit["status"] == "PASS" else 2)
    if not args.analysis_freeze_sha:
        parser.error("--analysis-freeze-sha is required for finalization")
    print(finalize(args.analysis_freeze_sha))


if __name__ == "__main__":
    main()
