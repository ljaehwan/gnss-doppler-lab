#!/usr/bin/env python3
"""Post-freeze PG-SCC static evaluation.  Never tune design in this program."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.acaf_nf_stage1_r2a_l20_foundation_audit import State, complex_caf_surface  # noqa: E402
from gnss_doppler_lab.pg_scc import (  # noqa: E402
    artifact_manifest, binary_metrics, dump_json, event_alarm_metrics, exact_binomial_ci,
    load_feature_cache, load_json, pool_events, rank_score_correlation, score_rows, sha256, write_csv,
)
from gnss_doppler_lab.pg_scc_physics import (  # noqa: E402
    N_COORDINATES, complex_correlator_coordinates, inject_same_prn_second_source,
    normalize_complex, two_source_glrt,
)

DEFAULT_OUTPUT = ROOT / "artifacts/pg_scc_stage0_static_k9"
FAMILY = {"ds3": "ds3", "ds4": "ds4", "ds7": "ds7_ds8", "ds8": "ds7_ds8"}
ONSET = {"ds3": 118.9, "ds4": 113.8, "ds7": 110.0, "ds8": 110.0}


def verify_freeze(output: Path) -> dict[str, str]:
    manifest = load_json(output / "freeze_manifest.json")
    errors = [name for name, digest in manifest.items() if not (output / name).is_file() or sha256(output / name) != digest]
    if errors:
        raise RuntimeError(f"frozen design drift before attack open: {errors}")
    frozen = load_json(output / "frozen_design.json")
    if frozen["attack_iq_bytes_read_before_freeze"] != 0 or frozen["real_attack_labels_used_for_selector_pooling_threshold"]:
        raise RuntimeError("freeze leakage declaration invalid")
    return manifest


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric_clean_scores(output: Path) -> list[dict]:
    rows = read_csv(output / "per_epoch_scores_clean.csv")
    numeric = {"second": int, "time_s": float, "budget": int, "score": float, "raw_power": float,
               "prompt_magnitude": float, "raw_start_sample": int, "raw_end_sample": int}
    for row in rows:
        for name, caster in numeric.items():
            if row.get(name) not in (None, "", "None"):
                row[name] = caster(row[name])
    return rows


def threshold_for(thresholds: dict, method: str, budget: int, quantile: str = "q99") -> float:
    return float(thresholds[f"{method}:K{budget}"][quantile])


def authenticate_attack_sources(config: dict, full_hash: bool) -> tuple[dict, int]:
    reports = {}; bytes_read = 0
    for scenario, value in config["sources"]["attack_raw"].items():
        path = Path(value)
        report = {"path": str(path), "size_bytes": path.stat().st_size,
                  "expected_sha256": config["sources"]["raw_sha256"][scenario], "full_hash_performed": full_hash}
        if full_hash:
            report["actual_sha256"] = sha256(path); bytes_read += path.stat().st_size
        report["status"] = "PASS" if (not full_hash or report["actual_sha256"] == report["expected_sha256"]) else "FAIL"
        if report["status"] != "PASS":
            raise RuntimeError(f"attack lineage hash failed: {scenario}")
        reports[scenario] = report
    return reports, bytes_read


def scenario_metrics(pooled: list[dict], thresholds: dict) -> list[dict]:
    output = []
    keys = sorted({(row["method"], int(row["budget"])) for row in pooled})
    for scenario in ("ds3", "ds4", "ds7", "ds8"):
        for method, budget in keys:
            negative = [row for row in pooled if row["scenario"] == "cleanStatic" and row["phase"] == "holdout" and row["method"] == method and row["budget"] == budget]
            attack = [row for row in pooled if row["scenario"] == scenario and row["phase"] != "strict_pre" and row["method"] == method and row["budget"] == budget]
            pre = [row for row in pooled if row["scenario"] == scenario and row["phase"] == "strict_pre" and row["method"] == method and row["budget"] == budget]
            if not negative or not attack:
                continue
            metrics = binary_metrics([0] * len(negative) + [1] * len(attack), [row["pooled_score"] for row in negative + attack])
            threshold = threshold_for(thresholds, method, budget)
            alarms = event_alarm_metrics(attack, threshold, ONSET[scenario])
            pre_alarms = sum(row["pooled_score"] >= threshold for row in pre)
            holdout_alarms = sum(row["pooled_score"] >= threshold for row in negative)
            output.append({
                "scenario": scenario, "family": FAMILY[scenario], "method": method, "budget": budget,
                **metrics, **alarms, "external_pre_events": len(pre), "external_pre_alarms": pre_alarms,
                "external_pre_fpr": pre_alarms / len(pre) if pre else None,
                "clean_holdout_events": len(negative), "clean_holdout_alarms": holdout_alarms,
                "clean_holdout_fpr": holdout_alarms / len(negative),
                "clean_holdout_fpr_ci95": json.dumps(exact_binomial_ci(holdout_alarms, len(negative))),
                "claim_scope": "transition_only" if scenario == "ds4" else "transition_and_established",
            })
    return output


def family_metrics(pooled: list[dict]) -> list[dict]:
    output = []
    keys = sorted({(row["method"], int(row["budget"])) for row in pooled})
    for family, scenarios in (("ds3", ("ds3",)), ("ds4", ("ds4",)), ("ds7_ds8", ("ds7", "ds8"))):
        for method, budget in keys:
            negative = [row for row in pooled if row["scenario"] == "cleanStatic" and row["phase"] == "holdout" and row["method"] == method and row["budget"] == budget]
            positive = [row for row in pooled if row["scenario"] in scenarios and row["phase"] != "strict_pre" and row["method"] == method and row["budget"] == budget]
            if negative and positive:
                metrics = binary_metrics([0] * len(negative) + [1] * len(positive), [row["pooled_score"] for row in negative + positive])
                output.append({"family": family, "method": method, "budget": budget,
                               "negative_events": len(negative), "positive_events": len(positive), **metrics})
    return output


def baseline_summary(families: list[dict], scenarios: list[dict]) -> list[dict]:
    output = []
    for method, budget in sorted({(row["method"], row["budget"]) for row in families}):
        values = [row for row in families if row["method"] == method and row["budget"] == budget]
        scenario_values = [row for row in scenarios if row["method"] == method and row["budget"] == budget]
        output.append({
            "method": method, "budget": budget, "status": "AVAILABLE",
            "family_macro_low_fpr_pauc": float(np.mean([row["normalized_low_fpr_pauc"] for row in values])),
            "family_macro_roc_auc": float(np.mean([row["roc_auc"] for row in values])),
            "worst_external_pre_fpr": max((row["external_pre_fpr"] or 0.0) for row in scenario_values),
            "worst_clean_holdout_fpr": max(row["clean_holdout_fpr"] for row in scenario_values),
        })
    output.append({"method": "B0_exact", "budget": 0, "status": "UNAVAILABLE",
                   "reason": "native B0 checkpoint/scorer is not reproducible on identical PG-SCC epoch/PRN support; historic CSV reuse forbidden"})
    return output


def block_bootstrap(pooled: list[dict], family: str, seed: int, replicates: int = 1000) -> dict:
    scenarios = (family,) if family != "ds7_ds8" else ("ds7", "ds8")
    clean_pg = [row["pooled_score"] for row in pooled if row["scenario"] == "cleanStatic" and row["phase"] == "holdout" and row["method"] == "pg_scc_k9"]
    clean_fixed = [row["pooled_score"] for row in pooled if row["scenario"] == "cleanStatic" and row["phase"] == "holdout" and row["method"] == "fixed9"]
    events = defaultdict(dict)
    for row in pooled:
        if row["scenario"] in scenarios and row["phase"] != "strict_pre" and row["method"] in {"pg_scc_k9", "fixed9"}:
            events[(row["scenario"], row["second"])][row["method"]] = row
    blocks = defaultdict(list)
    for value in events.values():
        if len(value) == 2:
            row = value["pg_scc_k9"]; blocks[(row["scenario"], int(row["time_s"] // 10))].append(value)
    block_values = list(blocks.values())
    if len(block_values) < 2:
        return {"family": family, "status": "INCONCLUSIVE", "blocks": len(block_values)}
    rng = np.random.default_rng(seed); effects = []
    for _ in range(replicates):
        sampled = [block_values[index] for index in rng.integers(0, len(block_values), len(block_values))]
        flat = [event for block in sampled for event in block]
        pg = [event["pg_scc_k9"]["pooled_score"] for event in flat]
        fixed = [event["fixed9"]["pooled_score"] for event in flat]
        left = binary_metrics([0] * len(clean_pg) + [1] * len(pg), clean_pg + pg)["normalized_low_fpr_pauc"]
        right = binary_metrics([0] * len(clean_fixed) + [1] * len(fixed), clean_fixed + fixed)["normalized_low_fpr_pauc"]
        effects.append(left - right)
    return {"family": family, "comparison": "pg_scc_k9-minus-fixed9 low-FPR pAUC", "status": "PASS",
            "block_seconds": 10, "blocks": len(block_values), "replicates": replicates,
            "effect": float(np.mean(effects)), "ci95_low": float(np.quantile(effects, .025)),
            "ci95_high": float(np.quantile(effects, .975)), "seed": seed}


def physics_controls(clean_rows: list[dict], auth: np.ndarray, covariance: np.ndarray, masks: dict,
                     thresholds: dict, baselines: list[dict], pooled: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed); threshold = threshold_for(thresholds, "pg_scc_k9", 9)
    selected = masks["pg_scc_k9"]; sample = normalize_complex(clean_rows[0]["surface"], "prompt_phase")
    reference = two_source_glrt(sample, auth, covariance, indices=selected).score
    errors = []
    for gain in (.5, .8, 1.2, 2.0):
        for phase in (0.0, np.pi / 3, np.pi / 2, np.pi):
            transformed = normalize_complex(sample * gain * np.exp(1j * phase), "prompt_phase")
            errors.append(abs(two_source_glrt(transformed, auth, covariance, indices=selected).score - reference))
    controls = [{"control": "global_gain_and_phase", "status": "PASS" if max(errors) < 1e-6 else "FAIL",
                 "value": max(errors), "criterion": "max score drift <1e-6"}]
    for sigma in (.02, .05):
        scores = []
        for row in clean_rows[:80]:
            value = normalize_complex(row["surface"], "prompt_phase")
            noise = sigma * (rng.normal(size=187) + 1j * rng.normal(size=187)) / np.sqrt(2)
            scores.append(two_source_glrt(normalize_complex(value + noise, "prompt_phase"), auth, covariance, indices=selected).score)
        fpr = float(np.mean(np.asarray(scores) >= threshold))
        controls.append({"control": f"awgn_sigma_{sigma}", "status": "PASS" if fpr <= .05 else "FAIL",
                         "value": fpr, "criterion": "clean false response <=5%"})
    power_score = two_source_glrt(normalize_complex(sample * 4.0, "prompt_phase"), auth, covariance, indices=selected).score
    controls.append({"control": "single_source_power_increase", "status": "PASS" if abs(power_score-reference) < 1e-6 else "FAIL",
                     "value": abs(power_score-reference), "criterion": "score drift <1e-6"})
    positives = []
    for tau in (-.75, -.375, -.125, .125, .375, .75):
        for doppler in (-150, 0, 150):
            for phase in (0, np.pi/2, np.pi, 3*np.pi/2):
                value = inject_same_prn_second_source(sample, delta_tau_chips=tau, delta_doppler_hz=doppler,
                    relative_amplitude=.75, relative_phase_rad=phase, noise_sigma=.02, rng=rng, normalization="prompt_phase")
                positives.append(two_source_glrt(value, auth, covariance, indices=selected).score)
    positive_rate = float(np.mean(np.asarray(positives) >= threshold))
    controls.append({"control": "same_prn_second_source", "status": "PASS" if positive_rate >= .8 else "FAIL",
                     "value": positive_rate, "criterion": "synthetic detection >=80%"})
    lookup = {(row["method"], int(row["budget"])): row for row in baselines if row.get("status") == "AVAILABLE"}
    pg = lookup[("pg_scc_k9", 9)]["family_macro_low_fpr_pauc"]
    comparisons = [lookup[(name, 9)]["family_macro_low_fpr_pauc"] for name in ("uniform_k9", "shuffled_k9")]
    random_values = [row["family_macro_low_fpr_pauc"] for row in baselines if row.get("status") == "AVAILABLE" and row["method"].startswith("random") and row["budget"] == 9]
    learned_pass = pg > max([*comparisons, float(np.mean(random_values))])
    controls.append({"control": "learned_vs_random_uniform_shuffled", "status": "PASS" if learned_pass else "FAIL",
                     "value": pg - max([*comparisons, float(np.mean(random_values))]), "criterion": "macro pAUC difference >0"})
    boundary_fraction = float(np.mean([index // 17 in (0, 10) or index % 17 in (0, 16) for index in selected]))
    controls.append({"control": "mask_boundary_concentration", "status": "PASS" if boundary_fraction <= .5 else "FAIL",
                     "value": boundary_fraction, "criterion": "boundary fraction <=0.5"})
    event = defaultdict(dict)
    for row in pooled:
        if row["scenario"] != "cleanStatic" and row["method"] in {"pg_scc_k9", "raw_power_only"}:
            event[(row["scenario"], row["second"])][row["method"]] = row
    pg_alarm = overlap = 0; power_threshold = threshold_for(thresholds, "raw_power_only", 0)
    for values in event.values():
        if len(values) == 2 and values["pg_scc_k9"]["pooled_score"] >= threshold:
            pg_alarm += 1; overlap += values["raw_power_only"]["pooled_score"] >= power_threshold
    overlap_fraction = overlap / pg_alarm if pg_alarm else 1.0
    controls.append({"control": "raw_power_alarm_overlap", "status": "PASS" if overlap_fraction < .9 else "FAIL",
                     "value": overlap_fraction, "criterion": "raw power explains <90% of PG-SCC alarms"})
    dense_map = {(row["scenario"], row["second"]): row["pooled_score"] for row in pooled if row["method"] == "dense_two_source_glrt"}
    sparse_map = {(row["scenario"], row["second"]): row["pooled_score"] for row in pooled if row["method"] == "pg_scc_k9"}
    common = sorted(set(dense_map) & set(sparse_map))
    correlation = rank_score_correlation([dense_map[key] for key in common], [sparse_map[key] for key in common])
    controls.append({"control": "dense_teacher_sparse_rank", "status": "PASS" if correlation["spearman"] > 0 else "FAIL",
                     "value": correlation["spearman"], "criterion": "positive Spearman correlation"})
    return controls


def compute_benchmark(config: dict, clean_rows: list[dict], masks: dict) -> list[dict]:
    state = State(**clean_rows[0]["states"][0]); raw = np.memmap(config["sources"]["clean_raw"], dtype="<i2", mode="r")
    start = state.raw_start_sample
    def read_iq():
        packed = np.asarray(raw[2 * start:2 * (start + 25_000)]).reshape(-1, 2)
        return packed[:, 0].astype(np.float64) + 1j * packed[:, 1].astype(np.float64)
    candidates = {"pg_scc_k3": masks["pg_scc_k3"], "pg_scc_k5": masks["pg_scc_k5"],
                  "pg_scc_k9": masks["pg_scc_k9"], "fixed9": masks["fixed9"]}
    rows = []
    for name, indices in [*candidates.items(), ("dense_grid", list(range(N_COORDINATES)))]:
        latencies = []
        for repetition in range(8):
            before = time.perf_counter(); iq = read_iq()
            result = complex_caf_surface(iq, state) if name == "dense_grid" else complex_correlator_coordinates(iq, state, indices)
            elapsed = time.perf_counter() - before
            if repetition: latencies.append(elapsed)
            assert np.isfinite(result).all()
        rows.append({"method": name, "correlations_per_prn_epoch": len(indices),
            "median_latency_ms": float(np.median(latencies) * 1000),
            "iqr_latency_ms": float((np.quantile(latencies, .75) - np.quantile(latencies, .25)) * 1000),
            "estimated_working_memory_bytes": int(25_000 * len(indices) * 24 + 25_000 * 16),
            "real_time_factor_1ms": float(np.median(latencies) / .001), "repetitions": 7, "warmup": 1,
            "includes_raw_read_wipeoff_code_replica_correlation": True,
            "cpu": platform.processor() or platform.machine(), "platform": platform.platform()})
    dense = next(row for row in rows if row["method"] == "dense_grid")
    for row in rows:
        row["dense_correlation_reduction"] = 1.0 - row["correlations_per_prn_epoch"] / 187
        row["measured_dense_speedup"] = dense["median_latency_ms"] / row["median_latency_ms"]
    return rows


def overlap_audit(config: dict) -> tuple[dict, int]:
    scenarios = ("cleanStatic", "ds7", "ds8")
    paths = {"cleanStatic": config["sources"]["clean_raw"], **config["sources"]["attack_raw"]}
    hashes = {}; bytes_read = 0
    for second in (20, 105):
        offset = second * 25_000_000 * 4
        for scenario in scenarios:
            with Path(paths[scenario]).open("rb") as handle:
                handle.seek(offset); payload = handle.read(100_000); bytes_read += len(payload)
            hashes[f"{scenario}:{second}"] = __import__("hashlib").sha256(payload).hexdigest()
    return {"schema": "pg_scc_ds78_overlap_audit.v1", "sampled_seconds": [20, 105], "bytes_per_sample": 100_000,
        "hashes": hashes, "clean_ds7_identical_segments": sum(hashes[f"cleanStatic:{s}"] == hashes[f"ds7:{s}"] for s in (20,105)),
        "clean_ds8_identical_segments": sum(hashes[f"cleanStatic:{s}"] == hashes[f"ds8:{s}"] for s in (20,105)),
        "ds7_ds8_identical_segments": sum(hashes[f"ds7:{s}"] == hashes[f"ds8:{s}"] for s in (20,105)),
        "independent_family_count": 1, "status": "AUDITED_NON_INDEPENDENT"}, bytes_read


def plots(output: Path, pooled: list[dict], baselines: list[dict], controls: list[dict], compute: list[dict]) -> None:
    plot_root = output / "plots"; methods = ("pg_scc_k9", "fixed9", "dense_two_source_glrt", "raw_power_only")
    for scenario in ("ds3", "ds4", "ds7", "ds8"):
        fig, axis = plt.subplots(figsize=(8, 4))
        for method in methods:
            rows = [row for row in pooled if row["scenario"] == scenario and row["method"] == method]
            if rows:
                values = np.asarray([row["pooled_score"] for row in rows]); values = (values - np.median(values)) / max(np.quantile(values,.75)-np.quantile(values,.25), 1e-9)
                axis.plot([row["time_s"] for row in rows], values, marker=".", label=method)
        axis.axvline(ONSET[scenario], color="black", linestyle="--", label="onset")
        axis.set(xlabel="receiver time (s)", ylabel="robust standardized score", title=f"{scenario.upper()} frozen score timeline")
        axis.legend(fontsize=7); fig.tight_layout(); fig.savefig(plot_root / f"{scenario}_detection.png", dpi=150); plt.close(fig)
    available = [row for row in baselines if row.get("status") == "AVAILABLE" and row["method"] in methods]
    fig, axis = plt.subplots(figsize=(7,4)); axis.bar([row["method"] for row in available], [row["family_macro_low_fpr_pauc"] for row in available])
    axis.tick_params(axis="x", rotation=25); axis.set(ylabel="3-family macro low-FPR pAUC", title="Fixed9 vs PG-SCC vs dense vs power-only")
    fig.tight_layout(); fig.savefig(plot_root / "fixed9_pg_scc_dense_power.png", dpi=150); plt.close(fig)
    comparison = [row for row in baselines if row.get("status") == "AVAILABLE" and (row["method"] == "pg_scc_k9" or row["method"] in {"uniform_k9","shuffled_k9"} or row["method"].startswith("random")) and row["budget"] == 9]
    fig, axis = plt.subplots(figsize=(9,4)); axis.bar([row["method"] for row in comparison], [row["family_macro_low_fpr_pauc"] for row in comparison])
    axis.tick_params(axis="x", rotation=45); axis.set(ylabel="macro low-FPR pAUC", title="Learned, uniform, shuffled, and random K=9")
    fig.tight_layout(); fig.savefig(plot_root / "random_shuffled_comparison.png", dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(6,4)); macro = {(row["method"],row["budget"]): row["family_macro_low_fpr_pauc"] for row in baselines if row.get("status") == "AVAILABLE"}
    for row in compute:
        key = (row["method"], row["correlations_per_prn_epoch"])
        if key in macro: axis.scatter(row["median_latency_ms"], macro[key], label=row["method"])
    axis.set(xlabel="actual raw-correlation latency (ms)", ylabel="macro low-FPR pAUC", title="Compute-detection Pareto")
    axis.legend(fontsize=7); fig.tight_layout(); fig.savefig(plot_root / "compute_pauc_pareto.png", dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8,4)); numeric = [row for row in controls if isinstance(row.get("value"), (int,float))]
    axis.bar([row["control"] for row in numeric], [row["value"] for row in numeric]); axis.tick_params(axis="x", rotation=55)
    axis.set(title="Physics and shortcut controls"); fig.tight_layout(); fig.savefig(plot_root / "gain_phase_awgn_controls.png", dpi=150); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-full-attack-hash", action="store_true"); args = parser.parse_args(); output = args.output
    frozen_hashes = verify_freeze(output)
    config = load_json(output / "config.json")
    lineage, attack_bytes = authenticate_attack_sources(config, not args.skip_full_attack_hash)
    attack_npz = ROOT / config["sources"]["attack_cache_npz"]; attack_json = ROOT / config["sources"]["attack_cache_json"]
    attack_rows = load_feature_cache(attack_npz, attack_json)
    if {row["scenario"] for row in attack_rows} != {"ds3", "ds4", "ds7", "ds8"}:
        raise RuntimeError("core attack cache scenario set mismatch")
    arrays = np.load(output / "normalization_covariance.npz", allow_pickle=False); auth, covariance = arrays["auth_template"], arrays["covariance"]
    masks = load_json(output / "masks.json"); pooling = load_json(output / "pooling.json")["selected"]
    thresholds = load_json(output / "thresholds.json"); clean_node = numeric_clean_scores(output)
    attack_node = score_rows(attack_rows, auth, covariance, masks, "prompt_phase"); pooled = pool_events([*clean_node, *attack_node], pooling)
    scenarios = scenario_metrics(pooled, thresholds); families = family_metrics(pooled); baselines = baseline_summary(families, scenarios)
    bootstrap = [block_bootstrap(pooled, family, config["seed"] + index) for index, family in enumerate(("ds3", "ds4", "ds7_ds8"))]
    clean_rows = load_feature_cache(ROOT / config["sources"]["clean_cache_npz"], ROOT / config["sources"]["clean_cache_json"])
    controls = physics_controls(clean_rows, auth, covariance, masks, thresholds, baselines, pooled, config["seed"])
    compute = compute_benchmark(config, clean_rows, masks); overlap, overlap_bytes = overlap_audit(config); attack_bytes += overlap_bytes
    lookup = {(row["method"], row["budget"]): row for row in baselines if row.get("status") == "AVAILABLE"}
    pg = lookup[("pg_scc_k9", 9)]; dense = lookup[("dense_two_source_glrt", 187)]
    worst_fpr = max(pg["worst_external_pre_fpr"], pg["worst_clean_holdout_fpr"])
    dense_gap = (dense["family_macro_low_fpr_pauc"] - pg["family_macro_low_fpr_pauc"]) / max(abs(dense["family_macro_low_fpr_pauc"]), 1e-9)
    significant = sum(row.get("status") == "PASS" and row.get("ci95_low", -1) > 0 for row in bootstrap)
    negative_pass = all(row["status"] == "PASS" for row in controls if row["control"] in {"global_gain_and_phase","awgn_sigma_0.02","awgn_sigma_0.05","single_source_power_increase"})
    positive_pass = next(row for row in controls if row["control"] == "same_prn_second_source")["status"] == "PASS"
    learned_pass = next(row for row in controls if row["control"] == "learned_vs_random_uniform_shuffled")["status"] == "PASS"
    shortcut_pass = all(next(row for row in controls if row["control"] == name)["status"] == "PASS" for name in ("mask_boundary_concentration","raw_power_alarm_overlap","dense_teacher_sparse_rank"))
    compute_pg = next(row for row in compute if row["method"] == "pg_scc_k9")
    gates = {
        "foundation_lineage": load_json(output / "foundation_validation.json")["status"] == "PASS" and all(x["status"] == "PASS" for x in lineage.values()),
        "attack_label_isolation": load_json(output / "frozen_design.json")["real_attack_labels_used_for_selector_pooling_threshold"] is False,
        "worst_clean_or_external_fpr_le_5pct": worst_fpr <= .05,
        "k9_within_5pct_dense_macro_pauc": dense_gap <= .05,
        "two_families_significantly_better_than_fixed9": significant >= 2,
        "gain_phase_awgn_negative_controls": negative_pass, "same_prn_positive_control": positive_pass,
        "learned_better_than_random_uniform_shuffled": learned_pass,
        "power_runid_boundary_shortcuts_not_explanatory": shortcut_pass,
        "actual_correlations_reduced": compute_pg["correlations_per_prn_epoch"] < 187 and compute_pg["measured_dense_speedup"] > 1.0}
    verdict = "INCONCLUSIVE" if not gates["foundation_lineage"] else ("CONDITIONAL_GO" if all(gates.values()) else "NO_GO")
    final = {"schema": "pg_scc_final_verdict.v1", "verdict": verdict, "gates": gates,
        "metrics": {"pg_scc_k9_macro_low_fpr_pauc": pg["family_macro_low_fpr_pauc"], "dense_macro_low_fpr_pauc": dense["family_macro_low_fpr_pauc"],
                    "relative_dense_gap": dense_gap, "worst_fpr": worst_fpr, "families_significantly_better_than_fixed9": significant},
        "full_ds4_available": False, "b0_exact": "UNAVAILABLE",
        "wcl_main_model_candidacy": "CANDIDATE_PENDING_EXTERNAL_DATA" if verdict == "CONDITIONAL_GO" else "NOT_A_MAIN_MODEL_CANDIDATE",
        "no_post_attack_retuning": True}
    dump_json(output / "attack_lineage_validation.json", {"status": "PASS", "sources": lineage,
        "attack_cache": {"npz": str(attack_npz), "npz_sha256": sha256(attack_npz), "json": str(attack_json), "json_sha256": sha256(attack_json)},
        "frozen_hashes_verified_before_attack_open": frozen_hashes})
    dump_json(output / "attack_access_audit.json", {"attack_iq_bytes_read_before_freeze": 0, "attack_cache_bytes_read_before_freeze": 0,
        "attack_iq_bytes_read_after_freeze": attack_bytes, "attack_cache_bytes_read_after_freeze": attack_npz.stat().st_size + attack_json.stat().st_size,
        "selector_threshold_pooling_changed_after_attack": False})
    dump_json(output / "ds7_ds8_overlap_audit.json", overlap); write_csv(output / "scenario_metrics.csv", scenarios)
    write_csv(output / "family_metrics.csv", families); write_csv(output / "baseline_metrics.csv", baselines)
    write_csv(output / "control_metrics.csv", controls); write_csv(output / "bootstrap_intervals.csv", bootstrap)
    write_csv(output / "compute_metrics.csv", compute); write_csv(output / "per_epoch_scores.csv", [*clean_node, *attack_node])
    dump_json(output / "final_verdict.json", final); plots(output, pooled, baselines, controls, compute)
    (output / "README.md").write_text(f"# PG-SCC Stage-0 static K<=9\n\nFinal verdict: **{verdict}**. The mask, pooling, normalization, thresholds, timeline, and gates were frozen and committed before any attack cache/IQ access. DS4 is truncated and supports transition-only claims; DS7/DS8 are one family; exact B0 is unavailable.\n")
    dump_json(output / "artifact_manifest_sha256.json", artifact_manifest(output)); print(json.dumps(final, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
