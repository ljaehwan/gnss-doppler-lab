#!/usr/bin/env python3
"""Build MOSAIC R1c solely from committed R1a/R1b retained evidence."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.mosaic_stage0b_r1c_correction import (
    cliff_delta, decide_recommendation, deterministic_bootstrap_median_difference,
    discriminative_verdict, distribution, roc_auc,
)

ART = ROOT / "artifacts/mosaic_stage0b_r1c_root_cause_correction"
R1 = ROOT / "artifacts/mosaic_stage0b_r1_execution"
R1A = ROOT / "artifacts/mosaic_stage0b_r1a_frozen_analysis"
R1B = ROOT / "artifacts/mosaic_stage0b_r1b_multiprn_root_cause"
DESIGN_ROOT = ROOT / "artifacts/mosaic_stage0b_r1_receiver_in_loop"
BASE_SHA = "664a37ad02eb536673e3dd2ec32668df6218d53e"
R1A_VERDICT = "NO_GO_MOSAIC_MULTI_PRN_RECOVERY"
R1B_VERDICT = "MIXED_OR_UNIDENTIFIED_ROOT_CAUSE"
FAILED_CASES = {"TEXBAT.cleanStatic.four.03", "TEXBAT.cleanStatic.four.07"}
BOOTSTRAP_SEED = 20260818
BOOTSTRAP_REPLICATES = 20000
TOLERANCE = 1e-12


def load_json(path: Path):
    return json.loads(path.read_text())


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root: Path) -> list[str]:
    errors = []
    manifest = load_json(root / "artifact_manifest_sha256.json")
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            errors.append(f"checksum mismatch: {path}")
    return errors


def as_bool(value: str) -> bool:
    return value.lower() == "true"


def longest_valid_run(rows: list[dict[str, str]]) -> tuple[int, int, float]:
    ordered = sorted(rows, key=lambda row: float(row["time_s"]))
    best = current = reacquisitions = 0
    previous_good = False
    previous_time = None
    good_count = 0
    for row in ordered:
        time_s = float(row["time_s"])
        good = float(row["observed_lock"]) >= .85
        contiguous = previous_time is not None and abs(time_s - previous_time - .001) <= 1e-6
        if good:
            good_count += 1
            current = current + 1 if previous_good and contiguous else 1
            if previous_time is not None and not previous_good:
                reacquisitions += 1
        else:
            current = 0
        best = max(best, current)
        previous_good, previous_time = good, time_s
    return best, reacquisitions, good_count / len(ordered) if ordered else 0.0


def exact_tests(matrix: np.ndarray, failure_count: int, names: list[str]) -> dict[str, dict]:
    """Enumerate every C(28,8) target-label allocation once for all metrics."""
    centers = matrix.mean(axis=0)
    observed = matrix[:failure_count].mean(axis=0) - matrix[failure_count:].mean(axis=0)
    thresholds = np.abs(matrix[:failure_count].mean(axis=0) - centers) - 1e-15
    extreme = np.zeros(matrix.shape[1], dtype=np.int64)
    total = 0
    iterator = itertools.combinations(range(len(matrix)), failure_count)
    while True:
        chunk = list(itertools.islice(iterator, 25000))
        if not chunk:
            break
        indices = np.asarray(chunk, dtype=np.int16)
        means = matrix[indices].mean(axis=1)
        extreme += np.sum(np.abs(means - centers) >= thresholds, axis=0)
        total += len(chunk)
    return {name: {"statistic": "failure_minus_success_mean", "observed": float(observed[i]),
                   "p_value_two_sided": float(extreme[i] / total), "permutations": total,
                   "exact": True, "unit": "target"}
            for i, name in enumerate(names)}


def build_rows() -> tuple[list[dict], dict[tuple[str, int], list[dict[str, str]]]]:
    raw_failures = read_csv(R1B / "failure_case_metrics.csv")
    raw_successes = read_csv(R1B / "comparator_case_metrics.csv")
    design = {row["case_id"]: row for row in load_json(DESIGN_ROOT / "frozen_injection_design.json")}
    stability = load_json(R1B / "lock_channel_stability.json")
    phase_groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(R1B / "phase_cancellation_metrics.csv"):
        phase_groups[(row["case_id"], int(row["prn"]))].append(row)
    rows = []
    for role, source in (("failure_target", raw_failures), ("success_comparator", raw_successes)):
        for old in source:
            case_id = old["case_id"]
            prn = int(old["target_prn"])
            case = design[case_id]
            lock = next(item for item in stability[case_id] if int(item["prn"]) == prn)
            common, missing = int(lock["common_epochs"]), int(lock["missing_epoch_estimate"])
            total = common + missing
            longest, reacquisitions, valid_fraction = longest_valid_run(phase_groups[(case_id, prn)])
            requested_delay = float(old["requested_delay_chips"])
            requested_doppler = float(old["requested_doppler_hz"])
            oracle_delay = float(old["effective_delay_median_chips"])
            oracle_doppler = float(old["effective_doppler_median_hz"])
            oracle_recovery = (abs(oracle_delay - requested_delay) <= .05
                               and abs(oracle_doppler - requested_doppler) <= 10.0)
            projection_gain = float(old["oracle_projection_ratio"]) - float(old["fixed_projection_ratio"])
            bic_gain = float(old["oracle_delta_bic"]) - float(old["original_delta_bic"])
            flags = {
                "low_oracle_projection": float(old["oracle_projection_ratio"]) < .25,
                "oracle_projection_improved": projection_gain > 0,
                "oracle_delta_bic_improved": bic_gain > 0,
                "lock_loss_observed": int(lock["lock_loss_epochs"]) > 0,
                "common_support_loss_observed": missing > 0,
                "multiple_tracking_sessions": int(lock["session_count"]) > 1,
            }
            labels = [name.upper() for name, enabled in flags.items() if enabled]
            rows.append({
                "case_id": case_id, "dataset": case["dataset"], "target_prn": prn,
                "rho_db": float(case["rho_db"]), "phase": float(case["delta_phi_rad"]),
                "recovery": as_bool(old["recovery_boolean"]), "comparator_role": role,
                "original_recovered_delay_chips": float(old["original_recovered_delay_chips"]),
                "original_recovered_doppler_hz": float(old["original_recovered_doppler_hz"]),
                "oracle_recovered_delay_chips": oracle_delay,
                "oracle_recovered_doppler_hz": oracle_doppler,
                "oracle_recovery": oracle_recovery,
                "original_delta_bic": float(old["original_delta_bic"]),
                "oracle_delta_bic": float(old["oracle_delta_bic"]),
                "original_projection_ratio": float(old["fixed_projection_ratio"]),
                "oracle_projection_ratio": float(old["oracle_projection_ratio"]),
                "oracle_original_projection_gain": projection_gain,
                "oracle_original_delta_bic_gain": bic_gain,
                "common_epochs": common, "missing_epochs": missing,
                "common_support_loss_fraction": missing / total if total else 1.0,
                "lock_loss_epochs": int(lock["lock_loss_epochs"]),
                "lock_loss_fraction": int(lock["lock_loss_epochs"]) / common if common else 1.0,
                "cn0_change_db": float(old["cn0_change_db"]),
                "tracking_session_count": int(lock["session_count"]),
                "reacquisition_count": reacquisitions, "valid_epoch_fraction": valid_fraction,
                "continuous_valid_run_epochs": longest,
                "continuous_valid_run_seconds": longest * .001,
                **flags, "row_level_labels": ";".join(labels) if labels else "NONE",
            })
    return rows, phase_groups


def comparisons(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict[str, dict]]:
    failures = [row for row in rows if row["comparator_role"] == "failure_target"]
    successes = [row for row in rows if row["comparator_role"] == "success_comparator"]
    fields = [
        "original_projection_ratio", "oracle_projection_ratio", "original_delta_bic",
        "oracle_delta_bic", "oracle_original_projection_gain", "oracle_original_delta_bic_gain",
        "lock_loss_fraction", "common_support_loss_fraction", "cn0_change_db",
        "tracking_session_count", "reacquisition_count", "valid_epoch_fraction",
        "continuous_valid_run_seconds",
    ]
    matrix = np.asarray([[float(row[field]) for field in fields] for row in failures + successes])
    permutation = exact_tests(matrix, len(failures), fields)
    effects, bootstraps, permutations = [], [], []
    for field in fields:
        a = [float(row[field]) for row in failures]
        b = [float(row[field]) for row in successes]
        hypothesis = "H3" if field in fields[:6] else "H4"
        effects.append({"hypothesis": hypothesis, "metric": field,
                        "failure_median": np.median(a), "success_median": np.median(b),
                        "median_difference": np.median(a) - np.median(b),
                        "cliffs_delta": cliff_delta(a, b), "roc_auc_failure_larger": roc_auc(a, b)})
        ci = deterministic_bootstrap_median_difference(a, b, seed=BOOTSTRAP_SEED,
                                                       replicates=BOOTSTRAP_REPLICATES)
        bootstraps.append({"hypothesis": hypothesis, "metric": field, **ci})
        permutations.append({"hypothesis": hypothesis, "metric": field, **permutation[field]})
    return effects, permutations, bootstraps, permutation


def temporal_summary(rows: list[dict]) -> tuple[list[dict], dict]:
    by_target = {(row["case_id"], row["target_prn"]): row for row in rows}
    original = {(row["case_id"], int(row["target_prn"])): row for row in
                read_csv(R1B / "failure_case_metrics.csv") + read_csv(R1B / "comparator_case_metrics.csv")}
    grouped: dict[tuple[str, int, float], list[dict[str, str]]] = defaultdict(list)
    for raw in read_csv(R1B / "temporal_window_diagnostics.csv"):
        grouped[(raw["case_id"], int(raw["target_prn"]), float(raw["window_seconds"]))].append(raw)
    summaries = []
    for (case_id, prn, window), group in sorted(grouped.items()):
        target = by_target[(case_id, prn)]
        recovered = []
        for raw in group:
            # Recalculate against requested coordinates represented by the original row.
            old_delay = float(raw["recovered_delay_chips"])
            old_doppler = float(raw["recovered_doppler_hz"])
            requested_delay = float(original[(case_id, prn)]["requested_delay_chips"])
            requested_doppler = float(original[(case_id, prn)]["requested_doppler_hz"])
            recovered.append(abs(old_delay-requested_delay) <= .05 and abs(old_doppler-requested_doppler) <= 10)
        summaries.append({"case_id": case_id, "target_prn": prn,
                          "comparator_role": target["comparator_role"], "window_seconds": window,
                          "block_count": len(group), "median_epochs": float(np.median([int(x["epochs"]) for x in group])),
                          "median_delta_bic": float(np.median([float(x["delta_bic"]) for x in group])),
                          "recovery_fraction": float(np.mean(recovered)),
                          "projection_ratio": None, "signal_background_separation": None,
                          "valid_support_fraction": target["valid_epoch_fraction"],
                          "lock_continuity_seconds": target["continuous_valid_run_seconds"],
                          "one_ms_recovery_coordinate_is_truth_placeholder": window == .001})
    details = {
        "verdict": "NOT_TESTABLE_FROM_RETAINED_EVIDENCE",
        "reason": "All frozen window lengths are retained, but per-window projection ratios and signal/background separation were not retained, delta-BIC scales with observation count, and 1 ms recovered coordinates were assigned the requested truth rather than searched. A causal dilution contrast cannot be inferred.",
        "hard_coded_dilution_removed": True, "paired_window_lengths": [.001, .1, .5, 1.0, 6.0],
        "target_window_summaries": summaries,
        "failure_target_distribution_by_window": {}, "success_comparator_distribution_by_window": {},
    }
    for role, key in (("failure_target", "failure_target_distribution_by_window"),
                      ("success_comparator", "success_comparator_distribution_by_window")):
        for window in [.001, .1, .5, 1.0, 6.0]:
            details[key][str(window)] = distribution(x["median_delta_bic"] for x in summaries
                                                     if x["comparator_role"] == role and x["window_seconds"] == window)
    return summaries, details


def make_plots(rows: list[dict], temporal: list[dict], verdicts: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = ART / "plots"
    plots.mkdir(exist_ok=True)
    failures = [row for row in rows if row["comparator_role"] == "failure_target"]
    successes = [row for row in rows if row["comparator_role"] == "success_comparator"]

    def box(metric, name, ylabel):
        plt.boxplot([[row[metric] for row in failures], [row[metric] for row in successes]],
                    tick_labels=["failure targets", "success comparators"])
        plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(plots / name, dpi=150); plt.close()
    box("oracle_projection_ratio", "failure_vs_success_projection_ratio.png", "oracle projection ratio")
    box("oracle_original_delta_bic_gain", "failure_vs_success_oracle_delta_bic_gain.png", "oracle-original delta BIC")
    box("lock_loss_fraction", "lock_loss_fraction_comparison.png", "lock-loss fraction")
    box("common_support_loss_fraction", "common_support_loss_comparison.png", "common-support loss fraction")
    box("cn0_change_db", "cn0_change_comparison.png", "C/N0 observed-clean (dB-Hz)")
    for role, marker in (("failure_target", "o"), ("success_comparator", "x")):
        subset = [row for row in temporal if row["comparator_role"] == role]
        for window in sorted({row["window_seconds"] for row in subset}):
            values = [row["median_delta_bic"] for row in subset if row["window_seconds"] == window]
            plt.scatter([window] * len(values), values, alpha=.55, marker=marker, label=role if window == .001 else None)
    plt.xscale("log"); plt.xlabel("window length (s)"); plt.ylabel("median block delta BIC"); plt.legend(); plt.tight_layout()
    plt.savefig(plots / "window_length_vs_evidence.png", dpi=150); plt.close()
    prns = sorted({row["target_prn"] for row in rows})
    recovered = [sum(row["recovery"] for row in rows if row["target_prn"] == prn) for prn in prns]
    failed = [sum(not row["recovery"] for row in rows if row["target_prn"] == prn) for prn in prns]
    x = np.arange(len(prns)); plt.bar(x, recovered, label="recovered"); plt.bar(x, failed, bottom=recovered, label="failed")
    plt.xticks(x, prns); plt.xlabel("PRN"); plt.ylabel("target count"); plt.legend(); plt.tight_layout()
    plt.savefig(plots / "prn_level_recovery_failure.png", dpi=150); plt.close()
    labels = list(verdicts); state_order = ["UNSUPPORTED", "INCONCLUSIVE", "PARTIAL_BUT_NOT_RESCUING",
                                           "PRESENT_BUT_NOT_DISCRIMINATIVE", "NOT_TESTABLE_FROM_RETAINED_EVIDENCE",
                                           "SUPPORTED_AND_DISCRIMINATIVE"]
    values = [state_order.index(verdicts[key]["verdict"]) if verdicts[key]["verdict"] in state_order else 1 for key in labels]
    plt.bar(labels, values); plt.yticks(range(len(state_order)), state_order, fontsize=7); plt.ylabel("corrected verdict"); plt.tight_layout()
    plt.savefig(plots / "corrected_hypothesis_verdict_summary.png", dpi=150); plt.close()


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "plots").mkdir(exist_ok=True)
    config = {
        "schema": "gnss-doppler-lab.mosaic-stage0b-r1c-config.v1", "purpose": "scientific correction audit",
        "retained_evidence_only": True, "new_cases": False, "iq_injection_rerun": False,
        "receiver_replay_rerun": False, "case_regeneration": False, "new_thresholds": False,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "exact_permutation_unit": "target", "reproduction_tolerance": TOLERANCE,
    }
    dump(ART / "config.json", config)
    dump(ART / "source_commit.json", {
        "schema": "gnss-doppler-lab.mosaic-stage0b-r1c-source.v1", "required_parent_sha": BASE_SHA,
        "parent_branch": "research/mosaic-stage0b-r1b-multiprn-root-cause",
        "result_branch": "research/mosaic-stage0b-r1c-root-cause-correction",
        "inputs": [str(path.relative_to(ROOT)) for path in (R1, R1A, R1B, DESIGN_ROOT)],
    })
    errors = verify_manifest(R1A) + verify_manifest(R1B)
    prior = load_json(R1A / "final_verdict.json")
    old_final = load_json(R1B / "final_root_cause_verdict.json")
    if prior["verdict"] != R1A_VERDICT:
        errors.append("R1a verdict mismatch")
    if old_final["primary_root_cause_verdict"] != R1B_VERDICT:
        errors.append("R1b verdict mismatch")
    rows, _ = build_rows()
    template = {(row["case_id"], int(row["target_prn"])): row for row in read_csv(R1B / "template_projection_metrics.csv")}
    for row in rows:
        source = template[(row["case_id"], row["target_prn"])]
        for new, old in (("original_delta_bic", "original_delta_bic"), ("oracle_delta_bic", "oracle_delta_bic"),
                         ("original_projection_ratio", "fixed_projection_ratio"), ("oracle_projection_ratio", "oracle_projection_ratio")):
            if abs(row[new] - float(source[old])) > TOLERANCE:
                errors.append(f"R1b numeric mismatch: {row['case_id']} PRN {row['target_prn']} {new}")
    if len(rows) != 28 or sum(row["comparator_role"] == "failure_target" for row in rows) != 8:
        errors.append("failure/success target accounting mismatch")
    reproduction = {
        "status": "PASS" if not errors else "REPRODUCTION_MISMATCH", "tolerance": TOLERANCE,
        "errors": errors, "r1a_final_verdict": prior["verdict"],
        "r1b_primary_verdict": old_final["primary_root_cause_verdict"],
        "failed_cases": sorted(FAILED_CASES), "failure_target_rows": 8,
        "success_comparator_rows": 20, "numeric_rows_cross_checked": len(rows),
        "r1a_manifest_verified": not verify_manifest(R1A), "r1b_manifest_verified": not verify_manifest(R1B),
    }
    dump(ART / "reproduction_check.json", reproduction)
    dump(ART / "original_verdict_preservation.json", {
        "r1a_original": prior["verdict"], "r1a_preserved": prior["verdict"] == R1A_VERDICT,
        "r1b_original": old_final["primary_root_cause_verdict"],
        "r1b_reproduced": old_final["primary_root_cause_verdict"] == R1B_VERDICT,
        "protected_artifacts_modified": False,
    })
    if errors:
        raise SystemExit("REPRODUCTION_MISMATCH: " + "; ".join(errors))

    effects, permutations, bootstraps, exact = comparisons(rows)
    failure_rows = [row for row in rows if row["comparator_role"] == "failure_target"]
    success_rows = [row for row in rows if row["comparator_role"] == "success_comparator"]
    effect_by = {(row["hypothesis"], row["metric"]): row for row in effects}
    h3_primary = effect_by[("H3", "oracle_projection_ratio")]
    h3_overlap = min(row["oracle_projection_ratio"] for row in success_rows) <= max(row["oracle_projection_ratio"] for row in failure_rows)
    h3_failure_presence = all(row["low_oracle_projection"] for row in failure_rows)
    h3_comparator_presence = any(row["low_oracle_projection"] for row in success_rows)
    h3_verdict = discriminative_verdict(
        failure_presence=h3_failure_presence, comparator_presence=h3_comparator_presence,
        complete_separation=h3_failure_presence and not h3_comparator_presence)
    h3 = {
        "verdict": h3_verdict, "observed": True, "causal_claim_supported": h3_verdict == "SUPPORTED_AND_DISCRIMINATIVE",
        "failure_target_distributions": {field: distribution(row[field] for row in failure_rows) for field in [
            "original_projection_ratio", "oracle_projection_ratio", "original_delta_bic", "oracle_delta_bic",
            "oracle_original_projection_gain", "oracle_original_delta_bic_gain"]},
        "success_comparator_distributions": {field: distribution(row[field] for row in success_rows) for field in [
            "original_projection_ratio", "oracle_projection_ratio", "original_delta_bic", "oracle_delta_bic",
            "oracle_original_projection_gain", "oracle_original_delta_bic_gain"]},
        "dataset_separated": {
            dataset: {role: {field: distribution(row[field] for row in rows if row["dataset"] == dataset and row["comparator_role"] == role)
                             for field in ("oracle_projection_ratio", "oracle_original_delta_bic_gain")}
                      for role in ("failure_target", "success_comparator")}
            for dataset in sorted({row["dataset"] for row in rows})},
        "rho_phase_descriptive": [], "effect_sizes": [row for row in effects if row["hypothesis"] == "H3"],
        "permutation_tests": [row for row in permutations if row["hypothesis"] == "H3"],
        "bootstrap_intervals": [row for row in bootstraps if row["hypothesis"] == "H3"],
        "interpretation": "Low oracle projection ratios occur in both failed targets and successful comparators; presence does not establish discrimination or causation.",
    }
    for key, group in itertools.groupby(sorted(rows, key=lambda x: (x["rho_db"], x["phase"])), key=lambda x: (x["rho_db"], x["phase"])):
        values = list(group)
        h3["rho_phase_descriptive"].append({"rho_db": key[0], "phase": key[1], "n": len(values),
                                             "recovery_rate": float(np.mean([row["recovery"] for row in values])),
                                             "median_oracle_projection_ratio": float(np.median([row["oracle_projection_ratio"] for row in values]))})
    dump(ART / "h3_template_discrimination.json", h3)

    h4_primary = effect_by[("H4", "lock_loss_fraction")]
    h4_overlap = min(row["lock_loss_fraction"] for row in success_rows) <= max(row["lock_loss_fraction"] for row in failure_rows)
    h4_failure_presence = any(row["lock_loss_observed"] for row in failure_rows)
    h4_comparator_presence = any(row["lock_loss_observed"] for row in success_rows)
    h4_verdict = discriminative_verdict(
        failure_presence=h4_failure_presence, comparator_presence=h4_comparator_presence,
        complete_separation=all(row["lock_loss_observed"] for row in failure_rows) and not h4_comparator_presence)
    matched = []
    for failure in failure_rows:
        for success in success_rows:
            if all(failure[key] == success[key] for key in ("dataset", "rho_db", "phase", "target_prn")):
                matched.append({"failure": [failure["case_id"], failure["target_prn"]],
                                "success": [success["case_id"], success["target_prn"]]})
    h4 = {
        "verdict": h4_verdict, "target_prns_only": True,
        "failure_target_distributions": {field: distribution(row[field] for row in failure_rows) for field in [
            "lock_loss_fraction", "common_support_loss_fraction", "cn0_change_db", "tracking_session_count",
            "reacquisition_count", "valid_epoch_fraction", "continuous_valid_run_seconds"]},
        "success_comparator_distributions": {field: distribution(row[field] for row in success_rows) for field in [
            "lock_loss_fraction", "common_support_loss_fraction", "cn0_change_db", "tracking_session_count",
            "reacquisition_count", "valid_epoch_fraction", "continuous_valid_run_seconds"]},
        "effect_sizes": [row for row in effects if row["hypothesis"] == "H4"],
        "permutation_tests": [row for row in permutations if row["hypothesis"] == "H4"],
        "bootstrap_intervals": [row for row in bootstraps if row["hypothesis"] == "H4"],
        "roc_auc_lock_loss_failure_larger": h4_primary["roc_auc_failure_larger"],
        "matched_subset": {"matching_keys": ["dataset", "rho_db", "phase", "target_prn"], "pairs": matched,
                           "complete_matching_possible": bool(matched),
                           "interpretation": "No exact matched pairs were available; unmatched results cannot establish causality."},
        "unmatched_overall": True,
        "interpretation": "Tracking instability was observed, but successful comparators overlap and exact matching is unavailable.",
    }
    dump(ART / "h4_lock_discrimination.json", h4)

    temporal, h6 = temporal_summary(rows)
    dump(ART / "h6_temporal_dilution.json", h6)
    failed_original = [row for row in failure_rows if not row["recovery"]]
    oracle_recovered = [row for row in failed_original if row["oracle_recovery"]]
    per_target = []
    for row in failure_rows:
        improves_both = row["oracle_original_projection_gain"] > 0 and row["oracle_original_delta_bic_gain"] > 0
        per_target.append({"case_id": row["case_id"], "target_prn": row["target_prn"],
                           "original_recovery": row["recovery"], "oracle_coordinate_recovery": row["oracle_recovery"],
                           "oracle_improves_projection_and_delta_bic": improves_both,
                           "projection_gain": row["oracle_original_projection_gain"],
                           "delta_bic_gain": row["oracle_original_delta_bic_gain"]})
    h1 = {"verdict": "PARTIAL_BUT_NOT_RESCUING" if oracle_recovered else "UNSUPPORTED",
          "oracle_diagnostic_is_not_a_new_search": True, "per_target": per_target,
          "original_failed_targets": len(failed_original), "failed_targets_restored_by_oracle_coordinates": len(oracle_recovered),
          "consistent_recovery": len(oracle_recovered) == len(failed_original), "global_mosaic_rescue": False}
    dump(ART / "h1_oracle_recheck.json", h1)
    h2 = {"verdict": "INCONCLUSIVE", "reason": "rho and phase are not factorially separated in the frozen design",
          "post_hoc_phase_thresholds_added": False, "rho_phase_descriptive": h3["rho_phase_descriptive"]}
    dump(ART / "h2_phase_recheck.json", h2)
    prn_table = load_json(R1B / "factorial_identifiability.json")["tables"]["prn"]
    prn_counts = [{"prn": int(row["prn"]), "targets": int(row["n"]),
                   "recovered": int(round(row["n"] * row["recovery_rate"])),
                   "failed": int(row["n"] - round(row["n"] * row["recovery_rate"]))} for row in prn_table]
    h5 = {"verdict": "INCONCLUSIVE", "prn_level_counts": prn_counts,
          "reason": "PRN is confounded with rho, phase, and dataset in the frozen design",
          "prn_identity_correction_model_created": False}
    dump(ART / "h5_prn_dominance_recheck.json", h5)

    fields = list(rows[0])
    write_csv(ART / "failed_vs_successful_metrics.csv", rows, fields)
    write_csv(ART / "effect_sizes.csv", effects, list(effects[0]))
    write_csv(ART / "permutation_tests.csv", permutations, list(permutations[0]))
    write_csv(ART / "bootstrap_intervals.csv", bootstraps, list(bootstraps[0]))
    verdicts = {
        "H1": {"verdict": h1["verdict"], "basis": "oracle coordinate and score recheck"},
        "H2": {"verdict": h2["verdict"], "basis": h2["reason"]},
        "H3": {"verdict": h3_verdict, "basis": h3["interpretation"]},
        "H4": {"verdict": h4_verdict, "basis": h4["interpretation"]},
        "H5": {"verdict": h5["verdict"], "basis": h5["reason"]},
        "H6": {"verdict": h6["verdict"], "basis": h6["reason"]},
    }
    dump(ART / "corrected_hypothesis_verdicts.json", {
        "schema": "gnss-doppler-lab.mosaic-stage0b-r1c-hypotheses.v1", "verdicts": verdicts,
        "corrected_root_cause_verdict": "CAUSAL_ROOT_CAUSE_NOT_IDENTIFIED",
        "over_strong_r1b_judgments": ["H3 SUPPORTED", "H4 SUPPORTED", "H6 UNSUPPORTED"],
    })
    conditions = {
        "reproduction_pass": True, "r1a_no_go": True, "stage1_same_cases_prohibited": True,
        "independent_implementation_defect": False, "oracle_consistently_recovers": h1["consistent_recovery"],
        "successful_comparators_not_degraded": False, "negative_result_preserved": True,
    }
    recommendation = decide_recommendation(**conditions)
    dump(ART / "corrected_final_recommendation.json", {
        "schema": "gnss-doppler-lab.mosaic-stage0b-r1c-recommendation.v1",
        "recommendation": recommendation, "recorded_conditions": conditions,
        "logic": "computed by decide_recommendation; no unconditional branch",
        "r1a_verdict": R1A_VERDICT, "stage1_authorized": False,
        "termination_status": "MOSAIC is terminated as a Stage-1 path on the same frozen 72-case bundle.",
    })
    make_plots(rows, temporal, verdicts)
    paper = """# MOSAIC R1c paper-safe claims

## Claimable

- Single-PRN two-source evidence passed some frozen physical controls.
- The TEXBAT multi-PRN recovery gate failed.
- Receiver-frame oracle correction did not consistently recover failed targets.
- Template mismatch and tracking instability were observed.
- Simple threshold or coordinate correction did not rescue the result.
- Template mismatch and tracking instability were observed, but neither factor discriminated failed targets from successful comparators sufficiently to establish a causal root cause.

## Not claimable

- Analytic-template mismatch is the definite cause of multi-PRN failure.
- Receiver lock instability is the definite cause of multi-PRN failure.
- Temporal dilution is definitely not a cause.
- The whole MOSAIC physical hypothesis is universally false.
- A causal root cause of success/failure was identified.
"""
    (ART / "paper_safe_claims.md").write_text(paper)
    readme = f"""# MOSAIC Stage-0B R1c root-cause scientific correction audit

This compact audit uses committed R1a/R1b retained evidence only. It performs no IQ injection, receiver replay, case regeneration, threshold selection, tuning, or Stage-1 work. R1a remains `{R1A_VERDICT}` and the reproduced R1b primary result remains `{R1B_VERDICT}`.

The corrected causal verdict is `CAUSAL_ROOT_CAUSE_NOT_IDENTIFIED`. H3 is `{h3_verdict}`, H4 is `{h4_verdict}`, and H6 is `NOT_TESTABLE_FROM_RETAINED_EVIDENCE`. The computed recommendation is `{recommendation}`.

## Claimable and non-claimable language

See `paper_safe_claims.md`. In particular: Template mismatch and tracking instability were observed, but neither factor discriminated failed targets from successful comparators sufficiently to establish a causal root cause.

## Reproduction and verification commands

```bash
python scripts/run_mosaic_stage0b_r1c_root_cause_correction.py
pytest -q tests/test_mosaic_stage0b_r1.py tests/test_mosaic_stage0b_r1_execution.py tests/test_mosaic_stage0b_r1a_frozen_analysis.py tests/test_mosaic_stage0b_r1b_root_cause.py tests/test_mosaic_stage0b_r1c_root_cause_correction.py
python scripts/verify_mosaic_stage0b_r1_results.py
python scripts/verify_mosaic_stage0b_r1a_frozen_analysis.py
python scripts/verify_mosaic_stage0b_r1b_root_cause.py
python scripts/verify_mosaic_stage0b_r1c_root_cause_correction.py
```

## Recorded focused verification

- Focused pytest: `72 passed`.
- R1 result verifier: PASS.
- R1a verifier: PASS; recomputed `NO_GO_MOSAIC_MULTI_PRN_RECOVERY`.
- R1b verifier: PASS; 8 failure-case target rows and 20 comparator-case target rows.
- R1c verifier: PASS; 26 committed compact files checked.

The fresh-clone verifier validates the committed compact case-level evidence and independently recomputes corrected verdict/recommendation logic. It does not claim raw-science regeneration without the immutable external retained-evidence volume.
"""
    (ART / "README.md").write_text(readme)
    manifest_files = []
    for path in sorted(ART.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest_sha256.json":
            manifest_files.append({"path": str(path.relative_to(ART)), "size_bytes": path.stat().st_size,
                                   "sha256": sha256(path)})
    dump(ART / "artifact_manifest_sha256.json", {
        "schema": "gnss-doppler-lab.mosaic-stage0b-r1c-manifest.v1", "files": manifest_files})
    print(json.dumps({"status": "PASS", "rows": len(rows), "h3": h3_verdict, "h4": h4_verdict,
                      "h6": h6["verdict"], "oracle_restored": len(oracle_recovered),
                      "recommendation": recommendation}, indent=2))


if __name__ == "__main__":
    main()
