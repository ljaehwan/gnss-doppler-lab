#!/usr/bin/env python3
"""Fresh-clone verifier for the committed MOSAIC R1c compact artifact."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.mosaic_stage0b_r1c_correction import (
    cliff_delta, decide_recommendation, discriminative_verdict,
)

ART = ROOT / "artifacts/mosaic_stage0b_r1c_root_cause_correction"
R1A = ROOT / "artifacts/mosaic_stage0b_r1a_frozen_analysis"
R1B = ROOT / "artifacts/mosaic_stage0b_r1b_multiprn_root_cause"
REQUIRED = {
    "README.md", "config.json", "source_commit.json", "reproduction_check.json",
    "original_verdict_preservation.json", "failed_vs_successful_metrics.csv",
    "h3_template_discrimination.json", "h4_lock_discrimination.json",
    "h6_temporal_dilution.json", "h1_oracle_recheck.json", "h2_phase_recheck.json",
    "h5_prn_dominance_recheck.json", "effect_sizes.csv", "permutation_tests.csv",
    "bootstrap_intervals.csv", "corrected_hypothesis_verdicts.json",
    "corrected_final_recommendation.json", "paper_safe_claims.md",
    "plots/failure_vs_success_projection_ratio.png",
    "plots/failure_vs_success_oracle_delta_bic_gain.png",
    "plots/lock_loss_fraction_comparison.png", "plots/common_support_loss_comparison.png",
    "plots/cn0_change_comparison.png", "plots/window_length_vs_evidence.png",
    "plots/prn_level_recovery_failure.png", "plots/corrected_hypothesis_verdict_summary.png",
}


def load(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root: Path) -> int:
    manifest = load(root / "artifact_manifest_sha256.json")
    listed = set()
    for item in manifest["files"]:
        path = root / item["path"]
        listed.add(item["path"])
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
    if root == ART:
        if not REQUIRED <= listed:
            raise ValueError(f"missing required R1c files: {sorted(REQUIRED-listed)}")
        actual = {str(path.relative_to(root)) for path in root.rglob("*")
                  if path.is_file() and path.name != "artifact_manifest_sha256.json"}
        if actual != listed:
            raise ValueError("R1c manifest coverage mismatch")
    return len(listed)


def exact_two_metric(rows: list[dict], fields: list[str], failure_count: int) -> dict[str, float]:
    matrix = np.asarray([[float(row[field]) for field in fields] for row in rows])
    centers = matrix.mean(axis=0)
    thresholds = np.abs(matrix[:failure_count].mean(axis=0) - centers) - 1e-15
    extreme = np.zeros(len(fields), dtype=np.int64)
    total = 0
    iterator = itertools.combinations(range(len(matrix)), failure_count)
    while True:
        chunk = list(itertools.islice(iterator, 50000))
        if not chunk:
            break
        means = matrix[np.asarray(chunk, dtype=np.int16)].mean(axis=1)
        extreme += np.sum(np.abs(means-centers) >= thresholds, axis=0)
        total += len(chunk)
    return {field: float(extreme[i] / total) for i, field in enumerate(fields)}


def verify() -> dict:
    count = verify_manifest(ART)
    verify_manifest(R1A)
    verify_manifest(R1B)
    reproduction = load(ART / "reproduction_check.json")
    if reproduction["status"] != "PASS" or reproduction["r1a_final_verdict"] != "NO_GO_MOSAIC_MULTI_PRN_RECOVERY":
        raise ValueError("frozen result reproduction failed")
    if reproduction["r1b_primary_verdict"] != "MIXED_OR_UNIDENTIFIED_ROOT_CAUSE":
        raise ValueError("R1b primary verdict not reproduced")
    with (ART / "failed_vs_successful_metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    failures = [row for row in rows if row["comparator_role"] == "failure_target"]
    successes = [row for row in rows if row["comparator_role"] == "success_comparator"]
    if len(rows) != 28 or len(failures) != 8 or len(successes) != 20:
        raise ValueError("target accounting mismatch")
    if any("H3" in row["row_level_labels"] or "H4" in row["row_level_labels"] for row in rows):
        raise ValueError("global hypothesis label leaked into row labels")
    p_values = exact_two_metric(failures + successes,
                                ["oracle_projection_ratio", "lock_loss_fraction"], len(failures))
    h3_effect = cliff_delta([float(x["oracle_projection_ratio"]) for x in failures],
                            [float(x["oracle_projection_ratio"]) for x in successes])
    h4_effect = cliff_delta([float(x["lock_loss_fraction"]) for x in failures],
                            [float(x["lock_loss_fraction"]) for x in successes])
    h3_failure_presence = all(x["low_oracle_projection"] == "True" for x in failures)
    h3_comparator_presence = any(x["low_oracle_projection"] == "True" for x in successes)
    h3_expected = discriminative_verdict(failure_presence=h3_failure_presence,
                                         comparator_presence=h3_comparator_presence,
                                         complete_separation=h3_failure_presence and not h3_comparator_presence)
    h4_failure_presence = any(x["lock_loss_observed"] == "True" for x in failures)
    h4_comparator_presence = any(x["lock_loss_observed"] == "True" for x in successes)
    h4_expected = discriminative_verdict(failure_presence=h4_failure_presence,
                                         comparator_presence=h4_comparator_presence,
                                         complete_separation=all(x["lock_loss_observed"] == "True" for x in failures) and not h4_comparator_presence)
    hypotheses = load(ART / "corrected_hypothesis_verdicts.json")["verdicts"]
    if hypotheses["H3"]["verdict"] != h3_expected or hypotheses["H4"]["verdict"] != h4_expected:
        raise ValueError("corrected H3/H4 verdict does not recompute")
    if hypotheses["H6"]["verdict"] != "NOT_TESTABLE_FROM_RETAINED_EVIDENCE":
        raise ValueError("H6 retained-evidence limitation lost")
    h1 = load(ART / "h1_oracle_recheck.json")
    restored = sum(row["oracle_recovery"] == "True" and row["recovery"] == "False" for row in failures)
    if restored != h1["failed_targets_restored_by_oracle_coordinates"]:
        raise ValueError("oracle restoration count mismatch")
    stored_recommendation = load(ART / "corrected_final_recommendation.json")
    expected_recommendation = decide_recommendation(**stored_recommendation["recorded_conditions"])
    if stored_recommendation["recommendation"] != expected_recommendation:
        raise ValueError("recommendation truth table mismatch")
    for relative in REQUIRED:
        if relative.endswith(".png") and (ART / relative).read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"not a real PNG: {relative}")
    result = {"status": "PASS", "checksums": count, "failure_targets": len(failures),
              "success_comparators": len(successes), "h3": h3_expected, "h4": h4_expected,
              "h6": hypotheses["H6"]["verdict"], "oracle_restored_failed_targets": restored,
              "recommendation": expected_recommendation, "raw_science_regenerated": False}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    verify()
