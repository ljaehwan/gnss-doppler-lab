from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

from gnss_doppler_lab.jammertest_crpa_stage0d_power import (
    maximum_cardinality_minimum_cost_match,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/jammertest2025_crpa_stage0d_true_spoof_discrimination"


def brute_force_objective(left, right, caliper):
    best = (0, float("inf"))
    for size in range(1, min(len(left), len(right)) + 1):
        for left_subset in itertools.combinations(left, size):
            for right_subset in itertools.combinations(right, size):
                for permutation in itertools.permutations(right_subset):
                    differences = [abs(a[1] - b[1]) for a, b in zip(left_subset, permutation)]
                    if all(value <= caliper for value in differences):
                        candidate = (size, sum(differences))
                        if candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                            best = candidate
    return best


def test_matching_is_maximum_cardinality_then_minimum_cost() -> None:
    spoof = [(1, 0.00), (2, 0.19), (3, 0.41), (4, 0.90)]
    prn = [(11, 0.10), (12, 0.22), (13, 0.60)]
    caliper = 0.25
    matched = maximum_cardinality_minimum_cost_match(spoof, prn, caliper)
    expected = brute_force_objective(spoof, prn, caliper)
    assert len(matched) == expected[0]
    assert abs(sum(pair.absolute_power_difference_db for pair in matched) - expected[1]) < 1e-12
    assert len({pair.spoof_sample_index for pair in matched}) == len(matched)
    assert len({pair.prn_sample_index for pair in matched}) == len(matched)


def test_matching_is_deterministic_under_input_permutation() -> None:
    spoof = [(1, 0.0), (2, 0.2), (3, 0.4)]
    prn = [(11, 0.1), (12, 0.3), (13, 0.5)]
    first = maximum_cardinality_minimum_cost_match(spoof, prn, 0.21)
    second = maximum_cardinality_minimum_cost_match(list(reversed(spoof)), list(reversed(prn)), 0.21)
    assert first == second


def test_committed_matching_never_crosses_train_and_test() -> None:
    with (ARTIFACT / "split_manifest.csv").open(newline="", encoding="utf-8") as handle:
        split = list(csv.DictReader(handle))
    role = {(row["fold"], row["sample_index"]): row["role"] for row in split}
    with (ARTIFACT / "power_match_manifest.csv").open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    for pair in pairs:
        key_spoof = (pair["fold"], pair["spoof_sample_index"])
        key_prn = (pair["fold"], pair["prn_sample_index"])
        assert role[key_spoof] == pair["role"]
        assert role[key_prn] == pair["role"]


def test_power_match_freeze_is_pre_spatial_and_complete() -> None:
    freeze = json.loads((ARTIFACT / "power_match_freeze.json").read_text())
    assert freeze["status"] == "POWER_MATCH_FREEZE_PRE_SPATIAL_SCORING"
    assert freeze["spatial_feature_bytes_computed"] == 0
    assert freeze["train_test_cross_pairing"] is False
    assert set(freeze["calipers"]) == {"0.10", "0.25", "0.50", "1.00"}


def test_matching_manifest_has_no_replacement_within_fold_role_caliper() -> None:
    with (ARTIFACT / "power_match_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups = {}
    for row in rows:
        groups.setdefault((row["caliper_db"], row["fold"], row["role"]), []).append(row)
    for group in groups.values():
        assert len({row["spoof_sample_index"] for row in group}) == len(group)
        assert len({row["prn_sample_index"] for row in group}) == len(group)
