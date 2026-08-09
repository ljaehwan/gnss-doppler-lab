from __future__ import annotations

import copy
import json
from pathlib import Path

from gnss_doppler_lab.pg_scc_r2_outcomes import COMPARISONS, calibration_and_pairs


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_rows():
    events = []
    second = 0
    for source_role, scenario, phase, count in (
        ("clean", "cleanStatic", "calibration", 10),
        ("clean", "cleanStatic", "holdout", 10),
        ("attack", "ds3", "strict_pre", 10),
        ("attack", "ds3", "attack", 10),
        ("attack", "ds4", "transition", 10),
        ("attack", "ds7", "attack", 5),
        ("attack", "ds8", "attack", 5),
    ):
        for _ in range(count):
            for prn in range(1, 10):
                events.append({"source_role": source_role, "scenario": scenario, "phase": phase,
                               "second": second, "prn": prn})
            second += 10
    rows = []
    for method in sorted({method for values in COMPARISONS.values() for method in values}):
        for event in events:
            attacked = event["source_role"] == "attack" and event["phase"] != "strict_pre"
            base = 0.8 if attacked else (0.05 if event["phase"] == "calibration" else 0.04)
            if method.startswith("pg_scc"):
                score = base + (0.2 if attacked else 0.0)
            elif "shuffled" in method:
                score = base - (0.1 if attacked else 0.0)
            else:
                score = base
            rows.append({**event, "method": method, "score": score})
    return rows


def test_strict_calibration_uncertainty_false_alarm_and_relational_estimands():
    config = copy.deepcopy(json.loads((ROOT / "configs/pg_scc_stage0_r2_support_feasibility.json").read_text()))
    config["minimum_gates"]["bootstrap_iterations"] = 20
    calibration, paired = calibration_and_pairs(_synthetic_rows(), config)
    k9_focal = next(cell for cell in calibration["cells"]
                    if cell["family"] == "K9" and cell["support_stratum"] == "K9"
                    and cell["method"] == "pg_scc_k9")
    assert k9_focal["status"] == "AVAILABLE"
    assert k9_focal["clean_holdout_clopper_pearson_95"] is not None
    assert k9_focal["strict_external_pre_clopper_pearson_95"] is not None
    assert k9_focal["threshold_leave_one_block_out_range"] is not None
    assert k9_focal["threshold_block_bootstrap_ci95"] is not None
    assert k9_focal["false_alarm_gate"] is True
    permutation = [cell for cell in paired["cells"]
                   if cell["k_family"] == "K9" and cell["support_stratum"] == "K9" and cell["control_role"] == "RELATIONSHIP_PERMUTATION"]
    assert len(permutation) == 3
    assert all(cell["support_fingerprint_left"] == cell["support_fingerprint_right"] for cell in permutation)
    assert all(cell["paired_events"] == 10 for cell in permutation)
    assert all(cell["paired_pauc_difference"] is not None for cell in permutation)
    aggregate = paired["aggregate_estimands"]
    assert aggregate["total_preregistered_cells"] > 0
    assert aggregate["available_cells"] > 0
    assert aggregate["equal_stratum_mean_paired_effect"] is not None
    assert aggregate["equal_outcome_family_mean_paired_effect"] is not None
    assert aggregate["included_event_weight"] > 0
    assert aggregate["raw_scores_pooled_across_k_or_strata"] is False
