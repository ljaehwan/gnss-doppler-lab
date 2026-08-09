from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from gnss_doppler_lab.pg_scc_r2_support import (
    aggregate_accounting, assert_support_only_selection, common_support_by_event,
    family_is_eligible, method_support, require_identical_paired_support,
    support_stratum, universe_support,
)


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "artifacts/pg_scc_stage0_r1_root_cause_audit"
R1_HASHES = {
    "config.json": "ade700892fe3e055e0154ae859ba701a36f11a6ec9245db4b39fd087426eb0e6",
    "source_commit.json": "13ab9e544f06903dd6fde04c42e8aefb35a66946bbaf0456e95040fbb777d428",
    "r1_fail_closed_report.json": "041cc432cdc893e9dba867d6d3dc005e3ee7f2c8d25d542d54d4f374ca68e3f5",
}
FAMILIES = {
    "K9": ("pg_scc_k9", "fixed9", "shuffled_k9"),
    "K5": ("pg_scc_k5", "uniform_k5", "shuffled_k5"),
    "K3": ("pg_scc_k3", "epl3", "shuffled_k3"),
    "DENSE": ("dense_two_source_glrt",),
}


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows_for_counts(counts: list[int]) -> list[dict[str, object]]:
    rows = []
    for second, count in enumerate(counts):
        for prn in range(1, count + 1):
            rows.append({
                "source_role": "synthetic", "scenario": "unit", "phase": "holdout",
                "second": second, "time_s": second + prn / 1000, "prn": prn,
            })
    return rows


def method_rows(universe: list[dict[str, object]], missing: tuple[str, int, int] | None = None):
    output = []
    for method in sorted({name for values in FAMILIES.values() for name in values}):
        for row in universe:
            if missing and method == missing[0] and int(row["second"]) == missing[1] and int(row["prn"]) == missing[2]:
                continue
            output.append({**row, "method": method, "score": 999.0})
    return output


def test_support_boundaries_below3_exact3_4_5_8_9plus_and_k_eligibility():
    counts = [1, 2, 3, 4, 5, 8, 9, 11]
    assert [support_stratum(value) for value in counts] == [
        "DENSE_ONLY", "DENSE_ONLY", "K3", "K3", "K5", "K5", "K9", "K9",
    ]
    expected = {
        "K9": [False, False, False, False, False, False, True, True],
        "K5": [False, False, False, False, True, True, True, True],
        "K3": [False, False, True, True, True, True, True, True],
        "DENSE": [True] * 8,
    }
    for family, values in expected.items():
        assert [family_is_eligible(family, value) for value in counts] == values
    universe = rows_for_counts(counts)
    report = aggregate_accounting(universe, method_rows(universe), FAMILIES)
    assert report["no_event_drop"] is True
    assert report["universe"]["total_events"] == len(counts)
    assert report["universe"]["exclusive_support_strata"] == {
        "K9": 2, "K5": 2, "K3": 2, "DENSE_ONLY": 2, "UNSUPPORTED": 0,
    }


def test_method_missingness_reduces_common_cardinality_but_never_drops_event():
    universe = rows_for_counts([5, 9])
    scored = method_rows(universe, missing=("shuffled_k5", 0, 5))
    report = aggregate_accounting(universe, scored, FAMILIES)
    assert report["comparison_families"]["K5"]["total_events"] == 2
    assert report["comparison_families"]["K5"]["common_unique_prn_histogram"] == {"4": 1, "9": 1}
    assert report["method_availability"]["shuffled_k5"]["available_events"] == 2
    assert all(item["available_events"] + item["unavailable_events"] == 2
               for item in report["method_availability"].values())


def test_no_event_drop_and_denominator_or_extra_event_errors_fail_closed():
    universe_rows = rows_for_counts([3, 5])
    universe = universe_support(universe_rows)
    supports = method_support(method_rows(universe_rows), FAMILIES["K3"])
    common = common_support_by_event(universe, supports, FAMILIES["K3"])
    assert set(common) == set(universe)
    extra = {method: dict(values) for method, values in supports.items()}
    extra[FAMILIES["K3"][0]] = {**extra[FAMILIES["K3"][0]], ("synthetic", "unit", "holdout", 99): {1}}
    with pytest.raises(RuntimeError, match="outside universe"):
        common_support_by_event(universe, extra, FAMILIES["K3"])
    changed = {event: set(prns) for event, prns in common.items()}
    changed[next(iter(changed))].discard(1)
    with pytest.raises(RuntimeError, match="identical"):
        require_identical_paired_support(common, changed)
    with pytest.raises(RuntimeError, match="denominator"):
        require_identical_paired_support(common, dict(list(changed.items())[1:]))


def test_leakage_and_outcome_dependent_selection_are_rejected():
    assert_support_only_selection(("source_role", "scenario", "phase", "second", "prn", "method_availability"))
    for field in ("score", "label", "outcome", "threshold", "auroc", "effect"):
        with pytest.raises(RuntimeError, match="outcome-dependent"):
            assert_support_only_selection(("scenario", "phase", "second", "prn", field))
    with pytest.raises(RuntimeError, match="non-preregistered"):
        assert_support_only_selection(("scenario", "phase", "second", "prn", "time_s"))


def test_inventory_is_json_only_rejects_outcome_fields_and_emits_no_ids(tmp_path):
    inventory = load_script("r2_inventory", ROOT / "scripts/inventory_pg_scc_r2_support.py")
    with pytest.raises(RuntimeError, match="only clean_features"):
        inventory.load_support_metadata(tmp_path / "attack_features.npz")
    bad = tmp_path / "attack_features.json"
    bad.write_text(json.dumps([{"scenario": "x", "phase": "p", "second": 0, "prn": 1, "score": 3.0}]))
    with pytest.raises(RuntimeError, match="outcome-bearing"):
        inventory.load_support_metadata(bad)
    summary = inventory.support_histogram((("synthetic", rows_for_counts([3, 9])),))
    encoded = json.dumps(summary)
    assert "unit" not in encoded and '"prn":' not in encoded
    assert summary["committed_prn_sets"] is False
    assert summary["outcome_values_accessible"] is False


def test_relationship_permutation_controls_and_support_fingerprints_are_mandatory():
    config = json.loads((ROOT / "configs/pg_scc_stage0_r2_support_feasibility.json").read_text())
    for family in ("K9", "K5", "K3"):
        methods = config["comparison_families"][family]
        assert methods[0].startswith("pg_scc")
        assert "shuffled" in methods[2]
    producer = (ROOT / "scripts/run_pg_scc_r2_support_feasibility.py").read_text()
    outcomes = (ROOT / "src/gnss_doppler_lab/pg_scc_r2_outcomes.py").read_text()
    assert "strict_calibration_and_pairs(scored, config)" in producer
    assert "RELATIONSHIP_PERMUTATION" in outcomes
    assert "support_fingerprint_left" in outcomes and "support_fingerprint_right" in outcomes
    verifier = (ROOT / "scripts/verify_pg_scc_r2_support_feasibility.py").read_text()
    assert "relationship_permutation_missing" in verifier
    assert "paired_support_not_identical" in verifier


def test_manifest_tampering_is_detected_by_independent_rehash(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    payload = artifact / "paired_results.json"
    payload.write_text('{"status":"AVAILABLE"}\n')
    manifest = {payload.name: hashlib.sha256(payload.read_bytes()).hexdigest()}
    (artifact / "artifact_manifest_sha256.json").write_text(json.dumps(manifest))
    actual = hashlib.sha256(payload.read_bytes()).hexdigest()
    assert manifest[payload.name] == actual
    payload.write_text('{"status":"TAMPERED"}\n')
    assert manifest[payload.name] != hashlib.sha256(payload.read_bytes()).hexdigest()


def test_r1_immutability_and_fail_closed_report_remain_exact():
    for name, expected in R1_HASHES.items():
        assert hashlib.sha256((R1 / name).read_bytes()).hexdigest() == expected
    report = json.loads((R1 / "r1_fail_closed_report.json").read_text())
    assert report["status"] == "FAIL_CLOSED"
    assert report["blocker_code"] == "common_support_event_has_fewer_than_4_unique_prns"
    assert report["assertions"]["audit_rerun"] is False


def test_freeze_guard_precedes_metadata_and_protected_open_and_verifier_never_runs_producer():
    producer_path = ROOT / "scripts/run_pg_scc_r2_support_feasibility.py"
    tree = ast.parse(producer_path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [ast.unparse(node) for node in ast.walk(main) if isinstance(node, ast.Call)]
    freeze = min(i for i, call in enumerate(calls) if "verify_implementation_freeze" in call)
    metadata = min(i for i, call in enumerate(calls) if "load_support_metadata" in call)
    protected = min(i for i, call in enumerate(calls) if "load_protected_records" in call)
    assert freeze < metadata < protected
    verifier_tree = ast.parse((ROOT / "scripts/verify_pg_scc_r2_support_feasibility.py").read_text())
    subprocess_calls = [ast.unparse(node) for node in ast.walk(verifier_tree)
                       if isinstance(node, ast.Call) and "subprocess" in ast.unparse(node.func)]
    assert not any("run_pg_scc_r2_support_feasibility.py" in call for call in subprocess_calls)
