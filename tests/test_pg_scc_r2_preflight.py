from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
R1_ARTIFACT_HASHES = {
    "config.json": "ade700892fe3e055e0154ae859ba701a36f11a6ec9245db4b39fd087426eb0e6",
    "source_commit.json": "13ab9e544f06903dd6fde04c42e8aefb35a66946bbaf0456e95040fbb777d428",
    "r1_fail_closed_report.json": "041cc432cdc893e9dba867d6d3dc005e3ee7f2c8d25d542d54d4f374ca68e3f5",
}


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "pg_scc_root_cause_audit_r2", ROOT / "scripts/run_pg_scc_root_cause_audit.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_pg_scc_root_cause_audit_r2",
        ROOT / "scripts/verify_pg_scc_root_cause_audit.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event_rows(
    method: str,
    *,
    second: int,
    prns: tuple[int, ...],
    time_offsets: tuple[float, ...] | None = None,
    channels: tuple[int, ...] | None = None,
) -> list[dict[str, object]]:
    offsets = time_offsets or tuple(0.0 for _ in prns)
    channel_values = channels or tuple(0 for _ in prns)
    assert len(prns) == len(offsets) == len(channel_values)
    return [
        {
            "scenario": "cleanStatic",
            "phase": "holdout",
            "second": second,
            "time_s": float(second) + offset,
            "prn": prn,
            "channel": channel,
            "method": method,
            "score": float(prn),
        }
        for prn, offset, channel in zip(prns, offsets, channel_values)
    ]


def test_r2_gate_1_four_prns_with_tracker_specific_times_form_one_pooled_event():
    runner = load_runner()
    offsets = (0.001, 0.004, 0.007, 0.009)
    rows = []
    for method in ("a", "b"):
        rows.extend(
            _event_rows(
                method,
                second=12,
                prns=(3, 7, 11, 19),
                time_offsets=offsets,
                channels=(1, 2, 3, 4),
            )
        )
    report = runner.validate_common_support(rows, ("a", "b"), minimum_prns=4)
    assert report["status"] == "PASS"
    assert report["eligible_event_count"] == 1
    assert report["excluded_event_count"] == 0
    assert report["pooled_event_unique_prn_counts"] == [
        {
            "eligible": True,
            "identity": {"scenario": "cleanStatic", "phase": "holdout", "second": 12},
            "prn_count": 4,
            "prns": [3, 7, 11, 19],
        }
    ]


def test_r2_gate_2_true_three_prn_event_is_excluded_identically_and_recorded():
    runner = load_runner()
    rows = []
    for method in ("a", "b"):
        rows.extend(_event_rows(method, second=20, prns=(3, 7, 11, 19)))
        rows.extend(_event_rows(method, second=21, prns=(3, 7, 11)))
    report = runner.validate_common_support(rows, ("a", "b"), minimum_prns=4)
    assert report["eligible_event_count"] == 1
    assert report["excluded_event_count"] == 1
    assert report["excluded_pooled_events"] == [
        {
            "identity": {"scenario": "cleanStatic", "phase": "holdout", "second": 21},
            "prn_count": 3,
            "prns": [3, 7, 11],
        }
    ]
    assert len(set(report["per_detector_eligible_event_support_hashes"].values())) == 1


def test_r2_gate_3_different_detector_raw_row_support_fails_closed_precisely():
    runner = load_runner()
    rows = _event_rows("a", second=30, prns=(3, 7, 11, 19))
    rows += _event_rows("b", second=30, prns=(3, 7, 11))
    with pytest.raises(RuntimeError, match=r"raw-row support mismatch.*detector=b"):
        runner.validate_common_support(rows, ("a", "b"), minimum_prns=4)


def test_r2_gate_4_different_detector_eligible_event_support_fails_closed_precisely():
    runner = load_runner()
    event = ("cleanStatic", "holdout", 40)
    supports = {
        "a": {event: {"prns": [3, 7, 11, 19], "prn_count": 4}},
        "b": {},
    }
    with pytest.raises(
        RuntimeError,
        match=r"eligible-event support mismatch.*detector=b.*pooled_event=.*second.*40.*prn_count=4",
    ):
        runner.validate_eligible_event_support(supports)


def test_r2_gate_5_same_prn_on_multiple_channels_counts_once_for_minimum():
    runner = load_runner()
    rows = []
    for method in ("a", "b"):
        rows.extend(
            _event_rows(
                method,
                second=50,
                prns=(3, 7, 11, 11),
                channels=(1, 2, 3, 4),
            )
        )
    report = runner.validate_common_support(rows, ("a", "b"), minimum_prns=4)
    assert report["eligible_event_count"] == 0
    assert report["excluded_event_count"] == 1
    assert report["excluded_pooled_events"][0]["prn_count"] == 3
    assert report["excluded_pooled_events"][0]["prns"] == [3, 7, 11]


def test_r2_gate_6_frozen_pg_scc_pooled_event_count_is_reproduced():
    runner = load_runner()
    metadata = [
        {key: value for key, value in row.items() if key not in {"method", "score"}}
        for row in _event_rows(
            "reference",
            second=60,
            prns=(3, 7, 11, 19),
            time_offsets=(0.001, 0.004, 0.007, 0.009),
            channels=(1, 2, 3, 4),
        )
    ]
    expected = runner.frozen_pg_scc_pooled_event_count(metadata)
    assert expected == 1
    rows = [{**row, "method": method, "score": 0.0} for method in ("a", "b") for row in metadata]
    report = runner.validate_common_support(
        rows, ("a", "b"), minimum_prns=4, frozen_expected_count=expected
    )
    assert report["frozen_pooled_event_expected_count"] == 1
    assert report["reconstructed_pooled_event_count"] == 1
    assert report["frozen_pooled_event_count_match"] is True


def test_r2_gate_7_r1_artifacts_remain_byte_identical_by_checksum():
    runner = load_runner()
    report = runner.verify_r1_artifact_immutability()
    assert report["status"] == "PASS"
    assert report["sha256"] == R1_ARTIFACT_HASHES


def test_r2_gate_8_minimal_forged_pass_cannot_open_protected_inputs(tmp_path):
    runner = load_runner()
    calls: list[str] = []
    forged = {
        "schema": "pg_scc_stage0_r2_support_preflight.v1",
        "status": "PASS",
        "protected_score_fields_read": 0,
        "frozen_pooled_event_count_match": True,
    }
    preflight_path = tmp_path / "support_preflight.json"
    preflight_path.write_text(json.dumps(forged), encoding="utf-8")

    def protected_loader():
        calls.append("protected")
        return {"loaded": True}

    with pytest.raises(RuntimeError, match="FAIL_CLOSED_SUPPORT_PREFLIGHT"):
        runner.load_protected_inputs_after_preflight(
            preflight_path=preflight_path, protected_loader=protected_loader
        )
    assert calls == []


def test_r2_gate_9_threshold_mask_and_score_formulas_are_identical_to_r1():
    runner = load_runner()
    report = runner.assert_frozen_formula_identity()
    assert report["status"] == "PASS"
    assert {
        "batch_glrt",
        "diagnostic_scores",
        "metric_bundle",
        "pooled_events",
        "validate_nested_masks",
    } <= set(report["function_ast_sha256"])
    assert {"normalization", "score", "selector", "covariance", "synthetic", "mask", "threshold"} == set(
        report["groups"]
    )
    assert all(item["match"] for item in report["function_ast_sha256"].values())
    assert report["frozen_artifact_sha256"]["masks.json"]["match"] is True
    assert report["frozen_artifact_sha256"]["thresholds.json"]["match"] is True


def test_remediation_full_report_accounting_tamper_blocks_loader(tmp_path):
    runner = load_runner()
    report = json.loads(
        (ROOT / "artifacts/pg_scc_stage0_r2_root_cause_audit/support_preflight.json").read_text()
    )
    report["eligible_event_count"] += 1
    path = tmp_path / "support_preflight.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    called = []

    def protected_loader():
        called.append(True)
        return {}

    with pytest.raises(RuntimeError, match="FAIL_CLOSED_SUPPORT_PREFLIGHT"):
        runner.load_protected_inputs_after_preflight(
            preflight_path=path, protected_loader=protected_loader
        )
    assert called == []


def test_remediation_duplicate_channel_prn_is_removed_from_every_metric_input():
    runner = load_runner()
    metadata = []
    for second, prns, channels in (
        (70, (3, 7, 11, 19), (1, 2, 3, 4)),
        (71, (3, 7, 11, 11), (1, 2, 3, 4)),
    ):
        metadata.extend([
            {key: value for key, value in row.items() if key not in {"method", "score"}}
            for row in _event_rows(
                "reference", second=second, prns=prns, channels=channels
            )
        ])
    detector_rows = [
        {**row, "method": method, "score": float(index)}
        for method in ("a", "b")
        for index, row in enumerate(metadata)
    ]
    support = runner.validate_common_support(detector_rows, ("a", "b"), minimum_prns=4)
    scores = {
        method: np.arange(len(metadata), dtype=float) + offset
        for offset, method in enumerate(("a", "b"))
    }
    filtered_metadata, filtered_scores = runner.filter_metric_inputs_to_eligible_events(
        metadata, scores, support
    )
    assert {int(row["second"]) for row in filtered_metadata} == {70}
    for method in ("a", "b"):
        assert len(filtered_scores[method]) == 4
        events = runner.pooled_events(filtered_metadata, filtered_scores[method])
        assert [event["second"] for event in events] == [70]


def test_remediation_failure_diagnostics_report_accurate_unique_prns():
    runner = load_runner()
    rows = _event_rows("a", second=80, prns=(3, 7, 11, 19))
    rows += _event_rows("b", second=80, prns=(3, 7, 11))
    with pytest.raises(RuntimeError) as caught:
        runner.validate_common_support(rows, ("a", "b"), minimum_prns=4)
    message = str(caught.value)
    assert '"second": 80' in message
    assert "reference_unique_prn_count=4" in message
    assert "reference_prns=[3, 7, 11, 19]" in message
    assert "detector_unique_prn_count=3" in message
    assert "detector_prns=[3, 7, 11]" in message

    duplicated = _event_rows("a", second=81, prns=(3, 7, 11, 19))
    duplicated.append(dict(duplicated[0]))
    duplicated += _event_rows("b", second=81, prns=(3, 7, 11, 19))
    with pytest.raises(RuntimeError) as caught:
        runner.validate_common_support(duplicated, ("a", "b"), minimum_prns=4)
    message = str(caught.value)
    assert '"second": 81' in message
    assert "unique_prn_count=4" in message
    assert "prns=[3, 7, 11, 19]" in message


def test_remediation_expected_count_uses_immutable_frozen_record_projection():
    runner = load_runner()
    frozen = runner.load_frozen_expected_support()
    assert frozen["source"]["path"] == "artifacts/pg_scc_stage0_static_k9/per_epoch_scores.csv"
    assert frozen["source"]["sha256"] == "d3fd253807aea873dadce767fc027d3f0d4060ad8d3519e4c800b166e9e3bef5"
    assert frozen["source"]["immutable_commit"] == runner.R1_FAIL_CLOSED_SHA
    assert frozen["source"]["projection_fields"] == [
        "scenario", "phase", "second", "time_s", "channel", "prn", "method", "budget"
    ]
    assert frozen["source"]["filter"] == {"method": "pg_scc_k9", "budget": 9}
    assert frozen["eligible_event_count"] == 260
    assert len(frozen["event_records"]) == 260


def test_remediation_preflight_provenance_is_honest_about_container_projection():
    report = json.loads(
        (ROOT / "artifacts/pg_scc_stage0_r2_root_cause_audit/support_preflight.json").read_text()
    )
    assert report["protected_score_fields_projected_or_read"] == 0
    opened = report["score_bearing_containers_opened_for_metadata_projection"]
    assert opened == [
        {
            "path": "artifacts/pg_scc_stage0_static_k9/per_epoch_scores.csv",
            "sha256": "d3fd253807aea873dadce767fc027d3f0d4060ad8d3519e4c800b166e9e3bef5",
            "projection_fields": [
                "scenario", "phase", "second", "time_s", "channel", "prn", "method", "budget"
            ],
            "excluded_fields": ["score"],
        }
    ]


def test_remediation_all_eight_r1_binding_hashes_and_disallowed_paths_verified():
    runner = load_runner()
    report = runner.verify_r1_identity()
    assert report["status"] == "PASS"
    assert report["binding_hashes_verified"] == 8
    assert len(report["preserved_file_sha256"]) == 8
    assert report["disallowed_paths_unchanged"] is True
    assert report["unchanged_code_identity"]["status"] == "PASS"
    assert {"normalization", "score", "selector", "covariance", "synthetic", "mask", "threshold"} <= set(
        report["unchanged_code_identity"]["groups"]
    )


def test_remediation_independent_verifier_reconstructs_detector_hashes(tmp_path):
    verifier = load_verifier()
    source = ROOT / "artifacts/pg_scc_stage0_r2_root_cause_audit"
    for name in ("config.json", "source_commit.json", "r1_failure_binding.json", "support_preflight.json"):
        shutil.copy2(source / name, tmp_path / name)
    report = json.loads((tmp_path / "support_preflight.json").read_text())
    forged = "0" * 64
    report["per_detector_raw_row_support_hashes"] = {
        method: forged for method in report["per_detector_raw_row_support_hashes"]
    }
    (tmp_path / "support_preflight.json").write_text(json.dumps(report), encoding="utf-8")
    verification = verifier.verify_preexecution(tmp_path)
    assert verification["status"] == "FAIL"
    assert "reconstructed_support_mismatch:per_detector_raw_row_support_hashes" in verification["errors"]
