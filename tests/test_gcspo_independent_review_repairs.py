"""Regressions required by the post-independent-review freeze repair.

Fixtures are synthetic or consume committed metadata/scientific clean evidence.
No protected attack payload is opened by this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "artifacts/gcspo_stage0_static_rerun"


def _identity(path: Path):
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data)}


def test_committed_preaccess_capabilities_validate_before_claim_and_keep_unavailable_optional():
    from gnss_doppler_lab.gcspo_capabilities import validate_preaccess_capabilities

    result = validate_preaccess_capabilities(
        json.loads((ARTIFACT / "protected_capabilities.json").read_text())
    )
    assert set(result["available"]) == {"DS3", "DS7"}
    assert result["unavailable"]["DS4"]["status"] in {"LIMITED", "LIMITED_TRANSITION_ONLY", "UNAVAILABLE"}
    assert result["unavailable"]["DS8"] == {
        "status": "UNAVAILABLE", "reason": "DS8_MISSING_AUTHENTICATED_OBSERVABLES"
    }
    runner = (ROOT / "scripts/run_gcspo_stage0.py").read_text()
    assert runner.index("validate_preaccess_capabilities") < runner.index("claim_protected_attempt")

    tampered = json.loads((ARTIFACT / "protected_capabilities.json").read_text())
    tampered["scenarios"]["DS3"]["sidecar"]["root_manifest"].pop("size_bytes")
    with pytest.raises(ValueError, match="DS3|manifest|size"):
        validate_preaccess_capabilities(tampered)


def test_committed_clean_contrast_count_is_bound_preaccess_and_tamper_fails(tmp_path):
    from gnss_doppler_lab.gcspo_evaluate import load_clean_contrast_rows

    required = ("clean_only_report.json", "clean_ablation_report.json",
                "clean_a5_report.json", "clean_reproduction_evidence.json")
    identities = [_identity(ARTIFACT / name) for name in required]
    rows, source = load_clean_contrast_rows(ARTIFACT, identities)
    assert len(rows) == 4 * 237
    assert source["clean_reproduction_evidence.json"]["expected_holdout_windows"] == 237

    for name in required:
        (tmp_path / name).write_bytes((ARTIFACT / name).read_bytes())
    bad = json.loads((tmp_path / "clean_reproduction_evidence.json").read_text())
    bad["counts"]["clean_contrast_holdout_windows"] = 238
    (tmp_path / "clean_reproduction_evidence.json").write_text(json.dumps(bad) + "\n")
    bad_identities = [_identity(tmp_path / name) for name in required]
    with pytest.raises(ValueError, match="count"):
        load_clean_contrast_rows(tmp_path, bad_identities)

    runner = (ROOT / "scripts/run_gcspo_stage0.py").read_text()
    assert runner.index("validate_clean_contrast_preaccess") < runner.index("claim_protected_attempt")


def test_protected_b0_is_integrated_on_exact_full_epoch_prn_support(monkeypatch):
    import gnss_doppler_lab.gcspo_evaluate as evaluate

    support = tuple((epoch, (1, 2, 3, 4)) for epoch in range(50))
    full = {"window_start_s": 1.0, "availability_s": 2.0, "score": 9.0,
            "prns": [1, 2, 3, 4], "epoch_ids": tuple(range(50)),
            "epoch_prn_support": support}
    b0 = [{"window_start_s": 1.0, "availability_s": 2.0, "prn": prn,
           "event_score": float(prn), "epoch_ids": tuple(range(50)),
           "epoch_prn_support": tuple((epoch, (prn,)) for epoch in range(50))}
          for prn in (1, 2, 3, 4)]
    methods = {name: [] for name in ("A1", "A2", "A3", "A4", "A5", "Full")}
    methods["Full"] = [full]
    result = evaluate.integrate_protected_b0(methods, b0, score_column="event_score")
    assert set(result) == {"A0", "A1", "A2", "A3", "A4", "A5", "Full"}
    assert result["A0"] == [{"window_start_s": 1.0, "availability_s": 2.0,
                              "score": 2.5, "prns": [1, 2, 3, 4],
                              "epoch_ids": tuple(range(50)), "epoch_prn_support": support}]
    source = (ROOT / "src/gnss_doppler_lab/gcspo_evaluate.py").read_text()
    assert "score_protected_b0" in source and 'threshold_key = "A0_B0"' in source


def test_final_packaging_keeps_authenticated_clean_inputs_and_relation_reconstruction_matches(tmp_path, monkeypatch):
    from gnss_doppler_lab.gcspo_artifacts import prepare_valid_artifact_manifest
    from gnss_doppler_lab.gcspo_verify import FINAL_REQUIRED
    from gnss_doppler_lab.gcspo_verify_artifacts import reconstruct_relation_evidence

    reports = {"verifier_report.json", "fresh_clone_verifier_report.json"}
    for name in FINAL_REQUIRED - reports - {"artifact_manifest_sha256.json"}:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("contract artifact\n")
    for name in ("clean_only_report.json", "clean_ablation_report.json", "clean_a5_report.json",
                 "clean_reproduction_evidence.json", "reproduction_run_1.json", "reproduction_run_2.json"):
        (tmp_path / name).write_text("{}\n")
    (tmp_path / "plots").mkdir(); (tmp_path / "plots/numeric.csv").write_text("x\n1\n")
    (tmp_path / "implementation_manifest.json").write_text(json.dumps({
        "clean_scientific_artifacts": [_identity(tmp_path / name) for name in
            ("clean_only_report.json", "clean_ablation_report.json", "clean_a5_report.json",
             "clean_reproduction_evidence.json", "reproduction_run_1.json", "reproduction_run_2.json")]
    }) + "\n")
    common_access = {"actor": "gnss_doppler_lab.gcspo.AccessGate", "canonical_path": "/sealed/ds3.mat",
                     "scenario": "DS3", "phase": "transition", "purpose": "synthetic package",
                     "authorization_sha": "a" * 40, "run_identity": "a" * 40, "access_counter": 1,
                     "expected_sha256": "b" * 64, "expected_size": 7, "byte_range": "[0,7)",
                     "row_range": "ALL_ROWS_IN_BYTE_RANGE", "operation": "READ_HDF5", "kind": "MAT",
                     "identity_source": "AUTHENTICATED_MANIFEST:/sealed/manifest.json"}
    previous = "0" * 64; access = []
    for sequence, payload in enumerate((
        {**common_access, "record_type": "PRE", "outcome": "OPEN_PENDING",
         "timestamp_utc": "2026-08-12T01:02:03.000001Z"},
        {**common_access, "record_type": "POST", "outcome": "SUCCESS",
         "timestamp_utc": "2026-08-12T01:02:03.000002Z", "observed_sha256": "b" * 64,
         "observed_size": 7}), 1):
        row = {**payload, "sequence": sequence, "previous_record_sha256": previous}
        row["record_sha256"] = hashlib.sha256(json.dumps(
            row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        access.append(row); previous = row["record_sha256"]
    (tmp_path / "access_ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in access))
    synthetic_evidence = {"synthetic_package": True}
    (tmp_path / "final_verdict.json").write_text(json.dumps({
        "verdict": "NO_GO_PHYSICAL_HYPOTHESIS", "protected_run_count": 1,
        "evidence": synthetic_evidence}) + "\n")
    manifest = prepare_valid_artifact_manifest(tmp_path)
    paths = {row["path"] for row in manifest["files"]}
    assert {"clean_only_report.json", "clean_ablation_report.json", "clean_a5_report.json",
            "clean_reproduction_evidence.json"} <= paths

    common = {"scenario": "DS3", "phase": "transition", "label": True,
              "window_start_s": 120.0, "availability_s": 121.0,
              "phase_start_s": 118.9, "phase_end_s": 195.0,
              "prns": [1, 2, 3, 4], "epoch_ids": list(range(50)),
              "epoch_prn_support": [[epoch, [1, 2, 3, 4]] for epoch in range(50)]}
    rows = [{**common, "method": "Full", "score": 10.0},
            {**common, "method": "PER_PRN_TEMPORAL_SHIFT", "score": 5.0}]
    reconstructed = reconstruct_relation_evidence({"rows": rows}, scenarios=("DS3",))
    assert reconstructed["scenario_results"]["DS3"]["contrast"] == "PAIRED_SCORE_LOSS_NOT_BINARY_PAUC"

    import gnss_doppler_lab.gcspo_verify as verify
    monkeypatch.setattr(verify, "reconstruct_final_evidence", lambda root: synthetic_evidence)
    monkeypatch.setattr(verify, "verify_evidence_document", lambda document: {"verdict": document["verdict"]})
    monkeypatch.setattr(verify, "verify_reproduction_manifests", lambda root: {"status": "PASS"})
    assert verify.verify_final(tmp_path, strict=False)["status"] == "PASS"


def test_a5_reacquisition_forms_independent_segments_and_preserves_actual_support(monkeypatch):
    import gnss_doppler_lab.gcspo_a5 as a5

    epochs, prns, z = [], [], []
    for epoch in range(50):
        present = [1, 2, 3, 4] + ([5] if epoch != 10 else [])
        for prn in present:
            epochs.append(epoch); prns.append(prn); z.append(np.full(10, epoch + prn / 10))
    monkeypatch.setattr(a5, "residual_table", lambda *args: (
        np.asarray(epochs), np.asarray(prns), np.asarray(z), np.asarray(z)))
    monkeypatch.setattr(a5, "window_endpoints", lambda *_: np.asarray([1.0]))
    model = type("Model", (), {"coefficients": np.zeros((1, 10, 10))})()
    whitener = type("Whitener", (), {"inverse_sqrt": np.eye(10)})()
    rows = a5.role_a5_terms(object(), model, whitener, np.zeros((10, 10)),
                             {"code_error_chips"}, 0.0, 1.0)
    assert len(rows) == 1 and rows[0]["prns"] == [1, 2, 3, 4, 5]
    assert rows[0]["epoch_prn_support"][10] == (10, (1, 2, 3, 4))
    prn5 = [segment for segment in rows[0]["state_segments"] if segment["prn"] == 5]
    assert [segment["epoch_ids"] for segment in prn5] == [tuple(range(10)), tuple(range(11, 50))]
    prior = a5.a5_segment_prior_precision(rows[0]["state_segments"], smoothness=2.0)
    assert prior.shape == rows[0]["terms"][0].shape
    assert prior[prn5[0]["state_stop"] - 1, prn5[1]["state_start"]] == 0


def test_committed_reproduction_manifests_are_self_authenticating_and_verifier_consumed(tmp_path):
    from gnss_doppler_lab.gcspo_verify import verify_reproduction_manifests

    result = verify_reproduction_manifests(ARTIFACT)
    assert result["status"] == "PASS" and result["run_count"] == 2
    assert result["comparison"] == "BYTE_IDENTICAL_AFTER_EXPLICIT_TIMESTAMP_CANONICALIZATION"
    verifier = (ROOT / "src/gnss_doppler_lab/gcspo_verify.py").read_text()
    assert "verify_reproduction_manifests(artifact)" in verifier

    for name in ("reproduction_run_1.json", "reproduction_run_2.json",
                 "clean_reproduction_evidence.json", "clean_a5_report.json", "clean_b0_report.json",
                 "thresholds.json"):
        (tmp_path / name).write_bytes((ARTIFACT / name).read_bytes())
    bad = json.loads((tmp_path / "reproduction_run_2.json").read_text())
    bad["scientific_files"]["clean_a5_report.json"]["sha256"] = "0" * 64
    (tmp_path / "reproduction_run_2.json").write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="reproduction|scientific|identity"):
        verify_reproduction_manifests(tmp_path)


def test_repair_freeze_candidate_targets_immutable_parent_and_cannot_authorize_access(tmp_path):
    from gnss_doppler_lab.gcspo_freeze import (build_review_candidate_record,
                                               verify_review_candidate_record,
                                               verify_freeze_record)

    implementation = tmp_path / "implementation.py"; implementation.write_text("repaired = True\n")
    clean = tmp_path / "clean.json"; clean.write_text("{}\n")
    target = "b" * 40
    record = build_review_candidate_record(
        target_commit=target, config_sha256="c" * 64,
        implementation_files=[implementation], clean_files=[clean],
        rejected_freeze_commit="a" * 40)
    assert verify_review_candidate_record(record, target_commit=target)
    assert record["manifest_excludes_self"] is True
    assert record["protected_access_authorized"] is False
    with pytest.raises(ValueError, match="schema|state"):
        verify_freeze_record(record, target_commit=target)
