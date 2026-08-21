import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.crid_r4b_phase_a import (
    AUTHORITATIVE_THRESHOLDS,
    BindingError,
    _aggregate_output_sha,
    _load_checkpoint,
    _run_one_r4b,
    _score_hash,
    authoritative_alarms,
    evaluate_primary_gate,
    require_file_binding,
    sha256_file,
    support_masks,
    validate_authorization_documents,
    validate_completed_replay,
)


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts/verify_crid_r4b_phase_a.py"
SPEC = importlib.util.spec_from_file_location("verify_crid_r4b", VERIFIER_PATH)
VERIFIER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFIER)


def _authorization_docs():
    final = {
        "status": "PASS",
        "verdict": "THRESHOLD_DECISION_EQUIVALENCE_REPAIR_PASS",
        "next_state": "READY_TO_REPEAT_CRID_PHASE_A",
        "phase_a_executed": False,
        "phase_b_executed": False,
        "attack_bytes_read": 0,
        "authoritative_thresholds": dict(AUTHORITATIVE_THRESHOLDS),
    }
    numeric = {
        "status": "PASS",
        "authoritative_threshold_policy": "COMMITTED_R2_LITERALS_ONLY",
        "domains": {
            domain: {
                "status": "PASS",
                "authoritative_threshold_retained": threshold,
                "committed_q99": threshold,
            }
            for domain, threshold in AUTHORITATIVE_THRESHOLDS.items()
        },
    }
    holdout = {
        "status": "PASS",
        "comparison": "score > threshold",
        "domains": {
            domain: {
                "status": "PASS",
                "alarm_vectors_byte_identical": True,
                "false_positive_count_and_fpr_equal": True,
                "all_scored_epochs_finite_four_config_min_four_prn": True,
                "committed_fpr": 0.01,
                "holdout_score_count": 100,
                "committed_alarm_sha256": "a" * 64,
                "causal_delays_ms": {"C0": 0, "C1": 0, "C2": 0, "C3": 0},
            }
            for domain in AUTHORITATIVE_THRESHOLDS
        },
    }
    return final, numeric, holdout


def test_r4a_decision_equivalence_authorizes_literals_and_fails_tamper():
    final, numeric, holdout = _authorization_docs()
    result = validate_authorization_documents(final, numeric, holdout)
    assert result["status"] == "PASS"
    assert result["threshold_recomputation_executed"] is False
    assert result["domains"]["OAK"]["authoritative_threshold"] == -21.705587048010322
    numeric["domains"]["OAK"]["authoritative_threshold_retained"] = -1.0
    with pytest.raises(BindingError, match="authorization mismatch"):
        validate_authorization_documents(final, numeric, holdout)


def test_authoritative_comparison_is_strictly_greater():
    threshold = AUTHORITATIVE_THRESHOLDS["TEX"]
    values = np.array([np.nextafter(threshold, -np.inf), threshold, np.nextafter(threshold, np.inf)])
    assert authoritative_alarms(values, threshold).tolist() == [False, False, True]


def test_truth_support_masks_use_raw_sample_intervals():
    samples = np.array([99, 100, 109, 110, 119, 120, 130], dtype=np.int64)
    full, active = support_masks(
        samples,
        replacement_start=100,
        replacement_end=130,
        active_samples=np.array([110, 120], dtype=np.int64),
        cadence_samples=10,
    )
    assert full.tolist() == [False, True, True, True, True, True, False]
    assert active.tolist() == [False, False, False, True, True, True, False]


def _passing_metrics():
    rows = []
    for domain in ("OAK", "TEX"):
        for index in range(15):
            rows.append({"domain": domain, "case_id": f"{domain}.n.{index}", "family": "negative", "mode": "", "case_gate_status": "PASS"})
        for mode in ("single", "four"):
            for index in range(9):
                rows.append(
                    {
                        "domain": domain,
                        "case_id": f"{domain}.p.{mode}.{index}",
                        "family": "positive",
                        "mode": mode,
                        "case_gate_status": "PASS" if index == 0 else "FAIL",
                    }
                )
    return rows


def test_primary_gate_requires_all_negatives_and_all_positive_groups():
    clean = {"OAK": {"holdout_fpr_q99": 0.01}, "TEX": {"holdout_fpr_q99": 0.02}}
    metrics = _passing_metrics()
    assert evaluate_primary_gate(metrics, clean, True)["status"] == "PASS"
    metrics[0]["case_gate_status"] = "FAIL"
    assert evaluate_primary_gate(metrics, clean, True)["status"] == "FAIL"
    assert evaluate_primary_gate(_passing_metrics(), clean, False)["status"] == "FAIL"


def test_checkpoint_requires_exact_freeze_and_inventory(tmp_path):
    checkpoint = _load_checkpoint(tmp_path, "freeze", "inventory")
    checkpoint["completed"]["OAK|case|C0"] = {"status": "PASS"}
    (tmp_path / "checkpoint.json").write_text(json.dumps(checkpoint))
    assert "OAK|case|C0" in _load_checkpoint(tmp_path, "freeze", "inventory")["completed"]
    with pytest.raises(BindingError, match="checkpoint binding mismatch"):
        _load_checkpoint(tmp_path, "changed", "inventory")


def test_file_binding_detects_input_truth_config_and_receiver_tamper(tmp_path):
    for name in ("input.bin", "truth.json", "truth_epochs.bin", "receiver", "receiver.conf"):
        path = tmp_path / name
        path.write_bytes(b"bound")
        require_file_binding(path, path.stat().st_size, sha256_file(path))
        path.write_bytes(b"tampered")
        with pytest.raises(BindingError):
            require_file_binding(path, len(b"bound"), "0" * 64)


def _completed_replay(tmp_path):
    out = tmp_path / "replay"
    out.mkdir()
    dump = out / "trace_native_1ms_ch_0.bin"
    dump.write_bytes(b"trace")
    config_path = out / "receiver.conf"
    config_path.write_text("config\n")
    dumps = [{"path": str(dump), "size": dump.stat().st_size, "sha256": sha256_file(dump)}]
    manifest = {
        "schema": "gnss-doppler-lab.crid-r4b-replay.v1",
        "status": "PASS",
        "domain": "OAK",
        "case_id": "OAK.synthetic",
        "config": "C0",
        "input": {"sha256": "a" * 64},
        "receiver": {"sha256": "b" * 64},
        "termination": {"status": "PASS"},
        "native_trace_validation": {"status": "PASS"},
        "target_tracking_pass": True,
        "config_file": {"path": str(config_path), "size_bytes": config_path.stat().st_size, "sha256": sha256_file(config_path)},
        "dumps": dumps,
        "output_set_sha256": _aggregate_output_sha(dumps),
        "common_support": {"status": "PENDING_FOUR_CONFIG_ANALYSIS", "minimum_configurations": 4, "minimum_common_prns": 4},
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest))
    row = {"domain": "OAK", "case_id": "OAK.synthetic", "package_sha256": "a" * 64}
    return path, row, dump, config_path


def test_completed_replay_detects_output_config_receiver_input_and_manifest_tamper(tmp_path):
    manifest_path, row, dump, config_path = _completed_replay(tmp_path)
    validate_completed_replay(manifest_path, row, "C0", "b" * 64)
    dump.write_bytes(b"changed")
    with pytest.raises(BindingError, match="output hash mismatch"):
        validate_completed_replay(manifest_path, row, "C0", "b" * 64)
    dump.write_bytes(b"trace")
    config_path.write_text("changed\n")
    with pytest.raises(BindingError):
        validate_completed_replay(manifest_path, row, "C0", "b" * 64)
    config_path.write_text("config\n")
    changed = dict(row, package_sha256="d" * 64)
    with pytest.raises(BindingError, match="manifest contract mismatch"):
        validate_completed_replay(manifest_path, changed, "C0", "b" * 64)


def test_manifest_payload_and_manifest_hash_tamper_fail_closed(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    payload = artifact / "payload.txt"
    payload.write_text("bound\n")
    manifest = {
        "schema": "gnss-doppler-lab.crid-r4b-artifact-manifest.v1",
        "file_count": 1,
        "files": [{"path": "payload.txt", "size_bytes": payload.stat().st_size, "sha256": sha256_file(payload)}],
        "status": "PASS",
    }
    path = artifact / "artifact_manifest_sha256.json"
    path.write_text(json.dumps(manifest))
    assert VERIFIER.verify_manifest(artifact)[0] is True
    payload.write_text("tampered\n")
    assert VERIFIER.verify_manifest(artifact)[0] is False
    payload.write_text("bound\n")
    manifest["files"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    assert VERIFIER.verify_manifest(artifact)[0] is False


def test_incomplete_output_fails_before_receiver_and_score_hash_is_deterministic(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(BindingError, match="incomplete or uncheckpointed"):
        _run_one_r4b({}, "C0", out, "f" * 64)
    rows = [
        {
            "sample": 1,
            "prn_count": 4,
            "config_count": 4,
            "score": 1.0,
            "h0_loglike": 2.0,
            "h1_loglike": 3.0,
            "penalty": 4.0,
            "configuration_disagreement": 5.0,
        }
    ]
    assert _score_hash(rows) == _score_hash([dict(rows[0])])
    rows[0]["score"] = 2.0
    assert _score_hash(rows) != _score_hash([{**rows[0], "score": 1.0}])


def test_runner_has_no_threshold_recomputation_command():
    text = (ROOT / "scripts/run_crid_r4b_phase_a.py").read_text()
    assert "threshold-check" not in text
    assert "recompute_thresholds" not in text
