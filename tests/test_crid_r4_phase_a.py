import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.crid_r4_phase_a import (
    BindingError,
    _aggregate_output_sha,
    _load_checkpoint,
    evaluate_primary_gate,
    require_file_binding,
    sha256_file,
    support_masks,
    validate_completed_replay,
)


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts/verify_crid_r4_phase_a.py"
SPEC = importlib.util.spec_from_file_location("verify_crid_r4", VERIFIER_PATH)
VERIFIER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFIER)


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
            rows.append(
                {
                    "domain": domain,
                    "case_id": f"{domain}.negative.{index}",
                    "family": "negative",
                    "mode": "",
                    "case_gate_status": "PASS",
                }
            )
        for mode in ("single", "four"):
            for index in range(9):
                rows.append(
                    {
                        "domain": domain,
                        "case_id": f"{domain}.positive.{mode}.{index}",
                        "family": "positive",
                        "mode": mode,
                        "case_gate_status": "PASS" if index == 0 else "FAIL",
                    }
                )
    return rows


def test_primary_gate_requires_all_negatives_and_each_positive_group():
    clean = {"OAK": {"holdout_fpr_q99": 0.01}, "TEX": {"holdout_fpr_q99": 0.02}}
    metrics = _passing_metrics()
    result = evaluate_primary_gate(metrics, clean, technical_ok=True)
    assert result["status"] == "PASS"
    assert result["positive_pass_count"] == 4
    assert result["legacy_any_positive_pass"] is True
    assert result["legacy_any_positive_pass_use"] == "DIAGNOSTIC_ONLY_NOT_AUTHORIZATION"

    metrics[0]["case_gate_status"] = "FAIL"
    assert evaluate_primary_gate(metrics, clean, technical_ok=True)["status"] == "FAIL"
    metrics[0]["case_gate_status"] = "PASS"
    metrics[-9]["case_gate_status"] = "FAIL"
    assert evaluate_primary_gate(metrics, clean, technical_ok=True)["status"] == "FAIL"
    assert evaluate_primary_gate(_passing_metrics(), clean, technical_ok=False)["status"] == "FAIL"


def test_checkpoint_resume_requires_exact_freeze_and_inventory(tmp_path):
    checkpoint = _load_checkpoint(tmp_path, "freeze", "inventory")
    checkpoint["completed"]["case|C0"] = {"status": "PASS"}
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(checkpoint))
    loaded = _load_checkpoint(tmp_path, "freeze", "inventory")
    assert "case|C0" in loaded["completed"]
    with pytest.raises(BindingError, match="checkpoint binding mismatch"):
        _load_checkpoint(tmp_path, "other", "inventory")


def test_file_binding_detects_input_truth_config_and_receiver_tamper(tmp_path):
    for name in ("input.bin", "truth.json", "receiver", "receiver.conf"):
        path = tmp_path / name
        path.write_bytes(b"bound")
        expected = sha256_file(path)
        require_file_binding(path, path.stat().st_size, expected)
        path.write_bytes(b"tampered")
        with pytest.raises(BindingError):
            require_file_binding(path, len(b"bound"), expected)


def _synthetic_completed_replay(tmp_path):
    out = tmp_path / "replay"
    out.mkdir()
    dump = out / "trace_native_1ms_ch_0.bin"
    dump.write_bytes(b"trace")
    config_path = out / "receiver.conf"
    config_path.write_text("config\n")
    dumps = [{"path": str(dump), "size": dump.stat().st_size, "sha256": sha256_file(dump)}]
    manifest = {
        "status": "PASS",
        "domain": "OAK",
        "case_id": "OAK.synthetic",
        "config": "C0",
        "input": {"sha256": "a" * 64},
        "receiver": {"sha256": "b" * 64},
        "termination": {"status": "PASS"},
        "native_trace_validation": {"status": "PASS"},
        "target_tracking_pass": True,
        "config_file": {
            "path": str(config_path),
            "size_bytes": config_path.stat().st_size,
            "sha256": sha256_file(config_path),
        },
        "dumps": dumps,
        "output_set_sha256": _aggregate_output_sha(dumps),
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest))
    row = {"domain": "OAK", "case_id": "OAK.synthetic", "package_sha256": "a" * 64}
    return path, row, dump, config_path


def test_completed_replay_detects_output_config_receiver_and_input_hash_tamper(tmp_path):
    manifest_path, row, dump, config_path = _synthetic_completed_replay(tmp_path)
    validate_completed_replay(manifest_path, row, "C0", "b" * 64)

    dump.write_bytes(b"changed")
    with pytest.raises(BindingError, match="output hash mismatch"):
        validate_completed_replay(manifest_path, row, "C0", "b" * 64)
    dump.write_bytes(b"trace")
    config_path.write_text("changed\n")
    with pytest.raises(BindingError, match="file size binding mismatch"):
        validate_completed_replay(manifest_path, row, "C0", "b" * 64)
    config_path.write_text("config\n")
    with pytest.raises(BindingError, match="manifest contract mismatch"):
        validate_completed_replay(manifest_path, row, "C0", "c" * 64)
    changed_row = dict(row, package_sha256="d" * 64)
    with pytest.raises(BindingError, match="manifest contract mismatch"):
        validate_completed_replay(manifest_path, changed_row, "C0", "b" * 64)


def test_manifest_payload_and_manifest_hash_tamper_fail_closed(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    payload = artifact / "payload.txt"
    payload.write_text("bound\n")
    manifest = {
        "schema": "gnss-doppler-lab.crid-r4-artifact-manifest.v1",
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


def test_incomplete_output_directory_fails_before_receiver_execution(tmp_path):
    from gnss_doppler_lab.crid_r4_phase_a import run_one_replay

    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(BindingError, match="incomplete or uncheckpointed"):
        run_one_replay({}, "C0", out, "f" * 64)
