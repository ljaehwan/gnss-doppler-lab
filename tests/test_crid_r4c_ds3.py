from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab import crid_r4c_ds3 as r4c


def verifier_module():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_crid_r4c_ds3.py"
    spec = importlib.util.spec_from_file_location("verify_crid_r4c_ds3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def row(time_s: float, score: float, alarm: int, tracked: float = 8.0, lock: float = 0.95):
    partition = r4c.partition_for_time(time_s)
    return {
        "dataset": "TEXBAT.DS3",
        "sample": int(round(time_s * r4c.FS)),
        "time_s": time_s,
        "partition": partition,
        "score": score,
        "alarm": alarm,
        "label": 0 if partition == "pre_onset" else 1,
        "prn_count": 6,
        "config_count": 4,
        "cn0_median_db_hz": 40.0 + 0.01 * time_s,
        "lock_median": lock,
        "tracked_prn_count_min_config": tracked,
        "tracked_prn_count_median_config": tracked,
        "h0_loglike": -20.0,
        "h1_loglike": -10.0,
        "h1_improvement": 10.0,
        "penalty": 5.0,
        "configuration_disagreement": 0.2,
    }


def test_frozen_timeline_and_source_items():
    assert r4c.START_S == 78.9
    assert r4c.ONSET_S == 118.9
    assert r4c.PULL_OFF_S == 195.0
    assert r4c.END_S == 238.9
    assert r4c.END_S - r4c.START_S == pytest.approx(160.0)
    assert r4c.START_SAMPLE == int(r4c.START_S * r4c.FS)
    assert r4c.END_SAMPLE == int(r4c.END_S * r4c.FS)
    assert r4c.SOURCE_ITEMS == int(r4c.DURATION_S * r4c.FS * 2)


def test_partition_boundaries_are_half_open():
    assert r4c.partition_for_time(78.9) == "pre_onset"
    assert r4c.partition_for_time(118.899999) == "pre_onset"
    assert r4c.partition_for_time(118.9) == "transition"
    assert r4c.partition_for_time(195.0) == "established"
    assert r4c.partition_for_time(238.9) == "outside"


def test_threshold_comparison_is_strict():
    alarms = r4c.strict_alarms([r4c.THRESHOLD - 1e-9, r4c.THRESHOLD, r4c.THRESHOLD + 1e-9])
    assert alarms.tolist() == [False, False, True]


def test_persistent_alarm_bins_are_frozen_nonoverlapping():
    rows = [row(195.1, -20, 1), row(195.8, -20, 1), row(196.1, -22, 0), row(196.8, -22, 0)]
    result = r4c.persistent_alarm_metrics(rows)
    assert result["1s"]["eligible_complete_bins"] == 2
    assert result["1s"]["passing_bins"] == 1
    assert result["1s"]["persistent_alarm_ratio"] == 0.5
    assert result["5s"]["eligible_complete_bins"] == 1
    assert result["5s"]["passing_bins"] == 0


def test_metrics_and_signal_gate_pass_for_clear_synthetic_signal():
    rows = []
    for index in range(100):
        rows.append(row(100.0 + index * 0.01, -30.0 + index * 1e-6, 0))
        rows.append(row(130.0 + index * 0.01, -10.0 + index * 1e-6, 1))
        rows.append(row(200.0 + index * 0.01, -9.0 + index * 1e-6, 1))
    rows.sort(key=lambda value: value["time_s"])
    metrics = r4c.scenario_metrics(rows)
    shortcut = r4c.shortcut_audit(rows)
    gate = r4c.evaluate_signal_gate(metrics, shortcut, True)
    assert metrics["pre_onset_fpr"] == 0.0
    assert metrics["established_detection_rate"] == 1.0
    assert metrics["pauc_0_05"] == 1.0
    assert shortcut["explained_only_by_lock_loss_or_tracked_prn_collapse"] is False
    assert gate["status"] == "PASS"


def test_signal_gate_fails_preonset_false_alarms():
    rows = [row(100 + i * 0.01, -10, 1) for i in range(20)]
    rows += [row(130 + i * 0.01, -9, 1) for i in range(20)]
    rows += [row(200 + i * 0.01, -8, 1) for i in range(20)]
    rows.sort(key=lambda value: value["time_s"])
    metrics = r4c.scenario_metrics(rows)
    shortcut = r4c.shortcut_audit(rows)
    assert r4c.evaluate_signal_gate(metrics, shortcut, True)["status"] == "FAIL"


def test_shortcut_audit_detects_alarm_only_during_prn_collapse():
    rows = [row(100 + i * 0.01, -30, 0, tracked=10) for i in range(20)]
    rows += [row(130 + i * 0.01, -30, 0, tracked=10) for i in range(20)]
    rows += [row(200 + i * 0.01, -10, 1, tracked=4) for i in range(20)]
    audit = r4c.shortcut_audit(rows)
    assert audit["explained_only_by_lock_loss_or_tracked_prn_collapse"] is True
    assert audit["status"] == "FAIL"


def test_allowlist_rejects_other_attack_path_without_access():
    r4c.assert_allowed_attack_path(r4c.DS3_RAW)
    with pytest.raises(r4c.BindingError):
        r4c.assert_allowed_attack_path(Path("/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds7.bin"))


def test_model_hash_is_deterministic():
    arrays = {config: np.eye(2) for config in r4c.CONFIG_ORDER}
    model = r4c.NormalModel(
        order=1,
        ridge=1e-3,
        shrinkage=0.2,
        coefficients=arrays,
        means={config: np.ones(2) for config in r4c.CONFIG_ORDER},
        whiteners=arrays,
        h_matrices={config: np.ones((2, 1)) for config in r4c.CONFIG_ORDER},
        latent_dimension=1,
    )
    first = r4c._model_sha256(model, {config: 0 for config in r4c.CONFIG_ORDER})
    second = r4c._model_sha256(model, {config: 0 for config in r4c.CONFIG_ORDER})
    assert first == second and len(first) == 64


def test_completed_replay_tamper_is_rejected(tmp_path: Path):
    cfg = tmp_path / "receiver.conf"
    cfg.write_text("frozen\n")
    dump = tmp_path / "trace.bin"
    dump.write_bytes(b"trace")
    cfg_sha, dump_sha = r4c.sha256_file(cfg), r4c.sha256_file(dump)
    source = {
        "receiver": {"sha256": "a" * 64},
        "frozen_configs": {"C0": {"sha256": cfg_sha}},
        "reanchored_handoff": {"sha256": "b" * 64},
    }
    inventory = {"sha256": "c" * 64}
    dumps = [{"path": str(dump), "size_bytes": dump.stat().st_size, "sha256": dump_sha}]
    manifest = {
        "status": "PASS",
        "config": "C0",
        "input": {"sha256": "c" * 64},
        "receiver": {"sha256": "a" * 64},
        "config_file": {"path": str(cfg), "size_bytes": cfg.stat().st_size, "sha256": cfg_sha},
        "handoff": {"sha256": "b" * 64},
        "exit_code": 0,
        "termination": {"status": "PASS"},
        "native_trace_validation": {"status": "PASS"},
        "target_tracking_pass": True,
        "dumps": dumps,
        "output_set_sha256": r4c._output_set_sha(dumps),
    }
    path = tmp_path / "manifest.json"
    r4c.dump_json(path, manifest)
    r4c.validate_completed_replay(path, "C0", inventory, source)
    dump.write_bytes(b"tamper")
    with pytest.raises(r4c.BindingError):
        r4c.validate_completed_replay(path, "C0", inventory, source)


def test_manifest_tamper_is_rejected(tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    payload = artifact / "payload.json"
    payload.write_text("{}\n")
    r4c.seal_manifest(artifact)
    module = verifier_module()
    assert module.verify_manifest(artifact)[0] is True
    payload.write_text("tampered\n")
    assert module.verify_manifest(artifact)[0] is False


def test_no_threshold_recomputation_command_or_prohibited_claims():
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts/run_crid_r4c_ds3.py").read_text()
    source = (root / "src/gnss_doppler_lab/crid_r4c_ds3.py").read_text()
    assert "recompute-threshold" not in runner
    assert "empirical_threshold(" not in source
    for claim in ("PHASE_B_PASS", "SPOOFING_DETECTOR_VALIDATED", "READY_FOR_DEPLOYMENT"):
        assert claim not in source
