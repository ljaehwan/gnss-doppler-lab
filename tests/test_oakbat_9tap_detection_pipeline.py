import importlib.util
import json
import sys
import subprocess
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_oakbat_9tap_detection_pipeline.py"
    spec = importlib.util.spec_from_file_location("oakbat_pipeline_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_minimal_covered_receiver(mod, manifest):
    manifest.parent.mkdir(parents=True, exist_ok=True)
    (manifest.parent / "tracking.csv").write_text("time_s,prn\n0,G01\n479.5,G01\n")
    (manifest.parent / "tracking_summary.csv").write_text("end_time_s,prn,epoch_count\n479.5,G01,2\n")
    manifest.write_text(json.dumps({"status":"complete","tracking":{"coverage":mod.receiver_tracking_coverage(manifest.parent)}}))
    return manifest


def test_receiver_config_is_exact_oakbat_5mhz_method_a_contract(tmp_path):
    mod = _load_module()
    text = mod.receiver_config(tmp_path / "os1.bin", tmp_path / "run")
    assert "GNSS-SDR.internal_fs_sps=5000000" in text
    assert "SignalSource.item_type=ishort" in text
    assert "SignalSource.samples=0" in text
    assert "SignalSource.samples=4800000000" not in text
    assert "SignalSource.repeat=false" in text
    assert mod.EXPECTED_COMPLEX_SAMPLES == 2_400_000_000
    assert mod.AVAILABLE_SCALAR_INT16_ITEMS == 4_800_000_000
    assert mod.CONFIGURED_SIGNAL_SOURCE_SAMPLES == 0
    assert "DataTypeAdapter.implementation=Ishort_To_Complex" in text
    assert "Tracking_1C.tap_count=9" in text
    assert "Tracking_1C.tap_spacing_chips=0.125" in text


def test_iq_validation_requires_exact_480_second_interleaved_int16_size(tmp_path):
    mod = _load_module()
    iq = tmp_path / "os1.bin"
    iq.write_bytes(b"x" * 32)
    with pytest.raises(ValueError, match="exact.*size|size.*exact"):
        mod.validate_iq(iq)
    assert mod.EXPECTED_IQ_BYTES == 9_600_000_000
    assert mod.EXPECTED_COMPLEX_SAMPLES == 2_400_000_000


def test_scenarios_and_manifest_are_oakbat_frozen_evaluation_contract(tmp_path):
    mod = _load_module()
    assert set(mod.SCENARIOS) == {"os1", "os2", "os3", "os4", "cleanStatic", "cleanDynamic"}
    assert mod.CLEAN_SCENARIOS == {"cleanStatic", "cleanDynamic"}
    doc = mod.provenance_manifest(scenario="os1", iq=tmp_path / "os1.bin", iq_sha256="a" * 64, receiver_manifest=tmp_path / "receiver.json", node_csv=tmp_path / "nodes.csv", score_summary=tmp_path / "score.json", gate_summary=tmp_path / "gate.json")
    assert doc["source"]["dataset"] == "OAKBAT"
    assert doc["source"]["duration_s"] == 480.0
    assert doc["evaluation"] == {"onset_s": 120.0, "guard_s": 10.0}
    assert doc["adapter"]["feature_mode"] == "normalized_dmcpd"
    assert "trainer" not in json.dumps(doc).lower()



def _write_receiver_cache(mod, tmp_path, *, iq_bytes=b"iqiq", exe_bytes=b"exe"):
    iq = tmp_path / "os1.bin"; iq.write_bytes(iq_bytes)
    exe = tmp_path / "gnss-sdr"; exe.write_bytes(exe_bytes); exe.chmod(0o755)
    run_dir = tmp_path / "out" / "receiver" / "oakbat-os1-method-a-9tap"
    (run_dir / "raw").mkdir(parents=True)
    config = run_dir / "receiver.conf"; config.write_text(mod.receiver_config(iq, run_dir))
    (run_dir / "receiver.log").write_text("receiver log")
    (run_dir / "tracking.csv").write_text("time_s,prn\n0.0,G01\n479.5,G01\n")
    (run_dir / "tracking_summary.csv").write_text("end_time_s,prn,row_count\n479.5,G01,2\n")
    (run_dir / "raw" / "epl_tracking_ch_0.mat").write_bytes(b"mat")
    doc = mod.receiver_cache_contract("os1", iq, run_dir, exe)
    manifest = run_dir / "manifest.json"; manifest.write_text(json.dumps(doc))
    return iq, exe, manifest


@pytest.mark.parametrize("stale", ["iq_path", "iq_size", "iq_sha", "available_items", "configured_samples", "config_sha", "tap", "exe_sha", "output"])
def test_receiver_cache_fails_closed_on_stale_provenance_or_incomplete_outputs(tmp_path, monkeypatch, stale):
    mod = _load_module(); monkeypatch.setattr(mod, "EXPECTED_IQ_BYTES", 4)
    iq, exe, manifest = _write_receiver_cache(mod, tmp_path)
    doc = json.loads(manifest.read_text())
    if stale == "iq_path": doc["source"]["iq"] = str(tmp_path / "other.bin")
    elif stale == "iq_size": doc["source"]["iq_size_bytes"] = 3
    elif stale == "iq_sha": doc["source"]["iq_sha256"] = "0" * 64
    elif stale == "available_items": doc["source"]["available_scalar_int16_items"] = 0
    elif stale == "configured_samples": doc["source"]["configured_signal_source_samples"] = 4_800_000_000
    elif stale == "config_sha": doc["receiver"]["config_sha256"] = "0" * 64
    elif stale == "tap": doc["tracking"]["tap_count"] = 3
    elif stale == "exe_sha": doc["receiver"]["executable_sha256"] = "0" * 64
    else: (manifest.parent / "tracking.csv").unlink()
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="cache|cached|stale|output|contract"):
        mod.validate_cached_receiver(manifest, "os1", iq, exe)


def test_receiver_cache_rejects_explicit_full_boundary_sample_config(tmp_path):
    mod = _load_module()
    iq, exe, manifest = _write_receiver_cache(mod, tmp_path)
    boundary = mod.receiver_config(
        iq, manifest.parent, samples=mod.AVAILABLE_SCALAR_INT16_ITEMS
    )
    assert "SignalSource.samples=4800000000" in boundary
    (manifest.parent / "receiver.conf").write_text(boundary)
    with pytest.raises(ValueError, match="config contract"):
        mod.validate_cached_receiver(manifest, "os1", iq, exe)


def test_feature_cache_validates_receiver_relationship_schema_hash_and_finiteness(tmp_path):
    mod = _load_module()
    receiver = _write_minimal_covered_receiver(mod, tmp_path / "receiver" / "manifest.json")
    out = tmp_path / "scenario"; out.mkdir()
    features = out / "tap9_tracking_features_w1.0_s0.5.csv"
    features.write_text('run_id,prn,window_start_s,window_end_s,window_mid_s,tap_count,tap_layout,tap_E4_mean\nr,G01,0.0,1.0,0.5,9,"E4,E3,E2,E,P,L,L2,L3,L4",1.0\n')
    dataset = out / "multi_prn_method_a_9tap_w1.0_s0.5_normalized_dmcpd"; dataset.mkdir()
    node = dataset / "normal_prn_node_windows.csv"; cols = ["run_id", "prn", "window_bin_s", *mod.FROZEN_FEATURE_COLUMNS]
    node.write_text(",".join(cols) + "\n" + ",".join(["r", "G01", "0.5", *(["1.0"] * len(mod.FROZEN_FEATURE_COLUMNS))]) + "\n")
    mod.write_feature_cache_contract(out, receiver, features, node)
    assert mod.validate_cached_features(out, receiver) == node
    node.write_text(",".join(cols) + "\n" + ",".join(["r", "G01", "0.5", "nan", *(["1.0"] * (len(mod.FROZEN_FEATURE_COLUMNS) - 1))]) + "\n")
    with pytest.raises(ValueError, match="finite|hash|stale"):
        mod.validate_cached_features(out, receiver)


def test_default_scenarios_include_both_clean_negative_controls():
    mod = _load_module()
    args = mod.build_parser().parse_args([])
    assert args.scenarios == ["cleanStatic", "cleanDynamic", "os1", "os2", "os3", "os4"]


def test_top_manifest_contains_frozen_contract_and_artifact_hashes(tmp_path):
    mod = _load_module()
    paths = {}
    for name in ["iq", "receiver", "node", "score", "gate", "checkpoint", "calibration"]:
        paths[name] = tmp_path / name; paths[name].write_bytes((b"{}" if name == "calibration" else name.encode()))
    doc = mod.provenance_manifest(scenario="os1", iq=paths["iq"], iq_sha256=mod.sha256(paths["iq"]),
        receiver_manifest=paths["receiver"], node_csv=paths["node"], score_summary=paths["score"],
        gate_summary=paths["gate"], checkpoint=paths["checkpoint"], calibration_json=paths["calibration"])
    assert doc["frozen_detector"]["checkpoint_sha256"] == mod.sha256(paths["checkpoint"])
    assert doc["frozen_detector"]["calibration_sha256"] == mod.sha256(paths["calibration"])
    assert doc["adapter"]["feature_contract"]["tap_count"] == 9
    for artifact in doc["outputs"]["artifacts"].values():
        assert len(artifact["sha256"]) == 64



def test_feature_cache_rejects_hash_consistent_but_wrong_node_schema(tmp_path):
    mod = _load_module()
    receiver = _write_minimal_covered_receiver(mod, tmp_path / "receiver" / "manifest.json")
    out = tmp_path / "scenario"; out.mkdir()
    features = out / "features.csv"
    features.write_text('run_id,prn,window_start_s,window_end_s,window_mid_s,tap_count,tap_layout,tap_E4_mean\nr,G01,0.0,1.0,0.5,9,"E4,E3,E2,E,P,L,L2,L3,L4",1.0\n')
    node = out / "nodes.csv"
    cols = ["run_id", "prn", "window_bin_s", *mod.FROZEN_FEATURE_COLUMNS]
    node.write_text(",".join(cols) + "\n" + ",".join(["r", "G01", "0.5", *(["1.0"] * len(mod.FROZEN_FEATURE_COLUMNS))]) + "\n")
    contract = mod.write_feature_cache_contract(out, receiver, features, node)
    node.write_text("run_id,prn,window_bin_s,unrelated_feature\nr,G01,0.5,1.0\n")
    doc = json.loads(contract.read_text()); doc["node_table"] = mod._artifact(node); contract.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="schema|model features"):
        mod.validate_cached_features(out, receiver)


def test_receiver_streams_log_and_writes_failure_metadata_on_nonzero(tmp_path, monkeypatch):
    mod = _load_module(); monkeypatch.setattr(mod, 'EXPECTED_IQ_BYTES', 4)
    iq = tmp_path / 'os1.bin'; iq.write_bytes(b'iqiq')
    exe = tmp_path / 'fake_receiver.py'
    exe.write_text("""#!/usr/bin/env python3
import sys
if '--version' in sys.argv: print('fake 1'); raise SystemExit(0)
print('stdout marker', flush=True)
print('stderr marker', file=sys.stderr, flush=True)
raise SystemExit(7)
""")
    exe.chmod(0o755)
    with pytest.raises(RuntimeError, match='rc=7'):
        mod.run_receiver('os1', iq, tmp_path / 'out', exe=str(exe), timeout_s=5, force=True)
    run_dir = tmp_path / 'out/receiver/oakbat-os1-method-a-9tap'
    assert 'stdout marker' in (run_dir / 'receiver.log').read_text()
    assert 'stderr marker' in (run_dir / 'receiver.log').read_text()
    failure = json.loads((run_dir / 'receiver_failure.json').read_text())
    assert failure['status'] == 'failed' and failure['exit_code'] == 7
    assert failure['process_group_terminated'] is True


def test_receiver_rejects_iq_changed_during_execution(tmp_path, monkeypatch):
    mod = _load_module(); monkeypatch.setattr(mod, 'EXPECTED_IQ_BYTES', 4)
    iq = tmp_path / 'os1.bin'; iq.write_bytes(b'iqiq')
    exe = tmp_path / 'mutator.py'
    exe.write_text(f"""#!/usr/bin/env python3
import pathlib,sys
if '--version' in sys.argv: print('fake'); raise SystemExit(0)
pathlib.Path({str(iq)!r}).write_bytes(b'xxxx')
""")
    exe.chmod(0o755)
    with pytest.raises(RuntimeError, match='IQ.*changed|changed.*IQ'):
        mod.run_receiver('os1', iq, tmp_path / 'out', exe=str(exe), timeout_s=5, force=True)
    failure = json.loads((tmp_path / 'out/receiver/oakbat-os1-method-a-9tap/receiver_failure.json').read_text())
    assert failure['failure_kind'] == 'iq_identity_changed'
    assert failure['iq_identity_before']['sha256'] != failure['iq_identity_after']['sha256']


def test_output_space_preflight_enforces_configurable_floor_and_estimate(tmp_path, monkeypatch):
    mod = _load_module()
    usage = type('Usage', (), {'free': 5_000})()
    monkeypatch.setattr(mod.shutil, 'disk_usage', lambda _: usage)
    with pytest.raises(RuntimeError, match='disk space'):
        mod.preflight_output_space(tmp_path, scenario_count=2, minimum_free_bytes=4_000, estimated_bytes_per_scenario=3_000)
    result = mod.preflight_output_space(tmp_path, scenario_count=1, minimum_free_bytes=4_000, estimated_bytes_per_scenario=3_000)
    assert result['required_free_bytes'] == 4_000


def test_timing_contract_is_explicit_in_feature_and_top_provenance(tmp_path):
    mod = _load_module()
    assert mod.TIMING_CONTRACT['score_time_field'] == 'window_start_s'
    assert mod.TIMING_CONTRACT['window_availability_offset_s'] == 1.0
    doc = mod.provenance_manifest(scenario='os1', iq=tmp_path/'iq', iq_sha256='a'*64, receiver_manifest=tmp_path/'r', node_csv=tmp_path/'n', score_summary=tmp_path/'s', gate_summary=tmp_path/'g')
    assert doc['adapter']['timing_contract'] == mod.TIMING_CONTRACT



def test_receiver_cache_contract_uses_supplied_iq_identity_without_rehash(tmp_path, monkeypatch):
    mod = _load_module(); monkeypatch.setattr(mod, "EXPECTED_IQ_BYTES", 4)
    iq, exe, manifest = _write_receiver_cache(mod, tmp_path)
    verified = {"path": str(iq.resolve()), "size_bytes": 4, "sha256": "1" * 64}
    real_sha = mod.sha256
    def guarded_sha(path, *args, **kwargs):
        if Path(path) == iq:
            raise AssertionError("IQ was rehashed")
        return real_sha(path, *args, **kwargs)
    monkeypatch.setattr(mod, "sha256", guarded_sha)
    doc = mod.receiver_cache_contract("os1", iq, manifest.parent, exe, verified)
    assert doc["source"]["iq_size_bytes"] == 4
    assert doc["source"]["iq_sha256"] == "1" * 64


def _write_success_receiver(path, iq):
    path.write_text(f"""#!/usr/bin/env python3
import pathlib,sys
if '--version' in sys.argv: print('fake 1'); raise SystemExit(0)
pathlib.Path('raw/epl_tracking_ch_0.mat').write_bytes(b'mat')
""")
    path.chmod(0o755)


def test_receiver_rejects_iq_changed_during_tracking_export(tmp_path, monkeypatch):
    mod = _load_module(); monkeypatch.setattr(mod, "EXPECTED_IQ_BYTES", 4)
    iq = tmp_path / "os1.bin"; iq.write_bytes(b"iqiq")
    exe = tmp_path / "fake.py"; _write_success_receiver(exe, iq)
    def mutating_export(mats, output, summary, *, sample_rate_hz):
        Path(output).write_text("a\n1\n"); Path(summary).write_text("a\n1\n")
        iq.write_bytes(b"xxxx")
        return {"row_count":1,"prns":["G01"],"channel_count":1}
    monkeypatch.setattr(mod, "export_tracking_csv", mutating_export)
    with pytest.raises(RuntimeError, match="IQ.*changed|changed.*IQ"):
        mod.run_receiver("os1", iq, tmp_path / "out", exe=str(exe), timeout_s=5, force=True)
    failure = json.loads((tmp_path / "out/receiver/oakbat-os1-method-a-9tap/receiver_failure.json").read_text())
    assert failure["failure_kind"] == "iq_identity_changed_after_tracking_export"
    assert failure["iq_identity_before"]["sha256"] != failure["iq_identity_after"]["sha256"]


@pytest.mark.parametrize("mode",["timeout","nonzero"])
def test_version_probe_failures_write_metadata(tmp_path, monkeypatch, mode):
    mod = _load_module(); monkeypatch.setattr(mod, "EXPECTED_IQ_BYTES", 4)
    iq = tmp_path / "os1.bin"; iq.write_bytes(b"iqiq")
    exe = tmp_path / "fake"; exe.write_bytes(b"exe"); exe.chmod(0o755)
    if mode == "timeout":
        def fail_run(*args, **kwargs): raise subprocess.TimeoutExpired(args[0], 30)
    else:
        def fail_run(*args, **kwargs): return subprocess.CompletedProcess(args[0], 9, stdout="", stderr="bad version")
    monkeypatch.setattr(mod.subprocess, "run", fail_run)
    with pytest.raises(RuntimeError, match="version"):
        mod.run_receiver("os1", iq, tmp_path / "out", exe=str(exe), timeout_s=5, force=True)
    run_dir = tmp_path / "out/receiver/oakbat-os1-method-a-9tap"
    failure = json.loads((run_dir / "receiver_failure.json").read_text())
    assert failure["failure_kind"] == f"version_{mode}"
    assert failure["receiver_log"] == str((run_dir / "receiver.log").resolve())
    assert "--version" in failure["version_command"]


def test_tracking_export_exception_writes_failure_metadata(tmp_path, monkeypatch):
    mod = _load_module(); monkeypatch.setattr(mod, "EXPECTED_IQ_BYTES", 4)
    iq = tmp_path / "os1.bin"; iq.write_bytes(b"iqiq")
    exe = tmp_path / "fake.py"; _write_success_receiver(exe, iq)
    monkeypatch.setattr(mod, "export_tracking_csv", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad mat")))
    with pytest.raises(RuntimeError, match="tracking export"):
        mod.run_receiver("os1", iq, tmp_path / "out", exe=str(exe), timeout_s=5, force=True)
    run_dir = tmp_path / "out/receiver/oakbat-os1-method-a-9tap"
    failure = json.loads((run_dir / "receiver_failure.json").read_text())
    assert failure["failure_kind"] == "tracking_export_error"
    assert "bad mat" in failure["error"]
    assert failure["receiver_log"] == str((run_dir / "receiver.log").resolve())


def test_pipeline_path_defaults_are_rooted_and_explicit_paths_remain_accepted(tmp_path, monkeypatch):
    mod = _load_module()
    monkeypatch.chdir(tmp_path)
    defaults = mod.build_parser().parse_args([])
    assert Path(defaults.out_root) == mod.ROOT / "artifacts/oakbat_9tap_frozen_champion"
    assert Path(defaults.model_dir) == mod.ROOT / "artifacts/ai_morph_gru_cleanStatic_q70_frame"
    assert Path(defaults.calibration_json) == mod.ROOT / "configs/detectors/texbat_btail_gate_v1.json"
    explicit = mod.build_parser().parse_args(["--out-root", "rel/out", "--model-dir", "rel/model", "--calibration-json", "rel/cal.json"])
    assert explicit.out_root == "rel/out"
    assert explicit.model_dir == "rel/model"
    assert explicit.calibration_json == "rel/cal.json"


def test_pipeline_help_invocation_works_outside_repository(tmp_path):
    mod = _load_module()
    result = subprocess.run(
        [sys.executable, mod.__file__, "--help"], cwd=tmp_path,
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode == 0
    assert "--model-dir" in result.stdout and "--calibration-json" in result.stdout



def test_receiver_source_provenance_distinguishes_available_expected_and_configured_samples(tmp_path):
    mod = _load_module()
    iq, exe, manifest = _write_receiver_cache(mod, tmp_path)
    source = json.loads(manifest.read_text())["source"]
    assert source["available_scalar_int16_items"] == 4_800_000_000
    assert source["expected_complex_samples"] == 2_400_000_000
    assert source["configured_signal_source_samples"] == 0
    assert source["signal_source_samples_semantics"] == "auto_until_eof"
    assert source["signal_source_repeat"] is False
    assert "signal_source_samples" not in source
    assert source["configured_signal_source_samples"] != source["available_scalar_int16_items"]


def test_full_receiver_coverage_rejects_240_seconds_and_accepts_small_tail(tmp_path):
    mod = _load_module()
    run = tmp_path / "run"; run.mkdir()
    (run / "tracking.csv").write_text("time_s,prn\n0,G01\n240.08,G01\n")
    (run / "tracking_summary.csv").write_text("end_time_s,prn,row_count\n240.08,G01,2\n")
    with pytest.raises(ValueError, match="coverage"):
        mod.receiver_tracking_coverage(run)
    (run / "tracking.csv").write_text("time_s,prn\n0,G01\n479.5,G01\n")
    (run / "tracking_summary.csv").write_text("end_time_s,prn,row_count\n479.5,G01,2\n")
    got = mod.receiver_tracking_coverage(run)
    assert got["required_min_time_s"] == 478.0
    assert got["allowed_max_time_s"] == 481.0
    assert got["tracking_csv_max_time_s"] == 479.5
    assert got["tracking_summary_max_time_s"] == 479.5
    assert got["valid_tracking_prns"] == ["G01"]
    assert got["valid_tracking_row_count"] == 2


def test_receiver_coverage_rejects_invalid_prns_and_rows(tmp_path):
    mod = _load_module()
    (tmp_path / "tracking.csv").write_text("time_s,prn\n479.5,not-a-prn\n")
    (tmp_path / "tracking_summary.csv").write_text("end_time_s,prn,row_count\n479.5,not-a-prn,1\n")
    with pytest.raises(ValueError, match="PRN|tracking"):
        mod.receiver_tracking_coverage(tmp_path)


def test_incomplete_coverage_writes_structured_failure_and_no_complete_manifest(tmp_path, monkeypatch):
    mod = _load_module(); monkeypatch.setattr(mod, "EXPECTED_IQ_BYTES", 4)
    iq = tmp_path / "os1.bin"; iq.write_bytes(b"iqiq")
    exe = tmp_path / "fake.py"; _write_success_receiver(exe, iq)
    def incomplete(mats, output, summary, *, sample_rate_hz):
        Path(output).write_text("time_s,prn\n0,G01\n240.08,G01\n")
        Path(summary).write_text("end_time_s,prn,row_count\n240.08,G01,2\n")
        return {"row_count":2,"prns":["G01"],"channel_count":1}
    monkeypatch.setattr(mod, "export_tracking_csv", incomplete)
    with pytest.raises(RuntimeError, match="coverage"):
        mod.run_receiver("os1", iq, tmp_path / "out", exe=str(exe), timeout_s=5, force=True)
    run = tmp_path / "out/receiver/oakbat-os1-method-a-9tap"
    assert not (run / "manifest.json").exists()
    failure = json.loads((run / "receiver_failure.json").read_text())
    assert failure["failure_kind"] == "incomplete_tracking_coverage"
    assert failure["coverage"]["expected_duration_s"] == 480.0
    assert failure["coverage"]["required_min_time_s"] == 478.0
    assert failure["coverage"]["tracking_csv_max_time_s"] == 240.08


def test_receiver_cache_rejects_manifest_coverage_claim_and_artifact_tampering(tmp_path, monkeypatch):
    mod = _load_module(); monkeypatch.setattr(mod, "EXPECTED_IQ_BYTES", 4)
    iq, exe, manifest = _write_receiver_cache(mod, tmp_path)
    doc = json.loads(manifest.read_text())
    doc["tracking"]["coverage"]["tracking_csv_max_time_s"] = 480.0
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="coverage|cache"):
        mod.validate_cached_receiver(manifest, "os1", iq, exe)


def test_raw_feature_schema_uses_window_bounds_while_node_schema_keeps_window_bin(tmp_path):
    mod = _load_module()
    raw = tmp_path / "raw.csv"
    raw.write_text('run_id,prn,window_start_s,window_end_s,window_mid_s,tap_count,tap_layout,tap_E4_mean\nr,G01,0,1,0.5,9,"E4,E3,E2,E,P,L,L2,L3,L4",1\n')
    frame = mod._finite_csv(raw, mod.RAW_FEATURE_REQUIRED_COLUMNS)
    assert "window_bin_s" not in frame.columns
    assert "window_mid_s" in frame.columns
    wrong = tmp_path / "wrong.csv"
    wrong.write_text('run_id,prn,window_bin_s,tap_count,tap_layout\nr,G01,0.5,9,"E4,E3,E2,E,P,L,L2,L3,L4"\n')
    with pytest.raises(ValueError, match="schema"):
        mod._finite_csv(wrong, mod.RAW_FEATURE_REQUIRED_COLUMNS)


def test_feature_use_reauthenticates_receiver_full_coverage(tmp_path, monkeypatch):
    mod = _load_module(); monkeypatch.setattr(mod, "EXPECTED_IQ_BYTES", 4)
    iq, exe, receiver = _write_receiver_cache(mod, tmp_path)
    out = tmp_path / "scenario"; out.mkdir()
    features = out / "tap9_tracking_features_w1.0_s0.5.csv"
    features.write_text('run_id,prn,window_start_s,window_end_s,window_mid_s,tap_count,tap_layout,tap_E4_mean\nr,G01,0,1,0.5,9,"E4,E3,E2,E,P,L,L2,L3,L4",1\n')
    node = out / "nodes.csv"; cols=["run_id","prn","window_bin_s",*mod.FROZEN_FEATURE_COLUMNS]
    node.write_text(",".join(cols)+"\n"+",".join(["r","G01","0.5",*(["1"]*len(mod.FROZEN_FEATURE_COLUMNS))])+"\n")
    mod.write_feature_cache_contract(out, receiver, features, node)
    (receiver.parent / "tracking.csv").write_text("time_s,prn\n240.08,G01\n")
    with pytest.raises(ValueError, match="coverage|receiver"):
        mod.validate_cached_features(out, receiver)


def test_force_features_rebuilds_partial_failed_feature_file(tmp_path, monkeypatch):
    mod = _load_module()
    receiver = tmp_path / "receiver/manifest.json"; receiver.parent.mkdir(parents=True)
    (receiver.parent / "tracking.csv").write_text("time_s,prn\n0,G01\n479.5,G01\n")
    (receiver.parent / "tracking_summary.csv").write_text("end_time_s,prn,row_count\n479.5,G01,2\n")
    receiver.write_text(json.dumps({"status":"complete","tracking":{"coverage":mod.receiver_tracking_coverage(receiver.parent)}}))
    out = tmp_path / "scenario"; out.mkdir()
    partial = out / "tap9_tracking_features_w1.0_s0.5.csv"; partial.write_text("run_id,prn\nr,G01\n")
    with pytest.raises(ValueError, match="stale|schema|cache"):
        mod.build_features("os1", out, receiver, force=False)
    def export_raw(*args, output_path, **kwargs):
        output_path.write_text('run_id,prn,window_start_s,window_end_s,window_mid_s,tap_count,tap_layout,tap_E4_mean\nr,G01,0,1,0.5,9,"E4,E3,E2,E,P,L,L2,L3,L4",1\n')
    def export_nodes(features, *, output_dir, **kwargs):
        output_dir.mkdir(parents=True)
        node=output_dir/"normal_prn_node_windows.csv"; cols=["run_id","prn","window_bin_s",*mod.FROZEN_FEATURE_COLUMNS]
        node.write_text(",".join(cols)+"\n"+",".join(["r","G01","0.5",*(["1"]*len(mod.FROZEN_FEATURE_COLUMNS))])+"\n")
        return node, output_dir/"graph.csv", output_dir/"manifest.json"
    monkeypatch.setattr(mod,"export_receiver_run_tap_feature_csv",export_raw)
    monkeypatch.setattr(mod,"export_tap_multi_prn_dataset",export_nodes)
    assert mod.build_features("os1",out,receiver,force=True).name == "normal_prn_node_windows.csv"
    assert "window_start_s" in partial.read_text().splitlines()[0]
