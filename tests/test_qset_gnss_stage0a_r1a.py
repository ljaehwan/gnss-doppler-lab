from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN = load("qset_r1a_run", "scripts/run_qset_gnss_stage0a_r1a.py")
VERIFY = load("qset_r1a_verify", "scripts/verify_qset_gnss_stage0a_r1a.py")


def write_records(path: Path, prn: int, *, seconds: float = 11.0, doppler: float = 500.0, cn0: float = 40.0) -> None:
    count = int(seconds / 0.004) + 1
    records = np.zeros(count, dtype=RUN.R1.TRACK_RECORD_DTYPE)
    records["prn"] = prn
    records["sample"] = 100_000 + np.arange(count, dtype=np.uint64) * 16_000
    records["doppler_hz"] = doppler
    records["cn0_db_hz"] = cn0
    records["carrier_lock"] = 1.0
    records.tofile(path)


def make_dump_set(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for channel in range(RUN.EXPECTED_CHANNELS):
        (root / f"veml_tracking_ch_{channel}.dat").write_bytes(b"")


def test_controlled_terminal_drain_requires_eof_and_graceful_markers(tmp_path: Path) -> None:
    child = tmp_path / "child.py"
    child.write_text(
        "import signal,sys,time\n"
        "def stop(sig, frame):\n"
        " print('GNSS-SDR received 2 OS signal', flush=True)\n"
        " print('Received action STOP', flush=True)\n"
        " print('Flowgraph stopped', flush=True)\n"
        " raise SystemExit(0)\n"
        "signal.signal(signal.SIGINT, stop)\n"
        "print('doing acquisition, sample stamp: 999', flush=True)\n"
        "while True: time.sleep(0.05)\n",
        encoding="utf-8",
    )
    result = RUN.supervise_receiver(
        [sys.executable, str(child)], tmp_path, tmp_path / "receiver.log", 4_000_500,
        max_runtime_s=2.0, quiet_s=0.1, stop_grace_s=2.0, poll_s=0.02,
    )
    assert result["status"] == "PASS"
    assert result["terminal_drain"] is True
    assert result["exit_code"] == 0
    assert result["max_acquisition_sample_stamp"] == 999
    assert all(result["markers"].values())


def test_terminal_drain_without_eof_fails_closed(tmp_path: Path) -> None:
    child = tmp_path / "child.py"
    child.write_text(
        "import signal,time\n"
        "def stop(sig, frame):\n"
        " print('GNSS-SDR received 2 OS signal', flush=True)\n"
        " print('Received action STOP', flush=True)\n"
        " print('Flowgraph stopped', flush=True)\n"
        " raise SystemExit(0)\n"
        "signal.signal(signal.SIGINT, stop)\n"
        "print('sample stamp: 10', flush=True)\n"
        "while True: time.sleep(0.05)\n",
        encoding="utf-8",
    )
    result = RUN.supervise_receiver(
        [sys.executable, str(child)], tmp_path, tmp_path / "receiver.log", 10_000_000,
        max_runtime_s=0.25, quiet_s=0.05, stop_grace_s=2.0, poll_s=0.02,
    )
    assert result["status"] == "FAIL_CLOSED"
    assert result["terminal_drain"] is False
    assert result["eof_evidence_pass"] is False
    assert result["forced_signal"] == "SIGINT_MAX_RUNTIME_WITHOUT_EOF"


def test_complete_record_adapter_preserves_and_hashes_tail(tmp_path: Path) -> None:
    receiver = tmp_path / "receiver"
    make_dump_set(receiver)
    records = np.zeros(3, dtype=RUN.R1.TRACK_RECORD_DTYPE)
    records["prn"] = 7
    records["sample"] = [10, 20, 30]
    tail = b"partial-tail"
    target = receiver / "veml_tracking_ch_0.dat"
    target.write_bytes(records.tobytes() + tail)
    rows = RUN.tracking_dump_inventory(receiver)
    row = rows[0]
    assert row["complete_record_count"] == 3
    assert row["trailing_fragment_size_bytes"] == len(tail)
    assert row["trailing_fragment_sha256"] == hashlib.sha256(tail).hexdigest()
    assert row["sha256"] == RUN.sha256_file(target)
    parsed = RUN.read_complete_tracking_records(target)
    assert parsed["prn"].tolist() == [7, 7, 7]


def test_empty_dump_is_unavailable_not_failure(tmp_path: Path) -> None:
    make_dump_set(tmp_path)
    rows = RUN.tracking_dump_inventory(tmp_path)
    assert len(rows) == 12
    assert all(row["status"] == "EMPTY_UNAVAILABLE_CHANNEL" for row in rows)
    assert all(row["complete_record_count"] == 0 for row in rows)


def test_missing_dump_fails_closed(tmp_path: Path) -> None:
    make_dump_set(tmp_path)
    (tmp_path / "veml_tracking_ch_11.dat").unlink()
    with pytest.raises(RUN.RepairError):
        RUN.tracking_dump_inventory(tmp_path)


def test_dump_hash_and_tail_tamper_are_detectable(tmp_path: Path) -> None:
    make_dump_set(tmp_path)
    target = tmp_path / "veml_tracking_ch_0.dat"
    target.write_bytes(np.zeros(1, dtype=RUN.R1.TRACK_RECORD_DTYPE).tobytes() + b"x")
    before = RUN.tracking_dump_inventory(tmp_path)[0]
    target.write_bytes(target.read_bytes()[:-1] + b"y")
    after = RUN.tracking_dump_inventory(tmp_path)[0]
    assert before["sha256"] != after["sha256"]
    assert before["trailing_fragment_sha256"] != after["trailing_fragment_sha256"]


def test_dynamic_panel_analysis_with_four_common_prns(tmp_path: Path) -> None:
    segments = []
    for index, segment_id in enumerate(("segment_030_060", "segment_100_130")):
        receiver = tmp_path / segment_id / "receiver"
        make_dump_set(receiver)
        (receiver / "receiver.log").write_text("", encoding="utf-8")
        for channel, prn in enumerate((2, 11, 25, 30)):
            write_records(receiver / f"veml_tracking_ch_{channel}.dat", prn, doppler=500 + prn + index * 10, cn0=40 + index)
        segments.append({
            "segment_id": segment_id,
            "source_start_sample": (1_500_000_000, 5_000_000_000)[index],
            "source_sample_count": 1_500_000_000,
            "decoder": {"output_samples": 119_999_966},
        })
    result = RUN.analyze_outputs({"segments": segments}, tmp_path)
    assert result["dynamic_panel"] == [2, 11, 25, 30]
    assert result["common_panel"] == [2, 11, 25, 30]
    assert result["acquisition_prn_count"] == 4
    assert result["tracking_ge_10s_prn_count"] == 4
    assert result["all_tracking_finite"] is True
    assert result["time_mapping_pass"] is True
    assert result["physical_consistency_pass"] is True
    assert all(len(mask) == 4 for mask in result["availability_mask"].values())


def test_dynamic_panel_with_fewer_than_four_common_prns_fails_physical_gate(tmp_path: Path) -> None:
    segments = []
    panels = ((2, 11, 25, 30), (2, 11, 25, 31))
    for index, segment_id in enumerate(("segment_030_060", "segment_100_130")):
        receiver = tmp_path / segment_id / "receiver"
        make_dump_set(receiver)
        (receiver / "receiver.log").write_text("", encoding="utf-8")
        for channel, prn in enumerate(panels[index]):
            write_records(receiver / f"veml_tracking_ch_{channel}.dat", prn, doppler=500 + prn, cn0=40)
        segments.append({"segment_id": segment_id, "source_start_sample": index * 3_500_000_000, "source_sample_count": 1_500_000_000, "decoder": {"output_samples": 119_999_966}})
    result = RUN.analyze_outputs({"segments": segments}, tmp_path)
    assert result["common_panel"] == [2, 11, 25]
    assert result["physical_consistency_pass"] is False


def test_frozen_receiver_configuration_and_scientific_gates_unchanged() -> None:
    template = ROOT / RUN.R1.CONFIG_TEMPLATE_REL
    assert RUN.sha256_file(template) == RUN.EXPECTED_CONFIG_SHA256
    prereg = json.loads((ROOT / RUN.ARTIFACT_REL / "repair_preregistration.json").read_text())
    gate = prereg["frozen_scientific_contract"]
    assert gate["acquisition_distinct_prns_min"] == 4
    assert gate["tracking_distinct_prns_ge_10s_min"] == 4
    assert gate["all_tracking_finite"] is True
    assert gate["receiver_configuration_template_sha256"] == RUN.EXPECTED_CONFIG_SHA256
    assert gate["receiver_sha256"] == RUN.EXPECTED_RECEIVER_SHA256


def test_access_audit_tamper_detection() -> None:
    audit = {
        "status": "PASS",
        "c1": {"identity_hash_passes": 1, "identity_hash_bytes": VERIFY.RAW_SIZE, "format_window_bytes": VERIFY.FORMAT_BYTES, "receiver_decode_bytes": VERIFY.DECODE_BYTES, "total_payload_bytes_read": VERIFY.EXPECTED_C1_BYTES},
        "c3": {key: 0 for key in ("stats", "hashes", "opens", "mmaps", "bytes_read", "downloads")},
        "attack": {key: 0 for key in ("stats", "hashes", "opens", "mmaps", "bytes_read", "downloads")},
        "other_tuni2025_raw": {key: 0 for key in ("stats", "hashes", "opens", "mmaps", "bytes_read", "downloads")},
    }
    VERIFY.validate_access(audit)
    changed = copy.deepcopy(audit)
    changed["c3"]["opens"] = 1
    with pytest.raises(VERIFY.VerificationError):
        VERIFY.validate_access(changed)
    changed = copy.deepcopy(audit)
    changed["attack"]["bytes_read"] = 1
    with pytest.raises(VERIFY.VerificationError):
        VERIFY.validate_access(changed)


def test_compact_manifest_tamper_detection(tmp_path: Path) -> None:
    target = tmp_path / "x"
    target.write_bytes(b"before")
    before = VERIFY.compact_manifest(tmp_path)
    target.write_bytes(b"after")
    assert before != VERIFY.compact_manifest(tmp_path)


def test_receiver_output_hash_tamper_detection(tmp_path: Path) -> None:
    target = tmp_path / "receiver.dat"
    target.write_bytes(b"receiver-output")
    expected = RUN.sha256_file(target)
    target.write_bytes(b"receiver-tamper")
    assert RUN.sha256_file(target) != expected


def test_r1a_contains_no_training_threshold_or_attack_scoring_path() -> None:
    text = (ROOT / "scripts/run_qset_gnss_stage0a_r1a.py").read_text()
    assert "qset_training_performed\": False" in text
    assert "threshold_calibrated\": False" in text
    assert "attack_scoring_performed\": False" in text
    assert "READY_FOR_TUNI2025_C3_CLEAN_DOWNLOAD" in text


def test_preregistration_commit_is_ancestor_of_current_head() -> None:
    result = subprocess.run(["git", "merge-base", "--is-ancestor", RUN.PREREGISTRATION_SHA, "HEAD"], cwd=ROOT)
    assert result.returncode == 0
