from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gnss_doppler_lab import qset_stage0a_r2 as Q
from gnss_doppler_lab import qset_stage0a_r2_execution_repair_v5 as R


def test_empty_existing_trace_channels_are_bound_not_parsed(tmp_path: Path) -> None:
    paths = []
    for channel in range(Q.TRACE_CHANNELS):
        path = tmp_path / f"trace_native_1ms_ch_{channel}.bin"
        path.write_bytes(b"")
        paths.append(path)
    result = Q.validate_galileo_trace(paths, "C-1")
    assert result["status"] == "FAIL"
    assert result["tracked_prn_count"] == 0
    assert all(row["status"] == "EMPTY_OPTIONAL_CHANNEL" for row in result["files"])
    assert all(row["sha256"] == hashlib.sha256(b"").hexdigest() for row in result["files"])


def test_missing_trace_channel_fails_closed(tmp_path: Path) -> None:
    paths = []
    for channel in range(Q.TRACE_CHANNELS - 1):
        path = tmp_path / f"trace_native_1ms_ch_{channel}.bin"
        path.write_bytes(b"")
        paths.append(path)
    with pytest.raises(Q.QSetError):
        Q.validate_galileo_trace(paths, "C-1")


def test_veml_pointer_patch_binds_frozen_outer_taps() -> None:
    text = R.V5_PATCH.read_text(encoding="utf-8")
    assert "d_Very_Early = &d_correlator_outs[0]" in text
    assert "d_Very_Late = &d_correlator_outs[8]" in text
    assert "d_Early = &d_correlator_outs[3]" in text
    assert "d_Late = &d_correlator_outs[5]" in text


def test_segfault_attempt_preservation_is_versioned(tmp_path: Path) -> None:
    root = tmp_path / "replays" / "C-1"
    receiver = root / "receiver"
    receiver.mkdir(parents=True)
    (root / "decoded_4msps_gr_complex.bin").write_bytes(b"decoded")
    (receiver / "receiver.log").write_text("Flowgraph started\n", encoding="utf-8")
    for channel in range(Q.TRACE_CHANNELS):
        (receiver / f"trace_native_1ms_ch_{channel}.bin").write_bytes(b"")
    failure_log = tmp_path / "failure.log"
    failure_log.write_text("truncated TRACE header\n", encoding="utf-8")
    preserved = tmp_path / "historical" / "C-1-segfault"
    result = R.preserve_segfault_attempt(root, preserved, failure_log, expected_decoder_size=7)
    assert result["status"] == "PRESERVED_PRE_SCORE_RECEIVER_SIGSEGV_ATTEMPT"
    assert not root.exists()
    assert (preserved / "attempt_preservation.json").is_file()
    assert result["terminal_drain"] is False
    assert result["score_computed"] is False
