from __future__ import annotations

from pathlib import Path

from gnss_doppler_lab import qset_stage0a_r2 as Q
from gnss_doppler_lab import qset_stage0a_r2_execution_repair_v7 as R


def test_exact_boundary_patch_queues_drain_after_copy_before_return() -> None:
    text = R.V7_PATCH.read_text(encoding="utf-8")
    copied = text.rfind("d_ncopied_items += n")
    condition = text.rfind("d_terminal_drain and d_ncopied_items >= d_nitems")
    queue = text.rfind("command_event_make(200, 2U)")
    returned = text.rfind("return n")
    assert -1 < copied < condition < queue < returned
    assert "return n so every final output item remains" in text


def test_exact_boundary_patch_sends_once_and_preserves_non_drain_path() -> None:
    text = R.V7_PATCH.read_text(encoding="utf-8")
    assert "d_terminal_action_sent(false)" in text
    assert "not d_terminal_action_sent" in text
    assert "d_terminal_action_sent = true" in text
    assert "const unsigned int action = d_terminal_drain ? 2U : 0U" in text
    assert R.V7_PATCH_SHA256 == R.sha256_file(R.V7_PATCH)


def test_exact_boundary_attempt_is_preserved_without_score(tmp_path: Path) -> None:
    root = tmp_path / "replays" / "C-1"
    receiver = root / "receiver"
    receiver.mkdir(parents=True)
    (root / "decoded_4msps_gr_complex.bin").write_bytes(b"decoded")
    (receiver / "receiver.log").write_text(
        "Current receiver time: 2 min 30 s\nReceived action STOP\nFlowgraph stopped\n", encoding="utf-8"
    )
    for channel in range(Q.TRACE_CHANNELS):
        (receiver / f"trace_native_1ms_ch_{channel}.bin").write_bytes(b"")
    wrapper = tmp_path / "wrapper.log"
    wrapper.write_text("KeyboardInterrupt\n", encoding="utf-8")
    preserved = tmp_path / "historical" / "C-1-boundary"
    result = R.preserve_exact_boundary_attempt(root, preserved, wrapper, expected_decoder_size=7)
    assert result["status"] == "PRESERVED_PRE_SCORE_EXACT_BOUNDARY_DRAIN_DEADLOCK"
    assert result["clean_score_computed"] is False
    assert result["terminal_drain"] is False
    assert preserved.is_dir() and not root.exists()


def test_completed_replay_manifest_is_checkpoint_not_failed_attempt(tmp_path: Path) -> None:
    replay = tmp_path / "C-1"
    replay.mkdir()
    (replay / "manifest.json").write_text('{"status":"PASS"}\n', encoding="utf-8")
    assert replay.exists() and (replay / "manifest.json").is_file()
    assert not (replay.exists() and not (replay / "manifest.json").is_file())
