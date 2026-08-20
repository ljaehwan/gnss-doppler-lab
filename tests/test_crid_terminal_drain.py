import json
from pathlib import Path
import sys

from gnss_doppler_lab.crid_receiver_replay import _supervised_run


def test_verified_eof_and_stable_dumps_precede_graceful_sigint(tmp_path: Path):
    raw = tmp_path / "raw.bin"
    raw.write_bytes(b"x" * 1024)
    child = tmp_path / "child.py"
    child.write_text(
        """import signal,time,sys
from pathlib import Path
done=False
def stop(sig, frame):
 global done
 done=True
signal.signal(signal.SIGINT, stop)
with Path(sys.argv[1]).open('rb') as stream:
 while stream.read(127): pass
 for channel in range(10):
  with Path(f'trace_native_1ms_ch_{channel}.bin').open('wb') as dump:
   dump.write(b'TRACE' * 20)
 while not done: time.sleep(.01)
"""
    )
    with (tmp_path / "receiver.log").open("wb") as log:
        code, termination = _supervised_run(
            [sys.executable, str(child), str(raw)], cwd=tmp_path, log=log,
            raw=raw, expected_end_byte=1024, poll_s=.01, stable_s=.05, grace_s=1.
        )
    assert code == 0
    assert termination["status"] == "PASS"
    assert termination["exit_cause"] == "verified_eof_graceful_sigint"
    assert termination["input_bytes_consumed_exactly"] is True
    assert termination["max_raw_fd_position"] == 1024
    assert termination["dump_count"] == 10
    assert termination["pre_signal_stability_s"] >= .05
    assert termination["sigint_sent"] is True
    assert termination["sigterm_sent"] is False
    assert termination["sigkill_sent"] is False
    assert any(row["signal"] == "SIGINT" for row in termination["timeline"])


def test_terminal_contract_is_json_serializable(tmp_path: Path):
    assert json.dumps({"path": str(tmp_path), "signal": "SIGINT"})
