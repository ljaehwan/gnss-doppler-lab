from __future__ import annotations
import json, os, signal, sys, time
from pathlib import Path
import pytest

from gnss_doppler_lab.r2c_stage0_observer import ProgressObserver, atomic_publish, mark_partial

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from supervise_r2c_stage0 import supervise


def test_heartbeat_progress_and_byte_neutrality(tmp_path):
    scientific=lambda: json.dumps({"delays":[-.1,.2],"score":3.5},sort_keys=True).encode()
    before=scientific();observer=ProgressObserver(tmp_path/"attempt",interval_s=.01).start()
    observer.progress(stage="score",scenario="synthetic",completed_bins=1,total_bins=2,current_bin=1,
                      recent_bins_s=2.,eta_s=.5);time.sleep(.03);observer.stop()
    assert scientific()==before
    heart=json.loads((tmp_path/"attempt/heartbeat.json").read_text());assert heart["heartbeat_seq"]>=2
    rows=(tmp_path/"attempt/progress.jsonl").read_text().splitlines();assert len(rows)==1
    assert json.loads(rows[0])["progress_seq"]==1


def test_atomic_publish_no_partial_and_no_overwrite(tmp_path):
    canonical=tmp_path/"result";staging=tmp_path/".result.attempt"
    mark_partial(staging,"assembling");(staging/"value").write_bytes(b"complete")
    assert not canonical.exists();atomic_publish(staging,canonical)
    assert (canonical/"value").read_bytes()==b"complete"
    other=tmp_path/".result.other";other.mkdir()
    with pytest.raises(FileExistsError):atomic_publish(other,canonical)


def test_interruption_leaves_non_authoritative_staging(tmp_path):
    staging=tmp_path/".result.interrupted";canonical=tmp_path/"result"
    mark_partial(staging,"sigterm");(staging/"partial").write_text("x")
    assert not canonical.exists() and (staging/"PARTIAL_NON_AUTHORITATIVE").is_file()


def test_supervisor_duplicate_silent_dead_and_no_retry(tmp_path):
    attempt=tmp_path/"attempt"
    code=supervise([sys.executable,"-c","import time;time.sleep(.1)"],attempt,poll_s=.01,stale_s=.02)
    assert code==0
    doc=json.loads((attempt/"supervisor.json").read_text());assert doc["exit_code"]==0 and doc["auto_retries"]==0
    with pytest.raises(FileExistsError):supervise([sys.executable,"-c","pass"],attempt)
    dead=tmp_path/"dead";assert supervise([sys.executable,"-c","raise SystemExit(7)"],dead,poll_s=.01)==7
    assert json.loads((dead/"supervisor.json").read_text())["exit_code"]==7


def test_stale_heartbeat_warns_without_kill_or_retry(tmp_path):
    attempt=tmp_path/"stale"
    # Reservation is performed by supervise; the worker creates one heartbeat and then stays silent.
    script="from pathlib import Path;import time; p=Path(r'%s');p.write_text('{}');time.sleep(.15)"%(attempt/"heartbeat.json")
    assert supervise([sys.executable,"-c",script],attempt,poll_s=.01,stale_s=.03)==0
    assert (attempt/"stale-heartbeat-warning.json").exists()
