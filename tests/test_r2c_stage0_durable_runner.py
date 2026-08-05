from __future__ import annotations
import json, os, signal, sys, time
from pathlib import Path
import pytest
import importlib.util

from gnss_doppler_lab.r2c_stage0_observer import ProgressObserver, atomic_json, atomic_publish, mark_partial

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
    authoritative=list((tmp_path/"attempt/progress").glob("*.json"));assert len(authoritative)==1
    assert json.loads(authoritative[0].read_text())["progress_seq"]==1


def test_atomic_publish_no_partial_and_no_overwrite(tmp_path):
    canonical=tmp_path/"result";staging=tmp_path/".result.attempt"
    mark_partial(staging,"assembling");(staging/"value").write_bytes(b"complete")
    assert not canonical.exists();atomic_publish(staging,canonical)
    assert (canonical/"value").read_bytes()==b"complete"
    other=tmp_path/".result.other";mark_partial(other,"other")
    with pytest.raises(FileExistsError):atomic_publish(other,canonical)

def test_publish_preset_termination_never_commits(tmp_path):
    import threading
    staging=tmp_path/".result.pending";canonical=tmp_path/"result";mark_partial(staging,"pending");(staging/"x").write_text("x")
    event=threading.Event();event.set()
    with pytest.raises(InterruptedError):atomic_publish(staging,canonical,event)
    assert not canonical.exists() and (staging/"PARTIAL_NON_AUTHORITATIVE").exists()

def test_destination_creation_race_survives(tmp_path,monkeypatch):
    import gnss_doppler_lab.r2c_stage0_observer as module
    staging=tmp_path/".result.race";canonical=tmp_path/"result";mark_partial(staging,"race");(staging/"x").write_text("x")
    real=module._rename_noreplace
    def race(source,destination):destination.mkdir();(destination/"owner").write_text("other");return real(source,destination)
    monkeypatch.setattr(module,"_rename_noreplace",race)
    with pytest.raises(FileExistsError):atomic_publish(staging,canonical)
    assert (canonical/"owner").read_text()=="other" and (staging/"PARTIAL_NON_AUTHORITATIVE").exists()

def test_post_rename_parent_fsync_is_warning_not_failure(tmp_path,monkeypatch):
    import gnss_doppler_lab.r2c_stage0_observer as module
    staging=tmp_path/".result.fsync";canonical=tmp_path/"result";mark_partial(staging,"ready");(staging/"x").write_text("x")
    real=module.os.fsync;calls=[]
    def injected(fd):
        calls.append(fd)
        if len(calls)>3:raise OSError(5,"post rename fsync")
        return real(fd)
    monkeypatch.setattr(module.os,"fsync",injected);warnings=atomic_publish(staging,canonical)
    assert canonical.is_dir() and warnings


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
    dead=tmp_path/"other-campaign/dead";dead.parent.mkdir();assert supervise([sys.executable,"-c","raise SystemExit(7)"],dead,poll_s=.01)==7
    assert json.loads((dead/"supervisor.json").read_text())["exit_code"]==7


def test_stale_heartbeat_warns_without_kill_or_retry(tmp_path):
    attempt=tmp_path/"stale"
    # Reservation is performed by supervise; the worker creates one heartbeat and then stays silent.
    script="from pathlib import Path;import time; p=Path(r'%s');p.write_text('{}');time.sleep(.15)"%(attempt/"heartbeat.json")
    assert supervise([sys.executable,"-c",script],attempt,poll_s=.01,stale_s=.03)==0
    assert (attempt/"stale-heartbeat-warning.json").exists()

def test_missing_initial_heartbeat_warns_without_restart(tmp_path):
    attempt=tmp_path/"missing"
    assert supervise([sys.executable,"-c","import time;time.sleep(.08)"],attempt,poll_s=.01,initial_heartbeat_s=.02)==0
    assert (attempt/"missing-initial-heartbeat-warning.json").exists()
    assert json.loads((attempt/"supervisor.json").read_text())["auto_retries"]==0

def test_two_attempt_ids_share_global_exact_once_reservation(tmp_path):
    first=tmp_path/"attempt-a";second=tmp_path/"attempt-b"
    assert supervise([sys.executable,"-c","pass"],first,poll_s=.01)==0
    with pytest.raises(FileExistsError):supervise([sys.executable,"-c","pass"],second,poll_s=.01)
    assert not second.exists()

def test_atomic_progress_failure_has_no_authoritative_partial(tmp_path,monkeypatch):
    import gnss_doppler_lab.r2c_stage0_observer as module
    real_replace=module.os.replace
    def fail_replace(source,target):
        if Path(target).parent.name=="progress":raise OSError(28,"No space left on device")
        return real_replace(source,target)
    monkeypatch.setattr(module.os,"replace",fail_replace)
    observer=ProgressObserver(tmp_path/"attempt").start()
    with pytest.raises(OSError):observer.progress(global_completed_bins=1,global_total_bins=2)
    assert not list((tmp_path/"attempt/progress").glob("*.json"))
    observer.stop()

def test_observer_on_off_actual_synthetic_artifact_bytes_identical(tmp_path):
    root=Path(__file__).resolve().parents[1];path=root/"scripts/run_r2c_gnss_stage0_fix.py"
    spec=importlib.util.spec_from_file_location("durable_runner_synthetic",path);runner=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=runner;spec.loader.exec_module(runner);config=json.loads((root/"configs/r2c_gnss_stage0_fix.json").read_text())
    off=tmp_path/"off";on=tmp_path/"on";runner.freeze_determinism(config["seed"]);runner.run_synthetic(off,config,"fixture-source")
    observer=ProgressObserver(tmp_path/"runtime",interval_s=.01).start();runner.freeze_determinism(config["seed"]);runner.run_synthetic(on,config,"fixture-source");observer.stop()
    off_files={str(p.relative_to(off)):p.read_bytes() for p in off.rglob("*") if p.is_file()}
    on_files={str(p.relative_to(on)):p.read_bytes() for p in on.rglob("*") if p.is_file()}
    assert on_files==off_files
