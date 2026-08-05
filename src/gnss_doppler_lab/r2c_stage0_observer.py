"""Side-effect-isolated durability primitives for the Stage-0 one-shot runner."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import threading
import time
from typing import Mapping


def _proc_start_ticks(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().split()[21])
    except (OSError, ValueError, IndexError):
        return None


def _usage() -> tuple[int, float]:
    import resource
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return rss, time.process_time()


def atomic_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":")) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try: temporary.unlink()
        except FileNotFoundError: pass
        raise
    directory = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


@dataclass
class ProgressObserver:
    directory: Path
    interval_s: float = 15.
    enabled: bool = True
    started: float = field(default_factory=time.monotonic)
    heartbeat_seq: int = 0
    progress_seq: int = 0
    state: dict = field(default_factory=dict)
    termination: threading.Event = field(default_factory=threading.Event)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self):
        if not self.enabled: return self
        self.directory.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, name="r2c-heartbeat", daemon=True)
        self._thread.start(); self.heartbeat()
        return self

    def _document(self):
        rss, cpu = _usage()
        return {**self.state, "elapsed_s":time.monotonic()-self.started, "pid":os.getpid(),
                "start_ticks":_proc_start_ticks(os.getpid()), "rss_bytes":rss, "cpu_time_s":cpu,
                "heartbeat_seq":self.heartbeat_seq, "progress_seq":self.progress_seq}

    def heartbeat(self, *, terminal: str | None = None):
        if not self.enabled: return
        with self._lock:
            self.heartbeat_seq += 1; value=self._document()
            if terminal is not None: value["terminal"] = terminal
            atomic_json(self.directory/"heartbeat.json", value)

    def progress(self, **state):
        if not self.enabled: return
        with self._lock:
            previous_completed=self.state.get("global_completed_bins",self.state.get("completed_bins",0))
            self.state.update(state); self.progress_seq += 1; value=self._document()
            completed=value.get("global_completed_bins",value.get("completed_bins",0));total=value.get("global_total_bins",value.get("total_bins",completed))
            if not 0 <= completed <= total or completed < previous_completed:
                raise ValueError("progress counters must be monotonic and bounded")
            authoritative=self.directory/"progress"/f"{self.progress_seq:012d}.json"
            if authoritative.exists():raise FileExistsError("authoritative progress event already exists")
            atomic_json(authoritative,value)
            with (self.directory/"progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n")
                stream.flush(); os.fsync(stream.fileno())

    def _loop(self):
        while not self._stop.wait(self.interval_s): self.heartbeat()

    def stop(self, terminal="complete"):
        if not self.enabled: return
        self._stop.set()
        if self._thread is not None: self._thread.join(timeout=max(1., self.interval_s))
        self.heartbeat(terminal=terminal)

    def install_sigterm(self):
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_: self.termination.set())
        return previous


def mark_partial(staging: Path, reason: str) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    atomic_json(staging/"PARTIAL_NON_AUTHORITATIVE", {"reason":reason,"pid":os.getpid(),"time":time.time()})


def atomic_publish(staging: Path, canonical: Path) -> None:
    if canonical.exists(): raise FileExistsError("canonical output already exists")
    if staging.parent != canonical.parent: raise ValueError("staging and canonical output must share a filesystem directory")
    marker=staging/"PARTIAL_NON_AUTHORITATIVE"
    if marker.exists(): marker.unlink()
    for root, directories, files in os.walk(staging):
        for name in files:
            descriptor=os.open(Path(root)/name,os.O_RDONLY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
        descriptor=os.open(root,os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    os.rename(staging,canonical)
    descriptor=os.open(canonical.parent,os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
