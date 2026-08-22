from __future__ import annotations

from pathlib import Path

import json
import os
import subprocess

import pytest

from gnss_doppler_lab import qset_stage0a_r2 as Q
from gnss_doppler_lab import qset_stage0a_r2_execution_repair_v4 as R


def test_gnuradio_runtime_binding_is_explicit() -> None:
    code = "import json; from gnss_doppler_lab.qset_stage0a_r2_execution_repair_v4 import bind_gnuradio_runtime; print(json.dumps(bind_gnuradio_runtime()))"
    completed = subprocess.run(["/usr/bin/python3", "-c", code], text=True, capture_output=True, check=True, env={**os.environ, "PYTHONPATH": str(R.ROOT / "src")})
    runtime = json.loads(completed.stdout)
    assert runtime["status"] == "PASS"
    assert runtime["python_executable"] == "/usr/bin/python3"
    assert runtime["system_dist_packages"] == "/usr/lib/python3/dist-packages"
    assert runtime["gnuradio_version"]


def test_empty_failed_attempt_is_preserved_without_overwrite(tmp_path: Path) -> None:
    failed = tmp_path / "replays" / "C-1"
    failed.mkdir(parents=True)
    preserved = tmp_path / "historical" / "C-1"
    log = tmp_path / "failure.log"
    log.write_text("ModuleNotFoundError: gnuradio\n", encoding="utf-8")
    result = R.preserve_empty_attempt(failed, preserved, log)
    assert result["status"] == "PRESERVED_EMPTY_PRE_SCORE_ATTEMPT"
    assert not failed.exists()
    assert (preserved / "attempt_preservation.json").is_file()
    assert result["log"]["sha256"] == Q.sha256_file(log)


def test_nonempty_failed_attempt_fails_closed(tmp_path: Path) -> None:
    failed = tmp_path / "replays" / "C-1"
    failed.mkdir(parents=True)
    (failed / "partial.bin").write_bytes(b"preserve me")
    log = tmp_path / "failure.log"
    log.write_text("failure\n", encoding="utf-8")
    with pytest.raises(Q.QSetError):
        R.preserve_empty_attempt(failed, tmp_path / "historical" / "C-1", log)
    assert (failed / "partial.bin").read_bytes() == b"preserve me"
