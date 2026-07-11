from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnss_doppler_lab.research_sequence import latest_run, load_run_manifest, sequence_status


def _make_run(root: Path, name: str, mtime_ns: int, *, with_iq: bool = True) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"scenario": {"name": name}}))
    if with_iq:
        (run / "gps_l1ca_s8_iq.bin").write_bytes(b"\x01\x02" * 2048)
    (run / "manifest.json").touch()
    import os

    os.utime(run / "manifest.json", ns=(mtime_ns, mtime_ns))
    return run


def test_latest_run_selects_newest_complete_run(tmp_path: Path) -> None:
    _make_run(tmp_path, "older", 100)
    newest = _make_run(tmp_path, "newer", 200)
    _make_run(tmp_path, "incomplete", 300, with_iq=False)

    assert latest_run(tmp_path) == newest


def test_latest_run_explains_when_no_complete_run_exists(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="complete RF run"):
        latest_run(tmp_path)


def test_load_run_manifest_returns_mapping(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "run-a", 100)

    assert load_run_manifest(run)["scenario"]["name"] == "run-a"


def test_sequence_status_reports_stage_artifacts(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "rf_runs", "normal", 100)
    receiver = tmp_path / "receiver_runs" / "normal"
    receiver.mkdir(parents=True)
    (receiver / "observables.csv").write_text("time,prn,doppler_hz\n0,G01,1200\n")

    status = sequence_status(tmp_path)

    assert status["01_normal_iq"]["ready"] is True
    assert status["02_receiver_processing"]["ready"] is True
    assert status["03_spoofing_comparison"]["ready"] is False
    assert status["01_normal_iq"]["path"] == str(run)
