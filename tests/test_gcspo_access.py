from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from gnss_doppler_lab.gcspo_core import AccessGate


def _ready_gate(tmp_path: Path) -> AccessGate:
    gate = AccessGate(tmp_path / "ledger.jsonl")
    gate.set_preflight(clean_only_pass=True, reviews_pass=True, freeze_sha="a" * 40,
                       frozen_hashes={"config": "b" * 64})
    gate.set_remote_sync(local_sha="a" * 40, remote_sha="a" * 40, ahead=0, behind=0, clean=True)
    return gate


def _identity(path: Path):
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def test_capability_pre_post_identity_and_same_descriptor_text(tmp_path):
    path = tmp_path / "nmea.txt"
    path.write_text("$GPRMC,ok\n", encoding="utf-8")
    sha, size = _identity(path)
    gate = _ready_gate(tmp_path)
    gate.register_pinned(path, expected_sha256=sha, expected_size=size, kind="NMEA")
    assert gate.read_text(path, scenario="DS3", phase="pre", purpose="time anchor") == "$GPRMC,ok\n"
    rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [row["record_type"] for row in rows] == ["PRE", "POST"]
    assert rows[1]["outcome"] == "SUCCESS"
    assert rows[1]["observed_sha256"] == sha and rows[1]["observed_size"] == size
    assert rows[0]["byte_range"] == f"[0,{size})"
    assert [row["access_counter"] for row in rows] == [1, 1]
    assert all(row["run_identity"] == "a" * 40 for row in rows)
    assert all(row["timestamp_utc"].endswith("Z") for row in rows)
    assert rows[0]["timestamp_utc"] <= rows[1]["timestamp_utc"]
    assert rows[1]["record_sha256"] and rows[1]["previous_record_sha256"] == rows[0]["record_sha256"]


def test_capability_mismatch_before_exposure_and_denied_attempts(tmp_path):
    path = tmp_path / "tracking.mat"
    path.write_bytes(b"science")
    gate = _ready_gate(tmp_path)
    gate.register_pinned(path, expected_sha256="0" * 64, expected_size=7, kind="MAT")
    called = False
    def consume(_handle):
        nonlocal called
        called = True
    with pytest.raises(ValueError, match="identity mismatch"):
        gate.consume(path, scenario="DS3", phase="transition", purpose="score", consumer=consume)
    assert called is False
    with pytest.raises(PermissionError):
        gate.read_text(tmp_path / "unregistered.txt", scenario="DS3", phase="pre", purpose="deny")
    records = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert any(row["record_type"] == "POST" and row["outcome"] == "IDENTITY_MISMATCH" for row in records)
    assert any(row["record_type"] == "DENIED" for row in records)


def test_capability_rejects_symlink_glob_directory_and_prior_result(tmp_path):
    target = tmp_path / "source.txt"; target.write_text("x")
    link = tmp_path / "link.txt"; link.symlink_to(target)
    gate = _ready_gate(tmp_path)
    sha, size = _identity(target)
    for candidate in (link, tmp_path, tmp_path / "*.mat", tmp_path / "scenario_metrics.csv"):
        with pytest.raises((PermissionError, ValueError)):
            gate.register_pinned(candidate, expected_sha256=sha, expected_size=size, kind="TEST")
    records = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert len(records) == 4 and all(row["record_type"] == "DENIED" for row in records)


def test_capability_detects_path_replacement_during_consume(tmp_path):
    path = tmp_path / "file.xml"; path.write_text("<a/>")
    replacement = tmp_path / "replacement.xml"; replacement.write_text("<b/>")
    gate = _ready_gate(tmp_path); sha, size = _identity(path)
    gate.register_pinned(path, expected_sha256=sha, expected_size=size, kind="XML")
    def replace(handle):
        assert handle.read() == b"<a/>"
        os.replace(replacement, path)
        return "parsed"
    with pytest.raises(RuntimeError, match="replaced"):
        gate.consume(path, scenario="DS7", phase="pre", purpose="ephemeris", consumer=replace)
    post = json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[-1])
    assert post["outcome"] == "PATH_REPLACED"


def test_authenticated_manifest_is_required_for_child_mat_nmea_xml(tmp_path):
    child = tmp_path / "child.mat"
    with h5py.File(child, "w") as handle:
        handle["x"] = np.arange(3)
    sha, size = _identity(child)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": "child.mat", "sha256": sha, "size_bytes": size}]}) + "\n")
    manifest_sha, manifest_size = _identity(manifest)
    gate = _ready_gate(tmp_path)
    gate.register_pinned(manifest, expected_sha256=manifest_sha, expected_size=manifest_size, kind="RECEIVER_MANIFEST")
    gate.authenticate_manifest(manifest, scenario="DS7", phase="all", purpose="receiver output identities")
    arrays = gate.read_h5(child, datasets=["x"], scenario="DS7", phase="transition", purpose="tracking")
    assert arrays["x"].tolist() == [0, 1, 2]
    missing = tmp_path / "missing.xml"; missing.write_text("<x/>")
    with pytest.raises(PermissionError, match="manifest-derived identity"):
        gate.read_text(missing, scenario="DS7", phase="transition", purpose="xml")


def test_manifest_rejects_missing_or_malformed_child_identity(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": "a.mat", "size_bytes": 1}]}) + "\n")
    sha, size = _identity(manifest)
    gate = _ready_gate(tmp_path)
    gate.register_pinned(manifest, expected_sha256=sha, expected_size=size, kind="RECEIVER_MANIFEST")
    with pytest.raises(ValueError, match="child identity"):
        gate.authenticate_manifest(manifest, scenario="DS3", phase="all", purpose="manifest")
