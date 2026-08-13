from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


def _claim(path):
    from gnss_doppler_lab.gcspo_freeze import claim_protected_attempt
    try:
        claim_protected_attempt(path, {"target_commit": "a" * 40})
        return "CLAIMED"
    except FileExistsError:
        return "EXISTS"


def test_exact_once_marker_is_atomic_and_has_one_concurrent_claimant(tmp_path):
    marker = tmp_path / "protected_run_started.json"
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(_claim, [marker] * 8))
    assert outcomes.count("CLAIMED") == 1 and outcomes.count("EXISTS") == 7
    assert json.loads(marker.read_text())["target_commit"] == "a" * 40


def test_malformed_freeze_or_manifest_never_consumes_marker(tmp_path):
    from gnss_doppler_lab.gcspo_freeze import validate_then_claim

    marker = tmp_path / "marker.json"
    with pytest.raises(ValueError):
        validate_then_claim(marker, freeze={"validity_state": "VALID_FOR_PROTECTED_ACCESS"},
                            expected_commit="a" * 40, live_remote_sha="a" * 40,
                            implementation_hashes={}, clean_hashes={})
    assert not marker.exists()


def test_live_remote_sha_not_stale_tracking_ref_and_exact_sync_required(tmp_path, monkeypatch):
    from gnss_doppler_lab.gcspo_freeze import live_remote_snapshot

    calls = []
    class Result:
        def __init__(self, stdout): self.stdout = stdout
    def fake(command, **kwargs):
        calls.append(command)
        if "ls-remote" in command: return Result("b" * 40 + "\trefs/heads/topic\n")
        if command[-2:] == ["rev-parse", "HEAD"]: return Result("a" * 40 + "\n")
        if "status" in command: return Result("")
        raise AssertionError(command)
    monkeypatch.setattr("gnss_doppler_lab.gcspo_freeze.subprocess.run", fake)
    snapshot = live_remote_snapshot(tmp_path, "origin", "topic")
    assert any("ls-remote" in command for command in calls)
    assert snapshot["remote_sha"] == "b" * 40 and snapshot["synchronized"] is False


@pytest.mark.parametrize(("ordinary", "ignored", "expected"), [
    ("?? src/pkg/shadow.py\n", "", False),
    ("?? config.local.json\n", "", False),
    ("?? notes.txt\n", "", False),
    ("A  tracked.py\n", "", False),
    (" M tracked.py\n", "", False),
    ("", "!! src/pkg/shadow.py\n", False),
    ("", "!! config.runtime.json\n", False),
    ("", "!! artifacts/gcspo_stage0_static_rerun/clean_only_report.json\n", True),
    ("", "", True),
])
def test_live_remote_requires_full_status_and_rejects_runtime_relevant_ignored(
        tmp_path, monkeypatch, ordinary, ignored, expected):
    from gnss_doppler_lab.gcspo_freeze import live_remote_snapshot

    calls = []
    class Result:
        def __init__(self, stdout): self.stdout = stdout
    def fake(command, **kwargs):
        calls.append(command)
        if "ls-remote" in command: return Result("a" * 40 + "\trefs/heads/topic\n")
        if command[-2:] == ["rev-parse", "HEAD"]: return Result("a" * 40 + "\n")
        if "status" in command: return Result(ignored if "--ignored" in command else ordinary)
        raise AssertionError(command)
    monkeypatch.setattr("gnss_doppler_lab.gcspo_freeze.subprocess.run", fake)
    snapshot = live_remote_snapshot(tmp_path, "origin", "topic")
    assert snapshot["synchronized"] is expected
    status_calls = [call for call in calls if "status" in call]
    assert all("--untracked-files=no" not in call for call in status_calls)
    assert all("--untracked-files=all" in call for call in status_calls)
    assert any("--ignored" in call for call in status_calls)


def test_freeze_binds_ignored_science_bytes_and_rejects_mutation(tmp_path):
    from gnss_doppler_lab.gcspo_freeze import build_freeze_record, verify_freeze_record

    implementation = tmp_path / "implementation.py"; implementation.write_text("x=1\n")
    clean = tmp_path / "ignored_clean.json"; clean.write_text('{"score":1}\n')
    record = build_freeze_record(target_commit="a" * 40, config_sha256="b" * 64,
                                 implementation_files=[implementation], clean_files=[clean],
                                 review_evidence={"reviewer": "independent", "status": "PASS"})
    verify_freeze_record(record, target_commit="a" * 40)
    clean.write_text('{"score":2}\n')
    with pytest.raises(ValueError, match="clean artifact hash"):
        verify_freeze_record(record, target_commit="a" * 40)


def test_inventory_missing_authenticated_manifest_size_is_preaccess_invalid():
    from gnss_doppler_lab.gcspo_freeze import validate_protected_manifest_inventory

    inventory = {"scenario_inventory": [{"id": "DS3", "receiver_manifest_sha256": "a" * 64,
                                          "receiver_manifest_path": "/sealed/manifest.json"}]}
    with pytest.raises(ValueError, match="manifest identity"):
        validate_protected_manifest_inventory(inventory, required=("DS3",))
