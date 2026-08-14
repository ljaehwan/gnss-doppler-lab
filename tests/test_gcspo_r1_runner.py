from pathlib import Path

import pytest

from gnss_doppler_lab import gcspo_evaluate, gcspo_verify_artifacts
from gnss_doppler_lab.gcspo_r1_runner import (
    INVOCATION_ID,
    claim_once,
    install_and_verify_adapter,
    verify_zero_access_state,
)
from gnss_doppler_lab.gcspo_r1_support import exact_b0_full_contrast_r1


def test_install_and_verify_adapter_binds_evaluator_and_verifier():
    install_and_verify_adapter()
    assert gcspo_evaluate.exact_b0_full_contrast is exact_b0_full_contrast_r1
    assert gcspo_verify_artifacts.exact_b0_full_contrast is exact_b0_full_contrast_r1


def test_zero_access_state_rejects_marker_ledger_or_verdict(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    marker = tmp_path / "marker.json"
    verify_zero_access_state(artifact, marker)

    marker.write_text("{}\n")
    with pytest.raises(ValueError, match="marker"):
        verify_zero_access_state(artifact, marker)
    marker.unlink()

    (artifact / "access_ledger.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="ledger"):
        verify_zero_access_state(artifact, marker)
    (artifact / "access_ledger.jsonl").unlink()

    (artifact / "final_verdict.json").write_text("{}\n")
    with pytest.raises(ValueError, match="verdict"):
        verify_zero_access_state(artifact, marker)


def test_claim_is_o_excl_and_bound_to_new_invocation(tmp_path):
    marker = tmp_path / "marker.json"
    first = claim_once(marker, wrapper_commit="a" * 40, target_commit="b" * 40)
    assert first["invocation_id"] == INVOCATION_ID
    assert first["protected_run_count"] == 1
    with pytest.raises(FileExistsError):
        claim_once(marker, wrapper_commit="a" * 40, target_commit="b" * 40)
