from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from gnss_doppler_lab.jammertest_metadata_audit import (
    infer_crpa_npy_layout,
    parse_lfs_pointer,
    verify_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/jammertest2025_crpa_stage0_metadata_feasibility"


def test_pointer_parser_accepts_canonical_pointer() -> None:
    pointer = parse_lfs_pointer(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "a" * 64 + "\n"
        "size 12345\n"
    )
    assert pointer.oid_sha256 == "a" * 64
    assert pointer.size == 12345


@pytest.mark.parametrize(
    "mutation",
    [
        "version https://git-lfs.github.com/spec/v1\noid sha256:xyz\nsize 1\n",
        "version https://git-lfs.github.com/spec/v1\noid sha256:" + "a" * 64 + "\nsize 0\n",
        "payload bytes",
    ],
)
def test_pointer_parser_fails_closed(mutation: str) -> None:
    with pytest.raises(ValueError):
        parse_lfs_pointer(mutation)


def test_crpa_shape_arithmetic_is_inference_not_observation() -> None:
    layout = infer_crpa_npy_layout(1_398_308_992, 42_672)
    assert layout["candidate_shape"] == [42_673, 4, 1024]
    assert layout["candidate_dtype"] == "complex64"
    assert layout["candidate_npy_header_bytes"] == 128
    assert layout["size_exactly_consistent"] is True
    assert layout["status"] == "INFERRED_NOT_DIRECTLY_OBSERVED"


def test_committed_artifact_verifies() -> None:
    assert verify_artifact(ARTIFACT) == []


def test_gate_and_verdict_are_consistent() -> None:
    verdict = json.loads((ARTIFACT / "final_verdict.json").read_text())
    assert verdict["verdict"] == "INCONCLUSIVE_SCHEMA_REQUIRES_ONE_BOUNDED_H5_SAMPLE"
    assert verdict["gates"]["direct_relative_phase_preservation_evidence"] is False
    assert verdict["raw_download_authorized"] is False
    assert verdict["model_implementation_authorized"] is False


def test_access_audit_is_zero() -> None:
    audit = json.loads((ARTIFACT / "access_audit.json").read_text())
    assert audit["git_lfs_payload_bytes_downloaded"] == 0
    assert audit["raw_hdf5_bytes_downloaded"] == 0
    assert audit["raw_iq_bytes_opened"] == 0
    assert audit["models_implemented"] is False
    assert audit["scores_computed"] is False


def test_manifest_detects_artifact_byte_flip(tmp_path: Path) -> None:
    copied = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, copied)
    readme = copied / "README.md"
    readme.write_bytes(readme.read_bytes() + b"X")
    assert any("manifest hash mismatch: README.md" in error for error in verify_artifact(copied))


def test_ready_verdict_tamper_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, copied)
    verdict_path = copied / "final_verdict.json"
    verdict = json.loads(verdict_path.read_text())
    verdict["verdict"] = "READY_FOR_CRPA_MINIMAL_SUBSET_DOWNLOAD"
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    errors = verify_artifact(copied)
    assert any("verdict contradicts gate conjunction" in error for error in errors)


def test_bounded_object_tamper_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / "artifact"
    shutil.copytree(ARTIFACT, copied)
    plan_path = copied / "minimal_download_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["single_bounded_object_required_for_next_schema_step"]["logical_size_bytes"] += 1
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    errors = verify_artifact(copied)
    assert any("bounded next object" in error for error in errors)
