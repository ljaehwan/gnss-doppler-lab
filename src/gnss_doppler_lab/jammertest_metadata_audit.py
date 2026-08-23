"""Deterministic helpers for the Jammertest 2025 metadata-only audit.

This module deliberately has no HDF5, NumPy payload, Git-LFS, model, or
training dependency.  It operates only on pointer text and released split
metadata.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LFS_POINTER_RE = re.compile(
    r"\Aversion https://git-lfs\.github\.com/spec/v1\n"
    r"oid sha256:([0-9a-f]{64})\n"
    r"size ([0-9]+)\n?\Z"
)

ALLOWED_VERDICTS = {
    "READY_FOR_CRPA_MINIMAL_SUBSET_DOWNLOAD",
    "INCONCLUSIVE_SCHEMA_REQUIRES_ONE_BOUNDED_H5_SAMPLE",
    "NO_GO_NOT_PHASE_COHERENT_ARRAY",
    "NO_GO_LABEL_OR_DOMAIN_CONFOUND",
    "NO_GO_SPATIAL_IDENTIFIABILITY",
    "NO_GO_WCL_NOVELTY",
}

REQUIRED_ARTIFACTS = {
    "README.md",
    "official_source_binding.json",
    "repository_inventory.json",
    "lfs_pointer_inventory.csv",
    "logical_size_summary.json",
    "crpa_schema_audit.json",
    "array_phase_coherence_evidence.md",
    "label_distribution_audit.json",
    "campaign_alignment_audit.json",
    "shortcut_confound_matrix.csv",
    "spatial_identifiability.md",
    "literature_review.md",
    "literature_sources.json",
    "sparc_candidate_spec.md",
    "leakage_safe_split_plan.json",
    "destruction_controls.json",
    "minimal_download_plan.json",
    "ssd_capacity_audit.json",
    "access_audit.json",
    "final_verdict.json",
    "artifact_manifest_sha256.json",
    "verifier_output.txt",
    "test_output.txt",
}


@dataclass(frozen=True)
class LfsPointer:
    oid_sha256: str
    size: int


def parse_lfs_pointer(text: str) -> LfsPointer:
    """Parse a canonical Git-LFS pointer, failing closed on all other bytes."""

    match = LFS_POINTER_RE.fullmatch(text)
    if match is None:
        raise ValueError("not a canonical Git-LFS pointer")
    size = int(match.group(2))
    if size <= 0:
        raise ValueError("Git-LFS logical size must be positive")
    return LfsPointer(match.group(1), size)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_crpa_rows(paths: Iterable[tuple[Path, int, str]]) -> list[dict]:
    rows: list[dict] = []
    for path, area, split in paths:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            fields = line.split("\t")
            if len(fields) != 4:
                raise ValueError(f"{path}:{line_number}: expected four columns")
            rows.append(
                {
                    "sample_index": int(fields[0]),
                    "class_id": int(fields[1]),
                    "transmit_power_dbm": float(fields[2]),
                    "bandwidth_mhz": float(fields[3]),
                    "area": area,
                    "split": split,
                }
            )
    return rows


def counter_records(counter: Counter, key_name: str) -> list[dict]:
    return [
        {key_name: key, "count": count}
        for key, count in sorted(counter.items(), key=lambda item: item[0])
    ]


def infer_crpa_npy_layout(pointer_size: int, max_sample_index: int) -> dict:
    """Record the unique size-consistent layout hypothesis without opening NPY.

    The reader fixes four complex channels and 1024 samples per snapshot.  A
    NumPy v1 header is normally 128 bytes for this shape/dtype.  This function
    only establishes arithmetic consistency; it does not assert that the raw
    NPY header or dtype has been observed.
    """

    snapshot_count = max_sample_index + 1
    payload_bytes = snapshot_count * 4 * 1024 * 8
    header_bytes = pointer_size - payload_bytes
    return {
        "candidate_shape": [snapshot_count, 4, 1024],
        "candidate_dtype": "complex64",
        "bytes_per_complex_value": 8,
        "bytes_per_snapshot": 4 * 1024 * 8,
        "candidate_payload_bytes": payload_bytes,
        "candidate_npy_header_bytes": header_bytes,
        "size_exactly_consistent": header_bytes == 128,
        "status": "INFERRED_NOT_DIRECTLY_OBSERVED",
    }


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_manifest(artifact_dir: Path) -> dict:
    excluded = {
        "artifact_manifest_sha256.json",
        "verifier_output.txt",
        "test_output.txt",
    }
    entries = {
        path.relative_to(artifact_dir).as_posix(): sha256_file(path)
        for path in sorted(artifact_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    payload = {
        "algorithm": "sha256",
        "excluded_self_and_mutable_logs": sorted(excluded),
        "files": entries,
    }
    write_json(artifact_dir / "artifact_manifest_sha256.json", payload)
    return payload


def verify_artifact(artifact_dir: Path, *, require_logs: bool = True) -> list[str]:
    errors: list[str] = []
    required = REQUIRED_ARTIFACTS if require_logs else REQUIRED_ARTIFACTS - {
        "verifier_output.txt",
        "test_output.txt",
    }
    missing = sorted(name for name in required if not (artifact_dir / name).is_file())
    if missing:
        errors.append(f"missing required artifacts: {missing}")

    try:
        manifest = read_json(artifact_dir / "artifact_manifest_sha256.json")
        for relative, expected in manifest["files"].items():
            path = artifact_dir / relative
            if not path.is_file():
                errors.append(f"manifest target missing: {relative}")
            elif sha256_file(path) != expected:
                errors.append(f"manifest hash mismatch: {relative}")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"invalid manifest: {exc}")

    try:
        with (artifact_dir / "lfs_pointer_inventory.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            pointer_rows = list(csv.DictReader(handle))
        if len(pointer_rows) != 231:
            errors.append(f"pointer count {len(pointer_rows)} != 231")
        total = sum(int(row["logical_size_bytes"]) for row in pointer_rows)
        if total != 360_208_806_569:
            errors.append(f"pointer size sum {total} != 360208806569")
        crpa = [row for row in pointer_rows if row["path"] == "all_crpa_files.npy"]
        if len(crpa) != 1:
            errors.append("CRPA pointer inventory is not unique")
        elif (
            crpa[0]["oid_sha256"]
            != "d869fa20d552288002e4d2a5b6c5d1300083a6348c01a956cd6a34ff232e0a3f"
            or int(crpa[0]["logical_size_bytes"]) != 1_398_308_992
        ):
            errors.append("CRPA pointer binding mismatch")
    except (OSError, KeyError, ValueError) as exc:
        errors.append(f"invalid pointer inventory: {exc}")

    try:
        access = read_json(artifact_dir / "access_audit.json")
        for field in (
            "git_lfs_payload_bytes_downloaded",
            "raw_hdf5_bytes_downloaded",
            "raw_iq_bytes_opened",
            "tuni_raw_payload_bytes_accessed",
            "texbat_raw_payload_bytes_accessed",
            "oakbat_raw_payload_bytes_accessed",
        ):
            if access[field] != 0:
                errors.append(f"forbidden access counter nonzero: {field}")
        if access["models_implemented"] or access["scores_computed"]:
            errors.append("forbidden model/score activity recorded")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"invalid access audit: {exc}")

    try:
        verdict = read_json(artifact_dir / "final_verdict.json")
        name = verdict["verdict"]
        if name not in ALLOWED_VERDICTS:
            errors.append(f"unrecognized verdict: {name}")
        gates = verdict["gates"]
        ready = all(gates.values())
        if (name == "READY_FOR_CRPA_MINIMAL_SUBSET_DOWNLOAD") != ready:
            errors.append("verdict contradicts gate conjunction")
        if name == "INCONCLUSIVE_SCHEMA_REQUIRES_ONE_BOUNDED_H5_SAMPLE":
            if gates["direct_relative_phase_preservation_evidence"]:
                errors.append("inconclusive schema verdict contradicts phase gate")
            if verdict["raw_download_authorized"]:
                errors.append("inconclusive verdict authorizes raw download")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"invalid final verdict: {exc}")

    try:
        plan = read_json(artifact_dir / "minimal_download_plan.json")
        if plan["current_authorization"] != "NOT_AUTHORIZED":
            errors.append("minimal plan authorizes a download before closure")
        obj = plan["single_bounded_object_required_for_next_schema_step"]
        if obj["oid_sha256"] != (
            "d869fa20d552288002e4d2a5b6c5d1300083a6348c01a956cd6a34ff232e0a3f"
        ) or obj["logical_size_bytes"] != 1_398_308_992:
            errors.append("bounded next object is not the frozen CRPA object")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"invalid minimal download plan: {exc}")

    return errors
