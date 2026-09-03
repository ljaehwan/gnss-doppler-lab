#!/usr/bin/env python3
"""Validate the WCL CGC claim-to-code manifest without running experiments."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/paper/wcl_cgc_v1_manifest.json"


def _relative_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"repository path must be relative and contained: {value!r}")
    return ROOT.joinpath(*pure.parts)


def _tracked_paths() -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return {
        item.decode("utf-8")
        for item in completed.stdout.split(b"\0")
        if item
    }


def audit_manifest(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema") != "gnss-doppler-lab.paper-code-map":
        raise ValueError("unexpected manifest schema")
    if document.get("schema_version") != 1:
        raise ValueError("unexpected manifest schema version")
    allowed = set(document.get("classifications", []))
    required_classes = {
        "paper_core",
        "paper_support",
        "development_only",
        "negative_or_failed",
        "superseded",
        "legacy_other_direction",
        "shared_infrastructure",
        "uncommitted_review",
    }
    if allowed != required_classes:
        raise ValueError("classification vocabulary changed unexpectedly")

    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest entries must be a nonempty list")
    tracked = _tracked_paths()
    claim_ids: set[str] = set()
    checked_files = 0
    missing: list[str] = []
    untracked_required: list[str] = []
    class_counts = {name: 0 for name in sorted(allowed)}

    for entry in entries:
        claim_id = entry.get("claim_id")
        classification = entry.get("classification")
        if not isinstance(claim_id, str) or not claim_id or claim_id in claim_ids:
            raise ValueError(f"invalid or duplicate claim_id: {claim_id!r}")
        claim_ids.add(claim_id)
        if classification not in allowed:
            raise ValueError(f"invalid classification for {claim_id}: {classification!r}")
        class_counts[classification] += 1
        for relative in entry.get("tracked_files", []):
            path = _relative_path(relative)
            checked_files += 1
            if not path.is_file():
                missing.append(relative)
            if relative not in tracked:
                untracked_required.append(relative)
        for artifact in entry.get("local_artifacts", []):
            if not isinstance(artifact, dict) or "path" not in artifact:
                raise ValueError(f"invalid local artifact in {claim_id}")
            value = str(artifact["path"])
            path = Path(value) if Path(value).is_absolute() else _relative_path(value)
            if artifact.get("required_for_manifest_audit", False) and not path.exists():
                missing.append(value)

    review = document.get("uncommitted_review", [])
    review_missing: list[str] = []
    review_now_tracked: list[str] = []
    for item in review:
        relative = item.get("path")
        if not isinstance(relative, str):
            raise ValueError("uncommitted_review path must be a string")
        path = _relative_path(relative)
        if not path.exists():
            review_missing.append(relative)
        if relative in tracked:
            review_now_tracked.append(relative)
    class_counts["uncommitted_review"] = len(review)

    return {
        "status": "pass" if not missing and not untracked_required else "fail",
        "claim_count": len(claim_ids),
        "tracked_file_references_checked": checked_files,
        "classification_counts": class_counts,
        "missing_required_paths": sorted(set(missing)),
        "required_paths_not_git_tracked": sorted(set(untracked_required)),
        "review_queue_paths_missing": sorted(set(review_missing)),
        "review_queue_paths_now_tracked": sorted(set(review_now_tracked)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    document = json.loads(manifest.read_text(encoding="utf-8"))
    summary = audit_manifest(document)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
