#!/usr/bin/env python3
"""Create the two-file authorization wrapper after independent target approval.

This finalizer does not claim a marker or access protected data.  It is not run
while constructing target A.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.gcspo_successor_launch import (  # noqa: E402
    AUTHORIZATION_RELATIVE, CONTROL_RELATIVE, RECEIPT_RELATIVE,
    build_review_authorization_documents, strict_json_bytes, verify_launch_target,
)


def _write_exclusive(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--review-command", required=True)
    parser.add_argument("--passed", required=True, type=int)
    parser.add_argument("--evidence-sha256", required=True)
    args = parser.parse_args()
    target = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            text=True, capture_output=True).stdout.strip()
    verify_launch_target(ROOT, target)
    control = strict_json_bytes((ROOT / CONTROL_RELATIVE).read_bytes(), "launch control")
    receipt, execution = build_review_authorization_documents(
        control, target_commit=target, reviewer=args.reviewer,
        review_command=args.review_command, passed=args.passed, findings=[],
        evidence_sha256=args.evidence_sha256, repo=ROOT,
    )
    _write_exclusive(ROOT / RECEIPT_RELATIVE, receipt)
    try:
        _write_exclusive(ROOT / AUTHORIZATION_RELATIVE, execution)
    except Exception:
        (ROOT / RECEIPT_RELATIVE).unlink()
        raise
    print(f"AUTHORIZATION_WRAPPER_DOCUMENTS_READY target={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
