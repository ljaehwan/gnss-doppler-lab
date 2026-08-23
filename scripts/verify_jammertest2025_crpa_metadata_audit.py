#!/usr/bin/env python3
"""Verify the committed Jammertest 2025 metadata-only artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from gnss_doppler_lab.jammertest_metadata_audit import verify_artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact_dir",
        nargs="?",
        type=Path,
        default=Path("artifacts/jammertest2025_crpa_stage0_metadata_feasibility"),
    )
    parser.add_argument("--allow-missing-logs", action="store_true")
    args = parser.parse_args()
    errors = verify_artifact(args.artifact_dir, require_logs=not args.allow_missing_logs)
    if errors:
        print("JAMMERTEST2025_CRPA_METADATA_AUDIT_VERIFY_FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("JAMMERTEST2025_CRPA_METADATA_AUDIT_VERIFY_PASS")
    print("verdict=INCONCLUSIVE_SCHEMA_REQUIRES_ONE_BOUNDED_H5_SAMPLE")
    print("raw_iq_access_bytes=0")
    print("lfs_payload_download_bytes=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
