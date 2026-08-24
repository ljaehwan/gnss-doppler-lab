#!/usr/bin/env python3
"""Run the frozen SPLITCLOCK R2 clean-only experiment exactly once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gnss_doppler_lab.splitclock_r2_experiment import (  # noqa: E402
    GeometryUnavailable,
    execute,
    finalize_manifest,
    write_failure_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "artifact",
        "c1-raw",
        "c3-raw",
        "c1-output",
        "c3-output",
        "receiver",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--implementation-sha", required=True)
    parser.add_argument("--rehash-raw", action="store_true")
    args = parser.parse_args()

    try:
        final = execute(
            args.artifact,
            {"C-1": args.c1_raw, "C-3": args.c3_raw},
            {"C-1": args.c1_output, "C-3": args.c3_output},
            args.receiver,
            args.implementation_sha,
            args.rehash_raw,
        )
    except GeometryUnavailable as exc:
        final = write_failure_artifact(
            args.artifact,
            "STOP_SPLITCLOCK_R2_GEOMETRY_OR_PANEL",
            str(exc),
            args.implementation_sha,
            59_999_664_000 if args.rehash_raw else 0,
        )
    except Exception as exc:
        final = write_failure_artifact(
            args.artifact,
            "INCONCLUSIVE_SPLITCLOCK_R2_EXECUTION_OR_PROVENANCE",
            f"{type(exc).__name__}: {exc}",
            args.implementation_sha,
            59_999_664_000 if args.rehash_raw else 0,
        )
    finalize_manifest(args.artifact)
    print(json.dumps(final, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
