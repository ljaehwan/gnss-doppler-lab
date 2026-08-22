#!/usr/bin/env python3
"""Prepare and execute the frozen exploratory CRID R4d TEXBAT DS3 audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.crid_r4d_ds3 import (  # noqa: E402
    ARTIFACT_REL,
    SSD_ROOT,
    analyze,
    audit_r4c_c0,
    inventory_ds3,
    prepare_execution_repair_freeze,
    run_replays,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-repair-freeze")
    for name in ("audit-c0", "inventory", "replay", "analyze"):

        subparser = subparsers.add_parser(name)
        subparser.add_argument("--freeze-sha", required=True)
    args = parser.parse_args()
    artifact = ROOT / ARTIFACT_REL
    if args.command == "prepare-repair-freeze":
        result = prepare_execution_repair_freeze(ROOT, artifact)
    elif args.command == "audit-c0":
        result = audit_r4c_c0(ROOT, artifact, SSD_ROOT, args.freeze_sha)
    elif args.command == "inventory":
        result = inventory_ds3(ROOT, artifact, SSD_ROOT, args.freeze_sha)
    elif args.command == "replay":
        result = run_replays(ROOT, artifact, SSD_ROOT, args.freeze_sha)
    else:
        result = analyze(ROOT, artifact, SSD_ROOT, args.freeze_sha)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
