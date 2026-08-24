#!/usr/bin/env python3
"""Independent compact verifier for committed SPLITCLOCK Stage-0A evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnss_doppler_lab.splitclock_observable_audit import verify_artifact


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("artifact", type=Path); args = parser.parse_args()
    errors = verify_artifact(args.artifact)
    verdict = json.loads((args.artifact / "final_verdict.json").read_text())
    access = json.loads((args.artifact / "access_audit.json").read_text())
    sign = json.loads((args.artifact / "sign_unit_validation.json").read_text())
    if verdict["verdict"] != "INCONCLUSIVE_SPLITCLOCK_EXECUTION_OR_PROVENANCE": errors.append("verdict")
    if verdict["base_sha"] != "a0e687de330a8ae1844e57f936aaf906144f693f": errors.append("base_sha")
    if verdict["design_freeze_sha"] != "8b7de5722037c1269989f2ee8cbff89ac42e3773": errors.append("design_sha")
    if access["attack"]["bytes_read"] or access["jammertest_raw"]["bytes_read"]: errors.append("forbidden_access")
    if access["score_operations"] != 0 or sign["status"] != "FAIL": errors.append("fail_closed")
    print(json.dumps({"status": "PASS" if not errors else "FAIL", "errors": errors}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
