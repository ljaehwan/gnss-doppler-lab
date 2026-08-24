#!/usr/bin/env python3
"""Independent compact verifier for committed SPLITCLOCK R2 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gnss_doppler_lab.splitclock_r2_experiment import verify_artifact  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    artifact = args.artifact or (
        REPO / "artifacts/splitclock_stage0a_r2_terminal_contract_repair"
    )
    errors = verify_artifact(artifact, REPO)
    result = {"status": "PASS" if not errors else "FAIL", "errors": errors}
    print(json.dumps(result, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
