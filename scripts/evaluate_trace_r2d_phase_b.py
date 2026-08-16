#!/usr/bin/env python3
"""R2d path adapter for the byte-identical frozen TRACE Phase-B scorer."""

import json
from pathlib import Path
import sys

import evaluate_trace_r2c_phase_b as inherited

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2d_oakbat_clean_support_repair"
SSD = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "trace-stage0-r2d-oakbat-clean-support-repair"
)
inherited.ARTIFACT = ARTIFACT
inherited.frozen.ARTIFACT = ARTIFACT
inherited.frozen.SSD = SSD
inherited.frozen.WORK = SSD / "evaluation_work"


def main() -> int:
    code = inherited.main()
    if len(sys.argv) > 1 and sys.argv[1] == "finalize":
        path = ARTIFACT / "final_verdict.json"
        verdict = json.loads(path.read_text())
        verdict["schema"] = "gnss-doppler-lab.trace-r2d-final-verdict.v1"
        verdict["engineering_repair_status"] = "OAKBAT_CLEAN_SUPPORT_REPAIRED"
        path.write_text(json.dumps(verdict, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
