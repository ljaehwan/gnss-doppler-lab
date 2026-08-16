#!/usr/bin/env python3
"""R2b path-only adapter for the frozen TRACE-R2 Phase-B scorer."""

from pathlib import Path
import sys
import evaluate_trace_r2_phase_b as frozen

ROOT = Path(__file__).resolve().parents[1]
frozen.ARTIFACT = ROOT / "artifacts/trace_stage0_r2b_stable_handoff_repair"
frozen.SSD = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2b-stable-handoff-repair")
frozen.WORK = frozen.SSD / "evaluation_work"

if __name__ == "__main__":
    raise SystemExit(frozen.main())
