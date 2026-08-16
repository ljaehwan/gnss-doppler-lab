#!/usr/bin/env python3
"""R2d path adapter for the unchanged R2c Phase-A evaluator and terminal gates."""

from pathlib import Path

import evaluate_trace_r2c_phase_a as inherited

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2d_oakbat_clean_support_repair"
SSD = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "trace-stage0-r2d-oakbat-clean-support-repair"
)
inherited.ARTIFACT = ARTIFACT
inherited.SSD = SSD
inherited.evaluator.ARTIFACT = ARTIFACT
inherited.evaluator.SSD = SSD


if __name__ == "__main__":
    raise SystemExit(inherited.main())
