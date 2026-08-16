#!/usr/bin/env python3
"""Run the unchanged frozen R2a Phase-A evaluator against R2b paths."""

from pathlib import Path
import evaluate_trace_r2a_phase_a as evaluator

ROOT = Path(__file__).resolve().parents[1]
evaluator.ARTIFACT = ROOT / "artifacts/trace_stage0_r2b_stable_handoff_repair"
evaluator.SSD = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2b-stable-handoff-repair")

if __name__ == "__main__":
    raise SystemExit(evaluator.main())
