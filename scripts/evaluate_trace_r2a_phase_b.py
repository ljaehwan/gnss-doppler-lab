#!/usr/bin/env python3
"""R2a input adapter for the frozen TRACE-R2 Phase-B scorer.

Only artifact and dump roots are redirected.  Predictor, features, whitening,
score, thresholds, controls, action shuffle, statistics, and verdict gates are
the frozen implementation in ``evaluate_trace_r2_phase_b.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import evaluate_trace_r2_phase_b as frozen


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2a_reproducibility_repair"
SSD = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2a-reproducibility-repair")


def main() -> int:
    frozen.ARTIFACT = ARTIFACT
    frozen.SSD = SSD
    frozen.WORK = SSD / "evaluation_work"
    if len(sys.argv) > 1 and sys.argv[1] == "finalize":
        phase_a = json.loads((ARTIFACT / "rep3_rep4_reproduction_metrics.json").read_text())
        (ARTIFACT / "smoke_replay_results.json").write_text(
            json.dumps(phase_a, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    return frozen.main()


if __name__ == "__main__":
    raise SystemExit(main())
