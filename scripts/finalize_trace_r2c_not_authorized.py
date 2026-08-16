#!/usr/bin/env python3
"""Emit fail-closed R2c placeholders if any frozen Phase-A gate fails."""

import json
from pathlib import Path

import finalize_trace_r2a_not_authorized as inherited

ROOT = Path(__file__).resolve().parents[1]
inherited.ARTIFACT = ROOT / "artifacts/trace_stage0_r2c_terminal_drain_repair"
inherited.REASON = (
    "R2c Phase A did not pass every frozen source/causal/support/whole-row-set/"
    "terminal-count/semantic/block-key/alarm gate; Phase B is NOT_AUTHORIZED "
    "and no attack performance was read or computed."
)

if __name__ == "__main__":
    code = inherited.main()
    for path in inherited.ARTIFACT.glob("*.json"):
        value = json.loads(path.read_text())
        if isinstance(value.get("schema"), str):
            value["schema"] = value["schema"].replace("trace-r2a", "trace-r2c")
        if path.name == "final_verdict.json":
            value["next_action"] = (
                "Inspect the failed R2c receiver/input closure gate and repeat the same frozen "
                "Phase A without changing TRACE scoring, tolerances, windows, or gates."
            )
        path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    raise SystemExit(code)
