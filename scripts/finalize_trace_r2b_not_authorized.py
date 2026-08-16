#!/usr/bin/env python3
"""Emit fail-closed R2b placeholders when the frozen Phase A does not pass."""

import json
from pathlib import Path
import finalize_trace_r2a_not_authorized as inherited

ROOT = Path(__file__).resolve().parents[1]
inherited.ARTIFACT = ROOT / "artifacts/trace_stage0_r2b_stable_handoff_repair"
inherited.REASON = "R2b Phase A did not pass every preregistered source/causal/semantic/support gate; Phase B is NOT_AUTHORIZED and no attack performance was read or computed."

if __name__ == "__main__":
    code = inherited.main()
    for path in inherited.ARTIFACT.glob("*.json"):
        value = json.loads(path.read_text())
        if isinstance(value.get("schema"), str):
            value["schema"] = value["schema"].replace("trace-r2a", "trace-r2b")
            path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    raise SystemExit(code)
