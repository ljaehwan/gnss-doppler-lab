#!/usr/bin/env python3
"""MOSAIC Stage-0B runner.

Stage-0B is intentionally fail-closed unless Stage-0A and navigation-bit
provenance are already present in the artifact bundle.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/mosaic_stage0ab_foundation"


def main() -> int:
    verdict_path = ART / "final_verdict.json"
    if not verdict_path.exists():
        print("Stage-0A artifacts missing; run scripts/run_mosaic_stage0a.py first", file=sys.stderr)
        return 2
    verdict = json.loads(verdict_path.read_text())
    if verdict.get("stage0a_pass") is not True:
        print(json.dumps({"status": "NOT_RUN", "reason": "Stage-0A did not pass; Stage-0B receiver replay not authorized", "final_verdict": verdict.get("verdict")}, indent=2))
        return 0
    field = json.loads((ART / "receiver_field_contract.json").read_text())
    if field.get("navigation_bit_provenance", {}).get("status") != "PASS":
        verdict["verdict"] = "INCONCLUSIVE_NAVIGATION_BIT_PROVENANCE"
        verdict["go"] = False
        verdict["stage0b_run"] = False
        verdict["reason"] = "navigation-bit provenance unavailable; refusing +1 fallback"
        verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return 0
    print(json.dumps({"status": "NOT_IMPLEMENTED_FOR_CURRENT_INPUTS", "reason": "Receiver-in-loop injection requires validated nav bits and Stage-0A PASS"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
