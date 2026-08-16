#!/usr/bin/env python3
"""TRACE-R2b replay driver using the frozen R2a scenario contract."""

from pathlib import Path
import run_trace_stage0_r2a as driver

ROOT = Path(__file__).resolve().parents[1]
driver.ARTIFACT = ROOT / "artifacts/trace_stage0_r2b_stable_handoff_repair"
driver.SSD_ROOT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2b-stable-handoff-repair")
driver.RECEIVER = driver.SSD_ROOT / "receiver-build/src/main/gnss-sdr"
driver.HANDOFF_ROOT = driver.ARTIFACT / "handoffs"
driver.RELEASE = "r2b"

if __name__ == "__main__":
    raise SystemExit(driver.main())
