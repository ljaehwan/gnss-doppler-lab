#!/usr/bin/env python3
"""R2d receiver driver preserving R2c drain semantics with repaired clean support."""

from pathlib import Path

import run_trace_stage0_r2c as r2c

ROOT = Path(__file__).resolve().parents[1]
driver = r2c.driver
driver.ARTIFACT = ROOT / "artifacts/trace_stage0_r2d_oakbat_clean_support_repair"
driver.SSD_ROOT = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "trace-stage0-r2d-oakbat-clean-support-repair"
)
driver.RECEIVER = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "trace-stage0-r2c-terminal-drain-repair/receiver-build/src/main/gnss-sdr"
)
driver.HANDOFF_ROOT = driver.ARTIFACT / "handoffs"
driver.RELEASE = "r2d"
driver.PHASE_B_SCENARIOS["OAKBAT.cleanStatic"]["phase_b_skip_s"] = 0.0
driver.PHASE_B_SCENARIOS["OAKBAT.cleanStatic"]["phase_b_handoff"] = (
    "oakbat_cleanstatic.csv"
)


def frozen_config_text(name: str) -> str:
    return r2c.frozen_config_text(name)


def frozen_phase_b_config_text(name: str) -> str:
    return r2c.frozen_phase_b_config_text(name)


driver.frozen_config_text = frozen_config_text
driver.frozen_phase_b_config_text = frozen_phase_b_config_text


if __name__ == "__main__":
    raise SystemExit(driver.main())
