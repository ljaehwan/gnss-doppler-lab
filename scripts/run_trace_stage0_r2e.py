#!/usr/bin/env python3
"""R2e receiver driver preserving R2c/R2d repairs with attack handoffs."""

from pathlib import Path

import run_trace_stage0_r2d as r2d

ROOT = Path(__file__).resolve().parents[1]
driver = r2d.driver
driver.ARTIFACT = ROOT / "artifacts/trace_stage0_r2e_attack_support_repair"
driver.SSD_ROOT = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "trace-stage0-r2e-attack-support-repair"
)
driver.HANDOFF_ROOT = driver.SSD_ROOT / "handoffs"
driver.RELEASE = "r2e"
driver.PHASE_B_SCENARIOS["TEXBAT.DS7"]["phase_b_handoff"] = "texbat_ds7.csv"
driver.PHASE_B_SCENARIOS["OAKBAT.OS4"]["phase_b_handoff"] = "oakbat_os4.csv"


def frozen_config_text(name: str) -> str:
    return r2d.frozen_config_text(name)


def frozen_phase_b_config_text(name: str) -> str:
    return r2d.frozen_phase_b_config_text(name)


driver.frozen_config_text = frozen_config_text
driver.frozen_phase_b_config_text = frozen_phase_b_config_text


if __name__ == "__main__":
    raise SystemExit(driver.main())
