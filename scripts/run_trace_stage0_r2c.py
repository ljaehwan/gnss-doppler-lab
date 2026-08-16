#!/usr/bin/env python3
"""TRACE-R2c replay driver with opt-in finite-source terminal draining."""

from pathlib import Path

import run_trace_stage0_r2a as driver

ROOT = Path(__file__).resolve().parents[1]
driver.ARTIFACT = ROOT / "artifacts/trace_stage0_r2c_terminal_drain_repair"
driver.SSD_ROOT = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2c-terminal-drain-repair"
)
driver.RECEIVER = driver.SSD_ROOT / "receiver-build/src/main/gnss-sdr"
driver.HANDOFF_ROOT = driver.ARTIFACT / "handoffs"
driver.RELEASE = "r2c"

_r2b_phase_a_config = driver.frozen_config_text
_r2b_phase_b_config = driver.frozen_phase_b_config_text


def frozen_config_text(name: str) -> str:
    return driver.set_config_values(
        _r2b_phase_a_config(name), {"SignalSource.enable_terminal_drain": "true"}
    )


def frozen_phase_b_config_text(name: str) -> str:
    return driver.set_config_values(
        _r2b_phase_b_config(name), {"SignalSource.enable_terminal_drain": "true"}
    )


driver.frozen_config_text = frozen_config_text
driver.frozen_phase_b_config_text = frozen_phase_b_config_text


if __name__ == "__main__":
    raise SystemExit(driver.main())
