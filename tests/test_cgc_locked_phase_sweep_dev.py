from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_cgc_locked_phase_sweep_dev.py"
SPEC = importlib.util.spec_from_file_location("run_cgc_locked_phase_sweep_dev", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_phase_tag_normalizes_degrees() -> None:
    assert MODULE.phase_tag(90.0) == "phase-0090p0deg"
    assert MODULE.phase_tag(450.0) == "phase-0090p0deg"


def test_phase_composer_rejects_nonfinite_offset(tmp_path: Path) -> None:
    authentic = tmp_path / "auth.bin"
    counterfeit = tmp_path / "spoof.bin"
    authentic.write_bytes(bytes([1, 0, 1, 0]))
    counterfeit.write_bytes(bytes([1, 0, 1, 0]))
    with pytest.raises(ValueError, match="phase_offset_rad"):
        MODULE.compose_phase_shifted_locked_iq(
            authentic,
            counterfeit,
            tmp_path / "out.bin",
            phase_offset_rad=float("nan"),
            sample_rate_hz=1,
            receiver=MODULE.ImpairmentConfig(),
            reference={"complex_samples": 2},
            event=MODULE.SpoofEvent(1.0, 1.0, (1.0, 0.0, 0.0), -20.0, 0.0, 1.0),
        )
