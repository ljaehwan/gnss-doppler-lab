from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_cgc_locked_phase_root_cause.py"
SPEC = importlib.util.spec_from_file_location("audit_cgc_locked_phase_root_cause", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_quadrature_fraction_distinguishes_in_phase_from_quadrature_profile() -> None:
    in_phase = np.asarray([[0.2, 0.5, 1.0, 0.5, 0.2]], dtype=np.complex128)
    quadrature = in_phase.copy()
    quadrature[0, 0] += 0.4j

    assert MODULE.quadrature_fraction(in_phase, prompt_index=2)[0] == pytest.approx(0.0)
    assert MODULE.quadrature_fraction(quadrature, prompt_index=2)[0] > 0.0


def test_centered_truth_agreement_ignores_clock_and_global_sign() -> None:
    truth = np.asarray([-0.3, -0.1, 0.2, 0.4])
    estimated = -2.0 * truth + 0.17

    result = MODULE.centered_truth_agreement(estimated, truth)

    assert result["absolute_centered_correlation"] == pytest.approx(1.0)
    assert result["truth_direction_r2"] == pytest.approx(1.0)
    assert result["signed_affine_slope"] == pytest.approx(-2.0)
    assert result["estimated_to_truth_spread_ratio"] == pytest.approx(2.0)
