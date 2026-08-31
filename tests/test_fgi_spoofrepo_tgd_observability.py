from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_fgi_spoofrepo_tgd_observability.py"
SPEC = importlib.util.spec_from_file_location("fgi_tgd_observability", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def config() -> dict:
    return json.loads(MODULE.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_exploratory_contract_is_valid() -> None:
    MODULE.validate_config(config())


def test_linear_clean_baseline_recovers_post_displacement() -> None:
    los_rows = np.asarray([
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
        [1.0, 1.0, 1.0], [-1.0, 1.0, -1.0],
    ], dtype=np.float64)
    los_rows[6:] /= np.linalg.norm(los_rows[6:], axis=1, keepdims=True)
    prns = set(range(1, 9))
    residuals = {}
    los = {}
    displacement = np.asarray([25.0, -40.0, 15.0])
    for bin_index in range(0, 20):
        for offset, prn in enumerate(sorted(prns)):
            clean_nuisance = 1000.0 + 3.0 * prn + (0.2 + 0.01 * prn) * (bin_index + 0.5)
            attack = 0.0 if bin_index < 10 else -float(los_rows[offset] @ displacement) + 35.0
            residuals[bin_index, prn] = clean_nuisance + attack
            los[bin_index, prn] = los_rows[offset]
    coefficients = MODULE.fit_linear_prn_baselines(residuals, prns, 0, 10)
    fitted = MODULE.fit_pseudorange_geometry(
        residuals, los, coefficients, start_s=0, end_s=20, minimum_prns=8
    )
    assert np.allclose(fitted[15].theta[:3], displacement, atol=1e-9)
    assert np.isclose(fitted[15].theta[3], 35.0, atol=1e-9)
    assert fitted[15].clock_centered_normalized_residual < 1e-20


def test_comparison_separates_good_absolute_and_bad_local_vectors() -> None:
    class Fit:
        def __init__(self, vector: list[float]) -> None:
            self.theta = np.asarray([*vector, 0.0])
            self.clock_centered_normalized_residual = 0.1

    row = MODULE.comparison_row(
        170,
        np.asarray([30.0, 40.0, 0.0]),
        Fit([30.0, 40.0, 0.0]),
        Fit([-40.0, 30.0, 0.0]),
        np.zeros(3),
        np.zeros(3),
    )
    assert np.isclose(row["pseudorange_nmea_direction_cosine"], 1.0)
    assert np.isclose(row["pseudorange_nmea_vector_error_m"], 0.0)
    assert np.isclose(row["tap_nmea_direction_cosine"], 0.0)
    assert row["tap_nmea_vector_error_m"] > 70.0
