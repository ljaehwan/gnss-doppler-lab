"""Source-byte-bound closed-loop physical loading and VAR transfer proof."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .gcspo_core import apply_var_transfer, build_physical_loading

DISCRIMINATOR = "src/algorithms/tracking/libs/tracking_discriminators.cc"
TRACKING_LOOP = "src/algorithms/tracking/gnuradio_blocks/dll_pll_veml_tracking.cc"
ROWS = ("code_error_chips", "pll_phase_error_cycles", "carrier_doppler_hz", "code_frequency_offset_chips_s")
RANGE_ROWS = {"code_error_chips", "pll_phase_error_cycles"}
RATE_ROWS = {"carrier_doppler_hz", "code_frequency_offset_chips_s"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float64).tobytes()).hexdigest()


def method_availability(validated_rows):
    rows = set(validated_rows)
    a3 = bool(rows & RANGE_ROWS); a4 = bool(rows & RATE_ROWS)
    code_family = bool(rows & {"code_error_chips", "code_frequency_offset_chips_s"})
    carrier_family = bool(rows & {"pll_phase_error_cycles", "carrier_doppler_hz"})
    return {"A3": a3, "A4": a4, "Full": a3 and a4 and code_family and carrier_family}


def _oracle_transfer(direct, coefficients):
    """Independent explicit recurrence oracle; zero prehistory after reset."""
    result = np.zeros_like(direct)
    for epoch in range(len(direct)):
        result[epoch] = direct[epoch]
        for lag in range(1, len(coefficients) + 1):
            if epoch >= lag:
                result[epoch] -= coefficients[lag - 1].dot(direct[epoch - lag])
    return result


def _vectors():
    impulse = np.zeros((50, 8)); impulse[10, 3] = .01
    ramp = np.zeros((50, 8))
    for epoch in range(10, 50):
        ramp[epoch, 7] = .001
        ramp[epoch, 3] = (ramp[epoch - 1, 3] if epoch else 0.) + .001 * .02
    return {"range_impulse": impulse, "rate_ramp": ramp}


def prove_closed_loop_transfer(source_root, model, *, expected_source_hashes, expected_var_sha256):
    root = Path(source_root)
    paths = {relative: root / relative for relative in (DISCRIMINATOR, TRACKING_LOOP)}
    observed_hashes = {relative: _sha(path) if path.is_file() else None for relative, path in paths.items()}
    hash_ok = observed_hashes == {key: value.lower() for key, value in expected_source_hashes.items()}
    discriminator = paths[DISCRIMINATOR].read_text(errors="strict") if paths[DISCRIMINATOR].is_file() else ""
    loop = paths[TRACKING_LOOP].read_text(errors="strict") if paths[TRACKING_LOOP].is_file() else ""
    equation_ok = ("double dll_nc_e_minus_l_normalized(" in discriminator
                   and "d_code_error_chips = dll_nc_e_minus_l_normalized(" in loop)
    cadence_ok = loop.count("d_correlation_length_ms = 1;") >= 1 and "d_correlation_length_ms = 2;" not in loop
    sign_ok = "d_carr_phase_error_hz = pll_cloop_two_quadrant_atan(d_P_accu) / TWO_PI;" in loop
    wrap_ok = ("return gr::fast_atan2f(cross, dot) / (t2 - t1);" in discriminator
               and "return gr::fast_atan2f(prompt_s1.imag(), prompt_s1.real());" in discriminator)
    loop_ok = ("d_carrier_doppler_hz = d_carr_error_filt_hz;" in loop
               and "d_code_freq_chips = d_code_chip_rate - d_code_error_filt_chips;" in loop
               and "d_code_freq_chips += d_carrier_doppler_hz * d_code_chip_rate / d_signal_carrier_freq;" in loop)
    var_ok = _array_sha(model.coefficients) == expected_var_sha256
    numeric_wrap = np.asarray([np.arctan2(np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)) / (2 * np.pi)
                               for x in (.249999, -.249999)])
    wrap_ok = wrap_ok and np.allclose(numeric_wrap, [.249999, -.249999], rtol=0, atol=1e-12)
    checks = {"source_hash": "PASS" if hash_ok else "FAIL", "equation": "PASS" if equation_ok else "FAIL",
              "cadence": "PASS" if cadence_ok else "FAIL", "sign": "PASS" if sign_ok else "FAIL",
              "wrap_linear_range": "PASS" if wrap_ok else "FAIL", "loop_path": "PASS" if loop_ok else "FAIL",
              "var_coefficient_hash": "PASS" if var_ok else "FAIL"}
    prerequisites = all(value == "PASS" for value in checks.values())
    loading = build_physical_loading(np.asarray([1., 0., 0.]), validated_rows=set(ROWS))
    vectors = _vectors(); row_reports = []
    for row_index, row_name in zip((6, 7, 8, 9), ROWS):
        errors, signs = [], []
        for vector_name, states in vectors.items():
            direct = np.einsum("qi,ti->tq", loading, states)
            implementation = apply_var_transfer(direct[:, :, None], model.coefficients)[:, :, 0]
            oracle = _oracle_transfer(direct, model.coefficients)
            errors.append(float(np.max(np.abs(implementation[:, row_index] - oracle[:, row_index]))))
            if (row_name in RANGE_ROWS and vector_name == "range_impulse") or (row_name in RATE_ROWS and vector_name == "rate_ramp"):
                nonzero = direct[:, row_index][np.abs(direct[:, row_index]) > 0]
                signs.append(bool(len(nonzero) and np.all(nonzero < 0)))
        passed = prerequisites and max(errors) <= 1e-12 and all(signs)
        row_reports.append({"row": row_name, "status": "PASS" if passed else "FAIL",
                            "maximum_oracle_error": max(errors), "source_sign_negative": all(signs),
                            "unproved_loading": 0.0 if not passed else None})
    validated = [row["row"] for row in row_reports if row["status"] == "PASS"]
    availability = method_availability(validated)
    overall = prerequisites and len(validated) == 4 and availability["Full"]
    return {"schema": "gnss-doppler-lab.gcspo-stage0.closed-loop-transfer-preflight.v1",
            "overall_status": "PASS" if overall else "FAIL", "checks": checks,
            "expected_source_hashes": expected_source_hashes, "observed_source_hashes": observed_hashes,
            "expected_var_sha256": expected_var_sha256, "observed_var_sha256": _array_sha(model.coefficients),
            "vector_contract": {"epochs": 50, "range_impulse_epoch": 10, "rate_ramp_start_epoch": 10},
            "analytic_tolerance": {"atol": 1e-12, "rtol": 1e-9}, "var_transfer_application_count": 1,
            "rows": row_reports, "validated_rows": validated, "method_availability": availability}
