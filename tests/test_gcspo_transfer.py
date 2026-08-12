from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

import numpy as np
import pytest

from gnss_doppler_lab.gcspo_core import SharedVAR


SOURCE_ROOT = Path("/home/ubuntu/build-gnss-sdr-complex9")


def _copy_source(tmp_path):
    if not SOURCE_ROOT.is_dir():
        pytest.skip("pinned receiver source unavailable")
    root = tmp_path / "receiver"
    relatives = (
        "src/algorithms/tracking/libs/tracking_discriminators.cc",
        "src/algorithms/tracking/gnuradio_blocks/dll_pll_veml_tracking.cc",
    )
    for relative in relatives:
        target = root / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_ROOT / relative, target)
    return root


def _model():
    coefficients = np.zeros((10, 10, 10))
    coefficients[0] = np.eye(10) * .1
    return SharedVAR(np.zeros(10), coefficients)


def _coeff_hash(model):
    return hashlib.sha256(np.ascontiguousarray(model.coefficients, dtype=np.float64).tobytes()).hexdigest()


def _source_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*.cc")}


def test_transfer_oracle_50_epoch_impulse_ramp_and_exactly_once(tmp_path):
    from gnss_doppler_lab.gcspo_transfer import prove_closed_loop_transfer

    root = _copy_source(tmp_path); model = _model()
    report = prove_closed_loop_transfer(root, model, expected_source_hashes=_source_hashes(root),
                                        expected_var_sha256=_coeff_hash(model))
    assert report["overall_status"] == "PASS"
    assert report["vector_contract"] == {"epochs": 50, "range_impulse_epoch": 10, "rate_ramp_start_epoch": 10}
    assert report["var_transfer_application_count"] == 1
    assert set(report["validated_rows"]) == {"code_error_chips", "pll_phase_error_cycles", "carrier_doppler_hz", "code_frequency_offset_chips_s"}
    assert max(row["maximum_oracle_error"] for row in report["rows"]) <= 1e-12


def test_transfer_rejects_source_hash_and_var_mutations(tmp_path):
    from gnss_doppler_lab.gcspo_transfer import prove_closed_loop_transfer

    root = _copy_source(tmp_path); model = _model(); hashes = _source_hashes(root)
    path = root / "src/algorithms/tracking/libs/tracking_discriminators.cc"
    path.write_text(path.read_text() + "\n// mutation\n")
    assert prove_closed_loop_transfer(root, model, expected_source_hashes=hashes,
                                      expected_var_sha256=_coeff_hash(model))["overall_status"] == "FAIL"
    root = _copy_source(tmp_path / "second"); hashes = _source_hashes(root)
    mutated = SharedVAR(model.intercept, model.coefficients.copy()); mutated.coefficients[0, 0, 0] += .01
    report = prove_closed_loop_transfer(root, mutated, expected_source_hashes=hashes,
                                        expected_var_sha256=_coeff_hash(model))
    assert report["checks"]["var_coefficient_hash"] == "FAIL"
    assert report["validated_rows"] == []


@pytest.mark.parametrize("needle,replacement,failed_check", [
    ("dll_nc_e_minus_l_normalized", "dll_mutated", "equation"),
    ("d_correlation_length_ms = 1", "d_correlation_length_ms = 2", "cadence"),
    ("pll_cloop_two_quadrant_atan(d_P_accu) / TWO_PI", "-pll_cloop_two_quadrant_atan(d_P_accu) / TWO_PI", "sign"),
    ("d_carrier_doppler_hz = d_carr_error_filt_hz", "d_carrier_doppler_hz = -d_carr_error_filt_hz", "loop_path"),
])
def test_transfer_rejects_equation_cadence_sign_and_loop_mutations(tmp_path, needle, replacement, failed_check):
    from gnss_doppler_lab.gcspo_transfer import prove_closed_loop_transfer

    root = _copy_source(tmp_path); model = _model()
    path = next(path for path in root.rglob("*.cc") if needle in path.read_text())
    path.write_text(path.read_text().replace(needle, replacement, 1))
    report = prove_closed_loop_transfer(root, model, expected_source_hashes=_source_hashes(root),
                                        expected_var_sha256=_coeff_hash(model))
    assert report["checks"][failed_check] == "FAIL"
    assert report["overall_status"] == "FAIL"


def test_transfer_wrap_boundary_mutation_fails(tmp_path):
    from gnss_doppler_lab.gcspo_transfer import prove_closed_loop_transfer

    root = _copy_source(tmp_path); model = _model()
    path = root / "src/algorithms/tracking/libs/tracking_discriminators.cc"
    text = path.read_text()
    if "atan2" not in text:
        pytest.skip("wrap oracle source token is in another pinned translation unit")
    path.write_text(text.replace("atan2", "atan", 1))
    report = prove_closed_loop_transfer(root, model, expected_source_hashes=_source_hashes(root),
                                        expected_var_sha256=_coeff_hash(model))
    assert report["checks"]["wrap_linear_range"] == "FAIL"


def test_method_availability_fails_closed_by_physical_family():
    from gnss_doppler_lab.gcspo_transfer import method_availability

    assert method_availability({"code_error_chips", "carrier_doppler_hz"}) == {"A3": True, "A4": True, "Full": True}
    assert method_availability({"code_error_chips"}) == {"A3": True, "A4": False, "Full": False}
    assert method_availability({"carrier_doppler_hz"}) == {"A3": False, "A4": True, "Full": False}
    assert method_availability(set()) == {"A3": False, "A4": False, "Full": False}
