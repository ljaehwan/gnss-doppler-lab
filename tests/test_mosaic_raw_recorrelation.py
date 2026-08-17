import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.mosaic_raw_recorrelation import (
    FROZEN_GATE,
    TAP_OFFSETS_CHIPS,
    evaluate_recorrelation,
    fit_complex_amplitude,
    normalized_complex_cosine,
    prompt_normalize_safe,
    read_ishort_complex_window,
    receiver_carrier_wipeoff,
    receiver_code_replicas,
    receiver_l1ca_code,
    select_epoch_records,
    sha256_file,
)
from gnss_doppler_lab.trace_native_1ms import RECORD_DTYPE


def _action(**updates):
    row = np.zeros(1, dtype=RECORD_DTYPE)
    values = {
        "action_used_code_nco_rate_chips_s": 1_023_000.0,
        "action_used_carrier_doppler_hz": 1250.0,
        "action_used_residual_code_phase_chips": 0.1,
        "action_used_residual_code_phase_samples": 0.1,
        "action_used_code_phase_step_chips_per_sample": 1.0,
        "action_used_carrier_phase_step_rad_per_sample": 2 * np.pi * 1250 / 1_023_000,
    }
    values.update(updates)
    for key, value in values.items():
        row[key] = value
    return row[0]


def test_receiver_ca_known_vector_and_period():
    code = receiver_l1ca_code(1)
    assert code.shape == (1023,)
    assert code[:16].tolist() == [1, 1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1]
    assert set(np.unique(code)) == {-1.0, 1.0}


def test_exact_one_ms_code_coverage():
    replicas = receiver_code_replicas(3, 5000, 1_023_000 / 5_000_000, 0.0)
    prompt_indices = np.floor(np.arange(5000, dtype=np.float32) * np.float32(1_023_000 / 5_000_000)).astype(int)
    assert replicas.shape == (9, 5000)
    assert prompt_indices.min() == 0 and prompt_indices.max() == 1022


def test_sample_index_boundary(tmp_path):
    raw = np.asarray([1, -2, 3, -4, 5, -6], dtype="<i2")
    path = tmp_path / "iq.bin"
    path.write_bytes(raw.tobytes())
    assert read_ishort_complex_window(path, 1, 2).tolist() == [3 - 4j, 5 - 6j]
    with pytest.raises(ValueError, match="boundary"):
        read_ishort_complex_window(path, 2, 2)


def test_code_delay_sign_matches_receiver_shift_definition():
    replicas = receiver_code_replicas(7, 1023, 1.0, 0.0)
    prompt = replicas[4]
    correlations = np.abs(replicas @ prompt)
    assert int(np.argmax(correlations)) == 4
    assert np.array_equal(replicas[3, 1:], prompt[:-1])


def test_doppler_wipeoff_sign():
    fs = 1_023_000.0
    count = 1023
    freq = 2000.0
    phase = 0.37 + 2 * np.pi * freq * np.arange(count) / fs
    signal = np.exp(1j * phase).astype(np.complex64)
    correct = abs(np.sum(signal * receiver_carrier_wipeoff(count, 0.37, 2 * np.pi * freq / fs)))
    wrong = abs(np.sum(signal * receiver_carrier_wipeoff(count, 0.37, -2 * np.pi * freq / fs)))
    assert correct > 100 * wrong


def test_navigation_bit_invariance_after_epoch_complex_fit():
    rng = np.random.default_rng(4)
    reconstructed = rng.normal(size=9) + 1j * rng.normal(size=9)
    native_plus = (0.7 - 1.2j) * reconstructed
    native_minus = -native_plus
    _, fit_plus = fit_complex_amplitude(reconstructed, native_plus)
    _, fit_minus = fit_complex_amplitude(reconstructed, native_minus)
    assert normalized_complex_cosine(fit_plus, native_plus) == pytest.approx(1.0)
    assert normalized_complex_cosine(fit_minus, native_minus) == pytest.approx(1.0)
    assert np.allclose(prompt_normalize_safe(native_plus), prompt_normalize_safe(native_minus))


def test_global_complex_amplitude_invariance():
    taps = np.arange(1, 10) + 1j * np.arange(9)
    transformed = 3.2 * np.exp(0.9j) * taps
    alpha, fitted = fit_complex_amplitude(taps, transformed)
    assert alpha == pytest.approx(3.2 * np.exp(0.9j))
    assert normalized_complex_cosine(fitted, transformed) == pytest.approx(1.0)


def test_prompt_normalization_safety():
    taps = np.ones(9, dtype=np.complex128)
    taps[4] = 0
    with pytest.raises(ValueError, match="unsafe Prompt"):
        prompt_normalize_safe(taps)


def test_frozen_gate_pass_and_fail():
    native = np.arange(1, 10, dtype=float).astype(complex)
    result = evaluate_recorrelation(native * (2 + 3j), native, _action(), 1_023_000.0)
    assert result.gate_pass
    assert FROZEN_GATE == {
        "delay_center_error_abs_max_chips": 0.125,
        "doppler_center_error_abs_max_hz": 50.0,
        "complex_tap_cosine_min": 0.90,
        "magnitude_spearman_min": 0.90,
    }
    bad = evaluate_recorrelation(native[::-1] * (2 + 3j), native, _action(), 1_023_000.0)
    assert not bad.gate_pass


def test_sha256_binding(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"cleanStatic")
    assert sha256_file(path) == "9150ed680717378def16bca27d6cf9b16ad40e2c2c29000a751149a7e3ec53a1"


def test_runner_has_no_attack_input_and_frozen_clean_paths():
    text = (Path(__file__).parents[1] / "scripts/run_mosaic_stage0a_r1.py").read_text()
    assert "ds1.bin" not in text.lower()
    assert "ds3.bin" not in text.lower()
    assert "os3" not in text.lower()
    assert "cleanStatic_gps.bin" in text
    assert "texbat/raw/cleanStatic.bin" in text
    assert '"attack_data_used": False' in text


def test_artifact_manifest_verifier(tmp_path):
    module_path = Path(__file__).parents[1] / "scripts/verify_mosaic_stage0a_r1.py"
    spec = importlib.util.spec_from_file_location("verifier", module_path)
    verifier = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(verifier)
    for name in verifier.REQUIRED - {"artifact_manifest_sha256.json"}:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "config.json":
            path.write_text(json.dumps({"attack_data_used": False, "stage0b_run": False}))
        elif name == "final_verdict.json":
            path.write_text(json.dumps({"verdict": "STAGE0A_RAW_ALIGNMENT_PASS"}))
        else:
            path.write_text("test\n")
    (tmp_path / "artifact_manifest_sha256.json").write_text(json.dumps(verifier.build_manifest(tmp_path)))
    assert verifier.verify(tmp_path)["status"] == "PASS"
    (tmp_path / "README.md").write_text("tampered\n")
    assert verifier.verify(tmp_path)["status"] == "FAIL"


def test_deterministic_temporal_multi_prn_selection_on_real_clean_dumps():
    dump = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/mctd-stage0-static/dumps/phase_a/oakbat_cleanstatic/slow/rep1")
    if not dump.exists():
        pytest.skip("cleanStatic native dumps unavailable")
    first = select_epoch_records(dump, [10, 11, 21, 24, 27])
    second = select_epoch_records(dump, [10, 11, 21, 24, 27])
    assert first == second
    assert {row["prn"] for row in first} == {10, 11, 21, 24, 27}
    assert len(first) >= 120
    for prn in {10, 11, 21, 24, 27}:
        times = [row["receiver_timestamp_s"] for row in first if row["prn"] == prn]
        assert max(times) - min(times) > 10.0
