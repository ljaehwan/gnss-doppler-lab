import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.rf_impairments import (
    ImpairmentConfig,
    MultipathTap,
    apply_impairments,
    open_sky_normal,
)


def write_iq(path: Path, z: np.ndarray) -> None:
    raw = np.empty(z.size * 2, dtype=np.int8)
    raw[0::2] = np.rint(z.real).astype(np.int8)
    raw[1::2] = np.rint(z.imag).astype(np.int8)
    path.write_bytes(raw.tobytes())


def read_iq(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int8).astype(np.float64)
    return raw[0::2] + 1j * raw[1::2]


def base(**kwargs) -> ImpairmentConfig:
    values = dict(
        enabled=True, profile="explicit", seed=123, sample_snr_db=None,
        carrier_offset_hz=0.0, frequency_drift_hz_per_s=0.0,
        phase_noise_std_rad_per_sqrt_sample=0.0, multipath=(),
        frontend_cutoff_hz=None, frontend_order=4,
        iq_gain_imbalance_db=0.0, iq_phase_imbalance_deg=0.0,
        dc_i=0.0, dc_q=0.0, gain=1.0, fading_depth=0.0,
        fading_rate_hz=0.0, ripple_depth=0.0, ripple_rate_hz=0.0,
        clip_level=127.0, chunk_samples=257,
    )
    values.update(kwargs)
    return ImpairmentConfig(**values)


def test_output_is_seeded_and_chunk_partition_invariant(tmp_path):
    rng = np.random.default_rng(8)
    z = rng.integers(-35, 36, 5003) + 1j * rng.integers(-35, 36, 5003)
    src = tmp_path / "in.bin"; write_iq(src, z)
    cfg = base(sample_snr_db=14.0, carrier_offset_hz=73.0,
               frequency_drift_hz_per_s=2.0,
               phase_noise_std_rad_per_sqrt_sample=0.002,
               frontend_cutoff_hz=1800.0,
               multipath=(MultipathTap(2.25, -13.0, 35.0),))
    outputs = []
    for chunk in (31, 257, 1024):
        out = tmp_path / f"out-{chunk}.bin"
        apply_impairments(src, out, 8000.0, replace(cfg, chunk_samples=chunk))
        outputs.append(out.read_bytes())
    assert outputs[0] == outputs[1] == outputs[2]
    other = tmp_path / "other.bin"
    apply_impairments(src, other, 8000.0, replace(cfg, seed=124, chunk_samples=31))
    assert other.read_bytes() != outputs[0]


def test_awgn_achieves_clean_composite_sample_snr(tmp_path):
    rng = np.random.default_rng(7)
    z = (rng.choice([-30, 30], 120_000) + 1j * rng.choice([-30, 30], 120_000)).astype(complex)
    src = tmp_path / "in.bin"; out = tmp_path / "out.bin"; write_iq(src, z)
    report = apply_impairments(src, out, 20_000.0, base(sample_snr_db=12.0, chunk_samples=997))
    error = read_iq(out) - z
    measured = 10 * np.log10(np.mean(np.abs(z) ** 2) / np.mean(np.abs(error) ** 2))
    assert measured == pytest.approx(12.0, abs=0.35)
    assert report["realized"]["equivalent_composite_cn0_db_hz"] == pytest.approx(12 + 10*np.log10(20_000))
    assert "not per-PRN" in report["cn0_caveat"]


def test_oscillator_phase_is_continuous_and_matches_equation(tmp_path):
    fs = 10_000.0
    z = np.full(2000, 50 + 0j)
    src = tmp_path / "in.bin"; out = tmp_path / "out.bin"; write_iq(src, z)
    apply_impairments(src, out, fs, base(carrier_offset_hz=37.0, frequency_drift_hz_per_s=5.0, chunk_samples=113))
    got = read_iq(out)
    phase_step = np.angle(got[1:] * np.conj(got[:-1]))
    expected = 2*np.pi*(37.0/fs + 5.0*(np.arange(1, 2000)-0.5)/fs**2)
    assert np.max(np.abs(np.unwrap(phase_step) - expected)) < 0.035
    assert abs(phase_step[112] - phase_step[111]) < 0.035


def test_fractional_multipath_impulse_delay(tmp_path):
    z = np.zeros(20, complex); z[0] = 40
    src = tmp_path / "in.bin"; out = tmp_path / "out.bin"; write_iq(src, z)
    apply_impairments(src, out, 1000.0, base(multipath=(MultipathTap(2.5, 0.0, 0.0),), chunk_samples=3))
    got = read_iq(out).real
    assert got[0] == 40
    assert got[2] == pytest.approx(20, abs=1)
    assert got[3] == pytest.approx(20, abs=1)


def test_atomic_validation_and_cleanup(tmp_path):
    odd = tmp_path / "odd.bin"; odd.write_bytes(b"123")
    out = tmp_path / "out.bin"
    with pytest.raises(ValueError, match="even"):
        apply_impairments(odd, out, 1000.0, base())
    assert not out.exists() and not list(tmp_path.glob("*.tmp"))
    src = tmp_path / "in.bin"; src.write_bytes(b"12"); out.write_bytes(b"owned")
    with pytest.raises(FileExistsError):
        apply_impairments(src, out, 1000.0, base())
    assert out.read_bytes() == b"owned"


def test_open_sky_profile_is_sane_reproducible_and_diverse():
    a = open_sky_normal(44, 2_600_000); b = open_sky_normal(44, 2_600_000); c = open_sky_normal(45, 2_600_000)
    assert a == b and a != c
    assert -16 <= a.sample_snr_db <= -8
    assert abs(a.carrier_offset_hz) <= 150
    assert a.multipath == ()
    assert a.fading_depth == 0 and a.ripple_depth == 0


@pytest.mark.parametrize("sample_rate_hz", [1_000_000, 2_000_000, 2_600_000, 4_000_000, 5_000_000])
def test_open_sky_frontend_cutoff_tracks_nyquist(sample_rate_hz):
    cfg = open_sky_normal(987, sample_rate_hz)
    fraction = cfg.frontend_cutoff_hz / (sample_rate_hz / 2)
    assert 0.76 <= fraction <= 0.92


def test_receiver_agc_prevents_negative_snr_adc_saturation_and_reports_measurements(tmp_path):
    rng = np.random.default_rng(71)
    z = rng.integers(-45, 46, 200_000) + 1j * rng.integers(-45, 46, 200_000)
    src = tmp_path / "in.bin"; out = tmp_path / "out.bin"; write_iq(src, z)
    report = apply_impairments(
        src, out, 2_600_000,
        base(sample_snr_db=-12.0, agc_target_rms=24.0, chunk_samples=1009),
    )
    realized = report["realized"]
    measured = report["measurements"]
    assert 0 < realized["receiver_agc_applied_gain"] < 1
    assert measured["pre_agc_mean_complex_power"] > measured["post_agc_mean_complex_power"]
    assert np.sqrt(measured["post_agc_mean_complex_power"]) == pytest.approx(24, rel=0.03)
    assert report["output"]["clipping_fraction"] < 0.001
    assert measured["quantized_mean_complex_power"] > 0
    assert measured["achieved_quantized_sample_snr_db"] == pytest.approx(-12, abs=0.6)


def test_two_pass_processor_does_not_create_channel_intermediate(tmp_path, monkeypatch):
    import gnss_doppler_lab.rf_impairments as module
    z = np.full(4000, 20 + 5j)
    src = tmp_path / "in.bin"; out = tmp_path / "out.bin"; write_iq(src, z)
    real_mkstemp = module.tempfile.mkstemp
    prefixes = []
    def checked_mkstemp(*args, **kwargs):
        prefix = kwargs.get("prefix", args[0] if args else "")
        prefixes.append(prefix)
        assert "channel" not in prefix
        return real_mkstemp(*args, **kwargs)
    monkeypatch.setattr(module.tempfile, "mkstemp", checked_mkstemp)
    report = apply_impairments(src, out, 10_000, base(sample_snr_db=2, agc_target_rms=24))
    assert len(prefixes) == 1
    assert report["processing"]["passes"] == 2
    assert report["processing"]["channel_intermediate_bytes"] == 0


def test_manifest_records_runtime_filter_rng_and_layer_model(tmp_path):
    src = tmp_path / "in.bin"; out = tmp_path / "out.bin"
    write_iq(src, np.full(5000, 20 + 3j))
    report = apply_impairments(
        src, out, 10_000,
        base(frontend_cutoff_hz=4000, sample_snr_db=0, agc_target_rms=22),
    )
    assert report["runtime"]["numpy_version"]
    assert report["runtime"]["scipy_version"]
    assert report["runtime"]["rng_bit_generator"] == "PCG64"
    assert report["filter"]["normalized_cutoff_to_nyquist"] == pytest.approx(0.8)
    assert report["filter"]["sos"]
    assert report["layer_model_version"] >= 2
    assert "cross-version" in report["reproducibility_caveat"]
