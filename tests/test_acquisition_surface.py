from pathlib import Path
import numpy as np

from gnss_doppler_lab.acquisition_surface import (
    compute_acquisition_surface,
    gps_l1ca_code,
    read_s8_iq,
    render_acquisition_surface,
    sampled_ca_code,
)


def test_gps_l1ca_code_has_one_period_and_balance():
    code = gps_l1ca_code("G05")
    assert code.shape == (1023,)
    assert set(np.unique(code)) == {-1.0, 1.0}
    assert abs(float(code.sum())) < 100


def test_acquisition_surface_finds_synthetic_peak(tmp_path: Path):
    fs = 2_600_000
    samples = int(fs * 0.001)
    doppler = 2250.0
    code = sampled_ca_code("G05", fs, samples)
    t = np.arange(samples) / fs
    iq = np.roll(code, 321) * np.exp(1j * 2 * np.pi * doppler * t)
    surface = compute_acquisition_surface(
        iq,
        "G05",
        fs,
        doppler_min_hz=-5000,
        doppler_max_hz=5000,
        doppler_step_hz=250,
    )
    assert abs(surface.peak_doppler_hz - doppler) <= 250
    assert surface.peak_to_second_ratio > 1.1
    png = render_acquisition_surface(surface, tmp_path / "acq.png")
    assert png.exists() and png.stat().st_size > 0


def test_read_s8_iq(tmp_path: Path):
    p = tmp_path / "iq.bin"
    p.write_bytes(bytes([1, 2, 253, 4]))  # int8: 1+2j, -3+4j
    iq = read_s8_iq(p, 2)
    assert iq[0] == 1 + 2j
    assert iq[1] == -3 + 4j
