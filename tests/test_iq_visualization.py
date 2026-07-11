from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.iq_visualization import load_s8_iq, render_iq_dashboard, summarize_iq


def test_load_s8_iq_decodes_interleaved_signed_samples(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(np.array([1, -2, 3, -4], dtype=np.int8).tobytes())

    iq = load_s8_iq(path)

    np.testing.assert_array_equal(iq, np.array([1 - 2j, 3 - 4j], dtype=np.complex64))


def test_load_s8_iq_rejects_odd_byte_count(tmp_path: Path) -> None:
    path = tmp_path / "broken.bin"
    path.write_bytes(b"\x01\x02\x03")

    with pytest.raises(ValueError, match="even"):
        load_s8_iq(path)


def test_summarize_iq_reports_signal_statistics() -> None:
    iq = np.array([1 + 2j, -3 + 4j], dtype=np.complex64)

    summary = summarize_iq(iq, sample_rate_hz=2.0)

    assert summary["complex_samples"] == 2
    assert summary["duration_seconds"] == pytest.approx(1.0)
    assert summary["peak_magnitude"] == pytest.approx(5.0)
    assert summary["rms_magnitude"] == pytest.approx(np.sqrt(15.0))


def test_render_iq_dashboard_writes_png(tmp_path: Path) -> None:
    sample_rate_hz = 2_600_000.0
    t = np.arange(65_536, dtype=np.float32) / sample_rate_hz
    iq = (40.0 * np.exp(2j * np.pi * 100_000.0 * t)).astype(np.complex64)
    output = tmp_path / "dashboard.png"

    render_iq_dashboard(iq, sample_rate_hz=sample_rate_hz, output_path=output)

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 10_000
