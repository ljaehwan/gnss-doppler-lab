"""Utilities for reading and summarizing signed 8-bit interleaved IQ."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_s8_iq(path: str | Path, *, max_complex_samples: int | None = None) -> np.ndarray:
    """Load interleaved signed 8-bit I/Q samples as complex64."""
    iq_path = Path(path)
    byte_count = iq_path.stat().st_size
    if byte_count % 2:
        raise ValueError("s8 interleaved IQ file must contain an even byte count")
    count = -1 if max_complex_samples is None else max_complex_samples * 2
    raw = np.fromfile(iq_path, dtype=np.int8, count=count)
    return raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)


def summarize_iq(iq: np.ndarray, *, sample_rate_hz: float) -> dict[str, Any]:
    """Return compact, JSON-friendly statistics for complex IQ samples."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if iq.size == 0:
        raise ValueError("IQ array must not be empty")
    magnitude = np.abs(iq)
    return {
        "complex_samples": int(iq.size),
        "duration_seconds": float(iq.size / sample_rate_hz),
        "peak_magnitude": float(np.max(magnitude)),
        "rms_magnitude": float(np.sqrt(np.mean(np.square(magnitude)))),
        "mean_i": float(np.mean(iq.real)),
        "mean_q": float(np.mean(iq.imag)),
    }


def render_iq_dashboard(
    iq: np.ndarray,
    *,
    sample_rate_hz: float,
    output_path: str | Path,
    title: str = "GPS L1 C/A normal received IQ",
) -> Path:
    """Render waveform, constellation, spectrum, and spectrogram to a PNG."""
    if iq.size < 2_048:
        raise ValueError("at least 2048 complex samples are required for visualization")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    waveform_count = min(iq.size, 5_000)
    time_ms = np.arange(waveform_count) / sample_rate_hz * 1e3
    axes[0, 0].plot(time_ms, iq.real[:waveform_count], linewidth=0.7, label="I")
    axes[0, 0].plot(time_ms, iq.imag[:waveform_count], linewidth=0.7, alpha=0.8, label="Q")
    axes[0, 0].set(title="Time-domain waveform", xlabel="Time (ms)", ylabel="ADC level")
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].grid(alpha=0.25)

    stride = max(1, iq.size // 40_000)
    points = iq[::stride][:40_000]
    axes[0, 1].hexbin(points.real, points.imag, gridsize=80, mincnt=1, bins="log", cmap="viridis")
    axes[0, 1].set(title="I/Q density", xlabel="I", ylabel="Q")
    axes[0, 1].set_aspect("equal", adjustable="box")
    axes[0, 1].grid(alpha=0.2)

    nfft = min(65_536, 2 ** int(np.floor(np.log2(iq.size))))
    segment = iq[:nfft] * np.hanning(nfft)
    spectrum = np.fft.fftshift(np.fft.fft(segment))
    power_db = 20.0 * np.log10(np.maximum(np.abs(spectrum), 1e-12))
    power_db -= np.max(power_db)
    freq_mhz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate_hz)) / 1e6
    axes[1, 0].plot(freq_mhz, power_db, linewidth=0.8)
    axes[1, 0].set(title="Relative baseband spectrum", xlabel="Frequency offset (MHz)", ylabel="Relative power (dB)")
    axes[1, 0].set_ylim(-100, 5)
    axes[1, 0].grid(alpha=0.25)

    spectrogram_count = min(iq.size, int(sample_rate_hz * 0.2))
    axes[1, 1].specgram(
        iq[:spectrogram_count],
        NFFT=2_048,
        Fs=sample_rate_hz,
        noverlap=1_536,
        scale="dB",
        cmap="magma",
    )
    axes[1, 1].set(title="Short-time spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)")

    summary = summarize_iq(iq, sample_rate_hz=sample_rate_hz)
    fig.text(
        0.5,
        0.005,
        f"Loaded {summary['complex_samples']:,} samples ({summary['duration_seconds']:.3f} s) | "
        f"RMS {summary['rms_magnitude']:.2f} | Peak {summary['peak_magnitude']:.2f}",
        ha="center",
        fontsize=10,
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
