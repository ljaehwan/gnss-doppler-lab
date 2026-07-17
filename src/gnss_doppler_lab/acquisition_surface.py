"""GPS L1 C/A acquisition-stage delay-Doppler correlation visualization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


# G2 tap pairs for GPS L1 C/A PRNs 1..32 (IS-GPS-200 C/A assignment).
_CA_TAPS = {
    1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9), 6: (2, 10),
    7: (1, 8), 8: (2, 9), 9: (3, 10), 10: (2, 3), 11: (3, 4), 12: (5, 6),
    13: (6, 7), 14: (7, 8), 15: (8, 9), 16: (9, 10), 17: (1, 4), 18: (2, 5),
    19: (3, 6), 20: (4, 7), 21: (5, 8), 22: (6, 9), 23: (1, 3), 24: (4, 6),
    25: (5, 7), 26: (6, 8), 27: (7, 9), 28: (8, 10), 29: (1, 6), 30: (2, 7),
    31: (3, 8), 32: (4, 9),
}


@dataclass(frozen=True)
class AcquisitionSurface:
    prn: str
    sample_rate_hz: float
    coherent_ms: int
    noncoherent_ms: int
    doppler_bins_hz: np.ndarray
    code_delay_chips: np.ndarray
    magnitude: np.ndarray
    peak_doppler_hz: float
    peak_code_delay_chips: float
    peak_magnitude: float
    second_peak_magnitude: float
    peak_to_second_ratio: float


def normalize_prn(prn: str | int) -> int:
    if isinstance(prn, int):
        value = prn
    else:
        text = str(prn).upper().strip()
        value = int(text[1:] if text.startswith("G") else text)
    if value not in _CA_TAPS:
        raise ValueError(f"Only GPS L1 C/A PRN 1..32 is supported, got {prn!r}")
    return value


def gps_l1ca_code(prn: str | int) -> np.ndarray:
    """Return one GPS L1 C/A code period as ±1 chips."""
    prn_i = normalize_prn(prn)
    tap1, tap2 = _CA_TAPS[prn_i]
    g1 = np.ones(10, dtype=np.int8)
    g2 = np.ones(10, dtype=np.int8)
    code = np.empty(1023, dtype=np.float32)
    for i in range(1023):
        g2_out = g2[tap1 - 1] ^ g2[tap2 - 1]
        bit = g1[-1] ^ g2_out
        code[i] = 1.0 if bit == 0 else -1.0
        g1_feedback = g1[2] ^ g1[9]
        g2_feedback = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1[1:] = g1[:-1]; g1[0] = g1_feedback
        g2[1:] = g2[:-1]; g2[0] = g2_feedback
    return code


def sampled_ca_code(prn: str | int, sample_rate_hz: float, samples: int) -> np.ndarray:
    chips = gps_l1ca_code(prn)
    chip_rate = 1.023e6
    chip_index = np.floor(np.arange(samples) * chip_rate / sample_rate_hz).astype(int) % 1023
    return chips[chip_index].astype(np.complex64)


def read_s8_iq(path: str | Path, samples: int, offset_samples: int = 0) -> np.ndarray:
    path = Path(path)
    start = offset_samples * 2
    count = samples * 2
    with path.open("rb") as f:
        f.seek(start)
        raw = np.frombuffer(f.read(count), dtype=np.int8)
    if raw.size < count:
        raise ValueError(f"IQ file too short for {samples} complex samples at offset {offset_samples}: {path}")
    i = raw[0::2].astype(np.float32)
    q = raw[1::2].astype(np.float32)
    return i + 1j * q


def compute_acquisition_surface(
    iq: np.ndarray,
    prn: str | int,
    sample_rate_hz: float,
    *,
    coherent_ms: int = 1,
    doppler_min_hz: int = -10000,
    doppler_max_hz: int = 10000,
    doppler_step_hz: int = 250,
    noncoherent_ms: int | None = None,
) -> AcquisitionSurface:
    samples = int(round(sample_rate_hz * 0.001 * coherent_ms))
    noncoherent_ms = int(noncoherent_ms or coherent_ms)
    if noncoherent_ms < coherent_ms or noncoherent_ms % coherent_ms != 0:
        raise ValueError("noncoherent_ms must be a positive multiple of coherent_ms")
    total_samples = int(round(sample_rate_hz * 0.001 * noncoherent_ms))
    if iq.size < total_samples:
        raise ValueError("Not enough IQ samples for requested coherent/noncoherent integration")
    replica = sampled_ca_code(prn, sample_rate_hz, samples)
    replica_fft_conj = np.conj(np.fft.fft(replica))
    t = np.arange(samples, dtype=np.float64) / sample_rate_hz
    bins = np.arange(doppler_min_hz, doppler_max_hz + 1, doppler_step_hz, dtype=float)
    mag = np.zeros((bins.size, samples), dtype=np.float32)
    segments = noncoherent_ms // coherent_ms
    for seg in range(segments):
        lo = seg * samples
        hi = lo + samples
        x = iq[lo:hi].astype(np.complex64)
        x = x - np.mean(x)
        for r, doppler in enumerate(bins):
            wiped = x * np.exp(-1j * 2.0 * np.pi * doppler * t)
            corr = np.fft.ifft(np.fft.fft(wiped) * replica_fft_conj)
            mag[r] += np.abs(corr).astype(np.float32)
    code_delay_chips = np.arange(samples, dtype=float) * 1023.0 / samples
    peak_flat = int(np.argmax(mag))
    peak_r, peak_c = np.unravel_index(peak_flat, mag.shape)
    peak_val = float(mag[peak_r, peak_c])
    guard = max(2, int(samples * 0.01))
    masked = mag.copy()
    for rr in range(max(0, peak_r - 1), min(mag.shape[0], peak_r + 2)):
        lo = max(0, peak_c - guard); hi = min(samples, peak_c + guard + 1)
        masked[rr, lo:hi] = 0
    second = float(masked.max()) if masked.size else 0.0
    prn_i = normalize_prn(prn)
    return AcquisitionSurface(
        prn=f"G{prn_i:02d}",
        sample_rate_hz=float(sample_rate_hz),
        coherent_ms=int(coherent_ms),
        noncoherent_ms=int(noncoherent_ms),
        doppler_bins_hz=bins,
        code_delay_chips=code_delay_chips,
        magnitude=mag,
        peak_doppler_hz=float(bins[peak_r]),
        peak_code_delay_chips=float(code_delay_chips[peak_c]),
        peak_magnitude=peak_val,
        second_peak_magnitude=second,
        peak_to_second_ratio=float(peak_val / second) if second > 0 else float("inf"),
    )


def render_acquisition_surface(surface: AcquisitionSurface, output_path: str | Path, *, title: str | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mag = surface.magnitude
    mag_db = 20.0 * np.log10(mag / max(surface.peak_magnitude, 1e-12) + 1e-12)
    # Decimate the code-delay axis for readable 3D rendering.
    step = max(1, mag.shape[1] // 400)
    x = surface.code_delay_chips[::step]
    y = surface.doppler_bins_hz
    X, Y = np.meshgrid(x, y)
    Z = mag_db[:, ::step]

    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    fig.suptitle(title or f"{surface.prn} acquisition delay-Doppler correlation surface", fontsize=14)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax3d.set_xlabel("Code delay (chips)")
    ax3d.set_ylabel("Doppler (Hz)")
    ax3d.set_zlabel("Relative correlation (dB)")
    ax3d.set_title("Acquisition peak as a 3D mountain")
    ax3d.view_init(elev=28, azim=-135)

    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(
        mag_db,
        aspect="auto",
        origin="lower",
        extent=[surface.code_delay_chips[0], surface.code_delay_chips[-1], y[0], y[-1]],
        cmap="magma",
        vmin=-35,
        vmax=0,
    )
    ax2.scatter([surface.peak_code_delay_chips], [surface.peak_doppler_hz], c="cyan", s=50, marker="x", label="peak")
    ax2.set_xlabel("Code delay (chips)")
    ax2.set_ylabel("Doppler (Hz)")
    ax2.set_title("Delay-Doppler heatmap")
    ax2.legend(loc="upper right")
    fig.colorbar(im, ax=ax2, label="Relative correlation (dB)")
    fig.text(
        0.5, 0.02,
        f"peak: delay={surface.peak_code_delay_chips:.1f} chips, doppler={surface.peak_doppler_hz:.0f} Hz, "
        f"peak/2nd={surface.peak_to_second_ratio:.2f}",
        ha="center",
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_surface_summary(surface: AcquisitionSurface, output_path: str | Path, *, source_iq: str | Path | None = None) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "prn": surface.prn,
        "sample_rate_hz": surface.sample_rate_hz,
        "coherent_ms": surface.coherent_ms,
        "noncoherent_ms": surface.noncoherent_ms,
        "doppler_min_hz": float(surface.doppler_bins_hz[0]),
        "doppler_max_hz": float(surface.doppler_bins_hz[-1]),
        "doppler_step_hz": float(surface.doppler_bins_hz[1] - surface.doppler_bins_hz[0]) if len(surface.doppler_bins_hz) > 1 else None,
        "code_delay_bins": int(len(surface.code_delay_chips)),
        "peak_doppler_hz": surface.peak_doppler_hz,
        "peak_code_delay_chips": surface.peak_code_delay_chips,
        "peak_magnitude": surface.peak_magnitude,
        "second_peak_magnitude": surface.second_peak_magnitude,
        "peak_to_second_ratio": surface.peak_to_second_ratio,
        "source_iq": str(source_iq) if source_iq is not None else None,
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return output_path
