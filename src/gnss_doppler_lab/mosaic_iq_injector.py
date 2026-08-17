"""MOSAIC-GNSS Stage-0B physical raw-IQ injector utilities.

The functions here deliberately operate on short in-memory windows only.  They
reuse the verified acquisition-surface GPS L1 C/A generator and never persist
scenario-scale synthetic IQ.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .acquisition_surface import gps_l1ca_code

CHIP_RATE_HZ = 1.023e6


@dataclass(frozen=True)
class InjectionTheta:
    delay_chips: float
    doppler_hz: float
    power_ratio_db: float
    phase_rad: float


def sampled_prn_replica(
    prn: str | int,
    sample_rate_hz: float,
    sample_count: int,
    *,
    code_phase_chips: float = 0.0,
    doppler_hz: float = 0.0,
    carrier_phase_rad: float = 0.0,
    nav_bits: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Create a complex GPS L1 C/A PRN-specific sample replica.

    Positive ``code_phase_chips`` advances the local code index by that many
    chips.  ``nav_bits`` must be supplied by a caller with provenance; this
    utility accepts either a scalar for unit tests or one value per sample.
    """
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    fs = float(sample_rate_hz)
    if fs <= 0:
        raise ValueError("sample_rate_hz must be positive")
    chips = gps_l1ca_code(prn).astype(np.float64)
    n = np.arange(sample_count, dtype=np.float64)
    chip_index = np.floor(code_phase_chips + n * CHIP_RATE_HZ / fs).astype(np.int64) % 1023
    code = chips[chip_index]
    bits = np.asarray(nav_bits, dtype=np.float64)
    if bits.ndim == 0:
        bits = np.full(sample_count, float(bits), dtype=np.float64)
    if bits.shape != (sample_count,):
        raise ValueError("nav_bits must be scalar or one value per sample")
    if not np.all(np.isin(bits, [-1.0, 1.0])):
        raise ValueError("nav_bits must contain only ±1 values")
    carrier = np.exp(1j * (2.0 * np.pi * float(doppler_hz) * n / fs + float(carrier_phase_rad)))
    return (code * bits * carrier).astype(np.complex128)


def rms_power(iq: np.ndarray) -> float:
    x = np.asarray(iq, dtype=np.complex128)
    return float(np.mean(np.abs(x) ** 2)) if x.size else 0.0


def inject_counterfeit(
    clean_iq: np.ndarray,
    prn: str | int,
    sample_rate_hz: float,
    theta: InjectionTheta,
    *,
    authentic_code_phase_chips: float = 0.0,
    authentic_doppler_hz: float = 0.0,
    authentic_carrier_phase_rad: float = 0.0,
    nav_bits: np.ndarray | float | None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Add a PRN-specific counterfeit waveform at sample level.

    Fail-closed if navigation-bit provenance is absent (``nav_bits is None``).
    The injected amplitude is set relative to the clean window RMS power.
    """
    if nav_bits is None:
        raise ValueError("navigation-bit provenance is required; refusing +1 fallback")
    y = np.asarray(clean_iq, dtype=np.complex128)
    replica = sampled_prn_replica(
        prn,
        sample_rate_hz,
        y.size,
        code_phase_chips=authentic_code_phase_chips + theta.delay_chips,
        doppler_hz=authentic_doppler_hz + theta.doppler_hz,
        carrier_phase_rad=authentic_carrier_phase_rad + theta.phase_rad,
        nav_bits=nav_bits,
    )
    clean_power = rms_power(y)
    replica_power = max(rms_power(replica), 1e-30)
    target_power = clean_power * 10.0 ** (float(theta.power_ratio_db) / 10.0)
    amplitude = np.sqrt(target_power / replica_power) if clean_power > 0 else 0.0
    injected = y + amplitude * replica
    metrics = {
        "clean_power": clean_power,
        "replica_power_before_scale": replica_power,
        "target_injection_power": float(target_power),
        "amplitude_scale": float(amplitude),
        "output_power": rms_power(injected),
    }
    return injected, metrics


def quantize_interleaved_int8(iq: np.ndarray) -> tuple[bytes, dict[str, float]]:
    """Quantize complex samples to interleaved signed int8 without rescaling."""
    x = np.asarray(iq, dtype=np.complex128)
    i = np.rint(x.real)
    q = np.rint(x.imag)
    clipped = (i < -128) | (i > 127) | (q < -128) | (q > 127)
    interleaved = np.empty(x.size * 2, dtype=np.int8)
    interleaved[0::2] = np.clip(i, -128, 127).astype(np.int8)
    interleaved[1::2] = np.clip(q, -128, 127).astype(np.int8)
    headroom = 127.0 - float(max(np.max(np.abs(i)) if i.size else 0.0, np.max(np.abs(q)) if q.size else 0.0))
    return interleaved.tobytes(), {"clipping_rate": float(np.mean(clipped)) if clipped.size else 0.0, "headroom_counts": headroom}


def design_sha256(design: object) -> str:
    import json
    payload = json.dumps(design, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()
