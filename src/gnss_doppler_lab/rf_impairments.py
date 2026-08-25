"""Deterministic, bounded-memory RF impairments for interleaved signed-8-bit IQ.

The simulator output is a clean composite.  Common multipath and fading are
therefore explicit stress-test approximations, not PRN-specific propagation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

import numpy as np
import scipy
from scipy.signal import butter, sosfilt

LAYER_MODEL_VERSION = 2
CN0_CAVEAT = (
    "sample_snr_db and equivalent composite C/N0 reference total clean composite "
    "IQ power; it is not per-PRN. Approximate per-PRN C/N0 depends on visible "
    "PRN count and signal-power distribution; GNSS-SDR-specific calibration is required."
)
COMPOSITE_CHANNEL_CAVEAT = (
    "Composite multipath/fading applies one common channel to all PRNs and can create "
    "artificial cross-PRN correlation. It is disabled in open_sky_normal and is only "
    "an explicit composite-channel approximation for stress or hard-negative tests."
)


@dataclass(frozen=True)
class MultipathTap:
    delay_samples: float
    attenuation_db: float
    phase_deg: float

    def __post_init__(self) -> None:
        vals = (self.delay_samples, self.attenuation_db, self.phase_deg)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("multipath values must be finite")
        if self.delay_samples <= 0 or self.delay_samples > 1_000_000:
            raise ValueError("multipath delay_samples must be in (0, 1000000]")
        if not -80.0 <= self.attenuation_db <= 0.0:
            raise ValueError("multipath attenuation_db must be between -80 and 0")
        if not -360.0 <= self.phase_deg <= 360.0:
            raise ValueError("multipath phase_deg must be between -360 and 360")


@dataclass(frozen=True)
class ImpairmentConfig:
    enabled: bool = False
    profile: str = "clean"
    seed: int = 0
    sample_snr_db: float | None = None
    carrier_offset_hz: float = 0.0
    frequency_drift_hz_per_s: float = 0.0
    phase_noise_std_rad_per_sqrt_sample: float = 0.0
    multipath: tuple[MultipathTap, ...] = ()
    frontend_cutoff_hz: float | None = None
    frontend_order: int = 4
    iq_gain_imbalance_db: float = 0.0
    iq_phase_imbalance_deg: float = 0.0
    dc_i: float = 0.0
    dc_q: float = 0.0
    # Fixed composite signal-channel gain; this is deliberately not called AGC.
    gain: float = 1.0
    fading_depth: float = 0.0
    fading_rate_hz: float = 0.0
    fading_phase_deg: float = 0.0
    ripple_depth: float = 0.0
    ripple_rate_hz: float = 0.0
    ripple_phase_deg: float = 0.0
    # Receiver AGC is applied to signal + frontend-output-referred AWGN.
    agc_target_rms: float | None = None
    clip_level: float = 127.0
    chunk_samples: int = 1_048_576

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an integer in [0, 2^64)")
        if self.profile not in {"clean", "explicit", "open_sky_normal"}:
            raise ValueError("profile must be clean, explicit, or open_sky_normal")
        numeric = [
            self.carrier_offset_hz, self.frequency_drift_hz_per_s,
            self.phase_noise_std_rad_per_sqrt_sample, self.iq_gain_imbalance_db,
            self.iq_phase_imbalance_deg, self.dc_i, self.dc_q, self.gain,
            self.fading_depth, self.fading_rate_hz, self.fading_phase_deg,
            self.ripple_depth, self.ripple_rate_hz, self.ripple_phase_deg,
            self.clip_level,
        ]
        if self.sample_snr_db is not None: numeric.append(self.sample_snr_db)
        if self.frontend_cutoff_hz is not None: numeric.append(self.frontend_cutoff_hz)
        if self.agc_target_rms is not None: numeric.append(self.agc_target_rms)
        if not all(math.isfinite(float(v)) for v in numeric):
            raise ValueError("impairment parameters must be finite")
        if self.sample_snr_db is not None and not -30 <= self.sample_snr_db <= 100:
            raise ValueError("sample_snr_db must be between -30 and 100")
        if self.phase_noise_std_rad_per_sqrt_sample < 0:
            raise ValueError("phase noise must be nonnegative")
        if self.frontend_cutoff_hz is not None and self.frontend_cutoff_hz <= 0:
            raise ValueError("frontend_cutoff_hz must be positive")
        if not isinstance(self.frontend_order, int) or not 1 <= self.frontend_order <= 12:
            raise ValueError("frontend_order must be 1..12")
        if abs(self.iq_gain_imbalance_db) > 12 or abs(self.iq_phase_imbalance_deg) > 45:
            raise ValueError("IQ imbalance outside supported range")
        if self.gain <= 0 or self.gain > 20:
            raise ValueError("gain must be in (0, 20]")
        if not 0 <= self.fading_depth < 1 or not 0 <= self.ripple_depth < 1:
            raise ValueError("fading_depth and ripple_depth must be in [0, 1)")
        if self.fading_rate_hz < 0 or self.ripple_rate_hz < 0:
            raise ValueError("gain rates must be nonnegative")
        if self.agc_target_rms is not None and not 1 <= self.agc_target_rms <= 80:
            raise ValueError("agc_target_rms must be in [1, 80]")
        if not 1 <= self.clip_level <= 127:
            raise ValueError("clip_level must be in [1, 127]")
        if not isinstance(self.chunk_samples, int) or not 1 <= self.chunk_samples <= 16_777_216:
            raise ValueError("chunk_samples must be in [1, 16777216]")
        if not isinstance(self.multipath, tuple) or not all(isinstance(t, MultipathTap) for t in self.multipath):
            raise ValueError("multipath must be a tuple of MultipathTap")
        if len(self.multipath) > 16:
            raise ValueError("at most 16 multipath taps are supported")

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


def open_sky_normal(seed: int, sample_rate_hz: float) -> ImpairmentConfig:
    """Draw a benign frontend realization with no common composite propagation."""
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2^64)")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    r = np.random.default_rng(seed)
    nyquist = sample_rate_hz / 2
    return ImpairmentConfig(
        enabled=True, profile="open_sky_normal", seed=seed,
        sample_snr_db=float(r.uniform(-14, -8)),
        carrier_offset_hz=float(r.uniform(-120, 120)),
        frequency_drift_hz_per_s=float(r.uniform(-0.8, 0.8)),
        phase_noise_std_rad_per_sqrt_sample=float(r.uniform(1e-5, 8e-5)),
        multipath=(),
        frontend_cutoff_hz=float(r.uniform(0.76, 0.92) * nyquist),
        frontend_order=5,
        iq_gain_imbalance_db=float(r.uniform(-0.45, 0.45)),
        iq_phase_imbalance_deg=float(r.uniform(-1.5, 1.5)),
        dc_i=float(r.uniform(-1.2, 1.2)), dc_q=float(r.uniform(-1.2, 1.2)),
        gain=1.0, fading_depth=0.0, ripple_depth=0.0,
        agc_target_rms=float(r.uniform(22, 28)), clip_level=127.0,
    )


def clean_impairments() -> ImpairmentConfig:
    return ImpairmentConfig()


def _multipath(x: np.ndarray, taps: tuple[MultipathTap, ...], history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not taps:
        return x.copy(), history
    hlen = history.size
    joined = np.concatenate((history, x))
    y = x.astype(np.complex64, copy=True)
    idx = hlen + np.arange(x.size)
    for tap in taps:
        whole = int(math.floor(tap.delay_samples))
        frac = np.float32(tap.delay_samples - whole)
        delayed = (1 - frac) * joined[idx - whole] + frac * joined[idx - whole - 1]
        coefficient = np.complex64(10 ** (tap.attenuation_db / 20) * np.exp(1j * np.deg2rad(tap.phase_deg)))
        y += delayed.astype(np.complex64) * coefficient
    return y, joined[-hlen:].copy()


class CompositeChannelProcessor:
    """Stateful deterministic channel for IQ8 or in-memory complex samples.

    ``process_complex`` superposes sources before the common oscillator and
    frontend without an intermediate IQ8 quantization.
    """
    def __init__(self, fs: float, cfg: ImpairmentConfig):
        self.fs, self.cfg, self.offset = fs, cfg, 0
        max_delay = max((math.ceil(t.delay_samples) for t in cfg.multipath), default=0)
        self.history = np.zeros(max_delay + 1, np.complex64) if cfg.multipath else np.empty(0, np.complex64)
        self.sos = butter(cfg.frontend_order, cfg.frontend_cutoff_hz / (fs / 2), output="sos") if cfg.frontend_cutoff_hz else None
        self.zi = np.zeros((self.sos.shape[0], 2), np.complex128) if self.sos is not None else None
        self.phase_rng = np.random.default_rng(np.random.SeedSequence(cfg.seed).spawn(2)[0])
        self.random_phase = 0.0

    def process(self, raw: bytes) -> np.ndarray:
        """Backward-compatible IQ8 byte entry point."""
        a = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
        return self.process_complex(a[0::2] + 1j * a[1::2])

    def process_complex(self, samples: np.ndarray) -> np.ndarray:
        """Process complex samples without an intermediate IQ8 quantization."""
        x = np.asarray(samples, dtype=np.complex64)
        if x.ndim != 1:
            raise ValueError("complex channel input must be one-dimensional")
        x, self.history = _multipath(x, self.cfg.multipath, self.history)
        n = self.offset + np.arange(x.size, dtype=np.float64)
        t = n / self.fs
        phase = 2 * np.pi * (self.cfg.carrier_offset_hz * t + 0.5 * self.cfg.frequency_drift_hz_per_s * t * t)
        if self.cfg.phase_noise_std_rad_per_sqrt_sample:
            increments = self.phase_rng.standard_normal(x.size) * self.cfg.phase_noise_std_rad_per_sqrt_sample
            walk = self.random_phase + np.cumsum(increments)
            self.random_phase = float(walk[-1])
            phase += walk
        x *= np.exp(1j * phase).astype(np.complex64)
        if self.sos is not None:
            filtered, self.zi = sosfilt(self.sos, x, zi=self.zi)
            x = filtered.astype(np.complex64)
        envelope = self.cfg.gain * (
            1 + self.cfg.fading_depth * np.sin(2 * np.pi * self.cfg.fading_rate_hz * t + np.deg2rad(self.cfg.fading_phase_deg))
        )
        envelope *= 1 + self.cfg.ripple_depth * np.sin(
            2 * np.pi * self.cfg.ripple_rate_hz * t + np.deg2rad(self.cfg.ripple_phase_deg)
        )
        x *= envelope.astype(np.float32)
        self.offset += x.size
        return x


# Internal compatibility for the existing two-pass impairment implementation.
_ChannelPass = CompositeChannelProcessor


def _iq_transform(x: np.ndarray, cfg: ImpairmentConfig) -> np.ndarray:
    gi = np.float32(10 ** (cfg.iq_gain_imbalance_db / 40))
    gq = np.float32(1 / gi)
    phi = np.float32(np.deg2rad(cfg.iq_phase_imbalance_deg))
    i = gi * x.real
    q = gq * (x.imag * np.cos(phi) + x.real * np.sin(phi))
    return (i + 1j * q).astype(np.complex64)


def apply_iq_imbalance(x: np.ndarray, config: ImpairmentConfig) -> np.ndarray:
    """Apply receiver IQ imbalance without gain, clipping, or quantization."""
    if not isinstance(config, ImpairmentConfig):
        raise TypeError("config must be an ImpairmentConfig")
    return _iq_transform(np.asarray(x, dtype=np.complex64), config)


def _chunks(path: Path, chunk_samples: int) -> Iterator[bytes]:
    with path.open("rb") as f:
        while raw := f.read(chunk_samples * 2):
            yield raw


def _safe_db_ratio(numerator: float, denominator: float) -> float | None:
    if numerator <= 0 or denominator <= 0:
        return None
    return 10 * math.log10(numerator / denominator)


def apply_impairments(input_path: str | Path, output_path: str | Path, sample_rate_hz: float,
                      config: ImpairmentConfig) -> dict[str, Any]:
    """Apply a deterministic two-pass pipeline and atomically publish final s8 IQ."""
    src, dst = Path(input_path), Path(output_path)
    if not isinstance(config, ImpairmentConfig) or not config.enabled:
        raise ValueError("apply_impairments requires an enabled ImpairmentConfig")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    if config.frontend_cutoff_hz is not None and config.frontend_cutoff_hz >= sample_rate_hz / 2:
        raise ValueError("frontend_cutoff_hz must be below Nyquist")
    if src.resolve() == dst.resolve():
        raise ValueError("input and output must differ")
    if dst.exists():
        raise FileExistsError(dst)
    size = src.stat().st_size
    if size % 2:
        raise ValueError("s8 interleaved IQ input byte count must be even")
    if size == 0:
        raise ValueError("IQ input must not be empty")
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: hash clean bytes and measure the regenerated deterministic signal.
    clean_hash = hashlib.sha256()
    channel_hash_1 = hashlib.sha256()
    channel = _ChannelPass(sample_rate_hz, config)
    count = 0; clean_power_sum = 0.0; clean_peak = 0.0
    channel_power_sum = 0.0; noiseless_rx_power_sum = 0.0
    for raw in _chunks(src, config.chunk_samples):
        clean_hash.update(raw)
        a = np.frombuffer(raw, np.int8).astype(np.float32)
        clean_z = a[0::2] + 1j * a[1::2]
        clean_power_sum += float(np.vdot(clean_z, clean_z).real)
        if clean_z.size: clean_peak = max(clean_peak, float(np.max(np.abs(clean_z))))
        signal = channel.process(raw)
        channel_hash_1.update(signal.tobytes())
        channel_power_sum += float(np.vdot(signal, signal).real)
        noiseless_rx = _iq_transform(signal, config) + np.complex64(config.dc_i + 1j * config.dc_q)
        noiseless_rx_power_sum += float(np.vdot(noiseless_rx, noiseless_rx).real)
        count += signal.size
    channel_power = channel_power_sum / count
    noise_power = 0.0 if config.sample_snr_db is None else channel_power / 10 ** (config.sample_snr_db / 10)
    gi = 10 ** (config.iq_gain_imbalance_db / 40)
    gq = 1 / gi
    transformed_noise_power = noise_power * (gi * gi + gq * gq) / 2
    expected_pre_agc_power = noiseless_rx_power_sum / count + transformed_noise_power
    agc_gain = 1.0 if config.agc_target_rms is None else config.agc_target_rms / math.sqrt(expected_pre_agc_power)

    # Pass 2: reset all state/RNG, regenerate, add frontend-output/ADC-input AWGN,
    # apply receiver IQ/DC and AGC to signal+noise, then quantize directly to temp.
    final_tmp: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=dst.parent)
        os.close(fd); final_tmp = Path(name)
        channel = _ChannelPass(sample_rate_hz, config)
        channel_hash_2 = hashlib.sha256()
        noise_rng = np.random.default_rng(np.random.SeedSequence(config.seed).spawn(2)[1])
        output_hash = hashlib.sha256()
        written = clipped_complex = clipped_components = 0
        pre_power_sum = post_power_sum = quantized_power_sum = 0.0
        analog_signal_power_sum = analog_noise_power_sum = 0.0
        quantized_reference_power_sum = quantized_error_power_sum = 0.0
        output_peak = 0.0
        with final_tmp.open("wb") as out:
            for raw in _chunks(src, config.chunk_samples):
                signal = channel.process(raw)
                channel_hash_2.update(signal.tobytes())
                if noise_power:
                    draws = noise_rng.standard_normal(signal.size * 2).reshape(-1, 2).astype(np.float32)
                    noise = np.float32(math.sqrt(noise_power / 2)) * (draws[:, 0] + 1j * draws[:, 1])
                    noise = noise.astype(np.complex64)
                else:
                    noise = np.zeros(signal.size, np.complex64)
                rx_signal = _iq_transform(signal, config)
                rx_noise = _iq_transform(noise, config)
                dc = np.complex64(config.dc_i + 1j * config.dc_q)
                pre = rx_signal + rx_noise + dc
                post = pre * np.float32(agc_gain)
                reference = (rx_signal + dc) * np.float32(agc_gain)
                pre_power_sum += float(np.vdot(pre, pre).real)
                post_power_sum += float(np.vdot(post, post).real)
                analog_signal_power_sum += float(np.vdot(rx_signal * agc_gain, rx_signal * agc_gain).real)
                analog_noise_power_sum += float(np.vdot(rx_noise * agc_gain, rx_noise * agc_gain).real)
                over_i = np.abs(post.real) > config.clip_level
                over_q = np.abs(post.imag) > config.clip_level
                clipped_complex += int(np.count_nonzero(over_i | over_q))
                clipped_components += int(np.count_nonzero(over_i) + np.count_nonzero(over_q))
                i = np.rint(np.clip(post.real, -config.clip_level, config.clip_level)).astype(np.int8)
                q = np.rint(np.clip(post.imag, -config.clip_level, config.clip_level)).astype(np.int8)
                ri = np.rint(np.clip(reference.real, -config.clip_level, config.clip_level)).astype(np.int8)
                rq = np.rint(np.clip(reference.imag, -config.clip_level, config.clip_level)).astype(np.int8)
                inter = np.empty(signal.size * 2, np.int8); inter[0::2] = i; inter[1::2] = q
                payload = inter.tobytes(); out.write(payload); output_hash.update(payload)
                fi = i.astype(np.float32); fq = q.astype(np.float32)
                fri = ri.astype(np.float32); frq = rq.astype(np.float32)
                quantized_power_sum += float(np.sum(fi * fi + fq * fq))
                quantized_reference_power_sum += float(np.sum(fri * fri + frq * frq))
                quantized_error_power_sum += float(np.sum((fi - fri) ** 2 + (fq - frq) ** 2))
                if signal.size: output_peak = max(output_peak, float(np.max(np.hypot(fi, fq))))
                written += signal.size
            out.flush(); os.fsync(out.fileno())
        if written != count or channel_hash_2.digest() != channel_hash_1.digest():
            raise RuntimeError("deterministic channel regeneration mismatch between passes")
        os.replace(final_tmp, dst); final_tmp = None
        output_bytes = dst.stat().st_size
        filter_doc = {
            "type": "butterworth_lowpass_sos" if channel.sos is not None else None,
            "order": config.frontend_order if channel.sos is not None else None,
            "cutoff_hz": config.frontend_cutoff_hz,
            "normalized_cutoff_to_nyquist": (
                None if config.frontend_cutoff_hz is None else config.frontend_cutoff_hz / (sample_rate_hz / 2)
            ),
            "sos": None if channel.sos is None else channel.sos.tolist(),
        }
        composite_cn0 = None if config.sample_snr_db is None else config.sample_snr_db + 10 * math.log10(sample_rate_hz)
        requested = config.manifest()
        return {
            "layer_model_version": LAYER_MODEL_VERSION,
            "seed": config.seed,
            "profile": config.profile,
            "requested": requested,
            "config": requested,
            "realized": {
                **requested,
                "deterministic_composite_signal_power": channel_power,
                "awgn_complex_variance_frontend_output": noise_power,
                "receiver_agc_applied_gain": agc_gain,
                "equivalent_composite_cn0_db_hz": composite_cn0,
                "approx_equal_power_per_prn_cn0_db_hz_for_4_to_10_visible": (
                    None if composite_cn0 is None else [composite_cn0 - 10 * math.log10(10), composite_cn0 - 10 * math.log10(4)]
                ),
            },
            "measurements": {
                "expected_pre_agc_mean_complex_power": expected_pre_agc_power,
                "pre_agc_mean_complex_power": pre_power_sum / count,
                "post_agc_mean_complex_power": post_power_sum / count,
                "quantized_mean_complex_power": quantized_power_sum / count,
                "achieved_analog_sample_snr_db": _safe_db_ratio(analog_signal_power_sum, analog_noise_power_sum),
                "achieved_quantized_sample_snr_db": (
                    None if config.sample_snr_db is None else _safe_db_ratio(quantized_reference_power_sum, quantized_error_power_sum)
                ),
            },
            "clean_input": {
                "sha256": clean_hash.hexdigest(), "bytes": size, "complex_samples": count,
                "mean_complex_power": clean_power_sum / count, "peak_magnitude": clean_peak,
            },
            "deterministic_channel": {"complex64_sha256": channel_hash_1.hexdigest(), "complex_samples": count},
            "output": {
                "sha256": output_hash.hexdigest(), "bytes": output_bytes, "complex_samples": written,
                "mean_complex_power": quantized_power_sum / count, "peak_magnitude": output_peak,
                "clipped_complex_samples": clipped_complex,
                "clipped_components": clipped_components,
                "clipping_fraction": clipped_complex / count,
            },
            "filter": filter_doc,
            "runtime": {
                "numpy_version": np.__version__, "scipy_version": scipy.__version__,
                "rng": "numpy.random.Generator", "rng_bit_generator": "PCG64",
            },
            "processing": {"passes": 2, "channel_intermediate_bytes": 0, "chunk_samples": config.chunk_samples},
            "layer_order": [
                "optional_composite_multipath", "common_oscillator", "frontend_lowpass",
                "optional_composite_signal_gain_fading", "frontend_output_adc_input_referred_awgn",
                "receiver_iq_imbalance_and_dc", "receiver_agc_signal_plus_noise", "clip_round_s8",
            ],
            "awgn_reference": "frontend-output/ADC-input-referred; added after frontend low-pass",
            "multipath_model": COMPOSITE_CHANNEL_CAVEAT,
            "cn0_caveat": CN0_CAVEAT,
            "reproducibility_caveat": (
                "Chunk-size invariant for the recorded NumPy/SciPy runtime; cross-version byte identity is not claimed."
            ),
        }
    except Exception:
        dst.unlink(missing_ok=True)
        raise
    finally:
        if final_tmp is not None:
            final_tmp.unlink(missing_ok=True)
