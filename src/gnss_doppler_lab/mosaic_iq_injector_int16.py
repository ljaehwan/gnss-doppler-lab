"""Production-format little-endian interleaved int16 I/Q injection helpers."""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

BYTES_PER_COMPLEX_SAMPLE = 4
INT16_MIN = -32768
INT16_MAX = 32767


def decode_interleaved_int16(payload: bytes) -> np.ndarray:
    if len(payload) % BYTES_PER_COMPLEX_SAMPLE:
        raise ValueError("payload is not an integral number of complex int16 I/Q samples")
    values = np.frombuffer(payload, dtype="<i2")
    return values[0::2].astype(np.float64) + 1j * values[1::2].astype(np.float64)


def encode_interleaved_int16(iq: np.ndarray) -> tuple[bytes, dict[str, float | int]]:
    values = np.asarray(iq, dtype=np.complex128).reshape(-1)
    rounded_i = np.rint(values.real)
    rounded_q = np.rint(values.imag)
    clipped_i = (rounded_i < INT16_MIN) | (rounded_i > INT16_MAX)
    clipped_q = (rounded_q < INT16_MIN) | (rounded_q > INT16_MAX)
    clipped_samples = clipped_i | clipped_q
    interleaved = np.empty(values.size * 2, dtype="<i2")
    interleaved[0::2] = np.clip(rounded_i, INT16_MIN, INT16_MAX).astype("<i2")
    interleaved[1::2] = np.clip(rounded_q, INT16_MIN, INT16_MAX).astype("<i2")
    return interleaved.tobytes(), {
        "complex_sample_count": int(values.size),
        "clipped_component_count": int(clipped_i.sum() + clipped_q.sum()),
        "clipped_sample_count": int(clipped_samples.sum()),
        "clipping_rate": float(clipped_samples.mean()) if values.size else 0.0,
        "headroom_counts": float(INT16_MAX - max(np.max(np.abs(rounded_i)) if values.size else 0, np.max(np.abs(rounded_q)) if values.size else 0)),
    }


def inject_payload(payload: bytes, counterfeit: np.ndarray) -> tuple[bytes, dict[str, float | int | bool]]:
    addition = np.asarray(counterfeit, dtype=np.complex128).reshape(-1)
    if len(payload) != addition.size * BYTES_PER_COMPLEX_SAMPLE:
        raise ValueError("input/output sample count contract violated")
    if not np.any(addition):
        return payload, {"complex_sample_count": int(addition.size), "clipped_component_count": 0,
                         "clipped_sample_count": 0, "clipping_rate": 0.0,
                         "headroom_counts": float("nan"), "byte_identity": True}
    output, metrics = encode_interleaved_int16(decode_interleaved_int16(payload) + addition)
    if len(output) != len(payload):
        raise AssertionError("int16 injector changed sample count")
    return output, {**metrics, "byte_identity": output == payload}


def read_complex_int16(path: str | Path, start_sample: int, sample_count: int) -> np.ndarray:
    if start_sample < 0 or sample_count < 0:
        raise ValueError("negative sample range")
    with Path(path).open("rb") as stream:
        stream.seek(start_sample * BYTES_PER_COMPLEX_SAMPLE)
        payload = stream.read(sample_count * BYTES_PER_COMPLEX_SAMPLE)
    if len(payload) != sample_count * BYTES_PER_COMPLEX_SAMPLE:
        raise EOFError("requested int16 I/Q range exceeds file bounds")
    return decode_interleaved_int16(payload)


def iter_complex_int16(path: str | Path, start_sample: int, sample_count: int, *, chunk_samples: int) -> Iterator[tuple[int, bytes]]:
    if chunk_samples <= 0 or start_sample < 0 or sample_count < 0:
        raise ValueError("invalid streaming range")
    remaining = sample_count
    absolute = start_sample
    with Path(path).open("rb") as stream:
        stream.seek(start_sample * BYTES_PER_COMPLEX_SAMPLE)
        while remaining:
            count = min(remaining, chunk_samples)
            payload = stream.read(count * BYTES_PER_COMPLEX_SAMPLE)
            if len(payload) != count * BYTES_PER_COMPLEX_SAMPLE:
                raise EOFError("truncated int16 I/Q stream")
            yield absolute, payload
            absolute += count
            remaining -= count


def write_complex_int16(stream: BinaryIO, iq: np.ndarray) -> dict[str, float | int]:
    payload, metrics = encode_interleaved_int16(iq)
    written = stream.write(payload)
    if written != len(payload):
        raise IOError("short int16 I/Q write")
    return metrics


def validate_file_format(path: str | Path, *, probe_samples: int = 256) -> dict[str, object]:
    source = Path(path)
    size = source.stat().st_size
    if size % BYTES_PER_COMPLEX_SAMPLE:
        raise ValueError("raw file size is incompatible with complex int16 I/Q")
    count = min(probe_samples, size // BYTES_PER_COMPLEX_SAMPLE)
    payload = source.read_bytes()[:count * BYTES_PER_COMPLEX_SAMPLE] if size <= 16 * 1024 * 1024 else _read_prefix(source, count)
    decoded = decode_interleaved_int16(payload)
    encoded, metrics = encode_interleaved_int16(decoded)
    if encoded != payload:
        raise ValueError("little-endian I/Q round-trip failed")
    return {"sample_format": "little-endian interleaved signed int16 I/Q", "bytes_per_complex_sample": 4,
            "size_bytes": size, "complex_sample_count": size // 4, "probe_samples": count,
            "iq_ordering": "I_then_Q", "probe_byte_roundtrip": True, "clipped_sample_count": metrics["clipped_sample_count"]}


def _read_prefix(path: Path, count: int) -> bytes:
    with path.open("rb") as stream:
        return stream.read(count * BYTES_PER_COMPLEX_SAMPLE)
