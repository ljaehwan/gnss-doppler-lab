"""Parallel raw-IQ residual-token extraction for the frozen CORA protocol."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from .cora_common_origin import raw_residual_token
from .cora_trace_adapter import EpochAction, LegacyTraceIndex, NativeTraceIndex, validate_epoch_sequence
from .trace_native_1ms import RECORD_DTYPE


_RAW_FDS: dict[str, int] = {}


def _raw_iq(path: str, start: int, count: int) -> np.ndarray:
    fd = _RAW_FDS.get(path)
    if fd is None:
        fd = os.open(path, os.O_RDONLY); _RAW_FDS[path] = fd
    payload = os.pread(fd, count * 4, start * 4)
    if len(payload) != count * 4:
        raise ValueError("raw IQ read crossed source boundary")
    raw = np.frombuffer(payload, dtype="<i2")
    return raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)


def _worker(task: tuple[str, int, int, int, bytes, float]) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    raw_path, prn, start, end, action_bytes, sample_rate_hz = task
    action = np.frombuffer(action_bytes, dtype=RECORD_DTYPE, count=1)[0]
    iq = _raw_iq(raw_path, start, end - start)
    token, meta = raw_residual_token(iq, prn=prn, action=action, sample_rate_hz=sample_rate_hz)
    return token, (meta["raw_rms"], meta["residual_rms"], meta["token_prequotient_norm"], meta["ls_coefficient_abs"])


def fixed_epoch_targets(window_start_s: float) -> np.ndarray:
    return window_start_s + (np.arange(32, dtype=float) + 0.5) * 2.0 / 32.0


def extract_windows(
    *, raw_path: str | Path, trace_path: str | Path, adapter: str, sample_rate_hz: float,
    prns: Iterable[int], window_starts_s: Iterable[float], workers: int = 8,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    raw_path = str(raw_path); trace_path = str(trace_path); prns = tuple(map(int, prns)); window_starts_s = tuple(window_starts_s)
    index = NativeTraceIndex(trace_path) if adapter == "native" else LegacyTraceIndex(trace_path, sample_rate_hz)
    windows: list[float] = []; actions: list[list[list[EpochAction]]] = []; exclusions = []
    for start in window_starts_s:
        per_epoch = []
        try:
            for target in fixed_epoch_targets(float(start)):
                per_epoch.append([index.select(prn, float(target)) for prn in prns])
        except (KeyError, ValueError) as error:
            exclusions.append({"window_start_s": float(start), "reason": str(error)})
            continue
        windows.append(float(start)); actions.append(per_epoch)
    flat = [row for window in actions for epoch in window for row in epoch]
    raw_samples = Path(raw_path).stat().st_size // 4
    audit = validate_epoch_sequence(flat, raw_samples)
    if audit["status"] != "PASS":
        raise ValueError(f"TRACE/raw lineage failed: {audit}")
    tasks = [
        (raw_path, row.prn, row.raw_start_sample, row.raw_end_sample, row.action.tobytes(), float(sample_rate_hz))
        for row in flat
    ]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for number, result in enumerate(executor.map(_worker, tasks, chunksize=16), 1):
            results.append(result)
            if number % 2000 == 0 or number == len(tasks):
                print(f"raw-token extraction {number}/{len(tasks)}", flush=True)
    nw, ne, np_, nk = len(windows), 32, len(prns), 9
    tokens = np.asarray([value[0] for value in results], dtype=np.complex128).reshape(nw, ne, np_, nk)
    metadata = np.asarray([value[1] for value in results], dtype=float).reshape(nw, ne, np_, 4)
    cn0 = np.asarray([row.cn0_db_hz for row in flat], dtype=float).reshape(nw, ne, np_)
    doppler = np.asarray([row.carrier_doppler_hz for row in flat], dtype=float).reshape(nw, ne, np_)
    doppler_increment = np.diff(doppler, axis=1, prepend=doppler[:, :1, :])
    context = np.stack((cn0, metadata[..., 0], metadata[..., 1], metadata[..., 2],
                        doppler_increment, np.full_like(cn0, len(prns))), axis=-1)
    arrays = {
        "tokens": tokens, "context": context, "window_start_s": np.asarray(windows),
        "prns": np.asarray(prns, dtype=np.int16), "raw_rms": metadata[..., 0],
        "residual_rms": metadata[..., 1], "token_prequotient_norm": metadata[..., 2],
        "ls_coefficient_abs": metadata[..., 3], "cn0_db_hz": cn0, "doppler_hz": doppler,
    }
    provenance = {
        "adapter": adapter, "trace_path": trace_path, "raw_path": raw_path,
        "requested_window_count": len(window_starts_s), "retained_window_count": len(windows),
        "excluded_windows": exclusions, "epoch_count": len(flat), "prns": list(prns),
        "lineage_audit": audit,
        "source_paths": sorted({row.source_path for row in flat}),
        "raw_sample_bounds": [min(row.raw_start_sample for row in flat), max(row.raw_end_sample for row in flat)],
    }
    return arrays, provenance
