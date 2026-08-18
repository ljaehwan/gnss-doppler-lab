"""Frozen raw-IQ reconstruction, transient injection, replay, and cache primitives."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Iterable

import numpy as np

from .acquisition_surface import gps_l1ca_code
from .mirage_r1 import DELAYS, SCALES, epoch_features, nav_signs_for_epochs
from .mosaic_iq_injector_int16 import decode_interleaved_int16, inject_payload
from .mosaic_raw_recorrelation import correlate_nine_taps
from .trace_native_1ms import read_records, sha256_file

BYTES_PER_SAMPLE = 4
CHUNK_SAMPLES = 250_000


def load_mapping(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def trace_for_prn(directory: Path, prn: int) -> Path:
    matches = []
    for path in sorted(directory.glob("trace_native_1ms_ch_*.bin")):
        _, records = read_records(path, mmap=True)
        if len(records) and int(records["prn"][-1]) == int(prn):
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"TRACE lookup failed for PRN {prn}: {matches}")
    return matches[0]


def role_record_slice(trace_path: Path, lo: int, hi: int) -> np.ndarray:
    _, records = read_records(trace_path, mmap=True)
    starts = records["raw_interval_start_sample"].astype(np.int64)
    indices = np.flatnonzero((starts >= lo) & (records["raw_interval_end_sample"] <= hi))
    if not len(indices):
        raise ValueError("role has no TRACE rows")
    if not np.all(np.diff(indices) == 1):
        raise ValueError("role TRACE slice is not contiguous")
    return indices


def reconstruct_records(raw_path: Path, trace_path: Path, indices: Iterable[int], *,
                        mapping_rows: list[dict[str, str]], dataset: str,
                        absolute_sample_offset: int = 0, raw_transform=None,
                        chunk_epochs: int = 500) -> tuple[np.ndarray, dict[str, object]]:
    header, records = read_records(trace_path, mmap=True)
    selected = np.asarray(tuple(indices), dtype=np.int64)
    if not len(selected) or not np.all(np.diff(selected) == 1):
        raise ValueError("reconstruction indices must be one nonempty contiguous run")
    prn = int(records[selected[0]]["prn"])
    starts_local = records[selected]["raw_interval_start_sample"].astype(np.int64)
    starts_absolute = starts_local + int(absolute_sample_offset)
    signs = nav_signs_for_epochs(mapping_rows, dataset, prn, starts_absolute)
    taps = np.empty((len(selected), 9), dtype=np.complex64)
    power_sum = 0.0
    sample_count = 0
    for begin in range(0, len(selected), chunk_epochs):
        take = selected[begin:begin + chunk_epochs]
        lo = int(records[take[0]]["raw_interval_start_sample"])
        hi = int(records[take[-1]]["raw_interval_end_sample"])
        with raw_path.open("rb") as stream:
            stream.seek(lo * BYTES_PER_SAMPLE)
            payload = stream.read((hi - lo) * BYTES_PER_SAMPLE)
        if len(payload) != (hi - lo) * BYTES_PER_SAMPLE:
            raise EOFError("raw reconstruction chunk truncated")
        chunk = decode_interleaved_int16(payload)
        if raw_transform is not None:
            chunk = raw_transform(chunk, lo, begin)
        power_sum += float(np.vdot(chunk, chunk).real)
        sample_count += len(chunk)
        for j, index in enumerate(take):
            a = int(records[index]["raw_interval_start_sample"]) - lo
            b = int(records[index]["raw_interval_end_sample"]) - lo
            value = correlate_nine_taps(chunk[a:b], prn=prn, action=records[index],
                                        tap_offsets_chips=header.tap_offsets_chips)
            taps[begin + j] = (value * signs[begin + j]).astype(np.complex64)
    return taps, {
        "prn": prn, "epoch_count": len(selected), "raw_start_sample": int(starts_local[0]),
        "raw_end_sample_exclusive": int(records[selected[-1]]["raw_interval_end_sample"]),
        "mean_raw_power": power_sum / max(sample_count, 1),
        "mean_cn0_db_hz": float(np.mean(records[selected]["cn0_db_hz"])),
        "valid_tracking_fraction": float(np.mean(records[selected]["valid_tracking"] == 1)),
        "valid_lock_fraction": float(np.mean(records[selected]["valid_lock"] == 1)),
    }


def role_minor_bundle(raw_path: Path, trace_path: Path, role: dict[str, int], *,
                      mapping_rows: list[dict[str, str]], dataset: str) -> dict[str, object]:
    indices = role_record_slice(trace_path, role["raw_start_sample"], role["raw_end_sample_exclusive"])
    count = len(indices) - len(indices) % 500
    indices = indices[:count]
    taps, audit = reconstruct_records(raw_path, trace_path, indices, mapping_rows=mapping_rows, dataset=dataset)
    windows = []
    for i in range(0, len(taps), 500):
        feature = epoch_features(taps[i:i + 500])
        windows.append({
            "ordinal": i // 500,
            "raw_start_sample": audit["raw_start_sample"] if i == 0 else None,
            "minors": np.stack([feature["minors"][scale] for scale in SCALES]),
            "energy": feature["energy"],
            "svd": np.asarray([feature["svd"][scale] for scale in SCALES]),
            "magnitude_minor": np.asarray([feature["magnitude_minor"][scale] for scale in SCALES]),
        })
    return {"audit": audit, "windows": windows}


class AuthenticatedReplica:
    def __init__(self, dataset: str, prn: int, trace_path: Path,
                 mapping_rows: list[dict[str, str]], fs: int):
        self.dataset = dataset
        self.prn = int(prn)
        self.fs = int(fs)
        _, self.records = read_records(trace_path, mmap=True)
        self.starts = self.records["raw_interval_start_sample"].astype(np.int64)
        self.ends = self.records["raw_interval_end_sample"].astype(np.int64)
        self.code = gps_l1ca_code(prn).astype(np.float64)
        rows = sorted((row for row in mapping_rows if row["dataset"] == dataset and int(row["prn"]) == prn),
                      key=lambda row: int(row["raw_start_sample"]))
        self.nav_starts = np.asarray([int(row["raw_start_sample"]) for row in rows], np.int64)
        self.nav_ends = np.asarray([int(row["raw_end_sample_exclusive"]) for row in rows], np.int64)
        self.nav_signs = np.asarray([int(row["bit_value_pm1"]) for row in rows], np.float64)

    def render(self, absolute_start: int, count: int, *, delay_chips: float = 0,
               delta_f_hz: float = 0, phase_rad: float = 0,
               phase_reference_sample: int | None = None) -> np.ndarray:
        positions = int(absolute_start) + np.arange(count, dtype=np.int64)
        rows = np.searchsorted(self.starts, positions, side="right") - 1
        safe = np.maximum(rows, 0)
        if np.any(rows < 0) or np.any(positions >= self.ends[safe]):
            raise ValueError(f"PRN {self.prn}: outside TRACE NCO support")
        nav = np.searchsorted(self.nav_starts, positions, side="right") - 1
        nsafe = np.maximum(nav, 0)
        if np.any(nav < 0) or np.any(positions >= self.nav_ends[nsafe]):
            raise ValueError(f"PRN {self.prn}: outside authenticated NAV support")
        local = positions - self.starts[rows]
        code_phase = (local * self.records["action_used_code_phase_step_chips_per_sample"][rows]
                      - self.records["action_used_residual_code_phase_chips"][rows] + float(delay_chips))
        code = self.code[np.floor(code_phase).astype(np.int64) % 1023]
        phase = (self.records["action_used_residual_carrier_phase_rad"][rows]
                 + local * self.records["action_used_carrier_phase_step_rad_per_sample"][rows])
        if phase_reference_sample is None:
            phase_reference_sample = absolute_start
        extra = float(phase_rad) + 2 * np.pi * float(delta_f_hz) * (positions - phase_reference_sample) / self.fs
        return (code * self.nav_signs[nav] * np.exp(1j * (phase + extra))).astype(np.complex128)


def estimate_authentic_alpha(raw_path: Path, replica: AuthenticatedReplica,
                             anchor: int, duration_samples: int) -> complex:
    lo = anchor - duration_samples
    numerator = 0j
    denominator = 0.0
    with raw_path.open("rb") as stream:
        stream.seek(lo * BYTES_PER_SAMPLE)
        absolute = lo
        while absolute < anchor:
            count = min(CHUNK_SAMPLES, anchor - absolute)
            clean = decode_interleaved_int16(stream.read(count * BYTES_PER_SAMPLE))
            reference = replica.render(absolute, count)
            numerator += np.vdot(reference, clean)
            denominator += float(np.vdot(reference, reference).real)
            absolute += count
    if denominator <= 0:
        raise ValueError("authentic amplitude denominator is zero")
    return complex(numerator / denominator)


def raised_cosine_envelope(t: np.ndarray, duration_s: float = 2.0, edge_s: float = .25) -> np.ndarray:
    x = np.asarray(t, float)
    env = np.zeros_like(x)
    ramp = (x >= 0) & (x < edge_s)
    env[ramp] = .5 - .5 * np.cos(np.pi * x[ramp] / edge_s)
    hold = (x >= edge_s) & (x < duration_s - edge_s)
    env[hold] = 1
    release = (x >= duration_s - edge_s) & (x < duration_s)
    env[release] = .5 + .5 * np.cos(np.pi * (x[release] - (duration_s - edge_s)) / edge_s)
    return env


def generate_transient(raw_path: Path, output_path: Path, *, dataset: str, fs: int,
                       anchor: int, targets: list[int], rho_db: float,
                       delay_chips: float, doppler_hz: float, phase_rad: float,
                       trace_dir: Path, mapping_rows: list[dict[str, str]],
                       preroll_s: int = 30) -> dict[str, object]:
    started = time.monotonic()
    source_start = anchor - preroll_s * fs
    source_end = anchor + 2 * fs
    replicas = {prn: AuthenticatedReplica(dataset, prn, trace_for_prn(trace_dir, prn), mapping_rows, fs)
                for prn in targets}
    alpha = {prn: estimate_authentic_alpha(raw_path, replica, anchor, fs // 4)
             for prn, replica in replicas.items()}
    coefficient = {prn: alpha[prn] * 10 ** (rho_db / 20) * np.exp(1j * phase_rad) for prn in targets}
    output_path.parent.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    clipped = 0
    injected_samples = 0
    clean_energy = 0.0
    output_energy = 0.0
    with raw_path.open("rb") as source, output_path.open("xb") as output:
        source.seek(source_start * BYTES_PER_SAMPLE)
        absolute = source_start
        while absolute < source_end:
            count = min(CHUNK_SAMPLES, source_end - absolute)
            payload = source.read(count * BYTES_PER_SAMPLE)
            if len(payload) != count * BYTES_PER_SAMPLE:
                raise EOFError("transient source truncated")
            if absolute + count <= anchor:
                encoded = payload
            else:
                clean = decode_interleaved_int16(payload)
                positions = absolute + np.arange(count)
                env = raised_cosine_envelope((positions - anchor) / fs)
                counterfeit = np.zeros(count, np.complex128)
                for prn, replica in replicas.items():
                    counterfeit += coefficient[prn] * replica.render(
                        absolute, count, delay_chips=delay_chips, delta_f_hz=doppler_hz,
                        phase_rad=0, phase_reference_sample=anchor) * env
                encoded, metrics = inject_payload(payload, counterfeit)
                quantized = decode_interleaved_int16(encoded)
                clipped += int(metrics["clipped_sample_count"])
                injected_samples += int(np.count_nonzero(env))
                clean_energy += float(np.vdot(clean, clean).real)
                output_energy += float(np.vdot(quantized, quantized).real)
            output.write(encoded)
            digest.update(encoded)
            absolute += count
    return {
        "path": str(output_path), "sha256": digest.hexdigest(), "size_bytes": output_path.stat().st_size,
        "source_absolute_range": [source_start, source_end], "local_injection_range": [preroll_s * fs, (preroll_s + 2) * fs],
        "targets": targets, "alpha_auth": {str(p): [alpha[p].real, alpha[p].imag] for p in targets},
        "requested_relative_power_db": rho_db, "clipped_sample_count": clipped,
        "clipping_ratio": clipped / max(injected_samples, 1),
        "clean_injection_region_rms": float(np.sqrt(clean_energy / max(2 * fs, 1))),
        "output_injection_region_rms": float(np.sqrt(output_energy / max(2 * fs, 1))),
        "runtime_seconds": time.monotonic() - started,
    }


def render_receiver_config(base_path: Path, raw_path: Path) -> str:
    output = []
    changed = 0
    for line in base_path.read_text().splitlines():
        if line.startswith("SignalSource.filename="):
            output.append(f"SignalSource.filename={raw_path}")
            changed += 1
        else:
            output.append(line)
    if changed != 1:
        raise ValueError("receiver config filename allowlist violation")
    return "\n".join(output) + "\n"


def run_receiver(executable: Path, base_config: Path, raw_path: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    config = output_dir / "receiver.conf"
    config.write_text(render_receiver_config(base_config, raw_path))
    started = time.monotonic()
    with (output_dir / "receiver.log").open("wb") as log:
        result = subprocess.run([str(executable), f"--config_file={config}", "--keyboard=false"], cwd=output_dir,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    traces = sorted(output_dir.glob("trace_native_1ms_ch_*.bin"))
    return {
        "exit_code": result.returncode, "runtime_seconds": time.monotonic() - started,
        "receiver_binary_sha256": sha256_file(executable), "config_sha256": sha256_file(config),
        "trace_count": len(traces), "log_path": str(output_dir / "receiver.log"),
        "status": "PASS" if result.returncode == 0 and len(traces) >= 5 else "FAIL",
    }


def compact_receiver_traces(receiver_dir: Path, compact_dir: Path) -> list[dict[str, object]]:
    compact_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(receiver_dir.glob("trace_native_1ms_ch_*.bin")):
        target = compact_dir / path.name
        shutil.copy2(path, target)
        _, records = read_records(target, mmap=True)
        rows.append({"path": str(target), "sha256": sha256_file(target), "rows": len(records),
                     "prn": int(records["prn"][-1]) if len(records) else None})
    return rows
