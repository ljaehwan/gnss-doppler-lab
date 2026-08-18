"""Authenticated TRACE-R2 native complex nine-tap input for CRISP.

The loader is deliberately narrow: it accepts only the validated little-endian
TRC1MS02 schema and never substitutes magnitude taps, interpolated rows, or
zero-filled channels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Iterator

import numpy as np

MAGIC = b"TRC1MS02"
SCHEMA_VERSION = 2
HEADER_SIZE = 192
RECORD_SIZE = 416
ENDIAN_MARKER = 0x01020304
TAPS = ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")
HEADER_STRUCT = struct.Struct("<8sIIIIdff64s48s9fI")

MEASUREMENT_FIELDS = (
    "dll_discriminator_chips",
    "pll_phase_error_cycles",
    "fll_frequency_error_hz",
    "code_filter_output_chips_s",
    "carrier_filter_output_hz",
    "cn0_db_hz",
    "carrier_lock_test",
    "coherent_integration_s",
)
ACTION_VALUE_FIELDS = (
    "code_nco_rate_chips_s",
    "carrier_doppler_hz",
    "residual_code_phase_chips",
    "residual_carrier_phase_rad",
    "code_phase_step_chips_per_sample",
    "carrier_phase_step_rad_per_sample",
    "code_phase_rate_step_chips_per_sample2",
    "carrier_phase_rate_step_rad_per_sample2",
    "dll_filter_output_chips_s",
    "pll_fll_filter_output_hz",
    "carrier_phase_accumulator_rad",
    "residual_code_phase_samples",
)


def _record_dtype() -> np.dtype:
    fields: list[tuple[str, str]] = [
        ("loop_sequence", "<u8"),
        ("tracking_session_id", "<u8"),
        ("action_used_source_loop_sequence", "<u8"),
        ("raw_interval_start_sample", "<u8"),
        ("raw_interval_end_sample", "<u8"),
        ("receiver_timestamp_s", "<f8"),
        ("integration_duration_s", "<f8"),
        ("channel", "<u4"),
        ("prn", "<u4"),
        ("valid_tracking", "u1"),
        ("valid_lock", "u1"),
        ("data_symbol_boundary", "u1"),
        ("loop_update_boundary", "u1"),
        ("navigation_bit_wipeoff_applied", "u1"),
        ("pull_in_transitory", "u1"),
        ("receiver_state", "u1"),
        ("reserved", "u1"),
    ]
    for tap in TAPS:
        fields.extend(((f"{tap}_i", "<f4"), (f"{tap}_q", "<f4")))
    fields.extend((name, "<f8") for name in MEASUREMENT_FIELDS)
    for prefix in ("action_used", "action_next"):
        fields.extend((f"{prefix}_{name}", "<f8") for name in ACTION_VALUE_FIELDS)
        fields.append((f"{prefix}_interval_length_samples", "<u8"))
    dtype = np.dtype(fields, align=False)
    if dtype.itemsize != RECORD_SIZE:
        raise AssertionError(f"TRACE dtype {dtype.itemsize} != {RECORD_SIZE}")
    return dtype


RECORD_DTYPE = _record_dtype()


@dataclass(frozen=True)
class NativeHeader:
    path: Path
    sample_rate_hz: float
    tap_spacing_chips: float
    coherent_integration_s: float
    scenario_id: str
    receiver_source_base_commit: str
    tap_offsets_chips: tuple[float, ...]


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_header(path: Path | str) -> NativeHeader:
    path = Path(path)
    with path.open("rb") as stream:
        payload = stream.read(HEADER_SIZE)
    if len(payload) != HEADER_SIZE:
        raise ValueError(f"{path}: truncated TRACE header")
    values = HEADER_STRUCT.unpack(payload)
    if values[:5] != (MAGIC, SCHEMA_VERSION, HEADER_SIZE, RECORD_SIZE, ENDIAN_MARKER):
        raise ValueError(f"{path}: unsupported TRACE schema")
    scenario = values[8].split(b"\0", 1)[0].decode("utf-8", errors="strict")
    commit = values[9].split(b"\0", 1)[0].decode("ascii", errors="strict")
    return NativeHeader(
        path=path,
        sample_rate_hz=float(values[5]),
        tap_spacing_chips=float(values[6]),
        coherent_integration_s=float(values[7]),
        scenario_id=scenario,
        receiver_source_base_commit=commit,
        tap_offsets_chips=tuple(float(v) for v in values[10:19]),
    )


def read_records(path: Path | str) -> tuple[NativeHeader, np.ndarray]:
    path = Path(path)
    header = read_header(path)
    payload = path.stat().st_size - HEADER_SIZE
    if payload < 0 or payload % RECORD_SIZE:
        raise ValueError(f"{path}: non-integral TRACE record payload")
    count = payload // RECORD_SIZE
    records = np.memmap(path, dtype=RECORD_DTYPE, mode="r", offset=HEADER_SIZE, shape=(count,))
    return header, records


def complex_taps(records: np.ndarray, tap_indices: tuple[int, ...] | None = None) -> np.ndarray:
    indices = tap_indices if tap_indices is not None else tuple(range(len(TAPS)))
    result = np.empty((len(records), len(indices)), dtype=np.complex128)
    for column, index in enumerate(indices):
        tap = TAPS[index]
        result[:, column] = records[f"{tap}_i"] + 1j * records[f"{tap}_q"]
    return result


def scenario_files(directory: Path | str) -> list[Path]:
    files = sorted(Path(directory).glob("trace_native_1ms_ch_*.bin"))
    if len(files) < 4:
        raise ValueError(f"{directory}: fewer than four native TRACE channels")
    return files


def validate_scenario(directory: Path | str, expected: str) -> dict[str, object]:
    manifest_path = Path(directory) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_files = {Path(row["path"]).name: row for row in manifest["dump_files"]}
    summaries: list[dict[str, object]] = []
    prns: set[int] = set()
    receiver_commits: set[str] = set()
    sample_rates: set[float] = set()
    spacings: set[float] = set()
    for path in scenario_files(directory):
        header, records = read_records(path)
        if header.scenario_id != expected:
            raise ValueError(f"{path}: scenario {header.scenario_id!r} != {expected!r}")
        bound = expected_files.get(path.name)
        if bound is None or int(bound["size_bytes"]) != path.stat().st_size:
            raise ValueError(f"{path}: absent or size-mismatched manifest binding")
        observed_hash = sha256_file(path)
        if observed_hash != bound["sha256"]:
            raise ValueError(f"{path}: dump SHA mismatch")
        physical = records[(records["valid_tracking"] == 1) & (records["loop_update_boundary"] == 1)]
        prns.update(int(v) for v in np.unique(physical["prn"]))
        receiver_commits.add(header.receiver_source_base_commit)
        sample_rates.add(header.sample_rate_hz)
        spacings.add(header.tap_spacing_chips)
        summaries.append(
            {
                "path": str(path),
                "sha256": observed_hash,
                "size_bytes": path.stat().st_size,
                "record_count": int(len(records)),
                "physical_record_count": int(len(physical)),
                "prns": sorted(int(v) for v in np.unique(physical["prn"])),
            }
        )
    if len(receiver_commits) != 1 or len(sample_rates) != 1 or len(spacings) != 1:
        raise ValueError(f"{directory}: inconsistent native headers")
    return {
        "scenario": expected,
        "status": "PASS",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "raw_iq": manifest["raw_iq"],
        "receiver_executable": manifest["receiver_executable"],
        "receiver_config_sha256": manifest["receiver_config_sha256"],
        "receiver_patch_sha256": manifest["receiver_patch_sha256"],
        "receiver_source_base_commit": next(iter(receiver_commits)),
        "sample_rate_hz": next(iter(sample_rates)),
        "sample_format": "signed little-endian interleaved int16 I,Q",
        "tap_spacing_chips": next(iter(spacings)),
        "tap_offsets_chips": list(read_header(scenario_files(directory)[0]).tap_offsets_chips),
        "available_prns": sorted(prns),
        "files": summaries,
    }


def iter_record_chunks(records: np.ndarray, chunk_records: int) -> Iterator[tuple[int, np.ndarray]]:
    for start in range(0, len(records), chunk_records):
        yield start, records[start : min(len(records), start + chunk_records)]
