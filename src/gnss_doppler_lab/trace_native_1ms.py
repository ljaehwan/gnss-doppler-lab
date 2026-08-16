"""Authenticated native-1-ms TRACE receiver dump adapter.

This module only adapts the TRACE-R2 binary receiver schema into the frozen
TRACE-R1 ``TracePairs`` representation.  It does not change score math,
thresholds, gates, pooling, ablations, or verdict criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from typing import Iterable

import numpy as np

from .trace_action_warp import prompt_normalize, receiver_action, warp_complex_taps
from .trace_equivariance import TracePairs

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
        raise AssertionError(f"native TRACE dtype is {dtype.itemsize}, expected {RECORD_SIZE}")
    return dtype


RECORD_DTYPE = _record_dtype()


@dataclass(frozen=True)
class NativeDumpHeader:
    path: Path
    schema_version: int
    header_size: int
    record_size: int
    endian_marker: int
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


def read_header(path: Path | str) -> NativeDumpHeader:
    path = Path(path)
    with path.open("rb") as stream:
        payload = stream.read(HEADER_STRUCT.size)
    if len(payload) != HEADER_SIZE:
        raise ValueError(f"{path}: truncated TRACE header")
    unpacked = HEADER_STRUCT.unpack(payload)
    magic, version, header_size, record_size, marker, fs, spacing, coherent = unpacked[:8]
    if magic != MAGIC:
        raise ValueError(f"{path}: invalid TRACE magic {magic!r}")
    if (version, header_size, record_size, marker) != (
        SCHEMA_VERSION,
        HEADER_SIZE,
        RECORD_SIZE,
        ENDIAN_MARKER,
    ):
        raise ValueError(f"{path}: unsupported TRACE schema tuple")
    scenario = unpacked[8].split(b"\0", 1)[0].decode("utf-8", errors="strict")
    commit = unpacked[9].split(b"\0", 1)[0].decode("ascii", errors="strict")
    offsets = tuple(float(value) for value in unpacked[10:19])
    return NativeDumpHeader(
        path=path,
        schema_version=version,
        header_size=header_size,
        record_size=record_size,
        endian_marker=marker,
        sample_rate_hz=float(fs),
        tap_spacing_chips=float(spacing),
        coherent_integration_s=float(coherent),
        scenario_id=scenario,
        receiver_source_base_commit=commit,
        tap_offsets_chips=offsets,
    )


def read_records(path: Path | str, *, mmap: bool = True) -> tuple[NativeDumpHeader, np.ndarray]:
    path = Path(path)
    header = read_header(path)
    payload_size = path.stat().st_size - header.header_size
    if payload_size < 0 or payload_size % header.record_size:
        raise ValueError(f"{path}: record payload is not an integral schema-v2 record count")
    count = payload_size // header.record_size
    if mmap:
        records = np.memmap(path, dtype=RECORD_DTYPE, mode="r", offset=header.header_size, shape=(count,))
    else:
        with path.open("rb") as stream:
            stream.seek(header.header_size)
            records = np.fromfile(stream, dtype=RECORD_DTYPE, count=count)
    return header, records


def complex_taps(records: np.ndarray) -> np.ndarray:
    taps = np.empty((len(records), len(TAPS)), dtype=np.complex128)
    for index, tap in enumerate(TAPS):
        taps[:, index] = records[f"{tap}_i"] + 1j * records[f"{tap}_q"]
    return taps


def _action_matrix(records: np.ndarray, prefix: str) -> np.ndarray:
    return np.column_stack([records[f"{prefix}_{name}"] for name in ACTION_VALUE_FIELDS])


def validate_dump_files(
    paths: Iterable[Path | str],
    *,
    expected_scenario_id: str | None = None,
    minimum_prns: int = 4,
) -> dict[str, object]:
    """Validate native cadence, causal links, finite values, and multi-PRN support."""
    files = [Path(path) for path in paths]
    if not files:
        raise ValueError("no native TRACE dump files")
    summaries: list[dict[str, object]] = []
    epoch_prns: dict[int, set[int]] = {}
    scenario_ids: set[str] = set()
    causal_pairs = 0
    causal_value_mismatches = 0
    causal_sequence_mismatches = 0
    consume_span_mismatches = 0
    current_span_mismatches = 0
    timestamp_mismatches = 0
    duration_mismatches = 0
    finite_failures = 0
    zero_tap_rows = 0
    zero_action_rows = 0
    repeated_tap_rows = 0
    repeated_action_rows = 0
    native_pairs = 0
    cadence_pairs = 0
    reassignment_failures = 0

    for path in sorted(files):
        header, records = read_records(path)
        scenario_ids.add(header.scenario_id)
        if expected_scenario_id is not None and header.scenario_id != expected_scenario_id:
            raise ValueError(f"{path}: scenario {header.scenario_id!r} != {expected_scenario_id!r}")
        if not len(records):
            summaries.append({"path": str(path), "record_count": 0, "sha256": sha256_file(path)})
            continue
        taps = complex_taps(records)
        measurements = np.column_stack([records[name] for name in MEASUREMENT_FIELDS])
        used = _action_matrix(records, "action_used")
        next_action = _action_matrix(records, "action_next")
        finite_failures += int((~np.isfinite(taps.real).all(axis=1) | ~np.isfinite(taps.imag).all(axis=1)).sum())
        finite_failures += int((~np.isfinite(measurements).all(axis=1)).sum())
        finite_failures += int((~np.isfinite(used).all(axis=1) | ~np.isfinite(next_action).all(axis=1)).sum())
        zero_tap_rows += int(np.all(taps == 0.0, axis=1).sum())

        spans = records["raw_interval_end_sample"] - records["raw_interval_start_sample"]
        current_span_mismatches += int(
            (spans != records["action_used_interval_length_samples"]).sum()
        )
        timestamp_mismatches += int(
            (~np.isclose(
                records["receiver_timestamp_s"],
                records["raw_interval_start_sample"].astype(np.float64) / header.sample_rate_hz,
                rtol=0.0,
                atol=1e-12,
            )).sum()
        )
        duration_mismatches += int(
            (~np.isclose(
                records["integration_duration_s"],
                spans.astype(np.float64) / header.sample_rate_hz,
                rtol=0.0,
                atol=1e-12,
            )).sum()
        )

        same_track = (
            (records["tracking_session_id"][1:] == records["tracking_session_id"][:-1])
            & (records["prn"][1:] == records["prn"][:-1])
            & (records["loop_sequence"][1:] == records["loop_sequence"][:-1] + 1)
        )
        pair_indices = np.flatnonzero(same_track) + 1
        causal_pairs += len(pair_indices)
        if len(pair_indices):
            previous = pair_indices - 1
            causal_sequence_mismatches += int(
                (records["action_used_source_loop_sequence"][pair_indices] != records["loop_sequence"][previous]).sum()
            )
            causal_value_mismatches += int(
                np.any(used[pair_indices] != next_action[previous], axis=1).sum()
            )
            causal_value_mismatches += int(
                (records["action_used_interval_length_samples"][pair_indices]
                 != records["action_next_interval_length_samples"][previous]).sum()
            )
            start_delta = (
                records["raw_interval_start_sample"][pair_indices]
                - records["raw_interval_start_sample"][previous]
            )
            consume_span_mismatches += int(
                (start_delta != records["action_next_interval_length_samples"][previous]).sum()
            )
            cadence_pairs += len(pair_indices)
            dt = start_delta.astype(np.float64) / header.sample_rate_hz
            native_pairs += int(((dt >= 0.0009) & (dt <= 0.0011)).sum())
            repeated_tap_rows += int(np.all(taps[pair_indices] == taps[previous], axis=1).sum())
            repeated_action_rows += int(
                np.all(next_action[pair_indices] == next_action[previous], axis=1).sum()
            )

        first_in_session = np.r_[True, records["tracking_session_id"][1:] != records["tracking_session_id"][:-1]]
        reassignment_failures += int(
            (records["loop_sequence"][first_in_session] != 0).sum()
            + (records["action_used_source_loop_sequence"][first_in_session] != np.iinfo(np.uint64).max).sum()
        )
        zero_action_rows += int(
            np.all(next_action == 0.0, axis=1).sum()
            + np.all(used[~first_in_session] == 0.0, axis=1).sum()
        )
        quality = (records["valid_tracking"] == 1) & (records["loop_update_boundary"] == 1)
        epochs = np.rint(records["receiver_timestamp_s"][quality] * 1000.0).astype(np.int64)
        for epoch, prn in zip(epochs, records["prn"][quality], strict=True):
            epoch_prns.setdefault(int(epoch), set()).add(int(prn))
        summaries.append(
            {
                "path": str(path),
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
                "record_count": int(len(records)),
                "channel_values": sorted(map(int, np.unique(records["channel"]))),
                "prn_values": sorted(map(int, np.unique(records["prn"]))),
                "tracking_session_count": int(len(np.unique(records["tracking_session_id"]))),
            }
        )

    max_same_epoch_prns = max((len(prns) for prns in epoch_prns.values()), default=0)
    cadence_fraction = native_pairs / cadence_pairs if cadence_pairs else 0.0
    failures: list[str] = []
    if len(scenario_ids) != 1:
        failures.append("NATIVE_1MS_DUMP_IMPLEMENTATION_INVALID")
    if not causal_pairs or causal_sequence_mismatches or causal_value_mismatches or consume_span_mismatches:
        failures.append("ACTION_MAPPING_UNRESOLVED")
    if (
        cadence_fraction < 0.99
        or current_span_mismatches
        or timestamp_mismatches
        or duration_mismatches
        or finite_failures
        or zero_tap_rows
        or zero_action_rows
        or repeated_tap_rows
        or repeated_action_rows
        or reassignment_failures
    ):
        failures.append("NATIVE_1MS_DUMP_IMPLEMENTATION_INVALID")
    if max_same_epoch_prns < minimum_prns:
        failures.append("INSUFFICIENT_MULTI_PRN_SUPPORT")
    return {
        "schema": "gnss-doppler-lab.trace-native-1ms-validation.v2",
        "status": "PASS" if not failures else "FAIL",
        "failure_labels": sorted(set(failures)),
        "scenario_ids": sorted(scenario_ids),
        "file_summaries": summaries,
        "causal_pair_count": int(causal_pairs),
        "causal_sequence_mismatch_count": int(causal_sequence_mismatches),
        "causal_value_mismatch_count": int(causal_value_mismatches),
        "consume_span_mismatch_count": int(consume_span_mismatches),
        "current_span_mismatch_count": int(current_span_mismatches),
        "timestamp_mismatch_count": int(timestamp_mismatches),
        "duration_mismatch_count": int(duration_mismatches),
        "native_cadence_pair_count": int(native_pairs),
        "cadence_pair_count": int(cadence_pairs),
        "native_cadence_fraction": float(cadence_fraction),
        "finite_failure_count": int(finite_failures),
        "zero_tap_row_count": int(zero_tap_rows),
        "zero_action_row_count": int(zero_action_rows),
        "repeated_tap_row_count": int(repeated_tap_rows),
        "repeated_action_row_count": int(repeated_action_rows),
        "reassignment_failure_count": int(reassignment_failures),
        "maximum_valid_prns_same_rounded_ms_epoch": int(max_same_epoch_prns),
        "minimum_prns_required": int(minimum_prns),
    }


def load_native_trace_pairs(
    dump_dir: Path | str,
    *,
    cn0_min_db_hz: float = 28.0,
    lock_min: float = 0.85,
    prompt_epsilon: float = 1e-12,
) -> TracePairs:
    """Adapt valid row-t next actions and row-(t+1) taps to frozen TRACE-R1."""
    batches: dict[str, list[np.ndarray]] = {name: [] for name in TracePairs.__dataclass_fields__}
    for path in sorted(Path(dump_dir).glob("trace_native_1ms_ch_*.bin")):
        header, records = read_records(path)
        if len(records) < 2:
            continue
        taps, prompt_valid = prompt_normalize(complex_taps(records), prompt_epsilon)
        linked = (
            (records["tracking_session_id"][1:] == records["tracking_session_id"][:-1])
            & (records["prn"][1:] == records["prn"][:-1])
            & (records["loop_sequence"][1:] == records["loop_sequence"][:-1] + 1)
            & (records["action_used_source_loop_sequence"][1:] == records["loop_sequence"][:-1])
        )
        next_used = _action_matrix(records[1:], "action_used")
        previous_next = _action_matrix(records[:-1], "action_next")
        linked &= np.all(next_used == previous_next, axis=1)
        linked &= (
            records["action_used_interval_length_samples"][1:]
            == records["action_next_interval_length_samples"][:-1]
        )
        quality = (
            prompt_valid[:-1]
            & prompt_valid[1:]
            & (records["valid_tracking"][:-1] == 1)
            & (records["valid_tracking"][1:] == 1)
            & np.isfinite(records["cn0_db_hz"][:-1])
            & np.isfinite(records["carrier_lock_test"][:-1])
            & (records["cn0_db_hz"][:-1] >= cn0_min_db_hz)
            & (records["carrier_lock_test"][:-1] >= lock_min)
        )
        source = np.flatnonzero(linked & quality)
        if not len(source):
            continue
        target_rows = source + 1
        dt = (
            records["raw_interval_start_sample"][target_rows].astype(np.float64)
            - records["raw_interval_start_sample"][source].astype(np.float64)
        ) / header.sample_rate_hz
        current = taps[source]
        target = taps[target_rows]
        code_action = np.empty(len(source), dtype=np.float64)
        carrier_action = np.empty(len(source), dtype=np.float64)
        warped = np.empty_like(current)
        support = np.empty(current.shape, dtype=bool)
        for index, (row, duration) in enumerate(zip(source, dt, strict=True)):
            code_action[index], carrier_action[index] = receiver_action(
                records["action_next_code_nco_rate_chips_s"][row],
                records["action_next_carrier_doppler_hz"][row],
                duration,
            )
            # Prompt referencing already removes the global carrier phase.
            # Frozen R1 therefore retains carrier action as a predictor feature
            # but does not rotate the normalized nine-tap vector a second time.
            warped[index], support[index] = warp_complex_taps(current[index], code_action[index], 0.0)
        batches["current"].append(current)
        batches["target"].append(target)
        batches["warped"].append(warped)
        batches["valid_support"].append(support)
        batches["code_action"].append(code_action)
        batches["carrier_action"].append(carrier_action)
        batches["dt_s"].append(dt)
        batches["time_s"].append(records["receiver_timestamp_s"][target_rows].astype(np.float64))
        batches["sample_count"].append(records["raw_interval_start_sample"][target_rows])
        batches["prn"].append(records["prn"][target_rows].astype(np.int16))
        batches["channel"].append(records["channel"][target_rows].astype(np.int16))
        batches["cn0_db_hz"].append(records["cn0_db_hz"][source].astype(np.float64))
        batches["lock"].append(records["carrier_lock_test"][source].astype(np.float64))
        batches["source_row"].append(source.astype(np.int64))
    if not batches["current"]:
        raise ValueError(f"no quality-gated native TRACE pairs in {dump_dir}")
    return TracePairs(**{name: np.concatenate(parts, axis=0) for name, parts in batches.items()})
