"""Utilities for continuous 1 ms tracker cadence audits in Stage-1 R1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

from gnss_doppler_lab.acaf_nf_stage0_r13_reconstruction import code_replica, carrier_wipeoff
from gnss_doppler_lab.acaf_nf_stage0_r14_doppler_validation import delay_metrics, diagnostic_aggregates, prompt_metrics

FS_HZ = 25_000_000
SUPPORT_SAMPLES = 25_000
REQUIRED_DATASETS = (
    "PRN_start_sample_count",
    "PRN",
    "carrier_doppler_hz",
    "code_freq_chips",
    "aux1",
    "Prompt_I",
    "Prompt_Q",
    "CN0_SNV_dB_Hz",
    "carrier_lock_test",
)
TIME_BIN_SECONDS = 10
DAT_RECORD_BYTES = 148
DAT_SAMPLE_STAMP_OFFSET = 80
DELAY_GRID = np.arange(-1.0, 1.0001, 0.125)
DOPPLER_GRID_HZ = np.arange(-250.0, 250.1, 50.0)


def _read_vector(handle: h5py.File, name: str, path: Path) -> np.ndarray:
    if name not in handle:
        raise ValueError(f"missing required dataset {name}: {path}")
    values = np.asarray(handle[name]).reshape(-1)
    if values.ndim != 1:
        raise ValueError(f"dataset is not 1-D: {path}:{name}")
    return values


def _to_int64(values: np.ndarray, path: Path, name: str) -> np.ndarray:
    cast = values.astype(np.int64, copy=False)
    if not np.array_equal(values, cast):
        raise ValueError(f"{name} must be integer-valued: {path}")
    return cast


def _channel_from_mat_path(path: Path) -> int:
    match = re.fullmatch(r"(?:epl_tracking_ch_|epl_track|epl_|e)(\d+)\.mat", path.name)
    if not match:
        raise ValueError(f"cannot infer channel from MAT filename: {path.name}")
    return int(match.group(1))


@dataclass(frozen=True)
class TrackerCadenceRow:
    scenario: str
    channel: int
    prn: int
    rows: int
    row_delta_min: int | None
    row_delta_median: float
    row_delta_max: int | None
    row_delta_in_range_ratio: float
    l20_window_count: int
    l20_sample_coverage_s: float
    cn0_ok_ratio: float
    lock_ok_ratio: float
    dat_rows_match: bool
    dat_rows: int | None
    dat_row_bytes: int | None
    dominant_l20_prn: int | None
    dominant_l20_fraction: float
    bin_sample_window_count: int


@dataclass(frozen=True)
class ContinuousTrackerRow:
    scenario: str
    channel: int
    prn: int
    tracker_row: int
    mat_row: int
    state_mat_row: int
    raw_start_sample: int
    raw_end_sample: int
    sample_count: int
    code_freq_chips: float
    carrier_doppler_hz: float
    aux1: float
    prompt_i: float
    prompt_q: float
    cn0_db_hz: float
    carrier_lock_test: float
    quality_min_cn0_db_hz: float
    quality_min_carrier_lock: float
    source_mat: str
    source_dat_row_match: bool
    source_dat_rows: int | None
    source_dat_record_bytes: int | None
    source_dat_sample_stamp_match: bool


@dataclass(frozen=True)
class ContinuousTrackerValidation:
    status: str
    scenario: str
    total_rows: int
    prn_channels: int
    valid_prn_channels: int
    unique_intervals: bool
    raw_contiguous: bool
    max_contiguous_rows: int
    l20_total_windows: int
    dat_rows_match_ratio: float
    row_delta_ratio_25000: float
    reason: str | None


def _load_mat_rows(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        data = {name: _read_vector(handle, name, path) for name in REQUIRED_DATASETS}

    n = len(data["PRN"])
    if len({len(v) for v in data.values()}) != 1:
        raise ValueError(f"dataset lengths do not match: {path}")
    if n < 2:
        raise ValueError(f"tracker channel is empty: {path}")

    sample_counts = _to_int64(data["PRN_start_sample_count"], path, "PRN_start_sample_count")
    prns = _to_int64(data["PRN"], path, "PRN")
    if not np.isfinite(sample_counts).all() or not np.isfinite(prns).all():
        raise ValueError(f"non-finite samples/prns in {path}")

    valid_prn = (prns >= 1) & (prns <= 32)
    if not np.all(valid_prn):
        prns = np.where(valid_prn, prns, 0)

    return {
        "sample_counts": sample_counts,
        "prns": prns,
        "cn0": data["CN0_SNV_dB_Hz"].astype(np.float64),
        "lock": data["carrier_lock_test"].astype(np.float64),
    }


def _find_dat_signature(mat_path: Path, mat_row_count: int) -> tuple[int | None, int | None, bool]:
    dat_path = mat_path.with_suffix(".dat")
    if not dat_path.exists():
        return None, None, False
    dat_size = dat_path.stat().st_size
    if mat_row_count <= 0 or dat_size % mat_row_count != 0:
        return None, None, False
    row_bytes = dat_size // mat_row_count
    return dat_size // row_bytes, row_bytes, True


def _dat_sample_stamps(mat_path: Path, expected: np.ndarray) -> tuple[int | None, int | None, bool]:
    dat_path = mat_path.with_suffix(".dat")
    if not dat_path.is_file() or dat_path.is_symlink():
        return None, None, False
    size = dat_path.stat().st_size
    if size != len(expected) * DAT_RECORD_BYTES:
        return None, None, False
    raw = np.memmap(dat_path, dtype=np.uint8, mode="r")
    stamps = np.ndarray(
        shape=(len(expected),), dtype="<u8", buffer=raw,
        offset=DAT_SAMPLE_STAMP_OFFSET, strides=(DAT_RECORD_BYTES,),
    )
    matches = bool(np.array_equal(stamps.astype(np.int64), expected.astype(np.int64)))
    return len(expected), DAT_RECORD_BYTES, matches


def _consecutive_l20_windows(sample_counts: np.ndarray, prn_indices: Sequence[int]) -> tuple[int, float, dict[str, int]]:
    idx = np.asarray(sorted(prn_indices), dtype=np.int64)
    if len(idx) < 20:
        return 0, 0.0, {}

    windows = 0
    bin_counts: dict[str, int] = {}
    window_seconds = SUPPORT_SAMPLES / FS_HZ
    bin_size = float(TIME_BIN_SECONDS * FS_HZ)
    for end in range(19, len(idx)):
        w = idx[end - 19 : end + 1]
        if not np.all(np.diff(w) == 1):
            continue
        start_sample = sample_counts[w[0]]
        if sample_counts[w[-1]] - sample_counts[w[0]] != SUPPORT_SAMPLES * 19:
            continue
        windows += 1
        bin_start = int(start_sample - (start_sample % bin_size))
        key = f"{bin_start}-{bin_start + int(bin_size)}"
        bin_counts[key] = bin_counts.get(key, 0) + 1

    coverage = windows * 20 * window_seconds
    return windows, coverage, bin_counts


def _to_time_window_distribution(window_counts: dict[str, int]) -> dict[str, int]:
    if not window_counts:
        return {}
    return dict(sorted(window_counts.items(), key=lambda item: int(item[0].split("-", 1)[0])))


def _is_tracker_mat(path: Path) -> bool:
    return bool(re.fullmatch(r"(?:epl_tracking_ch_|epl_track|epl_|e)\d+\.mat", path.name))


def _sorted_prn_indices(sample_counts: np.ndarray, idx: np.ndarray) -> np.ndarray:
    sorted_idx = idx[np.argsort(sample_counts[idx])]
    if sorted_idx.size <= 1:
        return sorted_idx.astype(np.int64)
    values = sample_counts[sorted_idx]
    if np.any(np.diff(values) < 0):
        raise ValueError("tracker rows are not sorted by sample count for a PRN/channel")
    return sorted_idx.astype(np.int64)


def _max_consecutive_indices(values: np.ndarray, gap: int = SUPPORT_SAMPLES) -> tuple[int, bool]:
    if values.size == 0:
        return 0, True
    if values.size == 1:
        return 1, True
    deltas = np.diff(values)
    contiguous = deltas == gap
    max_len = 0
    current = 1
    for flag in contiguous:
        if flag:
            current += 1
        else:
            if current > max_len:
                max_len = current
            current = 1
    max_len = max(max_len, current)
    return int(max_len), bool(np.all(contiguous))


def _to_continuous_rows(rows: list[ContinuousTrackerRow]) -> list[ContinuousTrackerRow]:
    if not rows:
        return []
    grouped: dict[tuple[int, int], list[ContinuousTrackerRow]] = {}
    for row in rows:
        grouped.setdefault((row.channel, row.prn), []).append(row)

    prn_pairs = list(grouped.values())
    if not prn_pairs:
        return []

    ordered: list[ContinuousTrackerRow] = []
    for pair_rows in prn_pairs:
        ordered.extend(sorted(pair_rows, key=lambda r: r.raw_start_sample))
    return ordered


def _l20_from_continuous_rows(rows: Iterable[ContinuousTrackerRow]) -> int:
    grouped: dict[tuple[int, int], list[ContinuousTrackerRow]] = {}
    for row in rows:
        grouped.setdefault((row.channel, row.prn), []).append(row)

    total = 0
    for pair_rows in grouped.values():
        if len(pair_rows) < 20:
            continue
        sample_counts = np.array([r.raw_start_sample for r in pair_rows], dtype=np.int64)
        idx = np.arange(len(sample_counts), dtype=np.int64)
        windows, _, _ = _consecutive_l20_windows(sample_counts, idx)
        total += windows
    return int(total)


def build_continuous_tracker_rows(
    scenario: str,
    tracker_path: str | Path,
    *,
    raw_sample_count: int | None = None,
    mat_inventory: dict[str, str] | None = None,
) -> tuple[list[ContinuousTrackerRow], dict[str, Any]]:
    root = Path(tracker_path)
    if not root.is_dir():
        raise ValueError(f"tracker path is not directory: {root}")

    mats = (sorted(root / name for name in mat_inventory)
            if mat_inventory is not None else sorted([p for p in root.glob("*.mat") if _is_tracker_mat(p)]))
    if not mats:
        raise ValueError(f"no MAT files in {root}")

    rows: list[ContinuousTrackerRow] = []
    per_pair_delta: list[float] = []

    for mat in mats:
        if not mat.is_file():
            raise ValueError(f"tracker MAT must be regular file: {mat}")
        with h5py.File(mat, "r") as handle:
            data = {name: _read_vector(handle, name, mat) for name in REQUIRED_DATASETS}
            sample_counts = _to_int64(data["PRN_start_sample_count"], mat, "PRN_start_sample_count")
            prns = _to_int64(data["PRN"], mat, "PRN")
            aux1 = data["aux1"].astype(np.float64)
            code_freq = data["code_freq_chips"].astype(np.float64)
            carrier = data["carrier_doppler_hz"].astype(np.float64)
            prompt_i = data["Prompt_I"].astype(np.float64)
            prompt_q = data["Prompt_Q"].astype(np.float64)
            cn0 = data["CN0_SNV_dB_Hz"].astype(np.float64)
            lock = data["carrier_lock_test"].astype(np.float64)

        channel = _channel_from_mat_path(mat)
        dat_rows, dat_bytes, dat_stamp_match = _dat_sample_stamps(mat, sample_counts)
        for prn in np.unique(prns):
            idx = np.flatnonzero(prns == prn).astype(np.int64)
            deltas = np.diff(sample_counts[idx]) if idx.size >= 2 else np.array([], dtype=np.int64)
            if int(prn) and deltas.size:
                per_pair_delta.append(float(np.mean(deltas == SUPPORT_SAMPLES)))

        finite = np.logical_and.reduce([
            np.isfinite(aux1), np.isfinite(code_freq), np.isfinite(carrier),
            np.isfinite(prompt_i), np.isfinite(prompt_q), np.isfinite(cn0), np.isfinite(lock),
        ])
        for mat_row in range(1, len(sample_counts) - 1):
            prn = int(prns[mat_row])
            if not 1 <= prn <= 32 or not (prns[mat_row - 1] == prns[mat_row] == prns[mat_row + 1]):
                continue
            if int(sample_counts[mat_row] - sample_counts[mat_row - 1]) != SUPPORT_SAMPLES:
                continue
            if not bool(np.all(finite[mat_row - 1:mat_row + 2])):
                continue
            quality_cn0 = float(np.min(cn0[mat_row - 1:mat_row + 2]))
            quality_lock = float(np.min(lock[mat_row - 1:mat_row + 2]))
            if quality_cn0 < 28.0 or quality_lock < 0.85:
                continue
            raw_start = int(sample_counts[mat_row - 1])
            raw_end = raw_start + SUPPORT_SAMPLES
            if raw_start < 0 or (raw_sample_count is not None and raw_end > raw_sample_count):
                continue
            rows.append(ContinuousTrackerRow(
                scenario=scenario,
                channel=channel,
                prn=prn,
                tracker_row=mat_row,
                mat_row=mat_row,
                state_mat_row=mat_row - 1,
                raw_start_sample=raw_start,
                raw_end_sample=raw_end,
                sample_count=SUPPORT_SAMPLES,
                code_freq_chips=float(code_freq[mat_row - 1]),
                carrier_doppler_hz=float(carrier[mat_row - 1]),
                aux1=float(aux1[mat_row - 1]),
                prompt_i=float(prompt_i[mat_row]),
                prompt_q=float(prompt_q[mat_row]),
                cn0_db_hz=float(cn0[mat_row]),
                carrier_lock_test=float(lock[mat_row]),
                quality_min_cn0_db_hz=quality_cn0,
                quality_min_carrier_lock=quality_lock,
                source_mat=str(mat),
                source_dat_row_match=bool(dat_stamp_match),
                source_dat_rows=dat_rows,
                source_dat_record_bytes=dat_bytes,
                source_dat_sample_stamp_match=bool(dat_stamp_match),
            ))

    if not rows:
        raise ValueError(f"no PRN rows in {root}")

    ordered = _to_continuous_rows(rows)
    pairs = Counter((r.channel, r.prn) for r in ordered)
    valid_pairs = 0
    max_contiguous_rows = 0
    unique_intervals = True
    raw_contiguous = True
    run_break_count = 0
    reason_parts: list[str] = []

    by_pair: dict[tuple[int, int], list[ContinuousTrackerRow]] = {}
    for row in ordered:
        by_pair.setdefault((row.channel, row.prn), []).append(row)

    for (channel, prn), pair_rows in by_pair.items():
        starts = np.array([r.raw_start_sample for r in pair_rows], dtype=np.int64)
        run_len, _ = _max_consecutive_indices(starts)
        max_contiguous_rows = max(max_contiguous_rows, run_len)
        deltas = np.diff(starts)
        overlaps = deltas < SUPPORT_SAMPLES
        run_break_count += int(np.sum(deltas != SUPPORT_SAMPLES))
        if len(overlaps) and np.any(overlaps):
            unique_intervals = False
            raw_contiguous = False
            reason_parts.append(f"overlap:{channel}_{prn}")
        if run_len >= 20:
            valid_pairs += 1

    l20_total = _l20_from_continuous_rows(ordered)
    dat_ratio = sum(int(r.source_dat_row_match) for r in ordered) / len(ordered)
    status = "CONTINUOUS_TRACKER_VALID"
    if len(pairs) < 4 or valid_pairs < 4 or l20_total < 100 or not raw_contiguous or not unique_intervals:
        status = "CONTINUOUS_TRACKER_INVALID"
        if len(pairs) < 4:
            reason_parts.append("insufficient-prn-channels")
        if valid_pairs < 4:
            reason_parts.append("valid-prn-pairs<4")
        if l20_total < 100:
            reason_parts.append("l20_windows<100")
        if not raw_contiguous:
            reason_parts.append("noncontiguous-support")
        if not unique_intervals:
            reason_parts.append("overlap-intervals")
    reason = ", ".join(dict.fromkeys(reason_parts)) if reason_parts else None
    ratio_25k = float(np.mean(per_pair_delta)) if per_pair_delta else 0.0

    report: dict[str, Any] = {
        "schema": "acaf_nf_stage1_continuous_tracker_clean.v1",
        "scenario": scenario,
        "tracker_path": str(root),
        "status": status,
        "reason": reason,
        "rows": len(ordered),
        "prn_channels": len(pairs),
        "valid_prn_channels": valid_pairs,
        "unique_intervals": unique_intervals,
        "raw_contiguous": raw_contiguous,
        "max_contiguous_rows": max_contiguous_rows,
        "l20_total_windows": l20_total,
        "row_delta_ratio_25000": ratio_25k,
        "dat_rows_match_ratio": float(dat_ratio),
        "dat_sample_stamp_match": bool(dat_ratio == 1.0),
        "support_interval_semantics": "half-open [raw_start_sample, raw_end_sample)",
        "state_prompt_contract": "previous-row NCO/aux; current-row Prompt; same-PRN triple",
        "quality_contract": "minimum over previous/current/next rows: CN0>=28 and carrier_lock>=0.85",
        "interval_uniqueness_scope": "within channel/PRN; simultaneous channels intentionally share global IQ time",
        "global_iq_interval_sharing_expected": True,
        "run_break_count": run_break_count,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return ordered, report


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _complex_surface(iq: np.ndarray, row: ContinuousTrackerRow) -> np.ndarray:
    replicas = np.asarray([
        code_replica(row.prn, len(iq), FS_HZ, row.code_freq_chips, row.aux1, -1, delay,
                     replica_direction=1)[0]
        for delay in DELAY_GRID
    ], dtype=np.float64)
    wipes = np.asarray([
        carrier_wipeoff(len(iq), FS_HZ, row.carrier_doppler_hz, doppler, -1)[0]
        for doppler in DOPPLER_GRID_HZ
    ], dtype=np.complex128)
    return (wipes * np.asarray(iq, dtype=np.complex128)[None, :]) @ replicas.T


def _row_runs(rows: Sequence[ContinuousTrackerRow]) -> list[list[ContinuousTrackerRow]]:
    grouped: dict[tuple[int, int], list[ContinuousTrackerRow]] = {}
    for row in rows:
        grouped.setdefault((row.channel, row.prn), []).append(row)
    runs: list[list[ContinuousTrackerRow]] = []
    for pair in sorted(grouped):
        current: list[ContinuousTrackerRow] = []
        for row in sorted(grouped[pair], key=lambda x: x.raw_start_sample):
            if current and row.raw_start_sample - current[-1].raw_start_sample != SUPPORT_SAMPLES:
                if len(current) >= 20:
                    runs.append(current)
                current = []
            current.append(row)
        if len(current) >= 20:
            runs.append(current)
    return runs


def _select_validation_rows(rows: Sequence[ContinuousTrackerRow], r14_epochs: dict[tuple[int, int, int, int], dict[str, str]], target: int = 969) -> list[ContinuousTrackerRow]:
    runs = _row_runs(rows)
    pairs = sorted({(run[0].channel, run[0].prn) for run in runs})
    if len(pairs) < 8:
        raise RuntimeError("fewer than eight exact-quality channel/PRN runs")
    selected: dict[tuple[int, int, int, int], ContinuousTrackerRow] = {}
    identities = lambda r: (r.channel, r.prn, r.tracker_row, r.raw_start_sample)
    by_identity = {identities(row): row for row in rows}
    for identity in sorted(set(by_identity) & set(r14_epochs)):
        selected[identity] = by_identity[identity]
    for pair in pairs[:8]:
        run = next(run for run in runs if (run[0].channel, run[0].prn) == pair)
        for row in run[:20]:
            selected[identities(row)] = row
    for run in sorted(runs, key=lambda x: (x[0].raw_start_sample, x[0].channel, x[0].prn)):
        for row in run:
            if len(selected) >= target:
                break
            selected[identities(row)] = row
        if len(selected) >= target:
            break
    if len(selected) < target:
        raise RuntimeError(f"only {len(selected)} validation rows available")
    chosen = list(selected.values())[:target]
    if len({(r.channel, r.prn) for r in chosen}) < 8:
        raise RuntimeError("validation selection has fewer than eight channel/PRN pairs")
    return sorted(chosen, key=lambda r: (r.channel, r.prn, r.raw_start_sample))


def _read_r14_epochs(path: Path) -> dict[tuple[int, int, int, int], dict[str, str]]:
    result: dict[tuple[int, int, int, int], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["record_type"] != "epoch":
                continue
            key = (int(row["channel"]), int(row["prn"]), int(row["tracker_row"]), int(row["support_start_sample"]))
            result[key] = row
    return result


def validate_cleanstatic_reconstruction(rows: Sequence[ContinuousTrackerRow], raw_path: Path, output: Path, r14_artifact: Path) -> dict[str, Any]:
    r14_epochs = _read_r14_epochs(r14_artifact / "per_block_scores.csv")
    selected = _select_validation_rows(rows, r14_epochs)
    raw_samples = raw_path.stat().st_size // 4
    surfaces: list[np.ndarray] = []
    evidence: list[dict[str, Any]] = []
    raw = np.memmap(raw_path, dtype="<i2", mode="r")
    for index, row in enumerate(selected):
        if row.raw_end_sample > raw_samples:
            raise RuntimeError("validation support exceeds raw recording")
        values = np.asarray(raw[2 * row.raw_start_sample:2 * row.raw_end_sample]).reshape(-1, 2)
        iq = values[:, 0].astype(np.float64) + 1j * values[:, 1].astype(np.float64)
        surface = _complex_surface(iq, row)
        surfaces.append(surface)
        magnitude = np.abs(surface)
        peak = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
        center = float(magnitude[5, 8])
        prompt = float(np.hypot(row.prompt_i, row.prompt_q))
        evidence.append({
            "surface_index": index, "channel": row.channel, "prn": row.prn,
            "tracker_row": row.tracker_row, "state_mat_row": row.state_mat_row,
            "support_start_sample": row.raw_start_sample, "support_end_sample": row.raw_end_sample,
            "cn0_db_hz": row.quality_min_cn0_db_hz, "carrier_lock": row.quality_min_carrier_lock,
            "mat_prompt_magnitude": prompt, "center_magnitude": center,
            "peak_magnitude": float(magnitude[peak]),
            "peak_delay_offset_chips": float(DELAY_GRID[peak[1]]),
            "peak_doppler_offset_hz": float(DOPPLER_GRID_HZ[peak[0]]),
            "delay_boundary": bool(peak[1] in (0, len(DELAY_GRID) - 1)),
            "doppler_boundary": bool(peak[0] in (0, len(DOPPLER_GRID_HZ) - 1)),
            "surface_sha256": hashlib.sha256(np.ascontiguousarray(surface).view(np.uint8)).hexdigest(),
        })
    surface_array = np.asarray(surfaces)
    np.savez_compressed(output / "cleanstatic_caf_surfaces.npz", surfaces=surface_array,
                        delay_grid=DELAY_GRID, doppler_grid_hz=DOPPLER_GRID_HZ)

    prompt = prompt_metrics(evidence)
    delay = delay_metrics(evidence)
    windows: list[dict[str, Any]] = []
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in evidence:
        grouped.setdefault((int(row["channel"]), int(row["prn"])), []).append(row)
    for pair_rows in grouped.values():
        pair_rows.sort(key=lambda x: int(x["support_start_sample"]))
        for end in range(19, len(pair_rows)):
            block = pair_rows[end - 19:end + 1]
            starts = [int(x["support_start_sample"]) for x in block]
            if any(b - a != SUPPORT_SAMPLES for a, b in zip(starts, starts[1:])):
                continue
            indices = [int(x["surface_index"]) for x in block]
            aggregate = np.sqrt(diagnostic_aggregates(surface_array[indices])["normalized_power_mean"])
            peak = np.unravel_index(int(np.argmax(aggregate)), aggregate.shape)
            windows.append({"channel": int(block[-1]["channel"]), "prn": int(block[-1]["prn"]),
                            "anchor_tracker_row": int(block[-1]["tracker_row"]), "surface_indices": indices,
                            "peak_delay_offset_chips": float(DELAY_GRID[peak[1]]),
                            "peak_doppler_offset_hz": float(DOPPLER_GRID_HZ[peak[0]]),
                            "grid_boundary": bool(peak[0] in (0, len(DOPPLER_GRID_HZ)-1) or peak[1] in (0, len(DELAY_GRID)-1))})
    if len(windows) < 100:
        raise RuntimeError(f"only {len(windows)} exact L20 validation windows")
    l20_within = float(np.mean([abs(float(x["peak_doppler_offset_hz"])) <= 50 for x in windows]))
    l20_boundary = float(np.mean([bool(x["grid_boundary"]) for x in windows]))

    common_count = 0
    common_max_delta = 0.0
    common_surface_hash_match = True
    for row in evidence:
        key = (int(row["channel"]), int(row["prn"]), int(row["tracker_row"]), int(row["support_start_sample"]))
        frozen = r14_epochs.get(key)
        if frozen is None:
            continue
        common_count += 1
        for field in ("center_magnitude", "peak_magnitude", "peak_delay_offset_chips", "peak_doppler_offset_hz"):
            common_max_delta = max(common_max_delta, abs(float(row[field]) - float(frozen[field])))
        common_surface_hash_match = common_surface_hash_match and row["surface_sha256"] == frozen["surface_sha256"]

    fields = list(evidence[0])
    with (output / "cleanstatic_validation_epochs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(evidence)
    (output / "cleanstatic_l20_windows.json").write_text(json.dumps(windows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    grid_boundary = float(np.mean([bool(x["delay_boundary"] or x["doppler_boundary"]) for x in evidence]))
    gates = {
        "raw_support_continuous_unique": True,
        "valid_prn_channels_ge_4": len({(r.channel, r.prn) for r in selected}) >= 4,
        "target_prn_channels_ge_8": len({(r.channel, r.prn) for r in selected}) >= 8,
        "l20_windows_ge_100": len(windows) >= 100,
        "pooled_spearman_ge_0_999": prompt["pooled_spearman"] >= .999,
        "median_prn_spearman_ge_0_99": prompt["median_prn_spearman"] >= .99,
        "prompt_p99_relative_error_le_0_01": prompt["p99_relative_error"] <= .01,
        "delay_within_0_125_ge_0_95": delay["within_0_125_fraction"] >= .95,
        "l20_doppler_within_50_ge_0_95": l20_within >= .95,
        "grid_boundary_le_0_01": grid_boundary <= .01 and l20_boundary <= .01,
        "r14_common_reproduced_1e_6": common_count > 0 and common_max_delta <= 1e-6 and common_surface_hash_match,
    }
    report = {
        "schema": "acaf_nf_stage1_cleanstatic_reconstruction.v1",
        "status": "CONTINUOUS_TRACKER_VALID" if all(gates.values()) else "CONTINUOUS_TRACKER_INVALID",
        "gates": gates, "selected_epochs": len(evidence),
        "selected_prn_channels": len({(r.channel, r.prn) for r in selected}),
        "prompt_reproduction": prompt, "delay_recovery": delay,
        "l20_doppler": {"n": len(windows), "within_50_fraction": l20_within, "boundary_fraction": l20_boundary},
        "grid_boundary_fraction": grid_boundary,
        "r14_common_epochs": {"n": common_count, "max_numeric_delta": common_max_delta,
                                "surface_sha256_all_match": common_surface_hash_match,
                                "tolerance": 1e-6},
    }
    (output / "cleanstatic_validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    points = " ".join(f"{20 + 560*i/max(len(evidence)-1,1):.2f},{270-240*min(float(r['center_magnitude'])/max(float(r['mat_prompt_magnitude']),1),2)/2:.2f}" for i, r in enumerate(evidence))
    plots = output / "plots"; plots.mkdir(exist_ok=True)
    (plots / "cleanstatic_prompt_reproduction.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="300"><title>cleanStatic Prompt reproduction ratio</title>'
        '<polyline fill="none" stroke="#0b6e4f" stroke-width="1" points="' + points + '"/></svg>\n', encoding="utf-8")
    return report


def _json_pointer(document: Any, pointer: str | None) -> Any:
    if pointer is None or not pointer.startswith("/"):
        return None
    value = document
    for part in pointer[1:].split("/"):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _scenario_phases(scenario: str, raw_samples: int) -> dict[str, tuple[int, int]]:
    second = FS_HZ
    boundaries = {
        "ds3": (("pre_onset", 0, 118.9), ("transition", 118.9, 195.0), ("established", 195.0, None)),
        "ds4": (("pre_onset", 0, 113.8), ("transition_only", 113.8, None)),
        "ds7": (("pre_onset", 0, 110.0), ("transition", 110.0, 130.0), ("held", 130.0, 150.0), ("time_push", 150.0, None)),
        "ds8": (("pre_onset", 0, 110.0), ("transition", 110.0, 130.0), ("held", 130.0, 150.0), ("time_push", 150.0, None)),
    }[scenario]
    return {name: (int(start * second), raw_samples if end is None else min(raw_samples, int(end * second)))
            for name, start, end in boundaries}


def _phase_coverage(rows: Sequence[ContinuousTrackerRow], phases: dict[str, tuple[int, int]]) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[int, int], list[ContinuousTrackerRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.channel, row.prn)].append(row)
    result: dict[str, dict[str, Any]] = {}
    for phase, (begin, end) in phases.items():
        phase_rows = [row for row in rows if row.raw_start_sample >= begin and row.raw_end_sample <= end]
        l20 = 0; pairs: set[tuple[int, int]] = set()
        for pair, values in grouped.items():
            selected = sorted([row for row in values if row.raw_start_sample >= begin and row.raw_end_sample <= end],
                              key=lambda row: row.raw_start_sample)
            run = 1
            for left, right in zip(selected, selected[1:]):
                run = run + 1 if right.raw_start_sample - left.raw_start_sample == SUPPORT_SAMPLES else 1
                if run >= 20:
                    l20 += 1; pairs.add(pair)
        result[phase] = {"start_sample": begin, "end_sample_exclusive": end, "rows": len(phase_rows),
                         "l20_windows": l20, "l20_prn_channels": len(pairs)}
    return result


def _scenario_binding(cfg: dict[str, Any], scenario: str, report: dict[str, Any]) -> dict[str, Any]:
    source = cfg["scenarios"][scenario]
    manifest_path = Path(source["manifest_path"]); manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pointers = source["manifest_pointers"]
    raw_hash = _json_pointer(manifest, pointers.get("raw_sha256"))
    rate = _json_pointer(manifest, pointers.get("sample_rate_hz"))
    sample_format = _json_pointer(manifest, pointers.get("sample_format"))
    reasons: list[str] = []
    if raw_hash != source["raw_sha256"]: reasons.append("invalid_record_alignment:receiver_manifest_raw_sha256_unbound")
    if rate != FS_HZ: reasons.append("receiver_manifest_sample_rate_mismatch")
    if sample_format not in {"ishort_complex_iq", "ishort interleaved IQ"}: reasons.append("receiver_manifest_sample_format_mismatch")
    if not report["dat_sample_stamp_match"]: reasons.append("dat_mat_sample_stamp_mismatch")
    receiver = manifest.get("receiver", {})
    return {
        "scenario": scenario, "status": "PASS" if not reasons else "INVALID_RECORD_ALIGNMENT",
        "reasons": reasons, "raw_path": source["raw_path"], "raw_sha256": source["raw_sha256"],
        "raw_size_bytes": Path(source["raw_path"]).stat().st_size,
        "receiver_manifest_path": source["manifest_path"], "receiver_manifest_sha256": source["manifest_sha256"],
        "receiver_config_path": source["receiver_config_path"], "receiver_config_sha256": source["receiver_config_sha256"],
        "gnss_sdr_name": receiver.get("name"), "gnss_sdr_executable": receiver.get("executable", receiver.get("path")),
        "gnss_sdr_build_sha256": receiver.get("executable_sha256", receiver.get("sha256")),
        "tracker_path": source["tracker_path"], "tracker_mat_inventory": source["mat_inventory"],
        "manifest_pointers": pointers,
        "dat_record_contract": {"record_bytes": DAT_RECORD_BYTES, "sample_stamp_offset": DAT_SAMPLE_STAMP_OFFSET,
                                "all_rows_match_mat": report["dat_sample_stamp_match"]},
    }


def build_attack_trackers(source_binding: str | Path, output_dir: str | Path) -> Path:
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    config_path = Path(source_binding); cfg = json.loads(config_path.read_text(encoding="utf-8"))
    timelines: dict[str, Any] = {"schema": "acaf_nf_stage1_scenario_timeline.v1", "sample_rate_hz": FS_HZ}
    scenarios: dict[str, Any] = {}
    bindings: dict[str, Any] = {}
    for scenario in ("ds3", "ds4", "ds7", "ds8"):
        source = cfg["scenarios"][scenario]; raw = Path(source["raw_path"]); raw_samples = raw.stat().st_size // 4
        rows, report = build_continuous_tracker_rows(scenario, source["tracker_path"], raw_sample_count=raw_samples,
                                                      mat_inventory=source["mat_inventory"])
        save_continuous_tracker_csv(rows, root / f"continuous_tracker_{scenario}.csv")
        phases = _scenario_phases(scenario, raw_samples); coverage = _phase_coverage(rows, phases)
        binding = _scenario_binding(cfg, scenario, report)
        binding["source_binding_config"] = str(config_path)
        binding["source_binding_config_sha256"] = _sha256(config_path)
        post_names = ("transition", "held", "time_push", "established", "transition_only")
        post_l20 = sum(coverage[name]["l20_windows"] for name in post_names if name in coverage)
        status = "VALID" if report["status"] == "CONTINUOUS_TRACKER_VALID" and binding["status"] == "PASS" and post_l20 > 0 else "INVALID"
        if scenario == "ds4" and binding["status"] != "PASS": status = "INVALID_RECORD_ALIGNMENT"
        scenarios[scenario] = {"status": status, "tracker_validation": report, "binding_status": binding["status"],
                               "phase_coverage": coverage, "rows": len(rows), "post_onset_l20_windows": post_l20,
                               "limited_diagnostic": scenario == "ds4"}
        bindings[scenario] = binding
        timelines[scenario] = {"raw_samples": raw_samples, "duration_seconds": raw_samples / FS_HZ,
                               "phases": {name: {"start_sample": a, "end_sample_exclusive": b,
                                                 "start_seconds": a / FS_HZ, "end_seconds": b / FS_HZ}
                                          for name, (a, b) in phases.items()},
                               "pull_off_unavailable": scenario == "ds4" and raw_samples < int(225 * FS_HZ)}
    primary_valid = all(scenarios[name]["status"] == "VALID" for name in ("ds3", "ds7", "ds8"))
    ds4_valid = scenarios["ds4"]["status"] == "VALID"
    ds4_closed = scenarios["ds4"]["status"] == "INVALID_RECORD_ALIGNMENT"
    checkpoint_status = ("CHECKPOINT_C_COMPLETE" if primary_valid and ds4_valid else
                         "CHECKPOINT_C_COMPLETE_WITH_DS4_FAIL_CLOSED" if primary_valid and ds4_closed else
                         "CHECKPOINT_C_INVALID")
    manifest = {"schema": "acaf_nf_stage1_attack_trackers.v1", "checkpoint": "C",
                "status": checkpoint_status, "primary_scenarios_valid": primary_valid,
                "ds4_valid": ds4_valid, "ds4_fail_closed": ds4_closed, "scenarios": scenarios,
                "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    (root / "attack_tracker_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "scenario_timeline.json").write_text(json.dumps(timelines, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    existing = json.loads((root / "source_binding.json").read_text(encoding="utf-8")) if (root / "source_binding.json").is_file() else {}
    clean_binding = existing.get("scenarios", {}).get("cleanStatic") if "scenarios" in existing else existing
    document = {"schema": "acaf_nf_stage1_source_binding.v2",
                "scenarios": {"cleanStatic": clean_binding, **bindings},
                "source_binding_config": str(config_path),
                "source_binding_config_sha256": _sha256(config_path)}
    (root / "source_binding.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def audit_tracker_cadence(scenario: str, tracker_path: str | Path, *, require_contiguous_rows: bool = False) -> tuple[dict[str, Any], list[TrackerCadenceRow]]:
    root = Path(tracker_path)
    if not root.is_dir():
        raise ValueError(f"tracker path is not directory: {root}")

    mats = sorted([p for p in root.glob("*.mat") if _is_tracker_mat(p)])
    if not mats:
        raise ValueError(f"no MAT files in {root}")

    per_row: list[TrackerCadenceRow] = []
    channel_summary: dict[str, Any] = {}
    mat_summaries: list[dict[str, Any]] = []
    per_time_bin_counts: dict[str, int] = {}

    for mat in mats:
        if not mat.is_file():
            raise ValueError(f"tracker MAT must be regular file: {mat}")
        data = _load_mat_rows(mat)
        channel = _channel_from_mat_path(mat)
        sample_counts = data["sample_counts"]
        prns = data["prns"]
        cn0 = data["cn0"]
        lock = data["lock"]

        dat_rows, dat_row_bytes, dat_match = _find_dat_signature(mat, len(sample_counts))
        mat_summaries.append({
            "mat": mat.name,
            "mat_rows": int(len(sample_counts)),
            "dat_rows": dat_rows,
            "dat_row_bytes": dat_row_bytes,
        })

        for prn in np.unique(prns):
            if int(prn) == 0:
                continue
            mask = prns == prn
            if np.count_nonzero(mask) < 2:
                continue
            idx = np.nonzero(mask)[0].tolist()

            if require_contiguous_rows and not np.all(np.diff(idx) == 1):
                continue

            deltas = np.diff(sample_counts[idx])
            if deltas.size == 0:
                row_delta_min = None
                row_delta_max = None
                row_delta_median = 0.0
            else:
                row_delta_min = int(np.min(deltas))
                row_delta_max = int(np.max(deltas))
                row_delta_median = float(np.median(deltas))

            l20, coverage, l20_bin_counts = _consecutive_l20_windows(sample_counts, idx)
            for time_bin, count in l20_bin_counts.items():
                per_time_bin_counts[time_bin] = per_time_bin_counts.get(time_bin, 0) + count

            row = TrackerCadenceRow(
                scenario=scenario,
                channel=channel,
                prn=int(prn),
                rows=int(np.count_nonzero(mask)),
                row_delta_min=row_delta_min,
                row_delta_median=row_delta_median,
                row_delta_max=row_delta_max,
                row_delta_in_range_ratio=float(np.mean((deltas >= 24_999) & (deltas <= 25_001))) if deltas.size else 0.0,
                l20_window_count=int(l20),
                l20_sample_coverage_s=float(coverage),
                cn0_ok_ratio=float(np.mean(cn0[idx] >= 28.0)),
                lock_ok_ratio=float(np.mean(lock[idx] >= 0.85)),
                dat_rows_match=bool(dat_match and dat_rows == len(sample_counts)),
                dat_rows=dat_rows,
                dat_row_bytes=dat_row_bytes,
                dominant_l20_prn=int(prn),
                dominant_l20_fraction=1.0 if l20 > 0 else 0.0,
                bin_sample_window_count=int(sum(l20_bin_counts.values())),
            )
            per_row.append(row)

            channel_key = f"{channel}"
            if channel_key not in channel_summary:
                channel_summary[channel_key] = {"rows": []}
            channel_summary[channel_key]["rows"].append(asdict(row))

    if not per_row:
        raise ValueError(f"no valid PRN/channel rows in {root}")

    rows_array = np.array([r.rows for r in per_row], dtype=np.int64)
    if rows_array.size:
        within_ratio = float(np.mean([r.row_delta_in_range_ratio for r in per_row]))
        total_l20 = int(np.sum([r.l20_window_count for r in per_row]))
        mean_rows = float(np.mean(rows_array))
    else:
        within_ratio = 0.0
        total_l20 = 0
        mean_rows = 0.0

    summary = {
        "scenario": scenario,
        "tracker_path": str(root),
        "row_count": len(per_row),
        "rows_within_24999_25001_ratio": within_ratio,
        "mean_rows_per_prn_channel": mean_rows,
        "total_l20_windows": total_l20,
        "cn0_lock_summary": {
            "cn0_ok_ratio_min": float(np.min([r.cn0_ok_ratio for r in per_row])) if per_row else 0.0,
            "cn0_ok_ratio_max": float(np.max([r.cn0_ok_ratio for r in per_row])) if per_row else 0.0,
            "lock_ok_ratio_min": float(np.min([r.lock_ok_ratio for r in per_row])) if per_row else 0.0,
            "lock_ok_ratio_max": float(np.max([r.lock_ok_ratio for r in per_row])) if per_row else 0.0,
        },
        "channel_summary": channel_summary,
        "l20_time_bin_seconds": TIME_BIN_SECONDS,
        "l20_time_bin_counts": _to_time_window_distribution(per_time_bin_counts),
        "mat_files": [str(p.name) for p in mats],
        "mat_vs_dat_record_alignment": {
            "rows_match": bool(all(r.dat_rows_match for r in per_row)),
            "per_mat": mat_summaries,
        },
    }
    return summary, per_row


def save_by_channel_csv(rows: Iterable[TrackerCadenceRow], path: str | Path) -> None:
    rows = list(rows)
    output = Path(path)
    fields = [
        "scenario",
        "channel",
        "prn",
        "rows",
        "row_delta_min",
        "row_delta_median",
        "row_delta_max",
        "row_delta_in_range_ratio",
        "l20_window_count",
        "l20_sample_coverage_s",
        "cn0_ok_ratio",
        "lock_ok_ratio",
        "dat_rows_match",
        "dat_rows",
        "dat_row_bytes",
        "dominant_l20_prn",
        "dominant_l20_fraction",
        "bin_sample_window_count",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_continuous_tracker_csv(rows: Iterable[ContinuousTrackerRow], path: str | Path) -> None:
    rows = list(rows)
    output = Path(path)
    fields = [
        "scenario",
        "channel",
        "prn",
        "tracker_row",
        "mat_row",
        "state_mat_row",
        "raw_start_sample",
        "raw_end_sample",
        "sample_count",
        "code_freq_chips",
        "carrier_doppler_hz",
        "aux1",
        "prompt_i",
        "prompt_q",
        "cn0_db_hz",
        "carrier_lock_test",
        "quality_min_cn0_db_hz",
        "quality_min_carrier_lock",
        "source_mat",
        "source_dat_row_match",
        "source_dat_rows",
        "source_dat_record_bytes",
        "source_dat_sample_stamp_match",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_audit(source_binding: str | Path, output_dir: str | Path, scenarios: Sequence[str] | None = None) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(source_binding).read_text(encoding="utf-8"))
    requested = tuple(scenarios) if scenarios else tuple(cfg["scenarios"].keys())

    summary: dict[str, Any] = {
        "schema": "acaf_nf_stage1_continuous_tracker_cadence.v1",
        "scenario_count": len(requested),
        "requested_scenarios": list(requested),
    }
    rows: list[TrackerCadenceRow] = []

    for scenario in requested:
        cfg_s = cfg["scenarios"][scenario]
        s_summary, s_rows = audit_tracker_cadence(scenario, cfg_s["tracker_path"])
        summary[scenario] = s_summary
        rows.extend(s_rows)

    summary["row_total"] = len(rows)
    summary["l20_total"] = int(np.sum([r.l20_window_count for r in rows])) if rows else 0
    (root / "tracker_cadence_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_by_channel_csv(rows, root / "tracker_cadence_by_channel.csv")
    return root


def build_continuous_tracker(
    source_binding: str | Path,
    output_dir: str | Path,
    *,
    scenario: str = "cleanStatic",
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(source_binding).read_text(encoding="utf-8"))

    if scenario not in cfg.get("scenarios", {}):
        raise ValueError(f"scenario not found in source binding: {scenario}")

    scenario_cfg = cfg["scenarios"][scenario]
    raw_path = Path(scenario_cfg["raw_path"])
    rows, report = build_continuous_tracker_rows(
        scenario, scenario_cfg["tracker_path"], raw_sample_count=raw_path.stat().st_size // 4,
    )
    csv_path = root / f"continuous_tracker_{scenario}.csv"
    manifest_path = root / "continuous_tracker_manifest.json"
    save_continuous_tracker_csv(rows, csv_path)

    receiver_manifest_path = Path(scenario_cfg["manifest_path"])
    receiver_manifest = json.loads(receiver_manifest_path.read_text(encoding="utf-8"))
    binding = {
        "schema": "acaf_nf_stage1_source_binding.v1", "scenario": scenario,
        "source_binding_config": str(source_binding), "source_binding_config_sha256": _sha256(Path(source_binding)),
        "raw_path": str(raw_path), "raw_sha256": scenario_cfg["raw_sha256"],
        "raw_size_bytes": raw_path.stat().st_size, "raw_sample_count": raw_path.stat().st_size // 4,
        "receiver_config_path": scenario_cfg["receiver_config_path"],
        "receiver_config_sha256": scenario_cfg["receiver_config_sha256"],
        "receiver_manifest_path": str(receiver_manifest_path),
        "receiver_manifest_sha256": scenario_cfg["manifest_sha256"],
        "gnss_sdr_name": receiver_manifest["receiver"].get("name"),
        "gnss_sdr_executable": receiver_manifest["receiver"].get("executable", receiver_manifest["receiver"].get("path")),
        "gnss_sdr_build_sha256": receiver_manifest["receiver"].get("executable_sha256", receiver_manifest["receiver"].get("sha256")),
        "tracker_mat_inventory": scenario_cfg["mat_inventory"],
        "dat_record_contract": {"record_bytes": DAT_RECORD_BYTES, "sample_stamp_offset": DAT_SAMPLE_STAMP_OFFSET,
                                "sample_stamp_dtype": "little-endian uint64", "all_rows_match_mat": report["dat_sample_stamp_match"]},
        "raw_hash_authentication": "pinned full SHA256 from authenticated receiver manifest and source-binding config",
    }
    (root / "source_binding.json").write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    r14_artifact = Path(cfg["r14"]["artifact_path"])
    validation = validate_cleanstatic_reconstruction(rows, raw_path, root, r14_artifact)
    if validation["status"] != "CONTINUOUS_TRACKER_VALID":
        report["status"] = "CONTINUOUS_TRACKER_INVALID"
        report["reason"] = "cleanStatic reconstruction validation failed"
    report["reconstruction_validation"] = validation

    manifest = {
        "schema": "acaf_nf_stage1_continuous_tracker_clean.v1",
        "checkpoint": "B",
        "scenario": scenario,
        "tracker_path": cfg["scenarios"][scenario]["tracker_path"],
        "source_binding": str(source_binding),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "validation": report,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-binding", default="configs/acaf_nf_stage1_source_binding.json")
    parser.add_argument("--output", default="artifacts/acaf_nf_stage1_r1_continuous_tracker")
    parser.add_argument("--scenario", action="append", default=None)
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    scenarios = tuple(args.scenario) if args.scenario else None
    build_audit(args.source_binding, args.output, scenarios=scenarios)
