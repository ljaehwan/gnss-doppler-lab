"""Utilities for continuous 1 ms tracker cadence audits in Stage-1 R1."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

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
    match = re.search(r"(?:_ch_|_)(\d+)\.mat$", path.name)
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
    source_mat: str
    source_dat_row_match: bool
    source_dat_rows: int | None


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
    return bool(re.fullmatch(r"(?:epl_tracking_ch_|epl_)\d+\.mat", path.name))


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
) -> tuple[list[ContinuousTrackerRow], dict[str, Any]]:
    root = Path(tracker_path)
    if not root.is_dir():
        raise ValueError(f"tracker path is not directory: {root}")

    mats = sorted([p for p in root.glob("*.mat") if _is_tracker_mat(p)])
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
        dat_rows, _dat_bytes, dat_match = _find_dat_signature(mat, len(sample_counts))
        for prn in np.unique(prns):
            if int(prn) == 0:
                continue
            idx = np.flatnonzero(prns == prn).astype(np.int64)
            if idx.size == 0:
                continue
            idx = _sorted_prn_indices(sample_counts, idx)
            deltas = np.diff(sample_counts[idx]) if idx.size >= 2 else np.array([], dtype=np.int64)
            if deltas.size:
                per_pair_delta.append(float(np.mean((deltas >= 24_999) & (deltas <= 25_001))))

            for tracker_row, mat_row in enumerate(idx):
                row = ContinuousTrackerRow(
                    scenario=scenario,
                    channel=channel,
                    prn=int(prns[mat_row]),
                    tracker_row=int(tracker_row),
                    mat_row=int(mat_row),
                    raw_start_sample=int(sample_counts[mat_row]),
                    raw_end_sample=int(sample_counts[mat_row] + SUPPORT_SAMPLES - 1),
                    sample_count=SUPPORT_SAMPLES,
                    code_freq_chips=float(code_freq[mat_row]),
                    carrier_doppler_hz=float(carrier[mat_row]),
                    aux1=float(aux1[mat_row]),
                    prompt_i=float(prompt_i[mat_row]),
                    prompt_q=float(prompt_q[mat_row]),
                    cn0_db_hz=float(cn0[mat_row]),
                    carrier_lock_test=float(lock[mat_row]),
                    source_mat=str(mat),
                    source_dat_row_match=bool(dat_match and dat_rows == len(sample_counts)),
                    source_dat_rows=dat_rows,
                )
                rows.append(row)

    if not rows:
        raise ValueError(f"no PRN rows in {root}")

    ordered = _to_continuous_rows(rows)
    pairs = Counter((r.channel, r.prn) for r in ordered)
    valid_pairs = 0
    max_contiguous_rows = 0
    unique_intervals = True
    raw_contiguous = True
    reason_parts: list[str] = []

    by_pair: dict[tuple[int, int], list[ContinuousTrackerRow]] = {}
    for row in ordered:
        by_pair.setdefault((row.channel, row.prn), []).append(row)

    for (channel, prn), pair_rows in by_pair.items():
        starts = np.array([r.raw_start_sample for r in pair_rows], dtype=np.int64)
        ends = np.array([r.raw_end_sample for r in pair_rows], dtype=np.int64)
        run_len, is_contiguous = _max_consecutive_indices(starts)
        max_contiguous_rows = max(max_contiguous_rows, run_len)
        raw_contiguous = raw_contiguous and is_contiguous
        overlaps = np.diff(starts) < SUPPORT_SAMPLES
        if len(overlaps) and np.any(overlaps):
            unique_intervals = False
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
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return ordered, report


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
        "source_mat",
        "source_dat_row_match",
        "source_dat_rows",
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

    rows, report = build_continuous_tracker_rows(scenario, cfg["scenarios"][scenario]["tracker_path"])
    csv_path = root / f"continuous_tracker_{scenario}.csv"
    manifest_path = root / "continuous_tracker_manifest.json"
    save_continuous_tracker_csv(rows, csv_path)

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
