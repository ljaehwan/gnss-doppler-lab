"""Three-tap E/P/L tracking-window features for current paper baselines.

GNSS-SDR 0.0.19 GPS_L1_CA_DLL_PLL_Tracking computes real Early, Prompt,
and Late correlators.  MAT abs_VE/abs_VL fields are format placeholders and
must never be interpreted as measured taps.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import csv, hashlib
import numpy as np
from .tracking_peaks import TrackingPeakSeries, available_tracking_prns, load_receiver_tracking_peak_series_segments, tap_dataset_layout

EPSILON = 1e-6
REQUIRED_TAPS = ("E", "P", "L")

@dataclass(frozen=True)
class TrackingWindowFeatureRecord:
    run_id: str
    source_fingerprint: str
    label: str
    prn: str
    channel: int
    sample_rate_hz: int
    segment_index: int
    window_index: int
    window_start_s: float
    window_end_s: float
    window_mid_s: float
    epoch_count: int
    near_sym_mean: float
    near_sym_std: float
    sharp_narrow_mean: float
    sharp_narrow_std: float
    sharp_narrow_slope: float
    doppler_std: float
    doppler_slope: float
    cn0_std: float
    code_err_abs_mean: float
    code_err_std: float
    prompt_mag_cv: float
    def to_row(self) -> dict[str, object]: return asdict(self)

def _tap_columns(series: TrackingPeakSeries) -> dict[str,np.ndarray]:
    if tuple(series.tap_names) != REQUIRED_TAPS or series.magnitudes.ndim != 2 or series.magnitudes.shape[1] != 3:
        raise ValueError("Current-paper feature extraction requires exactly real E/P/L taps; VE/VL placeholders are forbidden")
    values=series.magnitudes.astype(np.float64)
    if not np.isfinite(values).all() or any(np.allclose(values[:,i],0.0) for i in range(3)):
        raise ValueError("Current-paper feature extraction requires non-zero finite real E/P/L taps")
    return {name: values[:,i] for i,name in enumerate(REQUIRED_TAPS)}

def _window_starts(t,window_s,stride_s):
    if window_s<=0: raise ValueError("window_s must be positive")
    if stride_s<=0: raise ValueError("stride_s must be positive")
    if not len(t): return []
    step=float(np.median(np.diff(t)[np.diff(t)>0])) if len(t)>1 and np.any(np.diff(t)>0) else 0.0
    end=float(t[-1])+step; start=float(t[0]); out=[]
    while start <= end-window_s+1e-9: out.append(round(start,10)); start+=stride_s
    return out

def _summary(v): return float(np.mean(v)),float(np.std(v,ddof=0))
def _slope(t,v): return 0.0 if len(t)<2 or np.allclose(t,t[0]) else float(np.polyfit(t,v,1)[0])

def compute_tracking_window_feature_records(series: TrackingPeakSeries, *, receiver_run_id: str,
 window_s: float=1.0, stride_s: float=0.5, min_epochs: int=4, label: str="normal",
 source_fingerprint: str="") -> list[TrackingWindowFeatureRecord]:
    if min_epochs<2: raise ValueError("min_epochs must be at least 2")
    taps=_tap_columns(series); e,p,l=taps["E"],taps["P"],taps["L"]
    prompt=np.hypot(series.prompt_i,series.prompt_q)
    near=(e-l)/(e+l+EPSILON); sharp=(2*p-e-l)/(p+EPSILON)
    records=[]
    for wi,start in enumerate(_window_starts(series.time_s,window_s,stride_s)):
        end=start+window_s; mask=(series.time_s>=start-1e-9)&(series.time_s<=end+1e-9); count=int(mask.sum())
        if count<min_epochs: continue
        tw=series.time_s[mask]; nm,ns=_summary(near[mask]); sm,ss=_summary(sharp[mask]); _,ds=_summary(series.carrier_doppler_hz[mask]); _,cs=_summary(series.cn0_db_hz[mask]); _,ces=_summary(series.code_error_chips[mask]); pm=float(np.mean(prompt[mask]))
        records.append(TrackingWindowFeatureRecord(receiver_run_id,source_fingerprint,label,series.prn,series.channel,series.sample_rate_hz,series.segment_index,wi,float(start),float(end),float((start+end)/2),count,nm,ns,sm,ss,_slope(tw,sharp[mask]),ds,_slope(tw,series.carrier_doppler_hz[mask]),cs,float(np.mean(np.abs(series.code_error_chips[mask]))),ces,float(np.std(prompt[mask])/(pm+EPSILON))))
    return records

def _source_fingerprint(run_dir: Path) -> str:
    manifest=__import__("json").loads((run_dir/"manifest.json").read_text())
    raw=run_dir/str(manifest.get("tracking",{}).get("raw_directory","raw"))
    paths=sorted(raw.glob("epl_tracking_ch_*.mat"))
    if not paths: raise ValueError(f"receiver run has no source tracking MAT files: {run_dir.name}")
    hashes=sorted(hashlib.sha256(p.read_bytes()).hexdigest() for p in paths)
    return hashlib.sha256("".join(hashes).encode()).hexdigest()

def collect_receiver_run_tracking_feature_records(receiver_run_dir: str|Path, *, window_s=1.0,stride_s=0.5,min_epochs=4,label="normal",prns=None):
    run=Path(receiver_run_dir); fp=_source_fingerprint(run); selected=prns if prns is not None else available_tracking_prns(run); out=[]
    for prn in selected:
        for segment in load_receiver_tracking_peak_series_segments(run, prn):
            out.extend(compute_tracking_window_feature_records(segment,receiver_run_id=run.name,window_s=window_s,stride_s=stride_s,min_epochs=min_epochs,label=label,source_fingerprint=fp))
    return out

def export_receiver_run_tracking_feature_csv(receiver_run_dir: str|Path, *, output_path: str|Path,window_s=1.0,stride_s=0.5,min_epochs=4,label="normal",prns=None):
    records=collect_receiver_run_tracking_feature_records(receiver_run_dir,window_s=window_s,stride_s=stride_s,min_epochs=min_epochs,label=label,prns=prns); output=Path(output_path); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(TrackingWindowFeatureRecord.__dataclass_fields__)); w.writeheader(); w.writerows(r.to_row() for r in records)
    return output



def _format_float(value: float) -> str:
    return format(float(value), ".12g")


def _tap_feature_layout(tap_count: int) -> tuple[str, ...]:
    return tuple(label for label, _ in tap_dataset_layout(tap_count))


def compute_tracking_tap_window_feature_rows(
    series: TrackingPeakSeries,
    *,
    receiver_run_id: str,
    window_s: float = 1.0,
    stride_s: float = 0.5,
    min_epochs: int = 4,
    label: str = "normal",
    source_fingerprint: str = "",
    tap_count: int | None = None,
) -> list[dict[str, object]]:
    """Compute configurable 3/5/9-tap window features as CSV-ready rows."""
    requested = int(tap_count or len(series.tap_names))
    layout = _tap_feature_layout(requested)
    if tuple(series.tap_names) != layout or series.magnitudes.ndim != 2 or series.magnitudes.shape[1] != requested:
        raise ValueError(f"tracking series does not match requested {requested}-tap layout")
    if min_epochs < 2:
        raise ValueError("min_epochs must be at least 2")
    taps = series.magnitudes.astype(np.float64)
    if not np.isfinite(taps).all() or any(np.allclose(taps[:, i], 0.0) for i in range(requested)):
        raise ValueError(f"{requested}-tap feature extraction requires non-zero finite real taps")
    prompt_index = layout.index("P")
    prompt = np.hypot(series.prompt_i, series.prompt_q)
    rows: list[dict[str, object]] = []
    for wi, start in enumerate(_window_starts(series.time_s, window_s, stride_s)):
        end = start + window_s
        mask = np.logical_and(series.time_s >= start - 1e-9, series.time_s <= end + 1e-9)
        count = int(mask.sum())
        if count < min_epochs:
            continue
        window_taps = taps[mask]
        tw = series.time_s[mask]
        left = window_taps[:, :prompt_index]
        right = window_taps[:, prompt_index + 1:]
        near = window_taps[:, prompt_index]
        left_sum = left.sum(axis=1) if left.size else np.zeros(count)
        right_sum = right.sum(axis=1) if right.size else np.zeros(count)
        argmax = np.argmax(window_taps, axis=1).astype(np.float64)
        tap_offsets = np.arange(requested, dtype=np.float64) - float(prompt_index)
        norm = window_taps.sum(axis=1) + EPSILON
        width = np.sqrt(((window_taps * ((tap_offsets[None, :]) ** 2)).sum(axis=1)) / norm)
        sharpness = (near - (left_sum + right_sum) / max(1, requested - 1)) / (near + EPSILON)
        row: dict[str, object] = {
            "run_id": receiver_run_id,
            "source_fingerprint": source_fingerprint,
            "label": label,
            "prn": series.prn,
            "channel": series.channel,
            "sample_rate_hz": series.sample_rate_hz,
            "segment_index": series.segment_index,
            "window_index": wi,
            "window_start_s": _format_float(start),
            "window_end_s": _format_float(end),
            "window_mid_s": _format_float((start + end) / 2),
            "epoch_count": count,
            "tap_count": requested,
            "tap_layout": ",".join(layout),
            "left_right_imbalance_mean": _format_float(np.mean((left_sum - right_sum) / (left_sum + right_sum + EPSILON))),
            "left_right_imbalance_std": _format_float(np.std((left_sum - right_sum) / (left_sum + right_sum + EPSILON), ddof=0)),
            "peak_index_mean": _format_float(np.mean(argmax)),
            "peak_index_std": _format_float(np.std(argmax, ddof=0)),
            "peak_width_mean": _format_float(np.mean(width)),
            "peak_width_std": _format_float(np.std(width, ddof=0)),
            "peak_sharpness_mean": _format_float(np.mean(sharpness)),
            "peak_sharpness_std": _format_float(np.std(sharpness, ddof=0)),
            "doppler_std": _format_float(np.std(series.carrier_doppler_hz[mask], ddof=0)),
            "doppler_slope": _format_float(_slope(tw, series.carrier_doppler_hz[mask])),
            "cn0_std": _format_float(np.std(series.cn0_db_hz[mask], ddof=0)),
            "code_err_abs_mean": _format_float(np.mean(np.abs(series.code_error_chips[mask]))),
            "code_err_std": _format_float(np.std(series.code_error_chips[mask], ddof=0)),
            "prompt_mag_cv": _format_float(np.std(prompt[mask], ddof=0) / (np.mean(prompt[mask]) + EPSILON)),
        }
        prompt_values = window_taps[:, prompt_index]
        normalized_taps = window_taps / (prompt_values[:, None] + EPSILON)
        sum_normalized_taps = window_taps / (window_taps.sum(axis=1)[:, None] + EPSILON)
        for tap_index, tap_name in enumerate(layout):
            values = window_taps[:, tap_index]
            mean = np.mean(values)
            row[f"tap_{tap_name}_mean"] = _format_float(mean)
            row[f"tap_{tap_name}_std"] = _format_float(np.std(values, ddof=0))
            row[f"tap_{tap_name}_cv"] = _format_float(np.std(values, ddof=0) / (mean + EPSILON))
            row[f"tap_{tap_name}_rel_prompt_mean"] = _format_float(np.mean(normalized_taps[:, tap_index]))
            row[f"tap_{tap_name}_rel_sum_mean"] = _format_float(np.mean(sum_normalized_taps[:, tap_index]))

        # DMCPD/SQM-inspired normalized peak-morphology features.
        # These avoid absolute correlator magnitude and focus on symmetry, width,
        # prompt dominance, secondary shoulder, and centroid shift.
        side_indices = [i for i in range(requested) if i != prompt_index]
        side_taps = window_taps[:, side_indices] if side_indices else np.empty((count, 0))
        if side_taps.size:
            largest_side = np.max(side_taps, axis=1)
            sorted_side = np.sort(side_taps, axis=1)
            second_side = sorted_side[:, -2] if side_taps.shape[1] >= 2 else largest_side
            row["dmcpd_prompt_dominance_mean"] = _format_float(np.mean(prompt_values / (np.mean(window_taps, axis=1) + EPSILON)))
            row["dmcpd_prompt_to_max_side_mean"] = _format_float(np.mean(prompt_values / (largest_side + EPSILON)))
            row["dmcpd_max_side_to_prompt_mean"] = _format_float(np.mean(largest_side / (prompt_values + EPSILON)))
            row["dmcpd_second_side_to_prompt_mean"] = _format_float(np.mean(second_side / (prompt_values + EPSILON)))
        centroid = (window_taps * tap_offsets[None, :]).sum(axis=1) / norm
        row["dmcpd_centroid_shift_mean"] = _format_float(np.mean(centroid))
        row["dmcpd_centroid_shift_std"] = _format_float(np.std(centroid, ddof=0))
        row["dmcpd_width_variance_mean"] = _format_float(np.mean((window_taps * ((tap_offsets[None, :] - centroid[:, None]) ** 2)).sum(axis=1) / norm))
        row["dmcpd_left_right_energy_abs_mean"] = _format_float(np.mean(np.abs(left_sum - right_sum) / (prompt_values + EPSILON)))
        row["dmcpd_curvature_e1l1_mean"] = _format_float(np.mean((prompt_values - 0.5 * (window_taps[:, prompt_index - 1] + window_taps[:, prompt_index + 1])) / (prompt_values + EPSILON))) if requested >= 3 and 0 < prompt_index < requested - 1 else _format_float(0.0)
        max_pair = min(prompt_index, requested - prompt_index - 1)
        for k in range(1, max_pair + 1):
            early = window_taps[:, prompt_index - k]
            late = window_taps[:, prompt_index + k]
            row[f"dmcpd_pair{k}_signed_asym_mean"] = _format_float(np.mean((early - late) / (prompt_values + EPSILON)))
            row[f"dmcpd_pair{k}_abs_asym_mean"] = _format_float(np.mean(np.abs(early - late) / (prompt_values + EPSILON)))
            row[f"dmcpd_pair{k}_ratio_mean"] = _format_float(np.mean(early / (late + EPSILON)))
            row[f"dmcpd_pair{k}_sum_to_prompt_mean"] = _format_float(np.mean((early + late) / (prompt_values + EPSILON)))
        rows.append(row)
    return rows


def collect_receiver_run_tap_feature_rows(receiver_run_dir: str | Path, *, tap_count: int | None = None, window_s=1.0, stride_s=0.5, min_epochs=4, label="normal", prns=None):
    run = Path(receiver_run_dir)
    manifest = __import__("json").loads((run / "manifest.json").read_text())
    requested = int(tap_count or manifest.get("tracking", {}).get("tap_count", 3))
    tap_dataset_layout(requested)
    fp = _source_fingerprint(run)
    selected = prns if prns is not None else available_tracking_prns(run)
    rows: list[dict[str, object]] = []
    for prn in selected:
        for segment in load_receiver_tracking_peak_series_segments(run, prn, tap_count=requested):
            rows.extend(compute_tracking_tap_window_feature_rows(segment, receiver_run_id=run.name, window_s=window_s, stride_s=stride_s, min_epochs=min_epochs, label=label, source_fingerprint=fp, tap_count=requested))
    return rows


def export_receiver_run_tap_feature_csv(receiver_run_dir: str | Path, *, output_path: str | Path, tap_count: int | None = None, window_s=1.0, stride_s=0.5, min_epochs=4, label="normal", prns=None):
    rows = collect_receiver_run_tap_feature_rows(receiver_run_dir, tap_count=tap_count, window_s=window_s, stride_s=stride_s, min_epochs=min_epochs, label=label, prns=prns)
    if not rows:
        raise ValueError("zero configurable tap feature rows generated")
    base_fields = list(rows[0].keys())
    extra_fields = sorted({key for row in rows for key in row} - set(base_fields))
    fields = base_fields + extra_fields
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output
