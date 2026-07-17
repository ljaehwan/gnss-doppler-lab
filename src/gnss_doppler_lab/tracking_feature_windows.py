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
from .tracking_peaks import TrackingPeakSeries, available_tracking_prns, load_receiver_tracking_peak_series_segments

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
