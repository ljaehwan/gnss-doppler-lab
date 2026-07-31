"""Causal, geometry-conditioned GCMR pair-relation events.

Tracking measurements are binned independently inside each half-open window;
this module never interpolates, rolls across a boundary, or joins MAT tracking
segments.  Geometry and receiver-quality values are emitted only as model
conditions, not as observation/reconstruction targets.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import combinations
import math
from pathlib import Path
import re
from typing import Iterable, Mapping

import h5py
import numpy as np
from scipy.stats import rankdata

from .gcmr_geometry import (GpsEphemeris, common_clock_removed_residuals,
                            ephemeris_health_selection, satellite_observation)

GPS_CA_CODE_RATE_CHIPS = 1_023_000.0
L1_TO_CA_CARRIER_RATIO = 1540.0
OBSERVATION_FEATURES = ("spearman_rD", "median_abs_diff_rD", "spearman_DLL",
 "median_abs_diff_DLL", "pearson_PLL", "median_abs_diff_PLL", "spearman_CC",
 "median_abs_diff_CC", "pearson_micro", "median_abs_diff_micro")
CONDITION_FEATURES = ("los_dot", "min_elevation_sin", "max_elevation_sin",
 "abs_predicted_doppler_diff", "min_cn0", "max_cn0", "min_lock", "max_lock")
_REQUIRED_DATASETS = ("PRN", "PRN_start_sample_count", "carrier_doppler_hz",
 "carr_error_filt_hz", "code_error_filt_chips", "code_freq_chips",
 "carrier_doppler_rate_hz", "code_freq_rate_chips", "CN0_SNV_dB_Hz",
 "carrier_lock_test", "Prompt_I", "Prompt_Q")

@dataclass(frozen=True)
class TrackingRow:
    prn: int
    time_s: float
    carrier_doppler_hz: float
    carr_error_filt_hz: float
    code_error_filt_chips: float
    code_freq_chips: float
    carrier_doppler_rate_hz: float
    code_freq_rate_chips: float
    CN0_SNV_dB_Hz: float
    carrier_lock_test: float
    Prompt_I: float
    Prompt_Q: float
    channel: int = 0
    segment_index: int = 0

@dataclass(frozen=True)
class GcmrPairRelationEvent:
    window_start_s: float
    window_end_s: float
    pair_prns: np.ndarray
    observations: np.ndarray
    observation_mask: np.ndarray
    conditions: np.ndarray
    def __post_init__(self):
        if not math.isfinite(float(self.window_start_s)) or not math.isfinite(float(self.window_end_s)) or self.window_start_s >= self.window_end_s:
            raise ValueError("event window must be finite and strictly ordered")
        p = len(self.pair_prns)
        if self.pair_prns.shape != (p, 2) or self.observations.shape != (p, 10) or self.observation_mask.shape != (p, 10) or self.conditions.shape != (p, 8):
            raise ValueError("invalid GCMR event array dimensions")
        if not np.issubdtype(self.pair_prns.dtype, np.integer):
            raise ValueError("pair PRNs must be integers")
        pairs=[tuple(map(int,row)) for row in self.pair_prns]
        if any(not (1 <= a < b <= 32) for a,b in pairs):
            raise ValueError("pair PRNs must be canonical unordered GPS PRNs in [1, 32]")
        if len(set(pairs)) != len(pairs):
            raise ValueError("pair PRNs must be unique")
        if self.observation_mask.dtype != np.bool_:
            raise ValueError("observation_mask must be boolean")
        if not np.isfinite(self.observations[self.observation_mask]).all() or not np.isfinite(self.conditions).all():
            raise ValueError("valid observations and conditions must be finite")

GCMRPairRelationEvent = GcmrPairRelationEvent


def _vector(handle, name, path):
    if name not in handle:
        raise ValueError(f"tracking MAT is missing dataset {name}: {path}")
    value=np.asarray(handle[name]).reshape(-1)
    if value.ndim != 1:
        raise ValueError(f"tracking MAT dataset is not a vector: {name}")
    return value


def _channel(path, fallback):
    match=re.search(r"_ch_(\d+)\.mat$",path.name)
    return int(match.group(1)) if match else fallback


def _row_payload(row):
    return tuple(getattr(row,f.name) for f in fields(TrackingRow) if f.name not in ("channel","segment_index"))


def load_gnss_sdr_tracking_rows(raw_directory, *, sample_rate_hz, gap_threshold_s=0.05):
    """Read GNSS-SDR MATLAB-v7.3 channel files into deterministic finite rows.

    PRN values outside GPS 1..32 are dump padding and are ignored.  Every field
    of an observed GPS row is required and finite.  Segments preserve channel
    runs and are additionally split whenever time does not increase or exceeds
    ``gap_threshold_s``.
    """
    raw=Path(raw_directory); fs=float(sample_rate_hz); gap=float(gap_threshold_s)
    if not math.isfinite(fs) or fs <= 0: raise ValueError("sample_rate_hz must be positive and finite")
    if not math.isfinite(gap) or gap <= 0: raise ValueError("gap_threshold_s must be positive and finite")
    paths=sorted(raw.glob("epl_tracking_ch_*.mat"))
    if not paths: raise ValueError(f"raw directory contains no tracking MAT files: {raw}")
    provisional=[]
    for fallback,path in enumerate(paths):
        try:
            with h5py.File(path,"r") as handle:
                data={name:_vector(handle,name,path) for name in _REQUIRED_DATASETS}
        except OSError as exc:
            raise ValueError(f"invalid MATLAB-v7.3 tracking file: {path}") from exc
        lengths={len(v) for v in data.values()}
        if len(lengths)!=1: raise ValueError(f"tracking MAT dataset length mismatch: {path}")
        ch=_channel(path,fallback); prior=None; block=-1
        for i in range(next(iter(lengths))):
            raw_prn=data["PRN"][i]
            if not np.isfinite(raw_prn): raise ValueError(f"nonfinite PRN in {path}")
            prn=int(raw_prn)
            if raw_prn != prn: raise ValueError(f"nonintegral PRN in {path}")
            if not 1 <= prn <= 32:
                prior=None
                continue
            if prior != prn: block+=1
            prior=prn
            values=[float(data[name][i]) for name in _REQUIRED_DATASETS[1:]]
            if not np.isfinite(values).all(): raise ValueError(f"nonfinite tracking row for PRN {prn}: {path}")
            provisional.append((prn, values[0]/fs, values[1:], ch, block))
    # Exact scientific duplicates are ambiguous even when copied to a new channel.
    seen=set()
    for prn,time,values,_,_ in provisional:
        key=(prn,time,*values)
        if key in seen: raise ValueError(f"duplicate exact tracking row for PRN {prn} at {time}")
        seen.add(key)
    # Assign stable per-PRN segment indices independent of file/input ordering.
    grouped={}
    for item in provisional: grouped.setdefault(item[0],[]).append(item)
    result=[]
    for prn in sorted(grouped):
        items=sorted(grouped[prn],key=lambda x:(x[3],x[4],x[1],x[2]))
        segment=-1; prior_key=None; prior_time=None
        for _,time,values,ch,block in items:
            key=(ch,block)
            if key != prior_key or prior_time is None or time <= prior_time or time-prior_time > gap+1e-12: segment+=1
            result.append(TrackingRow(prn,time,*values,channel=ch,segment_index=segment))
            prior_key,prior_time=key,time
    return sorted(result,key=lambda r:(r.prn,r.time_s,r.channel,r.segment_index))

read_gnss_sdr_tracking_rows = load_gnss_sdr_tracking_rows


def code_carrier_consistency_hz(row):
    """Code/carrier consistency with the explicit GPS L1/C/A sign convention."""
    return float(row.code_freq_chips-GPS_CA_CODE_RATE_CHIPS-row.carrier_doppler_hz/L1_TO_CA_CARRIER_RATIO)


def _validate_rows(rows):
    output=[]; seen=set()
    numeric=[f.name for f in fields(TrackingRow) if f.name not in ("prn","channel","segment_index")]
    for row in rows:
        if not isinstance(row,TrackingRow): raise ValueError("rows must contain TrackingRow values")
        if isinstance(row.prn,bool) or not 1 <= int(row.prn) <= 32 or int(row.prn)!=row.prn: raise ValueError("row PRN must be an integer in [1, 32]")
        if not all(math.isfinite(float(getattr(row,name))) for name in numeric): raise ValueError(f"nonfinite tracking row for PRN {row.prn}")
        key=_row_payload(row)
        if key in seen: raise ValueError(f"duplicate exact tracking row for PRN {row.prn} at {row.time_s}")
        seen.add(key); output.append(row)
    return sorted(output,key=lambda r:(r.time_s,r.prn,r.channel,r.segment_index,_row_payload(r)))


def _median_bin(rows,start,end,bin_s):
    width=int(round((end-start)/bin_s)); columns=("carrier_doppler_hz","code_error_filt_chips","carr_error_filt_hz","code_freq_chips","CN0_SNV_dB_Hz","carrier_lock_test")
    bins={}
    for row in rows:
        relative=(row.time_s-start)/bin_s
        index=int(math.floor(relative+1e-10))
        if 0 <= index < width: bins.setdefault(index,[]).append(row)
    result={}
    for index,values in bins.items():
        result[index]=np.asarray([np.median([getattr(r,name) for r in values]) for name in columns] + [np.median([r.time_s for r in values])],dtype=float)
    return result


def _robust_z_differences(values,indices):
    out=np.full(len(values),np.nan)
    if len(values)<2:return out
    adjacent=np.diff(indices)==1; diff=np.diff(values)
    valid=adjacent & np.isfinite(diff)
    if not valid.any():return out
    d=diff[valid]; center=np.median(d); centered=d-center
    scale=1.4826*np.median(np.abs(centered))
    if scale <= 1e-12:
        # Preserve an isolated step when MAD is zero; constants remain zero.
        scale=float(np.sqrt(np.mean(centered**2)))
    z=np.zeros_like(d) if scale <= 1e-12 else centered/scale
    positions=np.nonzero(valid)[0]+1; out[positions]=z
    return out


def _correlation(a,b,kind):
    valid=np.isfinite(a)&np.isfinite(b); x=a[valid]; y=b[valid]
    if len(x)<3 or np.ptp(x)<=1e-12 or np.ptp(y)<=1e-12:return np.nan,False
    if kind=="spearman": x,y=rankdata(x),rankdata(y)
    x=x-x.mean(); y=y-y.mean(); denom=np.linalg.norm(x)*np.linalg.norm(y)
    if denom<=1e-15:return np.nan,False
    value=float(np.dot(x,y)/denom)
    return (value,math.isfinite(value))


def _relation(a,b,kind):
    corr,corr_ok=_correlation(a,b,kind); valid=np.isfinite(a)&np.isfinite(b)
    mad=float(np.median(np.abs(a[valid]-b[valid]))) if valid.any() else np.nan
    return (corr,mad),(corr_ok,bool(valid.any()))


def _window_event(rows,start,end,ephemerides,receiver,tow0,bin_s,min_samples,min_prns):
    by_prn={}
    for row in rows:
        if start <= row.time_s < end: by_prn.setdefault(row.prn,[]).append(row)
    binned={}
    for prn,values in by_prn.items():
        # A window spanning a channel/gap boundary fails closed for this PRN.
        if len({(r.channel,r.segment_index) for r in values}) != 1: continue
        bins=_median_bin(values,start,end,bin_s)
        ordered=sorted(bins)
        # Never summarize two observed runs across a missing grid interval.
        contiguous=all(right-left == 1 for left,right in zip(ordered,ordered[1:]))
        if len(bins)>=min_samples and contiguous and prn in ephemerides:binned[prn]=bins
    if len(binned)<min_prns:return None
    common=set.intersection(*(set(value) for value in binned.values()))
    if len(common)<min_samples:return None
    indices=np.asarray(sorted(common),dtype=int); prns=sorted(binned)
    node={p:{"dop":np.asarray([binned[p][i][0] for i in indices]),
             "dll":np.asarray([binned[p][i][1] for i in indices]),
             "pll":np.asarray([binned[p][i][2] for i in indices]),
             "cc":np.asarray([binned[p][i][3]-GPS_CA_CODE_RATE_CHIPS-binned[p][i][0]/L1_TO_CA_CARRIER_RATIO for i in indices]),
             "cn0":np.asarray([binned[p][i][4] for i in indices]),
             "lock":np.asarray([binned[p][i][5] for i in indices])} for p in prns}
    geometry={p:[] for p in prns}
    for i in indices:
        observed={p:node[p]["dop"][np.searchsorted(indices,i)] for p in prns}; predicted={}
        for p in prns:
            # Each satellite's binned value is propagated at that bin's retained
            # median measurement epoch, not the nominal grid's left edge.
            tow=(tow0+float(binned[p][i][6]))%604800.0
            obs=satellite_observation(receiver,ephemerides[p],tow); geometry[p].append(obs); predicted[p]=obs.predicted_l1_doppler_hz
        residual=common_clock_removed_residuals(observed,predicted,visible_prns=prns)
        pos=np.searchsorted(indices,i)
        for p in prns: node[p].setdefault("rd",np.empty(len(indices)))[pos]=residual[p]
    for p in prns:
        zd=_robust_z_differences(node[p]["dll"],indices); zp=_robust_z_differences(node[p]["pll"],indices)
        micro=np.full(len(indices),np.nan); both=np.isfinite(zd)&np.isfinite(zp); micro[both]=.5*(zd[both]+zp[both]); node[p]["micro"]=micro
    pairs=[]; observations=[]; masks=[]; conditions=[]
    specs=(("rd","spearman"),("dll","spearman"),("pll","pearson"),("cc","spearman"),("micro","pearson"))
    for a,b in combinations(prns,2):
        feature=[]; valid=[]
        for name,kind in specs:
            values,ok=_relation(node[a][name],node[b][name],kind); feature.extend(values); valid.extend(ok)
        ga,gb=geometry[a],geometry[b]
        los_dot=np.median([np.dot(x.los_ecef,y.los_ecef) for x,y in zip(ga,gb)])
        elev_a=np.median([math.sin(math.radians(x.elevation_deg)) for x in ga]); elev_b=np.median([math.sin(math.radians(x.elevation_deg)) for x in gb])
        pdiff=np.median([abs(x.predicted_l1_doppler_hz-y.predicted_l1_doppler_hz) for x,y in zip(ga,gb)])
        cn=(float(np.median(node[a]["cn0"])),float(np.median(node[b]["cn0"]))); lock=(float(np.median(node[a]["lock"])),float(np.median(node[b]["lock"])))
        pairs.append((a,b)); observations.append(feature); masks.append(valid); conditions.append((los_dot,min(elev_a,elev_b),max(elev_a,elev_b),pdiff,min(cn),max(cn),min(lock),max(lock)))
    return GcmrPairRelationEvent(float(start),float(end),np.asarray(pairs,dtype=np.int64),np.asarray(observations,dtype=float),np.asarray(masks,dtype=bool),np.asarray(conditions,dtype=float))


def build_gcmr_pair_relation_events(rows: Iterable[TrackingRow], *, ephemerides: Mapping[int,GpsEphemeris], receiver_ecef, gps_tow_at_time_zero_s, window_s=1.0, stride_s=0.5, resample_bin_s=0.02, min_common_samples=4, min_prns=4):
    """Build canonical unordered pair events from strictly causal windows."""
    clean=_validate_rows(list(rows)); window=float(window_s); stride=float(stride_s); bin_s=float(resample_bin_s); tow0=float(gps_tow_at_time_zero_s)
    if not clean:raise ValueError("fewer than min healthy tracked PRNs: 0")
    if not all(math.isfinite(x) and x>0 for x in (window,stride,bin_s)):raise ValueError("window, stride, and resample bin must be positive and finite")
    if abs(window/bin_s-round(window/bin_s))>1e-9:raise ValueError("window_s must be an integer number of resample bins")
    if int(min_common_samples)!=min_common_samples or min_common_samples<1:raise ValueError("min_common_samples must be a positive integer")
    if int(min_prns)!=min_prns or min_prns<4:raise ValueError("min_prns must be an integer of at least 4")
    if not math.isfinite(tow0):raise ValueError("GPS TOW origin must be finite")
    receiver=tuple(float(x) for x in receiver_ecef)
    if len(receiver)!=3 or not np.isfinite(receiver).all():raise ValueError("receiver_ecef must contain three finite values")
    healthy_ephemerides,_=ephemeris_health_selection(ephemerides,tracked_prns={row.prn for row in clean},min_prns=int(min_prns))
    # Filter before windowing, common-clock estimation, and pair construction.
    clean=[row for row in clean if row.prn in healthy_ephemerides]
    start=math.floor(clean[0].time_s/stride+1e-10)*stride; coverage_end=clean[-1].time_s+bin_s; output=[]
    while start+window <= coverage_end+1e-9:
        event=_window_event(clean,start,start+window,healthy_ephemerides,receiver,tow0,bin_s,int(min_common_samples),int(min_prns))
        if event is not None:output.append(event)
        start=round(start+stride,12)
    return output

build_gcmr_relation_events = build_gcmr_pair_relation_events
