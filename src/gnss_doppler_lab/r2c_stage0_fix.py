"""Pre-campaign R2C Stage-0 scientific primitives.

This module has no scenario labels and performs no campaign I/O.  It contains
the frozen native-B0 adapter, wide-template provider, proper-complex whitening,
joint profile likelihoods, controls/statistics, and the pure two-layer decision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence
import hashlib
import json
import math
import time
from datetime import date, datetime, timezone

import numpy as np

C_M_S = 299_792_458.0
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
B0_FEATURES = (
    "tap_E4_rel_prompt_mean", "tap_E3_rel_prompt_mean", "tap_E2_rel_prompt_mean",
    "tap_E_rel_prompt_mean", "tap_P_rel_prompt_mean", "tap_L_rel_prompt_mean",
    "tap_L2_rel_prompt_mean", "tap_L3_rel_prompt_mean", "tap_L4_rel_prompt_mean",
)
B0_THRESHOLDS = {"q50": .0914354398846626, "q70": .12956311106681812,
                 "q80": .1630456149578094}
B0_EVENT_THRESHOLD = 4.169877716047041


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_b0_node_windows(rows, *, run_id: str, window_s: float = 1., stride_s: float = .5,
                          min_epochs: int = 4):
    """Produce historical inclusive-endpoint, Prompt-normalize-then-mean nodes.

    ``rows`` is a pandas frame with time_s, prn and complex or real tap_E4..tap_L4.
    A source epoch on either endpoint participates, matching the native producer.
    """
    import pandas as pd
    required = {"time_s", "prn", "tap_E4", "tap_E3", "tap_E2", "tap_E", "tap_P",
                "tap_L", "tap_L2", "tap_L3", "tap_L4"}
    if missing := sorted(required - set(rows.columns)):
        raise ValueError(f"B0 source rows missing columns: {missing}")
    frame = rows.copy()
    for column,default in (("segment_index",0),("channel",0)):
        if column not in frame: frame[column]=default
    if frame.duplicated(["time_s", "prn", "channel", "segment_index"]).any():
        raise ValueError("duplicate B0 source epoch/PRN")
    frame = frame.sort_values(["segment_index","channel","prn", "time_s"], kind="mergesort")
    if not np.isfinite(frame["time_s"].to_numpy(float)).all():
        raise ValueError("nonfinite B0 time")
    taps = ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")
    output = []
    first = math.ceil(float(frame.time_s.min()) / stride_s) * stride_s
    last = math.floor((float(frame.time_s.max()) - window_s) / stride_s) * stride_s
    for start in np.arange(first, last + stride_s / 2, stride_s):
        selected = frame[(frame.time_s >= start - 1e-9) & (frame.time_s <= start + window_s + 1e-9)]
        for (segment,channel,prn), group in selected.groupby(["segment_index","channel","prn"], sort=True):
            prompt = group["tap_P"].to_numpy()
            if len(group) < min_epochs or np.any(~np.isfinite(np.abs(prompt))) or np.any(np.abs(prompt) == 0):
                continue
            row = {"run_id": run_id, "prn": int(prn), "channel":int(channel),"segment_index":int(segment),
                   "window_bin_s": float(start+stride_s),
                   "window_start_s": float(start), "window_end_s": float(start + window_s),
                   "window_mid_s": float(start + window_s / 2), "epoch_count": int(len(group))}
            # Native features normalize every epoch by its own Prompt, then mean.
            for tap in taps:
                ratio = np.abs(group[f"tap_{tap}"].to_numpy() / prompt)
                row[f"tap_{tap}_rel_prompt_mean"] = float(np.mean(ratio))
            output.append(row)
    return pd.DataFrame(output, columns=["run_id", "prn", "channel","segment_index","window_bin_s", "window_start_s",
                                               "window_end_s", "window_mid_s", "epoch_count", *B0_FEATURES])


def validate_b0_nodes(frame, *, seq_len: int = 12) -> dict:
    required = ["run_id", "prn", "channel","segment_index","window_bin_s", "window_start_s", "window_end_s",
                "window_mid_s", "epoch_count", *B0_FEATURES]
    if missing := [c for c in required if c not in frame.columns]:
        raise ValueError(f"B0 node schema missing: {missing}")
    if [c for c in frame.columns if c in B0_FEATURES] != list(B0_FEATURES):
        raise ValueError("B0 node feature order mismatch")
    numeric = frame[[c for c in required if c not in ("run_id","prn")]].apply(lambda x: np.asarray(x, float))
    if frame.empty or not np.isfinite(numeric.to_numpy()).all() or np.any(frame.epoch_count < 4):
        raise ValueError("B0 nodes must be nonempty, finite, and supported")
    timing=(np.isclose(frame.window_end_s,frame.window_start_s+1.,atol=1e-9)&
            np.isclose(frame.window_mid_s,frame.window_start_s+.5,atol=1e-9)&
            np.isclose(frame.window_bin_s,np.round(frame.window_mid_s*2.)/2.,atol=1e-9))
    if not bool(np.all(timing)):raise ValueError("B0 node timing/window/stride mismatch")
    if frame.duplicated(["run_id", "prn","channel","segment_index", "window_bin_s"]).any():
        raise ValueError("duplicate B0 node")
    for _, group in frame.groupby(["run_id", "prn","channel","segment_index"], sort=False):
        bins = np.sort(group.window_bin_s.to_numpy(float))
        if len(bins) > 1 and not np.allclose(np.diff(bins), .5, atol=1e-9):
            raise ValueError("gap or ordering error in B0 PRN-local sequence")
    return {"feature_columns": list(B0_FEATURES), "seq_len": seq_len,
            "node_rows": int(len(frame)), "score_rows": int(sum(max(0, len(g)-seq_len)
            for _, g in frame.groupby(["run_id", "prn","channel","segment_index"]))) ,
            "availability": "target_window_end"}


def score_b0_nodes(frame, checkpoint: Mapping, *, device="cpu", checkpoint_path: Path | None=None,
                   expected_checkpoint_sha256: str | None=None):
    """Authentic checkpoint/scaler inference over contiguous native node sequences."""
    import pandas as pd
    import torch
    validate_b0_nodes(frame); identity=validate_b0_checkpoint(checkpoint,checkpoint_path)
    if expected_checkpoint_sha256 is not None and identity["checkpoint_sha256"]!=expected_checkpoint_sha256:
        raise ValueError("canonical B0 checkpoint hash mismatch")
    # Architecture is the tracked native producer's exact serialized config.
    import importlib.util
    producer=Path(__file__).resolve().parents[2]/"scripts/train_prn_node_gru.py"
    spec=importlib.util.spec_from_file_location("r2c_b0_producer",producer); module=importlib.util.module_from_spec(spec)
    import sys; sys.modules[spec.name]=module; spec.loader.exec_module(module)
    cfg=module.TrainConfig(**checkpoint["config"]); model=module.PrnLocalGRU(9,cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"],strict=True); model.eval()
    mean=np.asarray(checkpoint["standardizer"]["node_mean"],np.float32)
    std=np.asarray(checkpoint["standardizer"]["node_std"],np.float32); rows=[]
    keys=["run_id","prn","channel","segment_index"]
    with torch.no_grad():
        for identity,g in frame.groupby(keys,sort=True):
            g=g.sort_values("window_bin_s",kind="mergesort").reset_index(drop=True)
            x=module.standardize(g[list(B0_FEATURES)].to_numpy(np.float32),mean,std)
            for target in range(12,len(g)):
                history=g.iloc[target-12:target]
                expected=np.arange(float(g.iloc[target].window_bin_s)-6.,float(g.iloc[target].window_bin_s),.5)
                if not np.allclose(history.window_bin_s.to_numpy(float),expected,atol=1e-9): continue
                pred=model(torch.tensor(x[target-12:target][None],device=device)).cpu().numpy()[0]
                score=float(np.sqrt(np.mean((pred-x[target])**2))); src=g.iloc[target]
                rows.append({**dict(zip(keys,identity)),"target_window_index":target,
                  "window_bin_s":float(src.window_bin_s),"window_start_s":float(src.window_start_s),
                  "window_end_s":float(src.window_end_s),"window_mid_s":float(src.window_mid_s),
                  "availability_time_s":float(src.window_end_s),"prn_node_rmse":score})
    return pd.DataFrame(rows)


def validate_b0_checkpoint(checkpoint: Mapping, checkpoint_path: Path | None = None) -> dict:
    config, features, scaler = checkpoint.get("config", {}), checkpoint.get("node_feature_columns"), checkpoint.get("standardizer", {})
    if tuple(features or ()) != B0_FEATURES or int(config.get("seq_len", -1)) != 12:
        raise ValueError("checkpoint feature order or seq_len identity mismatch")
    mean, std = np.asarray(scaler.get("node_mean")), np.asarray(scaler.get("node_std"))
    if mean.shape != (9,) or std.shape != (9,) or not np.isfinite(mean).all() or np.any(std <= 0):
        raise ValueError("checkpoint scaler identity invalid")
    return {"checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path else None,
            "features": list(features), "seq_len": 12, "scaler_valid": True}


def _btail(k: int, n: int, p: float) -> float:
    if k <= 0: return 0.
    return -math.log(max(sum(math.comb(n, i)*p**i*(1-p)**(n-i) for i in range(k, n+1)), 1e-300))


def replay_b0_events(scores):
    """Replay canonical binomial-tail max and recording-local EWMA."""
    import pandas as pd
    required = {"run_id", "prn", "window_bin_s", "window_start_s", "window_mid_s", "prn_node_rmse"}
    if missing := sorted(required - set(scores.columns)): raise ValueError(f"native score columns missing: {missing}")
    if scores.duplicated(["run_id", "window_bin_s", "prn"]).any(): raise ValueError("duplicate native B0 score")
    start=scores.window_start_s.to_numpy(float);mid=scores.window_mid_s.to_numpy(float)
    end=scores.window_end_s.to_numpy(float) if "window_end_s" in scores else start+1.
    valid=np.isfinite(start)&np.isfinite(mid)&np.isfinite(end)&np.isclose(end,start+1.,atol=1e-9)&np.isclose(mid,start+.5,atol=1e-9)&np.isclose(scores.window_bin_s,np.round(mid*2)/2,atol=1e-9)
    if not np.all(valid):raise ValueError("invalid per-PRN target timing contract")
    if "availability_time_s" in scores and not np.allclose(scores.availability_time_s,end,rtol=0,atol=1e-9):raise ValueError("B0 score availability must equal own target window end")
    rows = []
    for (run, bin_s), group in scores.groupby(["run_id", "window_bin_s"], sort=True):
        vals = group.prn_node_rmse.to_numpy(float); n = len(vals)
        surprises = [_btail(int(np.sum(vals > B0_THRESHOLDS[name])), n, 1-q)
                     for name, q in (("q50", .5), ("q70", .7), ("q80", .8))]
        rows.append({"run_id": run, "window_bin_s": float(bin_s),
                     "window_start_s": float(group.window_start_s.min()),
                     "window_end_s":float((group.window_end_s if "window_end_s" in group else group.window_start_s+1.).max()),
                     "availability_time_s":float((group.window_end_s if "window_end_s" in group else group.window_start_s+1.).max()),
                     "tracked_prn_count": n, "btail_max_507080": max(surprises)})
    out = pd.DataFrame(rows).sort_values(["run_id", "window_bin_s"]).reset_index(drop=True)
    out["btail_max_507080_ewma075"] = 0.
    for _, idx in out.groupby("run_id", sort=False).groups.items():
        state = 0.
        for i in idx:
            state = .75*state + .25*float(out.at[i, "btail_max_507080"])
            out.at[i, "btail_max_507080_ewma075"] = state
    out["alarm"] = out.btail_max_507080_ewma075 > B0_EVENT_THRESHOLD
    return out


def historical_b0_status(paths: Mapping[str, Path], expected_hashes: Mapping[str, str] | None = None) -> dict:
    """Validate historical score availability without interpreting attack labels."""
    import pandas as pd
    reports = {}
    for scenario, path in paths.items():
        digest=None
        try:
            digest = _sha256(path)
            if expected_hashes and scenario in expected_hashes and digest != expected_hashes[scenario]:
                raise ValueError("source score hash mismatch")
            source = pd.read_csv(path)
            replay = replay_b0_events(source)
            reports[scenario] = {"status": "AVAILABLE_SAVED_NATIVE_SCORE_REPLAY_WITH_NODE_LINEAGE_GAP", "sha256": digest,
                                 "node_scores": len(source), "event_scores": len(replay),
                                 "availability": "target_window_end"}
        except (OSError, ValueError) as exc:
            reports[scenario] = {"status": "UNAVAILABLE_AUTHENTIC_INTERFACE", "reason": str(exc),
                                 "sha256":digest,"saved_score_hash_valid":bool(digest and (not expected_hashes or expected_hashes.get(scenario)==digest))}
    status = ("RECONSTRUCTABLE_WITH_LINEAGE_GAPS" if any(Path(p).exists() for p in paths.values())
              else "UNAVAILABLE_AUTHENTIC_INTERFACE")
    return {"status": status, "paper_comparison_eligible": False, "scenarios": reports}


def gps_week_tow(moment: datetime, leap_seconds: int) -> tuple[int, float]:
    """Convert authenticated UTC to GPS week/TOW with explicit leap offset."""
    if moment.tzinfo is None: raise ValueError("UTC datetime must be timezone-aware")
    seconds=(moment.astimezone(timezone.utc)-GPS_EPOCH).total_seconds()+int(leap_seconds)
    if seconds < 0: raise ValueError("date precedes GPS epoch")
    return int(seconds//604800), float(seconds%604800)


def resolve_nmea_rollover(times_hms: Sequence[float], utc_date: date) -> list[datetime]:
    """Bind HHMMSS NMEA values to dates, advancing on a midnight rollover."""
    from datetime import timedelta
    result=[]; day=utc_date; previous=None
    for raw in times_hms:
        h=int(raw//10000); m=int((raw-h*10000)//100); s=float(raw-h*10000-m*100)
        if h>23 or m>59 or s>=60: raise ValueError("invalid NMEA UTC")
        seconds=h*3600+m*60+s
        if previous is not None and seconds < previous-43200: day += timedelta(days=1)
        elif previous is not None and seconds < previous: raise ValueError("non-rollover NMEA time reversal")
        whole=int(s); micro=round((s-whole)*1e6)
        result.append(datetime(day.year,day.month,day.day,h,m,whole,micro,tzinfo=timezone.utc)); previous=seconds
    return result


def reconstruct_time_geometry(*, scenario: str, relative_times_s: Sequence[float],
                              authenticated_utc_start: datetime | None, leap_seconds: int,
                              observable_rx_time: Sequence[float] | None,
                              ephemeris_toe: Mapping[int,float], pvt_times: Sequence[float],
                              los_by_event: Mapping[float,Mapping[int,Sequence[float]]],
                              max_toe_age_s=7200., max_pvt_age_s=1., condition_limit=1e6) -> dict:
    """Independently report time, offline LOS, causal ephemeris, PVT and rank coverage."""
    times=np.asarray(relative_times_s,float)
    if authenticated_utc_start is None:
        absolute=False; week=None; start_tow=None
    else:
        week,start_tow=gps_week_tow(authenticated_utc_start,leap_seconds); absolute=True
    rx=np.asarray(observable_rx_time if observable_rx_time is not None else [],float)
    rx_bound=bool(len(rx) and np.isfinite(rx).all() and np.all(np.diff(rx)>=0))
    pvt=np.sort(np.asarray(pvt_times,float)); valid=0; causal_eph=0; pvt_ok=0; rank_ok=0
    conditions=[]; eligible=len(times)
    for t in times:
        event_tow=None if start_tow is None else start_tow+float(t)
        prns=los_by_event.get(float(t),{})
        matrix=[]
        for prn, vector in prns.items():
            u=np.asarray(vector,float)
            if u.shape==(3,) and np.isfinite(u).all(): matrix.append(u/np.linalg.norm(u))
            toe=ephemeris_toe.get(int(prn))
            if event_tow is not None and toe is not None and 0 <= event_tow-toe <= max_toe_age_s: causal_eph+=1
        past=pvt[pvt<=t]
        causal_pvt=bool(len(past) and t-past[-1]<=max_pvt_age_s)
        pvt_ok += causal_pvt
        if len(matrix)>=5:
            design=np.column_stack([-np.asarray(matrix),np.ones(len(matrix))]); rank=np.linalg.matrix_rank(design)
            cond=float(np.linalg.cond(design)); dof=len(matrix)-rank; conditions.append(cond)
            good=rank==4 and dof>=1 and cond<=condition_limit
            rank_ok += good
            valid += bool(good and causal_pvt)
    return {"scenario":scenario,
      "authenticated_absolute_time_binding":{"status":"PASS" if absolute and rx_bound else "UNAVAILABLE",
        "gps_week":week,"start_tow":start_tow,"observable_rx_time_bound":rx_bound},
      "offline_los_reproducibility":{"status":"PASS" if los_by_event else "UNAVAILABLE","events":len(los_by_event)},
      "event_time_causal_ephemeris":{"status":"PASS" if causal_eph else "UNAVAILABLE","valid_prn_events":causal_eph},
      "pvt_coverage":{"status":"PASS" if pvt_ok else "UNAVAILABLE","valid_events":pvt_ok,"eligible_events":eligible},
      "prn_rank_dof_condition_coverage":{"status":"PASS" if valid else "UNAVAILABLE","valid_events":valid,
        "eligible_events":eligible,"coverage":valid/max(eligible,1),"rank_valid_events":rank_ok,
        "maximum_condition":max(conditions) if conditions else None}}


@dataclass(frozen=True)
class TemplateProvider:
    offsets_chips: np.ndarray
    values: np.ndarray | None
    provenance: Mapping[str, object]
    analytic_approximation: bool
    paper_comparison_ready: bool

    @classmethod
    def analytic(cls):
        return cls(np.array([-1., 1.]), None, {"kind": "analytic_gps_ca_acf"}, True, False)

    @classmethod
    def empirical(cls, offsets, values, provenance):
        x, y = np.asarray(offsets, float), np.asarray(values, complex)
        if x.ndim != 1 or y.shape != x.shape or x[0] > -1 or x[-1] < 1 or np.max(np.diff(x)) > .125000001:
            raise ValueError("empirical template requires authenticated complex >= +/-1 chip support at <=.125 spacing")
        if not np.iscomplexobj(values) or not provenance.get("source_sha256"):
            raise ValueError("empirical template provenance is unauthenticated")
        return cls(x, y, dict(provenance), False, True)

    def evaluate(self, offsets):
        x = np.asarray(offsets, float)
        if self.analytic_approximation:
            return np.maximum(1-np.abs(x), 0).astype(complex)
        if x.min() < self.offsets_chips[0] or x.max() > self.offsets_chips[-1]:
            raise ValueError("empirical template out-of-support evaluation forbidden")
        return np.interp(x, self.offsets_chips, self.values.real) + 1j*np.interp(x, self.offsets_chips, self.values.imag)


@dataclass
class ComplexWhitener:
    shrinkage: float = .2
    eigen_floor_fraction: float = 1e-6
    covariance: np.ndarray | None = None
    pseudo_covariance: np.ndarray | None = None
    inverse_sqrt: np.ndarray | None = None
    diagnostics: dict | None = None
    mean: np.ndarray | None = None

    def fit(self, residuals, roles, row_ids: Sequence[str] | None = None):
        z = np.asarray(residuals, complex)
        if z.ndim != 2 or z.shape[1] != 9 or len(roles) != len(z) or set(roles) != {"normal_train"}:
            raise ValueError("whitening fit requires only cleanStatic normal_train nine-tap residuals")
        self.mean = z.mean(0)
        centered = z-self.mean; denom=max(len(z)-1, 1)
        cov = centered.T@centered.conj()/denom
        pseudo = centered.T@centered/denom
        target = np.trace(cov).real/len(cov)*np.eye(len(cov))
        cov = (1-self.shrinkage)*cov+self.shrinkage*target
        eig, vec = np.linalg.eigh((cov+cov.conj().T)/2)
        floor=max(float(eig.max())*self.eigen_floor_fraction, np.finfo(float).tiny)
        clipped=np.maximum(eig, floor)
        self.covariance=cov; self.pseudo_covariance=pseudo
        self.inverse_sqrt=(vec*(1/np.sqrt(clipped)))@vec.conj().T
        ids = list(row_ids or map(str, range(len(z))))
        self.diagnostics={"minimum_eigenvalue":float(eig.min()), "maximum_eigenvalue":float(eig.max()),
                          "eigenvalue_floor":floor, "condition_number":float(clipped.max()/clipped.min()),
                          "pseudo_covariance_frobenius":float(np.linalg.norm(pseudo)),
                          "train_rows_sha256":hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                          "fit_role":"cleanStatic normal_train"}
        return self

    def transform(self, residuals):
        if self.inverse_sqrt is None: raise RuntimeError("whitener is not fitted")
        return np.asarray(residuals, complex)@self.inverse_sqrt.T

    def serialize(self):
        if self.covariance is None: raise RuntimeError("whitener is not fitted")
        encode=lambda a: {"real":a.real.tolist(), "imag":a.imag.tolist()}
        return {"shrinkage":self.shrinkage, "eigen_floor_fraction":self.eigen_floor_fraction,
                "mean":encode(self.mean),
                "covariance":encode(self.covariance), "pseudo_covariance":encode(self.pseudo_covariance),
                "inverse_sqrt":encode(self.inverse_sqrt), "diagnostics":self.diagnostics}

    @classmethod
    def deserialize(cls, value):
        decode=lambda x:np.asarray(x["real"],float)+1j*np.asarray(x["imag"],float)
        obj=cls(float(value["shrinkage"]),float(value["eigen_floor_fraction"]))
        obj.mean=decode(value["mean"]); obj.covariance=decode(value["covariance"])
        obj.pseudo_covariance=decode(value["pseudo_covariance"]); obj.inverse_sqrt=decode(value["inverse_sqrt"])
        obj.diagnostics=dict(value["diagnostics"])
        return obj


@dataclass(frozen=True)
class JointFit:
    hypothesis: str; log_likelihood: float; rss: float; n: int; k: int; bic: float
    score: float; delays_chips: tuple[float, ...]; beta_m: tuple[float, ...] | None
    converged: bool; boundary: bool; valid: bool; reason: str | None; epoch_count: int; prn_count: int
    null_log_likelihood: float | None = None; null_k: int | None = None


@dataclass(frozen=True)
class CompiledProfilePlan:
    """Immutable fixed-template/fixed-whitener projection plan.

    Arrays are owned, read-only snapshots.  ``signature`` binds their values so
    callers never rely on mutable provider/whitener object identity.
    """
    taps: np.ndarray
    grid: np.ndarray
    whitener: np.ndarray
    mean: np.ndarray
    templates: np.ndarray
    h0_candidates: tuple[float, ...]
    h1_candidates: tuple[tuple[float, float], ...]
    h0_bases: tuple[np.ndarray | None, ...]
    h1_bases: tuple[np.ndarray | None, ...]
    h0_basis_bank: np.ndarray
    h1_basis_bank: np.ndarray
    h0_valid: np.ndarray
    h1_valid: np.ndarray
    row_chunk: int
    candidate_chunk: int
    signature: str
    counters: "ProfileRuntimeCounters" = field(compare=False, repr=False)
    provider_binding: str = ""


@dataclass
class ProfileRuntimeCounters:
    bank_evaluations: int = 0
    unique_winner_events: int = 0
    near_tie_events: int = 0
    scalar_fallback_events: int = 0
    scalar_fallback_candidates: int = 0
    scalar_fallback_epochs: int = 0
    scalar_fallback_lstsq_calls: int = 0
    scalar_fallback_elapsed_s: float = 0.

    def snapshot(self) -> dict[str, int | float]:
        return {name:getattr(self,name) for name in self.__dataclass_fields__}


def _readonly(value, dtype):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _projection_basis(design: np.ndarray) -> np.ndarray | None:
    """Legacy-compatible rank screening followed by a stable thin-SVD basis."""
    matrix = np.asarray(design, np.complex128)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    if not len(singular) or singular[0] == 0:
        return None
    legacy_rank = int(np.sum(singular > singular[0] * np.finfo(float).eps * max(matrix.shape)))
    if legacy_rank < matrix.shape[1]:
        return None
    if matrix.shape[1] == 2 and singular[-1] <= singular[0] * 1e-10:
        return None
    return _readonly(u[:, :matrix.shape[1]], np.complex128)


def compile_profile_plan(provider: TemplateProvider, taps: Sequence[float], delay_grid: Sequence[float],
                         whitener: ComplexWhitener | None = None, *, row_chunk: int = 4096,
                         candidate_chunk: int = 8) -> CompiledProfilePlan:
    taps_array = np.asarray(taps, np.float64); grid = np.unique(np.asarray(delay_grid, np.float64))
    if row_chunk < 1 or candidate_chunk < 1:
        raise ValueError("profile chunks must be positive")
    if whitener is None:
        W = np.eye(len(taps_array), dtype=np.complex128); mean = np.zeros(len(taps_array), np.complex128)
    else:
        if whitener.inverse_sqrt is None or whitener.mean is None:
            raise ValueError("frozen fitted whitener required")
        W = np.asarray(whitener.inverse_sqrt, np.complex128); mean = np.asarray(whitener.mean, np.complex128)
    templates = np.asarray([provider.evaluate(taps_array-float(delay)) for delay in grid], np.complex128)
    h0 = tuple(map(float, grid))
    h1 = tuple((float(da), float(ds)) for da in grid for ds in grid if abs(float(ds)) > 1e-10)
    by_delay = {float(delay): template for delay, template in zip(grid, templates)}
    h0_bases = tuple(_projection_basis(W @ by_delay[delay][:, None]) for delay in h0)
    h1_bases = tuple(_projection_basis(W @ np.column_stack(
        (by_delay[da], provider.evaluate(taps_array-da-ds)))) for da, ds in h1)
    def bank(bases,columns):
        valid=np.asarray([basis is not None for basis in bases],bool)
        values=np.zeros((len(bases),len(taps_array),columns),np.complex128)
        for index,basis in enumerate(bases):
            if basis is not None:values[index]=basis
        return _readonly(values,np.complex128),_readonly(valid,bool)
    h0_bank,h0_valid=bank(h0_bases,1);h1_bank,h1_valid=bank(h1_bases,2)
    frozen_taps = _readonly(taps_array, np.float64); frozen_grid = _readonly(grid, np.float64)
    frozen_W = _readonly(W, np.complex128); frozen_mean = _readonly(mean, np.complex128)
    frozen_templates = _readonly(templates, np.complex128)
    digest = hashlib.sha256()
    for value in (frozen_taps, frozen_grid, frozen_W, frozen_mean, frozen_templates):
        digest.update(value.tobytes(order="C"))
    provider_digest=hashlib.sha256();provider_digest.update(json.dumps({"mode":"analytic" if provider.values is None else "empirical",
      "analytic_approximation":provider.analytic_approximation,"paper_comparison_ready":provider.paper_comparison_ready,
      "provenance":provider.provenance}, sort_keys=True, default=str).encode())
    provider_digest.update(np.asarray(provider.offsets_chips,np.float64).tobytes(order="C"))
    if provider.values is not None:provider_digest.update(np.asarray(provider.values,np.complex128).tobytes(order="C"))
    provider_digest.update(frozen_templates.tobytes(order="C"))
    provider_binding=provider_digest.hexdigest();digest.update(provider_binding.encode())
    return CompiledProfilePlan(frozen_taps, frozen_grid, frozen_W, frozen_mean, frozen_templates,
                               h0, h1, h0_bases, h1_bases,h0_bank,h1_bank,h0_valid,h1_valid,int(row_chunk), int(candidate_chunk),
                               digest.hexdigest(),ProfileRuntimeCounters(),provider_binding)


@dataclass(frozen=True)
class ScoreResult:
    status: str
    score: float | None
    reason: str | None = None
    fit: JointFit | None = None

    @property
    def valid(self):
        return self.status == "AVAILABLE" and self.score is not None and np.isfinite(self.score)


def _profile_epoch(y, columns):
    d=np.column_stack(columns); amp, _, rank, _=np.linalg.lstsq(d,y,rcond=None)
    residual=y-d@amp
    return float(np.vdot(residual,residual).real), rank


def joint_profile_glrt(observations: Mapping[int, np.ndarray], los: Mapping[int, np.ndarray],
                       provider: TemplateProvider, taps: Sequence[float], delay_grid: Sequence[float],
                       *, hypothesis: str, whitener: ComplexWhitener | None = None,
                       profile_plan: CompiledProfilePlan | None = None, scalar_reference: bool = False,
                       beta_bounds_m: Sequence[tuple[float,float]] = ((-150,150),(-150,150),(-150,150),(-150,150)),
                       optimizer_starts: Sequence[Sequence[float]] | None = None,
                       maximum_condition_number: float = 1e6) -> JointFit:
    """Fixed-covariance proper-complex all-epoch profile likelihood.

    Delays and beta are optimized from observations. Evaluation residual variance
    is never estimated. ``optimizer_starts`` are fixed initializations, not
    candidate solutions or injected truth.
    """
    from scipy.optimize import minimize
    taps=np.asarray(taps,float); grid=np.unique(np.asarray(delay_grid,float)); prns=sorted(observations)
    arrays={p:np.asarray(observations[p],complex) for p in prns}
    if not prns or any(a.ndim!=2 or a.shape[1:]!=(len(taps),) or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("finite PRN epoch matrices aligned to taps required")
    epochs=sum(len(a) for a in arrays.values()); n=2*len(taps)*epochs
    if whitener is None:
        whitener=ComplexWhitener(shrinkage=0,eigen_floor_fraction=1e-12)
        whitener.mean=np.zeros(len(taps),complex); whitener.covariance=np.eye(len(taps)); whitener.inverse_sqrt=np.eye(len(taps)); whitener.pseudo_covariance=np.zeros((len(taps),len(taps)),complex); whitener.diagnostics={}
    if whitener.inverse_sqrt is None or whitener.mean is None: raise ValueError("frozen fitted whitener required")
    W=whitener.inverse_sqrt; mu=whitener.mean; logdet=float(np.linalg.slogdet(whitener.covariance)[1])
    def profile(y, columns):
        design=np.column_stack(columns); wd=W@design; wy=W@(y-mu)
        amp,_,rank,s=np.linalg.lstsq(wd,wy,rcond=None)
        if rank<len(columns) or (len(s)>1 and s[-1]<=s[0]*1e-10): return np.inf
        r=W@(y-design@amp-mu); return float(np.vdot(r,r).real)
    def prn_rss(p, authentic_delay, second_delay=None):
        first=provider.evaluate(taps-authentic_delay); columns=[first]
        if second_delay is not None:
            if abs(second_delay)<1e-10: return np.inf
            columns.append(provider.evaluate(taps-authentic_delay-second_delay))
        return sum(profile(y,columns) for y in arrays[p])
    if profile_plan is None and not scalar_reference:
        profile_plan = compile_profile_plan(provider, taps, grid, whitener)
    if profile_plan is not None:
        provider_digest=hashlib.sha256();provider_digest.update(json.dumps({"mode":"analytic" if provider.values is None else "empirical",
          "analytic_approximation":provider.analytic_approximation,"paper_comparison_ready":provider.paper_comparison_ready,
          "provenance":provider.provenance},sort_keys=True,default=str).encode())
        provider_digest.update(np.asarray(provider.offsets_chips,np.float64).tobytes(order="C"))
        if provider.values is not None:provider_digest.update(np.asarray(provider.values,np.complex128).tobytes(order="C"))
        provider_digest.update(np.asarray([provider.evaluate(taps-float(delay)) for delay in grid],np.complex128).tobytes(order="C"))
        if (not np.array_equal(profile_plan.taps, taps) or not np.array_equal(profile_plan.grid, grid) or
                not np.array_equal(profile_plan.whitener, W) or not np.array_equal(profile_plan.mean, mu) or
                profile_plan.provider_binding != provider_digest.hexdigest()):
            raise ValueError("compiled profile plan does not match fixed inputs")

        def batched_values(p, candidates, bases):
            profile_plan.counters.bank_evaluations += 1
            y = arrays[p]; totals = np.zeros(len(candidates), np.float64)
            if bases is profile_plan.h0_bases:bank,valid_mask=profile_plan.h0_basis_bank,profile_plan.h0_valid
            elif bases is profile_plan.h1_bases:bank,valid_mask=profile_plan.h1_basis_bank,profile_plan.h1_valid
            else:bank=None
            for row_start in range(0, len(y), profile_plan.row_chunk):
                block = (y[row_start:row_start+profile_plan.row_chunk]-mu) @ W.T
                for candidate_start in range(0, len(candidates), profile_plan.candidate_chunk):
                    stop = min(candidate_start+profile_plan.candidate_chunk, len(candidates))
                    valid=(np.flatnonzero(valid_mask[candidate_start:stop])+candidate_start).tolist() if bank is not None else [index for index in range(candidate_start,stop) if bases[index] is not None]
                    invalid=np.setdiff1d(np.arange(candidate_start,stop),valid,assume_unique=True);totals[invalid]=np.inf
                    if valid:
                        stacked=bank[valid] if bank is not None else np.stack([bases[index] for index in valid])
                        projected=np.einsum("rt,ctk->rck",block,stacked.conj(),optimize=False)
                        reconstructed=np.einsum("rck,ctk->rct",projected,stacked,optimize=False)
                        residual=block[:,None,:]-reconstructed
                        rss_rows=np.sum(residual.real*residual.real+residual.imag*residual.imag,axis=2,dtype=np.float64)
                        totals[np.asarray(valid)] += np.sum(rss_rows,axis=0,dtype=np.float64)
            return totals

        def refined_min(p, candidates, bases, second_override=None):
            values = batched_values(p, candidates, bases)
            minimum = float(np.min(values)); total_energy = float(np.sum(
                np.abs((arrays[p]-mu) @ W.T)**2, dtype=np.float64))
            tolerance = max(1e-12, 1e-10*max(total_energy, 1.))
            indices = np.flatnonzero(values <= minimum+tolerance)
            if len(indices) == 1:
                profile_plan.counters.unique_winner_events += 1
                return float(values[int(indices[0])]), int(indices[0])
            profile_plan.counters.near_tie_events += 1
            profile_plan.counters.scalar_fallback_events += 1
            profile_plan.counters.scalar_fallback_candidates += int(len(indices))
            profile_plan.counters.scalar_fallback_epochs += int(len(indices)*len(arrays[p]))
            started_refinement=time.perf_counter();refined = []
            for index in indices:
                candidate = candidates[int(index)]
                if second_override is not None:
                    da = float(candidate); ds = float(second_override)
                elif isinstance(candidate, tuple):
                    da, ds = candidate
                else:
                    da, ds = float(candidate), None
                before=profile_plan.counters.scalar_fallback_lstsq_calls
                exact=prn_rss(p,da,ds)
                profile_plan.counters.scalar_fallback_lstsq_calls=before+len(arrays[p])
                refined.append((exact, int(index)))
            profile_plan.counters.scalar_fallback_elapsed_s += time.perf_counter()-started_refinement
            return min(refined,key=lambda item:(item[0],item[1]))

        def vector_prn_min(p, second_delay=None):
            if second_delay is None:
                value, index = refined_min(p, profile_plan.h0_candidates, profile_plan.h0_bases)
                return value, profile_plan.h0_candidates[index]
            dynamic_bases = tuple(_projection_basis(W @ np.column_stack(
                (provider.evaluate(taps-da), provider.evaluate(taps-da-second_delay))))
                if abs(second_delay) > 1e-10 else None for da in profile_plan.h0_candidates)
            value, index = refined_min(p, profile_plan.h0_candidates, dynamic_bases, second_delay)
            return value, profile_plan.h0_candidates[index]
    authentic={}; rss0=0.
    for p in prns:
        value,d=(vector_prn_min(p) if profile_plan is not None else
                 min((prn_rss(p,float(d)),float(d)) for d in grid))
        rss0+=value; authentic[p]=d
    k0=2*epochs+len(prns)
    if hypothesis=="H0": rss,k,delays,beta,boundary,converged=rss0,k0,tuple(authentic[p] for p in prns),None,False,True
    elif hypothesis=="H1-independent":
        rss=0.; second={}; boundary=False
        for p in prns:
            if profile_plan is not None:
                value,index=refined_min(p,profile_plan.h1_candidates,profile_plan.h1_bases)
                da,d=profile_plan.h1_candidates[index]
            else:
                value,da,d=min((prn_rss(p,float(da),float(ds)),float(da),float(ds))
                               for da in grid for ds in grid if abs(ds)>1e-10)
            rss+=value; authentic[p]=da; second[p]=d
            boundary|=da in (grid[0],grid[-1]) or d in (grid[0],grid[-1])
        k=4*epochs+2*len(prns); delays=tuple(second[p] for p in prns); beta=None; converged=not boundary
    elif hypothesis=="H1-shared":
        missing=sorted(set(prns)-set(los))
        if missing:
            return JointFit(hypothesis,np.nan,np.nan,n,4*epochs+len(prns)+4,np.nan,np.nan,(),None,False,False,False,
                            "missing_los",epochs,len(prns))
        u=np.asarray([los[p] for p in prns],float); design=np.column_stack([-u,np.ones(len(u))])
        rank=np.linalg.matrix_rank(design); cond=np.linalg.cond(design)
        if len(prns)<5 or rank<4 or len(prns)-rank<1 or cond>maximum_condition_number:
            return JointFit(hypothesis,np.nan,np.nan,n,4*epochs+len(prns)+4,np.nan,np.nan,(),None,False,False,False,
                            "insufficient_prns" if len(prns)<5 else "rank_or_dof_or_condition",epochs,len(prns))
        bounds=tuple((float(a),float(b)) for a,b in beta_bounds_m)
        if len(bounds)!=4 or any(a>=b for a,b in bounds): raise ValueError("four valid fixed beta bounds required")
        def objective(b):
            seconds={p:float((-np.dot(los[p],b[:3])+b[3])/C_M_S*1_023_000) for p in prns}
            if any(abs(d)<1e-10 or d<grid[0] or d>grid[-1] for d in seconds.values()): return 1e100
            return sum((vector_prn_min(p,seconds[p])[0] if profile_plan is not None else
                        min(prn_rss(p,float(da),seconds[p]) for da in grid)) for p in prns)
        starts=list(optimizer_starts or ((0,0,0,25),(0,0,0,-25),(25,-25,10,50),(-25,25,-10,-50)))
        results=[minimize(objective,np.clip(np.asarray(s,float),[a for a,b in bounds],[b for a,b in bounds]),
                          method="Nelder-Mead",bounds=bounds,options={"maxiter":800,"xatol":1e-7,"fatol":1e-9}) for s in starts]
        result=min(results,key=lambda x:float(x.fun)); b=np.asarray(result.x,float); rss=float(result.fun)
        boundary=any(abs(b[i]-bounds[i][j])<1e-6 for i in range(4) for j in (0,1))
        seconds={p:float((-np.dot(los[p],b[:3])+b[3])/C_M_S*1_023_000) for p in prns}
        authentic={p:(vector_prn_min(p,seconds[p])[1] if profile_plan is not None else
                       min(grid,key=lambda da:prn_rss(p,float(da),seconds[p]))) for p in prns}
        delays=tuple(seconds[p] for p in prns); beta=tuple(map(float,b)); k=4*epochs+len(prns)+4
        converged=bool(result.success and np.isfinite(rss) and rss<1e99 and not boundary)
    else: raise ValueError("unknown hypothesis")
    ll0=-rss0-epochs*(len(taps)*math.log(math.pi)+logdet)
    ll=-rss-epochs*(len(taps)*math.log(math.pi)+logdet)
    bic=-2*ll+k*math.log(n); score=0. if hypothesis=="H0" else 2*(ll-ll0)-(k-k0)*math.log(n)
    valid=bool(converged and np.isfinite(rss) and not boundary)
    reason=None if valid else "boundary_or_nonconvergence"
    return JointFit(hypothesis,float(ll),float(rss),n,int(k),float(bic),float(score),delays,beta,
                    bool(converged),bool(boundary),valid,reason,epochs,len(prns),float(ll0),int(k0))


def detector_scores(individual_scores: Mapping[int,float], analytic_shared: JointFit | None,
                    neural_independent: JointFit, neural_shared: JointFit | None,
                    neural_energy: JointFit | None, power_score: float) -> dict:
    values=np.asarray(list(individual_scores.values()),float)
    if not len(values): raise ValueError("A1 requires per-PRN scores")
    return {"A1":float(np.max(values)), "A2":float(np.median(values)+np.mean(np.sort(values)[-min(4,len(values)):])),
            "A3":None if analytic_shared is None or not analytic_shared.valid else analytic_shared.score,
            "A4":None if not neural_independent.valid else neural_independent.score,
            "Full":None if neural_shared is None or not neural_shared.valid else neural_shared.score,
            "Neural-with-energy":None if neural_energy is None else float(neural_energy) if np.isscalar(neural_energy) else None if not neural_energy.valid else float(neural_energy.score),
            "Power-only":float(power_score)}


def normalized_pauc(labels, scores, maximum_fpr=.05):
    labels=np.asarray(labels,bool); scores=np.asarray(scores,float)
    if labels.all() or (~labels).all(): raise ValueError("both classes required")
    order=np.argsort(-scores,kind="mergesort"); y=labels[order]
    tp=np.r_[0,np.cumsum(y)]/y.sum(); fp=np.r_[0,np.cumsum(~y)]/(~y).sum()
    indices=np.where(fp<=maximum_fpr)[0].tolist()
    if fp[indices[-1]] < maximum_fpr:
        j=indices[-1]+1; frac=(maximum_fpr-fp[j-1])/(fp[j]-fp[j-1]);
        fp=np.r_[fp[indices],maximum_fpr]; tp=np.r_[tp[indices],tp[j-1]+frac*(tp[j]-tp[j-1])]
    else: fp=fp[indices]; tp=tp[indices]
    return float(np.trapezoid(tp,fp)/maximum_fpr)


def ranking_metrics(labels, scores):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y=np.asarray(labels,bool); s=np.asarray(scores,float)
    return {"auroc":float(roc_auc_score(y,s)),"pr_auc":float(average_precision_score(y,s)),
            "normalized_pauc_fpr_lte_0.05":normalized_pauc(y,s,.05)}


def calibration_thresholds(scores, roles):
    if set(roles)!={"normal_calibration"}: raise ValueError("threshold calibration is cleanStatic normal_calibration only")
    x=np.asarray(scores,float)
    return {"q99":float(np.quantile(x,.99,method="higher")),
            "q99.5":float(np.quantile(x,.995,method="higher")),
            "target_fpr_1pct":float(np.quantile(x,.99,method="higher"))}


def detection_metrics(availability_times, labels, alarms, recording_ids, *, cadence_s=.5, sustained_count=3,
                      attack_onset_s=None):
    """Causal alarm metrics using score availability, with recording/gap resets."""
    t=np.asarray(availability_times,float); y=np.asarray(labels,bool); a=np.asarray(alarms,bool); rec=np.asarray(recording_ids)
    if not (len(t)==len(y)==len(a)==len(rec)): raise ValueError("metric rows must align")
    sustained=np.zeros(len(t),bool); run=0; previous=None; previous_rec=None
    for i in np.argsort(np.rec.fromarrays([rec.astype(str),t])):
        if rec[i]!=previous_rec or previous is None or t[i]-previous>cadence_s+1e-9 or not a[i]: run=0
        if a[i]: run+=1
        if run>=sustained_count: sustained[i]=True
        previous=t[i]; previous_rec=rec[i]
    attack=np.flatnonzero(y); first=float(t[attack][np.argmax(sustained[attack])]) if len(attack) and sustained[attack].any() else None
    onset=float(attack_onset_s) if attack_onset_s is not None else float(t[attack].min()) if len(attack) else None
    return {"attack_detection_rate":float(a[y].mean()) if y.any() else None,
      "sustained_detection_rate":float(sustained[y].mean()) if y.any() else None,
      "first_sustained_delay_s":None if first is None else first-onset,
      "persistent_alarm_ratio":float(a[y].mean()) if y.any() else None,"causal_time_field":"availability_time_s"}


def paired_block_bootstrap(times, recording_ids, labels, left, right, *, repetitions=2000, seed=20260803, block_s=10.):
    """Paired percentile bootstrap of normalized-pAUC differences by physical blocks."""
    if repetitions!=2000 or block_s!=10.: raise ValueError("frozen bootstrap is exactly 2000 paired 10-second blocks")
    t=np.asarray(times,float); rec=np.asarray(recording_ids); y=np.asarray(labels,bool); a=np.asarray(left,float); b=np.asarray(right,float)
    common=np.isfinite(t)&np.isfinite(a)&np.isfinite(b)
    t,rec,y,a,b=t[common],rec[common],y[common],a[common],b[common]
    blocks=[];block_ids=[]
    for recording in np.unique(rec):
        idx=np.flatnonzero(rec==recording); origin=t[idx].min(); keys=np.floor((t[idx]-origin)/block_s).astype(int)
        for key in np.unique(keys):
            block=idx[keys==key]
            if t[block].max()-t[block].min()>=block_s-.5-1e-9:
                blocks.append(block);block_ids.append(f"{recording}:{int(key)}")
    if not blocks: raise ValueError("no common paired blocks")
    estimate=normalized_pauc(y,a)-normalized_pauc(y,b); rng=np.random.default_rng(seed); values=[]; draws=[]; attempts=0
    while len(values)<repetitions:
        attempts+=1
        if attempts>repetitions*100: raise ValueError("unable to obtain exactly 2000 valid paired draws")
        chosen=rng.integers(0,len(blocks),len(blocks)); idx=np.concatenate([blocks[i] for i in chosen])
        if y[idx].any() and (~y[idx]).any():
            values.append(normalized_pauc(y[idx],a[idx])-normalized_pauc(y[idx],b[idx])); draws.append(chosen.astype("<i8"))
    low,high=np.quantile(values,[.025,.975])
    draw_hash=hashlib.sha256(np.concatenate(draws).tobytes()).hexdigest()
    block_hash=hashlib.sha256("\n".join(block_ids).encode()).hexdigest()
    return {"repetitions":2000,"valid_draw_count":len(values),"attempt_count":attempts,"draw_index_sha256":draw_hash,"seed":seed,"block_s":10.,"common_events":int(common.sum()),
      "eligible_block_ids":block_ids,"eligible_block_ids_sha256":block_hash,"draw_representation":{"rng":"numpy.default_rng.PCG64","sample":"integers(0,n_blocks,n_blocks)","reject_single_class":True},
      "excluded_events":int((~common).sum()),"estimate":estimate,"ci95":[float(low),float(high)],
      "interpretation":"improvement demonstrated" if low>0 else "significant improvement not demonstrated"}


class SmallNuisanceConditioner:
    """Bounded deterministic MLP; no identity, label, score, future, or energy inputs."""
    ALLOWED=("prompt_normalized_complex_shape","cn0","h0_residual_quality","causal_tracking_quality","past_loop_error","elevation")
    def __init__(self, input_names, *, hidden=8, seed=20260803, with_energy=False):
        names=tuple(input_names); permitted=set(self.ALLOWED)|({"explicit_energy"} if with_energy else set())
        if not names or not set(names)<=permitted or ("explicit_energy" in names)!=with_energy:
            raise ValueError("conditioner input schema violates frozen no-energy/energy-ablation contract")
        self.input_names=names; self.hidden=int(hidden); self.seed=int(seed); self.with_energy=with_energy; self.summary=None; self.model=None

    def fit(self, x, residuals, roles, *, epochs=20, learning_rate=1e-3, require_gpu=False):
        import torch
        if set(roles)!={"normal_train"}: raise ValueError("conditioner fit is cleanStatic normal_train only")
        x=np.asarray(x,np.float32); z=np.asarray(residuals,complex)
        if x.shape!=(len(z),len(self.input_names)) or z.shape[1:]!=(9,): raise ValueError("conditioner rows do not align")
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if require_gpu and device.type!="cuda": raise RuntimeError("GPU required for campaign conditioner training")
        torch.manual_seed(self.seed); mean=x.mean(0); scale=x.std(0); scale[scale<1e-6]=1
        xt=torch.tensor((x-mean)/scale,device=device); yt=torch.tensor(np.c_[z.real,z.imag].astype(np.float32),device=device)
        model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],self.hidden),torch.nn.Tanh(),torch.nn.Linear(self.hidden,18)).to(device)
        opt=torch.optim.Adam(model.parameters(),lr=learning_rate); losses=[]
        for _ in range(int(epochs)):
            opt.zero_grad(); loss=torch.mean((model(xt)-yt)**2); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        state=b"".join(v.detach().cpu().numpy().tobytes() for _,v in sorted(model.state_dict().items()))
        self.model=model; self.mean=mean; self.scale=scale
        self.summary={"architecture":[x.shape[1],self.hidden,18],"seed":self.seed,"epochs":epochs,"loss":losses,
          "input_names":list(self.input_names),"with_energy":self.with_energy,"device":device.type,
          "scaler":{"mean":mean.tolist(),"scale":scale.tolist()},"weights_sha256":hashlib.sha256(state).hexdigest(),
          "fit_role":"cleanStatic normal_train"}
        return self

    def predict(self, x):
        import torch
        if self.model is None: raise RuntimeError("conditioner is not fitted")
        device=next(self.model.parameters()).device
        value=torch.tensor((np.asarray(x,np.float32)-self.mean)/self.scale,device=device)
        with torch.no_grad(): out=self.model(value).detach().cpu().numpy()
        return out[:,:9]+1j*out[:,9:]

    def serialize(self):
        import torch
        if self.model is None or self.summary is None: raise RuntimeError("conditioner is not fitted")
        state={k:v.detach().cpu().numpy().tolist() for k,v in self.model.state_dict().items()}
        return {"input_names":list(self.input_names),"hidden":self.hidden,"seed":self.seed,"with_energy":self.with_energy,
                "mean":self.mean.tolist(),"scale":self.scale.tolist(),"state_dict":state,"summary":self.summary}

    @classmethod
    def deserialize(cls, value, *, device="cpu"):
        import torch
        obj=cls(value["input_names"],hidden=int(value["hidden"]),seed=int(value["seed"]),with_energy=bool(value["with_energy"]))
        obj.mean=np.asarray(value["mean"],np.float32); obj.scale=np.asarray(value["scale"],np.float32)
        obj.model=torch.nn.Sequential(torch.nn.Linear(len(obj.input_names),obj.hidden),torch.nn.Tanh(),torch.nn.Linear(obj.hidden,18)).to(device)
        state={k:torch.tensor(v,dtype=obj.model.state_dict()[k].dtype,device=device) for k,v in value["state_dict"].items()}
        obj.model.load_state_dict(state); obj.summary=dict(value["summary"])
        packed=b"".join(v.detach().cpu().numpy().tobytes() for _,v in sorted(obj.model.state_dict().items()))
        if hashlib.sha256(packed).hexdigest()!=obj.summary["weights_sha256"]: raise ValueError("conditioner weights hash mismatch")
        return obj


DISQUALIFIERS=("clean_dynamic_fpr","gain_invariance","noise_gain_alarms","relation_destruction",
               "geometry_removal","complex_second_source","shortcut_controls")
REQUIRED=("complex_provenance","time_los_alignment","geometry_coverage","clean_dynamic_fpr",
          "gain_invariance","phase_invariance","noise_gain_alarms","relation_destruction",
          "full_improvement","full_a2_two_scenarios","shortcut_controls")


def derive_two_layer_decision(gates: Mapping[str,Mapping[str,object]]) -> dict:
    """Pure frozen decision with evaluated-disqualifier precedence."""
    bad=[name for name in DISQUALIFIERS if gates.get(name,{}).get("status")=="FAIL"]
    coverage_evidence={"time_los_alignment","geometry_coverage"}
    missing=[name for name in REQUIRED if gates.get(name,{}).get("status") not in {"PASS","FAIL"} or
             (name in coverage_evidence and gates.get(name,{}).get("status")!="PASS")]
    failed=[name for name in REQUIRED if name not in coverage_evidence and gates.get(name,{}).get("status")=="FAIL"]
    if bad: core="R2C_CORE_NOT_SUPPORTED"
    elif missing: core="R2C_CORE_INCONCLUSIVE"
    elif failed: core="R2C_CORE_NOT_SUPPORTED"
    else: core="R2C_CORE_SUPPORTED"
    paper_names=("b0_authentic_common_support","full_b0_comparison","empirical_wide_template","paper_gates")
    paper=all(gates.get(n,{}).get("status")=="PASS" for n in paper_names)
    overall="PHYSICS_SUPPORTED" if core=="R2C_CORE_SUPPORTED" and paper else core
    return {"core_physics_verdict":core,"paper_comparison_ready":paper,"verdict":overall,
            "observed_disqualifiers":bad,"missing_required_evidence":missing,"failed_required_gates":failed}


def run_full_controls(score_fn: Callable, observations: Mapping[int,np.ndarray], los, threshold: float,
                      provider: TemplateProvider | None = None, taps: Sequence[float] | None = None,
                      *, seed=20260803):
    """Run every control through the identical frozen Full scorer and threshold."""
    rng=np.random.default_rng(seed); original={p:np.asarray(y,complex).copy() for p,y in observations.items()}
    if not original: raise ValueError("Full controls require observations")
    provider=provider or TemplateProvider.analytic(); taps=np.asarray(taps if taps is not None else np.arange(-.5,.5001,.125),float)
    def evaluate(value,pairing,*,cn0_degradation_db=0.):
        try:
            result=score_fn(value,pairing,cn0_degradation_db=cn0_degradation_db)
        except TypeError:
            result=score_fn(value,pairing)
        if isinstance(result,ScoreResult): return result
        if isinstance(result,JointFit):
            return ScoreResult("AVAILABLE",float(result.score),fit=result) if result.valid else ScoreResult("UNAVAILABLE_INVALID_SHARED_FIT",None,result.reason,result)
        value=float(result)
        return ScoreResult("AVAILABLE",value) if np.isfinite(value) else ScoreResult("UNAVAILABLE_INVALID_SCORE",None,"nonfinite")
    baseline=evaluate(original,los); rows=[]
    def transform(fn): return {p:fn(p,y.copy()) for p,y in original.items()}
    def add(kind,params,changed,changed_los=los,*,cn0_degradation_db=0.):
        post=evaluate(changed,changed_los,cn0_degradation_db=cn0_degradation_db)
        both=baseline.valid and post.valid
        rows.append({"kind":kind,"parameters":params,"seed":seed,"threshold":float(threshold),
          "support_count":sum(len(x) for x in changed.values()),"baseline_status":baseline.status,"perturbed_status":post.status,
          "baseline_reason":baseline.reason,"perturbed_reason":post.reason,
          "pre_score":baseline.score,"post_score":post.score,
          "effect_size":post.score-baseline.score if both else None,
          "pre_alarm":bool(baseline.score>threshold) if both else None,"post_alarm":bool(post.score>threshold) if both else None})
    for gain in (.5,.75,1.,1.5,2.): add("gain",{"gain":gain},transform(lambda p,y:y*gain))
    add("slow_agc",{"minimum":.75,"maximum":1.25},transform(lambda p,y:y*np.linspace(.75,1.25,len(y))[:,None]))
    for phase in (0.,math.pi/4,math.pi/2,math.pi): add("global_phase",{"radians":phase},transform(lambda p,y:y*np.exp(1j*phase)))
    power=np.mean([np.mean(np.abs(y)**2) for y in original.values()])
    def noisy(scale): return transform(lambda p,y:y+scale*np.sqrt(power/2)*(rng.normal(size=y.shape)+1j*rng.normal(size=y.shape)))
    add("awgn",{"relative_sigma":.05},noisy(.05)); add("cn0_degradation",{"db":6.0},noisy(math.sqrt(10**(.6)-1)),cn0_degradation_db=6.)
    add("matched_power_noise",{"relative_power":1.0},transform(lambda p,y:np.sqrt(power/2)*(rng.normal(size=y.shape)+1j*rng.normal(size=y.shape))))
    def quant(p,y):
        scale=max(np.max(np.abs(y.real)),np.max(np.abs(y.imag)),1e-12)
        return np.round(y.real/scale*127)/127*scale+1j*np.round(y.imag/scale*127)/127*scale
    add("quantization",{"bits":8},transform(quant))
    multipath={}
    for index,(p,y) in enumerate(sorted(original.items())):
        delay=(-.45+.13*index)% .8-.4; amp=.1+.03*(index%4); phase=rng.uniform(-math.pi,math.pi)
        multipath[p]=y+amp*np.exp(1j*phase)*provider.evaluate(taps-delay)[None,:]*np.sqrt(np.mean(np.abs(y)**2))
    add("non_shared_multipath",{"per_prn":True},multipath)
    for delay in (-.35,.35):
        for ratio in (.1,.25,.5,1.):
            phase=float(rng.uniform(-math.pi,math.pi)); injected={}
            for p,y in original.items():
                component=provider.evaluate(taps-delay)[None,:]*np.exp(1j*phase)
                component*=math.sqrt(ratio*np.mean(np.abs(y)**2)/max(np.mean(np.abs(component)**2),1e-15))
                injected[p]=y+component
            add("second_source_injection",{"delay_chips":delay,"power_ratio":ratio,"phase_rad":phase},injected)
    keys=sorted(los); perm=keys[1:]+keys[:1]
    add("relation_destruction",{"permutation":perm},original,{p:los[q] for p,q in zip(keys,perm)})
    invariance=[r for r in rows if r["kind"] in {"gain","slow_agc","global_phase"}]
    status="PASS" if invariance and all(r["pre_alarm"] is not None and r["pre_alarm"]==r["post_alarm"] for r in invariance) else "FAIL"
    return {"schema":"gnss-doppler-lab.r2c-full-controls.v2","seed":seed,"threshold":float(threshold),"support_count":sum(len(x) for x in original.values()),
            "rows":rows,"computed_rows":len(rows),"status":status,
            "baseline_status":baseline.status,"baseline_score":baseline.score,"baseline_reason":baseline.reason,
            "criteria":{"invariance_alarm_agreement":sum(r["pre_alarm"] is not None and r["pre_alarm"]==r["post_alarm"] for r in invariance)/len(invariance)}}
