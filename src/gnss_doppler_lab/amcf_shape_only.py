"""Leakage-resistant core for the preregistered AMCF Shape-Only ablation.

The model receives only side-tap shape tensors. Prompt, quality fields, IDs,
and times are deliberately absent from its forward API. IDs and source times
are accepted only by causal indexing helpers.
"""
from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import copy
import hashlib
import math
from typing import Any
import numpy as np
import torch
from torch import nn

TAP_NAMES = ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")
PROMPT_INDEX = 4
SIDE_INDICES = (0, 1, 2, 3, 5, 6, 7, 8)
COMPLEX_SCHEMA = ("median_real", "median_imag", "mad_real", "mad_imag")
MAGNITUDE_SCHEMA = ("median_abs", "mad_abs")
ALLOWED_SEEDS = (101, 202, 303)

@dataclass(frozen=True)
class PromptGate:
    minimum: float
    eps: float = 1e-12
    quantile: float = .005
    def __post_init__(self):
        if not np.isfinite(self.minimum) or self.minimum < 0 or not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("finite nonnegative minimum and positive eps required")

@dataclass(frozen=True)
class RobustScaler:
    median: np.ndarray
    iqr: np.ndarray
    def __post_init__(self):
        med = np.asarray(self.median, dtype=np.float64).copy(); iqr = np.asarray(self.iqr, dtype=np.float64).copy()
        if med.ndim != 2 or med.shape != iqr.shape or not np.isfinite(med).all() or not np.isfinite(iqr).all() or np.any(iqr <= 0):
            raise ValueError("finite per-tap/per-dimension scaler required")
        med.setflags(write=False); iqr.setflags(write=False)
        object.__setattr__(self, "median", med); object.__setattr__(self, "iqr", iqr)

@dataclass(frozen=True)
class FeatureWindow:
    recording_id: str
    segment_id: str
    channel_id: str
    prn: Any
    role: str
    source_start: float
    source_end: float
    features: np.ndarray
    def __post_init__(self):
        f = np.asarray(self.features, dtype=np.float32).copy()
        if f.ndim != 2 or f.shape[0] != 8 or not np.isfinite(f).all(): raise ValueError("window features must be finite [8,D] side taps")
        if not self.source_start < self.source_end: raise ValueError("actual source interval must have positive width")
        f.setflags(write=False); object.__setattr__(self, "features", f)

@dataclass(frozen=True)
class HistoryExample:
    current: FeatureWindow
    history: tuple[FeatureWindow, ...]

@dataclass(frozen=True)
class CleanTensorSplit:
    current: np.ndarray
    history: np.ndarray
    role: str
    recording_id: str
    def __post_init__(self):
        c = np.asarray(self.current, dtype=np.float32).copy(); h = np.asarray(self.history, dtype=np.float32).copy()
        if c.ndim != 3 or c.shape[1] != 8 or h.shape != (len(c), 12, 8, c.shape[2]): raise ValueError("current [N,8,D] and full history [N,12,8,D] required")
        if not np.isfinite(c).all() or not np.isfinite(h).all(): raise ValueError("finite tensors required")
        c.setflags(write=False); h.setflags(write=False); object.__setattr__(self, "current", c); object.__setattr__(self, "history", h)

def _only(role, recording, expected_role):
    r = np.asarray(role, dtype=str); s = np.asarray(recording, dtype=str)
    if r.ndim != 1 or s.shape != r.shape or not len(r) or np.any(r != expected_role) or np.any(s != "cleanStatic"):
        raise ValueError(f"cleanStatic {expected_role} only")

def fit_prompt_gate(prompt, *, role, recording, quantile=.005, eps=1e-12):
    _only(role, recording, "train"); p = np.abs(np.asarray(prompt, dtype=np.complex128))
    if p.ndim != 1 or len(p) != len(role) or not np.isfinite(p).all(): raise ValueError("aligned finite Prompt is required")
    return PromptGate(float(np.quantile(p, quantile, method="higher")), float(eps), float(quantile))

def normalize_by_prompt(correlators, gate):
    c = np.asarray(correlators, dtype=np.complex128)
    if c.ndim != 2 or c.shape[1] != 9: raise ValueError("correlators must be [N,9] in fixed tap order")
    p = c[:, PROMPT_INDEX]; pmag = np.abs(p); finite = np.isfinite(c.real).all(axis=1) & np.isfinite(c.imag).all(axis=1)
    valid = finite & (pmag >= gate.minimum * (1.0 - 1e-14)) & (pmag > 0)
    out = np.full((len(c), 8), np.nan + 1j*np.nan, dtype=np.complex128); den = pmag[valid]**2 + gate.eps
    out[valid] = c[valid][:, SIDE_INDICES] * np.conjugate(p[valid, None]) / den[:, None]
    return out, valid

def robust_features(normalized_sides, representation):
    z = np.asarray(normalized_sides, dtype=np.complex128)
    if z.ndim != 2 or z.shape[1] != 8 or not len(z) or not np.isfinite(z).all(): raise ValueError("finite normalized side rows [N,8] required")
    if representation == "complex":
        mr = np.median(z.real, axis=0); mi = np.median(z.imag, axis=0)
        dr = np.median(np.abs(z.real-mr), axis=0); di = np.median(np.abs(z.imag-mi), axis=0)
        return np.stack((mr, mi, dr, di), axis=1).astype(np.float32)
    if representation == "magnitude":
        mag = np.abs(z); mm = np.median(mag, axis=0); dm = np.median(np.abs(mag-mm), axis=0)
        return np.stack((mm, dm), axis=1).astype(np.float32)
    raise ValueError("representation must be complex or magnitude")

def assert_target_iqr(clean_train_features, tolerance=1e-8):
    x = np.asarray(clean_train_features, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != 8 or not np.isfinite(x).all(): raise ValueError("finite clean train features [N,8,D] required")
    iqr = np.quantile(x, .75, axis=0, method="linear") - np.quantile(x, .25, axis=0, method="linear")
    if np.any(iqr <= tolerance): raise ValueError(f"every target dimension must have clean-train IQR above tolerance; failed {np.argwhere(iqr <= tolerance).tolist()}")
    return iqr

def fit_robust_scaler(features, *, role, recording, tolerance=1e-8):
    _only(role, recording, "train"); x = np.asarray(features, dtype=np.float64)
    if len(x) != len(role): raise ValueError("aligned provenance required")
    return RobustScaler(np.median(x, axis=0), assert_target_iqr(x, tolerance))

def transform_robust(features, scaler):
    x = np.asarray(features, dtype=np.float64)
    if x.shape[-2:] != scaler.median.shape: raise ValueError("feature schema does not match immutable scaler")
    return (x-scaler.median)/scaler.iqr

def build_history_examples(windows, *, history_length=12, cadence_s=.5, tolerance=1e-9):
    indexed = list(enumerate(windows)); indexed.sort(key=lambda q:(q[1].recording_id,str(q[1].segment_id),str(q[1].channel_id),str(q[1].prn),q[1].source_end,q[0]))
    groups = {}
    for _, w in indexed: groups.setdefault((w.recording_id,str(w.segment_id),str(w.channel_id),str(w.prn),w.role), []).append(w)
    out = []
    for rows in groups.values():
        rows.sort(key=lambda w:w.source_end)
        for i in range(history_length, len(rows)):
            hist=rows[i-history_length:i]; cur=rows[i]; ends=np.asarray([w.source_end for w in hist]+[cur.source_end])
            if np.allclose(np.diff(ends),cadence_s,rtol=0,atol=tolerance) and all(w.source_end < cur.source_end for w in hist): out.append(HistoryExample(cur,tuple(hist)))
    out.sort(key=lambda e:(e.current.source_end,e.current.recording_id,str(e.current.prn))); return out

class ShapeOnlyModel(nn.Module):
    def __init__(self, feature_dim, hidden=32, df=4.):
        super().__init__()
        if feature_dim not in (2,4): raise ValueError("feature_dim must be 4 complex or 2 magnitude")
        self.feature_dim=int(feature_dim); self.hidden=int(hidden); self.df=float(df)
        self.input_adapter=nn.Linear(self.feature_dim,self.hidden)
        self.history_gru=nn.GRU(self.hidden,self.hidden,num_layers=1,batch_first=True)
        self.decoder=nn.Sequential(nn.Linear(self.hidden*2,self.hidden*4),nn.GELU(),nn.Linear(self.hidden*4,self.hidden),nn.GELU())
        self.output_head=nn.Linear(self.hidden,self.feature_dim*2)
    @staticmethod
    def _masked_pool(tokens, mask):
        weights=mask.to(tokens.dtype)
        if torch.any(weights.sum(dim=-1)<=0): raise ValueError("at least one observed side tap required")
        return (tokens*weights[...,None]).sum(dim=-2)/weights.sum(dim=-1,keepdim=True)
    def forward(self, history, current, observed_mask):
        if history.ndim!=4 or current.ndim!=3 or observed_mask.ndim!=2: raise ValueError("history [B,12,8,D], current [B,8,D], mask [B,8] required")
        b=current.shape[0]
        if history.shape!=(b,12,8,self.feature_dim) or current.shape[1:]!=(8,self.feature_dim) or observed_mask.shape!=(b,8): raise ValueError("shape-only tensor schema mismatch")
        hp=self.input_adapter(history).mean(dim=2); _,state=self.history_gru(hp)
        cp=self._masked_pool(self.input_adapter(current),observed_mask.bool()); latent=self.decoder(torch.cat((state[-1],cp),dim=-1)); raw=self.output_head(latent)
        loc,rs=raw.chunk(2,dim=-1); return loc,torch.nn.functional.softplus(rs)+1e-4

def student_t_nll(target, location, scale, df=4.):
    v=torch.as_tensor(df,dtype=target.dtype,device=target.device)
    return torch.lgamma(v/2)-torch.lgamma((v+1)/2)+.5*torch.log(v*math.pi)+torch.log(scale)+(v+1)/2*torch.log1p(((target-location)/scale).square()/v)

def _loo_loss(model, history, current, target_pairs=None):
    if target_pairs is None: sample=np.repeat(np.arange(len(current)),8); target=np.tile(np.arange(8),len(current))
    else: sample=np.array(target_pairs[:,0],dtype=int,copy=True); target=np.array(target_pairs[:,1],dtype=int,copy=True)
    si=torch.as_tensor(sample,device=history.device); ti=torch.as_tensor(target,device=current.device); hs=history[si]; cs=current[si]
    mask=torch.ones((len(sample),8),dtype=torch.bool,device=current.device); mask[torch.arange(len(sample),device=current.device),ti]=False
    loc,scale=model(hs,cs,mask); y=cs[torch.arange(len(sample),device=current.device),ti]
    return student_t_nll(y,loc,scale,model.df).mean()

def all9_loo_score(model, history, current, order=None):
    h=np.asarray(history,dtype=np.float32); c=np.asarray(current,dtype=np.float32)
    if h.shape!=(12,8,model.feature_dim) or c.shape!=(8,model.feature_dim): raise ValueError("one full-history/current tensor required")
    use=list(range(8) if order is None else order)
    if sorted(use)!=list(range(8)): raise ValueError("all eight side targets exactly once required")
    device=next(model.parameters()).device; hh=torch.as_tensor(np.repeat(h[None],8,axis=0),device=device); cc=torch.as_tensor(np.repeat(c[None],8,axis=0),device=device)
    mask=torch.ones((8,8),dtype=torch.bool,device=device)
    for row,target in enumerate(use): mask[row,target]=False
    with torch.no_grad():
        loc,scale=model(hh,cc,mask); targets=cc[torch.arange(8,device=device),torch.as_tensor(use,device=device)]; side=student_t_nll(targets,loc,scale,model.df).mean(dim=1).cpu().numpy()
    return float(np.mean(np.sort(side)[-2:]))

def make_fixed_validation_bank(sample_count, seed):
    bank=np.asarray([(i,t) for i in range(int(sample_count)) for t in range(8)],dtype=np.int64); rng=np.random.default_rng(int(seed)); bank=bank[rng.permutation(len(bank))]
    digest=hashlib.sha256(bank.astype("<i8",copy=False).tobytes()).hexdigest(); bank.setflags(write=False); return bank,digest

def _validate_clean_split(data, role):
    if data.recording_id!="cleanStatic" or data.role!=role: raise ValueError(f"cleanStatic {role} split required")

def fit_clean_model(train, validation, *, seed, hidden=32, batch_size=256, learning_rate=1e-3, max_epochs=200, patience=20):
    _validate_clean_split(train,"train"); _validate_clean_split(validation,"validation")
    if int(seed) not in ALLOWED_SEEDS: raise ValueError(f"seed must be one of {ALLOWED_SEEDS}")
    if train.current.shape[2]!=validation.current.shape[2] or batch_size<=0 or max_epochs<=0 or patience<=0: raise ValueError("compatible positive training configuration required")
    torch.manual_seed(int(seed)); rng=np.random.default_rng(int(seed)); model=ShapeOnlyModel(train.current.shape[2],hidden=hidden); optimizer=torch.optim.AdamW(model.parameters(),lr=float(learning_rate))
    tc=torch.as_tensor(np.array(train.current, copy=True)); th=torch.as_tensor(np.array(train.history, copy=True)); vc=torch.as_tensor(np.array(validation.current, copy=True)); vh=torch.as_tensor(np.array(validation.history, copy=True)); bank,bank_hash=make_fixed_validation_bank(len(vc),seed)
    best_loss=math.inf; best_model=None; best_optimizer=None; best_epoch=-1; stale=0; updates=0; finite=True; stopped=False; hashes=[]; epochs_run=0
    for epoch in range(int(max_epochs)):
        model.train(); order=rng.permutation(len(tc))
        for a in range(0,len(order),int(batch_size)):
            ids=torch.as_tensor(order[a:a+int(batch_size)]); optimizer.zero_grad(set_to_none=True); loss=_loo_loss(model,th[ids],tc[ids])
            if not torch.isfinite(loss): finite=False; break
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()): finite=False; break
            optimizer.step(); updates+=1
        epochs_run=epoch+1
        if not finite: break
        model.eval()
        with torch.no_grad(): val_loss=float(_loo_loss(model,vh,vc,bank).item())
        hashes.append(bank_hash)
        if not np.isfinite(val_loss): finite=False; break
        if val_loss < best_loss-1e-12:
            best_loss=val_loss; best_epoch=epoch; best_model=copy.deepcopy(model.state_dict()); best_optimizer=copy.deepcopy(optimizer.state_dict()); stale=0
        else:
            stale+=1
            if stale>=patience: stopped=True; break
    rm=best_model is not None; ro=best_optimizer is not None
    if rm: model.load_state_dict(best_model)
    if ro: optimizer.load_state_dict(best_optimizer)
    converged=bool(stopped and finite and epochs_run<=max_epochs and updates>1)
    audit={"seed":int(seed),"optimizer":"AdamW","learning_rate":float(learning_rate),"batch_size":int(batch_size),"max_epochs":int(max_epochs),"patience":int(patience),"epochs_run":epochs_run,"optimizer_updates":updates,"best_epoch":best_epoch,"validation_bank_hash":bank_hash,"validation_bank_hash_epoch_invariant":len(set(hashes))<=1,"finite":bool(finite),"patience_early_stop":bool(stopped),"converged":converged,"stop_reason":"patience_converged" if converged else ("nonfinite" if not finite else "cap_nonconverged"),"best_checkpoint_restored":rm,"best_optimizer_restored":ro}
    return model,optimizer,audit

def primary_status(seed_audits):
    if any(not bool(a.get("converged")) for a in seed_audits): return "INCOMPLETE: nonconverged seed"
    seeds = [int(a.get("seed", -1)) for a in seed_audits]
    if sorted(seeds) != list(ALLOWED_SEEDS): return "INCOMPLETE: missing required seed"
    return "COMPLETE"

def conformal_evidence(calibration, scores):
    cal=np.asarray(calibration,dtype=float); x=np.asarray(scores,dtype=float)
    if cal.ndim!=1 or not len(cal) or not np.isfinite(cal).all() or not np.isfinite(x).all(): raise ValueError("finite one-dimensional calibration/scores required")
    p=(1+np.sum(cal[None,:]>=x[...,None],axis=-1))/(len(cal)+1); return p,-np.log(p)

def calibration_loo_evidence(calibration):
    cal=np.asarray(calibration,dtype=float)
    if cal.ndim!=1 or len(cal)<2 or not np.isfinite(cal).all(): raise ValueError("at least two finite calibration scores required")
    count=np.sum(cal[None,:]>=cal[:,None],axis=1)-1; return -np.log((1+count)/len(cal))

def higher_thresholds(ensemble_loo_evidence, *, role, recording):
    _only(role,recording,"calibration"); e=np.asarray(ensemble_loo_evidence,dtype=float)
    if e.ndim==1: e=e[None,:]
    if e.ndim!=2 or e.shape[1]!=len(role) or not np.isfinite(e).all(): raise ValueError("aligned finite seed evidence required")
    mean=e.mean(axis=0); return {"q99":float(np.quantile(mean,.99,method="higher")),"q995":float(np.quantile(mean,.995,method="higher")),"comparison":"strict_greater","primary":"q99","count":len(mean)}

def alarm_flags(scores, threshold): return np.asarray(scores,dtype=float)>float(threshold)

def phase_masks(source_start, source_end, onset_s):
    start=np.asarray(source_start,dtype=float); end=np.asarray(source_end,dtype=float); onset=float(onset_s)
    if start.shape!=end.shape or np.any(end<=start): raise ValueError("aligned actual source intervals required")
    return {"stable_pre":(start>=30.) & (end<=onset-20.),"post":start>=onset,"persistent":start>=onset+40.}

def _has_sustained_three(times, alarm, mask, tolerance=1e-9):
    ids=np.flatnonzero(mask)
    for i in range(max(0,len(ids)-2)):
        run=ids[i:i+3]
        if len(run)==3 and np.array_equal(run,np.arange(run[0],run[0]+3)) and alarm[run].all() and np.allclose(np.diff(times[run]),.5,rtol=0,atol=tolerance): return True
    return False

def sustained_three_delay(decision_times, alarms, wholly_post, *, stable_pre, onset_s):
    t=np.asarray(decision_times,dtype=float); a=np.asarray(alarms,dtype=bool); post=np.asarray(wholly_post,dtype=bool); stable=np.asarray(stable_pre,dtype=bool)
    if not (t.shape==a.shape==post.shape==stable.shape): raise ValueError("aligned alarm rows required")
    if _has_sustained_three(t,a,stable): return "N/A: already alarming in stable-pre"
    ids=np.flatnonzero(post)
    for i in range(max(0,len(ids)-2)):
        run=ids[i:i+3]
        if len(run)==3 and np.array_equal(run,np.arange(run[0],run[0]+3)) and a[run].all() and np.allclose(np.diff(t[run]),.5,rtol=0,atol=1e-9): return float(t[run[0]]-float(onset_s))
    return None

def common_timestamp_pairs(complex_values, comparator_values):
    keys=sorted(set(complex_values).intersection(comparator_values)); return np.asarray(keys,dtype=float),np.asarray([complex_values[k] for k in keys],dtype=float),np.asarray([comparator_values[k] for k in keys],dtype=float)

def paired_block_bootstrap(timestamps, complex_values, comparator_values, statistic, *, reps=2000, block_s=10., seed=0):
    t=np.asarray(timestamps,dtype=float); a=np.asarray(complex_values,dtype=float); b=np.asarray(comparator_values,dtype=float)
    if not (t.ndim==a.ndim==b.ndim==1 and len(t)==len(a)==len(b) and len(t)) or reps<=0 or block_s<=0: raise ValueError("nonempty same-timestamp paired rows required")
    order=np.argsort(t,kind="mergesort"); t=t[order]; a=a[order]; b=b[order]; group=np.floor((t-t[0])/block_s).astype(int); blocks=[np.flatnonzero(group==k) for k in np.unique(group)]; rng=np.random.default_rng(int(seed)); draws=[]
    for _ in range(int(reps)):
        ids=np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),len(blocks))])[:len(t)]; draws.append(float(statistic(a[ids])-statistic(b[ids])))
    return {"estimate":float(statistic(a)-statistic(b)),"ci_low":float(np.quantile(draws,.025)),"ci_high":float(np.quantile(draws,.975)),"reps":int(reps),"block_s":float(block_s),"delta_definition":"Complex-comparator","paired_on":"same timestamps"}

def schema_collapse_audit(complex_features, magnitude_features, complex_schema, magnitude_schema, tracked_prn_counts, stable_pre_alarms):
    c=np.asarray(complex_features,dtype=float); m=np.asarray(magnitude_features,dtype=float); finite=bool(np.isfinite(c).all() and np.isfinite(m).all()); nonconstant=bool(c.size and m.size and np.ptp(c)>0 and np.ptp(m)>0); schema=tuple(complex_schema)==COMPLEX_SCHEMA and tuple(magnitude_schema)==MAGNITUDE_SCHEMA; tracked=float(np.median(np.asarray(tracked_prn_counts,dtype=float))) if len(tracked_prn_counts) else 0.; alarms=np.asarray(stable_pre_alarms,dtype=bool); not_all=bool(len(alarms) and np.mean(alarms)!=1.)
    out={"identical_fixed_feature_schema":schema,"no_cn0_branch":True,"finite":finite,"nonconstant":nonconstant,"tracked_prn_median":tracked,"stable_pre_alarm_rate":float(np.mean(alarms)) if len(alarms) else None}; out["pass"]=bool(schema and finite and nonconstant and tracked>1 and not_all); return out


def make_feature_window(correlators, times, roles, *, source_end, recording_id, segment_id, channel_id, prn, gate, representation):
    """Summarize actual rows in the open-left, closed-right one-second window."""
    c = np.asarray(correlators, dtype=np.complex128); t = np.asarray(times, dtype=float); r = np.asarray(roles, dtype=str)
    if c.shape != (len(t), 9) or r.shape != t.shape or not np.isfinite(t).all(): raise ValueError("aligned source rows required")
    end = float(source_end); start = end - 1.0; ids = np.flatnonzero((t > start) & (t <= end))
    if not len(ids): raise ValueError("source window is empty")
    unique = np.unique(r[ids])
    if len(unique) != 1: raise ValueError("source rows must be wholly in the same role")
    normalized, valid = normalize_by_prompt(c[ids], gate)
    if not valid.any(): raise ValueError("source window has no valid Prompt-normalized rows")
    return FeatureWindow(str(recording_id), str(segment_id), str(channel_id), prn, str(unique[0]), start, end, robust_features(normalized[valid], representation))


def epl_loo_diagnostic(model, history, current):
    """Auxiliary E/L-labelled two-target LOO; Prompt is never an input."""
    h = np.asarray(history, dtype=np.float32); c = np.asarray(current, dtype=np.float32)
    if h.shape != (12, 8, model.feature_dim) or c.shape != (8, model.feature_dim): raise ValueError("one full-history/current tensor required")
    device = next(model.parameters()).device; hh = torch.as_tensor(np.repeat(h[None], 2, axis=0), device=device); cc = torch.as_tensor(np.repeat(c[None], 2, axis=0), device=device)
    mask = torch.zeros((2, 8), dtype=torch.bool, device=device); mask[0, 4] = True; mask[1, 3] = True; target = torch.as_tensor([3, 4], device=device)
    with torch.no_grad():
        loc, scale = model(hh, cc, mask); y = cc[torch.arange(2, device=device), target]; nll = student_t_nll(y, loc, scale, model.df).mean(dim=1).cpu().numpy()
    return {"E": float(nll[0]), "L": float(nll[1]), "mean": float(np.mean(nll))}


def epoch_prn_score(prn_raw_scores):
    """Permutation-invariant epoch aggregate over tracked PRNs."""
    values = np.asarray(list(prn_raw_scores.values()), dtype=float)
    if not len(values) or not np.isfinite(values).all(): raise ValueError("finite PRN scores required")
    return float(np.median(values))
