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
PRIMARY_QUANTILE = .005
PRIMARY_EPS = 1e-12
PRIMARY_HIDDEN = 32
PRIMARY_BATCH_SIZE = 256
PRIMARY_LEARNING_RATE = 1e-3
PRIMARY_MAX_EPOCHS = 200
PRIMARY_PATIENCE = 20
PRIMARY_HISTORY = 12
PRIMARY_STRIDE_S = .5
PRIMARY_SOURCE_WINDOW_S = 1.0
_FACTORY_TOKEN = object()
FORBIDDEN_FEATURE_FIELDS = frozenset({
    "cn0", "c_n0", "log_prompt", "prompt_magnitude", "valid_fraction",
    "valid_count", "rejected_count", "raw_count", "recording_id",
    "scenario_id", "prn", "context_id",
})

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


@dataclass(frozen=True)
class CausalAuditManifest:
    history_length: int
    cadence_s: float
    source_window_s: float
    causal_checks_passed: bool
    split_role_checked: bool
    row_digest: str


@dataclass(frozen=True, init=False)
class AuditedCleanSplit:
    """Fit-capable tensors constructible only by the causal example factory."""
    current: np.ndarray
    history: np.ndarray
    role: str
    recording_id: str
    manifest: CausalAuditManifest

    def __init__(self, current, history, role, recording_id, manifest, *, _factory_token=None):
        if _factory_token is not _FACTORY_TOKEN:
            raise TypeError("AuditedCleanSplit must be created by the validated factory")
        c = np.asarray(current, dtype=np.float32).copy()
        h = np.asarray(history, dtype=np.float32).copy()
        if c.ndim != 3 or c.shape[1] != 8 or h.shape != (len(c), 12, 8, c.shape[2]):
            raise ValueError("current [N,8,D] and full history [N,12,8,D] required")
        if not len(c) or not np.isfinite(c).all() or not np.isfinite(h).all():
            raise ValueError("nonempty finite tensors required")
        c.setflags(write=False); h.setflags(write=False)
        object.__setattr__(self, "current", c); object.__setattr__(self, "history", h)
        object.__setattr__(self, "role", str(role)); object.__setattr__(self, "recording_id", str(recording_id))
        object.__setattr__(self, "manifest", manifest)


def build_audited_clean_split(examples, *, role, cadence_s=.5, source_window_s=1.0):
    """Validate provenance and bind it immutably to fit tensors."""
    rows = tuple(examples)
    if not rows:
        raise ValueError("nonempty causal examples required")
    currents, histories, digest_rows = [], [], []
    for ex in rows:
        if not isinstance(ex, HistoryExample) or len(ex.history) != PRIMARY_HISTORY:
            raise ValueError("validated 12-history examples required")
        cur = ex.current; hist = ex.history
        identity = (cur.recording_id, cur.segment_id, cur.channel_id, str(cur.prn), cur.role)
        if cur.recording_id != "cleanStatic" or cur.role != role:
            raise ValueError(f"cleanStatic {role} split required")
        if any((w.recording_id, w.segment_id, w.channel_id, str(w.prn), w.role) != identity for w in hist):
            raise ValueError("history crossed recording/segment/channel/PRN/role boundary")
        ends = np.asarray([w.source_end for w in hist] + [cur.source_end], dtype=float)
        if not np.all(np.diff(ends) > 0) or not np.allclose(np.diff(ends), cadence_s, rtol=0, atol=1e-9):
            raise ValueError("history must be strictly causal at exact cadence")
        all_windows = (*hist, cur)
        if any(not math.isclose(w.source_end-w.source_start, source_window_s, rel_tol=0, abs_tol=1e-9) for w in all_windows):
            raise ValueError("actual source-window width mismatch")
        currents.append(cur.features); histories.append(np.stack([w.features for w in hist]))
        digest_rows.extend((w.recording_id, w.segment_id, w.channel_id, str(w.prn), w.role, w.source_start, w.source_end) for w in all_windows)
    digest = hashlib.sha256(repr(digest_rows).encode()).hexdigest()
    manifest = CausalAuditManifest(PRIMARY_HISTORY, float(cadence_s), float(source_window_s), True, True, digest)
    return AuditedCleanSplit(np.stack(currents), np.stack(histories), role, "cleanStatic", manifest, _factory_token=_FACTORY_TOKEN)

def _only(role, recording, expected_role):
    r = np.asarray(role, dtype=str); s = np.asarray(recording, dtype=str)
    if r.ndim != 1 or s.shape != r.shape or not len(r) or np.any(r != expected_role) or np.any(s != "cleanStatic"):
        raise ValueError(f"cleanStatic {expected_role} only")

def fit_prompt_gate(prompt, *, role, recording, quantile=.005, eps=1e-12, primary=True):
    if primary and (quantile != PRIMARY_QUANTILE or eps != PRIMARY_EPS):
        raise ValueError("primary frozen Prompt quantile/epsilon mismatch")
    _only(role, recording, "train"); p = np.abs(np.asarray(prompt, dtype=np.complex128))
    if p.ndim != 1 or len(p) != len(role) or not np.isfinite(p).all(): raise ValueError("aligned finite Prompt is required")
    return PromptGate(float(np.quantile(p, quantile, method="higher")), float(eps), float(quantile))

def normalize_by_prompt(correlators, gate):
    c = np.asarray(correlators, dtype=np.complex128)
    if c.ndim != 2 or c.shape[1] != 9: raise ValueError("correlators must be [N,9] in fixed tap order")
    p = c[:, PROMPT_INDEX]; pmag = np.abs(p); finite = np.isfinite(c.real).all(axis=1) & np.isfinite(c.imag).all(axis=1)
    valid = finite & (pmag >= gate.minimum) & (pmag > 0)
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

def make_target_tuple_bank(sample_count, target_indices=range(8)):
    targets = tuple(int(x) for x in target_indices)
    if int(sample_count) <= 0 or not targets or len(set(targets)) != len(targets) or any(x < 0 or x >= 8 for x in targets):
        raise ValueError("positive sample count and unique side target indices required")
    bank = np.asarray([(i, target) for i in range(int(sample_count)) for target in targets], dtype=np.int64)
    bank.setflags(write=False)
    return bank


def make_fixed_validation_bank(sample_count, seed):
    bank = np.array(make_target_tuple_bank(sample_count), copy=True)
    rng=np.random.default_rng(int(seed)); bank=bank[rng.permutation(len(bank))]
    digest=hashlib.sha256(bank.astype("<i8",copy=False).tobytes()).hexdigest(); bank.setflags(write=False); return bank,digest


def _validate_clean_split(data, role):
    if not isinstance(data, AuditedCleanSplit):
        raise TypeError("fit requires an audited split from build_audited_clean_split")
    if data.recording_id != "cleanStatic" or data.role != role:
        raise ValueError(f"cleanStatic {role} split required")
    if not data.manifest.causal_checks_passed or not data.manifest.split_role_checked:
        raise ValueError("causal audit manifest is incomplete")


def _primary_fit_guard(*, hidden, batch_size, learning_rate, max_epochs, patience):
    actual = (int(hidden), int(batch_size), float(learning_rate), int(max_epochs), int(patience))
    frozen = (PRIMARY_HIDDEN, PRIMARY_BATCH_SIZE, PRIMARY_LEARNING_RATE, PRIMARY_MAX_EPOCHS, PRIMARY_PATIENCE)
    if actual != frozen:
        raise ValueError(f"primary frozen training configuration mismatch: required {frozen}")


def fit_clean_model(train, validation, *, seed, hidden=32, batch_size=256,
                    learning_rate=1e-3, max_epochs=200, patience=20, primary=True):
    _validate_clean_split(train,"train"); _validate_clean_split(validation,"validation")
    if int(seed) not in ALLOWED_SEEDS: raise ValueError(f"seed must be one of {ALLOWED_SEEDS}")
    if primary:
        _primary_fit_guard(hidden=hidden, batch_size=batch_size, learning_rate=learning_rate,
                           max_epochs=max_epochs, patience=patience)
        for data in (train, validation):
            if (data.manifest.history_length, data.manifest.cadence_s, data.manifest.source_window_s) != (PRIMARY_HISTORY, PRIMARY_STRIDE_S, PRIMARY_SOURCE_WINDOW_S):
                raise ValueError("primary frozen history/stride/source-window configuration mismatch")
    if train.current.shape[2]!=validation.current.shape[2] or batch_size<=0 or max_epochs<=0 or patience<=0:
        raise ValueError("compatible positive training configuration required")
    torch.manual_seed(int(seed)); rng=np.random.default_rng(int(seed))
    model=ShapeOnlyModel(train.current.shape[2],hidden=hidden)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(learning_rate))
    tc=torch.as_tensor(np.array(train.current, copy=True)); th=torch.as_tensor(np.array(train.history, copy=True))
    vc=torch.as_tensor(np.array(validation.current, copy=True)); vh=torch.as_tensor(np.array(validation.history, copy=True))
    train_bank = make_target_tuple_bank(len(tc))
    bank,bank_hash=make_fixed_validation_bank(len(vc),seed)
    best_loss=math.inf; best_model=None; best_optimizer=None; best_epoch=-1; stale=0
    updates=0; finite=True; stopped=False; hashes=[]; epochs_run=0; updates_per_epoch=[]; tuple_batch_sizes_per_epoch=[]; maximum_tuple_batch=0
    for epoch in range(int(max_epochs)):
        model.train(); order=rng.permutation(len(train_bank)); epoch_updates=0; epoch_batch_sizes=[]
        for a in range(0,len(order),int(batch_size)):
            tuple_ids=order[a:a+int(batch_size)]
            pairs=train_bank[tuple_ids]
            maximum_tuple_batch=max(maximum_tuple_batch,len(pairs)); epoch_batch_sizes.append(len(pairs))
            optimizer.zero_grad(set_to_none=True); loss=_loo_loss(model,th,tc,pairs)
            if not torch.isfinite(loss): finite=False; break
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()): finite=False; break
            optimizer.step(); updates+=1; epoch_updates+=1
        updates_per_epoch.append(epoch_updates); tuple_batch_sizes_per_epoch.append(epoch_batch_sizes); epochs_run=epoch+1
        if not finite: break
        expected=math.ceil(len(train_bank)/int(batch_size))
        if epoch_updates != expected:
            raise RuntimeError("optimizer update audit mismatch")
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
    audit={"seed":int(seed),"optimizer":"AdamW","learning_rate":float(learning_rate),"batch_size":int(batch_size),
           "max_epochs":int(max_epochs),"patience":int(patience),"epochs_run":epochs_run,"optimizer_updates":updates,
           "optimizer_updates_per_epoch":updates_per_epoch,"optimizer_tuple_batch_sizes_per_epoch":tuple_batch_sizes_per_epoch,"train_target_tuples":len(train_bank),
           "target_tuples_per_sample":8,"maximum_optimizer_tuple_batch":maximum_tuple_batch,
           "expected_updates_per_full_epoch":math.ceil(len(train_bank)/int(batch_size)),"best_epoch":best_epoch,
           "validation_bank_hash":bank_hash,"validation_bank_hash_epoch_invariant":len(set(hashes))<=1,
           "finite":bool(finite),"patience_early_stop":bool(stopped),"converged":converged,
           "stop_reason":"patience_converged" if converged else ("nonfinite" if not finite else "cap_nonconverged"),
           "best_checkpoint_restored":rm,"best_optimizer_restored":ro,
           "fit_manifest_digest":train.manifest.row_digest,"primary":bool(primary)}
    return model,optimizer,audit

def primary_status(seed_audits):
    audits = tuple(seed_audits)
    seeds = [int(a.get("seed", -1)) for a in audits]
    if any(not bool(a.get("converged")) for a in audits):
        return "INCOMPLETE: nonconverged seed"
    if len(audits) != len(ALLOWED_SEEDS) or set(seeds) != set(ALLOWED_SEEDS) or len(set(seeds)) != len(seeds):
        return "INCOMPLETE: missing required seed"
    return "COMPLETE"


@dataclass(frozen=True)
class SeedCalibration:
    seed: int
    scores: np.ndarray
    timestamps: np.ndarray
    indices: np.ndarray
    def __post_init__(self):
        scores=np.asarray(self.scores,dtype=float).copy(); times=np.asarray(self.timestamps,dtype=float).copy(); ids=np.asarray(self.indices).copy()
        if self.seed not in ALLOWED_SEEDS or scores.ndim!=1 or not len(scores) or times.shape!=scores.shape or ids.shape!=scores.shape:
            raise ValueError("aligned seed calibration rows required")
        if not np.isfinite(scores).all() or not np.isfinite(times).all() or len(np.unique(ids))!=len(ids):
            raise ValueError("finite calibration rows with unique indices required")
        for x in (scores,times,ids): x.setflags(write=False)
        object.__setattr__(self,"scores",scores); object.__setattr__(self,"timestamps",times); object.__setattr__(self,"indices",ids)

def conformal_evidence(calibration, scores):
    cal=np.asarray(calibration,dtype=float); x=np.asarray(scores,dtype=float)
    if cal.ndim!=1 or not len(cal) or not np.isfinite(cal).all() or not np.isfinite(x).all(): raise ValueError("finite one-dimensional calibration/scores required")
    p=(1+np.sum(cal[None,:]>=x[...,None],axis=-1))/(len(cal)+1); return p,-np.log(p)

def calibration_loo_evidence(calibration):
    cal=np.asarray(calibration,dtype=float)
    if cal.ndim!=1 or len(cal)<2 or not np.isfinite(cal).all(): raise ValueError("at least two finite calibration scores required")
    count=np.sum(cal[None,:]>=cal[:,None],axis=1)-1; return -np.log((1+count)/len(cal))

def higher_thresholds(ensemble_loo_evidence, *, role, recording, primary=True):
    if primary:
        raise ValueError("primary metadata required; use primary_higher_thresholds")
    _only(role,recording,"calibration"); e=np.asarray(ensemble_loo_evidence,dtype=float)
    if e.ndim==1: e=e[None,:]
    if e.ndim!=2 or e.shape[1]!=len(role) or not np.isfinite(e).all(): raise ValueError("aligned finite seed evidence required")
    mean=e.mean(axis=0); return {"q99":float(np.quantile(mean,.99,method="higher")),"q995":float(np.quantile(mean,.995,method="higher")),"comparison":"strict_greater","primary":"q99","count":len(mean),"diagnostic":True}


def primary_higher_thresholds(seed_calibrations, seed_audits, *, role, recording):
    _only(role, recording, "calibration")
    if set(seed_calibrations) != set(ALLOWED_SEEDS) or len(seed_calibrations) != len(ALLOWED_SEEDS):
        raise ValueError(f"primary requires exactly seeds {ALLOWED_SEEDS}")
    if primary_status(seed_audits) != "COMPLETE":
        raise ValueError("all required primary seeds must be converged")
    rows = [seed_calibrations[seed] for seed in ALLOWED_SEEDS]
    if any(not isinstance(row, SeedCalibration) or row.seed != seed for row,seed in zip(rows,ALLOWED_SEEDS)):
        raise ValueError("seed calibration identity mismatch")
    if any(len(row.scores) != len(role) for row in rows):
        raise ValueError("calibration provenance length mismatch")
    ref=rows[0]
    if any(not np.array_equal(row.indices,ref.indices) or not np.array_equal(row.timestamps,ref.timestamps) for row in rows[1:]):
        raise ValueError("three-seed calibration timestamp/index alignment mismatch")
    evidence=np.vstack([calibration_loo_evidence(row.scores) for row in rows])
    out=higher_thresholds(evidence,role=role,recording=recording,primary=False)
    out.update({"required_seeds":list(ALLOWED_SEEDS),"seed_alignment_verified":True,"all_seeds_converged":True,"diagnostic":False})
    return out

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

def _roc_auc(labels, scores):
    y=np.asarray(labels,dtype=bool); x=np.asarray(scores,dtype=float)
    pos=x[y]; neg=x[~y]
    if not len(pos) or not len(neg):
        raise ValueError("ROC-AUC requires both label classes")
    return float(np.mean((pos[:,None]>neg[None,:]) + .5*(pos[:,None]==neg[None,:])))


def paired_block_bootstrap(timestamps, labels, complex_values, comparator_values, *, metric,
                           reps=2000, block_s=10., seed=0, phase_mask=None,
                           complex_threshold=None, comparator_threshold=None):
    # Paired common-timestamp bootstrap; phase-rate blocks form after filtering.
    t=np.asarray(timestamps,dtype=float); y=np.asarray(labels); a=np.asarray(complex_values,dtype=float); b=np.asarray(comparator_values,dtype=float)
    if not (t.ndim==y.ndim==a.ndim==b.ndim==1 and len(t)==len(y)==len(a)==len(b) and len(t)):
        raise ValueError("nonempty aligned labels and same-timestamp paired scores required")
    if int(reps) < 2000: raise ValueError("at least 2000 bootstrap replicates required")
    if block_s<=0 or not np.isfinite(t).all() or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("finite paired rows and positive blocks required")
    if metric not in {"roc_auc","post_detection","stable_pre_fpr"}:
        raise ValueError("metric must be roc_auc, post_detection, or stable_pre_fpr")
    phase_local=metric != "roc_auc"
    if phase_local:
        if phase_mask is None or np.asarray(phase_mask,dtype=bool).shape != t.shape:
            raise ValueError("aligned phase_mask required for phase-rate metrics")
        if complex_threshold is None or comparator_threshold is None:
            raise ValueError("representation-specific thresholds required for phase-rate metrics")
        keep=np.asarray(phase_mask,dtype=bool)
        t,y,a,b=t[keep],y[keep],a[keep],b[keep]
        if not len(t): raise ValueError("selected phase has no rows")
    order=np.argsort(t,kind="mergesort"); t=t[order]; y=y[order]; a=a[order]; b=b[order]
    group=np.floor((t-t[0])/block_s).astype(int)
    blocks=[np.flatnonzero(group==k) for k in np.unique(group)]
    def stat(labels_, scores_, which):
        if metric=="roc_auc": return _roc_auc(labels_,scores_)
        threshold=complex_threshold if which=="complex" else comparator_threshold
        return float(np.mean(scores_>float(threshold)))
    estimate=stat(y,a,"complex")-stat(y,b,"comparator")
    rng=np.random.default_rng(int(seed)); draws=[]; attempts=0
    while len(draws)<int(reps):
        attempts+=1
        if attempts>int(reps)*100: raise ValueError("unable to obtain defined phase-local bootstrap statistics")
        ids=np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),len(blocks))])[:len(t)]
        try: draws.append(float(stat(y[ids],a[ids],"complex")-stat(y[ids],b[ids],"comparator")))
        except ValueError: continue
    return {"estimate":float(estimate),"ci_low":float(np.quantile(draws,.025)),"ci_high":float(np.quantile(draws,.975)),
            "reps":int(reps),"block_s":float(block_s),"metric":metric,"phase_local":phase_local,
            "delta_definition":"Complex-comparator","paired_on":"same timestamps",
            "resampling":"paired labels/scores on common timestamp blocks"}

def schema_collapse_audit(clean_features, scenario_features, *, feature_schemas,
                          tracked_prn_counts, stable_pre_alarms, tolerance=1e-8):
    required={"DS7","DS8"}
    if set(scenario_features)!=required or set(tracked_prn_counts)!=required or set(stable_pre_alarms)!=required:
        raise ValueError("DS7 and DS8 must each be present in collapse audit")
    expected={"complex":COMPLEX_SCHEMA,"magnitude":MAGNITUDE_SCHEMA}
    if set(clean_features)!=set(expected) or set(feature_schemas)!=set(expected):
        raise ValueError("complex and magnitude representations required")
    schema_exact=all(tuple(feature_schemas[k])==expected[k] for k in expected)
    declared={str(field).lower() for fields in feature_schemas.values() for field in fields}
    forbidden_absent=not bool(declared & FORBIDDEN_FEATURE_FIELDS)
    def representation_ok(x, dims):
        z=np.asarray(x,dtype=float)
        if z.ndim!=3 or z.shape[1:]!=(8,dims) or not len(z) or not np.isfinite(z).all(): return False, []
        iqr=np.quantile(z,.75,axis=0,method="linear")-np.quantile(z,.25,axis=0,method="linear")
        failures=np.argwhere(iqr<=tolerance).tolist()
        return not failures,failures
    clean_checks={}; clean_pass=True
    for rep,dims in (("complex",4),("magnitude",2)):
        ok,failures=representation_ok(clean_features[rep],dims); clean_checks[rep]={"pass":ok,"iqr_failures":failures}; clean_pass &= ok
    scenario_out={}
    for ds in sorted(required):
        reps=scenario_features[ds]
        if set(reps)!=set(expected): raise ValueError(f"{ds} representation set mismatch")
        rep_checks={}; rep_pass=True
        for rep,dims in (("complex",4),("magnitude",2)):
            ok,failures=representation_ok(reps[rep],dims); rep_checks[rep]={"pass":ok,"iqr_failures":failures}; rep_pass &= ok
        tracked=np.asarray(tracked_prn_counts[ds],dtype=float); alarms=np.asarray(stable_pre_alarms[ds],dtype=bool)
        median=float(np.median(tracked)) if len(tracked) else 0.; rate=float(np.mean(alarms)) if len(alarms) else None
        row={"schema_exact":schema_exact,"forbidden_fields_absent":forbidden_absent,
             "each_tap_dimension_valid":bool(clean_pass and rep_pass),"representation_checks":rep_checks,
             "tracked_prn_median":median,"stable_pre_alarm_rate":rate}
        row["pass"]=bool(schema_exact and forbidden_absent and clean_pass and rep_pass and median>1 and rate is not None and rate<1.)
        scenario_out[ds]=row
    return {"expected_scenarios":["DS7","DS8"],"clean_checks":clean_checks,"scenarios":scenario_out,
            "pass":all(row["pass"] for row in scenario_out.values())}


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


@dataclass(frozen=True)
class ScenarioEvidence:
    scenario: str
    complex_stable_pre_fpr: float
    complex_auc: float
    magnitude_auc: float
    complex_minus_magnitude_auc_ci_low: float
    seed_complex_auc: Mapping[int, float]
    seed_magnitude_auc: Mapping[int, float]
    b0_auc: float
    complex_post_detection: float
    b0_post_detection: float
    b0_stable_pre_fpr: float
    collapsed: bool
    threshold_quantile: str = "q99"
    def __post_init__(self):
        scalars=(self.complex_stable_pre_fpr,self.complex_auc,self.magnitude_auc,
                 self.complex_minus_magnitude_auc_ci_low,self.b0_auc,
                 self.complex_post_detection,self.b0_post_detection,self.b0_stable_pre_fpr)
        if not np.isfinite(scalars).all(): raise ValueError("finite full-precision scenario evidence required")
        if set(self.seed_complex_auc)!=set(ALLOWED_SEEDS) or set(self.seed_magnitude_auc)!=set(ALLOWED_SEEDS):
            raise ValueError("exact same-seed AUC evidence required")
        if not np.isfinite(list(self.seed_complex_auc.values())).all() or not np.isfinite(list(self.seed_magnitude_auc.values())).all():
            raise ValueError("finite same-seed AUC evidence required")


@dataclass(frozen=True)
class PrimaryDecision:
    quantile: str
    final: str
    amcf_wcl: str
    criteria: Mapping[str, Mapping[str, Any]]
    scenarios: tuple[str, ...]


def q99_primary_decision(scenario_evidence, seed_audits, *, quantile="q99"):
    if quantile != "q99": raise ValueError("only q99 can enter the primary GO/NO-GO decision")
    rows=tuple(scenario_evidence); required=("DS1","DS2","DS3","DS7","DS8")
    names=[row.scenario for row in rows]
    if len(rows)!=5 or set(names)!=set(required) or len(set(names))!=5:
        raise ValueError(f"exactly scenarios {required} are required")
    by={row.scenario:row for row in rows}
    ordered=[by[x] for x in required]
    if any(row.threshold_quantile != "q99" for row in ordered):
        raise ValueError("q995 evidence cannot enter the q99 primary decision")
    criteria={}
    def add(name, passed, evidence): criteria[name]={"pass":bool(passed),"evidence":evidence}
    fprs={r.scenario:r.complex_stable_pre_fpr for r in ordered}
    add("stable_pre_fpr_all_below_0.05",all(v<.05 for v in fprs.values()),fprs)
    auc_directions={r.scenario:r.complex_auc-r.magnitude_auc for r in ordered}; auc_count=sum(v>0 for v in auc_directions.values())
    add("complex_auc_gt_magnitude_4_of_5",auc_count>=4,{"count":auc_count,"deltas":auc_directions})
    ci={r.scenario:r.complex_minus_magnitude_auc_ci_low for r in ordered}; ci_count=sum(v>0 for v in ci.values())
    add("auc_bootstrap_ci_lower_gt_zero_3_of_5",ci_count>=3,{"count":ci_count,"ci_lower":ci})
    seed_dirs={r.scenario:{seed:r.seed_complex_auc[seed]-r.seed_magnitude_auc[seed] for seed in ALLOWED_SEEDS} for r in ordered}
    seed_counts={ds:sum(v>0 for v in values.values()) for ds,values in seed_dirs.items()}
    add("same_seed_direction_each_scenario",all(v>=2 for v in seed_counts.values()),{"positive_counts":seed_counts,"deltas":seed_dirs})
    b0={}; b0_count=0
    for r in ordered:
        auc_delta=r.complex_auc-r.b0_auc; post_delta=r.complex_post_detection-r.b0_post_detection; fpr_delta=r.complex_stable_pre_fpr-r.b0_stable_pre_fpr
        passed=(auc_delta>=.02 or post_delta>=.05) and fpr_delta<=.01
        b0[r.scenario]={"pass":passed,"auc_delta":auc_delta,"post_delta":post_delta,"fpr_delta":fpr_delta}
        b0_count+=int(passed)
    add("beats_b0_with_fpr_guard_3_of_5",b0_count>=3,{"count":b0_count,"scenarios":b0})
    status=primary_status(seed_audits)
    add("all_required_seeds_converged",status=="COMPLETE",{"status":status,"audits":list(seed_audits)})
    collapse={ds:not by[ds].collapsed for ds in ("DS7","DS8")}
    add("ds7_ds8_no_collapse",all(collapse.values()),collapse)
    passed=all(row["pass"] for row in criteria.values())
    return PrimaryDecision("q99","GO" if passed else "NO-GO","GO candidate" if passed else "AMCF WCL no-go",criteria,required)
