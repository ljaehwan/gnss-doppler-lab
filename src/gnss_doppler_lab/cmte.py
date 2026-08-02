"""Leakage-safe CMTE primitives around the frozen B0 innovation extractor.

CMTE is a sequential conformal evidence detector with empirically calibrated
false-alarm control.  Distribution fitting, validation calibration and scoring
are deliberately separate operations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib, json, re
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import pandas as pd

TAP_ORDER=("E4","E3","E2","E","P","L","L2","L3","L4")
RESIDUAL_COLUMNS=tuple(f"residual_{i:03d}" for i in range(9))
SCORE_METHODS=("rmse","diag_mahalanobis","full_shrinkage_mahalanobis","max_standardized_tap")
KAPPAS=(.25,.5,.75); EPSILON=1e-8
_FORBIDDEN=re.compile(r"(?:ds[1-4]|attack|spoof|external[\s_-]*validation)",re.I)


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def _array(f): return f.loc[:,RESIDUAL_COLUMNS].to_numpy(float)

def validate_residual_frame(frame,*,require_history_reset=False):
    required={"run_id","prn","window_start_s","window_end_s","window_mid_s",*RESIDUAL_COLUMNS}
    missing=sorted(required-set(frame));
    if missing: raise ValueError(f"residual input missing columns: {missing}")
    extras=sorted(c for c in frame if c.startswith("residual_") and c not in RESIDUAL_COLUMNS)
    if extras: raise ValueError(f"exactly nine ordered residual columns required; extras={extras}")
    a=frame[["window_start_s","window_end_s","window_mid_s",*RESIDUAL_COLUMNS]].to_numpy(float)
    if not np.isfinite(a).all(): raise ValueError("timing and residual values must be finite")
    if np.any(frame.window_end_s.to_numpy(float)<=frame.window_start_s.to_numpy(float)): raise ValueError("causal windows require end > start")
    if "b0_prn_node_rmse" in frame:
        expected=np.sqrt(np.mean(_array(frame)**2,axis=1))
        if not np.allclose(expected,frame.b0_prn_node_rmse,atol=1e-9,rtol=1e-6): raise ValueError("B0 residual RMSE mismatch")
    if require_history_reset:
        if "target_window_index" not in frame: raise ValueError("target_window_index required for history reset audit")
        identity="history_id" if "history_id" in frame else "run_id"
        first=frame.sort_values([identity,"prn","window_end_s"],kind="mergesort").groupby([identity,"prn"],sort=False).head(1)
        if not first.target_window_index.astype(int).eq(12).all(): raise ValueError("history reset requires first target_window_index=12")
    return {"feature_columns":list(RESIDUAL_COLUMNS),"prn_identity_feature":False,"tap_order":list(TAP_ORDER),"availability_field":"window_end_s"}


def _flatten_strings(value):
    if isinstance(value,dict):
        for k,v in value.items(): yield str(k); yield from _flatten_strings(v)
    elif isinstance(value,(list,tuple)):
        for v in value: yield from _flatten_strings(v)
    else: yield str(value)


def validate_clean_provenance(manifest_path, frame, *, checkpoint_sha256=None, node_path=None):
    """Positive allowlist: exact cleanStatic identity plus authenticated producer manifest."""
    p=Path(manifest_path).resolve(strict=True); doc=json.loads(p.read_text())
    required={"schema","scenario","role","producer_grade","source_sha256","checkpoint_sha256","node_sha256"}
    missing=sorted(required-set(doc))
    if missing: raise ValueError(f"clean manifest missing {missing}")
    if doc["scenario"]!="cleanStatic" or doc["role"]!="normal_clean": raise ValueError("manifest must identify exact cleanStatic normal_clean role")
    if doc["producer_grade"] not in {"verified_node_artifact","reconstructed_equivalence"}: raise ValueError("untrusted producer grade")
    for key in ("source_sha256","checkpoint_sha256","node_sha256"):
        if not re.fullmatch(r"[0-9a-fA-F]{64}",str(doc[key])): raise ValueError(f"invalid {key}")
    corpus=[str(p),*list(_flatten_strings(doc))]
    corpus += [str(x) for c in frame.select_dtypes(include=["object","string"]).columns for x in frame[c].dropna().unique()]
    if node_path is not None:
        node=Path(node_path).resolve(strict=True); corpus.append(str(node))
        if file_sha256(node).lower()!=doc["node_sha256"].lower(): raise ValueError("node SHA mismatch")
    if _FORBIDDEN.search(" ".join(corpus)): raise ValueError("forbidden attack/spoof/external-validation provenance token")
    if checkpoint_sha256 and doc["checkpoint_sha256"].lower()!=checkpoint_sha256.lower(): raise ValueError("manifest checkpoint SHA mismatch")
    return doc


def audit_roles(roles):
    if set(roles)!={"train","validation","test"}: raise ValueError("roles must be exactly train, validation, test")
    seen=set(); sources={}
    for role in ("train","validation","test"):
        f=roles[role]; validate_residual_frame(f,require_history_reset=True); sources[role]=sorted(f.run_id.astype(str).unique())
        if role!="test" and _FORBIDDEN.search(" ".join(sources[role])): raise ValueError("normal-only role contains forbidden provenance")
        keys=set(zip(f.run_id.astype(str),f.prn.astype(str),f.window_end_s.astype(float)))
        if seen&keys: raise ValueError("roles are not disjoint")
        seen|=keys
    return {"disjoint":True,"fit_sources":sources["train"],"sources":sources,"normal_only_fit":True,
            "attack_prefix_fitted":False,"threshold_source":"validation_only","test_used_for_calibration":False}

@dataclass
class FitState:
    mean:np.ndarray; covariance:np.ndarray; diagonal_scales:np.ndarray
    calibration:dict[str,np.ndarray]=field(default_factory=dict)
    epsilon:float=EPSILON; checkpoint_sha256:str=""; metadata:dict=field(default_factory=dict)

@dataclass
class SequentialState:
    run_id:str|None=None; s1_capitals:list[float]=field(default_factory=list); s2_e_cusum:float=0.
    def reset(self,run_id=None): self.run_id=run_id; self.s1_capitals.clear(); self.s2_e_cusum=0.


def fit_distribution(frame,*,epsilon=EPSILON,shrinkage=None,checkpoint_sha256=""):
    validate_residual_frame(frame)
    if frame.empty or epsilon<=0: raise ValueError("normal train residuals must be nonempty and epsilon positive")
    if _FORBIDDEN.search(" ".join(frame.run_id.astype(str).unique())): raise ValueError("normal-only fit forbids attack provenance")
    x=_array(frame); mu=x.mean(0); d=x-mu; emp=d.T@d/max(1,len(x)-1)
    lam=min(1.,10./max(10.,float(len(x)))) if shrinkage is None else float(shrinkage)
    if not 0<=lam<=1: raise ValueError("shrinkage must be in [0,1]")
    diag=np.maximum(np.diag(emp),epsilon); cov=(1-lam)*emp+lam*np.diag(diag)+epsilon*np.eye(9)
    return FitState(mu,cov,np.sqrt(np.maximum(np.diag(cov),epsilon)),{},epsilon,checkpoint_sha256,
      {"fit_scope":"train_distribution_only_shared_no_prn_identity","covariance":"fixed deterministic diagonal shrinkage",
       "shrinkage":lam,"attack_method_selection":False,"raw_taps":"prompt-relative magnitudes",
       "residuals":"signed standardized target-prediction","tap_order":list(TAP_ORDER),"seq_len":12,
       "default_method":"full_shrinkage_mahalanobis","q_semantics":"squared quadratic"})


def _score_matrix(x,state):
    d=x-state.mean; inv=np.linalg.inv(state.covariance)
    return {"rmse":np.sqrt(np.mean(x*x,axis=1)),
      "diag_mahalanobis":np.sum((d/state.diagonal_scales)**2,axis=1),
      "full_shrinkage_mahalanobis":np.maximum(0,np.einsum("ni,ij,nj->n",d,inv,d)),
      "max_standardized_tap":np.max(np.abs(d/state.diagonal_scales),axis=1)}


def attach_calibration(state,validation,method=None):
    if state.calibration: raise ValueError("calibration already attached")
    validate_residual_frame(validation)
    if validation.empty: raise ValueError("validation residuals must be nonempty")
    if _FORBIDDEN.search(" ".join(validation.run_id.astype(str).unique())): raise ValueError("validation calibration forbids attack provenance")
    scores=_score_matrix(_array(validation),state); methods=SCORE_METHODS if method is None else (method,)
    if any(m not in SCORE_METHODS for m in methods): raise ValueError("unknown calibration method")
    state.calibration={m:np.sort(scores[m]) for m in methods}
    state.metadata.update({"calibration_source":"validation_only","calibration_n":len(validation)})
    return state


def fit_shared_state(*args,**kwargs):
    """Deprecated spelling; distribution-only, never attaches Q_cal."""
    return fit_distribution(*args,**kwargs)


def conformal_pvalues(calibration,query):
    cal=np.sort(np.asarray(calibration,float)); q=np.asarray(query,float)
    if not len(cal) or not np.isfinite(cal).all() or not np.isfinite(q).all(): raise ValueError("finite nonempty calibration required")
    return (1+len(cal)-np.searchsorted(cal,q,side="left"))/(len(cal)+1.)


def mixture_evalues(pvalues,*,clip=1e-15):
    p=np.asarray(pvalues,float)
    if np.any((p<0)|(p>1)|~np.isfinite(p)): raise ValueError("p-values must be finite in [0,1]")
    pc=np.maximum(p,clip); logs=np.stack([np.log(k)+(k-1)*np.log(pc) for k in KAPPAS]); m=logs.max(0)
    loge=m+np.log(np.mean(np.exp(logs-m),axis=0))
    return {"e":np.exp(loge),"log_e":loge,"kappas":list(KAPPAS),"clipped_count":int(np.count_nonzero(pc!=p)),"clip":clip}


def score_residuals(frame,state,*,require_calibration=True):
    validate_residual_frame(frame); out=frame.copy(); scores=_score_matrix(_array(frame),state)
    for method,q in scores.items():
        out[f"q_{method}"]=q
        if method in state.calibration:
            p=conformal_pvalues(state.calibration[method],q); mix=mixture_evalues(p)
            out[f"p_{method}"]=p; out[f"e_{method}"]=mix["e"]
        elif require_calibration: raise ValueError("state has no frozen validation Q_cal")
    if "full_shrinkage_mahalanobis" in state.calibration:
        out["q"]=out.q_full_shrinkage_mahalanobis; out["p"]=out.p_full_shrinkage_mahalanobis; out["e"]=out.e_full_shrinkage_mahalanobis
    return out


def aggregate_epochs(scored):
    cols=["recording_id","run_id","window_bin_s","availability_time_s","N","mean_e","mean_neg_log_p","min_p","median_p","max_e","top25_mean_e"]
    if scored.empty: out=pd.DataFrame(columns=cols); out.attrs["contract"]="skip_epoch; summary metrics NaN"; return out
    key="window_bin_s" if "window_bin_s" in scored else "window_end_s"; rows=[]
    identity="recording_id" if "recording_id" in scored else "run_id"
    if scored.duplicated([identity,key,"prn"]).any():
        raise ValueError("duplicate PRN in physical recording epoch")
    for (run,t),g in scored.groupby([identity,key],sort=True):
        e=np.sort(g.e.to_numpy(float)); p=g.p.to_numpy(float); top=max(1,int(np.ceil(.25*len(g))))
        rows.append({"recording_id":str(run),"run_id":str(run),"window_bin_s":float(t),"availability_time_s":float(g.window_end_s.max()),"N":len(g),
          "mean_e":float(e.mean()),"mean_neg_log_p":float(np.mean(-np.log(np.maximum(p,1e-300)))),"min_p":float(p.min()),
          "median_p":float(np.median(p)),"max_e":float(e.max()),"top25_mean_e":float(e[-top:].mean())})
    return pd.DataFrame(rows,columns=cols).sort_values(["run_id","window_bin_s"],kind="mergesort").reset_index(drop=True)


def baseline_epoch_scores(scored):
    rows=[]
    identity="recording_id" if "recording_id" in scored else "run_id"
    for (run,t),g in scored.groupby([identity,"window_bin_s"],sort=True):
        rows.append({"recording_id":run,"run_id":run,"window_bin_s":t,"availability_time_s":float(g.window_end_s.max()),
          "A0":float(g.b0_prn_node_rmse.max()),"A2":float(np.mean(-np.log(np.maximum(g.p,1e-300)))),"A4":float(g.e.mean())})
    return pd.DataFrame(rows)


def sequential_scores(log_e,run_ids,*,drift=0.,horizon=4096,capital_cap=1e300):
    values=np.asarray(log_e,float); runs=np.asarray(run_ids).astype(str)
    if len(values)!=len(runs) or drift<0 or horizon<1 or not np.isfinite(values).all(): raise ValueError("invalid sequential input")
    if not np.isfinite(capital_cap) or capital_cap<=0: raise ValueError("capital_cap must be positive and finite")
    output=[]; previous=None; log_capitals=[]; step=0; g=0.; resets=0; clipped=0
    log2=float(np.log(2.)); log_cap=float(np.log(capital_cap))
    for value,run in zip(values,runs):
        if run!=previous: log_capitals=[]; step=0; g=0.; previous=run; resets+=1
        if step>=horizon: raise ValueError("S1 fixed prior horizon exhausted")
        log_weight=-(step+1)*log2
        log_capitals=[c+float(value) for c in log_capitals]+[log_weight+float(value)]
        # Unallocated fixed-prior tail is exactly 2^-(step+1).
        log_total=float(np.logaddexp.reduce(np.asarray([log_weight,*log_capitals],float)))
        if log_total>log_cap: clipped+=1
        capital=float(np.exp(min(log_total,log_cap)))
        reserve=float(np.exp(max(log_weight,np.log(np.nextafter(0.,1.)))))
        g=float(np.clip(max(0.,g+float(value)-drift),0.,capital_cap))
        output.append({"s1_capital":capital,"s1_log_capital":log_total,"s1_total_fund":1.,"s1_reserve":reserve,"s2_e_cusum":g})
        step+=1
    frame=pd.DataFrame(output); frame.attrs.update({"capital_cap":float(capital_cap),"capital_clipped_count":clipped,"reset_count":resets,"primary":"s1_log_capital"}); return frame


def epoch_masks(times,*,onset_s=100.):
    t=np.asarray(times,float); return {"stable":(t>=onset_s-70)&(t<onset_s-10),"transition":(t>=onset_s-10)&(t<onset_s+10),"established":t>=onset_s+10}
def label_epochs(times,*,onset_s=100.):
    m=epoch_masks(times,onset_s=onset_s); out=np.full(len(np.asarray(times)),"outside",object)
    for label in ("stable","transition","established"): out[m[label]]=label
    return out


def _payload(s): return {"schema":"gnss-doppler-lab.cmte-v2","mean":s.mean.tolist(),"covariance":s.covariance.tolist(),"diagonal_scales":s.diagonal_scales.tolist(),"calibration":{k:v.tolist() for k,v in s.calibration.items()},"epsilon":s.epsilon,"checkpoint_sha256":s.checkpoint_sha256,"metadata":s.metadata}
def save_state(state,path):
    doc=_payload(state); canonical=json.dumps(doc,sort_keys=True,separators=(",",":")); doc["state_sha256"]=hashlib.sha256(canonical.encode()).hexdigest(); Path(path).write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
def load_state(path,*,expected_checkpoint_sha256=""):
    doc=json.loads(Path(path).read_text()); checksum=doc.pop("state_sha256",None); canonical=json.dumps(doc,sort_keys=True,separators=(",",":"))
    if checksum!=hashlib.sha256(canonical.encode()).hexdigest(): raise ValueError("state hash mismatch")
    if expected_checkpoint_sha256 and doc["checkpoint_sha256"].lower()!=expected_checkpoint_sha256.lower(): raise ValueError("checkpoint hash mismatch")
    return FitState(np.asarray(doc["mean"]),np.asarray(doc["covariance"]),np.asarray(doc["diagonal_scales"]),{k:np.asarray(v) for k,v in doc["calibration"].items()},doc["epsilon"],doc["checkpoint_sha256"],doc["metadata"])
