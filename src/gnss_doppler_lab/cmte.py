"""Core CMTE detector primitives for frozen-B0 nine-tap innovations.

All fitting is shared across PRNs and normal cleanStatic data only.  Functions
are deterministic, causal where sequential, and intentionally contain no file
search or attack-dependent tuning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

TAP_ORDER = ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")
RESIDUAL_COLUMNS = tuple(f"residual_{i:03d}" for i in range(9))
SCORE_METHODS = ("rmse", "diag_mahalanobis", "full_shrinkage_mahalanobis", "max_standardized_tap")
KAPPAS = (.25, .5, .75)
EPSILON = 1e-8


def _array(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, RESIDUAL_COLUMNS].to_numpy(float)


def validate_residual_frame(frame: pd.DataFrame, *, require_history_reset: bool = False) -> dict:
    required = {"run_id", "prn", "window_start_s", "window_end_s", "window_mid_s", *RESIDUAL_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"residual input missing columns: {missing}")
    extras = sorted(c for c in frame if c.startswith("residual_") and c not in RESIDUAL_COLUMNS)
    if extras:
        raise ValueError(f"exactly nine ordered residual columns required; extras={extras}")
    numeric = frame[["window_start_s", "window_end_s", "window_mid_s", *RESIDUAL_COLUMNS]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("timing and residual values must be finite")
    if np.any(frame.window_end_s.to_numpy(float) <= frame.window_start_s.to_numpy(float)):
        raise ValueError("causal windows require end > start")
    if np.any(frame.window_mid_s.to_numpy(float) > frame.window_end_s.to_numpy(float)):
        raise ValueError("window midpoint cannot be in the future")
    if "b0_prn_node_rmse" in frame:
        expected = np.sqrt(np.mean(_array(frame) ** 2, axis=1))
        if not np.allclose(expected, frame.b0_prn_node_rmse.to_numpy(float), atol=1e-9, rtol=1e-6):
            raise ValueError("B0 residual RMSE must equal sqrt(mean(r^2))")
    if require_history_reset and "target_window_index" in frame:
        first = frame.sort_values(["run_id", "prn", "window_end_s"], kind="mergesort").groupby(["run_id", "prn"], sort=False).head(1)
        if not (first.target_window_index.to_numpy(int) == 12).all():
            raise ValueError("B0 history reset requires first target_window_index=12 per split/run/PRN")
    return {"feature_columns": list(RESIDUAL_COLUMNS), "prn_identity_feature": False,
            "tap_order": list(TAP_ORDER), "availability_field": "window_end_s"}


def audit_roles(roles: Mapping[str, pd.DataFrame]) -> dict:
    if set(roles) != {"train", "validation", "test"}:
        raise ValueError("roles must be exactly train, validation, test")
    seen: set[tuple] = set()
    sources: dict[str, list[str]] = {}
    for role in ("train", "validation", "test"):
        f = roles[role]
        validate_residual_frame(f)
        runs = sorted(f.run_id.astype(str).unique().tolist())
        sources[role] = runs
        if role in {"train", "validation"} and any(r.lower().startswith("ds") or "scenario" in r.lower() for r in runs):
            raise ValueError("normal-only fit/calibration forbids scenario ds* rows or attack prefixes")
        keys = set(zip(f.run_id.astype(str), f.prn.astype(str), f.window_end_s.astype(float)))
        if seen.intersection(keys):
            raise ValueError("train/validation/test roles are not disjoint")
        seen.update(keys)
    return {"disjoint": True, "fit_sources": sources["train"], "sources": sources,
            "normal_only_fit": True, "attack_prefix_fitted": False,
            "threshold_source": "validation_only", "test_used_for_calibration": False}


@dataclass
class SequentialState:
    """Explicit resettable state for online adapters."""
    run_id: str | None = None
    s1_suffix_log_capitals: list[float] = field(default_factory=list)
    s2_e_cusum: float = 0.0

    def reset(self, run_id: str | None = None) -> None:
        self.run_id = run_id
        self.s1_suffix_log_capitals.clear()
        self.s2_e_cusum = 0.0


@dataclass
class FitState:
    mean: np.ndarray
    covariance: np.ndarray
    diagonal_scales: np.ndarray
    calibration: dict[str, np.ndarray]
    epsilon: float = EPSILON
    checkpoint_sha256: str = ""
    metadata: dict = field(default_factory=dict)


def fit_shared_state(frame: pd.DataFrame, *, epsilon: float = EPSILON,
                     shrinkage: float | None = None, checkpoint_sha256: str = "") -> FitState:
    validate_residual_frame(frame)
    if frame.empty or epsilon <= 0:
        raise ValueError("normal train residuals must be nonempty and epsilon positive")
    if any(str(x).lower().startswith("ds") for x in frame.run_id.unique()):
        raise ValueError("normal-only fit forbids scenario rows")
    x = _array(frame); mu = x.mean(axis=0); centered = x - mu
    empirical = centered.T @ centered / max(1, len(x) - 1)
    # Deterministic diagonal shrinkage; stronger for small samples, never attack-selected.
    lam = min(1.0, 10.0 / max(10.0, float(len(x)))) if shrinkage is None else float(shrinkage)
    if not 0 <= lam <= 1:
        raise ValueError("shrinkage must be in [0,1]")
    diag = np.maximum(np.diag(empirical), epsilon)
    cov = (1-lam)*empirical + lam*np.diag(diag) + epsilon*np.eye(9)
    scales = np.sqrt(np.maximum(np.diag(cov), epsilon))
    temp = FitState(mu, cov, scales, {}, epsilon, checkpoint_sha256,
                    {"fit_scope":"shared_no_prn_identity", "covariance":"deterministic diagonal shrinkage",
                     "shrinkage":lam, "raw_taps":"prompt-relative magnitudes",
                     "residuals":"signed standardized target-prediction", "tap_order":list(TAP_ORDER),
                     "seq_len":12, "default_method":"full_shrinkage_mahalanobis"})
    scores = _score_matrix(x, temp)
    temp.calibration = {name: np.sort(values) for name, values in scores.items()}
    return temp


def _score_matrix(x: np.ndarray, state: FitState) -> dict[str, np.ndarray]:
    d=x-state.mean; inv=np.linalg.inv(state.covariance)
    return {"rmse":np.sqrt(np.mean(x*x,axis=1)),
            "diag_mahalanobis":np.sqrt(np.mean((d/state.diagonal_scales)**2,axis=1)),
            "full_shrinkage_mahalanobis":np.sqrt(np.maximum(0,np.einsum("ni,ij,nj->n",d,inv,d))),
            "max_standardized_tap":np.max(np.abs(d/state.diagonal_scales),axis=1)}


def conformal_pvalues(calibration: Sequence[float], query: Sequence[float]) -> np.ndarray:
    cal=np.sort(np.asarray(calibration,float)); q=np.asarray(query,float)
    if cal.size == 0 or not np.isfinite(cal).all() or not np.isfinite(q).all():
        raise ValueError("conformal values must be finite and calibration nonempty")
    # inclusive >= ties: first index with cal >= q.
    count=cal.size-np.searchsorted(cal,q,side="left")
    return (1.0+count)/(cal.size+1.0)


def mixture_evalues(pvalues: Sequence[float], *, clip: float = 1e-15) -> dict:
    p=np.asarray(pvalues,float)
    if np.any((p<0)|(p>1)|~np.isfinite(p)):
        raise ValueError("p-values must be finite in [0,1]")
    pc=np.maximum(p,clip); logs=np.stack([np.log(k)+(k-1)*np.log(pc) for k in KAPPAS])
    m=logs.max(axis=0); loge=m+np.log(np.mean(np.exp(logs-m),axis=0))
    return {"e":np.exp(loge), "log_e":loge, "kappas":list(KAPPAS),
            "clipped_count":int(np.count_nonzero(pc!=p)), "clip":clip}


def score_residuals(frame: pd.DataFrame, state: FitState) -> pd.DataFrame:
    validate_residual_frame(frame); out=frame.copy(); scores=_score_matrix(_array(frame),state)
    for method, q in scores.items():
        p=conformal_pvalues(state.calibration[method],q); mix=mixture_evalues(p)
        out[f"q_{method}"]=q; out[f"p_{method}"]=p; out[f"e_{method}"]=mix["e"]
    out["q"]=out.q_full_shrinkage_mahalanobis
    out["p"]=out.p_full_shrinkage_mahalanobis
    out["e"]=out.e_full_shrinkage_mahalanobis
    return out


def aggregate_epochs(scored: pd.DataFrame) -> pd.DataFrame:
    columns=["run_id","window_bin_s","availability_time_s","N","mean_e","min_p","median_p","max_e","top25_mean_e"]
    if scored.empty:
        out=pd.DataFrame(columns=columns); out.attrs["contract"]="skip_epoch; summary metrics NaN"; return out
    key="window_bin_s" if "window_bin_s" in scored else "window_end_s"
    rows=[]
    for (run,t),g in scored.groupby(["run_id",key],sort=True):
        e=np.sort(g.e.to_numpy(float)); p=g.p.to_numpy(float); n=len(g); top=max(1,int(np.ceil(.25*n)))
        rows.append({"run_id":str(run),"window_bin_s":float(t),"availability_time_s":float(g.window_end_s.max()),
                     "N":n,"mean_e":float(e.mean()),"min_p":float(p.min()),"median_p":float(np.median(p)),
                     "max_e":float(e.max()),"top25_mean_e":float(e[-top:].mean())})
    return pd.DataFrame(rows,columns=columns).sort_values(["run_id","window_bin_s"],kind="mergesort").reset_index(drop=True)


def sequential_scores(log_e: Sequence[float], run_ids: Sequence[str], *, drift: float = 0.0) -> pd.DataFrame:
    values=np.asarray(log_e,float); runs=np.asarray(run_ids).astype(str)
    if len(values)!=len(runs) or drift<0 or not np.isfinite(values).all():
        raise ValueError("invalid sequential input")
    s1=[]; s2=[]; previous=None; suffix=[]; g=0.
    for value,run in zip(values,runs):
        if run!=previous: suffix=[]; g=0.; previous=run
        suffix=[x+value for x in suffix]+[value]
        # Equal mixture over no-bet and every restart time, floored at unit capital.
        a=np.asarray([0.,*suffix]); m=a.max(); logcap=float(m+np.log(np.exp(a-m).mean()))
        s1.append(max(0.,logcap)); g=max(0.,g+float(value)-drift); s2.append(g)
    return pd.DataFrame({"s1_log_capital":s1,"s2_e_cusum":s2})


def epoch_masks(times: Sequence[float], *, onset_s: float = 100.) -> dict[str,np.ndarray]:
    t=np.asarray(times,float)
    return {"stable":(t>=onset_s-70)&(t<onset_s-10),
            "transition":(t>=onset_s-10)&(t<onset_s+10),
            "established":t>=onset_s+10}


def label_epochs(times: Sequence[float], *, onset_s: float = 100.) -> np.ndarray:
    m=epoch_masks(times,onset_s=onset_s); out=np.full(len(np.asarray(times)),"outside",dtype=object)
    for label in ("stable","transition","established"): out[m[label]]=label
    return out


def _state_payload(state: FitState) -> dict:
    return {"schema":"gnss-doppler-lab.cmte-v1","mean":state.mean.tolist(),"covariance":state.covariance.tolist(),
            "diagonal_scales":state.diagonal_scales.tolist(),"calibration":{k:v.tolist() for k,v in state.calibration.items()},
            "epsilon":state.epsilon,"checkpoint_sha256":state.checkpoint_sha256,"metadata":state.metadata}


def save_state(state: FitState, path: str|Path) -> None:
    payload=_state_payload(state); canonical=json.dumps(payload,sort_keys=True,separators=(",",":"))
    payload["state_sha256"]=hashlib.sha256(canonical.encode()).hexdigest()
    Path(path).write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n")


def load_state(path: str|Path, *, expected_checkpoint_sha256: str="") -> FitState:
    doc=json.loads(Path(path).read_text()); checksum=doc.pop("state_sha256",None)
    canonical=json.dumps(doc,sort_keys=True,separators=(",",":"))
    if checksum!=hashlib.sha256(canonical.encode()).hexdigest(): raise ValueError("state hash mismatch")
    if expected_checkpoint_sha256 and doc["checkpoint_sha256"].lower()!=expected_checkpoint_sha256.lower():
        raise ValueError("checkpoint hash mismatch")
    return FitState(np.asarray(doc["mean"]),np.asarray(doc["covariance"]),np.asarray(doc["diagonal_scales"]),
                    {k:np.asarray(v) for k,v in doc["calibration"].items()},doc["epsilon"],doc["checkpoint_sha256"],doc["metadata"])
