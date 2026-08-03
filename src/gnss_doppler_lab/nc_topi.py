"""NC-TOPI Stage-0 preregistered mathematical primitives.

No experiment runner or attack scoring lives here. Estimators reject non-clean
fitting roles at their API boundary.
"""
from __future__ import annotations
from dataclasses import dataclass
import json, math
from pathlib import Path
from typing import Callable, Mapping, Sequence
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import roc_auc_score

CANONICAL_TAP_COORDS=np.array([-.5,-.375,-.25,-.125,0.,.125,.25,.375,.5])
SECOND_PEAK_POWERS=(.05,.1,.2,.4,.8)
SECOND_PEAK_SEPARATIONS=(.0625,.125,.25,.375,.5)
NONIDENTIFIABILITY_MARKER="legacy residual-only tangent is non-identifiable without exact actual and predicted peaks"
DEFAULT_SEED=20260803

def _array(value,name,ndim=None):
 try: out=np.asarray(value,dtype=float)
 except (TypeError,ValueError) as exc: raise ValueError(f"{name} must be numeric and finite") from exc
 if ndim is not None and out.ndim!=ndim: raise ValueError(f"{name} must be {ndim}-dimensional")
 if not np.isfinite(out).all(): raise ValueError(f"{name} must be finite")
 return out

def _default_config_path(): return Path(__file__).resolve().parents[2]/"configs"/"nc_topi_stage0.json"

def validate_config(config:Mapping[str,object])->None:
 if config.get("schema")!="gnss-doppler-lab.nc-topi-stage0.v1": raise ValueError("unexpected NC-TOPI Stage-0 schema")
 try: taps=config["taps"]; coords=taps["coordinates_chips"]; b0=config["b0"]; split=config["split"]; decision=config["decision"]
 except (KeyError,TypeError) as exc: raise ValueError("incomplete NC-TOPI Stage-0 config") from exc
 if list(coords)!=CANONICAL_TAP_COORDS.tolist(): raise ValueError("tap coordinates must be the explicit canonical tap coordinates")
 if "GNSS-SDR" not in str(taps.get("coordinate_provenance","")): raise ValueError("tap coordinate provenance must name GNSS-SDR")
 if b0.get("checkpoint_sha256")!="f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b": raise ValueError("frozen B0 checkpoint hash changed")
 if b0.get("feature_order")!=["E4","E3","E2","E","P","L","L2","L3","L4"]: raise ValueError("frozen B0 feature order changed")
 expected={"train":{"source_end_lte":300},"calibration":{"source_start_gte":320,"source_end_lte":400},"holdout":{"source_start_gte":420}}
 if split!=expected: raise ValueError("source-support split contract changed")
 if decision.get("go_primary")!="q99 NC-TOPI median only": raise ValueError("primary GO detector changed")

def load_config(path=None):
 with Path(path or _default_config_path()).open(encoding="utf-8") as f: config=json.load(f)
 validate_config(config); return config

@dataclass(frozen=True)
class TangentBasis:
 matrix:np.ndarray; raw:np.ndarray; names:tuple[str,...]; metadata:dict[str,object]

def normalize_tangents(predicted_peak,coords,W=None,*,include_width=True,input_kind="regenerated_predicted_peak",low_signal_epsilon=1e-12):
 if input_kind!="regenerated_predicted_peak": raise ValueError(NONIDENTIFIABILITY_MARKER)
 p,x=_array(predicted_peak,"predicted_peak",1),_array(coords,"coords",1)
 if p.shape!=x.shape or p.size<3 or np.any(np.diff(x)<=0): raise ValueError("predicted_peak and strictly increasing physical coords must match")
 if np.linalg.norm(p)<=low_signal_epsilon: raise ValueError("low-signal predicted peak cannot define tangents")
 first=np.gradient(p,x,edge_order=2); cols=[p,first]; names=["amplitude","shift"]
 if include_width: cols.append(np.gradient(first,x,edge_order=2)); names.append("width")
 raw=np.column_stack(cols); weight=np.eye(p.size) if W is None else _array(W,"W",2)
 if weight.shape!=(p.size,p.size): raise ValueError("W dimensions must match peak")
 weight=(weight+weight.T)/2; normalized=raw.copy()
 for j in range(raw.shape[1]):
  norm2=float(raw[:,j]@weight@raw[:,j])
  if not np.isfinite(norm2) or norm2<=low_signal_epsilon**2: raise ValueError(f"low-signal {names[j]} tangent")
  normalized[:,j]/=math.sqrt(norm2)
 metadata={"input_kind":input_kind,"derivative_coordinates":"explicit physical chip coordinates","normalization":"each column has unit W norm","normalization_caveat":"B0 inputs are prompt-relative magnitude ratios (epsilon 1e-6); global amplitude is attenuated and amplitude is an ablation nuisance direction","residual_only_allowed":False,"nonidentifiability_marker":NONIDENTIFIABILITY_MARKER}
 return TangentBasis(normalized,raw,tuple(names),metadata)

@dataclass(frozen=True)
class CovarianceFit:
 Sigma:np.ndarray; W:np.ndarray; Sigma_unfloored:np.ndarray; audit:dict[str,object]

def assert_fit_is_clean_only(roles,*,allowed=("clean_train","clean_calibration")):
 role_list=[str(x) for x in roles]; bad=sorted(set(role_list)-set(allowed))
 if bad: raise ValueError(f"attack data cannot be used for fit; clean-only roles required, got {bad}")
 if not role_list: raise ValueError("clean-only fit roles cannot be empty")

def fit_shrinkage_covariance(residuals,*,fit_roles=None,floor_relative=1e-8,pinv_rcond=1e-10):
 r=_array(residuals,"residuals",2)
 if r.shape[0]<2 or r.shape[1]<1: raise ValueError("covariance requires at least two rows and one dimension")
 roles=list(fit_roles) if fit_roles is not None else ["clean_train"]*len(r)
 if len(roles)!=len(r): raise ValueError("fit_roles length must match residual rows")
 assert_fit_is_clean_only(roles,allowed=("clean_train",))
 unf=np.asarray(LedoitWolf().fit(r).covariance_); unf=(unf+unf.T)/2
 nominal=float(floor_relative*np.trace(unf)/r.shape[1]); floor=max(nominal,np.finfo(float).eps)
 values,vectors=np.linalg.eigh(unf); sigma=(vectors*np.maximum(values,floor))@vectors.T; sigma=(sigma+sigma.T)/2
 weight=np.linalg.pinv(sigma,rcond=pinv_rcond,hermitian=True)
 audit={"estimator":"sklearn.covariance.LedoitWolf","fit_role":"clean_train residual_raw only","rows":len(r),"dimension":r.shape[1],"floor_relative":floor_relative,"floor_epsilon":floor,"nominal_floor_epsilon":nominal,"pinv_rcond":pinv_rcond}
 return CovarianceFit(sigma,weight,unf,audit)

@dataclass(frozen=True)
class ProjectionResult:
 coefficients:np.ndarray; fitted:np.ndarray; r_perp:np.ndarray; total_energy:float; tangent_energy:float; perp_energy:float; rank:int; normal_rank:int; condition:float; ridge:float

def weighted_project(residual,J,W,*,lambda_relative=1e-8,pinv_rcond=1e-10):
 r,basis,weight=_array(residual,"residual",1),_array(J,"J",2),_array(W,"W",2)
 if basis.shape[0]!=r.size or weight.shape!=(r.size,r.size): raise ValueError("residual, J, and W dimensions do not match")
 if lambda_relative<0 or pinv_rcond<=0: raise ValueError("projection regularization parameters invalid")
 weight=(weight+weight.T)/2; normal=basis.T@weight@basis; scale=float(np.trace(normal)/max(1,normal.shape[0])); ridge=float(lambda_relative*scale) if scale>0 else float(lambda_relative)
 regularized=normal+ridge*np.eye(normal.shape[0]); coefficients=np.linalg.pinv(regularized,rcond=pinv_rcond,hermitian=True)@basis.T@weight@r
 fitted=basis@coefficients; perp=r-fitted; energy=lambda v:max(0.,float(v@weight@v)); singular=np.linalg.svd(normal,compute_uv=False)
 condition=float(np.inf if singular.size==0 or singular[-1]==0 else singular[0]/singular[-1])
 return ProjectionResult(coefficients,fitted,perp,energy(r),energy(fitted),energy(perp),int(np.linalg.matrix_rank(basis)),int(np.linalg.matrix_rank(normal)),condition,ridge)

def b0_rmse(standardized_residual):
 v=_array(standardized_residual,"standardized_residual")
 if v.ndim==1:return float(np.sqrt(np.mean(v**2)))
 if v.ndim==2:return np.sqrt(np.mean(v**2,axis=1))
 raise ValueError("standardized_residual must be a vector or matrix")

@dataclass(frozen=True)
class AggregateResult:
 score:float; count:int; selected_count:int; ids:tuple[str,...]; method:str

def aggregate_prn_scores(prn_ids,scores,method="median",*,valid_mask=None):
 ids=np.asarray(prn_ids,dtype=str); values=np.asarray(scores,dtype=float)
 if ids.ndim!=1 or values.ndim!=1 or len(ids)!=len(values) or len(ids)==0: raise ValueError("PRN IDs and scores must be nonempty matching vectors")
 mask=np.ones(len(values),bool) if valid_mask is None else np.asarray(valid_mask,bool)
 if mask.shape!=values.shape or not mask.any(): raise ValueError("valid mask must match and retain at least one PRN")
 if not np.isfinite(values[mask]).all(): raise ValueError("active PRN scores must be finite")
 if len(set(ids[mask]))!=int(mask.sum()) or any(not x.strip() for x in ids[mask]): raise ValueError("active PRN IDs must be unique and nonempty")
 active=values[mask]
 if method=="median": score,selected=float(np.median(active)),len(active)
 elif method=="top25_mean": selected=int(math.ceil(.25*len(active))); score=float(np.mean(np.sort(active)[-selected:]))
 else: raise ValueError("aggregator must be median or top25_mean")
 return AggregateResult(score,len(active),selected,tuple(sorted(ids[mask].tolist())),method)

def higher_quantile(scores,q,*,fit_roles):
 values=_array(scores,"calibration scores",1)
 if len(values)!=len(fit_roles) or not 0<q<1: raise ValueError("quantile inputs invalid")
 assert_fit_is_clean_only(fit_roles,allowed=("clean_calibration",))
 return float(np.quantile(values,q,method="higher"))

def strict_alarms(scores,threshold):
 values=_array(scores,"scores",1)
 if not np.isfinite(threshold): raise ValueError("threshold must be finite")
 return values>threshold

@dataclass(frozen=True)
class SplitMasks:
 train:np.ndarray; calibration:np.ndarray; holdout:np.ndarray; unassigned:np.ndarray

def source_support_split(source_start,source_end,*,scenario):
 start,end=_array(source_start,"source_start",1),_array(source_end,"source_end",1)
 if start.shape!=end.shape or np.any(end<start): raise ValueError("source support is invalid")
 if scenario!="cleanStatic": raise ValueError("attack scenario can never fit source-support splits")
 train=end<=300; calibration=(start>=320)&(end<=400); holdout=start>=420; stack=np.stack([train,calibration,holdout])
 if np.any(stack.sum(axis=0)>1): raise AssertionError("source-support split overlap")
 return SplitMasks(train,calibration,holdout,~stack.any(axis=0))

@dataclass(frozen=True)
class PhaseMasks:
 stable_pre:np.ndarray; transition:np.ndarray; post:np.ndarray; persistent:np.ndarray

def phase_masks(source_start,source_end,onset):
 start,end=_array(source_start,"source_start",1),_array(source_end,"source_end",1)
 if start.shape!=end.shape or np.any(end<start) or not np.isfinite(onset): raise ValueError("phase support is invalid")
 post=start>=onset; stable=(start>=30)&(end<=onset-20); transition=(~post)&(~stable); persistent=start>=onset+40
 if np.any(stable&transition) or np.any(transition&post) or np.any(stable&post): raise AssertionError("phase overlap")
 return PhaseMasks(stable,transition,post,persistent)

@dataclass(frozen=True)
class IQContext:
 contexts:np.ndarray; valid:np.ndarray; block_indices:tuple[np.ndarray,...]

def build_causal_iq_context(target_source_start,block_end,block_features,*,history=4,target_groups=None,block_groups=None):
 targets,ends,features=_array(target_source_start,"target_source_start",1),_array(block_end,"block_end",1),_array(block_features,"block_features",2)
 if len(ends)!=len(features) or history<1: raise ValueError("IQ blocks/history dimensions invalid")
 tg=np.asarray(target_groups if target_groups is not None else ["_"]*len(targets),dtype=str); bg=np.asarray(block_groups if block_groups is not None else ["_"]*len(ends),dtype=str)
 if len(tg)!=len(targets) or len(bg)!=len(ends): raise ValueError("IQ group vectors must match rows")
 contexts=np.full((len(targets),history,features.shape[1]),np.nan); valid=np.zeros(len(targets),bool); selected=[]
 for i,(target,group) in enumerate(zip(targets,tg)):
  eligible=np.flatnonzero((bg==group)&(ends<=target)); eligible=eligible[np.argsort(ends[eligible],kind="mergesort")]; chosen=eligible[-history:]; selected.append(chosen)
  if len(chosen)==history: contexts[i]=features[chosen]; valid[i]=True
  if len(chosen) and np.any(ends[chosen]>target): raise AssertionError("causal IQ context includes current overlap/future")
 return IQContext(contexts,valid,tuple(selected))

class RobustConditioner:
 """Clean-only robust-standardized fixed Huber scale model."""
 def __init__(self,*,lower_epsilon=1e-8): self.lower_epsilon=float(lower_epsilon)
 def fit(self,X,y,*,roles,feature_names=None):
  predictors,target=_array(X,"IQ predictors",2),_array(y,"scale target",1)
  if len(predictors)!=len(target) or len(roles)!=len(target) or np.any(target<=0): raise ValueError("conditioner fit dimensions/positive scale invalid")
  assert_fit_is_clean_only(roles,allowed=("clean_train",))
  names=[str(x).lower() for x in (feature_names or [f"x{i}" for i in range(predictors.shape[1])])]; forbidden={"prn","prn_id","scenario","scenario_id","onset","onset_s"}
  if forbidden.intersection(names): raise ValueError("forbidden PRN/scenario/onset identity feature")
  self.median_=np.median(predictors,axis=0); q75,q25=np.percentile(predictors,[75,25],axis=0); self.iqr_=q75-q25; self.iqr_[self.iqr_<=self.lower_epsilon]=1.
  self.model_=HuberRegressor(epsilon=1.35,alpha=1e-4,max_iter=1000).fit((predictors-self.median_)/self.iqr_,target)
  self.feature_names_=tuple(feature_names or [f"x{i}" for i in range(predictors.shape[1])]); self.fit_manifest_={"roles":["clean_train"],"rows":len(target),"PRN_feature":False,"scenario_feature":False,"onset_feature":False,"epsilon":1.35,"alpha":1e-4,"max_iter":1000}; self.cap_=None
  return self
 def _raw_predict(self,X):
  if not hasattr(self,"model_"): raise RuntimeError("conditioner is not fit")
  predictors=_array(X,"IQ predictors",2)
  if predictors.shape[1]!=len(self.median_): raise ValueError("IQ predictor dimension changed")
  return np.maximum(np.asarray(self.model_.predict((predictors-self.median_)/self.iqr_)),self.lower_epsilon)
 def calibrate_cap(self,X_calibration,*,roles,q=.995):
  values=self._raw_predict(X_calibration)
  if len(values)!=len(roles): raise ValueError("calibration roles length mismatch")
  assert_fit_is_clean_only(roles,allowed=("clean_calibration",)); self.cap_=float(np.quantile(values,q,method="higher")); return self.cap_
 def predict_scale(self,X):
  if getattr(self,"cap_",None) is None: raise RuntimeError("clean calibration cap is not set")
  return np.minimum(self._raw_predict(X),self.cap_)

def shuffled_control_target(target,*,roles,seed=DEFAULT_SEED):
 values=_array(target,"shuffle target",1)
 if len(values)!=len(roles): raise ValueError("shuffle roles length mismatch")
 assert_fit_is_clean_only(roles,allowed=("clean_train",)); return values[np.random.default_rng(seed).permutation(len(values))]

def standardized_pauc(labels,scores,*,max_fpr=.05):
 y=np.asarray(labels); s=_array(scores,"scores",1)
 if y.ndim!=1 or len(y)!=len(s) or set(np.unique(y))!={0,1}: raise ValueError("partial AUC requires binary labels with both classes")
 return float(roc_auc_score(y,s,max_fpr=max_fpr))

@dataclass(frozen=True)
class SustainedAlarm:
 delay:float; alarm_time:float; already_alarming_stable_pre:bool

def sustained_alarm_delay(availability_source_end,alarms,*,onset,required=3,cadence=.5,stable_pre_mask=None):
 times=_array(availability_source_end,"availability source_end",1); flags=np.asarray(alarms,bool)
 if flags.shape!=times.shape or required<1 or cadence<=0 or np.any(np.diff(times)<=0): raise ValueError("sustained alarm inputs invalid")
 pre=np.zeros(len(flags),bool) if stable_pre_mask is None else np.asarray(stable_pre_mask,bool)
 if pre.shape!=flags.shape: raise ValueError("stable_pre_mask length mismatch")
 already=bool(np.any(flags&pre)); run=0; alarm_time=math.inf; previous=None
 for time,flag in zip(times,flags):
  if time<onset: continue
  contiguous=previous is not None and abs(time-previous-cadence)<=1e-8
  run=run+1 if flag and contiguous else (1 if flag else 0); previous=time
  if run>=required: alarm_time=float(time); break
 return SustainedAlarm(float(alarm_time-onset) if np.isfinite(alarm_time) else math.inf,alarm_time,already)

def common_epoch_exact_join(identity_a,scores_a,identity_b,scores_b):
 va,vb=_array(scores_a,"scores_a",1),_array(scores_b,"scores_b",1); ka,kb=[tuple(x) for x in identity_a],[tuple(x) for x in identity_b]
 if len(ka)!=len(va) or len(kb)!=len(vb) or len(set(ka))!=len(ka) or len(set(kb))!=len(kb): raise ValueError("epoch identities must be unique and match scores")
 ma,mb=dict(zip(ka,va)),dict(zip(kb,vb)); common=sorted(set(ma).intersection(mb)); return common,np.asarray([ma[k] for k in common]),np.asarray([mb[k] for k in common])

@dataclass(frozen=True)
class BootstrapResult:
 point_estimate:float; ci:tuple[float,float]; replicates:np.ndarray; complete_block_count:int; block_epoch_count:int; audit:dict[str,object]

def paired_gap_safe_block_bootstrap(times,values_a,values_b,*,statistic=np.mean,block_seconds=10.,cadence=.5,reps=2000,seed=DEFAULT_SEED):
 t,a,b=(_array(v,n,1) for v,n in ((times,"times"),(values_a,"values_a"),(values_b,"values_b")))
 if not (len(t)==len(a)==len(b)) or len(t)==0 or reps<1: raise ValueError("paired bootstrap inputs invalid")
 order=np.argsort(t,kind="mergesort"); t,a,b=t[order],a[order],b[order]
 if np.any(np.diff(t)<=0): raise ValueError("times must be unique")
 epochs=int(round(block_seconds/cadence))
 if epochs<1 or not np.isclose(epochs*cadence,block_seconds): raise ValueError("block duration must be an integer cadence count")
 bounds=np.r_[0,np.flatnonzero(~np.isclose(np.diff(t),cadence,atol=1e-8,rtol=0))+1,len(t)]; blocks=[]
 for left,right in zip(bounds[:-1],bounds[1:]):
  for start in range(int(left),int(right)-epochs+1,epochs):
   ix=np.arange(start,start+epochs)
   if np.allclose(np.diff(t[ix]),cadence,atol=1e-8,rtol=0): blocks.append(ix)
 if not blocks: raise ValueError("no complete gap-safe 10s blocks; no IID fallback")
 rng=np.random.default_rng(seed); samples=np.empty(reps)
 for i in range(reps):
  ix=np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),len(blocks))]); samples[i]=float(statistic(a[ix])-statistic(b[ix]))
 point=float(statistic(a)-statistic(b)); ci=tuple(float(x) for x in np.percentile(samples,[2.5,97.5]))
 audit={"resampling":"paired gap-safe nonoverlapping complete 10s blocks","iid_fallback":False,"reps":reps,"seed":seed,"point_estimate_rows":"all eligible epochs","percentile":95}
 return BootstrapResult(point,ci,samples,len(blocks),epochs,audit)

def shift_peak(peak,coords,shift_chips):
 values,x=_array(peak,"peak",1),_array(coords,"coords",1)
 if values.shape!=x.shape or np.any(np.diff(x)<=0) or not np.isfinite(shift_chips): raise ValueError("physical peak/coordinate interpolation inputs invalid")
 return np.interp(x-shift_chips,x,values,left=0.,right=0.)

def second_peak_perturbation(peak,coords,relative_power,separation_chips,*,enforce_stage0_grid=True):
 if enforce_stage0_grid and (relative_power not in SECOND_PEAK_POWERS or separation_chips not in SECOND_PEAK_SEPARATIONS): raise ValueError("second peak must use the frozen Stage-0 physical grid")
 if relative_power<0: raise ValueError("relative power must be nonnegative")
 values=_array(peak,"peak",1); return values+math.sqrt(relative_power)*shift_peak(values,coords,separation_chips)

def equal_w_norm(vector,reference,W):
 v,ref,weight=_array(vector,"vector",1),_array(reference,"reference",1),_array(W,"W",2)
 if v.shape!=ref.shape or weight.shape!=(len(v),len(v)): raise ValueError("equal norm dimensions invalid")
 vn,rn=float(v@weight@v),float(ref@weight@ref)
 if vn<=0 or rn<0: raise ValueError("equal norm requires positive vector norm")
 return v*math.sqrt(rn/vn)

def w_orthogonal_vector(J,W,*,seed=DEFAULT_SEED):
 basis,weight=_array(J,"J",2),_array(W,"W",2)
 if weight.shape!=(basis.shape[0],basis.shape[0]): raise ValueError("orthogonal dimensions invalid")
 rng=np.random.default_rng(seed)
 for _ in range(100):
  candidate=rng.normal(size=basis.shape[0]); perp=weighted_project(candidate,basis,weight,lambda_relative=0).r_perp
  if float(perp@weight@perp)>1e-18:return perp
 raise ValueError("tangent span has no stable W-orthogonal complement")
