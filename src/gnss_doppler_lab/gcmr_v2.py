"""GCMR v2 PRN-preserving scoring primitives."""
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
import numpy as np
SCHEMA="gcmr-clean-v2"; TEMPERATURE=1.0
class InsufficientSupport(ValueError): pass

def pair_errors(reconstruction,target,observation_mask,pair_mask=None,*,observation_scale=None):
 r=np.asarray(reconstruction,float); y=np.asarray(target,float); m=np.asarray(observation_mask,bool)
 if r.shape!=y.shape or m.shape!=y.shape or y.ndim!=2: raise ValueError("pair arrays must have matching [P,F] shapes")
 real=np.ones(len(y),bool) if pair_mask is None else np.asarray(pair_mask,bool)
 if real.shape!=(len(y),): raise ValueError("pair_mask must have shape [P]")
 scale=np.ones(y.shape[1]) if observation_scale is None else np.asarray(observation_scale,float)
 if scale.shape!=(y.shape[1],) or not np.isfinite(scale).all() or np.any(scale<=0): raise ValueError("invalid observation_scale")
 out=[]
 for k in np.flatnonzero(real):
  valid=m[k]
  if not valid.any(): raise ValueError("real pair has zero valid coordinates")
  d=(r[k,valid]-y[k,valid])/scale[valid]
  if not np.isfinite(d).all(): raise ValueError("nonfinite valid residual")
  out.append(float(np.mean(d*d)))
 return np.asarray(out)

def node_raw(pair_prns,errors):
 pairs=np.asarray(pair_prns); e=np.asarray(errors,float)
 if pairs.ndim!=2 or pairs.shape[1]!=2 or len(pairs)!=len(e): raise ValueError("pair identity/errors mismatch")
 canon=np.sort(pairs.astype(int),axis=1)
 if not np.isfinite(e).all() or np.any(canon[:,0]==canon[:,1]) or len({tuple(x) for x in canon})!=len(canon): raise InsufficientSupport("insufficient_support")
 prns=np.unique(canon)
 if len(prns)<4: raise InsufficientSupport("insufficient_support")
 expected={(int(prns[i]),int(prns[j])) for i in range(len(prns)) for j in range(i+1,len(prns))}
 if {tuple(x) for x in canon}!=expected: raise InsufficientSupport("insufficient_support: incomplete graph")
 return prns,np.asarray([np.median(e[np.any(canon==p,axis=1)]) for p in prns],float)

@dataclass(frozen=True)
class NodeNormalizer:
 center:float; scale:float
 @classmethod
 def fit(cls,groups):
  x=np.concatenate([np.asarray(v,float).reshape(-1) for v in groups])
  if not len(x) or not np.isfinite(x).all(): raise ValueError("invalid clean reference nodes")
  c=float(np.median(x)); return cls(c,max(float(1.4826*np.median(np.abs(x-c))),1e-9))
 def transform(self,x): return (np.asarray(x,float)-self.center)/self.scale

def linear_q99(values): return float(np.quantile(np.asarray(values,float),.99,method="linear"))
@dataclass(frozen=True)
class EventScore:
 prns:np.ndarray; raw:np.ndarray; z:np.ndarray; activation:np.ndarray
 single_fault_score:float; candidate_prn:int; single_alarm:bool
 multi_prn_score:float; hard_support:int; multi_alarm:bool; classification:str; diffuse_support:bool

def score_nodes(prns,raw,normalizer,tau_prn,multi_threshold):
 p=np.asarray(prns,int); a=np.asarray(raw,float)
 if p.ndim!=1 or a.shape!=p.shape or len(p)<4 or len(np.unique(p))!=len(p): raise InsufficientSupport("insufficient_support")
 order=np.argsort(p); p=p[order]; a=a[order]; z=normalizer.transform(a)
 act=1/(1+np.exp(-np.clip(z-float(tau_prn),-709,709)))
 single=float(np.max(z)); candidate=int(p[np.flatnonzero(z==single)[0]])
 R=float(np.sum(act)/len(p)); K=int(np.sum(z>tau_prn)); sa=bool(single>tau_prn); ma=bool(R>multi_threshold and K>=2)
 return EventScore(p,a,z,act,single,candidate,sa,R,K,ma,"multi" if ma else "single" if sa else "none",bool(R>multi_threshold and K<2))

def calibrate(clean_reference_nodes,calibration_nodes):
 norm=NodeNormalizer.fit(clean_reference_nodes); tau=linear_q99(np.concatenate([norm.transform(x) for x in clean_reference_nodes])); Rs=[]
 for x in calibration_nodes:
  z=norm.transform(x); Rs.append(float(np.mean(1/(1+np.exp(-np.clip(z-tau,-709,709))))))
 return norm,tau,TEMPERATURE,linear_q99(Rs)
def canonical_json_hash(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def save_checkpoint(path,*,payload,hashes):
 if set(hashes)!={"implementation","source","cache","role","config"}: raise ValueError("exact hash set required")
 doc={"schema":SCHEMA,"payload":payload,"hashes":hashes}; doc["integrity_sha256"]=canonical_json_hash(doc); Path(path).write_text(json.dumps(doc,sort_keys=True,separators=(",",":")))
def load_checkpoint(path,*,expected_hashes=None):
 doc=json.loads(Path(path).read_text()); integrity=doc.pop("integrity_sha256",None)
 if doc.get("schema")!=SCHEMA or integrity!=canonical_json_hash(doc): raise ValueError("checkpoint tamper/schema rejection")
 if expected_hashes is not None and doc.get("hashes")!=expected_hashes: raise ValueError("checkpoint provenance hash mismatch")
 return doc


# Real-model adapter and authenticated native checkpoint support.
def model_pair_errors(model, events, *, device=None):
 """Run the actual padded model batch and return pair identities/errors per event."""
 import torch
 from gnss_doppler_lab.gcmr_model import collate_gcmr_events
 events=list(events)
 if not events: return []
 dev=torch.device(device) if device else next(model.parameters()).device
 batch=collate_gcmr_events(events,device=dev); model.eval()
 with torch.no_grad(): reconstruction,_=model(**batch)
 scale=model.scaler.observation_scale.detach().cpu().numpy()
 rec=reconstruction.detach().cpu().numpy(); obs=batch["observations"].cpu().numpy()
 masks=batch["observation_mask"].cpu().numpy(); real=batch["pair_mask"].cpu().numpy()
 out=[]
 for i,event in enumerate(events):
  n=int(real[i].sum()); pairs=np.asarray(event.pair_prns,int)
  if pairs.shape!=(n,2): raise RuntimeError("model/output pair identity mismatch")
  errors=pair_errors(rec[i],obs[i],masks[i],real[i],observation_scale=scale)
  if len(errors)!=len(pairs): raise RuntimeError("model/output pair count mismatch")
  prns,raw=node_raw(pairs,errors)
  input_prns=np.unique(pairs)
  if not np.array_equal(prns,input_prns): raise RuntimeError("input/output PRN-set mismatch")
  out.append({"event":event,"pair_prns":pairs.copy(),"pair_errors":errors,"prns":prns,"raw":raw})
 return out

def score_model_events(model,events,normalizer,tau_prn,multi_threshold,*,device=None):
 rows=[]
 for x in model_pair_errors(model,events,device=device):
  score=score_nodes(x["prns"],x["raw"],normalizer,tau_prn,multi_threshold)
  if not np.array_equal(score.prns,np.unique(x["event"].pair_prns)): raise RuntimeError("input/output PRN-set mismatch")
  rows.append({**x,"score":score})
 return rows

def save_native_checkpoint(path,*,training,normalizer,tau_prn,multi_threshold,provenance):
 import os,torch
 from gnss_doppler_lab.gcmr_relations import OBSERVATION_FEATURES,CONDITION_FEATURES
 payload={"format":SCHEMA,"schema_version":2,"model_state":{k:v.detach().cpu() for k,v in training.model.state_dict().items()},
  "config":dict(training.config),"best_epoch":int(training.best_epoch),"node_center":float(normalizer.center),
  "node_scale":float(normalizer.scale),"tau_prn":float(tau_prn),"temperature":TEMPERATURE,
  "multi_threshold":float(multi_threshold),"feature_contract":{"observation":list(OBSERVATION_FEATURES),"condition":list(CONDITION_FEATURES)},
  "scoring_contract":{"pair":"masked standardized MSE","node":"median incident pair errors","single":"max(z) > tau_prn","multi":"mean(sigmoid(z-tau_prn)) > multi_threshold and K >= 2","priority":"multi/single/none","prn_filtering":False},
  "provenance":provenance}
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+".tmp");torch.save(payload,tmp);os.replace(tmp,p)

def load_native_checkpoint(path,*,expected_provenance=None,expected_sha256=None,device="cpu"):
 import torch
 from gnss_doppler_lab.gcmr_model import GcmrNet
 from gnss_doppler_lab.gcmr_relations import OBSERVATION_FEATURES,CONDITION_FEATURES
 if expected_sha256 is not None:
  import hashlib
  h=hashlib.sha256(Path(path).read_bytes()).hexdigest()
  if h!=expected_sha256: raise ValueError("checkpoint SHA256 mismatch/tamper rejection")
 try: p=torch.load(path,map_location="cpu",weights_only=True)
 except Exception as e: raise ValueError(f"invalid GCMR v2 checkpoint: {e}") from e
 if not isinstance(p,dict) or p.get("format")!=SCHEMA or p.get("schema_version")!=2: raise ValueError("checkpoint schema mismatch")
 if p.get("feature_contract")!={"observation":list(OBSERVATION_FEATURES),"condition":list(CONDITION_FEATURES)}: raise ValueError("feature contract mismatch")
 if expected_provenance is not None and p.get("provenance")!=expected_provenance: raise ValueError("checkpoint provenance mismatch")
 c=p.get("config",{}); required=("pair_hidden","event_hidden","latent_dim")
 if any(k not in c for k in required): raise ValueError("checkpoint architecture missing")
 model=GcmrNet(**{k:int(c[k]) for k in required})
 try:model.load_state_dict(p["model_state"],strict=True)
 except Exception as e:raise ValueError(f"checkpoint model mismatch: {e}") from e
 model.to(torch.device(device)).eval()
 vals=[p.get(k) for k in ("node_center","node_scale","tau_prn","temperature","multi_threshold")]
 if not all(np.isfinite(vals)) or vals[1]<=0 or vals[3]!=1.0:raise ValueError("invalid fitted scoring state")
 return model,NodeNormalizer(float(vals[0]),float(vals[1])),float(vals[2]),float(vals[4]),p
