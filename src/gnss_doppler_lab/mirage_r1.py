"""Frozen MIRAGE R1 raw-recorrelation, CAF, design, and scoring primitives."""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc

from .mosaic_raw_recorrelation import correlate_nine_taps, read_ishort_complex_window
from .trace_native_1ms import complex_taps, read_records

DELAYS=np.arange(-4,5,dtype=np.float64)*.125
SCALES=(.020,.100,.500)
XI=np.arange(-2,3,dtype=np.float64)
EPSILON_RELATIVE=1e-12


def relative_complex_minors(caf:np.ndarray,epsilon_relative:float=EPSILON_RELATIVE)->np.ndarray:
 matrix=np.asarray(caf,np.complex128)
 if matrix.shape!=(9,5) or epsilon_relative<=0 or not np.isfinite(matrix).all():raise ValueError("finite 9x5 CAF required")
 energy=float(np.mean(np.abs(matrix)**2))
 if energy==0:return np.zeros((8,4),np.float64)
 x=matrix[:-1,:-1]*matrix[1:,1:];y=matrix[:-1,1:]*matrix[1:,:-1]
 value=np.abs(x-y)/np.sqrt(np.abs(x)**2+np.abs(y)**2+epsilon_relative*energy**2)
 if not np.isfinite(value).all():raise AssertionError("nonfinite minor")
 return value


def magnitude_minors(caf:np.ndarray)->np.ndarray:
 return relative_complex_minors(np.abs(np.asarray(caf,np.complex128)).astype(np.complex128))


def svd_ratio(caf:np.ndarray)->float:
 singular=np.linalg.svd(np.asarray(caf,np.complex128),compute_uv=False)
 return float(singular[1]**2/max(np.sum(singular**2),1e-300))


def multiscale_cafs(taps_1ms:np.ndarray)->dict[float,np.ndarray]:
 taps=np.asarray(taps_1ms,np.complex128)
 if taps.shape!=(500,9):raise ValueError("one Full epoch requires 500x9 complex taps")
 output={}
 for scale in SCALES:
  count=int(round(scale*1000));values=taps[-count:]
  t=(np.arange(count,dtype=np.float64)+.5)*.001
  phasor=np.exp(-2j*np.pi*np.outer(t,XI/scale))
  output[scale]=values.T@phasor
 return output


def reconstruct_taps(raw_path:Path,trace_path:Path,start_sample:int,epoch_count:int=500,
                     raw_transform=None)->tuple[np.ndarray,list[dict[str,object]]]:
 header,records=read_records(trace_path)
 starts=records["raw_interval_start_sample"].astype(np.int64)
 first=int(np.searchsorted(starts,int(start_sample)))
 if first>=len(records) or int(starts[first])!=int(start_sample):raise ValueError("start must be an actual TRACE endpoint")
 if first+epoch_count>len(records):raise ValueError("TRACE range truncated")
 taps=[];audit=[]
 for index in range(first,first+epoch_count):
  record=records[index];lo=int(record["raw_interval_start_sample"]);hi=int(record["raw_interval_end_sample"])
  iq=read_ishort_complex_window(raw_path,lo,hi-lo)
  if raw_transform is not None:iq=raw_transform(iq,lo,index-first)
  value=correlate_nine_taps(iq,prn=int(record["prn"]),action=record,tap_offsets_chips=header.tap_offsets_chips)
  taps.append(value);audit.append({"record_index":index,"raw_start":lo,"raw_end":hi})
 return np.asarray(taps),audit


def nearest_trace_endpoint(trace_path:Path,target_sample:int)->int:
 _,records=read_records(trace_path);starts=records["raw_interval_start_sample"].astype(np.int64)
 i=int(np.searchsorted(starts,target_sample));options=[x for x in (i-1,i) if 0<=x<len(starts)]
 return int(starts[min(options,key=lambda x:abs(int(starts[x])-target_sample))])


def native_alignment(raw_path:Path,trace_path:Path,indices:Iterable[int])->list[dict[str,object]]:
 from .mosaic_raw_recorrelation import evaluate_recorrelation
 header,records=read_records(trace_path);native=complex_taps(records)
 rows=[]
 for index in indices:
  record=records[int(index)];lo=int(record["raw_interval_start_sample"]);hi=int(record["raw_interval_end_sample"])
  iq=read_ishort_complex_window(raw_path,lo,hi-lo)
  taps=correlate_nine_taps(iq,prn=int(record["prn"]),action=record,tap_offsets_chips=header.tap_offsets_chips)
  result=evaluate_recorrelation(taps,native[int(index)],record,header.sample_rate_hz)
  rows.append({"record_index":int(index),"prn":int(record["prn"]),"raw_start_sample":lo,
   "complex_cosine":result.complex_cosine,"magnitude_spearman":result.magnitude_spearman,
   "center_error_chips":result.delay_center_error_chips,
   "native_peak_offset_chips":float(DELAYS[int(np.argmax(np.abs(native[int(index)])))]),
   "reconstructed_peak_offset_chips":float(DELAYS[int(np.argmax(np.abs(taps)))]),
   "gate_pass":bool(result.complex_cosine>=.995 and result.magnitude_spearman>=.99
                    and abs(result.delay_center_error_chips)<=.125)})
 return rows


def robust_reference(fields:list[np.ndarray],floor_relative:float=1e-6)->dict[str,np.ndarray]:
 values=np.asarray(fields,np.float64)**2
 location=np.median(values,axis=0);mad=1.4826*np.median(np.abs(values-location),axis=0)
 scale=np.maximum(mad,floor_relative*np.maximum(np.abs(location),1.0))
 statistics=np.mean(np.maximum((values-location)/scale,0),axis=(1,2))
 return {"location":location,"scale":scale,"train_statistics":statistics}


def scale_surprise(field:np.ndarray,reference:dict[str,np.ndarray])->float:
 statistic=float(np.mean(np.maximum((np.asarray(field)**2-reference["location"])/reference["scale"],0)))
 train=reference["train_statistics"]
 cdf=(1+np.count_nonzero(train<=statistic))/(len(train)+1)
 return float(-np.log(max(1-cdf,1/(len(train)+1))))


def epoch_features(taps:np.ndarray,references:dict[float,dict[str,np.ndarray]]|None=None)->dict[str,object]:
 cafs=multiscale_cafs(taps);minors={s:relative_complex_minors(c) for s,c in cafs.items()}
 result={"cafs":cafs,"minors":minors,"energy":float(sum(np.sum(np.abs(c)**2) for c in cafs.values())),
  "svd":{s:svd_ratio(c) for s,c in cafs.items()},"magnitude_minor":{s:float(np.mean(magnitude_minors(c)**2)) for s,c in cafs.items()}}
 if references is not None:
  surprises={s:scale_surprise(minors[s],references[s]) for s in SCALES}
  result["scale_scores"]=surprises;result["node_score"]=max(surprises.values())
 return result


def full_score(nodes:Iterable[float],minimum:int=4)->float|None:
 values=np.asarray(tuple(nodes),float);values=values[np.isfinite(values)]
 return float(np.median(values)) if len(values)>=minimum else None


def factor_counts(rows:list[dict[str,object]],key:str)->dict[str,int]:
 return {str(k):v for k,v in sorted(Counter(r[key] for r in rows).items(),key=lambda x:str(x[0]))}


def cramers_v(rows:list[dict[str,object]],a:str,b:str)->float:
 av=sorted({r[a] for r in rows},key=str);bv=sorted({r[b] for r in rows},key=str)
 table=np.zeros((len(av),len(bv)),int)
 for row in rows:table[av.index(row[a]),bv.index(row[b])]+=1
 chi2=chi2_contingency(table,correction=False)[0];n=table.sum()
 return float(np.sqrt(chi2/max(n*min(len(av)-1,len(bv)-1),1)))


def balanced_factorial(seed:int,count:int=42)->list[dict[str,object]]:
 levels={"rho_db":[-10,-6,0],"delay_chips":[-.5,-.25,-.1,.1,.25,.5],"doppler_hz":[0,-2,2,-5,5],
  "phase_rad":[0.,float(np.pi/2),float(np.pi),float(3*np.pi/2)]}
 rng=np.random.default_rng(seed);best=None
 for _ in range(4000):
  columns={}
  for key,values in levels.items():
   base=(values*(count//len(values))+values[:count%len(values)]);base=np.asarray(base,dtype=float);rng.shuffle(base);columns[key]=base
  rows=[{k:(int(v[i]) if k in ("rho_db","doppler_hz") else float(v[i])) for k,v in columns.items()} for i in range(count)]
  if len({tuple(r.values()) for r in rows})<count:continue
  pairs=[("rho_db","delay_chips"),("rho_db","doppler_hz"),("rho_db","phase_rad"),("delay_chips","doppler_hz"),("delay_chips","phase_rad"),("doppler_hz","phase_rad")]
  score=max(cramers_v(rows,a,b) for a,b in pairs)
  if best is None or score<best[0]:best=(score,rows)
 if best is None:raise RuntimeError("balanced design search failed")
 return best[1]


def assign_cases(seed:int,dataset:str,prns:list[int],anchors:list[int])->list[dict[str,object]]:
 factors=balanced_factorial(seed,42);rng=np.random.default_rng(seed+991)
 single_prns=np.repeat(np.asarray(prns),6);rng.shuffle(single_prns)
 single_anchors=np.resize(np.asarray(anchors),30);rng.shuffle(single_anchors)
 exclusions=np.resize(np.asarray(prns),12);rng.shuffle(exclusions)
 rows=[]
 for i,factor in enumerate(factors):
  if i<30:mode="single_prn";targets=[int(single_prns[i])];anchor=int(single_anchors[i])
  else:
   mode="simultaneous_four_prn";excluded=int(exclusions[i-30]);targets=[p for p in prns if p!=excluded][:4];anchor=int(anchors[i-30])
  strong=factor["rho_db"]>=-6 and (abs(factor["delay_chips"])>=.25 or abs(factor["doppler_hz"])>=2)
  rows.append({"case_id":f"{dataset}.r1.{i:02d}","dataset":dataset,"mode":mode,"anchor_start_sample":anchor,
   "target_prns":targets,**factor,"strong_resolvable":bool(strong)})
 return rows


def design_balance(rows:list[dict[str,object]])->dict[str,object]:
 factors=("rho_db","delay_chips","doppler_hz","phase_rad");counts={k:factor_counts(rows,k) for k in factors}
 diffs={k:max(v.values())-min(v.values()) for k,v in counts.items()}
 associations={f"{a}__{b}":cramers_v(rows,a,b) for i,a in enumerate(factors) for b in factors[i+1:]}
 pairs={(r["rho_db"],r["delay_chips"]) for r in rows};phase_dop={(r["phase_rad"],r["doppler_hz"]) for r in rows}
 return {"counts":counts,"count_differences":diffs,"cramers_v":associations,
  "delay_power_not_one_to_one":len(pairs)>max(len(counts["rho_db"]),len(counts["delay_chips"])),
  "phase_doppler_not_one_to_one":len(phase_dop)>max(len(counts["phase_rad"]),len(counts["doppler_hz"])),
  "strong_cases":sum(bool(r.get("strong_resolvable",r["rho_db"]>=-6 and (abs(r["delay_chips"])>=.25 or abs(r["doppler_hz"])>=2))) for r in rows),"status":"PASS" if max(diffs.values())<=1 else "FAIL"}


def canonical_sha(value:object)->str:
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def roc_statistics(labels:Iterable[int],scores:Iterable[float])->dict[str,float]:
 y=np.asarray(tuple(labels),int);s=np.asarray(tuple(scores),float);fpr,tpr,_=roc_curve(y,s);mask=fpr<=.05
 if not np.any(fpr==.05):
  t=np.interp(.05,fpr,tpr);x=np.r_[fpr[mask],.05];z=np.r_[tpr[mask],t]
 else:x=fpr[mask];z=tpr[mask]
 return {"roc_auc":float(roc_auc_score(y,s)),"pr_auc":float(average_precision_score(y,s)),"partial_auc_fpr_05":float(auc(x,z)/.05)}


def paired_bootstrap(values:Iterable[float],seed:int=20260819,resamples:int=10000)->dict[str,float]:
 x=np.asarray(tuple(values),float);rng=np.random.default_rng(seed);means=[]
 for _ in range(resamples):means.append(float(np.mean(rng.choice(x,len(x),replace=True))))
 return {"mean":float(np.mean(x)),"lower_95":float(np.quantile(means,.025)),"upper_95":float(np.quantile(means,.975))}


def abs_spearman(a:Iterable[float],b:Iterable[float])->float:
 return float(abs(spearmanr(tuple(a),tuple(b)).statistic))


def read_mapping(path:Path)->list[dict[str,str]]:
 with gzip.open(path,"rt",newline="") as stream:return list(csv.DictReader(stream))
