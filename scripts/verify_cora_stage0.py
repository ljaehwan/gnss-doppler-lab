#!/usr/bin/env python3
"""Artifact-only independent metric and verdict verifier for CORA Stage-0."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_ART=ROOT/"artifacts/cora_stage0_cross_prn_common_origin"
BOOTSTRAPS=2000;BOOTSTRAP_SEED=20260821
REQUIRED={"README.md","config.json","configuration_freeze.json","source_commit.json","data_inventory.json","raw_source_binding.json","clean_split_audit.json","normal_model_summary.json","synthetic_control_metrics.json","thresholds.json","scenario_metrics.csv","family_metrics.csv","ablation_metrics.csv","per_block_scores.csv.gz","cross_prn_cumulant_matrices.npz","relation_destruction_metrics.json","shortcut_audit.json","leave_one_prn_out.csv","bootstrap_intervals.csv","literature_novelty_audit.md","final_verdict.json","artifact_manifest_sha256.json"}

def digest(path):
 d=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):d.update(chunk)
 return d.hexdigest()
def manifest(art):
 return {str(p.relative_to(art)):digest(p) for p in sorted(art.rglob("*")) if p.is_file() and p.name!="artifact_manifest_sha256.json"}
def csvrows(path,gz=False):
 opener=gzip.open if gz else open
 with opener(path,"rt" if gz else "r",newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def f(value):return float(value)
def bootstrap(left,right,blocks,seed):
 blocks=np.asarray(blocks,int);unique=np.unique(blocks);paired=np.array([np.mean(np.asarray(left)[blocks==b]-np.asarray(right)[blocks==b]) for b in unique]);rng=np.random.default_rng(seed);draw=[]
 for _ in range(BOOTSTRAPS):draw.append(rng.choice(paired,len(paired),replace=True).mean())
 return float(paired.mean()),float(np.quantile(draw,.025)),float(np.quantile(draw,.975))
def close(a,b,tol=1e-9):return abs(float(a)-float(b))<=tol*max(1,abs(float(a)),abs(float(b)))

def verify(art:Path)->dict:
 failures=[];checks={};present={str(p.relative_to(art)) for p in art.rglob("*") if p.is_file()}
 missing=sorted(REQUIRED-present);checks["required_files"]={"missing":missing};failures += [f"missing:{x}" for x in missing]
 plots=sorted((art/"plots").glob("*.png"));checks["plots"]={"count":len(plots)}
 if len(plots)<10:failures.append("plots")
 expected=json.loads((art/"artifact_manifest_sha256.json").read_text());actual=manifest(art);checks["manifest"]={"entry_count":len(actual),"match":expected==actual}
 if expected!=actual:failures.append("manifest")
 rows=csvrows(art/"per_block_scores.csv.gz",True);thresholds=json.loads((art/"thresholds.json").read_text())["primary"]
 for domain in ("OAK","TEX"):
  cal=np.array([f(r["score_Full"]) for r in rows if r["domain"]==domain and r["partition"]=="calibration"]);q=float(np.quantile(cal,.99,method="higher"));checks[f"threshold_{domain}"]={"recomputed":q,"reported":thresholds[domain]["q99"],"count":len(cal)}
  if not close(q,thresholds[domain]["q99"]):failures.append(f"threshold:{domain}")
 scenario_report={r["dataset"]:r for r in csvrows(art/"scenario_metrics.csv")};scenario_calc={}
 for name,reported in scenario_report.items():
  rr=[r for r in rows if r["dataset"]==name];y=np.array([int(f(r["label"])) for r in rr]);s=np.array([f(r["score_Full"]) for r in rr]);a=np.array([bool(int(f(r["alarm_Full"]))) for r in rr]);
  calc={"roc_auc":roc_auc_score(y,s),"pauc_0_05":roc_auc_score(y,s,max_fpr=.05),"pr_auc":average_precision_score(y,s),"preonset_fpr":a[y==0].mean(),"attack_detection_rate":a[y==1].mean()};scenario_calc[name]={k:float(v) for k,v in calc.items()}
  for key,value in calc.items():
   if not close(value,reported[key]):failures.append(f"scenario:{name}:{key}")
 family_report={r["family"]:r for r in csvrows(art/"family_metrics.csv")};family_calc={}
 for family,reported in family_report.items():
  rr=[r for r in rows if r["family"]==family and r["partition"] in ("preonset","attack")];y=np.array([int(f(r["label"])) for r in rr]);s=np.array([f(r["score_Full"]) for r in rr]);a=np.array([bool(int(f(r["alarm_Full"]))) for r in rr]);calc={"roc_auc":roc_auc_score(y,s),"pauc_0_05":roc_auc_score(y,s,max_fpr=.05),"pr_auc":average_precision_score(y,s),"preonset_fpr":a[y==0].mean(),"attack_detection_rate":a[y==1].mean()};family_calc[family]={k:float(v) for k,v in calc.items()}
  for key,value in calc.items():
   if not close(value,reported[key]):failures.append(f"family:{family}:{key}")
 # Recompute matrix likelihood scores without trusting score JSON/CSV.
 model=json.loads((art/"normal_model_summary.json").read_text());matrix_data=np.load(art/"cross_prn_cumulant_matrices.npz");matrix_mismatch=0
 for key in matrix_data.files:
  if not key.endswith("__matrices"):continue
  name=key[:-10];mats=matrix_data[key];counts=matrix_data[f"{name}__prn_count"];domain="OAK" if name.startswith("oak") else "TEX";nv=float(model[domain]["null_variance"]);rr=[r for r in rows if r["dataset"]==name]
  for matrix,count,row in zip(mats,counts,rr,strict=True):
   n=int(count);k=matrix[:n,:n];off=~np.eye(n,dtype=bool);sigma=max(nv,1e-8);h0=np.sum(k[off]**2)/sigma;values,vectors=np.linalg.eigh(k);top=max(float(values[-1]),0);rank=top*np.outer(vectors[:,-1],vectors[:,-1]);np.fill_diagonal(rank,0);h1=np.sum((k[off]-rank[off])**2)/sigma;score=max(float(h0-h1),0)-n*np.log(max(n*(n-1)//2,2));matrix_mismatch+=not close(score,row["score_Full"],1e-8)
 checks["matrix_score_recomputation"]={"mismatches":int(matrix_mismatch)}
 if matrix_mismatch:failures.append("matrix_scores")
 relation=json.loads((art/"relation_destruction_metrics.json").read_text());relation_mismatch=0
 for name,items in relation["datasets"].items():
  rr=[r for r in rows if r["dataset"]==name and r["partition"]=="attack"]
  for key,reported in items.items():
   base=np.array([f(r["score_Full"]) for r in rr]);destroy=np.array([f(r[f"relation_{key}"]) for r in rr]);blocks=np.array([int(f(r["bootstrap_block"])) for r in rr]);est,lo,hi=bootstrap(base,destroy,blocks,BOOTSTRAP_SEED+len(key));relation_mismatch+=not(close(est,reported["mean_score_drop"]) and close(lo,reported["ci95"][0]) and close(hi,reported["ci95"][1]))
 checks["relation_recomputation"]={"mismatches":int(relation_mismatch)}
 if relation_mismatch:failures.append("relation")
 final=json.loads((art/"final_verdict.json").read_text());synthetic=json.loads((art/"synthetic_control_metrics.json").read_text())["domains"];shortcut=json.loads((art/"shortcut_audit.json").read_text());lopo=csvrows(art/"leave_one_prn_out.csv");abl=csvrows(art/"ablation_metrics.csv")
 clean_fpr={d:np.mean([int(f(r["alarm_Full"])) for r in rows if r["domain"]==d and r["partition"]=="holdout"]) for d in ("OAK","TEX")};tex=sum(v["pauc_0_05"]>=.8 and v["attack_detection_rate"]>=.7 for k,v in family_calc.items() if k.startswith("TEX"));oak=family_calc["OAK_OS3_OS4"]
 superiority={}
 for family in family_calc:
  vals={r["method"]:f(r["pauc_0_05"]) for r in abl if r["scope"]==family};superiority[family]=all(vals["Full"]>vals[k] for k in ("A0","A2","A4"))
 gates={"clean_holdout_q99_fpr_le_0_02":max(clean_fpr.values())<=.02,"external_preonset_worst_fpr_le_0_05":max(v["preonset_fpr"] for v in scenario_calc.values())<=.05,"shared_synthetic_significant_both":all(v["significant"] for v in synthetic.values()),"receiver_nuisance_no_persistent_alarm":not any(v["receiver_nuisance_persistent_alarm"] for v in synthetic.values()),"two_tex_families":tex>=2,"oak_family":oak["pauc_0_05"]>=.8 and oak["attack_detection_rate"]>=.7,"full_beats_A0_A2_A4_required_families":superiority["OAK_OS3_OS4"] and sum(v for k,v in superiority.items() if k.startswith("TEX"))>=2,"attack_relation_destruction":all(x["pass"] for ds in relation["datasets"].values() for x in ds.values()),"clean_relation_shuffle_unchanged":all(x["unchanged"] for ds in relation["clean_holdout"].values() for x in ds.values()),"leave_one_prn_out_stable":all(r["status"]=="COMPUTED" and r["stable"]=="True" for r in lopo),"shortcut_audit_pass":shortcut["status"]=="PASS","B0_evidence_condition":False}
 gates={k:bool(v) for k,v in gates.items()};verdict="GO_FOR_CORA_NEURAL_STAGE1" if all(gates.values()) else "NO_GO_CORA_COMMON_ORIGIN_HYPOTHESIS";checks["verdict"]={"recomputed":verdict,"reported":final["verdict"],"gates":gates}
 if verdict!=final["verdict"] or gates!=final["gates"]:failures.append("verdict")
 return {"schema":"gnss-doppler-lab.cora-verification.v1","status":"PASS" if not failures else "FAIL","failures":failures,"checks":checks}

def main():
 p=argparse.ArgumentParser();p.add_argument("--artifact",type=Path,default=DEFAULT_ART);p.add_argument("--write-manifest",action="store_true");args=p.parse_args()
 if args.write_manifest:(args.artifact/"artifact_manifest_sha256.json").write_text(json.dumps(manifest(args.artifact),indent=2,sort_keys=True)+"\n")
 result=verify(args.artifact);print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result["status"]=="PASS" else 1)
if __name__=="__main__":main()
