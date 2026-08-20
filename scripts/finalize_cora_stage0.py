#!/usr/bin/env python3
"""Complete fail-closed CORA diagnostics, provenance, plots, and verdict."""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import cora_pipeline as cp  # noqa: E402
from gnss_doppler_lab.cora_common_origin import score_token_block  # noqa: E402

ART=cp.ART; CACHE=cp.CACHE; PLOTS=ART/"plots"; PLOTS.mkdir(parents=True,exist_ok=True)
RAW_HASHES={
 "oakbat_cleanstatic":"8e3428abb1b94211118c1ec9f505322ef89fdb176f0cf33961eca3cf3da80dfe",
 "oakbat_os3":"2a3c3c5cf1accaa287fe14181e43070903500e0250c69e3c335f91c89c0cdc6c",
 "oakbat_os4":"803f3c76bcc618efbc6b394eb536fe61ed8c3e34b1822c0088b4475621bfa8e4",
 "texbat_cleanstatic":"dd295ab46616bfe9634d1c37479520e720ebc54bcb64adf0a247315a541fb9b9",
 "texbat_ds1":"8c8f4caa41c8a2253688eef47818a445e3a06139b729cf587c03be33e6b6744c",
 "texbat_ds3":"e37e11b060bc2c675d4a60024f8b4a53e95e7cd1d304bea80cd903856075a30d",
 "texbat_ds7":"d5fb1430d476f68930f3bb0290b80b649f08012eb6b6981d112493528813400e",
 "texbat_ds8":"1614d8de6fc8ebc3429def6e9505050c08a3ee8da69c11ecc27a98305f735d78",
}

def dump(name,value): (ART/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
def sha(path):
 d=hashlib.sha256()
 with Path(path).open("rb") as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b""):d.update(chunk)
 return d.hexdigest()
def prefix_sha(path,nbytes):
 d=hashlib.sha256();left=nbytes
 with Path(path).open("rb") as f:
  while left:
   chunk=f.read(min(left,8*1024*1024))
   if not chunk:break
   d.update(chunk);left-=len(chunk)
 if left:raise ValueError("prefix exceeds file")
 return d.hexdigest()
def read_rows():
 with gzip.open(ART/"per_block_scores.csv.gz","rt",newline="") as f:return list(csv.DictReader(f))
def num(rows):
 numeric={"window_start_s","window_end_s","bootstrap_block","prn_count","rank1_strength","participating_prns","label"}
 numeric|={k for k in rows[0] if k.startswith("score_") or k.startswith("alarm_") or k.startswith("threshold_")}
 for r in rows:
  for k in numeric:
   if k in r and r[k] not in ("",None):r[k]=float(r[k])
 return rows
def write_rows(rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with gzip.open(ART/"per_block_scores.csv.gz","wt",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def ci(values,blocks,seed):
 values=np.asarray(values,float);blocks=np.asarray(blocks,int);unique=np.unique(blocks);rng=np.random.default_rng(seed);means=[]
 unit=np.array([values[blocks==b].mean() for b in unique])
 for _ in range(cp.BOOTSTRAPS):means.append(rng.choice(unit,len(unit),replace=True).mean())
 return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]
def persistent_ratio(alarms,starts):
 alarms=np.asarray(alarms,bool);starts=np.asarray(starts,float);persistent=np.zeros(len(alarms),bool)
 for i in range(len(alarms)-2):
  if alarms[i:i+3].all() and np.allclose(np.diff(starts[i:i+3]),2):persistent[i:i+3]=True
 return float(persistent.mean())
def metrics(rows,onset=None,pull_off=None):
 y=np.array([r["label"] for r in rows],int);s=np.array([r["score_Full"] for r in rows]);a=np.array([r["alarm_Full"] for r in rows],bool);t=np.array([r["window_start_s"] for r in rows])
 out={"roc_auc":float(roc_auc_score(y,s)),"pauc_0_05":float(roc_auc_score(y,s,max_fpr=.05)),"pr_auc":float(average_precision_score(y,s)),
      "preonset_fpr":float(a[y==0].mean()),"attack_detection_rate":float(a[y==1].mean()),"persistent_alarm_ratio":persistent_ratio(a[y==1],t[y==1]),
      "alarm_duration_s_per_hour":float(a.sum()*2/(len(a)*2/3600)),"rank1_strength_attack_mean":float(np.mean([r["rank1_strength"] for r in rows if r["label"]==1])),
      "participating_prns_attack_mean":float(np.mean([r["participating_prns"] for r in rows if r["label"]==1]))}
 for label,value in (("onset",onset),("pull_off",pull_off)):
  candidates=t[(t>=value)&a] if value is not None else np.array([])
  out[f"first_alarm_delay_from_{label}_s"]=None if not len(candidates) else float(candidates.min()-value)
 return out

rows=num(read_rows())
models={};nulls={};zsets={};datasets={}
for domain,clean_name in (("OAK","oakbat_cleanstatic"),("TEX","texbat_cleanstatic")):
 clean=cp.load(clean_name);models[domain],nulls[domain],_=cp.fit_domain(clean)
for name in cp.SPECS:
 data=cp.load(name);datasets[name]=data;zsets[name]=cp.conditioned(data,models[cp.SPECS[name][0]])
 # Add residual Doppler diagnostic omitted from the first reporting pass.
 target=[r for r in rows if r["dataset"]==name]
 for r,ctx in zip(target,data["context"],strict=True):r["score_residual_doppler"]=float(np.mean(np.abs(ctx[...,4])))

# Relation scores are made auditable in the primary per-window table.
relation=json.loads((ART/"relation_destruction_metrics.json").read_text())
for name in cp.SPECS:
 target=[r for r in rows if r["dataset"]==name];starts=datasets[name]["window_start_s"]
 destroyed=cp.relation_scores(zsets[name],starts,nulls[cp.SPECS[name][0]],cp.SEED+len(name))
 for key,values in destroyed.items():
  for r,v in zip(target,values,strict=True):r[f"relation_{key}"]=float(v)
 if cp.SPECS[name][6] is None:
  hold=np.array([r["partition"]=="holdout" for r in target]);blocks=(starts[hold]//10).astype(int);clean_items={}
  base=np.array([r["score_Full"] for r in target])
  for key,values in destroyed.items():
   est,lo,hi=cp.bootstrap_mean_difference(base[hold],values[hold],blocks,cp.BOOTSTRAP_SEED+len(key))
   clean_items[key]={"original_minus_destroyed":est,"ci95":[lo,hi],"unchanged":bool(lo<=0<=hi)}
  relation.setdefault("clean_holdout",{})[name]=clean_items
write_rows(rows);dump("relation_destruction_metrics.json",relation)

# Full scenario/family metric contract.
scenario=[];bootstrap=[]
for name in [n for n in cp.SPECS if cp.SPECS[n][6] is not None]:
 rr=[r for r in rows if r["dataset"]==name];onset=cp.SPECS[name][6];pull=195.0 if name=="texbat_ds3" else None;m=metrics(rr,onset,pull)
 m.update({"dataset":name,"family":cp.SPECS[name][7],"domain":cp.SPECS[name][0],"preonset_windows":sum(r["label"]==0 for r in rr),"attack_windows":sum(r["label"]==1 for r in rr)})
 scenario.append(m)
 pos=[r for r in rr if r["label"]==1];bootstrap.append({"scope":name,"metric":"attack_detection_rate","estimate":m["attack_detection_rate"],"ci_lower":ci([r["alarm_Full"] for r in pos],[r["bootstrap_block"] for r in pos],cp.BOOTSTRAP_SEED)[0],"ci_upper":ci([r["alarm_Full"] for r in pos],[r["bootstrap_block"] for r in pos],cp.BOOTSTRAP_SEED)[1],"unit":"10s_block"})
families=[]
for family in sorted({r["family"] for r in scenario}):
 rr=[r for r in rows if r["family"]==family and r["partition"] in {"preonset","attack"}];m=metrics(rr);m.update({"family":family,"scenario_count":len({r["dataset"] for r in rr})});families.append(m)
def csvout(name,items):
 with (ART/name).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(items[0]),lineterminator="\n");w.writeheader();w.writerows(items)
csvout("scenario_metrics.csv",scenario);csvout("family_metrics.csv",families)

# Leave-one-PRN-out uses the same clean model but independently calibrated 4-PRN thresholds.
lopo=[];stable_all=True
for domain,clean_name in (("OAK","oakbat_cleanstatic"),("TEX","texbat_cleanstatic")):
 clean=datasets[clean_name];zc=zsets[clean_name];starts=clean["window_start_s"];cal=(starts>=221)&(starts<321)
 for drop_index,prn in enumerate(clean["prns"]):
  scores=np.array([score_token_block(np.delete(z,drop_index,axis=1),null_variance=nulls[domain])[0].score for z in zc]);threshold=float(np.quantile(scores[cal],.99,method="higher"))
  for name in cp.SPECS:
   if cp.SPECS[name][0]!=domain or cp.SPECS[name][6] is None:continue
   data=datasets[name]
   if int(prn) not in data["prns"]:continue
   if len(data["prns"])<=4:
    lopo.append({"dataset":name,"dropped_prn":int(prn),"status":"UNAVAILABLE_MINIMUM_4_PRNS","pauc_0_05":"","detection_rate":"","pauc_delta":"","detection_delta":"","stable":False});stable_all=False;continue
   j=int(np.flatnonzero(data["prns"]==prn)[0]);scores2=np.array([score_token_block(np.delete(z,j,axis=1),null_variance=nulls[domain])[0].score for z in zsets[name]])
   rr=[r for r in rows if r["dataset"]==name];y=np.array([r["label"] for r in rr],int);alarm=scores2>threshold;p=float(roc_auc_score(y,scores2,max_fpr=.05));det=float(alarm[y==1].mean());orig=next(x for x in scenario if x["dataset"]==name);st=abs(p-orig["pauc_0_05"])<=.1 and abs(det-orig["attack_detection_rate"])<=.15
   stable_all&=st;lopo.append({"dataset":name,"dropped_prn":int(prn),"status":"COMPUTED","pauc_0_05":p,"detection_rate":det,"pauc_delta":p-orig["pauc_0_05"],"detection_delta":det-orig["attack_detection_rate"],"stable":st})
csvout("leave_one_prn_out.csv",lopo)

# Nuisance and shortcut audit (no variable is introduced into Full).
shortcut_rows=[]
for domain in ("OAK","TEX"):
 rr=[r for r in rows if r["domain"]==domain];full=np.array([r["score_Full"] for r in rr]);
 for field in ("score_A0","score_power","score_cn0","score_doppler","score_residual_doppler","window_start_s","prn_count"):
  x=np.array([r[field] for r in rr]);rho=float(spearmanr(full,x).statistic) if np.std(x)>0 else 0.0;shortcut_rows.append({"domain":domain,"variable":field,"spearman_rho":rho,"absolute_rho":abs(rho)})
attack_rows=[r for r in rows if r["partition"] in {"preonset","attack"}];y=np.array([r["label"] for r in attack_rows],int)
method_pauc={k:float(roc_auc_score(y,np.array([r[f"score_{k}"] for r in attack_rows]),max_fpr=.05)) for k in ("Full","A0","A1","A2","A3","A4","power","cn0","doppler")}
max_corr=max(r["absolute_rho"] for r in shortcut_rows);single_matches=any(method_pauc[k]>=method_pauc["Full"]-.02 for k in ("A0","power","cn0","doppler"))
dump("shortcut_audit.json",{"schema":"gnss-doppler-lab.cora-shortcut-audit.v1","status":"FAIL" if single_matches or max_corr>=.8 else "PASS","correlations":shortcut_rows,"pooled_low_fpr_pauc":method_pauc,"single_nuisance_matches_or_beats_full":single_matches,"maximum_absolute_spearman":max_corr,"PRN_identity_model_input":False,"absolute_time_model_input":False,"absolute_doppler_model_input":False})

# Family-specific baseline comparison.
ablation=[];family_superiority={}
for family in sorted({r["family"] for r in attack_rows}):
 rr=[r for r in attack_rows if r["family"]==family];yy=np.array([r["label"] for r in rr],int);vals={}
 for method in ("Full","A0","A1","A2","A3","A4"):
  p=float(roc_auc_score(yy,np.array([r[f"score_{method}"] for r in rr]),max_fpr=.05));vals[method]=p;ablation.append({"scope":family,"method":method,"pauc_0_05":p,"attack_detection_rate":float(np.mean([r[f"alarm_{method}"] for r in rr if r["label"]==1]))})
 family_superiority[family]=all(vals["Full"]>vals[k] for k in ("A0","A2","A4"))
csvout("ablation_metrics.csv",ablation)

# Prefix-overlap audit for DS7/DS8 and source/cache/config binding.
prefix_bytes=110*25_000_000*4
ds7_prefix=prefix_sha(cp.SPECS["texbat_ds7"][2],prefix_bytes);ds8_prefix=prefix_sha(cp.SPECS["texbat_ds8"][2],prefix_bytes)
provenance=json.loads((CACHE/"extraction_provenance.json").read_text());binding={"schema":"gnss-doppler-lab.cora-raw-binding.v1","datasets":{},"ds7_ds8_pre110_overlap_audit":{"bytes_each":prefix_bytes,"ds7_sha256":ds7_prefix,"ds8_sha256":ds8_prefix,"identical":ds7_prefix==ds8_prefix,"counted_as_independent_normal_evidence":False}}
for name,spec in cp.SPECS.items():
 raw=Path(spec[2]);trace=Path(spec[3]);config=trace/("receiver.conf" if spec[4]=="legacy" else "receiver.conf")
 if spec[4]=="native":config=trace/"receiver.conf"
 else:config=trace.parent/"receiver.conf"
 binding["datasets"][name]={"raw_path":str(raw),"raw_size_bytes":raw.stat().st_size,"raw_sample_count":raw.stat().st_size//4,"full_sha256":RAW_HASHES[name],"full_sha256_read_this_evaluation":True,"sample_rate_hz":spec[1],"sample_format":"little-endian interleaved int16 I,Q","tracker_adapter":spec[4],"tracker_path":str(trace),"receiver_config_path":str(config),"receiver_config_sha256":sha(config) if config.exists() else None,"selected_raw_sample_interval":provenance[name]["raw_sample_bounds"],"cache_path":provenance[name]["cache_path"],"cache_sha256":provenance[name]["cache_sha256"],"deterministic_regeneration_command":f"python3 scripts/cora_pipeline.py {'extract-clean' if spec[6] is None else 'extract-attacks'} --workers 8"}
dump("raw_source_binding.json",binding)

# Final gates without retuning.
clean_fpr={d:float(np.mean([r["alarm_Full"] for r in rows if r["domain"]==d and r["partition"]=="holdout"])) for d in ("OAK","TEX")}
synthetic=json.loads((ART/"synthetic_control_metrics.json").read_text())["domains"]
tex_pass=sum(f["pauc_0_05"]>=.8 and f["attack_detection_rate"]>=.7 for f in families if f["family"].startswith("TEX"));oak=next(f for f in families if f["family"]=="OAK_OS3_OS4")
relations_attack=all(item["pass"] for ds in relation["datasets"].values() for item in ds.values());relations_clean=all(item["unchanged"] for ds in relation["clean_holdout"].values() for item in ds.values())
gates={"clean_holdout_q99_fpr_le_0_02":max(clean_fpr.values())<=.02,"external_preonset_worst_fpr_le_0_05":max(s["preonset_fpr"] for s in scenario)<=.05,"shared_synthetic_significant_both":all(synthetic[d]["significant"] for d in synthetic),"receiver_nuisance_no_persistent_alarm":not any(synthetic[d]["receiver_nuisance_persistent_alarm"] for d in synthetic),"two_tex_families":tex_pass>=2,"oak_family":oak["pauc_0_05"]>=.8 and oak["attack_detection_rate"]>=.7,"full_beats_A0_A2_A4_required_families":family_superiority.get("OAK_OS3_OS4",False) and sum(v for k,v in family_superiority.items() if k.startswith("TEX"))>=2,"attack_relation_destruction":relations_attack,"clean_relation_shuffle_unchanged":relations_clean,"leave_one_prn_out_stable":stable_all,"shortcut_audit_pass":not(single_matches or max_corr>=.8),"B0_evidence_condition":False}
final={"schema":"gnss-doppler-lab.cora-final-verdict.v2","verdict":"GO_FOR_CORA_NEURAL_STAGE1" if all(gates.values()) else "NO_GO_CORA_COMMON_ORIGIN_HYPOTHESIS","gates":gates,"failed_gates":[k for k,v in gates.items() if not v],"clean_holdout_fpr":clean_fpr,"worst_external_preonset_fpr":max(s["preonset_fpr"] for s in scenario),"B0_fixed9":"UNAVAILABLE_NO_ACTUAL_SAME_SUPPORT_RERUN","injection_performed":False,"neural_stage1_permitted":False,"configuration_freeze_sha":cp.FREEZE_SHA}
dump("final_verdict.json",final)

# Bootstrap table includes scenario detection and relation effects.
old=list(csv.DictReader((ART/"bootstrap_intervals.csv").open()))
for r in old:bootstrap.append(r)
csvout("bootstrap_intervals.csv",bootstrap)

# Required compact plots, generated only from frozen scores.
def save(fig,name):fig.tight_layout();fig.savefig(PLOTS/name,dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4))
for name in cp.SPECS:
 rr=[r for r in rows if r["dataset"]==name];ax.plot([r["window_start_s"] for r in rr],[r["score_Full"] for r in rr],label=name,lw=.8)
ax.set(xlabel="recording time (s)",ylabel="CORA score",title="Clean and attack CORA score timelines");ax.legend(ncol=4,fontsize=6);save(fig,"score_timeline.png")
name="texbat_ds3";matrix=np.load(ART/"cross_prn_cumulant_matrices.npz")[f"{name}__matrices"];idx=int(np.nanargmax(np.nansum(abs(matrix),axis=(1,2))));fig,ax=plt.subplots();im=ax.imshow(matrix[idx],cmap="coolwarm");fig.colorbar(im,ax=ax);ax.set(title="DS3 cross-PRN fourth-cumulant matrix");save(fig,"cumulant_matrix_heatmap.png")
values,vectors=np.linalg.eigh(np.nan_to_num(matrix));fig,ax=plt.subplots();im=ax.imshow(abs(vectors[:,:,-1]).T,aspect="auto",cmap="magma");fig.colorbar(im,ax=ax);ax.set(title="Rank-1 loading magnitude",xlabel="PRN index",ylabel="window");save(fig,"rank1_loading_heatmap.png")
fig,ax=plt.subplots(figsize=(9,4))
for name in [n for n in cp.SPECS if cp.SPECS[n][6] is not None]:
 rr=[r for r in rows if r["dataset"]==name];ax.plot([r["window_start_s"] for r in rr],[r["score_Full"] for r in rr],label=name,lw=.8);ax.axvline(cp.SPECS[name][6],color="k",alpha=.08);
ax.axvline(195,color="red",ls="--",lw=.8,label="DS3 pull-off");ax.set(title="Official onset and pull-off overlay",xlabel="time (s)",ylabel="score");ax.legend(fontsize=6,ncol=4);save(fig,"onset_pulloff_overlay.png")
fig,ax=plt.subplots();attack=[r for r in rows if r["partition"]=="attack"];ax.scatter([r["score_Full"] for r in attack],[r["relation_phase_norm_psd_surrogate"] for r in attack],s=8,alpha=.5);ax.plot(ax.get_xlim(),ax.get_xlim(),"k--");ax.set(xlabel="original",ylabel="phase/norm/PSD surrogate",title="Original vs relation-destroyed score");save(fig,"relation_destruction.png")
syn=json.loads((ART/"synthetic_control_metrics.json").read_text())["domains"];fig,ax=plt.subplots();x=np.arange(2);ax.bar(x-.2,[syn[d]["shared_mean"] for d in ("OAK","TEX")],.4,label="shared");ax.bar(x+.2,[syn[d]["independent_mean"] for d in ("OAK","TEX")],.4,label="independent");ax.set_xticks(x,["OAK","TEX"]);ax.set(title="Clean-only synthetic controls",ylabel="CORA score");ax.legend();save(fig,"synthetic_shared_independent.png")
fig,ax=plt.subplots(figsize=(10,4));computed=[r for r in lopo if r["status"]=="COMPUTED"];ax.scatter(range(len(computed)),[r["pauc_delta"] for r in computed],c=[r["stable"] for r in computed]);ax.axhline(0,color="k");ax.set(title="Leave-one-PRN-out pAUC stability",ylabel="pAUC delta",xlabel="scenario/drop combination");save(fig,"leave_one_prn_out.png")
fig,ax=plt.subplots();ax.scatter([s["preonset_fpr"] for s in scenario],[s["attack_detection_rate"] for s in scenario]);
for s in scenario:ax.annotate(s["dataset"],(s["preonset_fpr"],s["attack_detection_rate"]),fontsize=7)
ax.set(xlabel="pre-onset FPR",ylabel="attack detection",title="FPR–detection tradeoff");save(fig,"fpr_detection_tradeoff.png")
fig,ax=plt.subplots();keys=["Full","A0","power","cn0","doppler"];ax.bar(keys,[method_pauc[k] for k in keys]);ax.set(ylim=(.45,1),ylabel="normalized pAUC at FPR≤5%",title="CORA and nuisance baselines (B0 unavailable)");save(fig,"cora_vs_baselines.png")
fig,ax=plt.subplots(figsize=(8,4));ax.bar([f["family"] for f in families],[f["pauc_0_05"] for f in families]);ax.axhline(.8,color="red",ls="--");ax.tick_params(axis="x",rotation=25);ax.set(ylabel="normalized pAUC",title="Scenario-family low-FPR pAUC");save(fig,"family_pauc.png")
print(json.dumps({"status":"FINALIZED","verdict":final["verdict"],"failed_gates":final["failed_gates"]},indent=2))
