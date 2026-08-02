#!/usr/bin/env python3
"""Evaluate frozen CMTE on explicit canonical TEXBAT node/manifest mappings."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.cmte import *
from gnss_doppler_lab.cmte_inputs import load_checkpoint,extract_recording_innovations

def binom(k,n,p): return sum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(k,n+1))
def a1(nodes,doc):
 rows=[]
 for (run,t),g in nodes.groupby(["run_id","window_bin_s"],sort=True):
  x=g.b0_prn_node_rmse.to_numpy(); surprises=[-math.log(max(binom(int(np.sum(x>doc["train_node_thresholds"][q])),len(x),doc["train_exceedance_rates"][q]),1e-300)) for q in ("50","70","80")]; rows.append((run,t,max(surprises)))
 f=pd.DataFrame(rows,columns=["run","t","z"]); return f.groupby("run",sort=False).z.transform(lambda x:x.ewm(alpha=.75,adjust=False).mean()).to_numpy()
def cusum(x,runs,drift):
 out=[];p=None;g=0.
 for z,r in zip(x,runs):
  if r!=p:g=0.;p=r
  g=max(0.,g+float(z)-drift);out.append(g)
 return np.asarray(out)
def summary_values(x,mask):
 a=np.asarray(x)[mask]; return {"median":None if not len(a) else float(np.median(a)),"q90":None if not len(a) else float(np.quantile(a,.9)),"q99":None if not len(a) else float(np.quantile(a,.99)),"n":int(len(a))}
def metric_row(scenario,method,score,times,threshold,clean_fpr):
 score=np.asarray(score,float); times=np.asarray(times,float); masks=epoch_masks(times); keep=masks["stable"]|masks["established"]; labels=masks["established"][keep].astype(int); alarm=score>threshold; stable=masks["stable"]; est=masks["established"]
 auc=float(roc_auc_score(labels,score[keep])) if len(np.unique(labels))==2 else float("nan"); pr=float(average_precision_score(labels,score[keep])) if len(np.unique(labels))==2 else float("nan")
 duration=max(1e-9,stable.sum()*.5/60); hits=np.flatnonzero(alarm&est); first=None if not len(hits) else float(times[hits[0]])
 return {"scenario":scenario,"method":method,"threshold":threshold,"roc_auc":auc,"pr_auc":pr,"independent_clean_fpr":clean_fpr,
  "stable_pre_fpr":float(alarm[stable].mean()) if stable.any() else float("nan"),"false_alarms_per_min":float(alarm[stable].sum()/duration) if stable.any() else float("nan"),
  "detection":bool(np.any(alarm&est)),"first_alarm_availability_s":first,"first_alarm_delay_s":None if first is None else first-100.,
  "persistent_detection":float(alarm[est].mean()) if est.any() else float("nan"),"pre_summary":json.dumps(summary_values(score,stable),sort_keys=True),"post_summary":json.dumps(summary_values(score,est),sort_keys=True)}
def save_plot(path,epoch,scores,threshold,scenario):
 fig,ax=plt.subplots(figsize=(10,4));
 if "mean_e" in epoch: ax.plot(epoch.availability_time_s,epoch.mean_e,label="epoch conformal mean e",alpha=.7)
 ax.plot(epoch.availability_time_s,scores,label="Full sequential");ax.axhline(threshold,color="r",ls="--",label="Full threshold");ax.axvline(100,color="k",ls=":");ax.set(title=f"{scenario}: conformal and sequential scores (N shown)",xlabel="availability s",ylabel="score");ax2=ax.twinx();ax2.plot(epoch.availability_time_s,epoch.N,color="gray",alpha=.25);ax2.set_ylabel("N");ax.legend();fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--state-dir",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--expected-sha",required=True);p.add_argument("--scenario",action="append",required=True,help="DS1=/node.csv=/manifest.json");p.add_argument("--out",required=True);p.add_argument("--device",default="cpu")
 a=p.parse_args(argv);sd=Path(a.state_dir).resolve(strict=True);ck=Path(a.checkpoint).resolve(strict=True);actual=file_sha256(ck)
 if actual.lower()!=a.expected_sha.lower():raise ValueError("checkpoint bytes do not match expected SHA")
 config=json.loads((sd/"config.json").read_text());
 if config["checkpoint_sha256"].lower()!=actual.lower():raise ValueError("state/checkpoint mismatch")
 state=load_state(sd/"model_state.json",expected_checkpoint_sha256=actual);thresholds=json.loads((sd/"thresholds.json").read_text());model,features,mean,std,cfg=load_checkpoint(ck,actual,a.device,ROOT)
 specs=[]
 for s in a.scenario:
  parts=s.split("=",2)
  if len(parts)!=3:raise ValueError("scenario mapping must be NAME=/node.csv=/manifest.json")
  name,node,man=parts
  if name.upper() not in {"DS1","DS2","DS3","DS4"}:raise ValueError("scenarios must be DS1-DS4")
  specs.append((name.upper(),Path(node).resolve(strict=True),Path(man).resolve(strict=True)))
 if {x[0] for x in specs}!={"DS1","DS2","DS3","DS4"} or len(specs)!=4:raise ValueError("exactly one mapping for every DS1-DS4 is required")
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 for d in ("per_epoch","per_prn","plots","diagnostics","provenance"): (out/d).mkdir()
 for artifact in ("config.json","training_summary.json","calibration_summary.json","thresholds.json","code_hashes.json","residual_manifest.json"):(out/artifact).write_bytes((sd/artifact).read_bytes())
 clean_prn=pd.read_csv(sd/"clean_per_prn.csv"); clean=clean_prn[clean_prn.split=="test"].copy(); clean_epoch=aggregate_epochs(clean)
 clean_base=baseline_epoch_scores(clean);clean_rmse=clean.groupby(["run_id","window_bin_s"],sort=True).b0_prn_node_rmse.mean().to_numpy(); clean_scores={"A0":clean_base.A0,"A1":a1(clean,thresholds["baselines"]["A1"]),"A2":clean_base.A2,"A3":cusum(clean_rmse,clean_epoch.run_id,thresholds["baselines"]["A3"]["drift"]),"A4":clean_base.A4}
 cs=sequential_scores(np.log(np.maximum(clean_epoch.mean_e,1e-300)),clean_epoch.run_id,drift=thresholds["sequence"]["drift"]);clean_scores["Full"]=cs.s1_log_capital if thresholds["sequence"]["choice"]=="S1" else cs.s2_e_cusum
 th={k:v["threshold"] for k,v in thresholds["baselines"].items()};th["Full"]=thresholds["sequence"]["threshold"];clean_fpr={k:float(np.mean(np.asarray(v)>th[k])) for k,v in clean_scores.items()}
 metrics=[];ablations=[];prn_summary=[];provenance={"checkpoint":str(ck),"checkpoint_sha256":actual,"state_dir":str(sd),"scenario_inputs":{},"frozen_parameters":True,"attack_tuning":False};clip_total=0
 for name,node,man in specs:
  doc=json.loads(man.read_text());
  if doc.get("scenario","").upper()!=name or doc.get("role")!="evaluation_only" or doc.get("checkpoint_sha256","").lower()!=actual.lower():raise ValueError("scenario manifest identity/checkpoint mismatch")
  if doc.get("producer_grade") not in {"reconstructed_equivalence","verified_node_artifact"}:raise ValueError("scenario producer grade not allowed")
  if not all(__import__("re").fullmatch(r"[0-9a-fA-F]{64}",str(doc.get(k,""))) for k in ("source_sha256","node_sha256","checkpoint_sha256")):raise ValueError("scenario manifest hashes invalid")
  if file_sha256(node).lower()!=doc.get("node_sha256","").lower():raise ValueError("scenario node SHA mismatch")
  raw=pd.read_csv(node); residual=extract_recording_innovations(raw,model,features,mean,std,scenario=name,seq_len=cfg.seq_len,device=a.device);validate_residual_frame(residual,require_history_reset=True)
  nodes=score_residuals(residual,state);nodes.to_csv(out/"per_prn"/f"{name}.csv",index=False);epoch=aggregate_epochs(nodes);base=baseline_epoch_scores(nodes);rmse=nodes.groupby(["run_id","window_bin_s"],sort=True).b0_prn_node_rmse.mean().to_numpy()
  seq=sequential_scores(np.log(np.maximum(epoch.mean_e,1e-300)),epoch.run_id,drift=thresholds["sequence"]["drift"]);full=seq.s1_log_capital if thresholds["sequence"]["choice"]=="S1" else seq.s2_e_cusum
  scores={"A0":base.A0.to_numpy(),"A1":a1(nodes,thresholds["baselines"]["A1"]),"A2":base.A2.to_numpy(),"A3":cusum(rmse,epoch.run_id,thresholds["baselines"]["A3"]["drift"]),"A4":base.A4.to_numpy(),"Full":np.asarray(full)}
  for k,v in scores.items():epoch[f"score_{k}"]=v;metrics.append(metric_row(name,k,v,epoch.availability_time_s,th[k],clean_fpr[k]))
  epoch["phase"]=label_epochs(epoch.availability_time_s);epoch=pd.concat([epoch,seq],axis=1);epoch.to_csv(out/"per_epoch"/f"{name}.csv",index=False)
  for method in SCORE_METHODS:
   tmp=nodes.copy();tmp["p"]=tmp[f"p_{method}"];tmp["e"]=tmp[f"e_{method}"];z=aggregate_epochs(tmp);m=metric_row(name,method,z.mean_e,z.availability_time_s,float("nan"),float("nan"));ablations.append(m)
  for prn,g in nodes.groupby("prn"):prn_summary.append({"scenario":name,"prn":prn,"rows":len(g),"p_median":g.p.median(),"e_q99":g.e.quantile(.99)})
  save_plot(out/"plots"/f"{name}_scores.png",epoch,full,th["Full"],name)
  pivot=nodes.pivot_table(index="prn",columns="window_bin_s",values="e",aggfunc="mean");fig,ax=plt.subplots(figsize=(10,4));im=ax.imshow(np.log1p(pivot),aspect="auto");fig.colorbar(im,ax=ax);ax.set_title(f"{name} PRN log(1+e)");fig.tight_layout();fig.savefig(out/"plots"/f"{name}_prn_heatmap.png",dpi=120);plt.close(fig)
  fig,ax=plt.subplots();ax.hist(nodes.p,bins=20);ax.set_title(f"{name} conformal p distribution");fig.savefig(out/"plots"/f"{name}_pvalues.png",dpi=120);plt.close(fig)
  perm=epoch.sample(frac=1,random_state=2026);alt=sequential_scores(np.log(np.maximum(perm.mean_e,1e-300)),[name]*len(perm),drift=thresholds["sequence"]["drift"]);diag={"diagnostic_only":True,"epoch_multiset_invariant":np.allclose(np.sort(epoch.mean_e),np.sort(perm.mean_e)),"sequential_final_delta":float(abs(seq.s2_e_cusum.iloc[-1]-alt.s2_e_cusum.iloc[-1])),"trajectory_l2_delta":float(np.linalg.norm(np.sort(seq.s2_e_cusum)-np.sort(alt.s2_e_cusum))),"e_clip":1e-15,"e_clipped_count":int((nodes.p<1e-15).sum())};clip_total+=diag["e_clipped_count"];(out/"diagnostics"/f"{name}_order_shuffle.json").write_text(json.dumps(diag,indent=2)+"\n")
  provenance["scenario_inputs"][name]={"node":str(node),"node_sha256":file_sha256(node),"manifest":str(man),"manifest_sha256":file_sha256(man),"producer_grade":doc["producer_grade"]}
 pd.DataFrame(metrics).to_csv(out/"scenario_metrics.csv",index=False);pd.DataFrame(ablations).to_csv(out/"ablation_metrics.csv",index=False);pd.DataFrame(prn_summary).to_csv(out/"per_prn_evidence_summary.csv",index=False)
 # baseline-vs-Full and DS1 pre-onset zoom are actual plots.
 mf=pd.DataFrame(metrics);fig,ax=plt.subplots(figsize=(10,4));
 for method,g in mf.groupby("method"):ax.plot(g.scenario,g.roc_auc,marker="o",label=method)
 ax.legend(ncol=3);ax.set_ylabel("ROC-AUC");fig.tight_layout();fig.savefig(out/"plots"/"baseline_vs_full.png",dpi=120);plt.close(fig)
 ds1=pd.read_csv(out/"per_epoch"/"DS1.csv");fig,ax=plt.subplots();pre=ds1.availability_time_s<110;ax.plot(ds1.loc[pre,"availability_time_s"],ds1.loc[pre,"score_Full"]);ax.axhline(th["Full"],color="r");fig.tight_layout();fig.savefig(out/"plots"/"DS1_pre_zoom.png",dpi=120);plt.close(fig)
 full=mf[mf.method=="Full"];go={"all_ds_detected":bool(full.detection.all()),"all_ds_roc_auc_ge_0_8":bool((full.roc_auc.fillna(0)>=.8).all()),"independent_clean_fpr_le_0_01":bool((full.independent_clean_fpr<=.01).all())};go["pass"]=all(go.values())
 (out/"test_summary.json").write_text(json.dumps({"status":"finalized_by_actual_eval","go_criteria":go},indent=2)+"\n");(out/"diagnostics"/"summary.json").write_text(json.dumps({"e_clip":1e-15,"e_clipped_count":clip_total,"validation_order_shuffle":"see training candidate and clean validation; deterministic seed 2026"},indent=2)+"\n");(out/"provenance"/"provenance.json").write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n")
 (out/"README.md").write_text("CMTE sequential conformal evidence detector with empirically calibrated false-alarm control. Thresholds are frozen from normal validation only. Historical B0 summaries are historical_noncomparable. Empty metrics are explicit NaN/null.\n")
 checks={str(x.relative_to(out)):file_sha256(x) for x in sorted(out.rglob("*")) if x.is_file()};(out/"checksums.json").write_text(json.dumps(checks,indent=2,sort_keys=True)+"\n");print(json.dumps({"out":str(out),"go":go},sort_keys=True))
if __name__=="__main__":main()
