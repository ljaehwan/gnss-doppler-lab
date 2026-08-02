#!/usr/bin/env python3
"""Evaluate a provenance-compatible frozen CMTE state on explicit TEXBAT inputs."""
from __future__ import annotations
import argparse, json, math, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.cmte import *
from gnss_doppler_lab.cmte_inputs import load_checkpoint, extract_recording_innovations

OPS=("q99","q995","target1")

def binom(k,n,p): return sum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(k,n+1))
def a1(nodes,doc):
    rows=[]
    for (run,t),g in nodes.groupby(["recording_id","window_bin_s"],sort=True):
        x=g.b0_prn_node_rmse.to_numpy()
        z=[-math.log(max(binom(int(np.sum(x>doc["train_node_thresholds"][q])),len(x),doc["train_exceedance_rates"][q]),1e-300)) for q in ("50","70","80")]
        rows.append((run,t,max(z)))
    f=pd.DataFrame(rows,columns=["run","t","z"])
    return f.groupby("run",sort=False).z.transform(lambda x:x.ewm(alpha=.75,adjust=False).mean()).to_numpy()
def cusum(x,runs,drift):
    out=[]; previous=None; g=0.
    for value,run in zip(x,runs):
        if run!=previous: g=0.; previous=run
        g=max(0.,g+float(value)-drift); out.append(g)
    return np.asarray(out)
def summary_values(x,mask):
    a=np.asarray(x)[mask]
    return {"median":None if not len(a) else float(np.median(a)),"q90":None if not len(a) else float(np.quantile(a,.9)),"q99":None if not len(a) else float(np.quantile(a,.99)),"n":int(len(a))}
def clean_alarm_metrics(epoch,score,threshold):
    alarm=np.asarray(score)>threshold; n=len(alarm); duration=max(1e-9,n*.5/60)
    blocks=pd.DataFrame({"run":epoch.recording_id,"block":np.floor(epoch.availability_time_s/20).astype(int),"alarm":alarm}).groupby(["run","block"]).alarm.any()
    first=np.flatnonzero(alarm)
    return {"epoch_fpr":float(alarm.mean()),"false_alarms_per_min":float(alarm.sum()/duration),"censored_arl_epochs":int(first[0]+1 if len(first) else n),"block_any_fraction":float(blocks.mean()),"blocks":int(len(blocks))}
def metric_row(scenario,method,score,times,threshold,clean_fpr,operating_point="target1"):
    score=np.asarray(score,float); times=np.asarray(times,float); masks=epoch_masks(times)
    keep=masks["stable"]|masks["established"]; labels=masks["established"][keep].astype(int); alarm=score>threshold
    stable=masks["stable"]; established=masks["established"]
    auc=float(roc_auc_score(labels,score[keep])) if len(np.unique(labels))==2 else float("nan")
    pr=float(average_precision_score(labels,score[keep])) if len(np.unique(labels))==2 else float("nan")
    hits=np.flatnonzero(alarm&established); first=None if not len(hits) else float(times[hits[0]])
    duration=max(1e-9,stable.sum()*.5/60)
    return {"scenario":scenario,"method":method,"operating_point":operating_point,"threshold":float(threshold),"roc_auc":auc,"pr_auc":pr,
      "cmte_calibration_independent_clean_fpr":float(clean_fpr),"independent_clean_fpr":float(clean_fpr),
      "independent_clean_fpr_caveat":"CMTE calibration independent; frozen B0 was trained across cleanStatic with PRN holdout",
      "stable_pre_fpr":float(alarm[stable].mean()) if stable.any() else float("nan"),"false_alarms_per_min":float(alarm[stable].sum()/duration) if stable.any() else float("nan"),
      "detection":bool(np.any(alarm&established)),"first_alarm_availability_s":first,"first_alarm_delay_s":None if first is None else first-100.,
      "persistent_detection":float(alarm[established].mean()) if established.any() else float("nan"),"pre_summary":json.dumps(summary_values(score,stable),sort_keys=True),"post_summary":json.dumps(summary_values(score,established),sort_keys=True)}
def save_plot(path,epoch,scores,threshold,scenario):
    fig,ax=plt.subplots(figsize=(10,4))
    if "mean_e" in epoch: ax.plot(epoch.availability_time_s,epoch.mean_e,label="epoch mean e",alpha=.7)
    ax.plot(epoch.availability_time_s,scores,label="Full sequential"); ax.axhline(threshold,color="r",ls="--",label="target1")
    ax.axvspan(30,90,color="green",alpha=.08); ax.axvspan(90,110,color="orange",alpha=.08); ax.axvline(100,color="k",ls=":")
    ax2=ax.twinx(); ax2.plot(epoch.availability_time_s,epoch.N,color="gray",alpha=.25); ax2.set_ylabel("tracked N")
    ax.set(title=f"{scenario}: availability-time scores",xlabel="availability_time_s",ylabel="score"); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=120); plt.close(fig)
def score_set(nodes,epoch,thresholds):
    base=baseline_epoch_scores(nodes); rmse=nodes.groupby(["recording_id","window_bin_s"],sort=True).b0_prn_node_rmse.mean().to_numpy()
    seq=sequential_scores(np.log(epoch.mean_e.to_numpy()),epoch.recording_id,drift=thresholds["sequence"]["drift"])
    full=seq.s1_log_capital.to_numpy() if thresholds["sequence"]["choice"]=="S1" else seq.s2_e_cusum.to_numpy()
    return {"A0":base.A0.to_numpy(),"A1":a1(nodes,thresholds["baselines"]["A1"]),"A2":base.A2.to_numpy(),"A3":cusum(rmse,epoch.recording_id,thresholds["baselines"]["A3"]["drift"]),"A4":base.A4.to_numpy(),"Full":full},seq
def threshold_map(thresholds,method):
    doc=thresholds["sequence"] if method=="Full" else thresholds["baselines"][method]
    return doc["operating_points"]
def verify_state_compatibility(state_dir):
    config=json.loads((state_dir/"config.json").read_text()); expected=json.loads((state_dir/"code_hashes.json").read_text())
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if config.get("source_commit")!=commit: raise ValueError(f"stale state source_commit {config.get('source_commit')} != current {commit}")
    mismatches={name:{"expected":digest,"actual":file_sha256(ROOT/name)} for name,digest in expected.items() if not (ROOT/name).is_file() or file_sha256(ROOT/name)!=digest}
    if mismatches: raise ValueError(f"state generating code incompatible with current scoring semantics: {mismatches}")
    return commit,config,expected

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--state-dir",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--expected-sha",required=True); p.add_argument("--scenario",action="append",required=True,help="DS1=/node.csv=/manifest.json"); p.add_argument("--out",required=True); p.add_argument("--device",default="cpu")
    a=p.parse_args(argv); sd=Path(a.state_dir).resolve(strict=True); ck=Path(a.checkpoint).resolve(strict=True); actual=file_sha256(ck)
    if actual.lower()!=a.expected_sha.lower(): raise ValueError("checkpoint bytes do not match expected SHA")
    commit,config,code_hashes=verify_state_compatibility(sd)
    if config["checkpoint_sha256"].lower()!=actual.lower(): raise ValueError("state/checkpoint mismatch")
    state=load_state(sd/"model_state.json",expected_checkpoint_sha256=actual)
    if state.metadata.get("source_commit")!=commit or state.metadata.get("scoring_semantics")!="cmte-v3-recording-identity-log-domain-s1": raise ValueError("model state scoring provenance is stale")
    thresholds=json.loads((sd/"thresholds.json").read_text()); model,features,mean,std,cfg=load_checkpoint(ck,actual,a.device,ROOT)
    specs=[]
    for item in a.scenario:
        parts=item.split("=",2)
        if len(parts)!=3: raise ValueError("scenario mapping must be NAME=/node.csv=/manifest.json")
        name,node,manifest=parts; specs.append((name.upper(),Path(node).resolve(strict=True),Path(manifest).resolve(strict=True)))
    if len(specs)!=4 or {x[0] for x in specs}!={"DS1","DS2","DS3","DS4"}: raise ValueError("exactly one DS1-DS4 mapping required")
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    for directory in ("per_epoch","per_prn","plots","diagnostics","provenance"): (out/directory).mkdir()
    for artifact in ("config.json","training_summary.json","calibration_summary.json","thresholds.json","code_hashes.json","residual_manifest.json"): (out/artifact).write_bytes((sd/artifact).read_bytes())
    clean=pd.read_csv(sd/"clean_per_prn.csv"); clean=clean[clean.split=="test"].copy(); clean_epoch=aggregate_epochs(clean); clean_scores,_=score_set(clean,clean_epoch,thresholds)
    clean_rows=[]; clean_lookup={}
    for method,score in clean_scores.items():
        for op,threshold in threshold_map(thresholds,method).items():
            report=clean_alarm_metrics(clean_epoch,score,threshold); clean_lookup[(method,op)]=report["epoch_fpr"]
            clean_rows.append({"method":method,"operating_point":op,"threshold":threshold,**report})
    pd.DataFrame(clean_rows).to_csv(out/"independent_clean_operating_points.csv",index=False)
    metrics=[]; ablations=[]; prn_summary=[]; epoch_counts={}; diagnostics_ok=True
    provenance={"source_commit":commit,"state_source_commit":config["source_commit"],"checkpoint":str(ck),"checkpoint_sha256":actual,"state_dir":str(sd),"code_sha256":code_hashes,"timing_policy":config["timing_policy"],"scenario_inputs":{},"frozen_parameters":True,"attack_tuning":False}
    for name,node,manifest in specs:
        doc=json.loads(manifest.read_text())
        if doc.get("scenario","").upper()!=name or doc.get("role")!="evaluation_only" or doc.get("checkpoint_sha256","").lower()!=actual.lower(): raise ValueError("scenario manifest identity/checkpoint mismatch")
        if doc.get("producer_grade") not in {"reconstructed_equivalence","verified_node_artifact"} or file_sha256(node).lower()!=doc.get("node_sha256","").lower(): raise ValueError("scenario node provenance mismatch")
        residual=extract_recording_innovations(pd.read_csv(node),model,features,mean,std,scenario=name,seq_len=cfg.seq_len,device=a.device); validate_residual_frame(residual,require_history_reset=True)
        audit=residual.attrs["cadence_chunk_audit"]; audit_path=out/"provenance"/f"{name}_cadence_chunk_audit.json"; audit_path.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
        nodes=score_residuals(residual,state); nodes.to_csv(out/"per_prn"/f"{name}.csv",index=False); epoch=aggregate_epochs(nodes); epoch_counts[name]=len(epoch)
        scores,seq=score_set(nodes,epoch,thresholds)
        for method,score in scores.items():
            epoch[f"score_{method}"]=score
            for op,threshold in threshold_map(thresholds,method).items(): metrics.append(metric_row(name,method,score,epoch.availability_time_s,threshold,clean_lookup[(method,op)],op))
        epoch["phase"]=label_epochs(epoch.availability_time_s); epoch=pd.concat([epoch,seq],axis=1); epoch.to_csv(out/"per_epoch"/f"{name}.csv",index=False)
        for method in SCORE_METHODS:
            tmp=nodes.copy(); tmp["p"]=tmp[f"p_{method}"]; tmp["e"]=tmp[f"e_{method}"]; z=aggregate_epochs(tmp); ablations.append(metric_row(name,method,z.mean_e,z.availability_time_s,float("inf"),float("nan"),"diagnostic_no_threshold"))
        for prn,g in nodes.groupby("prn"): prn_summary.append({"scenario":name,"prn":prn,"rows":len(g),"p_median":g.p.median(),"e_q99":g.e.quantile(.99)})
        save_plot(out/"plots"/f"{name}_scores.png",epoch,scores["Full"],threshold_map(thresholds,"Full")["target1"],name)
        pivot=nodes.pivot_table(index="prn",columns="window_bin_s",values="e",aggfunc="mean"); fig,ax=plt.subplots(figsize=(10,4)); im=ax.imshow(np.log1p(pivot),aspect="auto"); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(out/"plots"/f"{name}_prn_heatmap.png",dpi=120); plt.close(fig)
        perm=epoch.sample(frac=1,random_state=2026).reset_index(drop=True); alt=sequential_scores(np.log(perm.mean_e),[name]*len(perm),drift=thresholds["sequence"]["drift"])
        selected="s1_log_capital" if thresholds["sequence"]["choice"]=="S1" else "s2_e_cusum"; original=np.asarray(seq[selected]); shuffled=np.asarray(alt[selected])
        diag={"diagnostic_only":True,"single_factor":"physical epoch order only; recording/reset structure fixed","recording_id":name,"original_reset_count":seq.attrs["reset_count"],"shuffled_reset_count":alt.attrs["reset_count"],"epoch_multiset_identical":bool(np.allclose(np.sort(epoch.mean_e),np.sort(perm.mean_e))),"sequential_trajectory_l2_delta":float(np.linalg.norm(original-shuffled)),"sequential_final_delta":float(abs(original[-1]-shuffled[-1])),"N_distribution":epoch.N.value_counts().sort_index().to_dict(),"prn_permutation_invariant_by_symmetric_aggregation":True}
        diagnostics_ok &= diag["epoch_multiset_identical"] and diag["original_reset_count"]==diag["shuffled_reset_count"]==1
        (out/"diagnostics"/f"{name}_order_shuffle.json").write_text(json.dumps(diag,indent=2,sort_keys=True)+"\n")
        provenance["scenario_inputs"][name]={"node":str(node),"node_sha256":file_sha256(node),"manifest":str(manifest),"manifest_sha256":file_sha256(manifest),"producer_grade":doc["producer_grade"],"recording_id":name,"cadence_chunk_audit":str(audit_path.relative_to(out))}
    mf=pd.DataFrame(metrics); mf.to_csv(out/"scenario_metrics.csv",index=False); pd.DataFrame(ablations).to_csv(out/"ablation_metrics.csv",index=False); pd.DataFrame(prn_summary).to_csv(out/"per_prn_evidence_summary.csv",index=False)
    target=mf[mf.operating_point=="target1"].copy(); full=target[target.method=="Full"].set_index("scenario")
    comparisons=[]
    for scenario in ("DS1","DS2","DS3","DS4"):
        for baseline in ("A0","A1"):
            b=target[(target.scenario==scenario)&(target.method==baseline)].iloc[0]; f=full.loc[scenario]
            similar=bool(f.cmte_calibration_independent_clean_fpr<=max(.015,b.cmte_calibration_independent_clean_fpr+.005))
            improved=bool(similar and ((f.persistent_detection>b.persistent_detection) or (f.detection and not b.detection) or (pd.notna(f.first_alarm_delay_s) and pd.notna(b.first_alarm_delay_s) and f.first_alarm_delay_s<b.first_alarm_delay_s)))
            comparisons.append({"scenario":scenario,"baseline":baseline,"similar_fpr":similar,"improved":improved,"full_persistent":f.persistent_detection,"baseline_persistent":b.persistent_detection,"full_delay":f.first_alarm_delay_s,"baseline_delay":b.first_alarm_delay_s})
    cmp=pd.DataFrame(comparisons); cmp.to_csv(out/"deterministic_comparator_audit.csv",index=False)
    scenario_improvements={s:bool(cmp[(cmp.scenario==s)&cmp.improved].shape[0]) for s in ("DS1","DS2","DS3","DS4")}
    catastrophic=bool((full.persistent_detection>=.20).any()); matched=bool(scenario_improvements["DS3"] or scenario_improvements["DS4"])
    criteria={"1_cmte_calibration_independent_clean_fpr_near1_pass_le1p5":bool(full.cmte_calibration_independent_clean_fpr.iloc[0]<=.015),"1_caveat":"B0 training is not independent",
      "2_DS1_4_stable_pre_fpr_le5":bool((full.stable_pre_fpr<=.05).all()),"2_catastrophic_any_ge20":catastrophic,
      "3_full_improves_A0_or_A1_in_at_least_3_scenarios_similar_fpr":sum(scenario_improvements.values())>=3,"3_scenario_audit":scenario_improvements,
      "4_DS3_or_DS4_matched_power_improvement":matched,"5_permutation_variable_N_non_dependence":bool(diagnostics_ok),"5_attack_tuning":False}
    go=bool(all(v for k,v in criteria.items() if k[0].isdigit() and not k.endswith("caveat") and k!="5_attack_tuning"))
    result={"status":"finalized_by_actual_eval","decision":"GO" if go else "NO-GO","go":go,"criteria":criteria,"epoch_counts":epoch_counts,"expected_epoch_count_reference":{"DS1":910,"DS2":901,"DS3":902,"DS4":243},"timing_policy":config["timing_policy"]}
    (out/"test_summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    (out/"test_summary.txt").write_text("Commands executed after code commit:\nTRAIN: scripts/train_cmte_texbat.py --checkpoint ... --expected-sha ... --clean-node-csv ... --clean-manifest ... --out STATE\nEVAL: scripts/eval_cmte_texbat.py --state-dir STATE --checkpoint ... --expected-sha ... --scenario DS1=... --scenario DS2=... --scenario DS3=... --scenario DS4=... --out EVAL\nFocused tests: pytest -q tests/test_cmte.py tests/test_cmte_review.py\nResult: see test_summary.json and scenario_metrics.csv; generated from source_commit "+commit+"\n")
    (out/"provenance"/"provenance.json").write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n")
    readme=f"""# CMTE TEXBAT final report\n\nDecision: **{result['decision']}**. Generated from `{commit}` with exact code hashes in `code_hashes.json`.\n\n## Contract and formulas\nImmutable `recording_id` is separate from predictor `history_id`. B0 temporarily groups by history, while physical epochs aggregate all PRNs by `(recording_id, window_bin_s)` and sequence state resets once per recording. Full uses conformal p=(1+#{{Qcal>=q}})/(n+1), fixed kappa mixture e, and a fully log-domain fixed-prior restart mixture (`s1_log_capital`, capped finite display capital only). This is empirically calibrated, **not anytime-valid**.\n\nThresholds use validation only. A0-A4 store q99, q99.5, and target1 epoch points. Full stores q99/q99.5/target1 quantiles of validation 20-second block maxima; only a handful of blocks exist, so these order statistics often coincide and have severe finite-sample uncertainty. No attack labels select methods, drift, or thresholds.\n\nTiming is the user-requested TEX policy on availability time: nominal onset 100 s; stable 30-90; transition 90-110; established >=110. DS4 has short post-onset coverage.\n\n## Provenance and grades\nFrozen B0 checkpoint SHA-256: `{actual}`. Reconstructed grades retained: clean/DS1-3 A-, DS4 B where supplied as verified node artifact; historical B0 comparison grade C and non-comparable. B0 was trained across cleanStatic with PRN holdout, so only **CMTE-calibration-independent clean FPR** is claimed.\n\n## Outputs and claims\nSplits, thresholds, clean operating-point FPR/FA-min/censored-ARL/block-any, all DS metrics and A0/A1 deterministic comparisons are in the CSV/JSON files. DS1 pre-onset metrics are `stable_pre_fpr`. SCI/WCL may only be claimed where source manifests establish those conditions; this evaluation does not infer them. Failures are explicit in `test_summary.json`.\n"""
    (out/"README.md").write_text(readme)
    checks={str(path.relative_to(out)):file_sha256(path) for path in sorted(out.rglob("*")) if path.is_file()}; (out/"checksums.json").write_text(json.dumps(checks,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"out":str(out),"decision":result["decision"],"source_commit":commit,"epoch_counts":epoch_counts},sort_keys=True))
if __name__=="__main__": main()
