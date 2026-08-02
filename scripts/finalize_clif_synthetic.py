#!/usr/bin/env python3
"""Finalize R4 artifacts from live bundles; no campaign counts are hard-coded."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
REQUIRED=("config.json","synthetic_run_manifest.csv","synthetic_bundle_ledger.csv","generation_summary.json","impairment_distribution.json","training_summary.json","predictor_comparison.csv","scenario_metrics.csv","domain_gap_metrics.csv","domain_gap_group_summaries.csv","alignment_destruction_metrics.json","alignment_destruction_raw_metrics.csv","evaluation_provenance.json","provenance_manifest.json","test_summary.txt","README.md")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def git(*args):return subprocess.check_output(["git",*args],text=True).strip()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"))
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("artifacts/clif_ip_synthetic_normal_r4"));a=ap.parse_args();root=a.root;(root/"plots").mkdir(exist_ok=True)
 idx=pd.read_csv(root/"synthetic_run_manifest.csv");imps=[json.loads(x) for x in idx.impairments_json];dist={k:{"min":min(float(x[k]) for x in imps),"max":max(float(x[k]) for x in imps)} for k in imps[0] if isinstance(imps[0][k],(int,float)) and not isinstance(imps[0][k],bool)}
 (root/"impairment_distribution.json").write_text(json.dumps({"axes":dist,"attack":False,"spoofing":False},indent=2)+"\n")
 # Repository-sized ledger verifies every externally retained run feature bundle.
 ledger=[]
 for row in idx.itertuples(index=False):
  rd=root/"runs"/row.run_id;m=json.loads((rd/"manifest.json").read_text());recv=rd/"receiver/manifest.json";b0=rd/"b0_nodes.csv";m1=rd/"m1_features.csv"
  if len(pd.read_csv(b0))!=int(m["b0_rows"]) or len(pd.read_csv(m1))!=int(m["m1_rows"]):raise RuntimeError(f"row-count mismatch {row.run_id}")
  rec={"run_id":row.run_id,"domain":row.domain,"split":row.split,"b0_rows":int(m["b0_rows"]),"m1_rows":int(m["m1_rows"]),"b0_sha256":sha(b0),"m1_sha256":sha(m1),"run_manifest_sha256":sha(rd/"manifest.json"),"iq_sha256":m["iq_sha256"],"iq_source_sha256":m["generator"]["transform"]["source_sha256"],"generator_sha256":m["generator"]["binary_sha256"],"receiver_manifest_sha256":sha(recv),"receiver_config_sha256":sha(rd/"receiver/receiver.conf"),"tracked_rows":int(m["receiver"]["tracking_rows"])}
  rec["leaf_sha256"]=hashlib.sha256(canonical(rec).encode()).hexdigest();ledger.append(rec)
 ld=pd.DataFrame(ledger).sort_values("run_id");ld.to_csv(root/"synthetic_bundle_ledger.csv",index=False);merkle=hashlib.sha256("".join(ld.leaf_sha256).encode()).hexdigest()
 counts={"success_runs":int(len(ld)),"b0_rows":int(ld.b0_rows.sum()),"m1_rows":int(ld.m1_rows.sum()),"tracked_rows":int(ld.tracked_rows.sum())}
 cfg={"schema":"clif-ip.synthetic-normal.r4.final.v3","final_runs":counts["success_runs"],"per_domain":idx.groupby(["domain","split"]).size().unstack(fill_value=0).to_dict("index"),"duration_s":float(idx.duration_s.unique()[0]),"tap_layout":["E4","E3","E2","E","P","L","L2","L3","L4"],"tap_semantics":"Method-A prompt-relative magnitudes; signed vectors are x-xhat innovations","regimes":{"R0_OAK":"immutable historical R3 only; noncomparable scope","R0_TEX":"real-only cleanStatic train 0-240, validation 250-330, test >=340","S0":"synthetic only","S1":"S0 B0 fine-tune; frozen M1 normalization/PCA with real-clean AR/distribution adaptation; residual Ridge corrections on S0 P1-P3"},"evaluation":{"score_time":"available_s","OAK":{"nominal_onset_s":120,"stable_pre":"t<110","excluded":"110<=t<130","established_post":"t>=130"},"TEX":{"nominal_onset_s":100,"stable_pre":"30<=t<90","excluded":"90<=t<110","established_post":"t>=110"}},"destruction":{"replicates":199,"block_epochs":8,"p_value_resolution":.005,"models":["P2","P3","Full"],"actual_9d_rescoring":True}}
 (root/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
 s=pd.read_csv(root/"scenario_metrics.csv");q=s[(s.regime.isin(["R0","S0","S1"]))&s.model.eq("Full")&pd.to_numeric(s.roc_auc,errors="coerce").notna()&s.comparison_scope.eq("r4_same_protocol")].copy();q["roc_auc"]=pd.to_numeric(q.roc_auc)
 fig,ax=plt.subplots(figsize=(10,4));q.assign(key=q.regime+" "+q.target_domain+" "+q.scenario).plot.bar(x="key",y="roc_auc",legend=False,ax=ax);ax.set_ylim(0,1);ax.set_ylabel("ROC-AUC");fig.tight_layout();fig.savefig(root/"plots/scenario_full_auc.png",dpi=150);plt.close(fig)
 d=pd.read_csv(root/"domain_gap_metrics.csv");fig,ax=plt.subplots(figsize=(9,4));d.assign(key=d.regime+" "+d.target_domain+" "+d.feature_group).plot.bar(x="key",y="rmse_ratio",legend=False,ax=ax);ax.axhline(5.5,color="r",ls="--");ax.set_ylabel("actual/synthetic RMSE ratio");fig.tight_layout();fig.savefig(root/"plots/domain_gap_rmse_ratio.png",dpi=150);plt.close(fig)
 training=json.loads((root/"training_summary.json").read_text());evalprov=json.loads((root/"evaluation_provenance.json").read_text());code={x:sha(Path(x)) for x in ("src/gnss_doppler_lab/clif_ip_synthetic.py","scripts/train_clif_synthetic.py","scripts/eval_clif_synthetic.py","scripts/finalize_clif_synthetic.py","tests/test_clif_synthetic.py")}
 manifest={"schema":"clif-ip.synthetic-normal.r4.provenance.v3","finalized_utc":datetime.now(timezone.utc).isoformat(),"source_commit_parent":git("rev-parse","HEAD"),"source_tree":git("write-tree"),"code_sha256":code,"campaign":counts,"bundle_ledger_sha256":sha(root/"synthetic_bundle_ledger.csv"),"bundle_merkle_sha256":merkle,"evaluation_inputs":evalprov["inputs"],"training_sources":{d:training["regimes"]["S1"][d]["allowed_source_hashes"] for d in ("SYN-OAK","SYN-TEX")},"immutability":{"R3_modified":False,"R0_OAK_comparison_scope":"historical_r3_noncomparable"},"clone_completeness":"Clone includes ledger and reported artifacts but not approximately 3 GB of ignored per-run feature bundles; verify retained local bundles against ledger.","manifest_self_excluded_from_checksums":True}
 (root/"provenance_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 missing=[x for x in REQUIRED if not (root/x).is_file()]
 if missing:raise SystemExit(f"missing required artifacts: {missing}")
 files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name not in {"checksums.json","provenance_manifest.json"} and "/runs/" not in str(p) and "/smoke/" not in str(p)}
 (root/"checksums.json").write_text(json.dumps({"generated_utc":datetime.now(timezone.utc).isoformat(),"source_commit_parent":git("rev-parse","HEAD"),"source_tree":git("write-tree"),"manifest_self_excluded":True,"files":files},indent=2)+"\n")
if __name__=="__main__":main()
