#!/usr/bin/env python3
"""Finalize R4 config, plots, provenance and checksums after tests."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
REQUIRED=("config.json","synthetic_run_manifest.csv","generation_summary.json","impairment_distribution.json","training_summary.json","predictor_comparison.csv","scenario_metrics.csv","domain_gap_metrics.csv","alignment_destruction_metrics.json","alignment_destruction_raw_metrics.csv","provenance_manifest.json","test_summary.txt","README.md")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def git(*args):return subprocess.check_output(["git",*args],text=True).strip()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("artifacts/clif_ip_synthetic_normal_r4"));a=ap.parse_args();root=a.root;(root/"plots").mkdir(exist_ok=True)
 idx=pd.read_csv(root/"synthetic_run_manifest.csv");imps=[json.loads(x) for x in idx.impairments_json];dist={k:{"min":min(float(x[k]) for x in imps),"max":max(float(x[k]) for x in imps)} for k in imps[0] if isinstance(imps[0][k],(int,float)) and not isinstance(imps[0][k],bool)}
 (root/"impairment_distribution.json").write_text(json.dumps({"axes":dist,"attack":False,"spoofing":False},indent=2)+"\n")
 cfg={"schema":"clif-ip.synthetic-normal.r4.final.v2","final_runs":60,"per_domain":{"train":24,"validation":3,"synthetic_test":3},"duration_s":120,"tap_layout":["E4","E3","E2","E","P","L","L2","L3","L4"],"tap_semantics":"Method-A prompt-relative magnitudes; signed vectors are x-xhat innovations","regimes":{"R0":"read-only exact R3 OAK; TEX unavailable with explicit reason","S0":"synthetic only; native synthetic validation threshold","S1":"S0 initialization plus real clean 0-240 adaptation; real clean 250-330 threshold; >=340 test"},"attacks":{"OAKBAT":{"scenarios":["os1","os2","os3","os4"],"onset_s":120},"TEXBAT":{"scenarios":["DS1","DS2","DS3","DS4"],"nominal_onset_s":100,"stable_pre_s":[30,90],"excluded_s":[90,110],"post_start_s":110}},"permutations":199,"permutation_block_epochs":8,"p_value_resolution":.005}
 (root/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
 s=pd.read_csv(root/"scenario_metrics.csv");q=s[(s.regime.isin(["S0","S1"]))&s.model.eq("Full")&pd.to_numeric(s.roc_auc,errors="coerce").notna()].copy();q["roc_auc"]=pd.to_numeric(q.roc_auc)
 fig,ax=plt.subplots(figsize=(9,4));q.assign(key=q.regime+" "+q.target_domain+" "+q.scenario).plot.bar(x="key",y="roc_auc",legend=False,ax=ax);ax.set_ylim(0,1);ax.set_ylabel("ROC-AUC");fig.tight_layout();fig.savefig(root/"plots/scenario_full_auc.png",dpi=150);plt.close(fig)
 d=pd.read_csv(root/"domain_gap_metrics.csv");fig,ax=plt.subplots(figsize=(8,4));d.assign(key=d.regime+" "+d.target_domain+" "+d.feature_group).plot.bar(x="key",y="rmse_ratio_5p5x",legend=False,ax=ax);ax.axhline(5.5,color="r",ls="--");ax.set_ylabel("actual/synthetic RMSE ratio");fig.tight_layout();fig.savefig(root/"plots/domain_gap_rmse_ratio.png",dpi=150);plt.close(fig)
 training=json.loads((root/"training_summary.json").read_text());manifest={"schema":"clif-ip.synthetic-normal.r4.provenance.v2","finalized_utc":datetime.now(timezone.utc).isoformat(),"source_commit_before_r4_finalize":git("rev-parse","HEAD"),"source_tree_sha256":hashlib.sha256(git("diff","--binary","HEAD").encode()).hexdigest(),"campaign":{"success_runs":int(idx.run_id.nunique()),"b0_rows":83202,"m1_rows":14400,"attack_rows_in_generation":0},"training_sources":{d:training["regimes"]["S1"][d]["allowed_source_hashes"] for d in ("SYN-OAK","SYN-TEX")},"immutability":{"R3_modified":False,"R0_mode":"read-only"},"unavailable":{"R0_TEX":"no complete chronological compatible Method-A node training export","S0_S1_TEX_DS1_DS3":"compatible Method-A attack node rows absent; values not fabricated"},"manifest_self_excluded_from_checksums":True}
 (root/"provenance_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 missing=[x for x in REQUIRED if not (root/x).is_file()]
 if missing:raise SystemExit(f"missing required artifacts: {missing}")
 files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name not in {"checksums.json","provenance_manifest.json"} and "/runs/" not in str(p) and "/smoke/" not in str(p)}
 (root/"checksums.json").write_text(json.dumps({"generated_utc":datetime.now(timezone.utc).isoformat(),"source_commit":git("rev-parse","HEAD"),"manifest_self_excluded":True,"files":files},indent=2)+"\n")
if __name__=="__main__":main()
