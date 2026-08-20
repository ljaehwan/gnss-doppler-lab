#!/usr/bin/env python3
"""Score frozen raw-IQ physical controls before any attack replay."""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from gnss_doppler_lab.crid import CONFIG_ORDER,load_response,score_aligned
from gnss_doppler_lab.crid_experiment import fit_domain
from run_crid_stage0 import ART,SSD

def load(root):return {c:load_response(c,(root/c).glob("trace_native_1ms_ch_*.bin")) for c in CONFIG_ORDER}
def main():
 p=argparse.ArgumentParser();p.add_argument("--dataset",choices=("oak_clean","tex_clean"),required=True);a=p.parse_args();domain="OAK" if a.dataset.startswith("oak") else "TEX"
 model,delays,split,thresholds,clean,audit=fit_domain(load(SSD/"replays"/a.dataset));rows=[];root=SSD/"control_replays"/a.dataset
 for control in sorted(root.iterdir() if root.exists() else []):
  scores=score_aligned(load(control),model,delays);values=np.array([r["score"] for r in scores]);positive=control.name.startswith("second_source")
  alarm=float(np.mean(values>thresholds["q99"])) if len(values) else 1.;status=("PASS" if (alarm>=.70 if positive else alarm<=.05) else "FAIL")
  rows.append({"domain":domain,"control":control.name,"kind":"positive" if positive else "negative","status":status,"score":float(np.median(values)) if len(values) else None,"alarm_ratio":alarm,"epoch_count":len(values)})
 path=ART/"physical_control_metrics.csv";prior=[]
 if path.exists():
  with path.open() as f:prior=[r for r in csv.DictReader(f) if r["domain"]!=domain]
 with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=["domain","control","kind","status","score","alarm_ratio","epoch_count"]);w.writeheader();w.writerows(prior+rows)
 print({"domain":domain,"controls":len(rows),"all_negative_pass":all(r["status"]=="PASS" for r in rows if r["kind"]=="negative"),"any_positive_pass":any(r["status"]=="PASS" for r in rows if r["kind"]=="positive")})
if __name__=="__main__":main()
