#!/usr/bin/env python3
"""Deterministically regenerate AMCF-R1 C1--C9 and independent decisions."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path

def _rows(path):
 p=Path(path)
 if not p.is_file() or not p.read_text().strip():return []
 with p.open(newline="") as f:return list(csv.DictReader(f))
def _f(r,k):
 try:
  x=float(r.get(k,""));return x if math.isfinite(x) else None
 except (ValueError,TypeError):return None
def _pick(rows,model,scenario,op="q99"):
 return next((r for r in rows if r.get("model")==model and r.get("scenario")==scenario and r.get("operating_point")==op),None)
def _mean(v):return sum(v)/len(v) if v else None

def summarize(out:Path):
 out=Path(out);metrics=_rows(out/"metrics.csv");seeds=_rows(out/"seed_metrics.csv");policy=_rows(out/"query_policy_metrics.csv")
 try:audit=json.loads((out/"model_audit.json").read_text())
 except Exception:audit={}
 scenarios=("DS1","DS2","DS3","DS7","DS8");primary="primary 3-seed mean complex IG K7"
 clean=_pick(metrics,primary,"cleanStatic","q99");c1=clean is not None and _f(clean,"held_out_clean_fpr") is not None and _f(clean,"held_out_clean_fpr")<=.015
 pre=[]
 for sc in scenarios:
  r=_pick(metrics,primary,sc,"q99");x=_f(r or {},"stable_pre_fpr");pre.append(x is not None and x<.05)
 c2=len(pre)==5 and all(pre)
 # C3: complex/magnitude same architecture, all9, across seed means.
 wins=0;available=0;c4parts=[];c5parts=[]
 for sc in scenarios:
  cx=_pick(metrics,"ensemble::complex all9::policyNone",sc);mg=_pick(metrics,"ensemble::magnitude all9::policyNone",sc);pd=_pick(metrics,"ensemble::complex all9 phase-destroyed::policyNone",sc);ts=_pick(metrics,"ensemble::complex all9 temporal-shuffled::policyNone",sc)
  a=_f(cx or {},"roc_auc");b=_f(mg or {},"roc_auc");d=_f(pd or {},"roc_auc");t=_f(ts or {},"roc_auc")
  if a is not None and b is not None:available+=1;wins+=a>b
  c4parts.append(a is not None and d is not None and a>d);c5parts.append(a is not None and t is not None and a>t)
 c3=available==5 and wins>=4;c4=len(c4parts)==5 and all(c4parts);c5=len(c5parts)==5 and all(c5parts)
 # C6: each budget's IG mean scenario ROC beats fixed and random for >=2 model seeds.
 c6parts=[]
 for k in (5,7):
  good=[]
  for ms in (101,202,303):
   def vals(token):return [_f(r,"roc_auc") for r in seeds if (str(r.get("model_seed"))==str(ms) or f"::model{ms}::" in r.get("model","")) and token in r.get("model","") and f"K{k}" in r.get("model","") and r.get("scenario") in scenarios and _f(r,"roc_auc") is not None]
   ig=vals("complex IG");fx=vals("complex fixed");rn=vals("complex random")
   if ig and fx and rn:good.append(_mean(ig)>_mean(fx) and _mean(ig)>_mean(rn))
  c6parts.append(len(good)>=2 and all(good))
 c6=all(c6parts)
 ig_policy=[r for r in policy if " IG K" in r.get("model","")];modal=[_f(r,"modal_fraction") for r in ig_policy];c7=bool(modal) and all(x is not None and x<.95 for x in modal)
 # C8: matched-FPR primary materially beats B0 in >=3 scenarios without pre-FPR worsening >.01.
 material=0
 for sc in scenarios:
  a=_pick(metrics,primary,sc,"matched_clean_diagnostic");b=_pick(metrics,"B0 Exact",sc,"matched_clean_diagnostic")
  if not a or not b:continue
  ap,bp=_f(a,"post_detection"),_f(b,"post_detection");ar,br=_f(a,"roc_auc"),_f(b,"roc_auc");af,bf=_f(a,"stable_pre_fpr"),_f(b,"stable_pre_fpr")
  no_pre=af is not None and bf is not None and af-bf<=.01+1e-12
  gain=(ap is not None and bp is not None and ap-bp>=.02-1e-12) or (ar is not None and br is not None and ar-br>=.02-1e-12)
  material+=bool(no_pre and gain)
 c8=material>=3
 c9=True
 for sc in ("DS7","DS8"):
  r=_pick(metrics,primary,sc,"q99");c9 &= bool(r and _f(r,"roc_auc") is not None and _f(r,"roc_auc")>=.80 and _f(r,"post_detection") is not None and _f(r,"post_detection")>=.50)
 criteria={"C1":bool(c1),"C2":bool(c2),"C3":bool(c3),"C4":bool(c4),"C5":bool(c5),"C6":bool(c6),"C7":bool(c7),"C8":bool(c8),"C9":bool(c9)}
 primary_audit=all(isinstance(audit.get(f"complex_seed{x}"),dict) and audit[f"complex_seed{x}"].get("finite") is True and audit[f"complex_seed{x}"].get("best_restored") is True for x in (101,202,303))
 decisions={"Detector operating point":"GO" if c1 and c2 else "NO-GO","Complex":"GO" if c3 and c4 and c5 else "NO-GO","Active":"GO" if c6 and c7 else "NO-GO","WCL":"GO" if c1 and c2 and c8 and c9 and primary_audit else "NO-GO"}
 doc={"schema":"gnss-doppler-lab.amcf-r1-decision.v2","criteria":criteria,"decisions":decisions,"primary_source_audit":primary_audit,"C3_scenario_wins":wins,"C8_material_scenarios":material,"status":"developmental: DS1-3; post-exposure exploratory: DS7/8; DS4 NA","independence_note":"clean calibration/test are held-out chronological segments, not independent datasets"}
 (out/"decision.json").write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n",newline="\n")
 lines=["# AMCF-R1 campaign summary","",doc["status"],"","## Decisions"]+[f"- **{k}:** {v}" for k,v in decisions.items()]+["","## Exact criteria"]+[f"- {k}: {'PASS' if criteria[k] else 'FAIL'}" for k in criteria]+["","## Failure attribution","- **Front-end:** Prompt-gate utilization, phase/scenario QA, and DS1 scale-vs-missingness diagnostics are in `window_qa.json`.","- **Calibration:** C1/C2 and q99/q99.5 strict-`>` behavior isolate operating-point failures.","- **Model/representation:** C3-C5 isolate complex, phase, and temporal representation effects under the same architecture.","- **Selector:** C6/C7 isolate same-budget IG performance and path collapse; all budget claims are offline replay only.","- **Baseline/WCL:** C8 uses common timestamps and matched-clean-FPR diagnostics; primary q99 columns are never overwritten. C9 guards DS7/8 collapse.","","All thresholds use clean calibration only. Clean segments are chronological held-outs, never independent; q99.5 can be a second-maximum order statistic at small calibration N."]
 (out/"README.md").write_text("\n".join(lines)+"\n",newline="\n");return doc

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("out",type=Path);a=p.parse_args(argv);print(json.dumps(summarize(a.out),sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
