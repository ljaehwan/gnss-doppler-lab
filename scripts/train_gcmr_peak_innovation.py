#!/usr/bin/env python3
"""Train/freeze GCMR-PI on OAKBAT cleanStatic only (fail-closed campaign runner)."""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess
from pathlib import Path
import numpy as np, torch
from gnss_doppler_lab.gcmr_experiment import DEFAULT_ROLES, select_role_events, validate_roles
from gnss_doppler_lab.gcmr_peak_innovation_adapter import aggregate_peak_windows, build_event_record, CausalEventBuildError
from gnss_doppler_lab.gcmr_peak_innovation_pipeline import GCMRPeakInnovationPipeline
from gnss_doppler_lab.tracking_peaks import available_tracking_prns, load_receiver_tracking_peak_series_segments
from run_gcmr_oakbat_poc import SCENARIOS, TIMING, load_scenario

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def windows(events): return sorted({(float(e.window_start_s),float(e.window_end_s)) for e in events})
def records(events, root, history):
 ws=windows(events); peaks={}
 for p in available_tracking_prns(root):
  rows=[]
  for s in load_receiver_tracking_peak_series_segments(root,p,tap_count=3): rows.extend(aggregate_peak_windows(s,ws,min_epochs=1))
  peaks[p]=rows
 out=[]; rejected=[]
 for e in events:
  try: out.append(build_event_record(e,peaks,history_window=history))
  except (CausalEventBuildError,ValueError) as x: rejected.append({'start':float(e.window_start_s),'end':float(e.window_end_s),'reason':str(x)})
 return out,rejected
def score(pipe, events, seed): return [pipe.score_attack(e,destruction_seed=seed+i) for i,e in enumerate(events)]
def emit_scores(path, scores):
 fields=['time','N','S_common','N_eff','loading_count','S_pair','energy','scalar_rmse','relation_destruction','A0','A1','A2','A3','A4','Full']
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for x in scores:
   d=x.diagnostics;w.writerow({'time':x.time,'N':x.n,'S_common':d.s_common,'N_eff':d.n_eff,'loading_count':d.loading_count,'S_pair':d.s_pair,'energy':d.energy,'scalar_rmse':d.scalar_rmse,'relation_destruction':x.destroyed_pair_score,**x.scores})
def metrics(scores, thresholds): return {a:{q:{'threshold':t,'events':len(scores),'alarms':int(sum(x.scores[a]>t for x in scores)),'rate':float(np.mean([x.scores[a]>t for x in scores])) if scores else None} for q,t in qs.items()} for a,qs in thresholds.items()}
def plots(out,scores):
 import matplotlib.pyplot as plt
 if not scores:return
 t=np.array([x.time for x in scores]); series={'score':[x.scores['Full'] for x in scores],'S_common':[x.diagnostics.s_common for x in scores],'N_eff':[x.diagnostics.n_eff for x in scores],'S_pair':[x.diagnostics.s_pair for x in scores],'N':[x.n for x in scores],'PRN_loading':[x.diagnostics.loading_count for x in scores]}
 for n,y in series.items():
  fig,ax=plt.subplots();ax.plot(t,y);ax.set(xlabel='window end (s)',ylabel=n);fig.tight_layout();fig.savefig(out/f'{n}.png',dpi=120);plt.close(fig)
def manifest(out):
 ps=sorted(p for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS');(out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in ps))
def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,default=Path('artifacts/gcmr_peak_innovation'));ap.add_argument('--history',type=int,default=4);ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--seed',type=int,default=7);ap.add_argument('--force-cache',action='store_true');ap.add_argument('--open-attacks',action='store_true');a=ap.parse_args(argv)
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);validate_roles()
 try:
  clean,meta,_=load_scenario('cleanStatic',out/'cache',a.force_cache);roles={r.name:select_role_events(clean,r) for r in DEFAULT_ROLES};need=('train','selection_val','clean_reference','event_calibration')
  if any(not roles[k] for k in need):raise RuntimeError('missing cleanStatic temporal role events')
  built={};reject={}
  for k,v in roles.items():built[k],reject[k]=records(v,SCENARIOS['cleanStatic'],a.history)
  if any(not built[k] for k in need):raise RuntimeError('GCMR relation events cannot be joined to real three-tap E/P/L windows')
  pipe=GCMRPeakInnovationPipeline(a.history,epochs=a.epochs,seed=a.seed).fit_normal(built['train'],built['selection_val']);cal=score(pipe,built['event_calibration'],a.seed)
  thresholds={ab:{q:float(np.quantile([x.scores[ab] for x in cal],v)) for q,v in {'q99':.99,'q995':.995,'FPR1':.99}.items()} for ab in ('A0','A1','A2','A3','A4','Full')};held=score(pipe,built['sealed_held'],a.seed);emit_scores(out/'score_summary.csv',held);plots(out,held);torch.save(pipe,out/'model.pt')
  result={'cleanStatic_sealed':metrics(held,thresholds),'rejected':{k:len(v) for k,v in reject.items()},'normal_only':True}
  if a.open_attacks:
   for name in ('os1','os2','os3','os4'):
    ev,_,_=load_scenario(name,out/'cache',a.force_cache);b,r=records(ev,SCENARIOS[name],a.history);s=score(pipe,b,a.seed);emit_scores(out/f'{name}_scores.csv',s);result[name]={'metrics':metrics(s,thresholds),'rejected':len(r),'attack_gate':'explicit; inference only'}
  (out/'thresholds.json').write_text(json.dumps(thresholds,indent=2)+'\n');(out/'scenario_metrics.json').write_text(json.dumps(result,indent=2)+'\n');(out/'ablations.json').write_text(json.dumps({'A0':'scalar RMSE','A1':'binomial tail diagnostic','A2':'energy','A3':'S_common','A4':'S_pair','Full':'normal-calibrated combination excludes btail'},indent=2)+'\n');(out/'training_summary.json').write_text(json.dumps({'normal_source':'OAKBAT cleanStatic only','roles':{k:len(v) for k,v in built.items()},'seed':a.seed,'epochs':a.epochs,'geometry':meta['geometry_preflight']},indent=2,default=str)+'\n');(out/'config.json').write_text(json.dumps({'history':a.history,'timing':TIMING,'open_attacks':a.open_attacks,'contract':'real E/P/L only; PRN labels are joins only'},indent=2)+'\n')
 except Exception as e:
  (out/'blocker_evidence.json').write_text(json.dumps({'status':'actual_campaign_not_run','exception_type':type(e).__name__,'exception':str(e),'requirement':'No numerical campaign metric was fabricated. Synthetic smoke is separately available.'},indent=2)+'\n');print(f'BLOCKED: {e}')
 manifest(out);return 0
if __name__=='__main__':raise SystemExit(main())
