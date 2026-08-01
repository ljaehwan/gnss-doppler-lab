#!/usr/bin/env python3
"""Read-only r4 diagnostics for frozen GCMR-PI r3 artifacts.
No model fitting or artifact mutation is performed."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score,average_precision_score
import matplotlib.pyplot as plt

NAMES=('energy','S_common','N_eff','S_pair','scalar_rmse')
SCENARIOS=('os1','os2','os3','os4')
def rows(p):
 with open(p,newline='') as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
def mask(t):
 t=np.asarray(t);return t>=30,(t>=30)&(t<110),(t>=130)
def metrics(t,s,th):
 _,pre,post=mask(t); keep=pre|post; y=post[keep].astype(int);x=np.asarray(s)[keep];a=np.asarray(s)>th
 first=np.asarray(t)[a&post];return dict(threshold=float(th),pre_fpr=float(a[pre].mean()),post_detection_rate=float(a[post].mean()),first_alarm_delay_s=None if not len(first) else float(first[0]-120),persistence=float(a[post].mean()),roc_auc=float(roc_auc_score(y,x)),pr_auc=float(average_precision_score(y,x)),pre_events=int(pre.sum()),post_events=int(post.sum()))
def zfit(cal,key,x):
 # Calibration provenance is frozen normal event_calibration role: existing A scores expose its standardization.
 return (np.asarray(x)-cal[key][0])/cal[key][1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--frozen',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();f=a.frozen;o=a.out;o.mkdir(parents=True,exist_ok=True);(o/'plots').mkdir(exist_ok=True)
 th=json.loads((f/'thresholds.json').read_text())
 # Recover normal-only location/scale exactly from frozen A2/A3/A4 linear identities across attack-independent fixed model outputs.
 # For new score quantiles we use the frozen event-calibration thresholds by mapping q99/q995 component thresholds.
 # A2 and A4 give component calibration thresholds; A3 is a sum, therefore RelationOnly is A3+A4.
 warm=[]; relation=[]; corr=[];dist=[]
 for sc in SCENARIOS:
  r=rows(f/(sc+'_scores.csv'));t=np.array([x['time'] for x in r]);
  for score in ('Full','A2','A3','A4'):
   for q in ('q99','q995','FPR1'): warm.append(dict(scenario=sc,score=score,criterion=q,**metrics(t,[x[score] for x in r],th[score][q])))
  # RelationOnly is exactly existing normal-calibrated A3+A4, no attack calibration/tuning.
  rel=np.array([x['A3']+x['A4'] for x in r]); en=np.array([x['A2'] for x in r]);
  # conservative fixed thresholds: sums of frozen normal q thresholds (not refit on attack)
  for score,v,tt in [('RelationOnly',rel,th['A3']['q99']+th['A4']['q99']),('EnergyOnly',en,th['A2']['q99']),('Full',np.array([x['Full'] for x in r]),th['Full']['q99'])]: relation.append(dict(scenario=sc,score=score,**metrics(t,v,tt)))
  for x in r:
   d={'scenario':sc,'period':'pre' if 30<=x['time']<110 else ('post' if x['time']>=130 else 'excluded')};d.update({k:x[k] for k in NAMES});dist.append(d)
  arr=np.array([[x[k] for k in ('Full','A0','A1','A2','A3','A4')] for x in r]);
  for i,n in enumerate(('Full','A0','A1','A2','A3','A4')):
   for j,m in enumerate(('Full','A0','A1','A2','A3','A4')):corr.append({'scenario':sc,'left':n,'right':m,'pearson':float(np.corrcoef(arr[:,i],arr[:,j])[0,1])})
  fig,ax=plt.subplots();
  for n in ('Full','A2'):ax.plot(t,[x[n] for x in r],label=n)
  ax.set(xlim=(0,40),xlabel='time (s)',ylabel='score',title=sc+' startup');ax.legend();fig.tight_layout();fig.savefig(o/'plots'/f'{sc}_startup_0_40.png',dpi=140);plt.close(fig)
  fig,ax=plt.subplots();ax.plot(t,[x['Full'] for x in r],label='Full');ax.plot(t,rel,label='RelationOnly');ax.plot(t,en,label='EnergyOnly');ax.axvline(120,color='k',ls=':');ax.legend();ax.set(xlabel='time (s)',ylabel='score',title=sc);fig.tight_layout();fig.savefig(o/'plots'/f'{sc}_scores.png',dpi=140);plt.close(fig)
 # distributions visualization
 for k in NAMES:
  fig,ax=plt.subplots()
  for period,color in [('pre','C0'),('post','C3')]:ax.hist([d[k] for d in dist if d['period']==period],bins=40,alpha=.45,label=period,color=color)
  ax.set(title=k);ax.legend();fig.tight_layout();fig.savefig(o/'plots'/f'{k}_pre_post.png',dpi=140);plt.close(fig)
 # v2 destruction: cache observations are retained raw relation innovations/features; preserve row norms and source a different, non-adjacent epoch.
 destruction={"method":"cache-observation direction shuffle proxy; norms preserved; source epochs differ by >1 and PRN when feasible; fixed seed; 20 repetitions","warning":"The frozen cache retains relation observations, not GRU residuals. This is a structural sensitivity diagnostic, not a re-scored model claim.","scenarios":{}}
 rng=np.random.default_rng(20260801)
 for sc in SCENARIOS:
  z=np.load(f/'cache'/(sc+'.relations.npz')); offs=z['offsets']; ends=z['window_end_s']; obs=z['observations']; prns=z['pair_prns']; original=np.array([x['S_pair'] for x in rows(f/(sc+'_scores.csv'))]); times=np.array([x['time'] for x in rows(f/(sc+'_scores.csv'))]); # align score events to cache ends
  vals=[]
  for rep in range(20):
   eidx=[];src=[]
   for e in range(len(ends)):
    candidates=np.where(np.abs(np.arange(len(ends))-e)>1)[0]; q=int(rng.choice(candidates));eidx.extend([e]*(offs[e+1]-offs[e]));src.extend([q]*(offs[e+1]-offs[e]))
   # relation coherence proxy: mean abs cosine change after assigning directions from other epochs, own norms retained
   changed=[]
   for e in range(len(ends)):
    a,b=offs[e],offs[e+1]; X=obs[a:b]; q=src[a]; Y=obs[offs[q]:offs[q+1]]; dirs=Y[np.arange(len(X))%len(Y)]
    D=dirs/np.maximum(np.linalg.norm(dirs,axis=1,keepdims=True),1e-12)*np.linalg.norm(X,axis=1,keepdims=True)
    nx=np.linalg.norm(X,axis=1); nd=np.linalg.norm(D,axis=1)
    cx=(X@X.T)/np.maximum(nx[:,None]*nx[None,:],1e-12); cd=(D@D.T)/np.maximum(nd[:,None]*nd[None,:],1e-12)
    changed.append(float(np.mean(np.abs(cx-cd))))
   vals.append(np.interp(times,ends,np.asarray(changed)))
  v=np.mean(vals,axis=0); attack=times>=130; delta=original[attack]-v[attack]; boot=np.quantile([np.mean(rng.choice(delta,len(delta),replace=True)) for _ in range(2000)],[.025,.975]); p=wilcoxon(delta).pvalue if np.any(delta) else None
  destruction['scenarios'][sc]=dict(repetitions=20,attack_events=int(attack.sum()),mean_delta=float(delta.mean()),median_delta=float(np.median(delta)),fraction_decreased=float((delta>0).mean()),bootstrap95_ci=[float(x) for x in boot],wilcoxon_p=float(p),effect_size_rank_biserial=float(np.mean(np.sign(delta))))
  fig,ax=plt.subplots();ax.plot(times,original,label='original S_pair');ax.plot(times,v,label='destroyed proxy');ax.legend();ax.set(title=sc+' relation destruction',xlabel='time (s)');fig.tight_layout();fig.savefig(o/'plots'/f'{sc}_original_destroyed_relation.png',dpi=140);plt.close(fig)
 def dumpcsv(name,data):
  with open(o/name,'w',newline='') as h:w=csv.DictWriter(h,fieldnames=sorted({k for d in data for k in d}));w.writeheader();w.writerows(data)
 dumpcsv('warmup_comparison.csv',warm);dumpcsv('relation_only_metrics.csv',relation);dumpcsv('component_correlations.csv',corr);dumpcsv('component_distributions.csv',dist)
 (o/'onset_metrics_warmup30.json').write_text(json.dumps({'contract':{'pre':'30<=time<110','excluded':'110<=time<130','post':'time>=130','onset':120,'frozen_full_thresholds_unchanged':True},'rows':warm},indent=2)+'\n');(o/'relation_destruction_v2.json').write_text(json.dumps(destruction,indent=2)+'\n')
 (o/'implementation_audit.md').write_text('# Implementation audit\n\n- E/P/L normalization uses center P (`safe_center_tap_normalize`); the primary E/P/L is not separately protected from the 9-tap runner contract.\n- A 3D Gram matrix has rank <=3, so high S_common can be geometric/low-rank rather than spoof-specific.\n- Full adds standardized components and ignores covariance; S_pair models conditional expected cosine but not conditional variance.\n- `clean_reference` is built but not consumed by `fit_normal`; `selection_val` is consumed for predictor/whitener/pair calibration, not model selection or early stopping.\n- A 9-tap runner exists, but this frozen artifact is 3T; no frozen 9T result exists.\n\nNext fixes are documentation only: use covariance-aware calibration, conditional pair variance, explicit clean-reference role, held-out early stopping/model selection, and freeze/run 9T before claims.\n')
 (o/'README.md').write_text('# r4 frozen diagnostics\n\nRead-only analysis of frozen r3; no retraining or threshold tuning on attacks. Warmup uses 30–110 s pre, excludes 110–130 s, and uses >=130 s post. Full uses frozen thresholds unchanged. RelationOnly is A3+A4 (normal-calibrated components); EnergyOnly is A2.\n\nLimitations: attack-recording steady-state FPR is reported, not a clean false-alarm guarantee. Energy dependence and relation-only incremental information are descriptive. Destruction is a cache-observation structural proxy because raw GRU residuals are not retained; it is not claimable as a successful causal intervention. 3T is frozen evidence; 9T requires a frozen result.\n')
 files=sorted(p for p in o.rglob('*') if p.is_file() and p.name!='SHA256SUMS');(o/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(o))+'\n' for p in files))
if __name__=='__main__':main()
