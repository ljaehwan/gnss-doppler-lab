"""CRID metrics, block bootstrap and frozen verdict gates."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

def metrics(labels,scores,alarms)->dict:
 y=np.asarray(labels,int);s=np.asarray(scores,float);a=np.asarray(alarms,bool)
 return {"roc_auc":float(roc_auc_score(y,s)),"pauc_0_05":float(roc_auc_score(y,s,max_fpr=.05)),
  "pr_auc":float(average_precision_score(y,s)),"preonset_fpr":float(a[y==0].mean()),
  "attack_detection_rate":float(a[y==1].mean())}

def paired_block_bootstrap(left,right,blocks,seed=20260821,replicates=2000)->dict:
 left=np.asarray(left,float);right=np.asarray(right,float);blocks=np.asarray(blocks);unique=np.unique(blocks)
 paired=np.array([(left[blocks==b]-right[blocks==b]).mean() for b in unique]);rng=np.random.default_rng(seed)
 draw=np.array([rng.choice(paired,len(paired),replace=True).mean() for _ in range(replicates)])
 return {"estimate":float(paired.mean()),"ci95":[float(np.quantile(draw,.025)),float(np.quantile(draw,.975))],
  "block_count":len(unique),"status":"PASS" if len(unique)>=10 else "LIMITED"}

def verdict(gates:dict[str,bool],replay_alignment_ok:bool,positive_identifiable:bool,baseline_available:bool)->str:
 if not replay_alignment_ok:return "INCONCLUSIVE_RECEIVER_REPLAY_OR_ALIGNMENT"
 if not positive_identifiable:return "NO_GO_CRID_CLEAN_PHYSICAL_IDENTIFIABILITY"
 if all(gates.values()):return "GO_FOR_CRID_NEURAL_STAGE1" if baseline_available else "GO_PHYSICS_BASELINE_PENDING"
 return "NO_GO_CRID_COUNTERFACTUAL_INVARIANCE"
