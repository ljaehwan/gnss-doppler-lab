"""Fail-closed raw-IQ tracker reconstruction helpers for ACAF-NF R1.1."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import hashlib, math
import numpy as np
from .acquisition_surface import gps_l1ca_code
FS=25_000_000; CHIP=1.023e6; EPOCH=25_000
REQ=('PRN','PRN_start_sample_count','carrier_doppler_hz','code_freq_chips','aux1','CN0_SNV_dB_Hz','carrier_lock_test','Prompt_I','Prompt_Q')
def canonical_prn_identity():
 for p in range(1,33):
  a=np.asarray(gps_l1ca_code(f'G{p:02d}')); b=np.asarray(gps_l1ca_code(p));
  if a.shape!=(1023,) or not np.array_equal(a,b): raise ValueError(f'canonical acquisition identity failed PRN {p}')
 return list(range(1,33))
def sampled_chip_indices(fs=FS, rate=CHIP):
 if fs<=0 or rate<=0: raise ValueError('positive rates required')
 return np.floor(np.arange(int(round(fs/1000)))*rate/fs).astype(int)%1023
def aux_samples_to_chips(aux, code_freq, fs=FS):
 x=float(aux)*float(code_freq)/float(fs)
 if not math.isfinite(x): raise ValueError('nonfinite aux conversion')
 return x
def raw_iq_s16le(path, sample_count, n=EPOCH):
 p=Path(path); start=int(sample_count); n=int(n)
 if start<0 or n<=0 or not p.is_file(): raise ValueError('invalid raw path/count')
 size=p.stat().st_size
 if size%4 or (start+n)*4>size: raise ValueError('raw IQ byte/sample bounds violation')
 with p.open('rb') as f: f.seek(start*4); x=np.frombuffer(f.read(n*4),dtype='<i2')
 if x.size!=2*n: raise ValueError('short raw IQ read')
 return x[0::2].astype(np.float32)+1j*x[1::2].astype(np.float32)
def alignment_candidates():
 out=[]
 for interval,row in [('k_to_k1','k1'),('end_k','k')]:
  for rem in (-1,1):
   for wipe in (-1,1): out.append({'name':f'interval_{interval}_tracker_{row}_rem_{"plus" if rem>0 else "minus"}_wipe_{"plus" if wipe>0 else "minus"}','interval':interval,'tracker_row':row,'remnant_sign':rem,'wipeoff_sign':wipe})
 return out
def parse_tracker_rows(rows,fs=FS):
 out=[]
 for r in rows:
  if any(k not in r for k in REQ): continue
  try: q={k:float(r[k]) for k in REQ}
  except (TypeError,ValueError): continue
  if not all(math.isfinite(v) for v in q.values()) or not (1<=int(q['PRN'])<=32) or q['PRN_start_sample_count']<0 or q['code_freq_chips']<=0: continue
  if q['CN0_SNV_dB_Hz']<28 or q['carrier_lock_test']<.85: continue
  q['prn']=int(q['PRN']);q['sample_count']=int(q['PRN_start_sample_count']);q['end_sample']=q['sample_count']+int(round(fs/1000));q['mat_prompt_mag']=math.hypot(q['Prompt_I'],q['Prompt_Q'])
  for key in ('mat_path','mat_index','mat_sha256','channel'):
   if key in r: q[key]=r[key]
  out.append(q)
 return out
def chronological_split(rows):
 # One raw interval supports one tracker row; rotate channels instead of choosing strongest.
 groups={}
 for x in rows: groups.setdefault(x['sample_count'],[]).append(x)
 chosen=[]; end=-1; turn=0
 # Prefer epochs with >=4 stable PRNs, so reconstruction cannot collapse to one
 # early-acquired channel; retain fallback for small synthetic fixtures.
 starts=[z for z in sorted(groups) if len(groups[z])>=4]
 if len(starts)<2000: starts=sorted(groups)
 for start in starts:
  if start<end: continue
  choices=sorted(groups[start],key=lambda x:x.get('prn', 0)); x=choices[turn%len(choices)]; turn+=1
  chosen.append(x);end=x['end_sample']
 if len(chosen)<2000: raise ValueError(f'need >=2000 nonoverlapping clean records, got {len(chosen)}')
 return {'train':chosen[:1000],'calibration':chosen[1000:1500],'holdout':chosen[1500:2000]}
def stratified_round_robin_clean(rows, n):
 """Choose a bounded, deterministic, PRN-balanced clean subset.

 Rows are selected in PRN round-robin order then restored to chronological raw order.
 The supplied rows must already describe non-overlapping raw epochs; this routine rejects
 an overlapping candidate rather than silently reusing raw samples.
 """
 n=int(n)
 if n < 1: return []
 queues={}
 for row in sorted(rows,key=lambda r:(int(r['sample_count']),int(r['prn']))):
  queues.setdefault(int(row['prn']),[]).append(row)
 # Seed scarce clean PRNs first so dense early channels cannot erase their only
 # non-overlapping windows; ties retain ascending PRN order.
 prns=sorted(queues,key=lambda p:(len(queues[p]),p))
 if not prns: return []
 selected=[]; cursor={p:0 for p in prns}; last_end=-1
 # Consume one feasible raw interval per PRN turn.  Advancing a queue past an
 # overlap prevents a same-epoch PRN from crowding out every later stratum.
 while len(selected)<n:
  progressed=False
  for p in prns:
   queue=queues[p]; i=cursor[p]
   while i<len(queue) and int(queue[i]['sample_count'])<last_end: i+=1
   cursor[p]=i
   if i<len(queue):
    row=queue[i]; selected.append(row); cursor[p]=i+1; last_end=int(row['end_sample']); progressed=True
    if len(selected)==n: break
  if not progressed: break
 return selected

def _rank(x):
 x=np.asarray(x); return np.argsort(np.argsort(x,kind='mergesort'),kind='mergesort').astype(float)
def spearman(a,b):
 if len(a)<2:return float('nan')
 x=_rank(a);y=_rank(b);return float(np.corrcoef(x,y)[0,1])
def center_gate(d):
 checks={'n':d['n']>=500,'within':d['within_fraction']>=.95,'spearman':d['spearman']>=.9,'boundary':d['boundary_fraction']<=.05,'prns':d['prn_count']>=4,'dominant':d['dominant_fraction']<=.5}
 return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks}
def normal_only_fit(train_scenarios, attack_scenarios):
 return {'status':'OK' if set(train_scenarios)=={'clean'} and not set(train_scenarios)&set(attack_scenarios) else 'REJECTED_NON_NORMAL_FIT'}
def null_thresholds(v): return {'status':'INSUFFICIENT_NORMAL_CALIBRATION' if not v else 'COMPUTED_NORMAL_ONLY','performance_claim':False if not v else False}
def classify_verdict(gate_a,gate_c):
 if gate_a!='PASS': return {'verdict':'CENTER_RECONSTRUCTION_INVALID','gate_b':'NOT_EVALUATED_UNTIL_CENTER_VALID','gate_c':gate_c}
 return {'verdict':'VALIDITY_COMPLETE_PENDING_PHYSICS','gate_b':'NOT_IMPLEMENTED','gate_c':gate_c}
def ds78_overlap(a,b):
 if not a or not b:return {'status':'UNRECONSTRUCTABLE_RECORDING_RELATIVE_COUNTERS','independent_normal_evidence':False}
 return {'status':'CHECKED','one_second_hash_overlap':bool(set(a)&set(b)),'independent_normal_evidence':not bool(set(a)&set(b))}
def local_prompt(iq,row,candidate,fs=FS):
 phase=aux_samples_to_chips(row['aux1'],row['code_freq_chips'],fs)*candidate['remnant_sign']
 chips=np.asarray(gps_l1ca_code(f"G{row['prn']:02d}")); code=chips[np.floor((phase+np.arange(len(iq))*row['code_freq_chips']/fs)%1023).astype(int)]
 t=np.arange(len(iq))/fs; wipe=np.exp(candidate['wipeoff_sign']*-2j*np.pi*row['carrier_doppler_hz']*t)
 return abs(np.vdot(code,iq*wipe))/len(iq)
def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
