#!/usr/bin/env python3
"""Run Stage-0-R1 with hash-bound tracker centers and raw-IQ complex CAFs.

No attack labels enter fit/calibration/query selection.  If lineage gates fail, the
only valid decision is EXPERIMENT_BLOCKED_<reason>; descriptive scores are tagged
NOT_FOR_DECISION rather than converted into a NO-GO claim.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, subprocess
from pathlib import Path
import h5py, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gnss_doppler_lab.acaf_nf_stage0_r1 import *

FS=25_000_000; SEED=20260806
RAW=Path('/home/ubuntu/unraid_hdd/texbat/raw')
HASH_LOG=Path('/home/ubuntu/acaf_r1_logs/raw_full_sha256.txt')
ROOTS={
 'cleanStatic':'/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/receiver/cleanStatic-complex9',
 'ds1':'/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/receiver/ds1-complex9',
 'ds2':'/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/receiver/ds2-complex9',
 'ds3':'/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/receiver/ds3-complex9',
 'ds4':None,
 'ds7':'/home/ubuntu/ssd_data/gnss-early-detection/artifacts/ds7-sealed-input/receiver/ds7-complex9',
 'ds8':'/home/ubuntu/ssd_data/gnss-early-detection/artifacts/cmte-a2-ds8-complex-d67f813/receiver',
}
ONSET={'ds1':125.0,'ds2':110.1,'ds3':118.9,'ds4':113.8,'ds7':110.0,'ds8':110.0}
PULL={'ds1':155.,'ds2':145.,'ds3':195.,'ds4':225.,'ds7':150.,'ds8':150.}
REQ=('PRN','PRN_start_sample_count','carrier_doppler_hz','code_freq_chips','aux1','CN0_SNV_dB_Hz','carrier_lock_test')

def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def writecsv(p,rows):
 fields=sorted({k for r in rows for k in r}) if rows else ['status']
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def raw_hashes():
 if not HASH_LOG.exists(): return {}
 out={}
 for line in HASH_LOG.read_text().splitlines():
  parts=line.split(maxsplit=1)
  if len(parts)==2: out[Path(parts[1]).stem]=parts[0]
 return out
def tracker_files(scenario):
 root=ROOTS.get(scenario)
 return sorted(Path(root,'raw').glob('epl_tracking_ch_*.mat')) if root and Path(root,'raw').is_dir() else []
def tracker_digest(scenario):
 files=tracker_files(scenario)
 return [{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in files]
def nearest_center(scenario,target_s):
 candidates=[]
 for p in tracker_files(scenario):
  with h5py.File(p,'r') as f:
   if not all(k in f for k in REQ): continue
   n=len(f['PRN']); ss=np.asarray(f['PRN_start_sample_count']).reshape(-1); time=ss/FS
   cn=np.asarray(f['CN0_SNV_dB_Hz']).reshape(-1); lk=np.asarray(f['carrier_lock_test']).reshape(-1)
   ok=(cn>=28)&(lk>=0.85)&np.isfinite(time)
   if not np.any(ok): continue
   ix=np.where(ok)[0][np.argmin(abs(time[ok]-target_s))]
   row={k:np.asarray(f[k]).reshape(-1)[ix].item() for k in REQ}
   row['mat_path']=str(p);row['tracker_time_s']=float(time[ix]);row['time_error_s']=float(abs(time[ix]-target_s));row['scenario']=scenario
   candidates.append(row)
 if not candidates: return None
 # prefer actual time then C/N0
 candidates.sort(key=lambda r:(r['time_error_s'],-float(r['CN0_SNV_dB_Hz'])))
 return candidates[0]
def observation(scenario,target_s):
 row=nearest_center(scenario,target_s)
 if row is None: return {'scenario':scenario,'target_s':target_s,'status':'NO_TRACKER_CENTER'},None,None
 try: center=tracker_center_from_row(row,FS); x=raw_iq_epoch(str(RAW/(scenario+'.bin')),center.sample_count,FS)
 except Exception as exc: return {'scenario':scenario,'target_s':target_s,'status':'RAW_CENTER_ERROR:'+type(exc).__name__},None,None
 c=caf_complex_grid(x,center,FS); cn,anchor=normalize_complex_caf(c)
 peak=np.unravel_index(np.abs(c).argmax(),c.shape); edge=peak[0] in (0,c.shape[0]-1) or peak[1] in (0,c.shape[1]-1)
 r={**row,'prn':center.prn,'sample_count':center.sample_count,'code_phase_chips':center.code_phase_chips,'code_rate_hz':center.code_rate_hz,'center_doppler_hz':center.carrier_doppler_hz,'status':'OK','caf_peak_delay_chip':float(CA_DELAY_CHIPS[peak[1]]),'caf_peak_doppler_offset_hz':float(CA_DOPPLER_HZ[peak[0]]),'caf_peak_on_boundary':bool(edge),'caf_center_to_peak_cells':int(abs(peak[0]-len(CA_DOPPLER_HZ)//2)+abs(peak[1]-len(CA_DELAY_CHIPS)//2)),**anchor,'raw_power':float(np.mean(abs(x)**2))}
 return r,cn,c
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='artifacts/acaf_nf_stage0_static_r1');args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'plots').mkdir(exist_ok=True)
 hashes=raw_hashes(); missing_hash=[s for s in ['cleanStatic','ds1','ds2','ds3','ds4','ds7','ds8'] if s not in hashes]
 tracker_info={s:tracker_digest(s) for s in ROOTS}; missing_tracker=[s for s,v in tracker_info.items() if not v]
 config={'stage':'ACAF-NF Stage-0-R1','seed':SEED,'raw_format':'little-endian interleaved signed int16 IQ','sample_rate_hz':FS,'coherent_ms':1,'chip_rate_hz':GPS_CA_CHIP_RATE_HZ,'grid':{'delay_chips':CA_DELAY_CHIPS.tolist(),'doppler_hz':CA_DOPPLER_HZ.tolist()},'center_contract':['PRN','PRN_start_sample_count','carrier_doppler_hz','code_freq_chips','aux1'],'query_unit':'one complex delay-Doppler coordinate','normal_only':True,'attack_labels_used_for_fit_calibration_threshold_or_query':False};dump(out/'config.json',config)
 manifest={'raw_full_sha256':hashes,'raw_hash_log':str(HASH_LOG),'tracker_full_sha256':tracker_info,'tracker_roots':ROOTS,'missing_raw_full_hash':missing_hash,'missing_tracker':missing_tracker};dump(out/'data_manifest.json',manifest)
 # 0.5 s chronological clean roles, deliberately far from any attack evaluation data.
 role_times={'train':[30+i*.5 for i in range(50)],'calibration':[55+i*.5 for i in range(30)],'holdout':[70+i*.5 for i in range(30)]}
 split={'recording':'cleanStatic','chronological':True,'roles':role_times,'n_requested':{k:len(v) for k,v in role_times.items()},'attack_data_used':False,'ds78_exact_sample_overlap_provenance':'NOT_AVAILABLE_FROM_SEPARATE_RECORDING_COUNTERS'};dump(out/'normal_split.json',split)
 scenario_meta={s:{'official_candidate_onset_s':ONSET[s],'transition_start_s':ONSET[s],'transition_end_s':PULL[s],'established_start_s':PULL[s],'processed_time_origin':'tracker PRN_start_sample_count / 25 MHz, recording-relative','time_origin_verified_against_official_document':False} for s in ONSET};dump(out/'scenario_metadata.json',scenario_meta)
 rows=[]; fields=[]; originals={}
 for role,times in role_times.items():
  for t in times:
   r,cn,c=observation('cleanStatic',t);r.update({'role':role,'phase':'normal','requested_time_s':t});rows.append(r)
   if cn is not None: fields.append((role,cn,r)); originals[('cleanStatic',t)]=c
 for s in ONSET:
  times=[max(2.,ONSET[s]-30+i*5) for i in range(6)]+[ONSET[s]+i*2 for i in range(11)]+[PULL[s]+i*3 for i in range(10)]
  for t in times:
   r,cn,c=observation(s,t);phase='pre_onset' if t<ONSET[s] else ('transition' if t<PULL[s] else 'established');r.update({'role':'external','phase':phase,'requested_time_s':t});rows.append(r)
   if cn is not None: fields.append(('external',cn,r)); originals[(s,t)]=c
 valid_clean=[(cn,r) for role,cn,r in fields if role=='train']
 blockers=[]
 if missing_hash: blockers.append('MISSING_FULL_RAW_SHA_'+'_'.join(missing_hash))
 if missing_tracker: blockers.append('MISSING_TRACKER_'+'_'.join(missing_tracker))
 try:
  import pandas  # Native B0 scorer dependency; do not substitute frozen CSV results.
 except ModuleNotFoundError:
  blockers.append('NATIVE_B0_SAME_EPOCH_PANDAS_DEPENDENCY_MISSING')
 if len(valid_clean)<20: blockers.append('INSUFFICIENT_CONTIGUOUS_CLEAN_TRACKER_CAF')
 # A center at the CAF grid boundary is a failed local-center recovery, not evidence.
 clean_ok=[r for r in rows if r.get('role') in {'train','calibration','holdout'} and r.get('status')=='OK']
 boundary_frac=np.mean([r['caf_peak_on_boundary'] for r in clean_ok]) if clean_ok else 1.0
 if boundary_frac>.20: blockers.append('TRACKER_CENTER_CAF_BOUNDARY_RATE_GT_20PCT')
 h0=None
 if not blockers:
  h0=robust_h0_fit(np.stack([complex_vector(cn) for cn,r in valid_clean]))
  for role,cn,r in fields: r['raw_complex_caf_norm']=float(np.linalg.norm(cn));r['normalized_h0_score']=h0_score(complex_vector(cn),h0);r.update(two_source_same_prn_fit(originals[(r['scenario'],r['requested_time_s'])],CA_DELAY_CHIPS,CA_DOPPLER_HZ))
 else:
  for r in rows: r['normalized_h0_score']=None
 cal=[r['normalized_h0_score'] for r in rows if r.get('role')=='calibration' and r.get('normalized_h0_score') is not None]
 thresholds={'source':'cleanStatic chronological calibration only','calibration_n':len(cal),'q99':float(np.quantile(cal,.99)) if cal else None,'q99_5':float(np.quantile(cal,.995)) if cal else None,'target_fpr_1pct':float(np.quantile(cal,.99)) if cal else None};dump(out/'thresholds.json',thresholds)
 writecsv(out/'per_epoch_scores.csv',rows)
 diag=[{k:r.get(k) for k in ['scenario','requested_time_s','tracker_time_s','prn','single_residual','two_residual','bic_improvement','second_delay_chip','second_doppler_hz','second_amplitude_ratio','grid_boundary','status']} for r in rows];writecsv(out/'two_source_diagnostics.csv',diag)
 # Query selection runs only from clean train fields.  Coordinate indices remain complex cells, never real/imag scalars.
 budgets=[]
 if h0:
  x=np.stack([complex_vector(cn) for cn,r in valid_clean]); shape=(len(CA_DOPPLER_HZ),len(CA_DELAY_CHIPS))
  for k in (3,5,9,16):
   chosen=select_complex_coordinates(x,shape,k); budgets.append({'method':'clean_only_variance_selected','K':k,'complex_coordinate_indices':chosen,'coordinate_count':len(chosen),'status':'COMPUTED_CLEAN_ONLY'})
   for method,coords in [('EPL',list(range(min(k,3)))),('fixed_9tap',list(range(min(k,9)))),('random',np.random.default_rng(SEED+k).choice(shape[0]*shape[1],k,replace=False).tolist()),('shuffled',list(reversed(chosen)))]: budgets.append({'method':method,'K':k,'complex_coordinate_indices':coords,'coordinate_count':len(coords),'status':'DESCRIPTIVE_ONLY_BLOCKED' if blockers else 'READY_FOR_COMMON_SUPPORT_EVAL'})
 else:
  for k in (3,5,9,16): budgets.append({'method':'all','K':k,'complex_coordinate_indices':[],'coordinate_count':0,'status':'BLOCKED'})
 writecsv(out/'budget_metrics.csv',budgets)
 # Raw IQ controls use one clean raw record and the repaired center. They remain actual raw->CAF calculations.
 controls=[]; cleanref=next((r for r in rows if r.get('role')=='train' and r.get('status')=='OK'),None)
 if cleanref:
  center=TrackerCenter(cleanref['prn'],cleanref['code_phase_chips'],cleanref['center_doppler_hz'],cleanref['code_rate_hz'],cleanref['sample_count']);x=raw_iq_epoch(str(RAW/'cleanStatic.bin'),center.sample_count,FS)
  rng=np.random.default_rng(SEED)
  for gain in (.5,.8,1.2,2.):
   for phase in (0.,np.pi/2,np.pi,3*np.pi/2):
    y=x*gain*np.exp(1j*phase); f,_=normalize_complex_caf(caf_complex_grid(y,center,FS)); controls.append({'kind':'gain_phase_raw_iq','gain':gain,'phase_rad':phase,'normalized_distance_to_reference':float(np.linalg.norm(f-fields[0][1]))})
  base=np.std(x.real)
  for ratio in (.05,.15,.30):
   y=x+ratio*base*(rng.normal(size=len(x))+1j*rng.normal(size=len(x)))/np.sqrt(2); f,_=normalize_complex_caf(caf_complex_grid(y,center,FS));controls.append({'kind':'awgn_raw_iq','relative_sigma':ratio,'normalized_distance_to_reference':float(np.linalg.norm(f-fields[0][1]))})
 dump(out/'physical_controls.json',{'controls':controls,'status':'ACTUAL_RAW_IQ_REREAD_CAF' if controls else 'BLOCKED'})
 metrics=[]
 for s in ONSET:
  rs=[r for r in rows if r.get('scenario')==s]
  metrics.append({'scenario':s,'n_rows':len(rs),'n_ok':sum(r.get('status')=='OK' for r in rs),'pre_onset_score_mean':np.mean([r['normalized_h0_score'] for r in rs if r.get('phase')=='pre_onset' and r.get('normalized_h0_score') is not None]) if h0 else None,'transition_score_mean':np.mean([r['normalized_h0_score'] for r in rs if r.get('phase')=='transition' and r.get('normalized_h0_score') is not None]) if h0 else None,'established_score_mean':np.mean([r['normalized_h0_score'] for r in rs if r.get('phase')=='established' and r.get('normalized_h0_score') is not None]) if h0 else None,'evaluation_status':'BLOCKED' if blockers else 'VALID'})
 writecsv(out/'scenario_metrics.csv',metrics)
 dump(out/'bootstrap_results.json',{'block_seconds':10,'status':'BLOCKED' if blockers else 'NOT_IMPLEMENTED','reason':blockers or 'must be computed on common 10-second epoch blocks'})
 verdict='EXPERIMENT_BLOCKED_'+('_'.join(blockers) if blockers else 'UNIMPLEMENTED_DECISION_GATE')
 dump(out/'go_no_go.json',{'execution_validity':'BLOCKED' if blockers else 'VALID','blockers':blockers,'verdict':verdict,'no_go_claim_made':False,'boundary_fraction_clean':float(boundary_frac),'full_raw_sha_complete':not bool(missing_hash)})
 # Actual figures, never placeholder text.
 ok=[r for r in rows if r.get('status')=='OK']
 plt.figure(figsize=(10,4));
 for s in sorted(set(r['scenario'] for r in ok)):
  q=[r for r in ok if r['scenario']==s];plt.scatter([r['requested_time_s'] for r in q],[r.get('raw_power') for r in q],s=8,label=s)
 plt.legend(ncol=4);plt.xlabel('recording-relative tracker target s');plt.ylabel('raw IQ mean power');plt.tight_layout();plt.savefig(out/'plots'/'actual_raw_power_by_scenario.png',dpi=140);plt.close()
 plt.figure(figsize=(8,4));plt.hist([r.get('caf_center_to_peak_cells',0) for r in clean_ok],bins=20);plt.xlabel('center-to-CAF-peak Manhattan cells');plt.ylabel('clean epochs');plt.tight_layout();plt.savefig(out/'plots'/'clean_center_recovery.png',dpi=140);plt.close()
 if controls:
  plt.figure(figsize=(8,4));plt.scatter(range(len(controls)),[c['normalized_distance_to_reference'] for c in controls]);plt.ylabel('normalized CAF distance');plt.xlabel('raw IQ control trial');plt.tight_layout();plt.savefig(out/'plots'/'actual_gain_phase_awgn_controls.png',dpi=140);plt.close()
 readme=f'''# ACAF-NF Stage-0-R1\n\nThis is a static-only raw-IQ CAF foundation repair. It corrects the prior 1023/fs code-rate and negative G2 indexing errors: GPS L1 C/A uses **1.023e6 chips/s**, and ICD G2 taps are one-based positive stages. Every stored CAF is reread from s16 complex raw IQ and centered with tracker PRN, PRN_start_sample_count, carrier_doppler_hz, code_freq_chips, and aux1. Query budgets count complex CAF coordinates, not real/imag scalars. Two-source diagnostics use shifts of one same-PRN CAF template.\n\n**Execution validity:** {"BLOCKED" if blockers else "VALID"}. **Verdict:** `{verdict}`. A blocked R1 is neither PHYSICS_NO_GO nor a detection claim. Missing full raw hashes, tracker coverage, center recovery, common-support bootstrap, or same-epoch native B0 validity must be repaired before GO/NO-GO. Existing B0 is not reported as comparable here because this R1 run has not completed its same-epoch native B0 input/score gate.\n''';(out/'README.md').write_text(readme)
 files=[p for p in out.rglob('*') if p.is_file() and p.name!='checksums.json'];dump(out/'checksums.json',{'algorithm':'sha256','files':{str(p.relative_to(out)):sha(p) for p in files}})
 print(json.dumps({'verdict':verdict,'rows':len(rows),'blockers':blockers}))
if __name__=='__main__': main()
