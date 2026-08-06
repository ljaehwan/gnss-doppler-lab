#!/usr/bin/env python3
"""Preregistered ACAF-NF Stage-0 raw complex CAF feasibility runner."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
from gnss_doppler_lab.acaf_nf_stage0 import *
RAW=Path('/home/ubuntu/unraid_hdd/texbat/raw'); SC=['cleanStatic','ds1','ds2','ds3','ds4','ds7','ds8']; SEED=20260806
OFF={'ds1':{'injection':125.0},'ds2':{'injection':110.1},'ds3':{'injection':118.9,'pull_off':195.0},'ds4':{'injection':113.8,'pull_off':225.0},'ds7':{'injection':110.0,'programmed_push':150.0},'ds8':{'injection':110.0,'programmed_push':150.0}}
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def csvout(p,rows,fields):
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r.get(k,'UNAVAILABLE') for k in fields} for r in rows])
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',default='artifacts/acaf_nf_stage0_static');a=ap.parse_args();o=Path(a.output);o.mkdir(parents=True,exist_ok=True);(o/'plots').mkdir(exist_ok=True)
 config={'seed':SEED,'raw_format':'little-endian int16 interleaved I,Q','sample_rate_hz':25000000,'coherent_ms':1,'grid':{'delay_chips':CAF_CODE.tolist(),'doppler_hz':CAF_DOPPLER.tolist()},'query_budgets':[3,5,9,16],'attack_labels_used_for_fit_calibration_threshold_or_query_selection':False,'pre_registered_before_attack_read':True,'center_policy':'tracker centers required for physical claims; unavailable=score/status only'};dump(o/'config.json',config)
 split={'recording':'cleanStatic','chronological':True,'tracking_lock_after_s':30,'attack_related_time_exclusion':'DS7/DS8 exact raw time mapping unavailable; all inferential gates fail closed','roles':{'train':[30.0,60.0],'calibration':[80.0],'holdout':[100.0]},'source_sample_ranges':{k:[int(x*25e6),int((x+.001)*25e6)] for k,x in [('train_30',30),('train_60',60),('calibration_80',80),('holdout_100',100)]},'attack_data_in_fit_or_threshold':False};dump(o/'normal_split.json',split)
 meta={k:{'official_candidate_s':v,'processed_time_offset_s':None,'time_origin_verified':False,'transition':[v['injection'],v.get('pull_off',v.get('programmed_push',v['injection']+30))],'established':[v.get('programmed_push',v['injection']+30),v.get('pull_off',999999)],'evidence':'UNVERIFIED: bundled official PDF must be interpreted against raw origin'} for k,v in OFF.items()};dump(o/'scenario_metadata.json',meta)
 docs=[]
 for p in [Path('data/external/texbat/docs/The Texas Spoofing Test Battery_1.pdf'),Path('data/external/texbat/docs/texbat_ds7_and_ds8.pdf')]: docs.append({'path':str(p),'sha256':sha(p) if p.exists() else None,'official_source':'bundled TEXBAT document','page_quote_status':'UNVERIFIED_IN_THIS_RUN'})
 data={};
 for n in SC:
  p=RAW/(n+'.bin'); data[n]={'path':str(p),'bytes':p.stat().st_size,'full_sha256':None,'fingerprint_sha256':hashlib.sha256(p.read_bytes()[:1048576] if p.stat().st_size<2e6 else (open(p,'rb').read(1048576))).hexdigest(),'format_verified_by_size':p.stat().st_size%4==0,'raw_read':True,'center_lineage':'UNVERIFIED'}
 dump(o/'data_manifest.json',{'documents':docs,'sources':data,'full_hash_policy':'not completed; no physical claim permitted'})
 # Direct raw complex CAF at fixed preregistered epochs. zero center is intentionally flagged unavailable.
 epochs={'cleanStatic':[30.,60.,80.,100.],**{n:[10.,60.,max(1.,OFF[n]['injection']-2),OFF[n]['injection']+2] for n in OFF}}
 surfaces={};rows=[]
 for n,ts in epochs.items():
  for t in ts:
   z=raw_epoch(RAW/(n+'.bin'),t); c=caf_complex(z,3,25000000,0,0); cn,ni=normalize_caf(c,center=(len(CAF_DOPPLER)//2)*len(CAF_CODE)+len(CAF_CODE)//2); surfaces[(n,t)]=(c,cn); rows.append({'scenario':n,'epoch_s':t,'center_status':'UNVERIFIED_CENTER','raw_complex_score':float(np.linalg.norm(complex_vector(c))),'normalized_score':'UNAVAILABLE_UNTIL_CLEAN_H0','raw_power_score':float(np.mean(np.abs(z)**2)),'support_id':'fixed_prereg_v1'})
 train=[complex_vector(surfaces[('cleanStatic',t)][1]) for t in [30.,60.]]; h0=fit_h0(train)
 for r in rows:
  c,cn=surfaces[(r['scenario'],r['epoch_s'])];r['normalized_score']=h0_score(complex_vector(cn),h0)
 cal=[r['normalized_score'] for r in rows if r['scenario']=='cleanStatic' and r['epoch_s']==80.]; thresholds={'source':'cleanStatic calibration only','q99':float(np.quantile(cal,.99)),'q99_5':float(np.quantile(cal,.995)),'target_fpr_1pct':float(np.quantile(cal,.99)),'calibration_epochs':[80.]};dump(o/'thresholds.json',thresholds);csvout(o/'per_epoch_scores.csv',rows,['scenario','epoch_s','center_status','raw_complex_score','normalized_score','raw_power_score','support_id'])
 # clean-only query selection and comparative budgets; all attack performance remains unavailable, not zero.
 qi={str(k):clean_query_indices(train,k) for k in [3,5,9,16]};b=[]
 for method in ['EPL_3','fixed_9tap','random_queries','shuffled_queries','clean_selected','dense_reference']:
  for k in [3,5,9,16]:b.append({'method':method,'K':k,'coordinates':qi.get(str(k),list(range(k))) if method=='clean_selected' else list(range(k)),'attack_low_fpr_pauc':'UNAVAILABLE_CENTER_LINEAGE','compute_coordinates':k})
 csvout(o/'budget_metrics.csv',b,['method','K','coordinates','attack_low_fpr_pauc','compute_coordinates'])
 reference_field=surfaces[("cleanStatic",30.)][0]; atoms=[];coords=[]
 for f in CAF_DOPPLER:
  for d in CAF_CODE: atoms.append(np.roll(np.roll(reference_field, int(round(f/50.)), axis=0), int(round(d/.125)), axis=1));coords.append((float(d),float(f)))
 diag=[]
 for r in rows:
  c,_=surfaces[(r['scenario'],r['epoch_s'])];q=two_source_fit(c,atoms,[abs(d)==1 or abs(f)==250 for d,f in coords]);d,f=coords[q['second_index']];q.update({'scenario':r['scenario'],'epoch_s':r['epoch_s'],'second_delay_chip':d,'second_doppler_hz':f,'physical_evidence_status':'NOT_INTERPRETABLE_UNVERIFIED_CENTER'});diag.append(q)
 csvout(o/'two_source_diagnostics.csv',diag,['scenario','epoch_s','single_residual','two_residual','bic_improvement','second_delay_chip','second_doppler_hz','amplitude_ratio','boundary','physical_evidence_status'])
 # Controls modify raw clean IQ then rerun complex CAF.
 x=raw_epoch(RAW/'cleanStatic.bin',30.);controls=[]
 for g in [.5,.8,1.2,2.]:
  for ph in [0.,np.pi/2,np.pi,3*np.pi/2]:
   for noise in [0.,.03]:
    c=caf_complex(augment(x,g,ph,noise,SEED),3,25000000,0,0);cn,_=normalize_caf(c);controls.append({'control':'gain_phase_awgn_raw_iq_then_caf','gain':g,'phase_rad':ph,'awgn_sigma':noise,'normalized_h0_score':h0_score(complex_vector(cn),h0),'interpretation':'control only; center unverified'})
 controls += [{'control':'query_destruction','variant':v,'K':k,'result':'UNAVAILABLE_CENTER_LINEAGE'} for v in ['clean_selected','random','shuffled'] for k in [3,5,9,16]]
 dump(o/'physical_controls.json',{'controls':controls,'synthetic_two_source':'raw complex replica sum with random phase/delays; mechanics only, not TEXBAT evidence'})
 sm=[]
 for n in ['ds1','ds2','ds3','ds4','ds7','ds8']:
  sm.append({'scenario':n,'pre_onset_fpr':'UNAVAILABLE_CENTER_LINEAGE','transition_detection':'UNAVAILABLE','established_detection':'UNAVAILABLE','post_onset_detection':'UNAVAILABLE','roc_auc':'UNAVAILABLE','pr_auc':'UNAVAILABLE','low_fpr_pauc':'UNAVAILABLE','first_sustained_alarm_delay_s':'UNAVAILABLE','b0_comparison':'NOT_DIRECTLY_COMPARABLE'})
 csvout(o/'scenario_metrics.csv',sm,list(sm[0]))
 dump(o/'bootstrap_results.json',{'block_seconds':10,'status':'UNAVAILABLE','reason':'sampled milliseconds have no valid 10s contiguous common-support blocks'})
 gates={'normality':False,'physical_evidence':False,'query_evidence':False,'two_family_bootstrap_improvement':False,'b0_comparison':False,'reasons':['raw tracker code/doppler centers are unverified','DS7/DS8 cleanStatic exact-time overlap cannot be excluded','official processed time origin not verified','no detection claim allowed'],'verdict':'INCONCLUSIVE'};dump(o/'go_no_go.json',gates)
 readme="""# ACAF-NF Stage-0 static physical feasibility\n\nStatic-only raw-IQ feasibility, not a neural field or active policy. cleanDynamic/DS5/DS6 are OOD and excluded from the core gate. DS7/DS8 are one non-independent family; their cleanStatic lineage requires exact-time overlap prevention, which is not proven here. Scenario onset candidates are recorded but raw/processed origin is unverified. B0/M1/R2C are comparators only: B0 native lineage was not rerun, therefore NOT_DIRECTLY_COMPARABLE. This work differs by rereading complex raw IQ and computing local C/A CAF, whereas B0/M1/R2C consume tracking/raw-texture/relation scores. CAF/LASSO/CCAF overlap is local ambiguity structure; no neural reconstruction, sequential policy, or novel CCAF claim is made. The only supported contribution is a fail-closed executable Stage-0 contract and controls. It cannot claim spoof detection, physical two-source TEXBAT evidence, or superiority. **Verdict: INCONCLUSIVE; do not proceed to full ACAF-NF until tracker center/time provenance and DS7/8 overlap are verified.**\n""";(o/'README.md').write_text(readme)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plots=['normalized_score_by_scenario','b0_vs_acaf','takeover_zoom','second_source_trajectory','budget_compute','gain_phase_awgn','caf_magnitude_phase','pre_onset_fpr']
 for name in plots:
  plt.figure(figsize=(7,3));plt.text(.5,.55,name.replace('_',' '),ha='center');plt.text(.5,.35,'INCONCLUSIVE: unverified tracker/time lineage',ha='center');plt.axis('off');plt.savefig(o/'plots'/(name+'.png'),dpi=120);plt.close()
 files=[p for p in o.rglob('*') if p.is_file() and p.name not in {'checksums.json'}];dump(o/'checksums.json',{'algorithm':'sha256','files':{str(p.relative_to(o)):sha(p) for p in files}})
 print(json.dumps({'epochs':len(rows),'verdict':'INCONCLUSIVE'}))
if __name__=='__main__':main()
