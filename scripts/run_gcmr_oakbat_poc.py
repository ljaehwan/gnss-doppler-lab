#!/usr/bin/env python3
"""Run the leakage-safe clean-only GCMR OAKBAT proof of concept."""
from __future__ import annotations
import argparse, json, platform, subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from gnss_doppler_lab.gcmr_experiment import (DEFAULT_ROLES, ExperimentGate,
 ablated_events, cache_events, calibration_threshold, load_event_cache,
 implementation_manifest, parse_preonset_nmea_position, preflight_oakbat_geometry, save_checkpoint, save_score_csv, score_events,
 select_role_events, source_hashes, train_clean_model, validate_roles, write_summary)
from gnss_doppler_lab.gcmr_geometry import parse_gnss_sdr_gps_ephemeris_xml
from gnss_doppler_lab.gcmr_model import CleanReferenceScoreCalibrator
from gnss_doppler_lab.gcmr_relations import build_gcmr_pair_relation_events, load_gnss_sdr_tracking_rows

LAB=Path('/home/ubuntu/projects/gnss-doppler-lab')
SCENARIOS={'cleanStatic':LAB/'artifacts/oakbat_9tap_frozen_champion/cleanStatic/receiver/oakbat-cleanStatic-method-a-9tap',**{f'os{i}':LAB/f'artifacts/oakbat_cleanstatic_detector_eval_v1/preprocessed/os{i}/receiver/oakbat-os{i}-method-a-9tap' for i in range(1,5)}}
TIMING={'sample_rate_hz':5e6,'gps_tow_at_time_zero_s':381618.0,'onset_s':120.0,'window_s':1.0,'stride_s':.5,'resample_bin_s':.02,'min_common_samples':20,'min_prns':4,'max_toe_age_s':7200.0,'tow0_rx_tolerance_s':0.05,'score_available_at':'window_end_s','window_interval':'[start,end)'}

def sources(root):return [root/'nmea_pvt.nmea',root/'gps_ephemeris.xml',root/'raw/observables.mat',*sorted((root/'raw').glob('epl_tracking_ch_*.mat'))]
def load_scenario(name,cache_dir,force):
 root=SCENARIOS[name];src=sources(root);meta={'scenario':name,'timing':TIMING}
 eph=parse_gnss_sdr_gps_ephemeris_xml(root/'gps_ephemeris.xml')
 rows=load_gnss_sdr_tracking_rows(root/'raw',sample_rate_hz=TIMING['sample_rate_hz'])
 preflight=preflight_oakbat_geometry(root/'raw/observables.mat',root/'nmea_pvt.nmea',eph,configured_tow0_s=TIMING['gps_tow_at_time_zero_s'],max_toe_age_s=TIMING['max_toe_age_s'],tow_tolerance_s=TIMING['tow0_rx_tolerance_s'],onset_s=TIMING['onset_s'],tracked_prns={row.prn for row in rows},min_prns=TIMING['min_prns'])
 meta['geometry_preflight']=preflight
 meta['ephemeris_health']=preflight['ephemeris_health']
 position=parse_preonset_nmea_position(root/'nmea_pvt.nmea',gps_tow_at_time_zero_s=TIMING['gps_tow_at_time_zero_s'],onset_s=TIMING['onset_s'])
 meta['receiver_position_contract']={k:position[k] for k in ('llh','ecef','sample_count','timing','assumption')}
 target=cache_dir/f'{name}.relations.npz'
 if target.exists() and not force:
  return load_event_cache(target,source_paths=src,expected_metadata=meta)[0],meta,src
 ev=build_gcmr_pair_relation_events(rows,ephemerides=eph,receiver_ecef=position['ecef'],gps_tow_at_time_zero_s=TIMING['gps_tow_at_time_zero_s'],window_s=1.,stride_s=.5,resample_bin_s=.02,min_common_samples=20,min_prns=4)
 cache_events(target,ev,source_paths=src,metadata=meta);return ev,meta,src

def metrics(scored,threshold,mask=None):
 mask=np.ones(len(scored['combined_score']),bool) if mask is None else mask;s=np.asarray(scored['combined_score'])[mask];t=np.asarray(scored['availability_s'])[mask];alarm=s>threshold
 return {'events':int(len(s)),'alarm_count':int(alarm.sum()),'alarm_rate':float(alarm.mean()) if len(s) else None,'score_median':float(np.median(s)) if len(s) else None,'score_q99':float(np.quantile(s,.99)) if len(s) else None,'first_alarm_s':float(t[np.flatnonzero(alarm)[0]]) if alarm.any() else None}
def scenario_metrics(scored,threshold,onset=120):
 t=scored['availability_s'];out={}
 for name,mask in [('pre',t<110),('transition',(t>=110)&(t<130)),('post',t>=130)]:out[name]=metrics(scored,threshold,mask)
 alarms=(scored['combined_score']>threshold)&(t>=onset);first=float(t[np.flatnonzero(alarms)[0]]) if alarms.any() else None
 out['first_alarm_s']=first;out['first_alarm_delay_s']=None if first is None else first-onset;return out
def timeline(path,scored,threshold,onset=None):
 fig,ax=plt.subplots(figsize=(12,4));ax.plot(scored['availability_s'],scored['combined_score'],lw=1,label='combined score');ax.axhline(threshold,color='r',ls='--',label='clean calibration q99')
 if onset is not None:ax.axvline(onset,color='k',ls=':',label='onset')
 ax.set(xlabel='score availability / window end (s)',ylabel='score');ax.legend();fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
def git(command):
 try:return subprocess.check_output(command,text=True,cwd=Path(__file__).resolve().parents[1]).strip()
 except Exception:return 'unavailable'

def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,default=Path('artifacts/gcmr_oakbat_poc'));ap.add_argument('--cache-dir',type=Path);ap.add_argument('--max-epochs',type=int,default=40);ap.add_argument('--force-cache',action='store_true');ap.add_argument('--open-attacks',action='store_true');ap.add_argument('--scenarios',nargs='+',choices=SCENARIOS);ap.add_argument('--seed',type=int,default=7);a=ap.parse_args(argv)
 chosen=a.scenarios or (['cleanStatic',*('os1','os2','os3','os4')] if a.open_attacks else ['cleanStatic'])
 if 'cleanStatic' not in chosen:ap.error('cleanStatic is mandatory for clean-only fitting')
 attacks=[x for x in chosen if x!='cleanStatic']
 if attacks and not a.open_attacks:ap.error('attack scenarios require explicit --open-attacks')
 validate_roles();out=a.output_dir.resolve();cache=(a.cache_dir or out/'cache').resolve();out.mkdir(parents=True,exist_ok=True)
 clean,clean_meta,clean_src=load_scenario('cleanStatic',cache,a.force_cache)
 fit_roles=DEFAULT_ROLES[:-1]
 role_events={r.name:select_role_events(clean,r) for r in fit_roles}
 missing=[k for k,v in role_events.items() if not v]
 if missing:raise RuntimeError(f'cleanStatic has no wholly contained events for roles: {missing}')
 training=train_clean_model(role_events['train'],role_events['selection_val'],seed=a.seed,max_epochs=a.max_epochs)
 reference_raw=score_events(training.model,role_events['clean_reference']);calibrator=CleanReferenceScoreCalibrator().fit(reference_raw['reconstruction'],reference_raw['latent'])
 calibration=score_events(training.model,role_events['event_calibration'],calibrator);threshold=calibration_threshold(calibration['combined_score'],quantile=.99)
 provenance={'implementation':implementation_manifest(),'runner':'clean-only GCMR OAKBAT','git_commit':git(['git','rev-parse','HEAD']),'git_status':git(['git','status','--short']),'python':platform.python_version(),'torch':torch.__version__,'seed':a.seed,'timing_contract':TIMING,'roles':[vars(x) for x in DEFAULT_ROLES],'clean_sources_sha256':source_hashes(clean_src),'receiver_position':clean_meta['receiver_position_contract'],'geometry_preflight':clean_meta['geometry_preflight'],'ephemeris_health':clean_meta['ephemeris_health'],'oracle_deployment_assumption':clean_meta['receiver_position_contract']['assumption'],'leakage_contract':{'scaler_model_fit':'train only','epoch_selection':'selection_val only','score_calibrator_fit':'clean_reference only','threshold_q99':'event_calibration only','sealed_held':'read only after freeze','attacks':'never fit any artifact'}}
 gate=ExperimentGate();save_checkpoint(out/'model.pt',training,calibrator,threshold,provenance=provenance);(out/'provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n');gate.freeze()
 # The sealed role is not selected or inspected until all fitted artifacts are frozen.
 held_events=select_role_events(clean,DEFAULT_ROLES[-1])
 if not held_events:raise RuntimeError('cleanStatic has no wholly contained sealed held events')
 held=score_events(training.model,held_events,calibrator);save_score_csv(out/'cleanStatic_scores.csv',held,threshold);timeline(out/'cleanStatic_timeline.png',held,threshold)
 results={'sealed_held':{**metrics(held,threshold),'normal_context':'same_recording_held_normal','role':'sealed_held','ephemeris_health':clean_meta['ephemeris_health']}};gate.mark_held_evaluated()
 # These are inference perturbations, not retraining comparisons.
 diagnostics={}
 for mode in ('geometry_channels_permutation','geometry_channels_zero'):
  ss=score_events(training.model,ablated_events(held_events,mode=mode,seed=a.seed),calibrator);diagnostics[mode]={**metrics(ss,threshold),'claim':'deterministic inference ablation; model not retrained'}
 if a.open_attacks:
  gate.open_attacks(explicit=True)
  for name in attacks:
   ev,meta,src=load_scenario(name,cache,a.force_cache);scored=score_events(training.model,ev,calibrator);save_score_csv(out/f'{name}_scores.csv',scored,threshold);timeline(out/f'{name}_timeline.png',scored,threshold,onset=TIMING['onset_s'])
   results[name]={**scenario_metrics(scored,threshold),'normal_context':'ood_normal_pre_attack','attack_context':'evaluation_only_never_used_for_fit','source_sha256':source_hashes(src),'receiver_position':meta['receiver_position_contract'],'geometry_preflight':meta['geometry_preflight']}
 summary=write_summary(out/'summary.json',results,threshold=threshold,threshold_source='cleanStatic event_calibration only q99',best_epoch=training.best_epoch,training_history=training.history,inference_diagnostics=diagnostics,attacks_open=gate.attacks_open,geometry_preflight=clean_meta['geometry_preflight'],ephemeris_health=clean_meta['ephemeris_health'],provenance=provenance)
 print(json.dumps({'output_dir':str(out),'threshold':threshold,'best_epoch':training.best_epoch,'results':results},indent=2,default=str));return 0
if __name__=='__main__':raise SystemExit(main())
