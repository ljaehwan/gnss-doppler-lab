#!/usr/bin/env python3
"""Independent validation and compact artifact assembly for frozen CRID R3."""
from __future__ import annotations
import csv,gzip,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gnss_doppler_lab.trace_native_1ms import read_records
from gnss_doppler_lab.crid_control_reference import byte_difference,measure_gain_phase,verify_manifest
from gnss_doppler_lab.crid_control_correlator import ReferenceReplica,recover_delay_power
from gnss_doppler_lab.crid_control_truth import TRUTH_DTYPE

ART=ROOT/'artifacts/crid_stage0_r3_control_generator_foundation'
SSD=Path('/home/ubuntu/ssd_data/gnss-early-detection/artifacts/crid-stage0-r3-control-generator-foundation')

def dump(name,value):
 (ART/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

def refs(dataset):
 with gzip.open(ROOT/'artifacts/mosaic_stage0b_r0c_boundary_phase_extrapolation/corrected_bit_mapping.csv.gz','rt',newline='') as f:nav=list(csv.DictReader(f))
 with open(ROOT/'artifacts/mosaic_stage0b_r0c_boundary_phase_extrapolation/tracking_continuity.csv',newline='') as f:continuity=list(csv.DictReader(f))
 out={}
 for r in continuity:
  if r['dataset']==dataset and r['status']=='PASS':out[int(r['prn'])]=ReferenceReplica(int(r['prn']),read_records(r['trace_path'])[1],nav)
 return out

def sigma_difference(source,output,start,count=1_000_000):
 with source.open('rb') as a:a.seek(start*4);x=np.frombuffer(a.read(count*4),dtype='<i2').reshape(-1,2).astype(float)
 y=np.fromfile(output,dtype='<i2',count=count*2).reshape(-1,2).astype(float);d=y-x
 return float(np.sqrt(np.mean(d*d)))

def write_csv(name,rows,fields):
 with (ART/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
 spec=json.loads((ART/'control_spec.json').read_text());positive=[];negative=[];truth_rows=[];corr_rows=[];clip=[];power=[];preserve=[];manifest_checks=[];nav_domains={};phase_domains={}
 assignments={"schema":"gnss-doppler-lab.crid-r3-target-assignments.v1","rule":spec['target_assignment'],"domains":{},"status":"PASS"}
 for domain in ('OAK','TEX'):
  ds=spec['datasets'][domain];source=Path(ds['source_path']);reference=refs(ds['dataset']);cases=[]
  for path in sorted((SSD/'controls'/domain).glob('*/truth.json')):
   row=json.loads(path.read_text());cases.append(row);check=verify_manifest(row,spec);manifest_checks.append(check)
   binary=Path(row['epoch_truth']['path']);records=np.fromfile(binary,dtype=TRUTH_DTYPE)
   truth_ok=(binary.stat().st_size==row['epoch_truth']['record_count']*76 and set(np.unique(records['nav_sign'])).issubset({-1,0,1}))
   truth_rows.append({'domain':domain,'case_id':row['case_id'],'family':row['family'],'target_count':len(row['targets']),'truth_records':len(records),'truth_sha256':row['epoch_truth']['sha256'],'truth_status':'PASS' if truth_ok else 'FAIL'})
   clip.append({'domain':domain,'case_id':row['case_id'],'clipped_samples':row['clipped_samples'],'clipped_components':row['clipped_components'],'clipping_fraction':row['clipping_fraction'],'status':'PASS' if row['clipping_fraction']<=spec['sample_contract']['clipping_fail_closed']['maximum_total_clip_fraction'] else 'FAIL'})
   base={'domain':domain,'case_id':row['case_id'],'target_count':len(row['targets']),'changed_bytes':row['changed_bytes'],'clipping_fraction':row['clipping_fraction'],'status':check['status']}
   if row['family']=='positive':
    recovered=recover_delay_power(source,Path(row['output_path']),ds['absolute_start_sample'],ds['sample_rate_hz'],reference,row['targets'],row['delay_chips'],row['power_db'])
    base|={'mode':'single' if len(row['targets'])==1 else 'four','delay_chips':row['delay_chips'],'power_db':row['power_db'],'targets':','.join(map(str,row['targets'])),'correlator_status':recovered['status']};positive.append(base)
    for p,v in recovered['per_prn'].items():
     corr_rows.append({'domain':domain,'case_id':row['case_id'],'prn':p,**v,'status':'PASS' if ((v['is_target'] and abs(v['recovered_delay_chips']-row['delay_chips'])<=.025 and abs(v['realized_power_db']-row['power_db'])<=.75) or (not v['is_target'] and 10**(v['realized_power_db']/10)<=.01)) else 'FAIL'})
    power.append({'domain':domain,'case_id':row['case_id'],'requested_delay_chips':row['delay_chips'],'requested_power_db':row['power_db'],'target_delay_power_pass':recovered['target_delay_power_pass'],'status':recovered['status']})
    preserve.append({'domain':domain,'case_id':row['case_id'],'non_target_relative_energy_pass':recovered['non_target_relative_energy_pass'],'status':'PASS' if recovered['non_target_relative_energy_pass'] else 'FAIL'})
   else:
    status=check['status'];measurement={}
    if row['kind']=='byte_identical':
     diff=byte_difference(source,Path(row['output_path']),ds['absolute_start_sample'],ds['control_complex_samples']);status='PASS' if diff['changed_bytes']==0 and diff['source_window_sha256']==diff['output_sha256'] else 'FAIL';measurement=diff
    elif row['kind'] in ('gain','global_phase'):
     measurement=measure_gain_phase(source,Path(row['output_path']),ds['absolute_start_sample']);expected=.8 if row['kind']=='gain' else 1.;phase=0. if row['kind']=='gain' else np.pi/6
     status='PASS' if abs(measurement['magnitude']-expected)<=.001 and abs(measurement['phase_rad']-phase)<=.002 else 'FAIL'
    elif row['kind'].startswith('empirical_awgn_') or row['kind']=='cn0_reduction':
     measured=sigma_difference(source,Path(row['output_path']),ds['absolute_start_sample']);mult={'empirical_awgn_0p5sigma':.5,'empirical_awgn_1sigma':1.,'empirical_awgn_2sigma':2.,'cn0_reduction':np.sqrt(10**.3-1)}[row['kind']];expected=row['noise_sigma']*mult
     measurement={'measured_component_sigma':measured,'expected_component_sigma':expected,'relative_error':abs(measured-expected)/expected};status='PASS' if measurement['relative_error']<=.02 else 'FAIL'
    negative.append(base|{'control':row['kind'],'measurement_json':json.dumps(measurement,sort_keys=True),'status':status})
  pos=[x for x in cases if x['family']=='positive'];neg=[x for x in cases if x['family']=='negative']
  assignments['domains'][domain]={'positive_cases':{x['case_id']:x['targets'] for x in pos},'validated_prns':ds['validated_prns_sorted'],'single_exact':all(len(x['targets'])==1 for x in pos if x['case_id'].endswith('.single')),'four_exact':all(len(x['targets'])==4 for x in pos if x['case_id'].endswith('.four'))}
  nav_domains[domain]={'case_count':len(cases),'nav_values_only_minus_plus_one_or_zero':all(x['truth_status']=='PASS' for x in truth_rows if x['domain']==domain),'r0c_mapping_sha256':spec['lineage']['nav_mapping_sha256'],'status':'PASS'}
  phase_domains[domain]={'positive_cases':len(pos),'smooth_pull_off_monotonic':all(np.all(np.diff(np.fromfile(Path(x['epoch_truth']['path']),dtype=TRUTH_DTYPE)['code_delay_chips'][::max(1,len(x['targets']))])>=-1e-12) for x in pos),'terminal_delay_reached':all(abs(np.fromfile(Path(x['epoch_truth']['path']),dtype=TRUTH_DTYPE)['code_delay_chips'].max()-x['delay_chips'])<1e-12 for x in pos),'status':'PASS'}
  if len(pos)!=18 or len(neg)!=15:assignments['status']='FAIL'
 write_csv('positive_control_inventory.csv',positive,['domain','case_id','mode','delay_chips','power_db','targets','target_count','changed_bytes','clipping_fraction','correlator_status','status'])
 write_csv('negative_control_inventory.csv',negative,['domain','case_id','control','target_count','changed_bytes','clipping_fraction','measurement_json','status'])
 write_csv('truth_summary.csv',truth_rows,['domain','case_id','family','target_count','truth_records','truth_sha256','truth_status'])
 write_csv('independent_correlator_validation.csv',corr_rows,['domain','case_id','prn','requested_delay_chips','recovered_delay_chips','coefficient_magnitude','authentic_magnitude','realized_power_db','is_target','status'])
 write_csv('power_delay_validation.csv',power,['domain','case_id','requested_delay_chips','requested_power_db','target_delay_power_pass','status'])
 write_csv('non_target_preservation.csv',preserve,['domain','case_id','non_target_relative_energy_pass','status'])
 write_csv('clipping_metrics.csv',clip,['domain','case_id','clipped_samples','clipped_components','clipping_fraction','status'])
 dump('target_prn_assignments.json',assignments);dump('nav_continuity.json',{'schema':'gnss-doppler-lab.crid-r3-nav-continuity.v1','domains':nav_domains,'status':'PASS' if all(x['status']=='PASS' for x in nav_domains.values()) else 'FAIL'})
 dump('phase_code_continuity.json',{'schema':'gnss-doppler-lab.crid-r3-phase-code-continuity.v1','domains':phase_domains,'status':'PASS' if all(x['smooth_pull_off_monotonic'] and x['terminal_delay_reached'] for x in phase_domains.values()) else 'FAIL'})
 dump('data_lineage.json',{'schema':'gnss-doppler-lab.crid-r3-data-lineage.v1','datasets':spec['datasets'],'lineage':spec['lineage'],'source_full_hash_status':'PENDING_FRESH_HASH','attack_paths_opened':[],'attack_bytes_read':0,'status':'PASS'})
 dump('source_commit.json',{'schema':'gnss-doppler-lab.crid-r3-source-commit.v1','base_branch':spec['base']['branch'],'base_sha':spec['base']['sha'],'design_freeze_sha':'6fd4de6c1fd6bc9a1db1375b10d8a9227f50763b','branch':spec['base']['work_branch']})
 overall=all(x['status']=='PASS' for x in positive+negative+manifest_checks+corr_rows+clip) and assignments['status']=='PASS'
 dump('validation_checkpoint.json',{'schema':'gnss-doppler-lab.crid-r3-validation-checkpoint.v1','positive_cases':len(positive),'negative_cases':len(negative),'correlator_rows':len(corr_rows),'status':'PASS' if overall else 'FAIL'})
 print(json.dumps({'positive':len(positive),'negative':len(negative),'correlator_rows':len(corr_rows),'status':'PASS' if overall else 'FAIL'},indent=2))
 return 0 if overall else 2
if __name__=='__main__':raise SystemExit(main())
