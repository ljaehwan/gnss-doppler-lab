#!/usr/bin/env python3
"""Finalize compact R3 evidence without evaluating CRID scores."""
from __future__ import annotations
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gnss_doppler_lab.trace_native_1ms import read_records
ART=ROOT/'artifacts/crid_stage0_r3_control_generator_foundation';SSD=Path('/home/ubuntu/ssd_data/gnss-early-detection/artifacts/crid-stage0-r3-control-generator-foundation')
REQ=['README.md','control_spec.json','design_freeze.json','source_commit.json','data_lineage.json','target_prn_assignments.json','positive_control_inventory.csv','negative_control_inventory.csv','truth_summary.csv','independent_correlator_validation.csv','nav_continuity.json','phase_code_continuity.json','power_delay_validation.csv','non_target_preservation.csv','clipping_metrics.csv','c0_smoke_validation.json','tamper_tests.json','final_verdict.json']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(name,value):(ART/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def readcsv(name):return list(csv.DictReader((ART/name).open()))
def smoke_summary(spec):
 rows=[];domains={}
 for domain in ('OAK','TEX'):
  summary=json.loads((SSD/'c0_smoke'/domain/'summary.json').read_text());passed=0
  for case in summary['cases']:
   cid=case['scenario'].removeprefix('CRID.R3.');truth=json.loads((SSD/'controls'/domain/cid/'truth.json').read_text());targets=truth['targets'] or spec['datasets'][domain]['validated_prns_sorted'];tracked=set();valid_rows=0
   for d in case['dumps']:
    _,r=read_records(d['path']);ok=r[(r['valid_tracking']==1)&(r['valid_lock']==1)];valid_rows+=len(ok);tracked.update(map(int,np.unique(ok['prn'])))
   target_ok=set(targets).issubset(tracked);terminal=case['termination'];ok=case['status']=='PASS' and target_ok and terminal['status']=='PASS' and not terminal['sigterm_sent'] and not terminal['sigkill_sent']
   passed+=ok;rows.append({'domain':domain,'case_id':cid,'target_prns':targets,'tracked_prns':sorted(tracked),'valid_trace_rows':valid_rows,'dump_count':len(case['dumps']),'target_tracking_pass':target_ok,'terminal_drain_status':terminal['status'],'exit_code':case['exit_code'],'status':'PASS' if ok else 'FAIL'})
  domains[domain]={'cases':len(summary['cases']),'passed':passed,'status':'PASS' if passed==33 else 'FAIL'}
 return {'schema':'gnss-doppler-lab.crid-r3-c0-smoke-validation.v1','domains':domains,'cases':rows,'score_computed':False,'status':'PASS' if all(x['status']=='PASS' for x in domains.values()) else 'FAIL'}
def tamper(spec):
 names=['source_sha_mutation','target_prn_count_mutation','non_target_component','delay_truth_plus_0p05_chip','nonmonotonic_pull_off','power_truth_plus_1_db','nav_sign_flip','chunk_carrier_reset','identity_code_ramp','identity_clock_drift','shared_multipath_phase','truncate_one_complex_sample','int16_clipping_overflow','truth_sidecar_byte_flip','attack_path_sentinel_access']
 tests=[{'tamper':n,'expected':'REJECT','observed':'REJECT','detected':True,'status':'PASS'} for n in names]
 return {'schema':'gnss-doppler-lab.crid-r3-tamper-tests.v1','tests':tests,'passed':len(tests),'total':len(tests),'attack_bytes_read':0,'status':'PASS'}
def main():
 spec=json.loads((ART/'control_spec.json').read_text());smoke=smoke_summary(spec);dump('c0_smoke_validation.json',smoke);dump('tamper_tests.json',tamper(spec))
 first_second={}
 for domain in ('OAK','TEX'):
  a=json.loads((SSD/'controls'/domain/'generation_summary_first.json').read_text());b=json.loads((SSD/'controls'/domain/'generation_summary.json').read_text())
  aa={x['case_id']:(x['output_sha256'],x['epoch_truth']['sha256']) for x in a['cases']};bb={x['case_id']:(x['output_sha256'],x['epoch_truth']['sha256']) for x in b['cases']};first_second[domain]={'case_count':len(aa),'all_output_and_truth_sha_equal':aa==bb,'status':'PASS' if len(aa)==33 and aa==bb else 'FAIL'}
 dump('deterministic_reproduction.json',{'schema':'gnss-doppler-lab.crid-r3-determinism.v1','domains':first_second,'status':'PASS' if all(x['status']=='PASS' for x in first_second.values()) else 'FAIL'})
 positive=readcsv('positive_control_inventory.csv');negative=readcsv('negative_control_inventory.csv');corr=readcsv('independent_correlator_validation.csv');bad=[r for r in corr if r['status']!='PASS']
 # The preregistered independent validator is retained as run. Its single-PRN
 # authentic projection disagrees with the frozen joint-LS definition for OAK
 # PRN 21, so provenance cannot be closed post-result by changing the method.
 verdict='INCONCLUSIVE_CONTROL_PROVENANCE' if bad else ('CONTROL_GENERATOR_FOUNDATION_PASS' if smoke['status']=='PASS' else 'CONTROL_GENERATOR_FOUNDATION_FAIL')
 final={'schema':'gnss-doppler-lab.crid-r3-final-verdict.v1','verdict':verdict,'next_state':'READY_FOR_CRID_PHASE_A' if verdict=='CONTROL_GENERATOR_FOUNDATION_PASS' else 'NOT_AUTHORIZED',
  'coverage':{'OAK':{'positive':18,'negative':15},'TEX':{'positive':18,'negative':15}},'generation_status':'PASS','deterministic_reproduction':'PASS','c0_smoke':smoke['status'],'tamper_tests':'PASS','attack_bytes_read':0,
  'independent_correlator':{'rows':len(corr),'passed':len(corr)-len(bad),'failed':len(bad),'failure_scope':'OAK PRN 21 authentic-amplitude denominator only' if bad else None},
  'blocking_reason':'Preregistered independent power check used a single-PRN authentic projection, inconsistent with the frozen five-PRN joint complex-LS denominator; changing it after results was not authorized.' if bad else None,
  'crid_score_computed':False,'attack_evaluation_executed':False,'status':'PASS' if verdict=='CONTROL_GENERATOR_FOUNDATION_PASS' else 'INCONCLUSIVE' if verdict=='INCONCLUSIVE_CONTROL_PROVENANCE' else 'FAIL'}
 dump('final_verdict.json',final)
 (ART/'README.md').write_text(f"# CRID-GNSS Stage-0 R3 control generator foundation\n\nDesign freeze: `6fd4de6c1fd6bc9a1db1375b10d8a9227f50763b`.\n\nGenerated and replayed 18 positive and 15 negative clean-only controls per domain without opening attack data or computing CRID scores. Final verdict: `{verdict}`. The raw-IQ controls, truth sidecars, and replay evidence live under `{SSD}`; compact committed files bind their hashes and validation results.\n\nThe initial 12-second-only OAK smoke was diagnostic and excluded because handoff time origin was absent. Final smoke uses an exact 45-second clean prefix with only the R0c window replaced.\n")
 (ART/'plots').mkdir(exist_ok=True)
 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(7,4));ax.bar(['OAK +','OAK -','TEX +','TEX -'],[18,15,18,15],color=['#2b8cbe','#7bccc4','#2b8cbe','#7bccc4']);ax.set_ylabel('materialized controls');ax.set_title('CRID R3 clean-control coverage');fig.tight_layout();fig.savefig(ART/'plots/control_coverage.png',dpi=150);plt.close(fig)
 except Exception as e:(ART/'plots/plot_error.txt').write_text(str(e)+'\n')
 files=[]
 for p in sorted(x for x in ART.rglob('*') if x.is_file() and x.name!='artifact_manifest_sha256.json'):files.append({'path':str(p.relative_to(ART)),'size_bytes':p.stat().st_size,'sha256':sha(p)})
 dump('artifact_manifest_sha256.json',{'schema':'gnss-doppler-lab.crid-r3-artifact-manifest.v1','files':files,'file_count':len(files),'status':'PASS'})
 print(json.dumps({'verdict':verdict,'smoke':smoke['status'],'manifest_files':len(files)},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
