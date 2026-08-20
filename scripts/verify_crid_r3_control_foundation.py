#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/crid_stage0_r3_control_generator_foundation'
REQ={'README.md','control_spec.json','design_freeze.json','source_commit.json','data_lineage.json','target_prn_assignments.json','positive_control_inventory.csv','negative_control_inventory.csv','truth_summary.csv','independent_correlator_validation.csv','nav_continuity.json','phase_code_continuity.json','power_delay_validation.csv','non_target_preservation.csv','clipping_metrics.csv','c0_smoke_validation.json','tamper_tests.json','final_verdict.json','artifact_manifest_sha256.json'}
def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def main():
 failures=[]
 for name in REQ:
  if not (ART/name).is_file():failures.append('missing:'+name)
 if failures:print(json.dumps({'status':'FAIL','failures':failures},indent=2));return 2
 spec=json.loads((ART/'control_spec.json').read_text());freeze=json.loads((ART/'design_freeze.json').read_text())
 if sha(ART/'control_spec.json')!=freeze['control_spec_sha256']:failures.append('control_spec_freeze_hash')
 pos=list(csv.DictReader((ART/'positive_control_inventory.csv').open()));neg=list(csv.DictReader((ART/'negative_control_inventory.csv').open()))
 for d in ('OAK','TEX'):
  if sum(x['domain']==d for x in pos)!=18:failures.append('positive_coverage:'+d)
  if sum(x['domain']==d for x in neg)!=15:failures.append('negative_coverage:'+d)
 manifest=json.loads((ART/'artifact_manifest_sha256.json').read_text())
 for x in manifest['files']:
  p=ART/x['path']
  if not p.is_file() or p.stat().st_size!=x['size_bytes'] or sha(p)!=x['sha256']:failures.append('manifest:'+x['path'])
 final=json.loads((ART/'final_verdict.json').read_text())
 if final['verdict'] not in ('CONTROL_GENERATOR_FOUNDATION_PASS','CONTROL_GENERATOR_FOUNDATION_FAIL','INCONCLUSIVE_CONTROL_PROVENANCE'):failures.append('verdict_enum')
 if final['attack_bytes_read']!=0 or final['crid_score_computed'] or final['attack_evaluation_executed']:failures.append('scope')
 out={'schema':'gnss-doppler-lab.crid-r3-verification.v1','verdict':final['verdict'],'positive_cases':len(pos),'negative_cases':len(neg),'manifest_files':manifest['file_count'],'failures':failures,'status':'PASS' if not failures else 'FAIL'};print(json.dumps(out,indent=2));return 0 if not failures else 2
if __name__=='__main__':raise SystemExit(main())
