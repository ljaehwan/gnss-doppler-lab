#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gnss_doppler_lab.crid_r3_smoke import run_c0_smoke
from gnss_doppler_lab.crid_r3_smoke_package import compose_prefix
RECEIVER=Path('/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2c-terminal-drain-repair/receiver-build/src/main/gnss-sdr')
BASE={'TEX':Path('/home/ubuntu/ssd_data/gnss-early-detection/artifacts/crid-stage0-counterfactual-receiver-invariance/replays/tex_clean/C0/receiver.conf'),'OAK':Path('/home/ubuntu/ssd_data/gnss-early-detection/artifacts/crid-stage0-counterfactual-receiver-invariance/replays/oak_clean/C0/receiver.conf')}
SSD=Path('/home/ubuntu/ssd_data/gnss-early-detection/artifacts/crid-stage0-r3-control-generator-foundation')
def main():
 p=argparse.ArgumentParser();p.add_argument('--domain',choices=('OAK','TEX'),required=True);p.add_argument('--case');a=p.parse_args();spec=json.loads((ROOT/'artifacts/crid_stage0_r3_control_generator_foundation/control_spec.json').read_text());ds=spec['datasets'][a.domain]
 truths=sorted((SSD/'controls'/a.domain).glob('*/truth.json'))
 if a.case:truths=[x for x in truths if json.loads(x.read_text())['case_id']==a.case]
 rows=[]
 for path in truths:
  truth=json.loads(path.read_text());case=truth['case_id'];out=SSD/'c0_smoke'/a.domain/case
  composite=SSD/'smoke_prefixes'/a.domain/case/'control_45s.bin';package=compose_prefix(Path(ds['source_path']),Path(truth['output_path']),composite,absolute_start=ds['absolute_start_sample'],control_samples=ds['control_complex_samples'],total_samples=45*ds['sample_rate_hz'])
  result=run_c0_smoke(receiver=RECEIVER,base_config=BASE[a.domain],raw=composite,out=out,scenario=f'CRID.R3.{case}',fs=ds['sample_rate_hz'],absolute_start=0,complex_samples=45*ds['sample_rate_hz'],expected_dumps=11 if a.domain=='OAK' else 10,raw_sha256=package['sha256']);result['smoke_package']=package;rows.append(result)
  print(json.dumps({'case':case,'status':rows[-1]['status'],'dump_count':len(rows[-1]['dumps'])}),flush=True)
 (SSD/'c0_smoke'/a.domain/'summary.json').write_text(json.dumps({'domain':a.domain,'case_count':len(rows),'cases':rows},indent=2,sort_keys=True)+'\n')
 return 0 if rows and all(x['status']=='PASS' for x in rows) else 2
if __name__=='__main__':raise SystemExit(main())
