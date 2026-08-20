#!/usr/bin/env python3
"""Materialize frozen CRID R3 clean controls; never opens attack data."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gnss_doppler_lab.crid_control_generator import FrozenContext,enumerate_cases,estimate_joint_amplitudes,empirical_noise_sigma,generate_case

def main()->int:
 p=argparse.ArgumentParser();p.add_argument('--domain',choices=('OAK','TEX'),required=True);p.add_argument('--case');p.add_argument('--output-root',type=Path,default=Path('/home/ubuntu/ssd_data/gnss-early-detection/artifacts/crid-stage0-r3-control-generator-foundation'));p.add_argument('--overwrite',action='store_true');a=p.parse_args()
 ctx=FrozenContext(ROOT,a.domain);cases=enumerate_cases(ctx.spec,a.domain)
 if a.case:cases=[c for c in cases if c.case_id==a.case]
 if not cases:raise SystemExit('no matching frozen case')
 alpha=estimate_joint_amplitudes(ctx);sigma=empirical_noise_sigma(ctx);rows=[]
 for case in cases:
  out=a.output_root/'controls'/a.domain/case.case_id/'control.bin';row=generate_case(ctx,case,out,alpha,sigma,overwrite=a.overwrite)
  side=out.parent/'truth.json';side.write_text(json.dumps(row,indent=2,sort_keys=True)+'\n');rows.append(row);print(json.dumps({'case':case.case_id,'sha256':row['output_sha256'],'status':'PASS'}),flush=True)
 summary=a.output_root/'controls'/a.domain/'generation_summary.json';summary.write_text(json.dumps({'domain':a.domain,'case_count':len(rows),'cases':rows},indent=2,sort_keys=True)+'\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
