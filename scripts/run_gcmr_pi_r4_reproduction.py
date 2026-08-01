#!/usr/bin/env python3
"""CUDA-first r3 score-reproduction diagnosis; never emits corrected diagnostics."""
from __future__ import annotations
import argparse,csv,json,platform,subprocess
from pathlib import Path
import numpy as np, torch
from train_gcmr_peak_innovation import records,peak_indexes
from run_gcmr_oakbat_poc import SCENARIOS as SOURCE_SCENARIOS,load_scenario
from gnss_doppler_lab.gcmr_pi_r4_corrected import reconstruct_event_innovation,rescore_from_innovations
from gnss_doppler_lab.gcmr_pi_r4_reproduction import component_agreement
FIELDS=("S_common","N_eff","S_pair","energy","Full")
SCENARIOS=("os1","os2","os3","os4")

def configure(seed, device):
 torch.manual_seed(seed); np.random.seed(seed)
 if device.type=='cuda': torch.cuda.manual_seed_all(seed)
 torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

def load_pipe(path,device,seed):
 configure(seed,device); p=torch.load(path,map_location=device,weights_only=False);p.device=device;p.network.to(device).eval();return p

def rows(path):
 with open(path,newline='') as f:return list(csv.DictReader(f))
def runtime():
 return {'python':platform.python_version(),'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
def source_commit():
 return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
def compare(pipe,frozen,scenario):
 ev,_,_=load_scenario(scenario,frozen/'cache',False); built,_=records(ev,SOURCE_SCENARIOS[scenario],pipe.window,peak_indexes(ev,SOURCE_SCENARIOS[scenario],pipe.feature_dim)); ref=rows(frozen/f'{scenario}_scores.csv')
 if len(built)!=len(ref):raise RuntimeError(f'{scenario}: record count mismatch {len(built)} != {len(ref)}')
 t=[]; actual={k:[] for k in FIELDS}; expected={k:[] for k in FIELDS}
 for event,row in zip(built,ref):
  d,s=rescore_from_innovations(pipe,event,*reconstruct_event_innovation(pipe,event)); values={'S_common':d.s_common,'N_eff':d.n_eff,'S_pair':d.s_pair,'energy':d.energy,'Full':s['Full']}
  t.append(event.time)
  for k in FIELDS:actual[k].append(values[k]);expected[k].append(float(row[k]))
 thresholds=json.loads((frozen/'thresholds.json').read_text())
 return {k:component_agreement(expected[k],actual[k],threshold=thresholds['Full']['q99'] if k=='Full' else float('inf'),times=t) for k in FIELDS}
def main():
 a=argparse.ArgumentParser();a.add_argument('--frozen',type=Path,required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 seed=json.loads((args.frozen/'training_summary.json').read_text())['seed']; configure(seed, torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')); report={'artifact_commit':'6f51f4809bbd4c012b39424003f8bcc40e22cf4a','current_commit':source_commit(),'runtime':runtime(),'seed':seed,'modes':{}}
 if not torch.cuda.is_available(): report['status']='blocked_no_cuda';(args.out/'reproduction_diagnosis.json').write_text(json.dumps(report,indent=2)+'\n');return 2
 for name,device in [('cuda',torch.device('cuda')),('cpu',torch.device('cpu'))]:
  try:
   pipe=load_pipe(args.frozen/'model.pt',device,seed);report['modes'][name]={sc:compare(pipe,args.frozen,sc) for sc in SCENARIOS}
  except Exception as e:report['modes'][name]={'error':f'{type(e).__name__}: {e}'}
 report['status']='diagnostic_complete_not_reproduction_pass';(args.out/'reproduction_diagnosis.json').write_text(json.dumps(report,indent=2)+'\n');return 0
if __name__=='__main__':raise SystemExit(main())
