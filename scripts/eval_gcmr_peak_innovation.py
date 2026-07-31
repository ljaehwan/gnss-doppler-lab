#!/usr/bin/env python3
"""Frozen GCMR-PI evaluator for TEXBAT normal-only contained protocol.

TEXBAT cleanStatic+cleanDynamic are normal inference scenarios only; DS attack
files are refused.  This is explicitly within-corpus protocol, never external
validation, unless the loaded model was frozen from an independent corpus.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import torch
from train_gcmr_peak_innovation import records,score,emit_scores,metrics,plots,manifest

def digest(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--thresholds',type=Path,required=True);ap.add_argument('--output-dir',type=Path,default=Path('artifacts/gcmr_peak_innovation/texbat'));ap.add_argument('--scenario',choices=('cleanStatic','cleanDynamic'),required=True);ap.add_argument('--events',type=Path,help='prebuilt, authenticated EventRecord torch payload; no DS input accepted');a=ap.parse_args(argv)
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 if not a.model.exists() or not a.thresholds.exists():raise SystemExit('frozen model and normal-only thresholds are required')
 if not a.events:raise SystemExit('BLOCKED: no authenticated normal EventRecord payload supplied; DS attack data are intentionally unsupported')
 pipe=torch.load(a.model,weights_only=False,map_location='cpu');payload=torch.load(a.events,weights_only=False,map_location='cpu');events=payload['events'] if isinstance(payload,dict) else payload
 thresholds=json.loads(a.thresholds.read_text());s=score(pipe,events,0);emit_scores(out/f'{a.scenario}_scores.csv',s);plots(out,s)
 summary={'protocol':'TEXBAT contained normal-only inference; not external validation','scenario':a.scenario,'model_sha256':digest(a.model),'thresholds_sha256':digest(a.thresholds),'metrics':metrics(s,thresholds),'ds_attack_data_used':False}
 (out/'scenario_metrics.json').write_text(json.dumps(summary,indent=2)+'\n');manifest(out);print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
