#!/usr/bin/env python3
"""Score a run with a frozen Peak–Floor CPC artifact without refitting."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_TRAIN_PATH = Path(__file__).with_name("train_peak_floor_contrastive_predictive.py")
_spec = importlib.util.spec_from_file_location("_pf_cpc_train", _TRAIN_PATH)
m = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = m
assert _spec.loader is not None; _spec.loader.exec_module(m)


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def score_run(artifact_dir: Path, morph_csv: Path, floor_csv: Path, output_csv: Path, *,
              device: str = "auto", batch_size: int = 256) -> dict:
    artifact_dir=Path(artifact_dir); morph_csv=Path(morph_csv); floor_csv=Path(floor_csv); output_csv=Path(output_csv)
    checkpoint=torch.load(artifact_dir/"model.pt",map_location="cpu",weights_only=True)
    cfg=m.ModelConfig(**checkpoint["model_config"])
    model=m.PeakFloorCPC(cfg); model.load_state_dict(checkpoint["model_state_dict"])
    dev=_device(device); model.to(dev)
    scalers_raw=json.loads((artifact_dir/"scalers.json").read_text())
    scalers={k:np.asarray(v,dtype=np.float32) for k,v in scalers_raw.items()}
    morph=pd.read_csv(morph_csv); floor=pd.read_csv(floor_csv)
    aligned=m.align_modalities(morph,floor,m.DEFAULT_MORPH_FEATURES,m.DEFAULT_FLOOR_FEATURES)
    pairs=m.make_predictive_pairs(m.apply_scalers(aligned,scalers),context_len=cfg.context_len)
    scores=m.score_pairs(model,pairs,batch_size,dev)
    calibration=pd.read_csv(artifact_dir/"calibration_scores.csv")["pf_cpc_surprisal"].to_numpy(float)
    scores["conformal_p_value"]=m._tail_pvalues(scores.pf_cpc_surprisal.to_numpy(),calibration)
    output_csv.parent.mkdir(parents=True,exist_ok=True); scores.to_csv(output_csv,index=False)
    result={
        "schema":"gnss-doppler-lab.peak-floor-cpc-score.v1",
        "windows":len(scores),
        "mean_surprisal":float(scores.pf_cpc_surprisal.mean()),
        "median_p_value":float(scores.conformal_p_value.median()),
        "fraction_p_le_0_05":float((scores.conformal_p_value<=0.05).mean()),
        "fraction_p_le_0_01":float((scores.conformal_p_value<=0.01).mean()),
        "model_sha256":m.sha256(artifact_dir/"model.pt"),
        "morph_sha256":m.sha256(morph_csv),"floor_sha256":m.sha256(floor_csv),
        "output_csv":str(output_csv),
    }
    (output_csv.with_suffix(".summary.json")).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir",type=Path,required=True); p.add_argument("--morph-csv",type=Path,required=True)
    p.add_argument("--floor-csv",type=Path,required=True); p.add_argument("--output-csv",type=Path,required=True)
    p.add_argument("--device",default="auto"); p.add_argument("--batch-size",type=int,default=256); return p.parse_args()


def main():
    a=parse_args(); print(json.dumps(score_run(a.artifact_dir,a.morph_csv,a.floor_csv,a.output_csv,
        device=a.device,batch_size=a.batch_size),indent=2,sort_keys=True))


if __name__ == "__main__": main()
