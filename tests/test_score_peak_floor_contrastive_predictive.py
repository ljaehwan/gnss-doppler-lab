from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train_peak_floor_contrastive_predictive.py"
SCORE = ROOT / "scripts" / "score_peak_floor_contrastive_predictive.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    assert spec.loader is not None; spec.loader.exec_module(module); return module


def frames(m, epochs=64, label="clean", scenario="clean"):
    mr=[]; fr=[]
    for i in range(epochs):
        t=.5*(i+1)
        for pi,prn in enumerate(("G01","G02","G03")):
            row={"run_id":"run","source_fingerprint":"source","label":label,"tap_count":9,
                 "prn":prn,"window_bin_s":t,"window_start_s":t-.48,"window_end_s":t}
            for j,c in enumerate(m.DEFAULT_MORPH_FEATURES): row[c]=np.sin(t/8+j/17)+pi*.03
            mr.append(row)
    for i in range(epochs+1):
        t=.5*i; row={"scenario":scenario,"window_start_s":t,"window_end_s":t+.01}
        for j,c in enumerate(m.DEFAULT_FLOOR_FEATURES): row[c]=np.cos(t/9+j/13)
        fr.append(row)
    return pd.DataFrame(mr),pd.DataFrame(fr)


def test_frozen_cpc_scores_attack_run_without_refitting(tmp_path):
    m=load(TRAIN,"pf_cpc_train_for_score_test")
    clean_m,clean_f=frames(m)
    cm=tmp_path/"cm.csv"; cf=tmp_path/"cf.csv"; clean_m.to_csv(cm,index=False); clean_f.to_csv(cf,index=False)
    artifact=tmp_path/"artifact"
    m.run_campaign(cm,cf,artifact,epochs=1,batch_size=8,context_len=4,hidden_dim=24,embedding_dim=12,
        token_layers=1,token_heads=4,split_rules={"train":(None,12.),"validation":(13.,18.),
        "calibration":(19.,24.),"held_clean":(25.,None)},device="cpu",seed=7)
    attack_m,attack_f=frames(m,epochs=20,label="attack",scenario="os-test")
    am=tmp_path/"am.csv"; af=tmp_path/"af.csv"; attack_m.to_csv(am,index=False); attack_f.to_csv(af,index=False)
    s=load(SCORE,"pf_cpc_score_test")
    out=tmp_path/"scores.csv"
    result=s.score_run(artifact,am,af,out,device="cpu",batch_size=8)
    scored=pd.read_csv(out)
    assert result["windows"]==len(scored)>0
    assert {"pf_cpc_surprisal","conformal_p_value","window_start_s","available_time_s"} <= set(scored)
    assert scored.conformal_p_value.between(0,1).all()
    assert result["model_sha256"]
