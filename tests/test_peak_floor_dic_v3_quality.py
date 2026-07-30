from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/"scripts"/"train_peak_floor_dynamic_innovation_copula.py"
SCORE=ROOT/"scripts"/"score_peak_floor_dynamic_innovation_copula.py"

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

def frames(m,n=64,label="clean",scenario="clean"):
    mr=[]; fr=[]
    for i in range(n):
        t=.5*(i+1)
        for k,prn in enumerate(("G01","G02","G03")):
            r={"label":label,"run_id":"r","source_fingerprint":"s","tap_count":9,"prn":prn,"window_bin_s":t,"window_start_s":t-.48,"window_end_s":t+.02}
            r.update({c:np.sin(t/7+j/19)+k*.02 for j,c in enumerate(m.MORPH_FEATURES)}); mr.append(r)
        q={"scenario":scenario,"window_start_s":t,"window_end_s":t+.04}; q.update({c:np.cos(t/9+j/17) for j,c in enumerate(m.FLOOR_FEATURES)}); fr.append(q)
    return pd.DataFrame(mr),pd.DataFrame(fr)

def test_empirical_gaussianization_and_rank_aware_projection():
    from gnss_doppler_lab import peak_floor_dic as d
    rng=np.random.default_rng(7); x=rng.standard_t(5,size=(1001,4)); x=np.c_[x,x[:,0]]
    projection=d.fit_pca_projection(x,10)
    assert projection["components"].shape[0]==4
    assert np.allclose(projection["components"]@projection["components"].T,np.eye(4),atol=1e-10)
    z=d.transform_and_gaussianize(x,projection)
    assert np.isfinite(z).all() and np.max(abs(z.mean(0)))<.02
    assert np.max(abs(z.std(0)-1))<.03

def test_config_ranges_distribution_nu_and_explicit_cuda(monkeypatch):
    from gnss_doppler_lab import peak_floor_dic as d
    with pytest.raises(ValueError,match="distribution"): d.ModelConfig(2,2,distribution="student")
    with pytest.raises(ValueError,match="nu"): d.ModelConfig(2,2,distribution="student_t",nu=2)
    with pytest.raises(ValueError): d.validate_training_options(epochs=0,batch_size=1,context_len=1,hidden_dim=1,pca_dim=1,rank=1,lr=.1,shrinkage=.1)
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    with pytest.raises(RuntimeError,match="CUDA.*unavailable"): d.resolve_device("cuda")
    dev,meta=d.resolve_device("auto"); assert str(dev)=="cpu" and meta["requested"]=="auto" and meta["resolved"]=="cpu"

def test_time_nan_rejected_and_cli_options_visible():
    from gnss_doppler_lab.peak_floor_contract import align_modalities
    m=load(TRAIN,"pfdic_quality_cli"); a,b=frames(m,8); a.loc[0,"window_bin_s"]=np.nan
    with pytest.raises(ValueError,match="time|finite"): align_modalities(a,b)
    help_text=Path(TRAIN).read_text()
    for option in ("--distribution","--nu","--lr","--shrinkage","--split-rules"):
        assert option in help_text

@pytest.fixture()
def artifact(tmp_path):
    m=load(TRAIN,"pfdic_quality_train"); a,b=frames(m)
    cm=tmp_path/"cm.csv"; cf=tmp_path/"cf.csv"; a.to_csv(cm,index=False); b.to_csv(cf,index=False)
    art=tmp_path/"artifact"
    m.run_campaign(cm,cf,art,epochs=1,batch_size=8,context_len=3,hidden_dim=12,pca_dim=4,rank=2,device="cpu",seed=2,split_rules={"train":(None,12.),"validation":(12.5,18.),"calibration":(18.5,24.),"held_clean":(24.5,None)})
    return m,art,cm,cf

def rehash(art,name):
    p=art/name; h=hashlib.sha256(p.read_bytes()).hexdigest(); manifest=json.loads((art/"campaign_manifest.json").read_text()); manifest["artifacts"][name]=h; (art/"campaign_manifest.json").write_text(json.dumps(manifest))

def test_v3_train_center_and_trainer_scorer_numerical_equivalence(artifact,tmp_path):
    m,art,cm,cf=artifact; s=load(SCORE,"pfdic_quality_score")
    manifest=json.loads((art/"campaign_manifest.json").read_text()); assert manifest["schema"].endswith(".v3")
    rel=np.load(art/"relation.npz"); center=float(rel["relation_center"])
    # relation center is frozen from train, not calibration.
    assert np.isclose(center,float(rel["train_relation_score_median"]))
    cal=pd.read_csv(art/"calibration_scores.csv"); assert not np.isclose(center,float(cal.relation_score.median()),rtol=0,atol=1e-12)
    out=tmp_path/"score.csv"; s.score_run(art,cm,cf,out,device="cpu",batch_size=7)
    frozen=pd.read_csv(out)
    # Score the same full input through trainer implementation and compare all score columns.
    ck=torch.load(art/"model.pt",map_location="cpu",weights_only=True); cfg=m.ModelConfig(**ck["model_config"]); model=m.DynamicInnovationModel(cfg); model.load_state_dict(ck["model_state_dict"])
    raw=json.loads((art/"scalers.json").read_text()); sc={k:np.asarray(v,np.float32) if k!="fit_scope" else v for k,v in raw.items()}
    pairs=m.make_causal_pairs(m.apply_scalers(m.align_modalities(pd.read_csv(cm),pd.read_csv(cf)),sc),cfg.context_len)
    proj=m.load_relation_projection(art/"relation.npz"); train_df,_,_=m.score_frame(model,pairs,proj["peak"],proj["floor"],proj["relation"],7,torch.device("cpu"),pd.read_csv(art/"calibration_scores.csv"),center)
    cols=[c for c in train_df if c.endswith("score") or c.endswith("p_value")]
    assert np.allclose(train_df[cols],frozen[cols],rtol=1e-6,atol=1e-6)

def test_semantic_validation_and_manifest_traversal_even_when_rehashed(artifact,tmp_path):
    _,art,_,_=artifact; s=load(SCORE,"pfdic_quality_validate")
    # Hash-valid but semantically invalid covariance must fail.
    z=dict(np.load(art/"relation.npz")); z["R"]=z["R"].copy(); z["R"][0,0]=-1
    np.savez(art/"relation.npz",**z); rehash(art,"relation.npz")
    with pytest.raises(ValueError,match="positive definite|symmetric|relation"): s.verify_artifact(art)
    # Traversal names are forbidden before accessing them.
    manifest=json.loads((art/"campaign_manifest.json").read_text()); manifest["artifacts"]["../outside"]="0"*64; (art/"campaign_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="filename|path|allowlist"): s.verify_artifact(art)

def test_model_weights_scores_and_onset_must_be_finite(artifact,tmp_path,monkeypatch):
    _,art,cm,cf=artifact; s=load(SCORE,"pfdic_quality_finite")
    ck=torch.load(art/"model.pt",map_location="cpu",weights_only=True)
    key=next(iter(ck["model_state_dict"])); ck["model_state_dict"][key].view(-1)[0]=float("nan")
    torch.save(ck,art/"model.pt"); rehash(art,"model.pt")
    with pytest.raises(ValueError,match="weight|finite|model"):
        s.verify_artifact(art)

    # Rebuild a clean fixture, then verify non-finite onset is rejected before scoring.
    m=load(TRAIN,"pfdic_quality_finite_rebuild");a,b=frames(m);cm2=tmp_path/"cm2.csv";cf2=tmp_path/"cf2.csv";a.to_csv(cm2,index=False);b.to_csv(cf2,index=False)
    art2=tmp_path/"artifact2";m.run_campaign(cm2,cf2,art2,epochs=1,batch_size=8,context_len=3,hidden_dim=12,pca_dim=4,rank=2,device="cpu",seed=2,split_rules={"train":(None,12.),"validation":(12.5,18.),"calibration":(18.5,24.),"held_clean":(24.5,None)})
    with pytest.raises(ValueError,match="onset.*finite"):
        s.score_run(art2,cm2,cf2,tmp_path/"nan-onset.csv",device="cpu",onset_s=float("nan"))

    original=s.innovations
    def bad_innovations(*args,**kwargs):
        P,F,PS,FS=original(*args,**kwargs);P=P.copy();P[0,0]=np.nan;return P,F,PS,FS
    monkeypatch.setattr(s,"innovations",bad_innovations)
    with pytest.raises(ValueError,match="score|innovation|finite"):
        s.score_run(art2,cm2,cf2,tmp_path/"nan-score.csv",device="cpu")
