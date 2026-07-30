from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
import numpy as np,pandas as pd,pytest
ROOT=Path(__file__).resolve().parents[1]; TRAIN=ROOT/"scripts"/"train_peak_floor_dynamic_innovation_copula.py"; SCORE=ROOT/"scripts"/"score_peak_floor_dynamic_innovation_copula.py"
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def frames(m,n=64,label="clean",scenario="clean"):
 mr=[];fr=[]
 for i in range(n):
  t=.5*(i+1)
  for k,p in enumerate(("G01","G02","G03")):
   r={"label":label,"run_id":"r","source_fingerprint":"s","tap_count":9,"prn":p,"window_bin_s":t,"window_start_s":t-.48,"window_end_s":t+.02};r.update({c:np.sin(t/7+j/19)+k*.02 for j,c in enumerate(m.MORPH_FEATURES)});mr.append(r)
  q={"scenario":scenario,"window_start_s":t,"window_end_s":t+.04};q.update({c:np.cos(t/9+j/17) for j,c in enumerate(m.FLOOR_FEATURES)});fr.append(q)
 return pd.DataFrame(mr),pd.DataFrame(fr)
def test_tiny_cpu_campaign_frozen_score_and_tamper_rejection(tmp_path):
 m=load(TRAIN,"pfdic_t"); a,b=frames(m); cm=tmp_path/"cm.csv";cf=tmp_path/"cf.csv";a.to_csv(cm,index=False);b.to_csv(cf,index=False)
 art=tmp_path/"artifact"; m.run_campaign(cm,cf,art,epochs=1,batch_size=8,context_len=3,hidden_dim=12,pca_dim=4,rank=2,device="cpu",seed=2,split_rules={"train":(None,12.),"validation":(12.5,18.),"calibration":(18.5,24.),"held_clean":(24.5,None)})
 required={"model.pt","scalers.json","relation.npz","calibration_scores.csv","calibration.json","held_clean_scores.csv","permutation_diagnostic.json","provenance.json","campaign_manifest.json"}
 assert required<={p.name for p in art.iterdir()}
 s=load(SCORE,"pfdic_s"); am,af=frames(m,24,"attack","os2"); x=tmp_path/"am.csv";y=tmp_path/"af.csv";am.to_csv(x,index=False);af.to_csv(y,index=False);out=tmp_path/"scores.csv"
 result=s.score_run(art,x,y,out,device="cpu",batch_size=7,onset_s=8.)
 z=pd.read_csv(out); assert result["windows"]==len(z)>0
 assert {"peak_score","floor_score","joint_score","relation_score","relation_deviation_score","relation_p_value","relation_deviation_p_value","support_class"}<=set(z)
 assert set(z.support_class)<={"pre","post","uncertain"}
 # frozen scorer must reject any artifact mutation before loading/scoring
 with (art/"scalers.json").open("a") as fh: fh.write(" ")
 with pytest.raises(ValueError,match="hash|integrity|tamper"): s.score_run(art,x,y,tmp_path/"bad.csv",device="cpu")

def test_manifest_cannot_omit_required_frozen_artifact(tmp_path):
    m=load(TRAIN,"pfdic_t_required"); a,b=frames(m); cm=tmp_path/"cm.csv";cf=tmp_path/"cf.csv";a.to_csv(cm,index=False);b.to_csv(cf,index=False)
    art=tmp_path/"artifact";m.run_campaign(cm,cf,art,epochs=1,batch_size=8,context_len=3,hidden_dim=12,pca_dim=4,rank=2,device="cpu",seed=2,split_rules={"train":(None,12.),"validation":(12.5,18.),"calibration":(18.5,24.),"held_clean":(24.5,None)})
    manifest=json.loads((art/"campaign_manifest.json").read_text());manifest["artifacts"].pop("relation.npz");(art/"campaign_manifest.json").write_text(json.dumps(manifest))
    s=load(SCORE,"pfdic_s_required")
    with pytest.raises(ValueError,match="required|manifest"):
        s.verify_artifact(art)

def test_post_label_uses_raw_support_start_and_sidecar_refuses_overwrite(tmp_path):
    m=load(TRAIN,"pfdic_t_support");a,b=frames(m);cm=tmp_path/"cm.csv";cf=tmp_path/"cf.csv";a.to_csv(cm,index=False);b.to_csv(cf,index=False)
    art=tmp_path/"artifact";m.run_campaign(cm,cf,art,epochs=1,batch_size=8,context_len=3,hidden_dim=12,pca_dim=4,rank=2,device="cpu",seed=2,split_rules={"train":(None,12.),"validation":(12.5,18.),"calibration":(18.5,24.),"held_clean":(24.5,None)})
    s=load(SCORE,"pfdic_s_support");am,af=frames(m,24,"attack","os2");x=tmp_path/"am.csv";y=tmp_path/"af.csv";am.to_csv(x,index=False);af.to_csv(y,index=False)
    out=tmp_path/"scores.csv";side=out.with_suffix(out.suffix+".provenance.json");side.write_text("occupied")
    with pytest.raises(FileExistsError,match="overwrite"):
        s.score_run(art,x,y,out,device="cpu",onset_s=2.5)
