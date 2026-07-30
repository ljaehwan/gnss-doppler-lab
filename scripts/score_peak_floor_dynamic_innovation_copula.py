#!/usr/bin/env python3
"""Strict integrity/semantics-checked frozen scorer for PF-DIC schema v3."""
from __future__ import annotations
import argparse, json, math, os, tempfile
from pathlib import Path, PurePosixPath
import numpy as np
import pandas as pd
import torch
from gnss_doppler_lab.peak_floor_contract import MORPH_FEATURES,FLOOR_FEATURES,align_modalities,apply_scalers,make_causal_pairs,sha256
from gnss_doppler_lab.peak_floor_dic import (ModelConfig,DynamicInnovationModel,PairDataset,masked_peak_summary,
 innovation_nll,innovations,resolve_device,transform_and_gaussianize,relation_scores,relation_deviation_scores,
 tail_pvalues,projection_from_npz)
SCHEMA="gnss-doppler-lab.pf-dic.v3"
SCORE_SCHEMA="gnss-doppler-lab.pf-dic-score.v3"
ALLOWED_ARTIFACTS={"model.pt","scalers.json","relation.npz","training_history.csv","calibration_scores.csv","held_clean_scores.csv","calibration.json","permutation_diagnostic.json","provenance.json"}
CAL_COLUMNS={"window_start_s","available_time_s","peak_score","floor_score","relation_score","relation_deviation_score","joint_score"}
summary=masked_peak_summary; DS=PairDataset

def nll(y,mu,ls,cfg): return innovation_nll(y,mu,ls,cfg,"none")
def _safe_name(name):
    p=PurePosixPath(name)
    return bool(name) and not p.is_absolute() and len(p.parts)==1 and p.parts[0] not in (".","..") and "\\" not in name

def _finite_array(value,name,shape=None,positive=False):
    a=np.asarray(value)
    if shape is not None and a.shape!=shape: raise ValueError(f"{name} shape mismatch: {a.shape} != {shape}")
    if not np.issubdtype(a.dtype,np.number) or not np.isfinite(a).all(): raise ValueError(f"{name} must be finite numeric")
    if positive and not (a>0).all(): raise ValueError(f"{name} must be positive")
    return a

def verify_artifact(directory):
    d=Path(directory); manifest_path=d/"campaign_manifest.json"
    try: manifest=json.loads(manifest_path.read_text())
    except Exception as exc: raise ValueError(f"invalid manifest: {exc}") from exc
    if manifest.get("schema")!=SCHEMA: raise ValueError(f"manifest schema must be {SCHEMA}")
    if manifest.get("hash_algorithm")!="sha256": raise ValueError("manifest hash_algorithm must be sha256")
    artifacts=manifest.get("artifacts")
    if not isinstance(artifacts,dict) or set(artifacts)!=ALLOWED_ARTIFACTS: raise ValueError(f"manifest artifact allowlist mismatch; required exact set {sorted(ALLOWED_ARTIFACTS)}")
    for name,want in artifacts.items():
        if not _safe_name(name): raise ValueError(f"manifest filename/path is not allowed: {name!r}")
        if not isinstance(want,str) or len(want)!=64 or any(c not in "0123456789abcdef" for c in want): raise ValueError(f"invalid sha256 for {name}")
        path=d/name
        if not path.is_file() or sha256(path)!=want: raise ValueError(f"artifact hash/integrity/corruption failure: {name}")
    try: provenance=json.loads((d/"provenance.json").read_text())
    except Exception as exc: raise ValueError(f"invalid provenance: {exc}") from exc
    if provenance.get("schema")!=SCHEMA or provenance.get("model_name")!="PF-DIC": raise ValueError("provenance schema/model mismatch")
    snapshots=provenance.get("source_snapshot_hashes",{})
    valid_hex=lambda x:isinstance(x,str) and len(x)==64 and all(c in "0123456789abcdef" for c in x)
    if set(snapshots)!={"trainer","scorer","contract","shared"} or any(not valid_hex(x) for x in snapshots.values()): raise ValueError("provenance source snapshot hashes invalid")
    try: ck=torch.load(d/"model.pt",map_location="cpu",weights_only=True)
    except Exception as exc: raise ValueError(f"invalid model artifact: {exc}") from exc
    if ck.get("schema")!=SCHEMA or not ck.get("frozen") or ck.get("feature_contract")!={"morph":MORPH_FEATURES,"floor":FLOOR_FEATURES}: raise ValueError("frozen model schema/feature contract mismatch")
    state=ck.get("model_state_dict")
    if not isinstance(state,dict) or not state: raise ValueError("model state_dict is missing or empty")
    for name,value in state.items():
        if not isinstance(value,torch.Tensor) or not torch.isfinite(value).all(): raise ValueError(f"model weight must be a finite tensor: {name}")
    try: cfg=ModelConfig(**ck["model_config"])
    except Exception as exc: raise ValueError(f"invalid model config: {exc}") from exc
    if cfg.peak_input_dim!=2*len(MORPH_FEATURES) or cfg.floor_input_dim!=len(FLOOR_FEATURES): raise ValueError("model config feature dimensions mismatch")
    try: raw=json.loads((d/"scalers.json").read_text())
    except Exception as exc: raise ValueError(f"invalid scalers: {exc}") from exc
    if set(raw)!={"morph_median","morph_scale","floor_median","floor_scale","fit_scope"} or raw.get("fit_scope")!="train_only": raise ValueError("scaler schema/provenance mismatch")
    _finite_array(raw["morph_median"],"morph_median",(len(MORPH_FEATURES),)); _finite_array(raw["morph_scale"],"morph_scale",(len(MORPH_FEATURES),),True)
    _finite_array(raw["floor_median"],"floor_median",(len(FLOOR_FEATURES),)); _finite_array(raw["floor_scale"],"floor_scale",(len(FLOOR_FEATURES),),True)
    required_npz={"peak_mean","peak_components","peak_references","peak_numerical_rank","floor_mean","floor_components","floor_references","floor_numerical_rank","C","R","inverse","logdet","relation_center","train_relation_score_median"}
    try:
        with np.load(d/"relation.npz",allow_pickle=False) as z:
            if set(z.files)!=required_npz: raise ValueError("relation npz schema mismatch")
            pm=_finite_array(z["peak_mean"],"peak_mean"); pc=_finite_array(z["peak_components"],"peak_components"); pr=_finite_array(z["peak_references"],"peak_references")
            fm=_finite_array(z["floor_mean"],"floor_mean"); fc=_finite_array(z["floor_components"],"floor_components"); fr=_finite_array(z["floor_references"],"floor_references")
            if pm.shape!=(cfg.peak_input_dim,) or fm.shape!=(cfg.floor_input_dim,) or pc.ndim!=2 or fc.ndim!=2 or pc.shape[1]!=len(pm) or fc.shape[1]!=len(fm): raise ValueError("PCA projection shape mismatch")
            if pr.ndim!=2 or fr.ndim!=2 or pr.shape[1]!=pc.shape[0] or fr.shape[1]!=fc.shape[0] or len(pr)<2 or len(fr)<2: raise ValueError("Gaussianization reference shape mismatch")
            if np.any(np.diff(pr,axis=0)<0) or np.any(np.diff(fr,axis=0)<0): raise ValueError("Gaussianization references must be sorted")
            if not np.allclose(pc@pc.T,np.eye(len(pc)),rtol=1e-7,atol=1e-8) or not np.allclose(fc@fc.T,np.eye(len(fc)),rtol=1e-7,atol=1e-8): raise ValueError("PCA components are not orthonormal")
            c=_finite_array(z["C"],"C",(len(pc),len(fc))); R=_finite_array(z["R"],"R",(len(pc)+len(fc),)*2); stored_inv=_finite_array(z["inverse"],"inverse",R.shape)
            stored_ld=float(_finite_array(z["logdet"],"logdet")); center=float(_finite_array(z["relation_center"],"relation_center")); train_center=float(_finite_array(z["train_relation_score_median"],"train relation median"))
            expected=np.block([[np.eye(len(pc)),c],[c.T,np.eye(len(fc))]])
            if not np.allclose(R,R.T,rtol=1e-10,atol=1e-10) or not np.allclose(R,expected,rtol=1e-8,atol=1e-10): raise ValueError("relation R is not symmetric/consistent")
            try: chol=np.linalg.cholesky(R)
            except np.linalg.LinAlgError as exc: raise ValueError("relation R is not positive definite") from exc
            inverse=np.linalg.inv(R); logdet=2*np.log(np.diag(chol)).sum()
            if not np.allclose(stored_inv,inverse,rtol=1e-7,atol=1e-9) or not np.isclose(stored_ld,logdet,rtol=1e-8,atol=1e-10): raise ValueError("relation inverse/logdet inconsistent with R")
            if not np.isclose(center,train_center,rtol=0,atol=1e-12): raise ValueError("relation center is not frozen train median")
    except ValueError: raise
    except Exception as exc: raise ValueError(f"invalid relation artifact: {exc}") from exc
    try: cal=pd.read_csv(d/"calibration_scores.csv")
    except Exception as exc: raise ValueError(f"invalid calibration scores: {exc}") from exc
    if cal.empty or not CAL_COLUMNS<=set(cal): raise ValueError("calibration must be nonempty with required score columns")
    values=cal[list(CAL_COLUMNS)].apply(pd.to_numeric,errors="coerce").to_numpy()
    if not np.isfinite(values).all(): raise ValueError("calibration required columns must be finite")
    return manifest

def _atomic_write(frame,output_csv,meta,sidecar):
    output_csv.parent.mkdir(parents=True,exist_ok=True); csv_tmp=side_tmp=None
    try:
        fd,csv_name=tempfile.mkstemp(prefix=f".{output_csv.name}.",suffix=".tmp",dir=output_csv.parent); os.close(fd); csv_tmp=Path(csv_name)
        frame.to_csv(csv_tmp,index=False); meta["output_sha256"]=sha256(csv_tmp)
        fd,side_name=tempfile.mkstemp(prefix=f".{sidecar.name}.",suffix=".tmp",dir=sidecar.parent); os.close(fd); side_tmp=Path(side_name)
        side_tmp.write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n")
        if output_csv.exists() or sidecar.exists(): raise FileExistsError(f"refusing overwrite: {output_csv} or {sidecar}")
        os.replace(csv_tmp,output_csv); csv_tmp=None; os.replace(side_tmp,sidecar); side_tmp=None
    finally:
        for tmp in (csv_tmp,side_tmp):
            if tmp is not None: tmp.unlink(missing_ok=True)

def score_run(artifact_dir,morph_csv,floor_csv,output_csv,*,device="auto",batch_size=128,onset_s=None):
    artifact_dir=Path(artifact_dir); morph_csv=Path(morph_csv); floor_csv=Path(floor_csv); output_csv=Path(output_csv); sidecar=output_csv.with_suffix(output_csv.suffix+".provenance.json")
    if batch_size<1: raise ValueError("batch_size must be positive")
    if onset_s is not None:
        onset_s=float(onset_s)
        if not math.isfinite(onset_s) or onset_s<0: raise ValueError("onset_s must be finite and non-negative")
    if output_csv.exists() or sidecar.exists():raise FileExistsError(f"refusing overwrite: {output_csv} or {sidecar}")
    before={"morph":sha256(morph_csv),"floor":sha256(floor_csv)}; manifest=verify_artifact(artifact_dir)
    ck=torch.load(artifact_dir/"model.pt",map_location="cpu",weights_only=True); cfg=ModelConfig(**ck["model_config"])
    model=DynamicInnovationModel(cfg); model.load_state_dict(ck["model_state_dict"]); dev,devmeta=resolve_device(device); model.to(dev).eval()
    raw=json.loads((artifact_dir/"scalers.json").read_text()); scalers={k:np.asarray(v,np.float32) if k!="fit_scope" else v for k,v in raw.items()}
    morph=pd.read_csv(morph_csv); floor=pd.read_csv(floor_csv); after={"morph":sha256(morph_csv),"floor":sha256(floor_csv)}
    if before!=after: raise RuntimeError("input mutation detected while reading CSV")
    pairs=make_causal_pairs(apply_scalers(align_modalities(morph,floor,validate_clean=False),scalers),cfg.context_len)
    with np.load(artifact_dir/"relation.npz",allow_pickle=False) as z:
        pp=projection_from_npz(z,"peak"); fp=projection_from_npz(z,"floor"); relation={"inverse":np.linalg.inv(z["R"]),"logdet":float(np.linalg.slogdet(z["R"])[1])}; center=float(z["relation_center"])
    P,F,PS,FS=innovations(model,pairs,batch_size,dev)
    for name,value in (("peak innovations",P),("floor innovations",F),("peak scores",PS),("floor scores",FS)):
        if not np.isfinite(value).all(): raise ValueError(f"{name} must be finite")
    P=transform_and_gaussianize(P,pp); F=transform_and_gaussianize(F,fp); rs=relation_scores(P,F,relation); rds=relation_deviation_scores(rs,center)
    out=pd.DataFrame({"window_start_s":pairs.target_times,"available_time_s":pairs.available_times,"peak_score":PS,"floor_score":FS,"relation_score":rs,"relation_deviation_score":rds,"joint_score":PS+FS+rs})
    score_columns=("peak_score","floor_score","relation_score","relation_deviation_score","joint_score")
    if not np.isfinite(out[list(score_columns)].to_numpy()).all(): raise ValueError("final score columns must be finite")
    cal=pd.read_csv(artifact_dir/"calibration_scores.csv")
    for col in score_columns: out[col.replace("score","p_value")]=tail_pvalues(out[col].to_numpy(),cal[col].to_numpy())
    if onset_s is not None:
        onset=float(onset_s); out["support_class"]=np.where(pairs.available_times<onset,"pre",np.where(pairs.support_start_times>=onset,"post","uncertain"))
    else:out["support_class"]="uncertain"
    meta={"schema":SCORE_SCHEMA,"windows":len(out),"artifact_manifest_sha256":sha256(artifact_dir/"campaign_manifest.json"),"model_sha256":manifest["artifacts"]["model.pt"],"input_hashes":after,"frozen":True,"refit":False,"onset_s":onset_s,"device":devmeta,"requested_device":device,"resolved_device":str(dev),"batch_size":batch_size,"scorer_source_sha256":sha256(Path(__file__)),"integrity_notice":"SHA-256 detects accidental corruption; it is not a security signature."}
    _atomic_write(out,output_csv,meta,sidecar); return meta

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--artifact-dir",type=Path,required=True);p.add_argument("--morph-csv",type=Path,required=True);p.add_argument("--floor-csv",type=Path,required=True);p.add_argument("--output-csv",type=Path,required=True);p.add_argument("--device",default="auto");p.add_argument("--batch-size",type=int,default=128);p.add_argument("--onset-s",type=float);return p.parse_args(argv)
def main():
    a=parse_args();print(json.dumps(score_run(a.artifact_dir,a.morph_csv,a.floor_csv,a.output_csv,device=a.device,batch_size=a.batch_size,onset_s=a.onset_s),indent=2))
if __name__=="__main__":main()
