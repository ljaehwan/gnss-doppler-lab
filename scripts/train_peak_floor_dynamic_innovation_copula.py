#!/usr/bin/env python3
"""Train PF-DIC v3: train-frozen empirical Gaussian copula on clean data only."""
from __future__ import annotations
import argparse, json, math, platform, subprocess
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from gnss_doppler_lab.peak_floor_contract import (MAX_PRNS,MORPH_FEATURES,FLOOR_FEATURES,DEFAULT_SPLIT_RULES,
 align_modalities,validate_normal_only_inputs,partition_aligned,fit_robust_scalers,apply_scalers,make_causal_pairs,sha256)
from gnss_doppler_lab.peak_floor_dic import (ModelConfig,DynamicInnovationModel,PairDataset,masked_peak_summary,
 gaussian_nll,student_t_nll,innovation_nll,innovations,resolve_device,seed_all,validate_training_options,
 fit_pca_projection,transform_and_gaussianize,fit_relation,relation_nll,relation_scores,
 relation_deviation_scores,tail_pvalues,load_relation_projection)
SCHEMA="gnss-doppler-lab.pf-dic.v3"
ARTIFACT_FILES={"model.pt","scalers.json","relation.npz","training_history.csv","calibration_scores.csv","held_clean_scores.csv","calibration.json","permutation_diagnostic.json","provenance.json"}
_loss=innovation_nll

def train_model(train,val,cfg,epochs,batch_size,lr,device):
    model=DynamicInnovationModel(cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=lr)
    bestp=bestf=math.inf; bp=bf=None; hist=[]
    for ep in range(1,epochs+1):
        model.train(); p_sum=f_sum=0.; count=0
        for batch in DataLoader(PairDataset(train),batch_size=batch_size,shuffle=True):
            pc,fc,pt,ft=[x.to(device) for x in batch]; o=model(pc,fc)
            lp=_loss(pt,o["peak_mean"],o["peak_log_scale"],cfg); lf=_loss(ft,o["floor_mean"],o["floor_log_scale"],cfg)
            opt.zero_grad(); (lp+lf).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
            n=len(pt); p_sum+=float(lp.detach())*n; f_sum+=float(lf.detach())*n; count+=n
        model.eval(); vp_sum=vf_sum=0.; vcount=0
        with torch.no_grad():
            for batch in DataLoader(PairDataset(val),batch_size=batch_size):
                pc,fc,pt,ft=[x.to(device) for x in batch]; o=model(pc,fc); n=len(pt)
                vp_sum+=float(_loss(pt,o["peak_mean"],o["peak_log_scale"],cfg))*n
                vf_sum+=float(_loss(ft,o["floor_mean"],o["floor_log_scale"],cfg))*n; vcount+=n
        pv=vp_sum/vcount; fv=vf_sum/vcount
        hist.append({"epoch":ep,"train_peak_nll":p_sum/count,"train_floor_nll":f_sum/count,"validation_peak_nll":pv,"validation_floor_nll":fv})
        if pv<bestp: bestp=pv; bp={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith("peak_")}
        if fv<bestf: bestf=fv; bf={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith("floor_")}
    model.load_state_dict({**bp,**bf}); return model,hist,{"peak":bestp,"floor":bestf}

def score_frame(model,pairs,peak_projection,floor_projection,relation,batch,device,cal=None,relation_center=None):
    p,f,ps,fs=innovations(model,pairs,batch,device)
    p=transform_and_gaussianize(p,peak_projection); f=transform_and_gaussianize(f,floor_projection)
    rs=relation_scores(p,f,relation)
    frame=pd.DataFrame({"window_start_s":pairs.target_times,"available_time_s":pairs.available_times,"peak_score":ps,"floor_score":fs,"relation_score":rs,"joint_score":ps+fs+rs})
    if relation_center is not None: frame["relation_deviation_score"]=relation_deviation_scores(rs,relation_center)
    if cal is not None:
        for col in ("peak_score","floor_score","relation_score","relation_deviation_score","joint_score"):
            if col in frame and col in cal: frame[col.replace("score","p_value")]=tail_pvalues(frame[col].to_numpy(),cal[col].to_numpy())
    return frame,p,f

def valid_circular_shifts(n,guard,max_shifts=99):
    candidates=np.arange(max(1,int(guard)),max(int(guard),n-int(guard)+1),dtype=int)
    if len(candidates)<=max_shifts:return candidates.tolist()
    return np.unique(np.linspace(candidates[0],candidates[-1],max_shifts).round().astype(int)).tolist()

def _json(path,obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else (x.item() if isinstance(x,np.generic) else str(x)))+"\n")
def _git(root):
    try:return {"sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),"dirty":bool(subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).strip())}
    except Exception as exc:return {"sha":None,"dirty":None,"error":str(exc)}

def run_campaign(morph_csv,floor_csv,output_dir,*,epochs=20,batch_size=128,context_len=12,hidden_dim=64,pca_dim=8,rank=4,shrinkage=.1,distribution="gaussian",nu=5.,lr=1e-3,split_rules=None,device="auto",seed=17):
    validate_training_options(epochs=epochs,batch_size=batch_size,context_len=context_len,hidden_dim=hidden_dim,pca_dim=pca_dim,rank=rank,lr=lr,shrinkage=shrinkage)
    cfg=ModelConfig(2*len(MORPH_FEATURES),len(FLOOR_FEATURES),hidden_dim,1,0.,distribution,nu,context_len)
    dev,devmeta=resolve_device(device); seed_all(seed,deterministic=True); morph_csv=Path(morph_csv); floor_csv=Path(floor_csv); output_dir=Path(output_dir)
    if output_dir.exists():raise FileExistsError(f"refusing overwrite: {output_dir}")
    m=pd.read_csv(morph_csv); f=pd.read_csv(floor_csv); contract=validate_normal_only_inputs(m,f); aligned=align_modalities(m,f)
    rules=dict(split_rules or DEFAULT_SPLIT_RULES); parts=partition_aligned(aligned,rules); scalers=fit_robust_scalers(parts["train"])
    parts={k:apply_scalers(v,scalers) for k,v in parts.items()}; pairs={k:make_causal_pairs(v,context_len) for k,v in parts.items()}
    model,hist,best=train_model(pairs["train"],pairs["validation"],cfg,epochs,batch_size,lr,dev)
    raw_peak,raw_floor,_,_=innovations(model,pairs["train"],batch_size,dev)
    peak_projection=fit_pca_projection(raw_peak,pca_dim); floor_projection=fit_pca_projection(raw_floor,pca_dim)
    train_peak=transform_and_gaussianize(raw_peak,peak_projection); train_floor=transform_and_gaussianize(raw_floor,floor_projection)
    relation=fit_relation(train_peak,train_floor,min(rank,4),shrinkage)
    train_signed=relation_scores(train_peak,train_floor,relation); relation_center=float(np.median(train_signed))
    cal,_,_=score_frame(model,pairs["calibration"],peak_projection,floor_projection,relation,batch_size,dev,relation_center=relation_center)
    held,hp,hf=score_frame(model,pairs["held_clean"],peak_projection,floor_projection,relation,batch_size,dev,cal,relation_center)
    aligned_nll=float(relation_nll(hp,hf,relation).mean()); guard=context_len+1; shifts=valid_circular_shifts(len(hf),guard)
    shifted=[float(relation_nll(hp,np.roll(hf,k,0),relation).mean()) for k in shifts]
    permutation={"diagnostic_only":True,"null":"held-clean circular shift outside causal guard band; temporal dependence means this is not iid coverage","guard_band_epochs":guard,"tested_shifts":shifts,"aligned_mean_relation_nll":aligned_nll,"shifted_mean_relation_nll":shifted,"permutation_p_value":float((1+sum(v<=aligned_nll for v in shifted))/(1+len(shifted)))}
    output_dir.mkdir(parents=True,exist_ok=False)
    torch.save({"schema":SCHEMA,"model_state_dict":model.state_dict(),"model_config":asdict(cfg),"feature_contract":{"morph":MORPH_FEATURES,"floor":FLOOR_FEATURES},"frozen":True},output_dir/"model.pt")
    _json(output_dir/"scalers.json",scalers)
    np.savez(output_dir/"relation.npz",peak_mean=peak_projection["mean"],peak_components=peak_projection["components"],peak_references=peak_projection["references"],peak_numerical_rank=np.array(peak_projection["numerical_rank"]),floor_mean=floor_projection["mean"],floor_components=floor_projection["components"],floor_references=floor_projection["references"],floor_numerical_rank=np.array(floor_projection["numerical_rank"]),C=relation["C"],R=relation["R"],inverse=relation["inverse"],logdet=np.array(relation["logdet"]),relation_center=np.array(relation_center),train_relation_score_median=np.array(np.median(train_signed)))
    pd.DataFrame(hist).to_csv(output_dir/"training_history.csv",index=False); cal.to_csv(output_dir/"calibration_scores.csv",index=False); held.to_csv(output_dir/"held_clean_scores.csv",index=False)
    _json(output_dir/"calibration.json",{"schema":SCHEMA,"method":"empirical_upper_tail_rank","role":"reference_distribution_only; no fitted center or model parameters","coverage_disclaimer":"scores overlap in time and are time-dependent; p-values are empirical ranks, not guaranteed iid coverage","counts":len(cal)})
    _json(output_dir/"permutation_diagnostic.json",permutation)
    root=Path(__file__).resolve().parents[1]; scorer=root/"scripts/score_peak_floor_dynamic_innovation_copula.py"; shared=root/"src/gnss_doppler_lab/peak_floor_dic.py"; contract_path=root/"src/gnss_doppler_lab/peak_floor_contract.py"
    provenance={"schema":SCHEMA,"model_name":"PF-DIC","normal_only_training":True,"integrity_notice":"SHA-256 hashes detect accidental corruption; this manifest is not a security signature or authenticity proof.","contract":contract,"training":{"seed":seed,"epochs":epochs,"batch_size":batch_size,"learning_rate":lr,"context_len":context_len,"deterministic_algorithms":True},"sources":{"morph":{"path":str(morph_csv.resolve()),"sha256":sha256(morph_csv)},"floor":{"path":str(floor_csv.resolve()),"sha256":sha256(floor_csv)}},"git":_git(root),"packages":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"torch":torch.__version__},"device":devmeta,"feature_order":{"morph":MORPH_FEATURES,"floor":FLOOR_FEATURES},"splits":{k:{"epochs":len(parts[k].times),"pairs":len(pairs[k]),"rule":rules[k]} for k in rules},"model_config":asdict(cfg),"relation":{"requested_pca_dim":pca_dim,"peak_dim":len(peak_projection["components"]),"floor_dim":len(floor_projection["components"]),"rank":relation["rank"],"shrinkage":shrinkage,"gaussianization":"train-only empirical PIT to normal score","relation_center_fit_scope":"train_signed_score_median","nested_null":"R=I"},"source_snapshot_hashes":{"contract":sha256(contract_path),"shared":sha256(shared),"trainer":sha256(Path(__file__)),"scorer":sha256(scorer)}}
    provenance["output_file_hashes_before_provenance"]={p.name:sha256(p) for p in output_dir.iterdir() if p.is_file()}; _json(output_dir/"provenance.json",provenance)
    files={p.name:sha256(p) for p in output_dir.iterdir() if p.is_file()}
    if set(files)!=ARTIFACT_FILES: raise RuntimeError("internal artifact allowlist mismatch")
    _json(output_dir/"campaign_manifest.json",{"schema":SCHEMA,"hash_algorithm":"sha256","integrity_purpose":"accidental corruption detection only; not a security signature","artifacts":files})
    return {"output_dir":str(output_dir),"held_clean":{"windows":len(held),"mean_relation_p_value":float(held.relation_p_value.mean())},"best_validation":best}

def _split_rules(value):
    try:
        text=Path(value).read_text() if Path(value).is_file() else value; raw=json.loads(text)
        if set(raw)!={"train","validation","calibration","held_clean"}: raise ValueError("split rules require train, validation, calibration, held_clean")
        return {k:tuple(v) for k,v in raw.items()}
    except (OSError,json.JSONDecodeError,TypeError) as exc: raise argparse.ArgumentTypeError(f"invalid split-rules JSON/path: {exc}") from exc

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--morph-csv",type=Path,required=True); p.add_argument("--floor-csv",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--context-len",type=int,default=12); p.add_argument("--hidden-dim",type=int,default=64); p.add_argument("--pca-dim",type=int,default=8); p.add_argument("--rank",type=int,default=4); p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--shrinkage",type=float,default=.1); p.add_argument("--distribution",choices=sorted({"gaussian","student_t"}),default="gaussian"); p.add_argument("--nu",type=float,default=5.); p.add_argument("--split-rules",type=_split_rules); p.add_argument("--device",default="auto"); p.add_argument("--seed",type=int,default=17)
    args=p.parse_args(argv)
    try: validate_training_options(epochs=args.epochs,batch_size=args.batch_size,context_len=args.context_len,hidden_dim=args.hidden_dim,pca_dim=args.pca_dim,rank=args.rank,lr=args.lr,shrinkage=args.shrinkage); ModelConfig(2*len(MORPH_FEATURES),len(FLOOR_FEATURES),args.hidden_dim,distribution=args.distribution,nu=args.nu,context_len=args.context_len)
    except ValueError as exc: p.error(str(exc))
    return args

def main():
    a=parse_args(); print(json.dumps(run_campaign(a.morph_csv,a.floor_csv,a.output_dir,epochs=a.epochs,batch_size=a.batch_size,context_len=a.context_len,hidden_dim=a.hidden_dim,pca_dim=a.pca_dim,rank=a.rank,lr=a.lr,shrinkage=a.shrinkage,distribution=a.distribution,nu=a.nu,split_rules=a.split_rules,device=a.device,seed=a.seed),indent=2))
if __name__=="__main__":main()
