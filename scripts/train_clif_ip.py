#!/usr/bin/env python3
"""Prepare os1 M1 features and train the CLIF-only B0 on clean 0--240 s."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
LAB=Path("/home/ubuntu/projects/gnss-doppler-lab")
B0_SOURCE=LAB/"artifacts/oakbat_9tap_frozen_champion/cleanStatic/multi_prn_method_a_9tap_w1.0_s0.5_normalized_dmcpd/normal_prn_node_windows.csv"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()


def load_trainer():
    source=ROOT/"scripts/train_prn_node_gru.py"
    spec=importlib.util.spec_from_file_location("clif_b0_base",source)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod,source


def clean_train_frame(df):
    """Return only whole-contained CLIF clean-train windows."""
    return df[(df.window_start_s>=0)&(df.window_end_s<=240)].copy()


def training_input_digest(df, feature_cols):
    """Digest only clean-train inputs, so future/test mutations cannot affect it."""
    d=clean_train_frame(df).sort_values(["run_id","prn","window_bin_s"])
    cols=["run_id","prn","window_start_s","window_end_s","window_bin_s",*feature_cols]
    return hashlib.sha256(pd.util.hash_pandas_object(d[cols],index=False).values.tobytes()).hexdigest()


def prepare_os1(out,force=False):
    cache=out/"input_cache";cache.mkdir(parents=True,exist_ok=True)
    dest=cache/"oakbat_os1_raw_iq_noise_features.csv"
    raw=Path("/home/ubuntu/unraid_hdd/oakbat/gps_l1ca/raw/os1.bin")
    source=LAB/"scripts/iq_noise_continuity_detector.py"
    if force or not dest.exists():
        spec=importlib.util.spec_from_file_location("m1extract",source)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.FS=5_000_000
        df=mod.extract_feature_frame(raw,"os1",block_ms=10.,stride_s=.5,max_s=None);df.to_csv(dest,index=False)
    else: df=pd.read_csv(dest)
    manifest={
        "schema":"clif-ip.r3.preparation.v3","scenario":"os1","raw":str(raw),
        "raw_bytes":raw.stat().st_size,"raw_mtime_ns":raw.stat().st_mtime_ns,
        "raw_sha256":"e9ef8ab33a3e59c5e55b3f6fb9b8bb3ba18aaf380402ae00abbe535858b1deb7",
        "raw_hash_method":"cached canonical digest; current path/stat recorded but digest not reverified this run",
        "digest_reverified_this_run":False,"extractor_source":str(source),"extractor_sha256":sha(source),
        "sample_rate_hz":5000000,"sample_format":"interleaved int16 IQ","recording_start_sample":0,
        "seek_samples":0,"block_ms":10.,"stride_s":.5,"rows":len(df),"output":str(dest),
        "output_sha256":sha(dest),"note":"feature extraction only; no M1 fit"}
    (cache/"os1_extraction_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


def train_b0(out,epochs=40,batch_size=256):
    base,trainer_source=load_trainer();base.seed_all(11)
    raw=pd.read_csv(B0_SOURCE);d=clean_train_frame(raw)
    if d.empty or float(d.window_end_s.max())>240: raise RuntimeError("invalid clean-train B0 scope")
    cfg=base.TrainConfig(node_csv=str(B0_SOURCE),output_dir=str(out),seq_len=12,epochs=epochs,
                         batch_size=batch_size,feature_subset="tap_rel_prompt_mean")
    features=base.select_feature_columns(d,cfg.feature_subset)
    prns=sorted(d.prn.astype(str).unique());val_prns=set(prns[-max(1,len(prns)//5):])
    train=d[~d.prn.astype(str).isin(val_prns)].copy();val=d[d.prn.astype(str).isin(val_prns)].copy()
    mean,std=base.fit_standardizer(train[features].to_numpy(np.float32))
    train_ds=base.PrnSequenceDataset(base.build_series(train,features,mean,std),cfg.seq_len)
    val_ds=base.PrnSequenceDataset(base.build_series(val,features,mean,std),cfg.seq_len)
    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=base.PrnLocalGRU(len(features),cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    best=math.inf;history=[];ckpath=out/"clif_b0_prn_local_gru.pt"
    provenance={
        "scope":"CLIF clean train whole-contained windows only","min_window_start_s":float(d.window_start_s.min()),
        "max_window_end_s":float(d.window_end_s.max()),"selected_rows":len(d),"model_fit_rows":len(train),
        "validation_rows":len(val),"train_windows":len(train_ds),"validation_windows":len(val_ds),
        "train_prns":sorted(train.prn.astype(str).unique()),"validation_prns":sorted(val_prns),
        "training_input_sha256":training_input_digest(raw,features),"source_csv_sha256_live":sha(B0_SOURCE),
    }
    for epoch in range(1,epochs+1):
        tr=base.run_epoch(model,train_loader,opt,device);va=base.run_epoch(model,val_loader,None,device)
        row={"epoch":epoch,**{f"train_{k}":v for k,v in tr.items()},**{f"val_{k}":v for k,v in va.items()}}
        history.append(row);print(json.dumps(row,sort_keys=True),flush=True)
        if va["loss"]<best:
            best=va["loss"]
            torch.save({"model_state_dict":model.state_dict(),"config":asdict(cfg),
                "node_feature_columns":features,"standardizer":{"node_mean":mean.tolist(),"node_std":std.tolist()},
                "architecture_note":"existing B0 architecture retrained/frozen on CLIF clean train; shared PRN-local GRU; no PRN ID",
                "training_provenance":provenance},ckpath)
    pd.DataFrame(history).to_csv(out/"clif_b0_training_history.csv",index=False)
    metadata={"schema":"clif-ip.r3.b0-training.v1","architecture":"existing B0 architecture retrained/frozen on CLIF clean train",
        "fit_scope":provenance,"standardizer":{"fit_scope":"CLIF clean train rows only","fit_rows":len(train),
        "fit_prns":provenance["train_prns"]},"best_validation_loss":best,"epochs":epochs,"device":str(device),
        "source_csv":{"path":str(B0_SOURCE),"sha256_live":provenance["source_csv_sha256_live"]},
        "reused_loader_source":{"path":str(trainer_source),"sha256_live":sha(trainer_source)},
        "checkpoint":{"path":str(ckpath),"sha256":sha(ckpath)}}
    (out/"clif_b0_training_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=Path("artifacts/clif_ip_cross_layer_r3"))
    ap.add_argument("--force-os1",action="store_true");ap.add_argument("--skip-os1",action="store_true")
    ap.add_argument("--skip-b0",action="store_true");ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--batch-size",type=int,default=256);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if not a.skip_os1: prepare_os1(a.out,a.force_os1)
    if not a.skip_b0: train_b0(a.out,a.epochs,a.batch_size)

if __name__=="__main__":main()
