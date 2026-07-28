#!/usr/bin/env python3
"""Train and freeze the normal-only OAKBAT cleanStatic detector.

The command has no attack, raw-IQ, or scenario inputs.  It chronologically
splits one authenticated clean node CSV, trains on the first partition,
selects by validation loss, calibrates the frozen checkpoint, and finally
measures held-clean false positives.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
def _load_script(name):
 path=ROOT/"scripts"/name; spec=importlib.util.spec_from_file_location("_oakbat_"+path.stem,path); module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); return module
train_lib=_load_script("train_prn_node_gru.py")
gate_lib=_load_script("eval_btail_support_gate.py")
FEATURE_COLUMNS=["tap_E4_rel_prompt_mean","tap_E3_rel_prompt_mean","tap_E2_rel_prompt_mean","tap_E_rel_prompt_mean","tap_P_rel_prompt_mean","tap_L_rel_prompt_mean","tap_L2_rel_prompt_mean","tap_L3_rel_prompt_mean","tap_L4_rel_prompt_mean"]
TAP_LAYOUT="E4,E3,E2,E,P,L,L2,L3,L4"
PARTITION_RULES={"train":(None,240.0),"validation":(250.0,330.0),"calibration":(340.0,410.0),"held_clean":(420.0,None)}
PURGES=((240.0,250.0),(330.0,340.0),(410.0,420.0))
SCHEMA="gnss-doppler-lab.oakbat-cleanstatic-freeze.v1"
NODE_SCHEMA="gnss-doppler-lab.method-a-9tap-multi-prn-dataset"
FEATURE_CACHE_SCHEMA="gnss-doppler-lab.oakbat-feature-cache.v1"
RUN_ID="oakbat-cleanStatic-method-a-9tap"
SEQ_LEN=12
CADENCE_S=0.5
TIME_TOLERANCE_S=1e-6
MAX_MODEL_DIM=4096
MAX_CHECKPOINT_TENSORS=100_000_000
ARTIFACT_ROSTER={"model.pt","training_history.csv","split_manifest.json","model_metadata.json","calibration_prn_scores.csv","calibration.json","held_clean_prn_scores.csv","held_clean_event_scores.csv","held_clean_fpr.json","partitions/train.csv","partitions/validation.csv","partitions/calibration.csv","partitions/held_clean.csv"}
DEFAULTS={"seq_len":12,"epochs":25,"batch_size":256,"lr":.001,"weight_decay":.0001,"hidden_dim":128,"emb_dim":128,"dropout":.05,"seed":11}

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
 return h.hexdigest()

def _json(path, description):
 try:
  value=json.loads(Path(path).read_text(encoding="utf-8"))
 except Exception as exc: raise ValueError(f"missing/invalid {description}") from exc
 if not isinstance(value,dict): raise ValueError(f"invalid {description}")
 return value

def _identity(document,key,description):
 value=document.get(key)
 if not isinstance(value,dict) or not isinstance(value.get("path"),str) or not isinstance(value.get("sha256"),str): raise ValueError(f"invalid {description} identity")
 return value

def _exact_file_identity(identity, expected_path, description):
 expected=Path(expected_path).resolve(); claimed=Path(identity["path"]).resolve()
 if claimed!=expected or not expected.is_file() or identity["sha256"]!=sha256(expected): raise ValueError(f"{description} path/hash mismatch")

def _coverage_values(receiver):
 tracking=receiver.get("tracking",{}); coverage=tracking.get("coverage",{}) if isinstance(tracking,dict) else {}
 if isinstance(coverage,(int,float)): return [float(coverage)]
 keys=("coverage_seconds","coverage_s","duration_seconds","duration_s","coverage_end_s","tracking_csv_max_time_s","tracking_summary_max_time_s","end_time_s","max_time_s")
 values=[]
 for doc in (tracking,coverage):
  if isinstance(doc,dict):
   for key in keys:
    if key in doc:
     try: values.append(float(doc[key]))
     except (TypeError,ValueError): raise ValueError("invalid tracking coverage")
 return values

def authenticate_clean_input(node_csv):
 source=Path(node_csv).resolve()
 if not source.is_file(): raise FileNotFoundError(source)
 node_path=source.parent/"manifest.json"; feature_path=source.parents[1]/"oakbat_feature_cache_manifest.json"
 node=_json(node_path,"node dataset manifest")
 if node.get("schema")!=NODE_SCHEMA or node.get("tap_count")!=9 or node.get("tap_layout")!=TAP_LAYOUT: raise ValueError("node manifest schema/tap contract mismatch")
 _exact_file_identity(_identity(node,"node_table","node table"),source,"node manifest node_table")
 feature=_json(feature_path,"OAKBAT feature cache manifest")
 if feature.get("schema")!=FEATURE_CACHE_SCHEMA: raise ValueError("feature cache manifest schema mismatch")
 feature_contract=feature.get("feature_contract",{})
 if (not isinstance(feature_contract,dict) or feature_contract.get("feature_mode")!="normalized_dmcpd" or feature_contract.get("tap_count")!=9 or feature_contract.get("node_feature_columns")!=FEATURE_COLUMNS): raise ValueError("feature cache feature contract mismatch")
 _exact_file_identity(_identity(feature,"node_table","feature cache node table"),source,"feature cache node_table")
 receiver_identity=_identity(feature,"receiver_manifest","receiver manifest")
 receiver_path=Path(receiver_identity["path"]).resolve(); _exact_file_identity(receiver_identity,receiver_path,"receiver manifest")
 receiver=_json(receiver_path,"receiver manifest")
 if receiver.get("schema_version") not in (3,4): raise ValueError("receiver manifest schema mismatch")
 receiver_source=receiver.get("source",{})
 iq_path=receiver_source.get("iq",receiver_source.get("path",""))
 status=receiver.get("status",receiver.get("state"))
 if receiver_source.get("dataset")!="OAKBAT" or receiver_source.get("scenario_id")!="cleanStatic": raise ValueError("receiver source is not OAKBAT cleanStatic")
 if receiver.get("receiver_run_id")!=RUN_ID: raise ValueError("receiver run_id mismatch")
 tracking=receiver.get("tracking",{})
 if not isinstance(tracking,dict) or tracking.get("tap_count")!=9: raise ValueError("receiver nine-tap contract mismatch")
 iq_hash=receiver_source.get("iq_sha256")
 if not isinstance(iq_hash,str) or len(iq_hash)!=64 or any(c not in "0123456789abcdef" for c in iq_hash.lower()): raise ValueError("receiver IQ hash identity missing/invalid")
 if Path(str(iq_path)).name!="cleanStatic_gps.bin": raise ValueError("receiver source IQ basename mismatch")
 if status!="complete": raise ValueError("receiver status is not complete")
 coverage=_coverage_values(receiver)
 if not coverage or any(not math.isfinite(v) or v<478. or v>481. for v in coverage): raise ValueError("receiver tracking coverage outside [478, 481]")
 frame=pd.read_csv(source)
 if "run_id" not in frame or not (frame.run_id.astype(str)==RUN_ID).all(): raise ValueError("CSV run_id must be authenticated clean run")
 if "label" not in frame or not (frame.label.astype(str)=="oakbat_cleanStatic_9tap").all(): raise ValueError("CSV label must be exactly oakbat_cleanStatic_9tap")
 if "source_fingerprint" not in frame or frame.source_fingerprint.isna().any() or frame.source_fingerprint.astype(str).nunique()!=1 or not str(frame.source_fingerprint.iloc[0]): raise ValueError("CSV source_fingerprint must be one consistent non-empty value")
 manifests={"node_dataset":{"path":str(node_path.resolve()),"sha256":sha256(node_path)},"feature_cache":{"path":str(feature_path.resolve()),"sha256":sha256(feature_path)},"receiver":{"path":str(receiver_path),"sha256":sha256(receiver_path)}}
 return source,frame,manifests

def atomic_json(path,value):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix="."+path.name,dir=path.parent)
 try:
  with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(value,f,indent=2,sort_keys=True,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp): os.unlink(tmp)

def atomic_csv(path,frame):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix="."+path.name,dir=path.parent); os.close(fd)
 try: frame.to_csv(tmp,index=False); os.replace(tmp,path)
 finally:
  if os.path.exists(tmp): os.unlink(tmp)

def validate_clean_frame(df):
 required={"run_id","prn","window_bin_s","window_start_s","window_mid_s","window_end_s","tap_count","tap_layout",*FEATURE_COLUMNS}
 missing=sorted(required-set(df.columns))
 if missing: raise ValueError(f"missing required feature/data columns: {missing}")
 tap_features=[c for c in df.columns if c.startswith("tap_") and c.endswith("_rel_prompt_mean")]
 if tap_features!=FEATURE_COLUMNS: raise ValueError("feature order/layout must be exactly the nine E4,E3,E2,E,P,L,L2,L3,L4 rel_prompt_mean columns")
 if not (pd.to_numeric(df.tap_count,errors="coerce")==9).all(): raise ValueError("tap_count must be 9")
 if not (df.tap_layout.astype(str)==TAP_LAYOUT).all(): raise ValueError("tap_layout mismatch")
 if df[["run_id","prn"]].isna().any().any() or (df.run_id.astype(str).str.len()==0).any() or (df.prn.astype(str).str.len()==0).any(): raise ValueError("missing run_id/PRN")
 numeric=["window_bin_s","window_start_s","window_mid_s","window_end_s","tap_count",*FEATURE_COLUMNS]
 converted=df[numeric].apply(pd.to_numeric,errors="coerce")
 if not np.isfinite(converted.to_numpy(float)).all(): raise ValueError("non-finite required data")
 bins=converted["window_bin_s"].to_numpy(float); starts=converted["window_start_s"].to_numpy(float)
 if not np.allclose(bins/CADENCE_S,np.rint(bins/CADENCE_S),rtol=0.,atol=TIME_TOLERANCE_S/CADENCE_S): raise ValueError("window_bin_s is not on the expected 0.5-s grid")
 offsets=starts-bins
 if not np.allclose(offsets,offsets[0],rtol=0.,atol=TIME_TOLERANCE_S): raise ValueError("window_start_s/window_bin_s global offset disagreement")
 key=["run_id","window_bin_s","prn"]
 if df.duplicated(key,keep=False).any(): raise ValueError("duplicate/non-unique (run_id,window_bin_s,prn)")
 for _,group in df.assign(_bin=bins).groupby(["run_id","prn"],sort=False):
  if len(group)>1 and not (np.diff(group._bin.to_numpy(float))>TIME_TOLERANCE_S).all(): raise ValueError("window bins must be monotonic within each PRN")

def _sort(df):
 cols=[c for c in ("run_id","prn","window_bin_s","window_start_s") if c in df.columns]
 return df.sort_values(cols,kind="mergesort").reset_index(drop=True)

def validate_partitions(parts,seq_len):
 if list(parts)!=list(PARTITION_RULES): raise ValueError("partition set/order mismatch")
 seen=set()
 for name,part in parts.items():
  if part.empty: raise ValueError(f"empty partition: {name}")
  keys=set(map(tuple,part[["run_id","window_bin_s","prn"]].itertuples(index=False,name=None)))
  if seen & keys: raise ValueError("partition overlap")
  seen |= keys
  lo,hi=PARTITION_RULES[name]; t=part.window_start_s.astype(float)
  if (lo is not None and (t<lo).any()) or (hi is not None and (t>=hi).any()): raise ValueError(f"partition boundary overlap: {name}")
  if part.prn.astype(str).nunique()<2: raise ValueError(f"insufficient PRNs in {name}")
  counts=part.groupby(["run_id","prn"],dropna=False).size()
  if counts.empty or (counts<seq_len+1).any(): raise ValueError(f"insufficient PRN history in {name}; need seq_len+1 per series")
  for _,group in part.groupby(["run_id","prn"],sort=False):
   bins=np.sort(group.window_bin_s.to_numpy(float))
   if len(bins)>1 and not np.allclose(np.diff(bins),CADENCE_S,rtol=0.,atol=TIME_TOLERANCE_S): raise ValueError(f"temporal cadence/gap in {name}; sequences cannot bridge gaps")

def split_chronologically(df,seq_len=12):
 validate_clean_frame(df); t=df.window_start_s.astype(float); parts={}
 for name,(lo,hi) in PARTITION_RULES.items():
  mask=np.ones(len(df),dtype=bool)
  if lo is not None: mask &= t.to_numpy()>=lo
  if hi is not None: mask &= t.to_numpy()<hi
  parts[name]=_sort(df.loc[mask].copy())
 validate_partitions(parts,seq_len); return parts

def sequence_target_times(part,seq_len=12):
 result=[]
 for _,group in part.groupby(["run_id","prn"],sort=True):
  group=group.sort_values("window_bin_s",kind="mergesort"); result.extend(group.window_start_s.iloc[seq_len:].astype(float))
 return sorted(result)

def fit_train_standardizer(parts):
 values=parts["train"][FEATURE_COLUMNS].to_numpy(np.float32)
 if not np.isfinite(values).all(): raise ValueError("non-finite training features")
 return train_lib.fit_standardizer(values)

def _validate_hparams(hparams):
 for name in ("seq_len","epochs","batch_size","hidden_dim","emb_dim"):
  value=hparams.get(name)
  if isinstance(value,bool) or not isinstance(value,int) or value<1: raise ValueError(f"{name} must be a positive integer")
 if hparams["hidden_dim"]>MAX_MODEL_DIM or hparams["emb_dim"]>MAX_MODEL_DIM: raise ValueError("hidden_dim/emb_dim exceeds safety limit")
 for name in ("lr","weight_decay","dropout"):
  value=hparams.get(name)
  if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(float(value)): raise ValueError(f"{name} must be finite")
 if hparams["lr"]<=0: raise ValueError("lr must be positive")
 if hparams["weight_decay"]<0: raise ValueError("weight_decay must be nonnegative")
 if not 0<=hparams["dropout"]<1: raise ValueError("dropout must satisfy 0 <= dropout < 1")
 if isinstance(hparams.get("seed"),bool) or not isinstance(hparams.get("seed"),int): raise ValueError("seed must be an integer")

def _config(source,out,**hparams):
 _validate_hparams(hparams)
 return train_lib.TrainConfig(node_csv=str(source),output_dir=str(out),feature_subset="tap_rel_prompt_mean",**hparams)

def _datasets(parts,cfg,mean,std):
 result={}
 for name,part in parts.items(): result[name]=train_lib.PrnSequenceDataset(train_lib.build_series(part,FEATURE_COLUMNS,mean,std),cfg.seq_len)
 return result

def train_model(parts,cfg,mean,std,out):
 random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
 datasets=_datasets(parts,cfg,mean,std); gen=torch.Generator().manual_seed(cfg.seed)
 train_loader=DataLoader(datasets["train"],batch_size=cfg.batch_size,shuffle=True,generator=gen,num_workers=0)
 val_loader=DataLoader(datasets["validation"],batch_size=cfg.batch_size,shuffle=False,num_workers=0)
 model=train_lib.PrnLocalGRU(len(FEATURE_COLUMNS),cfg); opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
 best=math.inf; best_epoch=None; best_state=None; history=[]
 for epoch in range(1,cfg.epochs+1):
  tr=train_lib.run_epoch(model,train_loader,opt,torch.device("cpu")); va=train_lib.run_epoch(model,val_loader,None,torch.device("cpu"))
  history.append({"epoch":epoch,"train_loss":tr["loss"],"validation_loss":va["loss"],"train_sequences":tr["count"],"validation_sequences":va["count"]})
  if va["loss"]<best: best=va["loss"]; best_epoch=epoch; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 if best_state is None or not math.isfinite(best): raise ValueError("no finite validation checkpoint")
 checkpoint=Path(out)/"model.pt"
 atomic_torch_save(checkpoint,{"model_state_dict":best_state,"config":asdict(cfg),"node_feature_columns":FEATURE_COLUMNS,"standardizer":{"node_mean":mean.tolist(),"node_std":std.tolist()},"selected_epoch":best_epoch,"best_validation_loss":best,"checkpoint_selection":"minimum validation loss"})
 atomic_csv(Path(out)/"training_history.csv",pd.DataFrame(history)); return checkpoint,best_epoch,best

def atomic_torch_save(path,value):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix="."+path.name,dir=path.parent); os.close(fd)
 try:
  torch.save(value,tmp)
  with open(tmp,"rb") as f: os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp): os.unlink(tmp)


def _validated_checkpoint_payload(payload):
 expected={"model_state_dict","config","node_feature_columns","standardizer","selected_epoch","best_validation_loss","checkpoint_selection"}
 if not isinstance(payload,dict) or set(payload)!=expected: raise ValueError("checkpoint payload key/type mismatch")
 config=payload["config"]; config_keys=set(train_lib.TrainConfig.__dataclass_fields__)
 if not isinstance(config,dict) or set(config)!=config_keys: raise ValueError("checkpoint config key/type mismatch")
 if not isinstance(config.get("node_csv"),str) or not isinstance(config.get("output_dir"),str) or config.get("feature_subset")!="tap_rel_prompt_mean": raise ValueError("checkpoint config contract mismatch")
 _validate_hparams({k:config[k] for k in DEFAULTS})
 if config["seq_len"]!=SEQ_LEN: raise ValueError("checkpoint seq_len contract mismatch")
 if payload["node_feature_columns"]!=FEATURE_COLUMNS or payload["checkpoint_selection"]!="minimum validation loss": raise ValueError("checkpoint feature/selection contract mismatch")
 epoch=payload["selected_epoch"]; loss=payload["best_validation_loss"]
 if isinstance(epoch,bool) or not isinstance(epoch,int) or not 1<=epoch<=config["epochs"]: raise ValueError("checkpoint selected_epoch contract mismatch")
 if isinstance(loss,bool) or not isinstance(loss,(int,float)) or not math.isfinite(float(loss)): raise ValueError("checkpoint validation loss contract mismatch")
 standardizer=payload["standardizer"]
 if not isinstance(standardizer,dict) or set(standardizer)!={"node_mean","node_std"}: raise ValueError("checkpoint standardizer key/type mismatch")
 for name in ("node_mean","node_std"):
  values=standardizer[name]
  if not isinstance(values,(list,tuple)) or len(values)!=len(FEATURE_COLUMNS) or any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in values): raise ValueError("checkpoint standardizer nested type mismatch")
 state=payload["model_state_dict"]
 if not isinstance(state,dict) or not state or any(not isinstance(k,str) or not isinstance(v,torch.Tensor) for k,v in state.items()): raise ValueError("checkpoint model state must contain only named tensors")
 if sum(v.numel() for v in state.values())>MAX_CHECKPOINT_TENSORS: raise ValueError("checkpoint model state exceeds tensor safety limit")
 for tensor in state.values():
  if tensor.layout!=torch.strided or tensor.device.type!="cpu" or tensor.dtype not in (torch.float16,torch.float32,torch.float64) or not torch.isfinite(tensor).all().item(): raise ValueError("checkpoint model state tensor contract mismatch")
 return config


def _open_model(checkpoint):
 try: payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
 except Exception as exc: raise ValueError("invalid or unsupported safe checkpoint payload") from exc
 config=_validated_checkpoint_payload(payload)
 try: cfg=train_lib.TrainConfig(**config)
 except (TypeError,ValueError) as exc: raise ValueError("checkpoint config construction mismatch") from exc
 mean=np.asarray(payload["standardizer"]["node_mean"],np.float32); std=np.asarray(payload["standardizer"]["node_std"],np.float32)
 if mean.shape!=(9,) or std.shape!=(9,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any(): raise ValueError("checkpoint standardizer contract mismatch")
 model=train_lib.PrnLocalGRU(len(FEATURE_COLUMNS),cfg)
 try: model.load_state_dict(payload["model_state_dict"],strict=True)
 except (RuntimeError,TypeError,ValueError) as exc: raise ValueError("checkpoint model state shape/key mismatch") from exc
 model.eval(); return payload,cfg,model,mean,std

def score_partition(part,checkpoint):
 _,cfg,model,mean,std=_open_model(checkpoint); rows=[]
 with torch.no_grad():
  for key,group in part.groupby(["run_id","prn"],sort=True):
   group=group.sort_values("window_bin_s",kind="mergesort")
   x=(group[FEATURE_COLUMNS].to_numpy(np.float32)-mean)/std
   if not np.isfinite(x).all(): raise ValueError("non-finite score input")
   for start in range(len(x)-cfg.seq_len):
    target=start+cfg.seq_len; pred=model(torch.from_numpy(x[start:target][None])).numpy()[0]; source=group.iloc[target]
    rows.append({"run_id":str(source.run_id),"prn":str(source.prn),"window_bin_s":float(source.window_bin_s),"window_start_s":float(source.window_start_s),"window_mid_s":float(source.window_mid_s),"window_end_s":float(source.window_end_s),"prn_node_rmse":float(np.sqrt(np.mean((pred-x[target])**2)))})
 scores=pd.DataFrame(rows)
 if scores.empty or not np.isfinite(scores.prn_node_rmse).all(): raise ValueError("empty/non-finite partition scores")
 return scores

def derive_calibration(calibration_scores):
 calibration_scores=calibration_scores.copy()
 if "window_start_s" not in calibration_scores: calibration_scores["window_start_s"]=calibration_scores["window_mid_s"]
 if "window_bin_s" not in calibration_scores: calibration_scores["window_bin_s"]=calibration_scores["window_start_s"]
 values=calibration_scores.prn_node_rmse.astype(float)
 if values.empty or not np.isfinite(values).all(): raise ValueError("empty/non-finite calibration scores")
 thresholds={name:float(values.quantile(q)) for name,q in gate_lib.NODE_QUANTILES.items()}
 events=gate_lib.build_event_scores(calibration_scores,thresholds,alpha=.75); event_q99=float(events[gate_lib.FINAL_SCORE].quantile(.99))
 if not math.isfinite(event_q99): raise ValueError("non-finite event threshold")
 return {"schema":"gnss-doppler-lab.oakbat-cleanstatic-calibration.v1","normal_only":True,"attack_inputs_read":False,"input_partition":"calibration","threshold_source_partition":"calibration","node_thresholds":thresholds,"event_q99_threshold":event_q99,"event_score":gate_lib.FINAL_SCORE,"event_builder":"build_event_scores","alpha":.75}

def held_clean_report(held_scores,calibration):
 held_scores=held_scores.copy()
 if "window_start_s" not in held_scores: held_scores["window_start_s"]=held_scores["window_mid_s"]
 if "window_bin_s" not in held_scores: held_scores["window_bin_s"]=held_scores["window_start_s"]
 events=gate_lib.build_event_scores(held_scores,calibration["node_thresholds"],alpha=.75); flags=events[gate_lib.FINAL_SCORE]>float(calibration["event_q99_threshold"])
 report={"schema":"gnss-doppler-lab.oakbat-held-clean-fpr.v1","threshold_source_partition":"calibration","partition":"held_clean","event_windows":int(len(events)),"false_positive_events":int(flags.sum()),"false_positive_rate":float(flags.mean())}
 return events.assign(gate_flag=flags),report

def _run_campaign(clean_node_csv,output_dir,**overrides):
 hparams=dict(DEFAULTS); unknown=set(overrides)-set(hparams)
 if unknown: raise TypeError(f"unknown training knobs: {sorted(unknown)}")
 hparams.update(overrides)
 if hparams["seq_len"]!=SEQ_LEN: raise ValueError("seq_len is frozen at 12")
 _validate_hparams(hparams)
 source,frame,parent_manifests=authenticate_clean_input(clean_node_csv); out=Path(output_dir).resolve()
 if out.exists() and any(out.iterdir()): raise FileExistsError("output directory must be absent or empty")
 out.mkdir(parents=True,exist_ok=True); source_hash=sha256(source); parts=split_chronologically(frame,SEQ_LEN)
 partition_docs={}
 for name,part in parts.items():
  path=out/"partitions"/(name+".csv"); atomic_csv(path,part); partition_docs[name]={"path":str(path.relative_to(out)),"sha256":sha256(path),"rows":int(len(part)),"prns":int(part.prn.nunique())}
 split_doc={"schema":"gnss-doppler-lab.oakbat-chronological-split.v1","clock":"window_start_s","boundaries":{n:{"start_inclusive":lo,"end_exclusive":hi} for n,(lo,hi) in PARTITION_RULES.items()},"purge_intervals":[{"start_inclusive":a,"end_exclusive":b} for a,b in PURGES],"seq_len":hparams["seq_len"],"history_contract":"each partition forms sequences independently; no history crosses boundaries","source_sha256":source_hash,"partition_csvs":partition_docs}
 atomic_json(out/"split_manifest.json",split_doc)
 mean,std=fit_train_standardizer(parts); cfg=_config(source,out,**hparams); checkpoint,best_epoch,best_loss=train_model(parts,cfg,mean,std,out)
 if sha256(source)!=source_hash: raise ValueError("source mutation detected during training")
 checkpoint_hash=sha256(checkpoint)
 metadata={"schema":"gnss-doppler-lab.oakbat-model-metadata.v1","architecture":"PrnLocalGRU","feature_columns":FEATURE_COLUMNS,"feature_count":9,"hparams":hparams,"standardizer_fit_partition":"train","checkpoint_selection":"minimum validation loss","selected_epoch":best_epoch,"best_validation_loss":best_loss,"checkpoint":"model.pt","checkpoint_sha256":checkpoint_hash}
 atomic_json(out/"model_metadata.json",metadata)
 calibration_scores=score_partition(parts["calibration"],checkpoint); atomic_csv(out/"calibration_prn_scores.csv",calibration_scores)
 calibration=derive_calibration(calibration_scores); calibration.update({"input_csv":"calibration_prn_scores.csv","input_sha256":sha256(out/"calibration_prn_scores.csv"),"checkpoint_sha256":checkpoint_hash}); atomic_json(out/"calibration.json",calibration)
 held_scores=score_partition(parts["held_clean"],checkpoint); atomic_csv(out/"held_clean_prn_scores.csv",held_scores); held_events,fpr=held_clean_report(held_scores,calibration); atomic_csv(out/"held_clean_event_scores.csv",held_events); fpr.update({"score_input_sha256":sha256(out/"held_clean_prn_scores.csv"),"calibration_sha256":sha256(out/"calibration.json")}); atomic_json(out/"held_clean_fpr.json",fpr)
 if sha256(source)!=source_hash: raise ValueError("source mutation detected during campaign")
 for identity in parent_manifests.values():
  if not Path(identity["path"]).is_file() or sha256(identity["path"])!=identity["sha256"]: raise ValueError("parent manifest mutation detected during campaign")
 names=["model.pt","training_history.csv","split_manifest.json","model_metadata.json","calibration_prn_scores.csv","calibration.json","held_clean_prn_scores.csv","held_clean_event_scores.csv","held_clean_fpr.json",*[d["path"] for d in partition_docs.values()]]
 manifest={"schema":SCHEMA,"complete":True,"normal_only":True,"attack_inputs_read":False,"source":{"path":str(source),"sha256":source_hash,"parent_manifests":parent_manifests},"checkpoint":"model.pt","calibration":"calibration.json","split":"split_manifest.json","artifacts":{name:sha256(out/name) for name in names}}
 atomic_json(out/"campaign_manifest.json",manifest); return manifest

def run_campaign(clean_node_csv,output_dir,**overrides):
 hparams=dict(DEFAULTS); unknown=set(overrides)-set(hparams)
 if unknown: raise TypeError(f"unknown training knobs: {sorted(unknown)}")
 hparams.update(overrides)
 if hparams["seq_len"]!=SEQ_LEN: raise ValueError("seq_len is frozen at 12")
 _validate_hparams(hparams)
 out=Path(output_dir).resolve(); existed=out.exists()
 try: return _run_campaign(clean_node_csv,out,**overrides)
 except Exception:
  if not existed and out.exists(): shutil.rmtree(out)
  raise


def _under_root(root, relative, description):
 if not isinstance(relative,str) or not relative or Path(relative).is_absolute(): raise ValueError(f"invalid {description} pointer")
 path=(root/relative).resolve()
 if root!=path and root not in path.parents: raise ValueError(f"{description} path traversal")
 return path


def _semantic_equal(actual,expected,description):
 """Compare recomputed semantics with deterministic numeric tolerance."""
 if isinstance(expected,dict):
  if not isinstance(actual,dict) or set(actual)!=set(expected): raise ValueError(f"{description} semantic key mismatch")
  for key,value in expected.items(): _semantic_equal(actual[key],value,f"{description}.{key}")
 elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
  if isinstance(actual,bool) or not isinstance(actual,(int,float)) or not math.isfinite(float(actual)) or not math.isclose(float(actual),float(expected),rel_tol=1e-12,abs_tol=1e-12): raise ValueError(f"{description} semantic numeric mismatch")
 elif actual!=expected: raise ValueError(f"{description} semantic mismatch")

def _semantic_frame_equal(actual,expected,description):
 try: pd.testing.assert_frame_equal(actual.reset_index(drop=True),expected.reset_index(drop=True),check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
 except AssertionError as exc: raise ValueError(f"{description} semantic content mismatch") from exc

def load_frozen_artifacts(output_dir):
 root=Path(output_dir).resolve(); manifest=_json(root/"campaign_manifest.json","frozen campaign manifest")
 if manifest.get("schema")!=SCHEMA or manifest.get("complete") is not True or manifest.get("normal_only") is not True or manifest.get("attack_inputs_read") is not False: raise ValueError("invalid frozen campaign contract")
 if manifest.get("checkpoint")!="model.pt": raise ValueError("checkpoint pointer mismatch")
 if manifest.get("calibration")!="calibration.json": raise ValueError("calibration pointer mismatch")
 if manifest.get("split")!="split_manifest.json": raise ValueError("split pointer mismatch")
 source_doc=manifest.get("source",{}); source_path=source_doc.get("path")
 if not isinstance(source_path,str) or str(Path(source_path).resolve())!=source_path: raise ValueError("invalid authenticated source path")
 source=Path(source_path)
 if not source.is_file() or sha256(source)!=source_doc.get("sha256"): raise ValueError("source tamper/hash mismatch")
 parents=source_doc.get("parent_manifests")
 if not isinstance(parents,dict) or set(parents)!={"node_dataset","feature_cache","receiver"}: raise ValueError("invalid parent manifest inventory")
 for name,identity in parents.items():
  if not isinstance(identity,dict) or not isinstance(identity.get("path"),str) or str(Path(identity["path"]).resolve())!=identity["path"]: raise ValueError(f"invalid parent manifest path: {name}")
  path=Path(identity["path"])
  if not path.is_file() or sha256(path)!=identity.get("sha256"): raise ValueError(f"parent manifest tamper/hash mismatch: {name}")
 # Re-authentication detects parent substitution even if a caller rewrites hashes.
 auth_source,authenticated_frame,authenticated=authenticate_clean_input(source)
 if auth_source!=source or authenticated!=parents: raise ValueError("authenticated source chain substitution")
 artifacts=manifest.get("artifacts")
 if not isinstance(artifacts,dict) or set(artifacts)!=ARTIFACT_ROSTER: raise ValueError("frozen artifact inventory mismatch")
 for rel,expected in artifacts.items():
  path=_under_root(root,rel,"artifact")
  if not path.is_file() or not isinstance(expected,str) or sha256(path)!=expected: raise ValueError(f"artifact tamper/hash mismatch: {rel}")
 calibration=_json(_under_root(root,manifest["calibration"],"calibration"),"calibration")
 metadata=_json(root/"model_metadata.json","model metadata"); checkpoint=_under_root(root,manifest["checkpoint"],"checkpoint")
 checkpoint_hash=sha256(checkpoint)
 if calibration.get("checkpoint_sha256")!=checkpoint_hash or metadata.get("checkpoint")!="model.pt" or metadata.get("checkpoint_sha256")!=checkpoint_hash: raise ValueError("checkpoint linkage hash mismatch")
 if calibration.get("input_csv")!="calibration_prn_scores.csv" or calibration.get("input_sha256")!=sha256(root/"calibration_prn_scores.csv"): raise ValueError("calibration input linkage mismatch")
 split=_json(_under_root(root,manifest["split"],"split"),"split manifest")
 expected_boundaries={n:{"start_inclusive":lo,"end_exclusive":hi} for n,(lo,hi) in PARTITION_RULES.items()}
 expected_purges=[{"start_inclusive":a,"end_exclusive":b} for a,b in PURGES]
 if (split.get("schema")!="gnss-doppler-lab.oakbat-chronological-split.v1" or split.get("clock")!="window_start_s" or
     split.get("source_sha256")!=source_doc.get("sha256") or split.get("seq_len")!=SEQ_LEN or
     split.get("boundaries")!=expected_boundaries or split.get("purge_intervals")!=expected_purges or
     split.get("history_contract")!="each partition forms sequences independently; no history crosses boundaries"):
  raise ValueError("split source/sequence/boundary linkage mismatch")
 expected_parts=split_chronologically(authenticated_frame,SEQ_LEN)
 partition_csvs=split.get("partition_csvs")
 if not isinstance(partition_csvs,dict) or set(partition_csvs)!=set(PARTITION_RULES): raise ValueError("split partition inventory mismatch")
 for name,doc in partition_csvs.items():
  expected_rel=f"partitions/{name}.csv"
  if not isinstance(doc,dict) or doc.get("path")!=expected_rel or doc.get("sha256")!=artifacts.get(expected_rel): raise ValueError(f"split partition pointer/hash mismatch: {name}")
  frame=pd.read_csv(_under_root(root,expected_rel,"split partition"))
  if doc.get("rows")!=len(frame) or doc.get("prns")!=frame.prn.astype(str).nunique(): raise ValueError(f"split partition metadata mismatch: {name}")
  _semantic_frame_equal(frame,expected_parts[name],f"split partition {name}")
 payload,cfg,model,mean,std=_open_model(checkpoint)
 if (calibration.get("schema")!="gnss-doppler-lab.oakbat-cleanstatic-calibration.v1" or calibration.get("normal_only") is not True or
     calibration.get("attack_inputs_read") is not False or calibration.get("input_partition")!="calibration" or
     calibration.get("threshold_source_partition")!="calibration" or set(calibration.get("node_thresholds",{}))!={"q50","q70","q80"}):
  raise ValueError("calibration contract/linkage mismatch")
 calibration_scores=pd.read_csv(root/"calibration_prn_scores.csv")
 rescored_calibration=score_partition(expected_parts["calibration"],checkpoint)
 _semantic_frame_equal(calibration_scores,rescored_calibration,"calibration checkpoint scores")
 expected_calibration=derive_calibration(rescored_calibration)
 expected_calibration.update({"input_csv":"calibration_prn_scores.csv","input_sha256":sha256(root/"calibration_prn_scores.csv"),"checkpoint_sha256":checkpoint_hash})
 _semantic_equal(calibration,expected_calibration,"calibration")
 held=_json(root/"held_clean_fpr.json","held-clean report")
 if held.get("threshold_source_partition")!="calibration" or held.get("partition")!="held_clean" or held.get("calibration_sha256")!=artifacts.get("calibration.json") or held.get("score_input_sha256")!=artifacts.get("held_clean_prn_scores.csv"):
  raise ValueError("held-clean linkage mismatch")
 held_scores=pd.read_csv(root/"held_clean_prn_scores.csv")
 rescored_held=score_partition(expected_parts["held_clean"],checkpoint)
 _semantic_frame_equal(held_scores,rescored_held,"held-clean checkpoint scores")
 expected_events,expected_held=held_clean_report(rescored_held,expected_calibration)
 frozen_events=pd.read_csv(root/"held_clean_event_scores.csv")
 _semantic_frame_equal(frozen_events,expected_events,"held-clean event scores")
 expected_held.update({"score_input_sha256":sha256(root/"held_clean_prn_scores.csv"),"calibration_sha256":sha256(root/"calibration.json")})
 _semantic_equal(held,expected_held,"held-clean report")
 checkpoint_config=payload.get("config",{})
 if (cfg.seq_len!=SEQ_LEN or metadata.get("hparams")!= {k:checkpoint_config.get(k) for k in DEFAULTS} or
     metadata.get("hparams",{}).get("seq_len")!=SEQ_LEN or metadata.get("checkpoint_selection")!="minimum validation loss" or
     payload.get("checkpoint_selection")!="minimum validation loss" or metadata.get("selected_epoch")!=payload.get("selected_epoch") or
     metadata.get("best_validation_loss")!=payload.get("best_validation_loss")):
  raise ValueError("checkpoint metadata linkage mismatch")
 return {"manifest":manifest,"calibration":calibration,"metadata":metadata,"checkpoint":payload,"config":cfg,"model":model,"mean":mean,"std":std}

def build_parser():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--clean-node-csv",required=True); p.add_argument("--output-dir",required=True)
 p.add_argument("--epochs",type=int,default=25); p.add_argument("--batch-size",type=int,default=256); p.add_argument("--lr",type=float,default=.001); p.add_argument("--weight-decay",type=float,default=.0001); p.add_argument("--hidden-dim",type=int,default=128); p.add_argument("--emb-dim",type=int,default=128); p.add_argument("--dropout",type=float,default=.05); p.add_argument("--seed",type=int,default=11); return p
def main():
 args=vars(build_parser().parse_args()); source=args.pop("clean_node_csv"); out=args.pop("output_dir"); print(json.dumps(run_campaign(source,out,**args),indent=2,sort_keys=True))
if __name__=="__main__": main()
