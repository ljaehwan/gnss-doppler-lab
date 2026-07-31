#!/usr/bin/env python3
"""Parallel causal Tap-Residual Common-Drive V0 scorer for frozen B0."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, re, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path: sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.tap_residual_common_drive import (TIMING_CONTRACT, build_clean_calibration_document, calibrate_clean_only, causal_smooth_events, extract_b0_innovations, score_common_drive)
FROZEN_TAP_FEATURES = [f"tap_{x}_rel_prompt_mean" for x in ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")]
TAP_LAYOUT = "E4,E3,E2,E,P,L,L2,L3,L4"
CLEAN_LABEL = "texbat_cleanStatic_9tap_w1.0_s0.5"
CLEAN_RUN_ID = "texbat-cleanStatic-method-a-9tap-external-validation"

def _train_module():
 path=ROOT/"scripts"/"train_prn_node_gru.py"; spec=importlib.util.spec_from_file_location("trcd_frozen_b0_definition",path); module=importlib.util.module_from_spec(spec); assert spec.loader is not None; sys.modules[spec.name]=module; spec.loader.exec_module(module); return module

def _sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def load_frozen_b0(checkpoint_path: Path, device: str | torch.device, *, expected_sha256: str):
 actual=_sha256(checkpoint_path)
 if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256 or "") or actual.lower()!=expected_sha256.lower(): raise ValueError(f"frozen B0 checkpoint SHA-256 mismatch: got {actual}")
 checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
 required={"model_state_dict","config","node_feature_columns","standardizer"}
 if not isinstance(checkpoint,dict) or not required.issubset(checkpoint): raise ValueError("frozen B0 checkpoint lacks required structure")
 features=list(checkpoint["node_feature_columns"]); standardizer=checkpoint["standardizer"]
 try: mean=np.asarray(standardizer["node_mean"],dtype=np.float32); std=np.asarray(standardizer["node_std"],dtype=np.float32)
 except (KeyError,TypeError,ValueError) as exc: raise ValueError("invalid frozen B0 standardizer") from exc
 if features != FROZEN_TAP_FEATURES: raise ValueError("Tap-Residual V0 requires exact frozen nine-tap feature names/order")
 if mean.shape!=(9,) or std.shape!=(9,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0): raise ValueError("invalid frozen B0 nine-tap standardizer")
 train=_train_module()
 try: config=train.TrainConfig(**checkpoint["config"])
 except (TypeError,ValueError) as exc: raise ValueError("invalid frozen B0 semantic config") from exc
 if config.seq_len!=12: raise ValueError("Tap-Residual V0 requires frozen B0 seq_len=12")
 if getattr(config,"feature_subset",None)!="tap_rel_prompt_mean": raise ValueError("Tap-Residual V0 requires feature_subset=tap_rel_prompt_mean")
 model=train.PrnLocalGRU(9,config).to(device); model.load_state_dict(checkpoint["model_state_dict"],strict=True); model.eval()
 for parameter in model.parameters(): parameter.requires_grad_(False)
 return model,features,mean,std,config

def validate_node_csv_metadata(frame: pd.DataFrame, *, require_cleanstatic: bool) -> dict:
 required=("label","run_id","source_fingerprint","tap_count","tap_layout")
 missing=[c for c in required if c not in frame]
 if missing: raise ValueError(f"node CSV metadata missing required columns: {missing}")
 result={}
 for c in required:
  if frame[c].isna().any() or (frame[c].astype(str).str.strip()=="").any(): raise ValueError(f"node CSV metadata {c} must be non-null")
  values=frame[c].drop_duplicates().tolist()
  if len(values)!=1: raise ValueError(f"node CSV metadata {c} must be single-valued")
  result[c]=values[0]
 try: count=float(result["tap_count"])
 except (TypeError,ValueError) as exc: raise ValueError("node CSV metadata tap_count must equal 9") from exc
 if count!=9: raise ValueError("node CSV metadata tap_count must equal 9")
 if str(result["tap_layout"])!=TAP_LAYOUT: raise ValueError("node CSV metadata tap_layout is not the exact frozen layout")
 if require_cleanstatic and (str(result["label"])!=CLEAN_LABEL or str(result["run_id"])!=CLEAN_RUN_ID): raise ValueError("node CSV metadata does not identify the cleanStatic identity")
 return {k:(int(count) if k=="tap_count" else str(v)) for k,v in result.items()}

def score_node_csv(node_csv: Path, checkpoint_path: Path, *, expected_sha256: str, device: str | torch.device="cpu", alpha: float=.35, max_window_start_s: float|None=None, require_cleanstatic: bool=False):
 model,features,mean,std,config=load_frozen_b0(checkpoint_path,device,expected_sha256=expected_sha256); frame=pd.read_csv(node_csv); metadata=validate_node_csv_metadata(frame,require_cleanstatic=require_cleanstatic)
 if max_window_start_s is not None:
  if "window_start_s" not in frame: raise ValueError("node CSV missing window_start_s")
  frame=frame[pd.to_numeric(frame.window_start_s,errors="raise")<=max_window_start_s].copy()
 vectors=extract_b0_innovations(frame,model,features,mean,std,seq_len=config.seq_len,device=device); nodes,events=score_common_drive(vectors); return nodes,causal_smooth_events(events,alpha=alpha),metadata

def _validated_scenario_slug(value: str) -> str:
 if value=="cleanStatic" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*",value or "") or value in {".",".."}: raise ValueError("scenario must be a safe non-reserved slug")
 return value

def _output_paths(out: Path, scenario: str, include_score: bool):
 paths=[out/"cleanStatic_node_scores.csv",out/"cleanStatic_event_scores.csv",out/"calibration.json",out/"summary.json"]
 if include_score: paths += [out/f"{scenario}_node_scores.csv",out/f"{scenario}_event_scores.csv"]
 return paths

def _preflight_output_paths(out: Path, scenario: str, *, include_score: bool, overwrite: bool):
 scenario=_validated_scenario_slug(scenario) if include_score else scenario; base=out.resolve()
 paths=_output_paths(out,scenario,include_score)
 if any(p.resolve().parent!=base for p in paths): raise ValueError("all output paths must remain inside out-dir")
 existing=[p for p in paths if p.exists()]
 if existing and not overwrite: raise FileExistsError(f"refusing to overwrite outputs: {existing}")
 return paths

def _atomic_writer(path: Path, writer, *, overwrite: bool=False):
 path.parent.mkdir(parents=True,exist_ok=True); fd,temp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
 try:
  with os.fdopen(fd,"w",encoding="utf-8",newline="") as handle:
   writer(handle); handle.flush(); os.fsync(handle.fileno())
  if overwrite: os.replace(temp,path)
  else:
   os.link(temp,path); os.unlink(temp)
 except BaseException:
  try: os.unlink(temp)
  except FileNotFoundError: pass
  raise

def _write_scores(out: Path,name: str,nodes: pd.DataFrame,events: pd.DataFrame,*,overwrite: bool=False):
 if name=="cleanStatic": pass
 else: name=_validated_scenario_slug(name)
 node_path=out/f"{name}_node_scores.csv"; event_path=out/f"{name}_event_scores.csv"
 if not overwrite and (node_path.exists() or event_path.exists()): raise FileExistsError(f"refusing to overwrite score outputs for {name}")
 _atomic_writer(node_path,lambda handle:nodes.to_csv(handle,index=False),overwrite=overwrite); _atomic_writer(event_path,lambda handle:events.to_csv(handle,index=False),overwrite=overwrite); return node_path,event_path

def _atomic_json(path: Path,document: dict,*,overwrite: bool=False): _atomic_writer(path,lambda handle:handle.write(json.dumps(document,indent=2,sort_keys=True)+"\n"),overwrite=overwrite)

def _gate_metrics(events: pd.DataFrame,threshold: float,onset_s: float|None):
 flags=events.event_joint_evidence_causal>threshold; result={"windows":int(len(events)),"flags":int(flags.sum()),"flag_rate":float(flags.mean()) if len(flags) else 0.0}
 if onset_s is not None:
  pre=events.window_end_s<=onset_s; post=events.window_start_s>=onset_s; crossing=~(pre|post); post_flags=flags&post
  first_row=None
  if post_flags.any(): first_row=events.loc[post_flags].sort_values(["window_start_s","window_end_s","run_id"],kind="mergesort").iloc[0]
  score=None if first_row is None else float(first_row.window_start_s); available=None if first_row is None else float(first_row.window_end_s)
  result.update({"onset_s":onset_s,"pre_windows":int(pre.sum()),"pre_flags":int((flags&pre).sum()),"crossing_windows":int(crossing.sum()),"crossing_flags_uncertain":int((flags&crossing).sum()),"post_windows":int(post.sum()),"post_flags":int(post_flags.sum()),"post_flag_rate":float(post_flags.sum()/max(1,post.sum())),"first_post_flag_score_time_s":score,"first_post_flag_available_time_s":available,"score_time_delay_s":None if score is None else score-onset_s,"availability_time_delay_s":None if available is None else available-onset_s})
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--checkpoint",required=True); parser.add_argument("--expected-checkpoint-sha256",required=True); parser.add_argument("--cleanstatic-node-csv",required=True); parser.add_argument("--score-node-csv"); parser.add_argument("--scenario",default="diagnostic"); parser.add_argument("--out-dir",required=True); parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); parser.add_argument("--alpha",type=float,default=.35); parser.add_argument("--max-window-start-s",type=float); parser.add_argument("--onset-s",type=float); parser.add_argument("--overwrite",action="store_true"); args=parser.parse_args()
 checkpoint=Path(args.checkpoint).resolve(strict=True); clean_csv=Path(args.cleanstatic_node_csv).resolve(strict=True); score_csv=Path(args.score_node_csv).resolve(strict=True) if args.score_node_csv else None; scenario=_validated_scenario_slug(args.scenario) if score_csv else args.scenario; out=Path(args.out_dir); checkpoint_sha=_sha256(checkpoint)
 _preflight_output_paths(out,scenario,include_score=score_csv is not None,overwrite=args.overwrite)
 clean_nodes,clean_events,clean_meta=score_node_csv(clean_csv,checkpoint,expected_sha256=args.expected_checkpoint_sha256,device=args.device,alpha=args.alpha,max_window_start_s=args.max_window_start_s,require_cleanstatic=True)
 scored=None
 if score_csv: scored=(*score_node_csv(score_csv,checkpoint,expected_sha256=args.expected_checkpoint_sha256,device=args.device,alpha=args.alpha,max_window_start_s=args.max_window_start_s),score_csv)
 calibration=build_clean_calibration_document(clean_events,source_kind="cleanStatic",source_paths=[str(clean_csv)],source_fingerprint=clean_meta["source_fingerprint"],checkpoint_sha256=checkpoint_sha)
 threshold=calibration["thresholds"]["event_joint_evidence_causal"]["q0.99"]
 out.mkdir(parents=True,exist_ok=True); clean_node_path,clean_event_path=_write_scores(out,"cleanStatic",clean_nodes,clean_events,overwrite=args.overwrite); calibration_path=out/"calibration.json"; calibrate_clean_only(clean_events,calibration_path,source_kind="cleanStatic",source_paths=[str(clean_csv)],source_fingerprint=clean_meta["source_fingerprint"],checkpoint_sha256=checkpoint_sha,overwrite=args.overwrite)
 summary={"schema":"gnss-doppler-lab.tap-residual-common-drive-v0.run","architecture":"frozen B0 standardized nine-tap prediction innovation + same-event robust leave-one-out median common direction","checkpoint":str(checkpoint),"checkpoint_sha256":checkpoint_sha,"checkpoint_expected_sha256":args.expected_checkpoint_sha256.lower(),"b0_frozen":True,"b0_score_preserved":"sqrt(mean(standardized_9d_innovation**2))","timing_contract":TIMING_CONTRACT,"causal_alpha":args.alpha,"scaled_horizon_window_start_s":args.max_window_start_s,"calibration":str(calibration_path),"cleanStatic":{"source_metadata":clean_meta,"node_scores":str(clean_node_path),"event_scores":str(clean_event_path),"metrics_q99":_gate_metrics(clean_events,threshold,None)}}
 if scored:
  nodes,events,meta,_=scored; node_path,event_path=_write_scores(out,scenario,nodes,events,overwrite=args.overwrite); summary[scenario]={"source":str(score_csv),"source_metadata":meta,"role":"evaluation_only_never_fit","node_scores":str(node_path),"event_scores":str(event_path),"metrics_q99":_gate_metrics(events,threshold,args.onset_s)}
 _atomic_json(out/"summary.json",summary,overwrite=args.overwrite); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=="__main__": main()
