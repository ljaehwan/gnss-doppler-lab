#!/usr/bin/env python3
"""Train one cleanStatic-only GCMR, freeze it, then evaluate TEXBAT DS1--DS4."""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess
from pathlib import Path
import numpy as np
import torch
from gnss_doppler_lab.gcmr_experiment import (DEFAULT_ROLES,cache_events,calibration_threshold,implementation_manifest,load_checkpoint,load_event_cache,parse_preonset_nmea_position,preflight_oakbat_geometry,save_checkpoint,save_score_csv,score_events,select_role_events,source_hashes,train_clean_model,validate_roles)
from gnss_doppler_lab.gcmr_geometry import parse_gnss_sdr_gps_ephemeris_xml
from gnss_doppler_lab.gcmr_model import CleanReferenceScoreCalibrator
from gnss_doppler_lab.gcmr_relations import build_gcmr_pair_relation_events,load_gnss_sdr_tracking_rows

CLASSIFICATION="texbat_cleanstatic_only_frozen_external_scenario_evaluation"
CLEAN_ROOT=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/receiver/cleanStatic-complex9")
CLEAN_CACHE=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/gcmr-texbat-cleanstatic-event-cache-v1/cleanStatic.relations.npz")
DS_CACHE_ROOT=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/gcmr-texbat-event-cache-v1")
DEFAULT_OUTPUT=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/gcmr-texbat-cleanstatic-v1")
CLEAN_IQ_SHA256="dd295ab46616bfe9634d1c37479520e720ebc54bcb64adf0a247315a541fb9b9"
DS_CACHE_SHA256={"DS1":"8ddf6d7a6b70d4c7497ccb75ca0844a50fd6eeebd733ef030dbb11ce1bbbcfef","DS2":"c18201693e4240247b06647e5fae5eb7e51cf82a703018b838c03e678880a24d","DS3":"bc1ded77e5ce3d74262338980ea3668a4f2ba8eaf5c8bcbb25a4ea80821a35ed","DS4":"518be3f9154b52f58a777351f97f81d3a3e3998d6a3343f9cae6791d129c576f"}
CLEAN_EVENT_CONTRACT={"relation_contract_version":4,"window_s":1.0,"stride_s":0.5,"resample_bin_s":0.02,"min_common_samples":20,"min_prns":4,"healthy_ephemerides_only":True,"max_toe_age_s":7200.0,"sample_rate_hz":25e6,"tow0_s":477900.0,"window_interval":"[start,end)","score_available_at":"window_end_s","expected_event_count":959,"expected_range_s":[[0.,1.],[479.,480.]],"receiver_position_window_s":[20.,90.]}
TRAINING_CONFIG=dict(seed=23,max_epochs=40,patience=6,learning_rate=1e-3,compactness_weight=.01,warmup_epochs=5,pair_hidden=32,event_hidden=64,latent_dim=32)
ONSET_CONTRACT={"primary_nominal_onset_s":100.,"stable_pre":"start>=30,end<=90","transition":"start>=90,end<=110","stable_post":"start>=110","post120_sensitivity":"start>=120 secondary only","ds4_auxiliary_script_conflict":{"auxiliary_onset_s":110.,"primary_onset_s":100.,"resolution":"primary onset 100 retained; post>=120 is secondary sensitivity"}}

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()

def validate_receiver_manifest(path):
 try:m=json.loads(Path(path).read_text())
 except (OSError,json.JSONDecodeError) as e:raise ValueError(f"invalid receiver manifest: {e}") from e
 ok=(m.get("schema")=="gnss-doppler-lab.texbat-clean-complex9-receiver" and m.get("normal_only") is True and m.get("attack_inputs_read") is False and m.get("source",{}).get("iq_sha256")==CLEAN_IQ_SHA256 and m.get("source",{}).get("sample_rate_hz")==25000000)
 auth=m.get("authenticated_inputs")
 if not isinstance(auth,dict):ok=False
 else:
  before=auth.get("iq_before_receiver");after=auth.get("iq_after_receiver")
  if not isinstance(before,dict) or not isinstance(after,dict) or before.get("sha256")!=CLEAN_IQ_SHA256 or after.get("sha256")!=CLEAN_IQ_SHA256:ok=False
 if not ok:raise ValueError("receiver manifest / authenticated IQ contract mismatch")
 return m

def clean_sources(root=CLEAN_ROOT):
 root=Path(root);paths=[root/"manifest.json",root/"nmea_pvt.nmea",root/"gps_ephemeris.xml",root/"raw/observables.mat",*sorted((root/"raw").glob("epl_tracking_ch_*.mat"))]
 if any(not p.is_file() for p in paths):raise ValueError("cleanStatic source set is incomplete")
 return paths

def validate_clean_event_grid(events):
 events=list(events)
 if len(events)!=959:raise ValueError(f"cleanStatic cache must contain exactly 959 events, got {len(events)}")
 starts=np.asarray([e.window_start_s for e in events]);ends=np.asarray([e.window_end_s for e in events]);expected=np.arange(959)*.5
 if not np.array_equal(starts,expected) or not np.array_equal(ends,expected+1):raise ValueError("cleanStatic event grid/range does not match [0,1) through [479,480)")
 return events

def role_partitions(events):
 validate_roles();out={r.name:select_role_events(events,r) for r in DEFAULT_ROLES};expected={"train":299,"selection_val":139,"clean_reference":119,"event_calibration":119,"sealed_held":119}
 if {k:len(v) for k,v in out.items()}!=expected:raise ValueError("cleanStatic whole-window role count contract mismatch")
 ids=[id(e) for v in out.values() for e in v]
 if len(ids)!=len(set(ids)):raise ValueError("role contamination detected")
 return out

def clean_cache_metadata(root,preflight,position,manifest):
 return {"scenario":"cleanStatic","classification":"training_source_clean_only","event_contract":CLEAN_EVENT_CONTRACT,"geometry_preflight":preflight,"receiver_position_contract":{k:position[k] for k in ("llh","ecef","sample_count","timing","assumption")},"receiver_manifest":{"path":str((Path(root)/"manifest.json").resolve()),"sha256":sha256(Path(root)/"manifest.json"),"schema":manifest["schema"],"source_iq_sha256":CLEAN_IQ_SHA256},"implementation":implementation_manifest(),"runner":{"path":"scripts/run_gcmr_texbat_cleanstatic.py","sha256":sha256(__file__)}}

def load_clean_events(root=CLEAN_ROOT,cache_path=CLEAN_CACHE,force=False):
 root=Path(root);manifest=validate_receiver_manifest(root/"manifest.json");sources=clean_sources(root)
 eph=parse_gnss_sdr_gps_ephemeris_xml(root/"gps_ephemeris.xml");rows=load_gnss_sdr_tracking_rows(root/"raw",sample_rate_hz=25e6)
 preflight=preflight_oakbat_geometry(root/"raw/observables.mat",root/"nmea_pvt.nmea",eph,configured_tow0_s=477900.,max_toe_age_s=7200.,tow_tolerance_s=.05,onset_s=100.,tracked_prns={r.prn for r in rows},min_prns=4)
 position=parse_preonset_nmea_position(root/"nmea_pvt.nmea",gps_tow_at_time_zero_s=477900.,onset_s=100.,position_window_s=(20,90));meta=clean_cache_metadata(root,preflight,position,manifest);cache_path=Path(cache_path)
 if cache_path.exists() and not force:events=load_event_cache(cache_path,source_paths=sources,expected_metadata=meta)[0]
 else:
  events=build_gcmr_pair_relation_events(rows,ephemerides=eph,receiver_ecef=position["ecef"],gps_tow_at_time_zero_s=477900.,window_s=1.,stride_s=.5,resample_bin_s=.02,min_common_samples=20,min_prns=4)
  cache_events(cache_path,events,source_paths=sources,metadata=meta)
 # Mandatory immediate authenticated reload, including after reuse.
 events,saved=load_event_cache(cache_path,source_paths=sources,expected_metadata=meta);validate_clean_event_grid(events)
 return events,meta,sources,saved

class FreezeGate:
 def __init__(self):self.state=0
 def checkpoint_saved(self):
  if self.state!=0:raise RuntimeError("checkpoint may be saved exactly once")
  self.state=1
 def checkpoint_reloaded(self):
  if self.state!=1:raise RuntimeError("checkpoint must first be frozen")
  self.state=2
 def sealed_held_scored(self):
  if self.state!=2:raise RuntimeError("sealed held scoring requires reloaded frozen checkpoint")
  self.state=3
 def allow_external(self):
  if self.state!=3:raise RuntimeError("DS caches prohibited until checkpoint frozen, reloaded, and sealed held scored")

def load_pinned_ds_cache(name,gate,cache_root=DS_CACHE_ROOT):
 gate.allow_external();path=Path(cache_root)/f"{name.lower()}.relations.npz";before=sha256(path)
 if before!=DS_CACHE_SHA256[name]:raise ValueError(f"{name} cache SHA256 mismatch")
 try:
  with np.load(path,allow_pickle=False) as z:meta=json.loads(str(z["metadata_json"]))
 except Exception as e:raise ValueError(f"invalid {name} cache: {e}") from e
 source_map=meta.get("source_sha256");
 if not isinstance(source_map,dict) or not source_map:raise ValueError(f"{name} original source hashes missing")
 sources=[Path(p) for p in source_map]
 expected={k:v for k,v in meta.items() if k not in {"schema_version","relation_contract_version","observation_features","condition_features","source_sha256"}}
 events,validated=load_event_cache(path,source_paths=sources,expected_metadata=expected)
 if validated.get("source_sha256")!=source_map or sha256(path)!=before:raise ValueError(f"{name} cache/source identity changed during validation")
 return events,validated,before

def region_masks(starts,ends):
 s=np.asarray(starts,float);e=np.asarray(ends,float)
 return {"stable_pre":(s>=30)&(e<=90),"transition":(s>=90)&(e<=110),"stable_post":s>=110,"post120_sensitivity":s>=120}
def _metrics(scored,threshold,mask):
 x=np.asarray(scored["combined_score"])[mask];t=np.asarray(scored["availability_s"])[mask];a=x>threshold
 return {"event_count":int(len(x)),"alarm_count":int(a.sum()),"alarm_rate":float(a.mean()) if len(a) else None,"score_median":float(np.median(x)) if len(x) else None,"score_q99":float(np.quantile(x,.99)) if len(x) else None,"first_alarm_score_end_s":float(t[np.flatnonzero(a)[0]]) if a.any() else None}
def summarize_scores(name,scored,threshold):
 s=np.asarray(scored["window_start_s"]);e=np.asarray(scored["window_end_s"]);t=np.asarray(scored["availability_s"]);m=region_masks(s,e);out={k:_metrics(scored,threshold,v) for k,v in m.items()};alarm=(np.asarray(scored["combined_score"])>threshold)&(s>=100);first=float(t[np.flatnonzero(alarm)[0]]) if alarm.any() else None
 out.update({"event_count":int(len(s)),"window_range_s":[[float(s.min()),float(e[np.argmin(s)])],[float(s[np.argmax(e)]),float(e.max())]] if len(s) else None,"first_alarm_score_end_s":first,"first_alarm_delay_from_primary_onset_s":None if first is None else first-100.,"strict_alarm_rule":"score > threshold"})
 if name=="DS4":out["onset_conflict"]=ONSET_CONTRACT["ds4_auxiliary_script_conflict"]
 return out

def git(cmd):
 try:return subprocess.check_output(cmd,text=True,cwd=Path(__file__).resolve().parents[1]).strip()
 except Exception:return "unavailable"

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT);p.add_argument("--clean-cache",type=Path,default=CLEAN_CACHE);p.add_argument("--ds-cache-dir",type=Path,default=DS_CACHE_ROOT);p.add_argument("--force-clean-cache",action="store_true");p.add_argument("--max-epochs",type=int,default=40);p.add_argument("--device",default="cpu");a=p.parse_args(argv)
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 clean,cache_meta,clean_src,saved_meta=load_clean_events(cache_path=a.clean_cache,force=a.force_clean_cache);validate_roles();roles={r.name:select_role_events(clean,r) for r in DEFAULT_ROLES[:-1]}
 if {k:len(v) for k,v in roles.items()}!={"train":299,"selection_val":139,"clean_reference":119,"event_calibration":119}:raise RuntimeError("fit-role count contract mismatch")
 config={**TRAINING_CONFIG,"max_epochs":a.max_epochs,"device":a.device}
 training=train_clean_model(roles["train"],roles["selection_val"],**config)
 raw=score_events(training.model,roles["clean_reference"],device=a.device);calibrator=CleanReferenceScoreCalibrator().fit(raw["reconstruction"],raw["latent"])
 calibrated=score_events(training.model,roles["event_calibration"],calibrator,device=a.device);threshold=calibration_threshold(calibrated["combined_score"],quantile=.99)
 provenance={"classification":CLASSIFICATION,"implementation":implementation_manifest(),"runner":{"path":"scripts/run_gcmr_texbat_cleanstatic.py","sha256":sha256(__file__)},"git_commit":git(["git","rev-parse","HEAD"]),"git_status":git(["git","status","--short"]),"python":platform.python_version(),"torch":torch.__version__,"training_config":training.config,"roles":[vars(r) for r in DEFAULT_ROLES],"role_counts":{"train":299,"selection_val":139,"clean_reference":119,"event_calibration":119,"sealed_held":119},"clean_cache":{"path":str(Path(a.clean_cache).resolve()),"sha256":sha256(a.clean_cache),"metadata":saved_meta},"clean_source_sha256":source_hashes(clean_src),"leakage_contract":{"scaler_model_fit":"cleanStatic train only","epoch_selection":"cleanStatic selection_val only","score_calibrator_fit":"cleanStatic clean_reference only","threshold_q99":"cleanStatic event_calibration only","sealed_held":"after checkpoint freeze and reload","external_ds":"inference only; no DS adaptation"}}
 gate=FreezeGate();model_path=out/"model.pt";save_checkpoint(model_path,training,calibrator,threshold,provenance=provenance);checkpoint_hash=sha256(model_path);gate.checkpoint_saved()
 frozen=load_checkpoint(model_path,expected_provenance=provenance,device=a.device)
 if sha256(model_path)!=checkpoint_hash or frozen.threshold!=threshold or frozen.best_epoch!=training.best_epoch:raise RuntimeError("frozen checkpoint identity/threshold/epoch mismatch")
 gate.checkpoint_reloaded();held_events=select_role_events(clean,DEFAULT_ROLES[-1])
 if len(held_events)!=119:raise RuntimeError("sealed-held count contract mismatch")
 held=score_events(frozen.model,held_events,frozen.calibrator,device=a.device);save_score_csv(out/"cleanStatic_scores.csv",held,frozen.threshold);held_metrics=summarize_scores("cleanStatic",held,frozen.threshold);(out/"cleanStatic_metrics.json").write_text(json.dumps(held_metrics,indent=2,sort_keys=True)+"\n");gate.sealed_held_scored()
 results={}
 for name in ("DS1","DS2","DS3","DS4"):
  events,meta,cache_hash=load_pinned_ds_cache(name,gate,a.ds_cache_dir);scored=score_events(frozen.model,events,frozen.calibrator,device=a.device);save_score_csv(out/f"{name}_scores.csv",scored,frozen.threshold);metrics=summarize_scores(name,scored,frozen.threshold);metrics.update({"checkpoint_sha256":checkpoint_hash,"threshold":frozen.threshold,"cache_sha256":cache_hash,"source_sha256":meta["source_sha256"]});(out/f"{name}_metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n");results[name]=metrics
 summary={"classification":CLASSIFICATION,"checkpoint":str(model_path),"checkpoint_sha256":checkpoint_hash,"threshold":frozen.threshold,"threshold_source":"cleanStatic event_calibration q99 only","best_epoch":frozen.best_epoch,"same_checkpoint_and_threshold_every_ds":all(x["checkpoint_sha256"]==checkpoint_hash and x["threshold"]==frozen.threshold for x in results.values()),"sealed_held":held_metrics,"onset_contract":ONSET_CONTRACT,"results":results,"provenance":provenance};(out/"provenance.json").write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n");(out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");print(json.dumps({"output_dir":str(out),"threshold":threshold,"best_epoch":frozen.best_epoch},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
