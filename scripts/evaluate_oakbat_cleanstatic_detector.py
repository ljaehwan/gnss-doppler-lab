#!/usr/bin/env python3
import argparse,importlib.util,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def load(n,t):
 s=importlib.util.spec_from_file_location(t,ROOT/"scripts"/n);m=importlib.util.module_from_spec(s);sys.modules[t]=m;s.loader.exec_module(m);return m
trainer=load("train_oakbat_cleanstatic_detector.py","_oe_t");pipeline=load("run_oakbat_9tap_detection_pipeline.py","_oe_p");gate_lib=trainer.gate_lib
SCENARIOS=("os1","os2","os3","os4");SCHEMA="gnss-doppler-lab.oakbat-cleanstatic-attack-evaluation.v1"
def atomic_json(p,x):trainer.atomic_json(p,x)
def atomic_csv(p,x):trainer.atomic_csv(p,x)
def scenario_contract(s):
 if s not in SCENARIOS:raise ValueError("unsupported scenario")
 return {"scenario":s,"official_scenario_id":"os1a" if s=="os1" else s,"published_filename":s+".bin","onset_s":120.,"guard_s":10.,"onset_source":"OAKBAT official scenario metadata"}
def identity(p):
 p=Path(p).resolve();a=p.stat();h=trainer.sha256(p);b=p.stat()
 if (a.st_size,a.st_mtime_ns)!=(b.st_size,b.st_mtime_ns):raise RuntimeError("changed while hashing")
 return {"path":str(p),"size_bytes":b.st_size,"sha256":h}
def probability_auc(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float)
 if not len(a) or not len(b):raise ValueError("empty AUC region")
 if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError("nonfinite AUC")
 a=np.sort(a);lo=np.searchsorted(a,b,side="left");eq=np.searchsorted(a,b,side="right")-lo;return float(np.sum(lo+.5*eq)/(len(a)*len(b)))
def qs(x):
 x=np.asarray(x,float)
 if not len(x) or not np.isfinite(x).all():raise ValueError("empty/nonfinite region")
 return {"count":int(len(x)),"median":float(np.quantile(x,.5)),"q50":float(np.quantile(x,.5)),"q90":float(np.quantile(x,.9)),"q99":float(np.quantile(x,.99))}
def discrimination_report(s,e):
 r=s.prn_node_rmse.to_numpy(float);t=s.window_start_s.to_numpy(float);v=e[gate_lib.FINAL_SCORE].to_numpy(float);u=e.window_start_s.to_numpy(float);a=r[t<110];b=r[t>=130];c=v[u<110];d=v[u>=130]
 return {"regions":{"pre":"window_start_s < 110","post":"window_start_s >= 130"},"pooled_prn_rmse":{"pre":qs(a),"post":qs(b),"auc_probability_post_gt_pre_with_half_ties":probability_auc(a,b)},"event_score":{"pre":qs(c),"post":qs(d),"auc_probability_post_gt_pre_with_half_ties":probability_auc(c,d)}}
def dfld(p,t):
 if t is None:return {p+"score_timestamp_s":None,p+"online_availability_s":None,p+"delay_s":None}
 return {p+"score_timestamp_s":float(t),p+"online_availability_s":float(t+1),p+"delay_s":float(t-119)}
def detection_report(e,q):
 t=e.window_start_s.to_numpy(float);v=e[gate_lib.FINAL_SCORE].to_numpy(float);f=v>q;pre=t<110;post=t>=130
 if not len(t) or not np.isfinite(t).all() or not np.isfinite(v).all() or not pre.any() or not post.any():raise ValueError("empty/nonfinite event regions")
 z=np.flatnonzero(np.logical_and(f,post));first=float(t[z[0]]) if len(z) else None;sus=None
 for i in range(2,len(t)):
  if post[i-2] and f[i-2:i+1].all() and np.allclose(np.diff(t[i-2:i+1]),.5,rtol=0,atol=1e-6):sus=float(t[i]);break
 x={"threshold":float(q),"pre_onset_windows":int(pre.sum()),"pre_onset_false_positives":int(np.logical_and(f,pre).sum()),"pre_onset_false_positive_rate":float(np.logical_and(f,pre).sum()/pre.sum()),"post_windows":int(post.sum()),"post_flags":int(np.logical_and(f,post).sum())};x.update(dfld("first_detection_",first));x.update(dfld("first_three_consecutive_",sus));return x
def build_attack_events(s,c):
 if set(c.get("node_thresholds",{}))!={"q50","q70","q80"} or c.get("alpha")!=.75:raise ValueError("calibration contract")
 e=gate_lib.build_event_scores(s,dict(c["node_thresholds"]),alpha=.75)
 if e.empty or not np.isfinite(e.select_dtypes(include=[np.number]).to_numpy(float)).all():raise ValueError("empty/nonfinite event scores")
 return e
def readj(p):
 x=json.loads(Path(p).read_text());
 if not isinstance(x,dict):raise ValueError("invalid manifest")
 return x
def cadence(f):
 trainer.validate_clean_frame(f)
 for _,g in f.groupby(["run_id","prn"]):
  d=np.diff(np.sort(g.window_bin_s.to_numpy(float)))
  if len(d) and not np.allclose(d,.5,rtol=0,atol=1e-6):raise ValueError("cadence gap")
def authenticate(s,iq,r,node,exe):
 iq=Path(iq).resolve();r=Path(r).resolve();node=Path(node).resolve();pipeline.validate_iq(iq);ii=identity(iq);pipeline.validate_cached_receiver(r,s,iq,exe,ii);pipeline.validate_cached_features(node.parents[1],r);rd=readj(r);src=rd["source"];run=f"oakbat-{s}-method-a-9tap"
 if rd.get("receiver_run_id")!=run or src.get("scenario_id")!=s or src.get("iq")!=str(iq) or src.get("iq_sha256")!=ii["sha256"] or src.get("configured_signal_source_samples")!=0 or src.get("signal_source_repeat") is not False:raise ValueError("receiver identity")
 fp=node.parents[1]/"oakbat_feature_cache_manifest.json";fd=readj(fp);ni=identity(node);nm=node.parent/"manifest.json";nd=readj(nm)
 if fd.get("receiver_manifest")!={"path":str(r),"sha256":trainer.sha256(r)} or fd.get("node_table")!=ni:raise ValueError("feature linkage")
 if nd.get("schema")!=trainer.NODE_SCHEMA or nd.get("tap_count")!=9 or nd.get("tap_layout")!=trainer.TAP_LAYOUT or nd.get("node_table",{}).get("sha256")!=ni["sha256"]:raise ValueError("node contract")
 f=pd.read_csv(node);cadence(f)
 if not f.run_id.astype(str).eq(run).all() or "label" not in f or not f.label.astype(str).eq(f"oakbat_{s}_9tap").all():raise ValueError("run/label")
 return f,{"attack_iq":ii,"receiver_manifest":identity(r),"feature_cache_manifest":identity(fp),"node_manifest":identity(nm),"node_csv":ni,"scenario_contract":scenario_contract(s)}
def validate_scores(f,s):
 required={"run_id","prn","window_bin_s","window_start_s","window_mid_s","window_end_s","prn_node_rmse"}
 if f.empty or not required.issubset(f.columns) or f[["run_id","prn"]].isna().any().any() or not f.run_id.astype(str).eq(f"oakbat-{s}-method-a-9tap").all():raise ValueError("score key/run contract")
 if not np.isfinite(f[["window_bin_s","window_start_s","window_mid_s","window_end_s","prn_node_rmse"]].to_numpy(float)).all() or f.duplicated(["run_id","window_bin_s","prn"]).any():raise ValueError("score finite/unique contract")
 for _,g in f.groupby(["run_id","prn"]):
  d=np.diff(np.sort(g.window_bin_s.to_numpy(float)))
  if len(d) and not np.allclose(d,.5,rtol=0,atol=1e-6):raise ValueError("score cadence gap")
def build_report(s,scores,events,q,inputs,out):
 return {"schema":SCHEMA,"scenario":s,"official_scenario_id":scenario_contract(s)["official_scenario_id"],"published_filename":s+".bin","onset":{"seconds":120.,"guard_seconds":10.,"source":"OAKBAT official scenario metadata"},"timing":{"score_timestamp":"window_start_s","online_availability_offset_s":1.},"calibration_policy":{"normal_only_training":True,"attack_inputs_read_during_training":False,"node_quantiles":["q50","q70","q80"],"event_quantile":"q99","alpha":.75},"inputs":inputs,"outputs":{"prn_scores":identity(Path(out)/"attack_prn_scores.csv"),"event_scores":identity(Path(out)/"attack_event_scores.csv")},"raw_score_discrimination":discrimination_report(scores,events),"detection":detection_report(events,q)}
def load_valid_resume(s,out,inputs,calibration):
 try:
  out=Path(out);r=readj(out/"report.json");a=pd.read_csv(out/"attack_prn_scores.csv");b=pd.read_csv(out/"attack_event_scores.csv")
  validate_scores(a,s);expected_events=build_attack_events(a,calibration);trainer._semantic_frame_equal(b,expected_events,"resumed attack event scores")
  x=build_report(s,a,expected_events,float(calibration["event_q99_threshold"]),inputs,out);return r if r==x else None
 except Exception:return None
def evaluate_scenario(s,raw,pre,scored,c,frozen,exe,timeout_s,fr=False,ff=False,fs=False):
 iq=Path(raw)/(s+".bin");pipeline.validate_iq(iq);p=Path(pre)/s;p.mkdir(parents=True,exist_ok=True);r=pipeline.run_receiver(s,iq,p,exe=exe,timeout_s=timeout_s,force=fr);node=pipeline.build_features(s,p,r,force=ff);frame,attack=authenticate(s,iq,r,node,exe);inputs={"frozen":frozen,"attack":attack,"semantics":{"schema":SCHEMA,"onset_s":120.,"guard_s":10.,"pre_end_s":110.,"post_start_s":130.,"availability_offset_s":1.,"alpha":.75,"run_id":f"oakbat-{s}-method-a-9tap"}};out=Path(scored)/s;out.mkdir(parents=True,exist_ok=True)
 if not fs:
  old=load_valid_resume(s,out,inputs,c)
  if old:return old
 scores=trainer.score_partition(frame,Path(frozen["checkpoint"]["path"]));validate_scores(scores,s);events=build_attack_events(scores,c);atomic_csv(out/"attack_prn_scores.csv",scores);atomic_csv(out/"attack_event_scores.csv",events);x=build_report(s,scores,events,float(c["event_q99_threshold"]),inputs,out);atomic_json(out/"report.json",x);return x
def run_evaluation(campaign_root,raw_root,output_root,scenarios,exe,timeout_s=21600,preprocessing_root=None,force_receiver=False,force_features=False,force_scoring=False):
 scenarios=list(scenarios)
 if not scenarios or len(set(scenarios))!=len(scenarios) or any(s not in SCENARIOS for s in scenarios):raise ValueError("scenarios restricted to os1-os4")
 z=trainer.load_frozen_artifacts(campaign_root);c=z["calibration"];root=Path(campaign_root).resolve()
 if set(c.get("node_thresholds",{}))!={"q50","q70","q80"} or c.get("normal_only") is not True or c.get("attack_inputs_read") is not False:raise ValueError("frozen semantics")
 frozen={"campaign_manifest":identity(root/"campaign_manifest.json"),"checkpoint":identity(root/"model.pt"),"calibration":identity(root/"calibration.json")};out=Path(output_root);pre=Path(preprocessing_root) if preprocessing_root else out/"preprocessed";scored=out/"scored"
 for s in scenarios:evaluate_scenario(s,raw_root,pre,scored,c,frozen,exe,timeout_s,force_receiver,force_features,force_scoring)
 m={"schema":"gnss-doppler-lab.oakbat-cleanstatic-attack-evaluation-manifest.v1","complete":True,"selected_scenarios":scenarios,"serial":True,"frozen":frozen,"scenario_reports":{s:identity(scored/s/"report.json") for s in scenarios},"normal_only_training":True,"attack_inputs_read_during_training":False};atomic_json(out/"manifest.json",m);return m
def build_parser():
 p=argparse.ArgumentParser();p.add_argument("--campaign-root",required=True);p.add_argument("--raw-root",default="/home/ubuntu/unraid_hdd/oakbat/gps_l1ca/raw");p.add_argument("--output-root",default=str(ROOT/"artifacts/oakbat_cleanstatic_detector_eval_v1"));p.add_argument("--preprocessing-root");p.add_argument("--scenarios",nargs="+",choices=list(SCENARIOS),default=list(SCENARIOS));p.add_argument("--exe",default=os.environ.get("GNSS_SDR_METHOD_A_EXE",str(ROOT/".tools/gnss-sdr-method-a-9tap")));p.add_argument("--timeout-s",type=int,default=21600);p.add_argument("--force-receiver",action="store_true");p.add_argument("--force-features",action="store_true");p.add_argument("--force-scoring",action="store_true");return p
if __name__=="__main__":
 a=build_parser().parse_args();print(json.dumps(run_evaluation(a.campaign_root,a.raw_root,a.output_root,a.scenarios,a.exe,a.timeout_s,a.preprocessing_root,a.force_receiver,a.force_features,a.force_scoring),indent=2))
