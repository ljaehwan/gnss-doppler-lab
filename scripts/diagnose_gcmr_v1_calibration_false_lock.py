#!/usr/bin/env python3
"""Diagnostic-only GCMR-v1 calibration QC; never refits model/scaler/calibrator."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys,platform,importlib.metadata
from pathlib import Path
import numpy as np
import torch
import gnss_doppler_lab.gcmr_experiment as ge
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.gcmr_experiment import load_checkpoint,load_event_cache,score_events
from gnss_doppler_lab.gcmr_geometry import parse_gnss_sdr_gps_ephemeris_xml,satellite_observation,common_clock_removed_residuals
from gnss_doppler_lab.gcmr_relations import load_gnss_sdr_tracking_rows,GcmrPairRelationEvent
FROZEN=ROOT/"artifacts/frozen/gcmr-texbat-cleanstatic-frozen-v1-seed23"
CACHE=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/gcmr-texbat-cleanstatic-event-cache-v1/cleanStatic.relations.npz")
SOURCE=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/receiver/cleanStatic-complex9")
OUTPUT=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/gcmr-v1-calibration-diagnostic-seed23")
QC_RULE={"residual_abs_hz":{"operator":">","threshold":500.0},"median_carrier_lock_test":{"operator":"<","threshold":0.5},"median_cn0_db_hz":{"operator":"<","threshold":35.0},"conjunction":"all","strict":True}
QC_BINNING={"width_s":0.5,"range_s":[340.0,400.0],"interval":"half-open [start,end)","cadence":"receiver samples aggregated by median"}
UNIQUENESS={"minimum_ratio":10.0,"minimum_difference":10.0,"definition":"min healthy-control q99 / automatically-flagged exclusion q99 and corresponding difference"}
def sha256(p):
 h=hashlib.sha256();
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def invalid_qc(residual_hz,lock,cn0):return abs(float(residual_hz))>QC_RULE["residual_abs_hz"]["threshold"] and float(lock)<QC_RULE["median_carrier_lock_test"]["threshold"] and float(cn0)<QC_RULE["median_cn0_db_hz"]["threshold"]
def root_cause_uniqueness(target,healthy_min,*,min_ratio=10.0,min_difference=10.0):return healthy_min/target>=min_ratio and healthy_min-target>=min_difference
def write_sha256_manifest(out):
 files=sorted(x for x in Path(out).iterdir() if x.is_file() and x.name!="SHA256SUMS")
 (Path(out)/"SHA256SUMS").write_text("".join(f"{sha256(x)}  {x.name}\n" for x in files))
def overlaps(a,b,c,d):return a<d and c<b
def incident_pair_mask(pairs,bad):return np.asarray([int(a) not in bad and int(b) not in bad for a,b in pairs],bool)
def require_support(pairs,min_prns=4,min_pairs=6):
 prns=set(map(int,np.asarray(pairs).reshape(-1))) if len(pairs) else set()
 if len(prns)<min_prns or len(pairs)<min_pairs:raise ValueError(f"complete support failed closed: {len(prns)} PRNs/{len(pairs)} pairs")
def qc_bins(rows,eph,receiver,tow0,start=340.,end=400.,step=.5):
 out=[]
 for a in np.arange(start,end,step):
  by={}
  for r in rows:
   if a<=r.time_s<a+step and r.prn in eph:by.setdefault(r.prn,[]).append(r)
  observed={p:float(np.median([r.carrier_doppler_hz for r in rr])) for p,rr in by.items()}
  predicted={p:satellite_observation(receiver,eph[p],(tow0+float(np.median([r.time_s for r in rr])))%604800).predicted_l1_doppler_hz for p,rr in by.items()}
  residual=common_clock_removed_residuals(observed,predicted,visible_prns=sorted(by))
  for p,rr in sorted(by.items()):
   lock=float(np.median([r.carrier_lock_test for r in rr]));cn=float(np.median([r.CN0_SNV_dB_Hz for r in rr]));res=float(residual[p])
   out.append(dict(bin_start_s=float(a),bin_end_s=float(a+step),prn=int(p),residual_hz=res,median_lock=lock,median_cn0=cn,invalid=invalid_qc(res,lock,cn)))
 return out
def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUTPUT);a=ap.parse_args(argv);out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
 original={p.name:sha256(p) for p in sorted(FROZEN.glob("DS*_scores.csv"))}
 
 with np.load(CACHE,allow_pickle=False) as z:
  events=[GcmrPairRelationEvent(float(z["window_start_s"][i]),float(z["window_end_s"][i]),z["pair_prns"][z["offsets"][i]:z["offsets"][i+1]],z["observations"][z["offsets"][i]:z["offsets"][i+1]],z["observation_mask"][z["offsets"][i]:z["offsets"][i+1]],z["conditions"][z["offsets"][i]:z["offsets"][i+1]]) for i in range(len(z["window_start_s"]))]
 cal=[e for e in events if e.window_start_s>=340 and e.window_end_s<=400]
 if len(cal)!=119:raise ValueError(f"expected 119 calibration events, got {len(cal)}")
 prov=json.loads((FROZEN/"provenance.json").read_text()); receiver=prov["clean_cache"]["metadata"]["receiver_position_contract"]["ecef"]
 eph=parse_gnss_sdr_gps_ephemeris_xml(SOURCE/"gps_ephemeris.xml");rows=load_gnss_sdr_tracking_rows(SOURCE/"raw",sample_rate_hz=25e6)
 bins=qc_bins(rows,eph,receiver,477900.);badbins=[x for x in bins if x["invalid"]];filtered=[];removed=0;affected=[];target_removed_by_window={}
 for e in cal:
  bad={x["prn"] for x in badbins if overlaps(e.window_start_s,e.window_end_s,x["bin_start_s"],x["bin_end_s"])};keep=incident_pair_mask(e.pair_prns,bad); nremoved=int((~keep).sum()); removed+=nremoved
  if nremoved: affected.append(e.window_start_s);target_removed_by_window[e.window_start_s]=nremoved
  require_support(e.pair_prns[keep]);filtered.append(GcmrPairRelationEvent(e.window_start_s,e.window_end_s,e.pair_prns[keep],e.observations[keep],e.observation_mask[keep],e.conditions[keep]))
 payload=torch.load(FROZEN/"model.pt",map_location="cpu",weights_only=True); ge.implementation_manifest=lambda:payload["provenance"]["implementation"]; frozen=load_checkpoint(FROZEN/"model.pt",device="cpu");sc=score_events(frozen.model,filtered,frozen.calibrator,device="cpu");threshold=float(np.quantile(np.asarray(sc["combined_score"]),.99,method="linear"))
 # Topology-matched controls use exactly the target's 29 windows and edge-removal count per window.
 affected_events=[e for e in cal if e.window_start_s in target_removed_by_window]
 common=set.intersection(*(set(map(int,e.pair_prns.reshape(-1))) for e in affected_events))
 target_prns=sorted({x["prn"] for x in badbins}); healthy_controls=[]
 for prn in sorted(common-set(target_prns)):
  candidate=[];counts=[]
  for e in cal:
   keep=incident_pair_mask(e.pair_prns,{prn}) if e.window_start_s in target_removed_by_window else np.ones(len(e.pair_prns),bool)
   count=int((~keep).sum());
   if e.window_start_s in target_removed_by_window and count!=target_removed_by_window[e.window_start_s]: break
   require_support(e.pair_prns[keep]);counts.append(count);candidate.append(GcmrPairRelationEvent(e.window_start_s,e.window_end_s,e.pair_prns[keep],e.observations[keep],e.observation_mask[keep],e.conditions[keep]))
  else:
   cs=score_events(frozen.model,candidate,frozen.calibrator,device="cpu")["combined_score"];healthy_controls.append({"prn":prn,"removed_pairs":int(sum(counts)),"affected_windows":len(affected),"threshold_q99":float(np.quantile(np.asarray(cs),.99,method="linear"))})
 healthy_min=min(x["threshold_q99"] for x in healthy_controls)
 confirmed=root_cause_uniqueness(threshold,healthy_min,min_ratio=UNIQUENESS["minimum_ratio"],min_difference=UNIQUENESS["minimum_difference"])
 overlays={}
 for src in sorted(FROZEN.glob("DS*_scores.csv")):
  dst=out/(src.stem+"_alarm_overlay.csv"); n=alarms=0
  with src.open() as fi,dst.open("w",newline="") as fo:
   rd=csv.DictReader(fi); wr=csv.DictWriter(fo,fieldnames=["window_start_s","window_end_s","combined_score","diagnostic_threshold","diagnostic_alarm"]);wr.writeheader()
   for r in rd:
    score=float(r["combined_score"]);alarm=score>threshold;n+=1;alarms+=alarm;wr.writerow({"window_start_s":r["window_start_s"],"window_end_s":r["window_end_s"],"combined_score":r["combined_score"],"diagnostic_threshold":repr(threshold),"diagnostic_alarm":int(alarm)})
  overlays[src.stem]={"events":n,"alarms":int(alarms),"sha256":sha256(dst)}
 after={p.name:sha256(p) for p in sorted(FROZEN.glob("DS*_scores.csv"))}
 summary={"warning":"DIAGNOSTIC ONLY: phase-1 calibration QC; not a refit, retraining, model revision, or v2 detector.","threshold":threshold,"threshold_rule":"q99 numpy linear; strict alarm score > threshold","frozen_checkpoint_sha256":sha256(FROZEN/"model.pt"),"frozen_threshold":frozen.threshold,"calibration_events_rescored":len(filtered),"invalid_prns":sorted({x["prn"] for x in badbins}),"invalid_bin_count":len({(x["bin_start_s"],x["prn"]) for x in badbins}),"affected_window_count":len(affected),"affected_window_range_s":[min(affected),max(affected)] if affected else None,"incident_pairs_removed":removed,"support_contract":"each event >=4 PRNs and >=6 pairs; fail closed","same_checkpoint_scaler_calibrator":True,"model_or_training_modified":False,"original_ds_score_hashes_before":original,"original_ds_score_hashes_after":after,"original_ds_scores_immutable":original==after,"overlays":overlays,"topology_matched_healthy_prn_controls":healthy_controls,"uniqueness_contract":{**UNIQUENESS,"healthy_min_threshold":healthy_min,"ratio":healthy_min/threshold,"difference":healthy_min-threshold},"root_cause_confirmed":confirmed,"provenance":{"script":{"path":str(Path(__file__).resolve()),"sha256":sha256(__file__)},"qc_rule":QC_RULE,"qc_binning":QC_BINNING,"receiver":{"ecef_m":receiver,"gps_tow_at_receiver_time_zero_s":477900.0},"source_inputs":{"clean_cache":{"path":str(CACHE),"sha256":sha256(CACHE)},"ephemeris":{"path":str(SOURCE/"gps_ephemeris.xml"),"sha256":sha256(SOURCE/"gps_ephemeris.xml")},"raw_tracking":[{"path":str(x),"sha256":sha256(x)} for x in sorted((SOURCE/"raw").glob("epl_tracking_ch_*.dat"))]},"frozen_bundle":{"path":str(FROZEN.resolve()),"files":[{"path":x.name,"sha256":sha256(x)} for x in sorted(FROZEN.iterdir()) if x.is_file()]},"runtime":{"python":sys.version,"platform":platform.platform(),"libraries":{n:importlib.metadata.version(n) for n in ["numpy","torch","scipy","pandas"]}}}}
 (out/"qc_bins.json").write_text(json.dumps(bins,indent=2)+chr(10));summary["qc_bins_sha256"]=sha256(out/"qc_bins.json")
 (out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+chr(10));write_sha256_manifest(out)
 if not confirmed: raise RuntimeError("root cause unconfirmed: topology-matched healthy controls also collapse")
 print(json.dumps(summary,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
