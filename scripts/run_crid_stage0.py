#!/usr/bin/env python3
"""Freeze, replay, analyse, and finalize CRID Stage-0.

``freeze`` performs stat-only attack inventory. ``replay`` refuses attack
payload access unless the pushed freeze SHA is recorded and equals HEAD.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.crid import (CONFIG_ORDER,FEATURE_NAMES,chronological_split,
 empirical_threshold,estimate_causal_delays,fit_normal_model,load_response,
 receiver_configurations,score_aligned,sha256_bytes,canonical_json)
from gnss_doppler_lab.crid_receiver_replay import run_replay,sha256_file
from gnss_doppler_lab.crid_metrics import verdict

ART=ROOT/"artifacts/crid_stage0_counterfactual_receiver_invariance"
SSD=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/crid-stage0-counterfactual-receiver-invariance")
BRANCH="research/crid-stage0-counterfactual-receiver-invariance"
BASE="3e6149f529db8f9e52a215ed696c28896c761844"
RECEIVER=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2c-terminal-drain-repair/receiver-build/src/main/gnss-sdr")
FREEZE_TEXT="CRID configuration frozen before this CRID evaluation; TEXBAT/OAKBAT were previously inspected by the broader project."
DATA={
 "oak_clean":{"domain":"OAK","role":"clean","fs":5_000_000,"raw":"/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/cleanStatic_gps.bin","onset":None},
 "oak_os3":{"domain":"OAK","role":"core","family":"OAK_OS3_OS4","fs":5_000_000,"raw":"/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/os3.bin","onset":120.},
 "oak_os4":{"domain":"OAK","role":"core","family":"OAK_OS3_OS4","fs":5_000_000,"raw":"/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/os4.bin","onset":120.},
 "tex_clean":{"domain":"TEX","role":"clean","fs":25_000_000,"raw":"/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/cleanStatic.bin","onset":None},
 "tex_ds1":{"domain":"TEX","role":"diagnostic","family":"TEX_DS1","fs":25_000_000,"raw":"/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds1.bin","onset":125.},
 "tex_ds3":{"domain":"TEX","role":"core","family":"TEX_DS3","fs":25_000_000,"raw":"/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds3.bin","onset":118.9,"pull_off":195.},
 "tex_ds4":{"domain":"TEX","role":"appendix","family":"TEX_DS4","fs":25_000_000,"raw":"/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds4.bin","onset":130.},
 "tex_ds7":{"domain":"TEX","role":"core","family":"TEX_DS7_DS8","fs":25_000_000,"raw":"/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds7.bin","onset":110.,"time_push":150.},
 "tex_ds8":{"domain":"TEX","role":"core","family":"TEX_DS7_DS8","fs":25_000_000,"raw":"/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds8.bin","onset":110.,"time_push":150.}}
BASE_CONFIG={"OAK":Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_b/oakbat_cleanstatic/rep1/receiver.conf"),
 "TEX":Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_a/texbat_cleanstatic/rep4/receiver.conf")}

def git(*args):return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
def dump(name,value):
 ART.mkdir(parents=True,exist_ok=True);(ART/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
def science_config():
 return {"schema":"gnss-doppler-lab.crid-stage0-config.v1","model":"CRID-GNSS",
  "configuration_statement":FREEZE_TEXT,"receiver_configurations":receiver_configurations(),
  "fields":FEATURE_NAMES,"cn0_lock_usage":"mask/covariance audit only; forbidden in score",
  "cadence_ms":1,"minimum_configurations":4,"minimum_common_prns":4,
  "receiver_initialization":{"method":"TRACE-R2e frozen handoff plus fixed ChannelN.satellite map","TEX_handoff":"texbat_cleanstatic.csv","OAK_handoff":"oakbat_cleanstatic.csv","same_handoff_all_configs":True},
  "execution":{"worker_count":1,"reason":"byte-deterministic replay gate; R2c sequential runs","benchmark_peak_rss_kib":662400,"memory_gib":31,"swap_gib":0},
  "alignment":{"coordinate":"absolute raw sample","max_causal_group_delay_ms":20,
   "resampling":"past-only integer native-cadence shift","gap":"reset predictor; exclude reacquisition"},
  "predictor":{"type":"shared PRN-agnostic ridge causal FIR","order":4,"ridge":1e-3},
  "covariance":{"source":"clean calibration only","shrinkage":.2,"eigen_floor":1e-8},
  "hypotheses":{"H0":"z_ic=H_c x_i+epsilon_ic","H1":"z_ic=H_c x_i+B_c delta_ic+epsilon_ic",
   "latent_dimension":2,"H1_ridge":.1,"score":"median_i[RSS_H0-RSS_H1-(C-1)q log(Cd)]"},
  "pooling":"median over common PRNs","thresholds":{"q99":.99,"q99_5":.995,"target_fpr":.01},
  "clean_split":{"train_fraction":[0,.45],"guard1":[.45,.47],"calibration":[.47,.70],"guard2":[.70,.72],"holdout":[.72,1.]},
  "physical_controls":{"seed":20260820,"raw_iq_only":True,
   "negative":["byte_identical","gain","global_phase","nav_sign","awgn_0.5sigma","awgn_1sigma","awgn_2sigma","cn0_reduction","single_source_code_ramp","single_source_doppler_ramp","common_clock_drift","prn_drop_add","single_prn_disturbance","independent_multipath","zero_delay_collapsed_duplicate"],
   "positive":{"delays_chip":[.05,.15,.30],"powers_db":[-6,-3,0],"prns":[1,4],"phase":"seeded","smooth_pull_off":True}},
  "ablations":{"A0":"C0 residual norm","A1":"pairwise config spread","A2":"no dynamics","A3":"C0+C1","A4":"C0+C2+C3","A5":"independent deviation","Full":"C0+C1+C2+C3 H0/H1"},
  "bootstrap":{"block_s":10,"sensitivity_block_s":30,"replicates":2000,"seed":20260821},
  "gates":{"clean_holdout_fpr_max":.02,"external_preonset_fpr_max":.05,"pauc_min":.8,"detection_min":.70,"delta_pauc_min":.05},
  "datasets":DATA,"prohibited":["neural model","attack-label training","PRN identity","score fusion","power/CN0 score","post-hoc tuning"]}
def manifest():
 out={}
 for p in sorted(ART.rglob("*")):
  if p.is_file() and p.name!="artifact_manifest_sha256.json":out[str(p.relative_to(ART))]=sha256_file(p)
 return out
def cached_raw_hash(name,raw):
 cache=SSD/"raw_hashes";cache.mkdir(parents=True,exist_ok=True);path=cache/f"{name}.json"
 stat=raw.stat();identity={"size":stat.st_size,"mtime_ns":stat.st_mtime_ns,"path":str(raw)}
 if path.exists():
  old=json.loads(path.read_text())
  if old.get("identity")==identity:return old["sha256"]
 value=sha256_file(raw);path.write_text(json.dumps({"identity":identity,"sha256":value},indent=2,sort_keys=True)+"\n");return value
def do_freeze():
 if git("branch","--show-current")!=BRANCH:raise RuntimeError("wrong branch")
 cfg=science_config();dump("config.json",cfg);dump("preregistration.json",cfg)
 inv={"schema":"gnss-doppler-lab.crid-receiver-binary.v1","path":str(RECEIVER),"sha256":sha256_file(RECEIVER),
  "size_bytes":RECEIVER.stat().st_size,"version":subprocess.check_output([str(RECEIVER),"--version"],text=True).strip(),"peak_rss_benchmark":{"TEX_kib":662400,"OAK_kib":180864,"worker_count":1,"status":"PASS"}}
 dump("receiver_binary_inventory.json",inv);dump("receiver_configurations.json",receiver_configurations())
 dump("receiver_config_hashes.json",{k:sha256_bytes(canonical_json(v).encode()) for k,v in receiver_configurations().items()})
 rows=[]
 for name,spec in DATA.items():
  p=Path(spec["raw"]);rows.append({"dataset":name,**spec,"exists_stat_only":p.is_file(),"size_bytes_stat_only":p.stat().st_size if p.is_file() else None,
   "payload_opened":False,"hash_deferred":spec["role"]!="clean"})
 dump("data_inventory.json",{"rows":rows,"attack_payload_bytes_read":0,"operation":"stat only"})
 dump("source_commit.json",{"base_expected":BASE,"base_actual":git("merge-base","HEAD",BASE),"branch":BRANCH,
  "freeze_candidate_sha":git("rev-parse","HEAD"),"statement":FREEZE_TEXT})
 print(json.dumps({"status":"READY_TO_COMMIT_FREEZE","science_sha256":sha256_bytes(canonical_json(cfg).encode())},indent=2))
def assert_attack_authorized():
 auth=ART/"freeze_authorization.json"
 if not auth.exists():raise RuntimeError("freeze authorization missing")
 doc=json.loads(auth.read_text());head=git("rev-parse","HEAD")
 if not doc.get("remote_verified") or doc.get("freeze_sha")!=head:raise RuntimeError("HEAD is not remotely verified freeze SHA")
def do_replay(args):
 spec=DATA[args.dataset]
 if spec["role"]!="clean":assert_attack_authorized()
 skip=args.skip if args.skip is not None else (0. if spec["role"]=="clean" else max(0.,spec["onset"]-40.))
 duration=args.duration if args.duration is not None else (180. if spec["role"]=="clean" else 120.)
 raw=Path(spec["raw"]);raw_hash=cached_raw_hash(args.dataset,raw)
 names=CONFIG_ORDER if args.config=="all" else (args.config,)
 for c in names:
  out=SSD/"replays"/args.dataset/c
  m=run_replay(receiver=RECEIVER,base_config=BASE_CONFIG[spec["domain"]],raw=raw,out=out,
   scenario=args.dataset,config_name=c,fs=spec["fs"],skip_s=skip,duration_s=duration,raw_sha256=raw_hash)
  print(json.dumps({"dataset":args.dataset,"config":c,"exit_code":m["exit_code"],"peak_rss_kib":m["peak_rss_kib"]}))
def load_tables(dataset):
 return {c:load_response(c,(SSD/"replays"/dataset/c).glob("trace_native_1ms_ch_*.bin")) for c in CONFIG_ORDER}
def do_analyze_clean(args):
 tab=load_tables(args.dataset);delay=estimate_causal_delays(tab);samples=np.concatenate([t.sample for t in tab.values()]);split=chronological_split(samples)
 model=fit_normal_model(tab,split["train"],split["calibration"]);rows=score_aligned(tab,model,delay)
 cal=np.array([r["score"] for r in rows if split["calibration"][0]<=r["sample"]<=split["calibration"][-1]]);hold=np.array([r["score"] for r in rows if split["holdout"][0]<=r["sample"]<=split["holdout"][-1]])
 q99=empirical_threshold(cal,.99);q995=empirical_threshold(cal,.995)
 out={"dataset":args.dataset,"delays_ms":delay,"row_count":len(rows),"common_prn_min":min((r["prn_count"] for r in rows),default=0),
  "q99":q99,"q99_5":q995,"holdout_fpr_q99":float(np.mean(hold>q99)),"split":{k:[int(v[0]),int(v[-1]),len(v)] for k,v in split.items()}}
 dump(f"{args.dataset}_clean_analysis.json",out);print(json.dumps(out,indent=2))
def empty_csv(name,fields,rows=()):
 with (ART/name).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def do_inconclusive(args):
 """Fail-closed finalizer used only after a genuine replay/alignment failure."""
 reason=args.reason;v="INCONCLUSIVE_RECEIVER_REPLAY_OR_ALIGNMENT"
 dump("alignment_audit.json",{"status":"FAIL","reason":reason});dump("clean_split_audit.json",{"status":"NOT_EVALUABLE"})
 dump("normal_model_summary.json",{"status":"NOT_FIT"});dump("thresholds.json",{"status":"NOT_CALIBRATED"})
 dump("prompt_placeholder.json",{"note":"not a CRID required result"})
 empty_csv("physical_control_metrics.csv",["domain","control","kind","status","score"])
 empty_csv("scenario_metrics.csv",["dataset","family","status","pauc_0_05","attack_detection_rate"])
 empty_csv("ablation_metrics.csv",["scope","method","status","pauc_0_05"])
 empty_csv("common_support_metrics.csv",["dataset","status","common_prns","valid_configs"])
 empty_csv("external_static_fpr.csv",["dataset","status","fpr"])
 empty_csv("shortcut_audit.csv",["scalar","correlation","auc","status"])
 empty_csv("bootstrap_intervals.csv",["comparison","estimate","lower","upper","status"])
 with gzip.open(ART/"per_epoch_scores.csv.gz","wt",newline="") as f:csv.writer(f).writerow(["dataset","sample","score","alarm","label"])
 with gzip.open(ART/"per_config_state_estimates.csv.gz","wt",newline="") as f:csv.writer(f).writerow(["dataset","sample","prn","config","delay_state","carrier_state"])
 dump("counterfactual_validity.json",{"status":"NOT_EVALUABLE","reason":reason})
 dump("final_verdict.json",{"verdict":v,"reason":reason,"neural_stage1_implemented":False,"attack_data_accessed":False})
 (ART/"README.md").write_text(f"# CRID-GNSS Stage-0\n\n{FREEZE_TEXT}\n\nFinal verdict: `{v}`.\n\nReason: {reason}\n")
 dump("artifact_manifest_sha256.json",manifest());print(v)
def main():
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest="cmd",required=True)
 sp.add_parser("freeze")
 r=sp.add_parser("replay");r.add_argument("--dataset",choices=DATA,required=True);r.add_argument("--config",choices=(*CONFIG_ORDER,"all"),default="all");r.add_argument("--skip",type=float);r.add_argument("--duration",type=float)
 a=sp.add_parser("analyze-clean");a.add_argument("--dataset",choices=("oak_clean","tex_clean"),required=True)
 i=sp.add_parser("finalize-inconclusive");i.add_argument("--reason",required=True)
 args=p.parse_args()
 if args.cmd=="freeze":do_freeze()
 elif args.cmd=="replay":do_replay(args)
 elif args.cmd=="analyze-clean":do_analyze_clean(args)
 else:do_inconclusive(args)
if __name__=="__main__":main()
