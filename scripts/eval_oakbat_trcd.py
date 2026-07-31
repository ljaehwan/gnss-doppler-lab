#!/usr/bin/env python3
"""Strict OAKBAT-native frozen-B0 Tap-Residual Common-Drive evaluation."""
import argparse,copy,hashlib,importlib.util,json,math,os,re,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/"src") not in sys.path:sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.tap_residual_common_drive import causal_smooth_events,extract_b0_innovations,score_common_drive
SCENARIOS=("os1","os2","os3","os4");EXPECTED_CHECKPOINT_SHA256="ab5d77204194c64223292646b7e7c0b83b0d262bde769634efdd4c99bd11b0da";EXPECTED_LABEL="oakbat_cleanStatic_9tap";EXPECTED_RUN_ID="oakbat-cleanStatic-method-a-9tap"
EXPECTED_CAMPAIGN_MANIFEST_SHA256="4519aad0cf69a7f50efa71f713525381decff265228a8d2f05ca1dc1087bfcd4"
EXPECTED_SPLIT_MANIFEST_SHA256="33b06abf1fe632d6c6f770f3d19870eca038ed86c8727c10bce22a2155579353"
EXPECTED_ATTACK_NODE_MANIFEST_SHA256={"os1":"d583a2acf71f230a9a225c1cc14b283c16ed556f21ff8517934494feb3770684","os2":"e4114229a88dd048902e47871e5abf1c79210d978531ef2b8031300bd1751762","os3":"b1ad3dc50edfc5350088ba8480acf764945ded67f5b080eb8f31ed6fe2d97ffd","os4":"a7e003962f54d4ac8f3f5231f539b0a0f9fe5040de636f3e6408666e65c14170"}
EXPECTED_BOUNDARIES={"train":{"start_inclusive":None,"end_exclusive":240.},"validation":{"start_inclusive":250.,"end_exclusive":330.},"calibration":{"start_inclusive":340.,"end_exclusive":410.},"held_clean":{"start_inclusive":420.,"end_exclusive":None}}
ROSTER={"model.pt","training_history.csv","split_manifest.json","model_metadata.json","calibration_prn_scores.csv","calibration.json","held_clean_prn_scores.csv","held_clean_event_scores.csv","held_clean_fpr.json",*[f"partitions/{x}.csv" for x in EXPECTED_BOUNDARIES]};BRANCHES=("event_local_support","event_common_drive_support","event_joint_evidence","event_joint_evidence_causal");RELATION_GATES=tuple(f"local_q{a}_AND_common_q{b}" for a in (95,99) for b in (95,99));ALPHA=.35
def sha256(path,block=8*1024*1024):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for chunk in iter(lambda:f.read(block),b""):h.update(chunk)
 return h.hexdigest()
def verify_trusted_manifest(path,expected_sha256,allow_unpinned_manifests=False):
 actual=sha256(path)
 if actual!=expected_sha256 and not allow_unpinned_manifests:raise ValueError(f"trusted manifest SHA mismatch: {path}")
 return actual
def read_json(path):
 try:x=json.loads(Path(path).read_text())
 except Exception as e:raise ValueError(f"invalid JSON: {path}") from e
 if not isinstance(x,dict):raise ValueError(f"JSON must be object: {path}")
 return x
def verify_campaign_semantics(m):
 if m.get("complete") is not True:raise ValueError("campaign incomplete")
 if m.get("normal_only") is not True:raise ValueError("normal_only_training not true")
 if m.get("attack_inputs_read") is not False:raise ValueError("attack_inputs_read_during_training not false")
def safe_under(root,rel):
 root=Path(root).resolve()
 if not isinstance(rel,str) or not rel or Path(rel).is_absolute():raise ValueError("unsafe artifact pointer")
 p=(root/rel).resolve()
 if root!=p and root not in p.parents:raise ValueError("artifact traversal")
 return p
def verify_hash_roster(root,m,require_exact=False):
 root=Path(root).resolve();a=m.get("artifacts")
 if not isinstance(a,dict) or (require_exact and set(a)!=ROSTER):raise ValueError("frozen artifact roster mismatch")
 for rel,d in a.items():
  p=safe_under(root,rel)
  if not p.is_file():raise ValueError(f"missing artifact: {rel}")
  if not isinstance(d,str) or not re.fullmatch(r"[0-9a-f]{64}",d) or sha256(p)!=d:raise ValueError(f"artifact hash mismatch: {rel}")
def verify_partition_roles(split,frames):
 if split.get("clock")!="window_start_s" or split.get("seq_len")!=12 or split.get("history_contract")!="each partition forms sequences independently; no history crosses boundaries":raise ValueError("split history contract mismatch")
 if split.get("boundaries")!=EXPECTED_BOUNDARIES or set(split.get("partition_csvs",{}))!=set(EXPECTED_BOUNDARIES) or set(frames)!=set(EXPECTED_BOUNDARIES):raise ValueError("partition role contract mismatch")
 for role,b in EXPECTED_BOUNDARIES.items():
  f=frames[role];t=pd.to_numeric(f.get("window_start_s"),errors="coerce")
  bad=f.empty or not np.isfinite(t).all() or (b["start_inclusive"] is not None and (t<b["start_inclusive"]).any()) or (b["end_exclusive"] is not None and (t>=b["end_exclusive"]).any())
  if bad:raise ValueError(f"{role} partition boundary leakage")
  keys=[x for x in ("run_id","window_bin_s","prn") if x in f]
  if len(keys)==3 and f.duplicated(keys).any():raise ValueError(f"{role} duplicate keys")
def identity_frame(f,clean=True,scenario=None):
 req={"label","run_id","source_fingerprint","tap_count","tap_layout","window_start_s","window_end_s"}
 if f.empty or not req.issubset(f):raise ValueError("node identity missing")
 for c in ("label","run_id","source_fingerprint","tap_count","tap_layout"):
  if f[c].isna().any() or f[c].astype(str).nunique()!=1:raise ValueError(f"identity not single-valued: {c}")
 label,run,fp=str(f.label.iloc[0]),str(f.run_id.iloc[0]),str(f.source_fingerprint.iloc[0])
 if not re.fullmatch(r"[0-9a-f]{64}",fp):raise ValueError("fingerprint invalid")
 if clean and (label!=EXPECTED_LABEL or run!=EXPECTED_RUN_ID):raise ValueError("exact clean identity mismatch")
 if scenario and (label!=f"oakbat_{scenario}_9tap" or run!=f"oakbat-{scenario}-method-a-9tap"):raise ValueError("attack identity mismatch")
 if not (pd.to_numeric(f.tap_count,errors="coerce")==9).all() or not f.tap_layout.astype(str).eq("E4,E3,E2,E,P,L,L2,L3,L4").all():raise ValueError("tap identity mismatch")
 return {"label":label,"run_id":run,"source_fingerprint":fp}
def verify_frozen_campaign(root,allow_unpinned_manifests=False):
 root=Path(root).resolve();verify_trusted_manifest(root/"campaign_manifest.json",EXPECTED_CAMPAIGN_MANIFEST_SHA256,allow_unpinned_manifests);verify_trusted_manifest(root/"split_manifest.json",EXPECTED_SPLIT_MANIFEST_SHA256,allow_unpinned_manifests);m=read_json(root/"campaign_manifest.json");verify_campaign_semantics(m)
 if m.get("checkpoint")!="model.pt" or m.get("split")!="split_manifest.json" or m.get("schema")!="gnss-doppler-lab.oakbat-cleanstatic-freeze.v1":raise ValueError("campaign pointer/schema mismatch")
 verify_hash_roster(root,m,True)
 if m["artifacts"]["model.pt"]!=EXPECTED_CHECKPOINT_SHA256 or sha256(root/"model.pt")!=EXPECTED_CHECKPOINT_SHA256:raise ValueError("checkpoint SHA mismatch")
 split=read_json(root/"split_manifest.json");frames={}
 for role,doc in split.get("partition_csvs",{}).items():
  rel=f"partitions/{role}.csv"
  if doc.get("path")!=rel or doc.get("sha256")!=m["artifacts"].get(rel):raise ValueError(f"split partition hash mismatch: {role}")
  frames[role]=pd.read_csv(root/rel)
 verify_partition_roles(split,frames);ids={r:identity_frame(f) for r,f in frames.items()}
 if len({x["source_fingerprint"] for x in ids.values()})!=1:raise ValueError("partition fingerprint mismatch")
 meta=read_json(root/"model_metadata.json")
 if meta.get("checkpoint_sha256")!=EXPECTED_CHECKPOINT_SHA256 or meta.get("standardizer_fit_partition")!="train" or meta.get("hparams",{}).get("seq_len")!=12:raise ValueError("model metadata mismatch")
 return {"root":root,"manifest":m,"split":split,"frames":frames,"identity":ids["calibration"]}
def scorer_module():
 p=ROOT/"scripts"/"score_tap_residual_common_drive.py";s=importlib.util.spec_from_file_location("_oakbat_trcd_b0",p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
def load_model(camp,device="cpu"):return scorer_module().load_frozen_b0(camp["root"]/"model.pt",device,expected_sha256=EXPECTED_CHECKPOINT_SHA256)
def score_frame(frame,loaded,device="cpu"):
 model,features,mean,std,cfg=loaded;v=extract_b0_innovations(frame,model,features,mean,std,seq_len=cfg.seq_len,device=device);nodes,e=score_common_drive(v);return v,nodes,causal_smooth_events(e,alpha=ALPHA)
def calibrate_branches(e,partition_role):
 if partition_role!="calibration":raise ValueError("threshold source role must be calibration")
 if e.empty or any(c not in e for c in BRANCHES):raise ValueError("missing calibration branches")
 out={}
 for c in BRANCHES:
  v=pd.to_numeric(e[c],errors="coerce")
  if not np.isfinite(v).all():raise ValueError("non-finite calibration")
  out[c]={"q95":float(v.quantile(.95)),"q99":float(v.quantile(.99))}
 return out
def branch_flags(e,t):
 out={}
 for c in BRANCHES:
  for q in ("q95","q99"):out[f"{c}_{q}"]=pd.to_numeric(e[c],errors="coerce")>float(t[c][q])
 for a in (95,99):
  for b in (95,99):out[f"local_q{a}_AND_common_q{b}"]=np.logical_and(e.event_local_support>t["event_local_support"][f"q{a}"],e.event_common_drive_support>t["event_common_drive_support"][f"q{b}"])
 return out
def official_masks(e):
 t=pd.to_numeric(e.window_start_s,errors="coerce");return {"pre":(t<110).to_numpy(),"guard":np.logical_and(t>=110,t<130).to_numpy(),"post":(t>=130).to_numpy()}
def metrics(e,flags,mask):
 n=int(mask.sum());f=np.logical_and(np.asarray(flags),mask);rows=e.loc[f].sort_values(["window_start_s","window_end_s","run_id"],kind="mergesort");first=None if rows.empty else rows.iloc[0]
 return {"windows":n,"flags":int(f.sum()),"rate":float(f.sum()/n) if n else 0.,"first_alarm_score_time_s":None if first is None else float(first.window_start_s),"first_alarm_availability_time_s":None if first is None else float(first.window_end_s)}
def evaluate_branches(e,t,region="held"):
 mask=np.ones(len(e),bool);return {"region":region,"branches":{k:metrics(e,v,mask) for k,v in branch_flags(e,t).items()}}
def scenario_report(e,t,scenario):
 if scenario not in SCENARIOS:raise ValueError("unsupported scenario")
 onset=120.;masks=official_masks(e);branches={}
 for k,v in branch_flags(e,t).items():
  pre=metrics(e,v,masks["pre"]);post=metrics(e,v,masks["post"])
  post["score_time_delay_s"]=None if post["first_alarm_score_time_s"] is None else post["first_alarm_score_time_s"]-onset
  post["availability_time_delay_s"]=None if post["first_alarm_availability_time_s"] is None else post["first_alarm_availability_time_s"]-onset
  branches[k]={"pre":pre,"post":post,"guard_windows_excluded":int(masks["guard"].sum())}
 return {"scenario":scenario,"official_scenario_id":"os1a" if scenario=="os1" else scenario,"onset_s":onset,"guard_s":10.,"pre_contract":"window_start_s < 110","post_contract":"window_start_s >= 130","availability_field":"window_end_s event envelope","threshold_source_partition":"calibration","fitting":"none","adaptation":"none","branches":branches}
def assert_finite_json(x,path="root"):
 if isinstance(x,dict):
  for k,v in x.items():assert_finite_json(v,f"{path}.{k}")
 elif isinstance(x,list):
  for i,v in enumerate(x):assert_finite_json(v,f"{path}[{i}]")
 elif isinstance(x,float) and not math.isfinite(x):raise ValueError(f"non-finite output: {path}")
def create_unique_output_dir(parent,stamp=None):
 stamp=stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ");p=Path(parent)/f"oakbat_trcd_{stamp}";p.mkdir(parents=True,exist_ok=False);return p
def atomic_write(path,writer):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix="."+path.name+".",dir=path.parent);os.close(fd)
 try:
  writer(tmp)
  if path.exists():raise FileExistsError(path)
  os.link(tmp,path);os.unlink(tmp)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def atomic_json(path,x):assert_finite_json(x);atomic_write(path,lambda p:Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n"))
def atomic_csv(path,f):
 if f.empty or not np.isfinite(f.select_dtypes(include=[np.number]).to_numpy()).all():raise ValueError("empty/non-finite CSV")
 atomic_write(path,lambda p:f.to_csv(p,index=False))
def shuffle_innovations_within_prn(frame,seed=17,return_report=False):
 out=frame.copy();cols=[f"innovation_{i}" for i in range(9)]
 if set(c for c in out if c.startswith("innovation_"))!=set(cols):raise ValueError("requires exact 9D innovations")
 if not np.isfinite(frame[cols].to_numpy(float)).all():raise ValueError("requires finite innovations")
 rng=np.random.default_rng(seed);stats={"permuted_nonzero_rows":0,"fixed_nonzero_rows":0,"degenerate_zero_rows":0}
 for _,idx in out.groupby(["run_id","prn"],sort=True).groups.items():
  pos=np.asarray(list(idx));vectors=frame.loc[pos,cols].to_numpy(float);norms=np.linalg.norm(vectors,axis=1);nonzero=np.flatnonzero(norms>0)
  stats["degenerate_zero_rows"]+=int(len(pos)-len(nonzero))
  if len(nonzero):
   permutation=rng.permutation(len(nonzero));source=nonzero[permutation];directions=vectors[source]/norms[source,None]
   out.loc[pos[nonzero],cols]=directions*norms[nonzero,None]
   stats["fixed_nonzero_rows"]+=int(np.sum(permutation==np.arange(len(nonzero))))
   stats["permuted_nonzero_rows"]+=int(np.sum(permutation!=np.arange(len(nonzero))))
 out.attrs["relation_destruction_counts"]=dict(stats)
 return (out,dict(stats)) if return_report else out

def verify_relation_only_diagnostic(original_innovations,original_events,shuffled_innovations,shuffled_events,thresholds):
 cols=[f"innovation_{i}" for i in range(9)];meta=[c for c in original_innovations if c not in cols]
 pd.testing.assert_frame_equal(original_innovations[meta].reset_index(drop=True),shuffled_innovations[meta].reset_index(drop=True),check_exact=True)
 shuffled_rmse=np.sqrt(np.mean(np.square(shuffled_innovations[cols].to_numpy(float)),axis=1))
 if not np.allclose(shuffled_rmse,original_innovations.b0_prn_node_rmse.to_numpy(float),rtol=1e-6,atol=1e-9):raise ValueError("relation diagnostic changed row-local RMSE")
 keys=["run_id","window_bin_s"]
 if not original_events[keys].equals(shuffled_events[keys]):raise ValueError("relation diagnostic changed event keys")
 if not np.allclose(original_events.event_local_support,shuffled_events.event_local_support,rtol=1e-12,atol=1e-12):raise ValueError("relation diagnostic changed event_local_support")
 original_flags,shuffled_flags=branch_flags(original_events,thresholds),branch_flags(shuffled_events,thresholds);masks=official_masks(original_events)
 for region,mask in masks.items():
  for q in (95,99):
   name=f"event_local_support_q{q}"
   if not np.array_equal(np.asarray(original_flags[name])[mask],np.asarray(shuffled_flags[name])[mask]):raise ValueError(f"relation diagnostic changed {region} local flags")
def evaluate_in_order(read,freeze):
 cal=read("calibration");t=calibrate_branches(cal,"calibration");freeze(copy.deepcopy(t));reports={"held_clean":evaluate_branches(read("held_clean"),t)}
 for s in SCENARIOS:reports[s]=scenario_report(read(s),t,s)
 return t,reports
def authenticate_attack(path,scenario,allow_unpinned_manifests=False):
 path=Path(path).resolve();verify_trusted_manifest(path.parent/"manifest.json",EXPECTED_ATTACK_NODE_MANIFEST_SHA256[scenario],allow_unpinned_manifests);manifest=read_json(path.parent/"manifest.json");expected=manifest.get("node_table",{}).get("sha256")
 if not isinstance(expected,str) or sha256(path)!=expected:raise ValueError(f"{scenario} node hash mismatch")
 cache=read_json(path.parents[1]/"oakbat_feature_cache_manifest.json");ident={"path":str(path),"size_bytes":path.stat().st_size,"sha256":expected}
 if cache.get("node_table")!=ident:raise ValueError(f"{scenario} feature cache linkage mismatch")
 f=pd.read_csv(path);meta=identity_frame(f,clean=False,scenario=scenario);return f,meta,{"node_csv":ident,"node_manifest":{"path":str(path.parent/"manifest.json"),"sha256":sha256(path.parent/"manifest.json")},"feature_cache_manifest":{"path":str(path.parents[1]/"oakbat_feature_cache_manifest.json"),"sha256":sha256(path.parents[1]/"oakbat_feature_cache_manifest.json")}}
def run(campaign_root,attack_root,output_parent,device="cpu",shuffle_seed=17,allow_unpinned_manifests=False):
 started=datetime.now(timezone.utc).isoformat();camp=verify_frozen_campaign(campaign_root,allow_unpinned_manifests);loaded=load_model(camp,device);out=create_unique_output_dir(output_parent)
 _,_,cal_e=score_frame(camp["frames"]["calibration"],loaded,device);thresholds=calibrate_branches(cal_e,"calibration");calibration={"schema":"gnss-doppler-lab.oakbat-trcd-calibration.v1","threshold_source_partition":"calibration","held_clean_used":False,"attack_inputs_read":False,"row_count":len(cal_e),"thresholds":thresholds,"frozen_at":datetime.now(timezone.utc).isoformat()};atomic_csv(out/"calibration_event_scores.csv",cal_e);atomic_json(out/"calibration.json",calibration)
 _,_,held_e=score_frame(camp["frames"]["held_clean"],loaded,device);atomic_csv(out/"held_clean_event_scores.csv",held_e);held=evaluate_branches(held_e,thresholds,"held_clean");reports={};provenance={};base=Path(attack_root)
 for s in SCENARIOS:
  p=base/s/"multi_prn_method_a_9tap_w1.0_s0.5_normalized_dmcpd"/"normal_prn_node_windows.csv";frame,meta,prov=authenticate_attack(p,s,allow_unpinned_manifests);innov,_,events=score_frame(frame,loaded,device);atomic_csv(out/f"{s}_event_scores.csv",events);rep=scenario_report(events,thresholds,s);shuffle_result=shuffle_innovations_within_prn(innov,shuffle_seed);sh,shuffle_counts=(shuffle_result if isinstance(shuffle_result,tuple) else (shuffle_result,dict(getattr(shuffle_result,"attrs",{}).get("relation_destruction_counts",{}))));_,se=score_common_drive(sh);se=causal_smooth_events(se,alpha=ALPHA);verify_relation_only_diagnostic(innov,events,sh,se,thresholds);rep["relation_destruction_diagnostic"]={"seed":shuffle_seed,"selection_use":"none","method":"permute unit innovation directions only among nonzero rows within PRN, preserve each row norm, and keep zero rows fixed; causal state reset","local_invariance_verified":True,"counts":shuffle_counts,"report":scenario_report(se,thresholds,s)};reports[s]=rep;provenance[s]={"identity":meta,**prov}
 pin_provenance={"enforced":not allow_unpinned_manifests,"override_used":bool(allow_unpinned_manifests),"expected":{"campaign_manifest_sha256":EXPECTED_CAMPAIGN_MANIFEST_SHA256,"split_manifest_sha256":EXPECTED_SPLIT_MANIFEST_SHA256,"attack_node_manifest_sha256":dict(EXPECTED_ATTACK_NODE_MANIFEST_SHA256)},"observed":{"campaign_manifest_sha256":sha256(camp["root"]/"campaign_manifest.json"),"split_manifest_sha256":sha256(camp["root"]/"split_manifest.json"),"attack_node_manifest_sha256":{name:doc.get("node_manifest",{}).get("sha256") for name,doc in provenance.items()}}};summary={"schema":"gnss-doppler-lab.oakbat-native-trcd-evaluation.v1","complete":not allow_unpinned_manifests,"production_status":"production" if not allow_unpinned_manifests else "non-production-incomplete","manifest_pin_provenance":pin_provenance,"started_at":started,"completed_at":datetime.now(timezone.utc).isoformat(),"device":device,"model_frozen":True,"checkpoint_sha256":EXPECTED_CHECKPOINT_SHA256,"campaign_root":str(camp["root"]),"campaign_manifest_sha256":sha256(camp["root"]/"campaign_manifest.json"),"split_manifest_sha256":sha256(camp["root"]/"split_manifest.json"),"split_boundaries":EXPECTED_BOUNDARIES,"seq_len":12,"independent_partition_histories":True,"calibration":calibration,"held_clean":held,"no_attack_fitting":True,"no_adaptation":True,"causal_alpha":ALPHA,"attack_inputs":provenance,"scenarios":reports,"output_directory":str(out)};atomic_json(out/"summary.json",summary);return summary
def main():
 p=argparse.ArgumentParser();p.add_argument("--campaign-root",required=True);p.add_argument("--attack-preprocessed-root",required=True);p.add_argument("--output-parent",required=True);p.add_argument("--device",default="cpu");p.add_argument("--shuffle-seed",type=int,default=17);p.add_argument("--developer-allow-unpinned-manifests",action="store_true",help="developer-only override of trusted manifest SHA pins");a=p.parse_args();print(json.dumps(run(a.campaign_root,a.attack_preprocessed_root,a.output_parent,a.device,a.shuffle_seed,a.developer_allow_unpinned_manifests),indent=2,sort_keys=True))
if __name__=="__main__":main()
