#!/usr/bin/env python3
"""Frozen GCMR v2 cleanStatic/TEXBAT campaign runner (no PRN/QC filtering)."""
from __future__ import annotations
import argparse,csv,hashlib,json,platform,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gnss_doppler_lab.gcmr_v2 import (NodeNormalizer,linear_q99,model_pair_errors,score_model_events,save_native_checkpoint,load_native_checkpoint,canonical_json_hash)
CLEAN_CACHE=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/gcmr-texbat-cleanstatic-event-cache-v1/cleanStatic.relations.npz")
DS_CACHE_ROOT=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/gcmr-texbat-event-cache-v1")
OUTPUT=Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/gcmr-v2-texbat-cleanstatic-seed23")
ROLES={"train":(30,180),"selection":(190,260),"clean_reference":(270,330),"calibration":(340,400),"sealed":(410,470)}
ROLE_COUNTS={"train":299,"selection":139,"clean_reference":119,"calibration":119,"sealed":119}
CONFIG=dict(seed=23,max_epochs=40,patience=6,learning_rate=1e-3,compactness_weight=.01,warmup_epochs=5,pair_hidden=32,event_hidden=64,latent_dim=32)
HASH_KEYS=("checkpoint_sha256","implementation_hash","source_hash","cache_contract_hash","role_hash","config_hash")

CLEAN_CACHE_SHA256="b59c40eeb1c70b3ae1db123f4a7710001f9c0403a67432d37c6e2cf700f0910a"
V1_PRODUCER_AGGREGATE="437557c894d1b3b4132122b70f6a94e2dcf574eed7a766b794b6ebd44eda7d49"
_CACHE_STANDARD_KEYS={"schema_version","relation_contract_version","observation_features","condition_features","source_sha256"}

def load_frozen_producer_cache(path,*,expected_cache_sha256,expected_producer_aggregate):
 """Authenticate a cache against its immutable producer contract, never this consumer."""
 from gnss_doppler_lab.gcmr_experiment import load_event_cache
 path=Path(path); before=sha256(path)
 if before!=expected_cache_sha256: raise ValueError("cache SHA256 mismatch")
 try:
  with np.load(path,allow_pickle=False) as z: meta=json.loads(str(z["metadata_json"]))
 except Exception as exc: raise ValueError(f"invalid frozen cache schema: {exc}") from exc
 if meta.get("schema_version")!=4 or meta.get("relation_contract_version")!=4:
  raise ValueError("producer cache schema mismatch")
 producer=meta.get("implementation") if expected_producer_aggregate is not None else meta.get("external_evaluation_implementation")
 if expected_producer_aggregate is not None:
  if not isinstance(producer,dict) or producer.get("aggregate_sha256")!=expected_producer_aggregate or not isinstance(producer.get("files"),list): raise ValueError("producer implementation mismatch")
 elif not isinstance(producer,dict) or not isinstance(producer.get("external_adapter"),dict): raise ValueError("producer implementation mismatch")
 source_map=meta.get("source_sha256")
 if not isinstance(source_map,dict) or not source_map: raise ValueError("producer source hashes missing")
 expected={k:v for k,v in meta.items() if k not in _CACHE_STANDARD_KEYS}
 events,validated=load_event_cache(path,source_paths=[Path(x) for x in source_map],expected_metadata=expected)
 if validated!=meta or sha256(path)!=before: raise ValueError("frozen cache identity changed during validation")
 schema={k:meta[k] for k in ("schema_version","relation_contract_version","observation_features","condition_features")}
 identity={"cache_sha256":before,"producer_implementation":producer,"producer_implementation_hash":producer.get("aggregate_sha256",canonical_json_hash(producer)),
  "producer_source_sha256":source_map,"producer_source_hash":canonical_json_hash(source_map),"producer_schema":schema,
  "producer_schema_hash":canonical_json_hash(schema),"producer_role_hash":canonical_json_hash(meta.get("role_contract",meta.get("roles",{}))),
  "producer_config_hash":canonical_json_hash(meta.get("config_contract",meta.get("event_contract",{})))}
 return events,validated,identity

def implementation_provenance(producer_identity,consumer_implementation):
 return {"cache_producer_implementation":producer_identity["producer_implementation"],
  "cache_producer_identity":producer_identity,"campaign_consumer_implementation":consumer_implementation}

class FreezeGate:
 def __init__(self):self.state=0;self.training_count=0;self.checkpoint_count=0
 def trained(self):
  self.training_count+=1
  if self.training_count!=1:raise RuntimeError("one training only")
 def checkpoint_saved(self):
  self.checkpoint_count+=1
  if self.state!=0 or self.checkpoint_count!=1 or self.training_count!=1:raise RuntimeError("one checkpoint after training")
  self.state=1
 def checkpoint_reloaded(self):
  if self.state!=1:raise RuntimeError("reload after freeze required")
  self.state=2
 def sealed_scored(self):
  if self.state!=2:raise RuntimeError("sealed must use reloaded checkpoint")
  self.state=3
 def allow_ds(self):
  if self.state!=3:raise RuntimeError("DS cache prohibited until freeze/reload/sealed")
def assert_same_frozen_hashes(rows):
 if not rows or any(len({r[k] for r in rows})!=1 for k in HASH_KEYS):raise RuntimeError("all DS must use same frozen hashes")
 return True
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def role_partitions(events):
 out={}
 names={"train":"train","selection":"selection_val","clean_reference":"clean_reference","calibration":"event_calibration","sealed":"sealed_held"}
 for name,(lo,hi) in ROLES.items():out[name]=[e for e in events if e.window_start_s>=lo and e.window_end_s<=hi]
 if {k:len(x) for k,x in out.items()}!=ROLE_COUNTS:raise ValueError(f"exact role counts required: { {k:len(x) for k,x in out.items()} }")
 identities=[id(e) for x in out.values() for e in x]
 if len(identities)!=len(set(identities)):raise ValueError("role contamination")
 return out
def validate_event(event):
 pairs=np.asarray(event.pair_prns,int);prns=np.unique(pairs)
 if len(prns)<4:raise ValueError("complete graph requires >=4 PRNs")
 canonical={tuple(sorted(map(int,x))) for x in pairs}
 expected={(int(a),int(b)) for i,a in enumerate(prns) for b in prns[i+1:]}
 if len(canonical)!=len(pairs) or canonical!=expected:raise ValueError("event pair graph incomplete/duplicate")
 return prns
def score_groups(model,events,device):
 for e in events:validate_event(e)
 return model_pair_errors(model,events,device=device)
def _json(v):
 if isinstance(v,(np.integer,np.floating)):return v.item()
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,Path):return str(v)
 raise TypeError(type(v).__name__)
def write_scores(out,name,rows,identity):
 event_fields=["scenario","event_index","window_start_s","window_end_s","availability_s",*HASH_KEYS,"N","K","single_score","multi_score","tau_prn","multi_threshold","single_alarm","multi_alarm","classification","candidate_prn"]
 node_fields=["scenario","event_index","window_start_s","window_end_s","availability_s","prn","a","z","activation","degree"]
 pair_fields=["scenario","event_index","window_start_s","window_end_s","availability_s","prn_i","prn_j","e"]
 with (out/f"{name}-events.csv").open("w",newline="") as ef,(out/f"{name}-nodes.csv").open("w",newline="") as nf,(out/f"{name}-pairs.csv").open("w",newline="") as pf:
  ew=csv.DictWriter(ef,event_fields);nw=csv.DictWriter(nf,node_fields);pw=csv.DictWriter(pf,pair_fields);ew.writeheader();nw.writeheader();pw.writeheader()
  for idx,x in enumerate(rows):
   e=x["event"];s=x["score"];base={"scenario":name,"event_index":idx,"window_start_s":e.window_start_s,"window_end_s":e.window_end_s,"availability_s":e.window_end_s}
   ew.writerow({**base,**identity,"N":len(s.prns),"K":s.hard_support,"single_score":s.single_fault_score,"multi_score":s.multi_prn_score,"tau_prn":identity["tau_prn"],"multi_threshold":identity["multi_threshold"],"single_alarm":s.single_alarm,"multi_alarm":s.multi_alarm,"classification":s.classification,"candidate_prn":s.candidate_prn})
   degrees={int(p):int(np.sum(np.asarray(x["pair_prns"])==p)) for p in s.prns}
   for p,a,z,act in zip(s.prns,s.raw,s.z,s.activation):nw.writerow({**base,"prn":p,"a":a,"z":z,"activation":act,"degree":degrees[int(p)]})
   for pair,err in zip(x["pair_prns"],x["pair_errors"]):pw.writerow({**base,"prn_i":pair[0],"prn_j":pair[1],"e":err})
def metrics(rows):
 def one(mask):
  x=[r for r,m in zip(rows,mask) if m]; alarm=[r["score"].classification!="none" for r in x]
  return {"event_count":len(x),"alarm_count":sum(alarm),"alarm_rate":sum(alarm)/len(x) if x else None,"first_alarm_score_end_s":next((r["event"].window_end_s for r,a in zip(x,alarm) if a),None)}
 starts=np.asarray([r["event"].window_start_s for r in rows]);ends=np.asarray([r["event"].window_end_s for r in rows])
 return {k:one(m) for k,m in {"pre":(starts>=30)&(ends<=90),"transition":(starts>=90)&(ends<=110),"post":starts>=110}.items()}
def synthetic_events():
 from gnss_doppler_lab.gcmr_relations import GcmrPairRelationEvent
 rng=np.random.default_rng(23);events=[];prns=np.array([1,2,3,8]);pairs=np.array([(a,b) for i,a in enumerate(prns) for b in prns[i+1:]])
 for i in range(959):
  obs=rng.normal(size=(6,10));mask=np.ones((6,10),bool);cond=rng.normal(size=(6,8));events.append(GcmrPairRelationEvent(i*.5,i*.5+1,pairs,obs,mask,cond))
 return events
def git(cmd):
 try:return subprocess.check_output(cmd,text=True,cwd=Path(__file__).resolve().parents[1]).strip()
 except Exception:return "unavailable"
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",type=Path,default=OUTPUT);p.add_argument("--clean-cache",type=Path,default=CLEAN_CACHE);p.add_argument("--ds-cache-dir",type=Path,default=DS_CACHE_ROOT);p.add_argument("--device",default="cpu");p.add_argument("--max-epochs",type=int,default=40);p.add_argument("--synthetic-smoke",action="store_true");a=p.parse_args(argv)
 import torch,run_gcmr_texbat_cleanstatic as v1
 from gnss_doppler_lab.gcmr_experiment import train_clean_model,source_hashes
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);gate=FreezeGate()
 if a.synthetic_smoke:
  clean=synthetic_events();saved_meta={"synthetic":True,"generator":"synthetic_events","seed":23};sources=[]
  synthetic_producer={"kind":"synthetic-generator","generator":"synthetic_events","seed":23}
  clean_identity={"producer_kind":"synthetic-smoke","pinned_input":False,"producer_implementation":synthetic_producer,"producer_implementation_hash":canonical_json_hash(synthetic_producer)}
  clean_cache_hash=None

 else:
  clean,saved_meta,clean_identity=load_frozen_producer_cache(a.clean_cache,expected_cache_sha256=CLEAN_CACHE_SHA256,expected_producer_aggregate=V1_PRODUCER_AGGREGATE);sources=[Path(x) for x in saved_meta["source_sha256"]];clean_cache_hash=clean_identity["cache_sha256"]
 roles=role_partitions(clean)
 for group in roles.values():
  for event in group:validate_event(event)
 config={**CONFIG,"max_epochs":a.max_epochs,"device":a.device};training=train_clean_model(roles["train"],roles["selection"],**config);gate.trained()
 ref=score_groups(training.model,roles["clean_reference"],a.device);normalizer=NodeNormalizer.fit([x["raw"] for x in ref]);tau=linear_q99(np.concatenate([normalizer.transform(x["raw"]) for x in ref]))
 cal=score_groups(training.model,roles["calibration"],a.device)
 if len(cal)!=119:raise RuntimeError("all 119 calibration events required")
 cal_prn8=sum(8 in x["prns"] for x in cal)
 if not a.synthetic_smoke and cal_prn8==0:raise RuntimeError("PRN8 absent from calibration; retention cannot be proven")
 Rs=[float(np.mean(1/(1+np.exp(-np.clip(normalizer.transform(x["raw"])-tau,-709,709))))) for x in cal];multi=linear_q99(Rs)
 consumer_implementation=v1.implementation_manifest(); implementation=consumer_implementation; role_doc={k:list(v) for k,v in ROLES.items()};source_doc=source_hashes(sources) if sources else {"synthetic":"seed23"}
 provenance={"classification":"gcmr-v2-synthetic-smoke" if a.synthetic_smoke else "gcmr-v2-clean-only-frozen-external",**implementation_provenance(clean_identity,consumer_implementation),"source_sha256":source_doc,"clean_cache":{"path":None if a.synthetic_smoke else str(a.clean_cache.resolve()),"sha256":clean_cache_hash,"metadata":saved_meta},"roles":role_doc,"role_counts":ROLE_COUNTS,"config":config,"calibration":{"event_count":len(cal),"prn8_event_count":cal_prn8,"all_events_contributed":True},"no_filter_proof":{"prn_filtering":False,"qc_filtering":False,"geometry_residual_exclusion":False,"input_output_prn_set_asserted":True},"versions":{"python":platform.python_version(),"numpy":np.__version__,"torch":str(torch.__version__)},"git_commit":git(["git","rev-parse","HEAD"]),"git_status":git(["git","status","--short"])}
 model_path=out/"model-v2.pt";save_native_checkpoint(model_path,training=training,normalizer=normalizer,tau_prn=tau,multi_threshold=multi,provenance=provenance);checkpoint_hash=sha256(model_path);gate.checkpoint_saved()
 model,norm2,tau2,multi2,payload=load_native_checkpoint(model_path,expected_provenance=provenance,expected_sha256=checkpoint_hash,device=a.device);gate.checkpoint_reloaded()
 # Round-trip score identity is checked before any sealed/external I/O.
 ref2=score_groups(model,roles["clean_reference"][:2],a.device)
 for before,after in zip(ref[:2],ref2):
  if not np.array_equal(before["pair_prns"],after["pair_prns"]) or not np.allclose(before["pair_errors"],after["pair_errors"],rtol=0,atol=1e-7):raise RuntimeError("checkpoint score identity failure")
 hashes={"checkpoint_sha256":checkpoint_hash,"implementation_hash":implementation["aggregate_sha256"],"source_hash":canonical_json_hash(source_doc),"cache_contract_hash":canonical_json_hash(saved_meta),"role_hash":canonical_json_hash(role_doc),"config_hash":canonical_json_hash(config),"tau_prn":tau2,"multi_threshold":multi2}
 held=score_model_events(model,roles["sealed"],norm2,tau2,multi2,device=a.device);write_scores(out,"clean-sealed",held,hashes);(out/"clean-sealed-metrics.json").write_text(json.dumps(metrics(held),indent=2)+"\n");gate.sealed_scored()
 results={"clean-sealed":metrics(held)};ds_id=[]
 if not a.synthetic_smoke:
  for name in ("DS1","DS2","DS3","DS4"):
   gate.allow_ds();events,meta,ds_producer_identity=load_frozen_producer_cache(Path(a.ds_cache_dir)/f'{name.lower()}.relations.npz',expected_cache_sha256=v1.DS_CACHE_SHA256[name],expected_producer_aggregate=None);ch=ds_producer_identity['cache_sha256']
   rows=score_model_events(model,events,norm2,tau2,multi2,device=a.device);identity=dict(hashes);write_scores(out,name,rows,identity);m=metrics(rows);m.update({"cache_sha256":ch,"source_sha256":meta["source_sha256"],"cache_producer_identity":ds_producer_identity});(out/f"{name}-metrics.json").write_text(json.dumps(m,indent=2,default=_json)+"\n");results[name]=m;ds_id.append(identity)
  assert_same_frozen_hashes(ds_id)
 summary={"checkpoint":"model-v2.pt","best_epoch":training.best_epoch,"thresholds":{"tau_prn":tau2,"temperature":1.0,"multi_threshold":multi2},"results":results,"provenance":provenance};(out/"provenance.json").write_text(json.dumps(provenance,indent=2,sort_keys=True,default=_json)+"\n");(out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True,default=_json)+"\n")
 files=sorted(x for x in out.iterdir() if x.is_file() and x.name!="SHA256SUMS");(out/"SHA256SUMS").write_text("".join(f"{sha256(x)}  {x.name}\n" for x in files));print(json.dumps({"output_dir":str(out),"checkpoint_sha256":checkpoint_hash,"synthetic":a.synthetic_smoke},indent=2));return 0
class _GateAdapter:
 def __init__(self,g):self.g=g
 def allow_external(self):self.g.allow_ds()
if __name__=="__main__":raise SystemExit(main())
