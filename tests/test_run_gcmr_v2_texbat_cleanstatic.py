import importlib.util
from pathlib import Path
import pytest
P=Path(__file__).parents[1]/"scripts/run_gcmr_v2_texbat_cleanstatic.py"; s=importlib.util.spec_from_file_location("v2run",P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_contract_and_gate():
 assert m.ROLES["calibration"]==(340,400) and m.CONFIG["max_epochs"]==40 and "gcmr-v2-texbat-cleanstatic-seed23" in str(m.OUTPUT)
 g=m.FreezeGate()
 with pytest.raises(RuntimeError):g.allow_ds()
 g.trained();g.checkpoint_saved();g.checkpoint_reloaded()
 with pytest.raises(RuntimeError):g.allow_ds()
 g.sealed_scored();g.allow_ds()
 with pytest.raises(RuntimeError):g.trained()
def test_same_hashes_all_ds():
 row={k:"x" for k in ("checkpoint_sha256","implementation_hash","source_hash","cache_contract_hash","role_hash","config_hash")}
 assert m.assert_same_frozen_hashes([row.copy() for _ in range(4)])
 bad=[row.copy() for _ in range(4)];bad[-1]["config_hash"]="y"
 with pytest.raises(RuntimeError):m.assert_same_frozen_hashes(bad)


def test_frozen_producer_cache_contract_and_consumer_separation(tmp_path):
 import numpy as np
 from gnss_doppler_lab.gcmr_experiment import cache_events
 from gnss_doppler_lab.gcmr_relations import GcmrPairRelationEvent
 source=tmp_path/"source";source.write_bytes(b"source")
 event=GcmrPairRelationEvent(0,1,np.array([[1,2]]),np.zeros((1,10)),np.ones((1,10),bool),np.zeros((1,8)))
 producer={"aggregate_sha256":"a"*64,"files":[{"path":"producer.py","sha256":"b"*64}]}
 cache=tmp_path/"cache.npz";cache_events(cache,[event],source_paths=[source],metadata={"implementation":producer,"role_contract":{"train":[1,2]},"config_contract":{"x":1}})
 digest=m.sha256(cache)
 events,meta,identity=m.load_frozen_producer_cache(cache,expected_cache_sha256=digest,expected_producer_aggregate="a"*64)
 assert len(events)==1 and identity["producer_implementation"]==producer
 consumer={"aggregate_sha256":"c"*64,"files":[{"path":"new-v2.py","sha256":"d"*64}]}
 provenance=m.implementation_provenance(identity,consumer)
 assert provenance["cache_producer_implementation"]==producer
 assert provenance["campaign_consumer_implementation"]==consumer
 assert provenance["cache_producer_implementation"] != provenance["campaign_consumer_implementation"]
 m.load_frozen_producer_cache(cache,expected_cache_sha256=digest,expected_producer_aggregate="a"*64)
 with pytest.raises(ValueError,match="producer implementation"):m.load_frozen_producer_cache(cache,expected_cache_sha256=digest,expected_producer_aggregate="e"*64)
 with pytest.raises(ValueError,match="cache SHA256"):m.load_frozen_producer_cache(cache,expected_cache_sha256="0"*64,expected_producer_aggregate="a"*64)
 source.write_bytes(b"altered")
 with pytest.raises(ValueError,match="source hash"):m.load_frozen_producer_cache(cache,expected_cache_sha256=digest,expected_producer_aggregate="a"*64)

def test_frozen_producer_rejects_schema_tamper(tmp_path):
 import json,numpy as np
 bad=tmp_path/"bad.npz"
 np.savez_compressed(bad,metadata_json=np.asarray(json.dumps({"schema_version":999,"implementation":{"aggregate_sha256":"a"*64,"files":[]},"source_sha256":{"/missing":"x"}})))
 with pytest.raises(ValueError,match="schema"):
  m.load_frozen_producer_cache(bad,expected_cache_sha256=m.sha256(bad),expected_producer_aggregate="a"*64)


def test_synthetic_smoke_cli_records_truthful_unpinned_producer(tmp_path):
 import json, subprocess, sys
 output=tmp_path/"synthetic"
 completed=subprocess.run(
  [sys.executable,str(P),"--synthetic-smoke","--max-epochs","1","--output-dir",str(output)],
  text=True,capture_output=True,timeout=120,
 )
 assert completed.returncode==0, completed.stderr
 provenance=json.loads((output/"provenance.json").read_text())
 summary=json.loads((output/"summary.json").read_text())
 identity=provenance["cache_producer_identity"]
 assert identity["producer_kind"]=="synthetic-smoke"
 assert identity["producer_implementation"] != provenance["campaign_consumer_implementation"]
 assert identity["pinned_input"] is False
 assert provenance["clean_cache"]["path"] is None
 assert provenance["clean_cache"]["sha256"] is None
 assert provenance["score_batching_contract"]["unit"]=="full scenario/role batch"
 assert summary["checkpoint_roundtrip_identity"]["event_count"]==119
 assert summary["checkpoint_roundtrip_identity"]["batching"]=="full clean_reference role batch"
 assert summary["checkpoint_roundtrip_identity"]["max_abs_pair_error"] <= summary["checkpoint_roundtrip_identity"]["tolerance_abs"]


def test_metrics_explicitly_separate_single_multi_and_availability():
 from types import SimpleNamespace as NS
 def row(start,end,single=False,multi=False):
  classification="multi" if multi else "single" if single else "none"
  return {"event":NS(window_start_s=start,window_end_s=end),"score":NS(single_alarm=single,multi_alarm=multi,classification=classification)}
 result=m.metrics([row(30,31,True),row(40,41,False,True),row(89.5,90.5,False,True),row(110,111,True,True)])
 for region in ("all","pre","transition","post"):
  assert "alarm_count" not in result[region] and "alarm_rate" not in result[region]
  assert {"event_count","single_alarm_count","single_alarm_rate","first_single_alarm_availability_s","multi_alarm_count","multi_alarm_rate","first_multi_alarm_availability_s","any_alarm_count","any_alarm_rate"} <= result[region].keys()
 assert result["pre"]["first_multi_alarm_availability_s"]==41
 assert result["post"]["first_multi_alarm_availability_s"]==111
 assert result["crossing_windows_excluded"]=={"event_count":1,"definition":"windows straddling a region boundary (30, 90, or 110 s) are excluded from per-region metrics"}

def test_synthetic_emits_auditable_clean_role_traces_and_manifest(tmp_path):
 import csv,hashlib,json,subprocess,sys,numpy as np
 output=tmp_path/"audit"
 completed=subprocess.run([sys.executable,str(P),"--synthetic-smoke","--max-epochs","1","--output-dir",str(output)],text=True,capture_output=True,timeout=120)
 assert completed.returncode==0,completed.stderr
 summary=json.loads((output/"summary.json").read_text())
 for role in ("clean-reference","clean-calibration"):
  events=list(csv.DictReader((output/f"{role}-events.csv").open()))
  nodes=list(csv.DictReader((output/f"{role}-nodes.csv").open()))
  pairs=list(csv.DictReader((output/f"{role}-pairs.csv").open()))
  assert len(events)==119 and nodes and pairs
  assert {int(x["event_index"]) for x in nodes}==set(range(119))
  assert {int(x["event_index"]) for x in pairs}==set(range(119))
  assert any(int(x["prn"])==8 for x in nodes)
 ref=np.array([float(x["a"]) for x in csv.DictReader((output/"clean-reference-nodes.csv").open())])
 center=float(np.median(ref));scale=max(float(1.4826*np.median(np.abs(ref-center))),1e-9)
 z=(ref-center)/scale;tau=float(np.quantile(z,.99,method="linear"))
 cal=list(csv.DictReader((output/"clean-calibration-nodes.csv").open()))
 groups={}
 for x in cal:groups.setdefault(int(x["event_index"]),[]).append(float(x["a"]))
 Rs=[np.mean(1/(1+np.exp(-np.clip((np.array(v)-center)/scale-tau,-709,709)))) for _,v in sorted(groups.items())]
 multi=float(np.quantile(Rs,.99,method="linear"))
 audit=summary["threshold_trace_audit"]
 assert np.isclose(center,audit["recomputed"]["node_center"],rtol=0,atol=1e-12)
 assert np.isclose(scale,audit["recomputed"]["node_scale"],rtol=0,atol=1e-12)
 assert np.isclose(tau,summary["thresholds"]["tau_prn"],rtol=0,atol=1e-12)
 assert np.isclose(multi,summary["thresholds"]["multi_threshold"],rtol=0,atol=1e-12)
 assert set(summary["results"]) >= {"clean-reference","clean-calibration","clean-sealed"}
 manifest={line.split("  ",1)[1]:line.split("  ",1)[0] for line in (output/"SHA256SUMS").read_text().splitlines()}
 for path in output.iterdir():
  if path.is_file() and path.name!="SHA256SUMS":
   assert manifest[path.name]==hashlib.sha256(path.read_bytes()).hexdigest()
