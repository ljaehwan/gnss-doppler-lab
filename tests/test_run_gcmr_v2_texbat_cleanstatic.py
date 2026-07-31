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
