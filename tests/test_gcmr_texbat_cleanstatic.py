import ast,importlib.util,json
from pathlib import Path
import numpy as np,pytest
from gnss_doppler_lab.gcmr_experiment import DEFAULT_ROLES
from gnss_doppler_lab.gcmr_relations import GcmrPairRelationEvent
SCRIPT=Path(__file__).parents[1]/"scripts/run_gcmr_texbat_cleanstatic.py"
spec=importlib.util.spec_from_file_location("cleanstatic",SCRIPT);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def ev(s):return GcmrPairRelationEvent(s,s+1,np.array([[1,2]]),np.ones((1,10)),np.ones((1,10),bool),np.ones((1,8)))
def test_contract_roles_config():
 assert m.CLASSIFICATION=="texbat_cleanstatic_only_frozen_external_scenario_evaluation"
 assert m.CLEAN_EVENT_CONTRACT["relation_contract_version"]==4 and m.CLEAN_EVENT_CONTRACT["sample_rate_hz"]==25e6 and m.CLEAN_EVENT_CONTRACT["tow0_s"]==477900.
 assert m.TRAINING_CONFIG==dict(seed=23,max_epochs=40,patience=6,learning_rate=1e-3,compactness_weight=.01,warmup_epochs=5,pair_hidden=32,event_hidden=64,latent_dim=32)
 assert [(r.name,r.start_s,r.end_s) for r in DEFAULT_ROLES]==[("train",30,180),("selection_val",190,260),("clean_reference",270,330),("event_calibration",340,400),("sealed_held",410,470)]
def test_grid_roles_counts_contamination():
 events=[ev(i*.5) for i in range(959)];m.validate_clean_event_grid(events);roles=m.role_partitions(events)
 assert {k:len(v) for k,v in roles.items()}=={"train":299,"selection_val":139,"clean_reference":119,"event_calibration":119,"sealed_held":119}
 assert len({id(e) for v in roles.values() for e in v})==795
 with pytest.raises(ValueError,match="959"):m.validate_clean_event_grid(events[:-1])
 with pytest.raises(ValueError,match="grid"):m.validate_clean_event_grid(events[1:]+[ev(480)])
def receiver_manifest():
 auth_entry={"path":"/home/ubuntu/unraid/gnss-datasets/texbat/raw/cleanStatic.bin","sha256":m.CLEAN_IQ_SHA256,"size_bytes":48016392192,"fstat_before":{"device":47,"inode":6,"size_bytes":48016392192},"fstat_after":{"device":47,"inode":6,"size_bytes":48016392192}}
 return {"schema":"gnss-doppler-lab.texbat-clean-complex9-receiver","schema_version":1,"normal_only":True,"attack_inputs_read":False,"source":{"dataset":"TEXBAT","iq":"/home/ubuntu/unraid/gnss-datasets/texbat/raw/cleanStatic.bin","iq_sha256":m.CLEAN_IQ_SHA256,"sample_rate_hz":25000000,"size_bytes":48016392192},"authenticated_inputs":{"iq_before_receiver":dict(auth_entry),"iq_after_receiver":dict(auth_entry)}}
def test_manifest_auth(tmp_path):
 p=tmp_path/"manifest.json";base=receiver_manifest()
 p.write_text(json.dumps(base));assert m.validate_receiver_manifest(p)["normal_only"]
 base["source"]["iq_sha256"]="0"*64;p.write_text(json.dumps(base))
 with pytest.raises(ValueError,match="IQ|manifest"):m.validate_receiver_manifest(p)
@pytest.mark.parametrize("case",["missing","empty","before_only","after_only","before_mismatch","after_mismatch"])
def test_manifest_auth_fails_closed(tmp_path,case):
 p=tmp_path/"manifest.json";base=receiver_manifest();auth=base["authenticated_inputs"]
 if case=="missing":del base["authenticated_inputs"]
 elif case=="empty":base["authenticated_inputs"]={}
 elif case=="before_only":auth.pop("iq_after_receiver")
 elif case=="after_only":auth.pop("iq_before_receiver")
 elif case=="before_mismatch":auth["iq_before_receiver"]["sha256"]="0"*64
 elif case=="after_mismatch":auth["iq_after_receiver"]["sha256"]="0"*64
 p.write_text(json.dumps(base))
 with pytest.raises(ValueError,match="IQ|manifest"):m.validate_receiver_manifest(p)
def test_gate_prohibits_ds_until_reload_and_held():
 g=m.FreezeGate()
 with pytest.raises(RuntimeError):g.allow_external()
 g.checkpoint_saved()
 with pytest.raises(RuntimeError):g.allow_external()
 g.checkpoint_reloaded()
 with pytest.raises(RuntimeError):g.allow_external()
 g.sealed_held_scored();g.allow_external()
 with pytest.raises(RuntimeError):g.checkpoint_saved()
def test_pinned_hashes():
 assert m.DS_CACHE_SHA256=={"DS1":"8ddf6d7a6b70d4c7497ccb75ca0844a50fd6eeebd733ef030dbb11ce1bbbcfef","DS2":"c18201693e4240247b06647e5fae5eb7e51cf82a703018b838c03e678880a24d","DS3":"bc1ded77e5ce3d74262338980ea3668a4f2ba8eaf5c8bcbb25a4ea80821a35ed","DS4":"518be3f9154b52f58a777351f97f81d3a3e3998d6a3343f9cae6791d129c576f"}
def test_regions_latency_strict_and_ds4():
 s=np.array([29,30,89,89.5,90,109,109.5,110,119,120.]);e=s+1;mask=m.region_masks(s,e)
 assert np.flatnonzero(mask["stable_pre"]).tolist()==[1,2] and np.flatnonzero(mask["transition"]).tolist()==[4,5] and np.flatnonzero(mask["stable_post"]).tolist()==[7,8,9]
 scored={"window_start_s":s,"window_end_s":e,"availability_s":e,"combined_score":np.array([0,0,0,0,0,0,0,1,1,2.])};out=m.summarize_scores("DS4",scored,1.)
 assert out["stable_post"]["alarm_count"]==1 and out["first_alarm_score_end_s"]==121. and out["first_alarm_delay_from_primary_onset_s"]==21.
 assert out["onset_conflict"]["auxiliary_onset_s"]==110. and out["post120_sensitivity"]["event_count"]==1
def test_exactly_one_fit_checkpoint_threshold_no_loso_path():
 text=SCRIPT.read_text().lower();tree=ast.parse(text);calls=[n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else "" for n in ast.walk(tree) if isinstance(n,ast.Call)]
 assert "loso" not in text and "donor" not in text and "no ds adaptation" in text
 assert calls.count("train_clean_model")==calls.count("save_checkpoint")==calls.count("calibration_threshold")==1
