import importlib.util
from pathlib import Path
SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_clif_ip_provenance_r2.py"
def load_module():
 spec=importlib.util.spec_from_file_location("provenance_r2",SCRIPT);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def test_blocked_gate_never_invokes_evaluator():
 m=load_module();called=[];manifest={"scenarios":[{"scenario":"os2","same_recording_proven":False,"causal_alignment_proven":False,"permitted":False}]}
 assert m.invoke_evaluator_if_permitted(manifest,lambda names:called.append(names)) is False
 assert called==[]
def test_gate_requires_every_provenance_condition():
 m=load_module();row={key:True for key in m.REQUIRED_BOOLEAN_PROOFS};row.update({"scenario":"x","raw_iq_path":"/x","raw_iq_sha256":"a"*64,"sample_rate_hz":1,"recording_start_sample":0,"b0_source":{"path":"/b","sha256":"b"*64},"m1_source":{"path":"/m","sha256":"c"*64},"b0_window_sample_range":[0,1],"m1_block_sample_range":[0,1],"alignment_offset_samples":0})
 assert m.scenario_permitted(row);row["causal_alignment_proven"]=False;assert not m.scenario_permitted(row)
