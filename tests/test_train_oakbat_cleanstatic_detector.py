import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"train_oakbat_cleanstatic_detector.py"
def load_runner():
 spec=importlib.util.spec_from_file_location("oakbat_runner_test",SCRIPT); m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m); return m
def clean_frame():
 m=load_runner(); rows=[]
 for pi,prn in enumerate(("G01","G02")):
  for t in np.arange(0.,440.,.5):
   row={"run_id":"oakbat-cleanStatic-method-a-9tap","source_fingerprint":"frozen-clean-fingerprint","label":"oakbat_cleanStatic_9tap","prn":prn,"segment_index":0,"window_index":int(t*2),"window_bin_s":t,"window_start_s":t,"window_mid_s":t+.5,"window_end_s":t+1,"tap_count":9,"tap_layout":"E4,E3,E2,E,P,L,L2,L3,L4"}; row.update({n:pi+i+t/1000 for i,n in enumerate(m.FEATURE_COLUMNS)}); rows.append(row)
 return pd.DataFrame(rows)
def tiny_frame():
 df=clean_frame(); wanted=[]
 for base in (0.,250.,340.,420.): wanted.extend(base+np.arange(14)*.5)
 return df[df.window_start_s.isin(wanted)].copy()

def authenticated_source(tmp_path, frame=None):
 m=load_runner(); frame=tiny_frame() if frame is None else frame
 receiver_dir=tmp_path/"receiver"; receiver_dir.mkdir(parents=True)
 iq=receiver_dir/"cleanStatic_gps.bin"; iq.write_bytes(b"authenticated oakbat iq")
 receiver=receiver_dir/"manifest.json"
 receiver.write_text(json.dumps({"schema_version":3,"receiver_run_id":"oakbat-cleanStatic-method-a-9tap","status":"complete","source":{"dataset":"OAKBAT","scenario_id":"cleanStatic","iq":str(iq.resolve()),"iq_sha256":m.sha256(iq)},"tracking":{"tap_count":9,"tap_spacing_chips":0.125,"coverage_seconds":479.5}}))
 cache=tmp_path/"cache"; node_dir=cache/"node"; node_dir.mkdir(parents=True)
 source=node_dir/"clean.csv"; frame.to_csv(source,index=False)
 feature=cache/"oakbat_feature_cache_manifest.json"
 feature.write_text(json.dumps({"schema":"gnss-doppler-lab.oakbat-feature-cache.v1","feature_contract":{"feature_mode":"normalized_dmcpd","tap_count":9,"node_feature_columns":m.FEATURE_COLUMNS},"node_table":{"path":str(source.resolve()),"sha256":m.sha256(source)},"receiver_manifest":{"path":str(receiver.resolve()),"sha256":m.sha256(receiver)}}))
 node_manifest=node_dir/"manifest.json"
 node_manifest.write_text(json.dumps({"schema":"gnss-doppler-lab.method-a-9tap-multi-prn-dataset","tap_count":9,"tap_layout":m.TAP_LAYOUT,"node_table":{"path":str(source.resolve()),"sha256":m.sha256(source)}}))
 return source,{"node":node_manifest,"feature":feature,"receiver":receiver,"iq":iq}

def test_split_boundaries_purges_and_sequences_are_partition_local():
 m=load_runner(); p=m.split_chronologically(clean_frame(),12)
 assert [(n,p[n].window_start_s.min(),p[n].window_start_s.max()) for n in p]==[("train",0.,239.5),("validation",250.,329.5),("calibration",340.,409.5),("held_clean",420.,439.5)]
 included=set(pd.concat(p.values()).window_start_s)
 for a,b in ((240,250),(330,340),(410,420)): assert set(np.arange(a,b,.5)).isdisjoint(included)
 assert m.sequence_target_times(p["validation"],12)[0]==256.; assert m.sequence_target_times(p["calibration"],12)[0]==346.; assert m.sequence_target_times(p["held_clean"],12)[0]==426.
def test_cli_has_only_clean_input_output_and_normal_training_knobs():
 m=load_runner(); parser=m.build_parser(); options={o for a in parser._actions for o in a.option_strings}
 assert {"--clean-node-csv","--output-dir","--epochs","--batch-size","--lr","--weight-decay","--hidden-dim","--emb-dim","--dropout","--seed"}<=options
 assert "--seq-len" not in options
 assert not any(any(w in o for w in ("attack","raw","scenario")) for o in options)
 with pytest.raises(SystemExit): parser.parse_args(["--clean-node-csv","clean.csv","--output-dir","out","--attack-csv","bad"])
def test_feature_order_and_train_only_standardizer():
 m=load_runner(); assert m.FEATURE_COLUMNS==["tap_E4_rel_prompt_mean","tap_E3_rel_prompt_mean","tap_E2_rel_prompt_mean","tap_E_rel_prompt_mean","tap_P_rel_prompt_mean","tap_L_rel_prompt_mean","tap_L2_rel_prompt_mean","tap_L3_rel_prompt_mean","tap_L4_rel_prompt_mean"]
 p=m.split_chronologically(clean_frame(),12); mean,std=m.fit_train_standardizer(p); expected=p["train"][m.FEATURE_COLUMNS].to_numpy(np.float32); assert mean==pytest.approx(expected.mean(axis=0))
 for n,v in (("validation",1e8),("calibration",2e8),("held_clean",3e8)): p[n].loc[:,m.FEATURE_COLUMNS]=v
 assert m.fit_train_standardizer(p)[0]==pytest.approx(mean); assert np.all(std>0)
def test_thresholds_use_calibration_only_and_existing_event_builder(monkeypatch):
 m=load_runner(); c=pd.DataFrame({"run_id":["c"]*8,"prn":["G01","G02"]*4,"window_mid_s":[1,1,2,2,3,3,4,4],"prn_node_rmse":[1,2,2,3,3,4,4,5]}); held=c.assign(prn_node_rmse=999.); calls=[]; real=m.gate_lib.build_event_scores
 monkeypatch.setattr(m.gate_lib,"build_event_scores",lambda scores,thresholds,alpha:(calls.append(alpha) or real(scores,thresholds,alpha))); frozen=m.derive_calibration(c)
 assert frozen["node_thresholds"]==pytest.approx({"q50":3.,"q70":3.9,"q80":4.}); assert calls[0]==.75; before=json.dumps(frozen,sort_keys=True); m.held_clean_report(held,frozen); assert json.dumps(frozen,sort_keys=True)==before
 assert frozen["normal_only"] is True and frozen["attack_inputs_read"] is False and frozen["threshold_source_partition"]=="calibration"
@pytest.mark.parametrize("mutation,match",[(lambda d,m:d.assign(tap_count=3),"tap_count"),(lambda d,m:d.assign(tap_layout="E,P,L"),"tap_layout"),(lambda d,m:d.drop(columns=[m.FEATURE_COLUMNS[-1]]),"feature"),(lambda d,m:d.assign(**{m.FEATURE_COLUMNS[0]:np.inf}),"non-finite"),(lambda d,m:pd.concat([d,d.iloc[[0]]]),"duplicate")])
def test_fail_closed_data_contract(mutation,match):
 m=load_runner()
 with pytest.raises(ValueError,match=match): m.validate_clean_frame(mutation(clean_frame(),m))
def test_partition_validator_rejects_overlap_and_empty_partition():
 m=load_runner(); parts=m.split_chronologically(clean_frame(),12)
 parts["validation"]=pd.concat([parts["validation"],parts["train"].iloc[[0]]],ignore_index=True)
 with pytest.raises(ValueError,match="overlap"): m.validate_partitions(parts,12)
 parts=m.split_chronologically(clean_frame(),12); parts["calibration"]=parts["calibration"].iloc[0:0]
 with pytest.raises(ValueError,match="empty"): m.validate_partitions(parts,12)

def test_tiny_training_freezes_then_scores_held_clean_and_rejects_tamper(tmp_path):
 m=load_runner(); source,chain=authenticated_source(tmp_path); result=m.run_campaign(source,tmp_path/"frozen",epochs=1,batch_size=32,hidden_dim=8,emb_dim=8); root=tmp_path/"frozen"
 assert result["complete"] is True; assert m.load_frozen_artifacts(root)["manifest"]["complete"] is True
 calibration=json.loads((root/"calibration.json").read_text()); assert calibration["normal_only"] is True and calibration["attack_inputs_read"] is False and calibration["input_partition"]=="calibration"
 split=json.loads((root/"split_manifest.json").read_text()); assert split["history_contract"]=="each partition forms sequences independently; no history crosses boundaries"; assert set(split["partition_csvs"])=={"train","validation","calibration","held_clean"}
 model=json.loads((root/"model_metadata.json").read_text()); assert model["checkpoint_selection"]=="minimum validation loss" and model["standardizer_fit_partition"]=="train"; assert (root/"held_clean_fpr.json").is_file()
 (root/"calibration.json").write_text("{}")
 with pytest.raises(ValueError,match="tamper|hash"): m.load_frozen_artifacts(root)
def test_rejects_insufficient_prns_and_detects_source_mutation(tmp_path,monkeypatch):
 m=load_runner()
 with pytest.raises(ValueError,match="PRN"): m.split_chronologically(clean_frame().query("prn == 'G01'"),12)
 source,_=authenticated_source(tmp_path); original=m.train_model
 def mutate(*args,**kwargs): source.write_text(source.read_text()+"\n"); return original(*args,**kwargs)
 monkeypatch.setattr(m,"train_model",mutate)
 with pytest.raises(ValueError,match="source mutation"): m.run_campaign(source,tmp_path/"bad",epochs=1,hidden_dim=8,emb_dim=8)


def test_authentication_rejects_forged_path_attack_identity_and_bad_coverage(tmp_path):
 m=load_runner(); source,chain=authenticated_source(tmp_path)
 forged=tmp_path/"forged"/"clean.csv"; forged.parent.mkdir(); tiny_frame().to_csv(forged,index=False)
 with pytest.raises(ValueError,match="manifest|authenticated"): m.authenticate_clean_input(forged)
 for column,value in (("run_id","ds7-attack"),("label","attack"),("source_fingerprint",None)):
  bad=tiny_frame(); bad.loc[:,column]=(["a","b"]*(len(bad)//2) if value is None else value)
  case=tmp_path/column; case.mkdir(); candidate,_=authenticated_source(case,bad)
  with pytest.raises(ValueError,match="run_id|label|fingerprint"): m.authenticate_clean_input(candidate)
 receiver=json.loads(chain["receiver"].read_text()); receiver["tracking"]["coverage_seconds"]=100.; chain["receiver"].write_text(json.dumps(receiver))
 feature=json.loads(chain["feature"].read_text()); feature["receiver_manifest"]["sha256"]=m.sha256(chain["receiver"]); chain["feature"].write_text(json.dumps(feature))
 with pytest.raises(ValueError,match="coverage"): m.authenticate_clean_input(source)

def test_seq_len_is_frozen_and_cannot_be_overridden(tmp_path):
 m=load_runner(); source,_=authenticated_source(tmp_path)
 with pytest.raises((TypeError,ValueError),match="seq_len|training knobs"): m.run_campaign(source,tmp_path/"out",seq_len=11)
 assert m.DEFAULTS["seq_len"]==12

def test_loader_rejects_inventory_deletion_and_pointer_substitution(tmp_path):
 m=load_runner(); source,_=authenticated_source(tmp_path); root=tmp_path/"frozen"
 m.run_campaign(source,root,epochs=1,batch_size=32,hidden_dim=8,emb_dim=8)
 original=json.loads((root/"campaign_manifest.json").read_text()); manifest=json.loads(json.dumps(original)); manifest["artifacts"].pop("training_history.csv"); (root/"campaign_manifest.json").write_text(json.dumps(manifest))
 with pytest.raises(ValueError,match="inventory"): m.load_frozen_artifacts(root)
 manifest=json.loads(json.dumps(original)); manifest["artifacts"]["extra.txt"]="0"*64; (root/"campaign_manifest.json").write_text(json.dumps(manifest))
 with pytest.raises(ValueError,match="inventory"): m.load_frozen_artifacts(root)
 manifest=json.loads(json.dumps(original)); value=manifest["artifacts"].pop("training_history.csv"); manifest["artifacts"]["../training_history.csv"]=value; (root/"campaign_manifest.json").write_text(json.dumps(manifest))
 with pytest.raises(ValueError,match="inventory|traversal"): m.load_frozen_artifacts(root)
 (root/"campaign_manifest.json").write_text(json.dumps(original)); history=(root/"training_history.csv").read_bytes(); (root/"training_history.csv").unlink()
 with pytest.raises(ValueError,match="tamper|hash"): m.load_frozen_artifacts(root)
 (root/"training_history.csv").write_bytes(history)
 # Substitute frozen pointers with other inventoried artifacts.
 root2=tmp_path/"frozen2"; source2,_=authenticated_source(tmp_path/"second"); m.run_campaign(source2,root2,epochs=1,batch_size=32,hidden_dim=8,emb_dim=8)
 doc=json.loads((root2/"campaign_manifest.json").read_text()); original2=json.loads(json.dumps(doc)); doc["calibration"]="model_metadata.json"; (root2/"campaign_manifest.json").write_text(json.dumps(doc))
 with pytest.raises(ValueError,match="pointer|calibration"): m.load_frozen_artifacts(root2)
 original2["checkpoint"]="training_history.csv"; (root2/"campaign_manifest.json").write_text(json.dumps(original2))
 with pytest.raises(ValueError,match="pointer|checkpoint"): m.load_frozen_artifacts(root2)

def test_loader_rejects_parent_manifest_mutation_and_source_substitution(tmp_path):
 m=load_runner(); source,chain=authenticated_source(tmp_path); root=tmp_path/"frozen"; m.run_campaign(source,root,epochs=1,batch_size=32,hidden_dim=8,emb_dim=8)
 chain["receiver"].write_text(chain["receiver"].read_text()+" ")
 with pytest.raises(ValueError,match="source|manifest|tamper|hash"): m.load_frozen_artifacts(root)
 chain["receiver"].write_text(chain["receiver"].read_text().rstrip())
 doc=json.loads((root/"campaign_manifest.json").read_text()); doc["source"]["path"]=str((tmp_path/"substitute.csv").resolve()); (tmp_path/"substitute.csv").write_bytes(source.read_bytes()); (root/"campaign_manifest.json").write_text(json.dumps(doc))
 with pytest.raises(ValueError,match="source|authenticated|path|manifest"): m.load_frozen_artifacts(root)


def test_loader_rejects_split_semantic_pointer_tamper_even_with_rehashed_manifest(tmp_path):
 m=load_runner(); source,_=authenticated_source(tmp_path); root=tmp_path/"frozen"
 m.run_campaign(source,root,epochs=1,batch_size=32,hidden_dim=8,emb_dim=8)
 campaign=json.loads((root/"campaign_manifest.json").read_text())
 split=json.loads((root/"split_manifest.json").read_text())
 split["partition_csvs"]["calibration"]["path"]="partitions/train.csv"
 (root/"split_manifest.json").write_text(json.dumps(split))
 campaign["artifacts"]["split_manifest.json"]=m.sha256(root/"split_manifest.json")
 (root/"campaign_manifest.json").write_text(json.dumps(campaign))
 with pytest.raises(ValueError,match="split.*pointer|split.*linkage"):
  m.load_frozen_artifacts(root)



def test_loader_rejects_coordinated_calibration_and_held_clean_forge(tmp_path):
 m=load_runner(); source,_=authenticated_source(tmp_path); root=tmp_path/"frozen"
 m.run_campaign(source,root,epochs=1,batch_size=32,hidden_dim=8,emb_dim=8)
 campaign=json.loads((root/"campaign_manifest.json").read_text())
 calibration=json.loads((root/"calibration.json").read_text())
 calibration["event_q99_threshold"] += 1.0
 (root/"calibration.json").write_text(json.dumps(calibration))
 held=json.loads((root/"held_clean_fpr.json").read_text())
 held["false_positive_events"]=(held["false_positive_events"]+1)%(held["event_windows"]+1)
 held["false_positive_rate"]=(held["false_positive_rate"]+.25)%1.0
 held["calibration_sha256"]=m.sha256(root/"calibration.json")
 (root/"held_clean_fpr.json").write_text(json.dumps(held))
 campaign["artifacts"]["calibration.json"]=m.sha256(root/"calibration.json")
 campaign["artifacts"]["held_clean_fpr.json"]=m.sha256(root/"held_clean_fpr.json")
 (root/"campaign_manifest.json").write_text(json.dumps(campaign))
 with pytest.raises(ValueError,match="calibration|held-clean|semantic"):
  m.load_frozen_artifacts(root)


def test_authentication_rejects_receiver_schema_and_feature_contract_tamper(tmp_path):
 m=load_runner(); source,chain=authenticated_source(tmp_path)
 receiver=json.loads(chain["receiver"].read_text()); receiver["schema_version"]=2; chain["receiver"].write_text(json.dumps(receiver))
 feature=json.loads(chain["feature"].read_text()); feature["receiver_manifest"]["sha256"]=m.sha256(chain["receiver"]); chain["feature"].write_text(json.dumps(feature))
 with pytest.raises(ValueError,match="schema"):
  m.authenticate_clean_input(source)
 source,chain=authenticated_source(tmp_path/"feature_case")
 feature=json.loads(chain["feature"].read_text()); feature["feature_contract"]["node_feature_columns"]=list(reversed(m.FEATURE_COLUMNS)); chain["feature"].write_text(json.dumps(feature))
 with pytest.raises(ValueError,match="feature.*contract"):
  m.authenticate_clean_input(source)

def test_temporal_contract_accepts_stable_per_prn_acquisition_offsets():
    m=load_runner(); frame=clean_frame()
    frame.loc[frame.prn == "G02", "window_start_s"] += 0.01
    parts=m.split_chronologically(frame, 12)
    assert {name: len(part) for name,part in parts.items()} == {"train":960,"validation":320,"calibration":280,"held_clean":80}


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.assign(window_bin_s=d.window_bin_s + 0.1), "grid"),
    (lambda d: d.assign(window_start_s=d.window_start_s + np.where((d.prn == "G02") & (d.window_bin_s == 100.0), 0.01, 0.0)), "offset"),
])
def test_temporal_contract_rejects_grid_and_prn_clock_disagreement(mutation, match):
    m=load_runner()
    with pytest.raises(ValueError, match=match):
        m.split_chronologically(mutation(clean_frame()), 12)


def test_temporal_contract_rejects_gap_duplicate_and_nonmonotonic_input():
    m=load_runner(); frame=tiny_frame()
    gap=frame[~((frame.prn == "G01") & (frame.window_start_s == 2.0))]
    with pytest.raises(ValueError, match="cadence|gap"):
        m.split_chronologically(gap, 12)
    duplicate=pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate|unique"):
        m.split_chronologically(duplicate, 12)
    indices=list(frame.index); indices[0],indices[1]=indices[1],indices[0]
    with pytest.raises(ValueError, match="monotonic"):
        m.split_chronologically(frame.loc[indices].reset_index(drop=True), 12)


@pytest.mark.parametrize("knob,value", [
    ("lr", 0.0), ("lr", float("nan")), ("weight_decay", -1.0),
    ("weight_decay", float("inf")), ("dropout", -0.1), ("dropout", 1.0),
    ("hidden_dim", 0), ("emb_dim", -1), ("epochs", 0), ("batch_size", 0),
])
def test_invalid_hparams_fail_before_output_creation(tmp_path, knob, value):
    m=load_runner(); source,_=authenticated_source(tmp_path/"source")
    out=tmp_path/"must-not-exist"
    with pytest.raises((TypeError, ValueError), match=knob.replace("_", ".*")):
        m.run_campaign(source, out, **{knob:value})
    assert not out.exists()


def test_open_model_uses_weights_only_and_rejects_nested_non_tensor_payload(tmp_path, monkeypatch):
    m=load_runner(); source,_=authenticated_source(tmp_path/"source"); root=tmp_path/"frozen"
    m.run_campaign(source,root,epochs=1,batch_size=32,hidden_dim=8,emb_dim=8)
    real_load=m.torch.load; calls=[]
    def checked_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only")); return real_load(*args, **kwargs)
    monkeypatch.setattr(m.torch,"load",checked_load)
    m._open_model(root/"model.pt")
    assert calls == [True]
    payload=real_load(root/"model.pt",map_location="cpu",weights_only=True)
    payload["model_state_dict"][next(iter(payload["model_state_dict"]))] = {"unsupported":"nested payload"}
    bad=tmp_path/"malicious-like.pt"; m.torch.save(payload,bad)
    monkeypatch.setattr(m.train_lib,"PrnLocalGRU",lambda *a,**k: pytest.fail("model constructed before payload validation"))
    with pytest.raises(ValueError, match="checkpoint.*state|tensor"):
        m._open_model(bad)


def test_loader_rejects_coordinated_checkpoint_metadata_and_score_substitution(tmp_path):
    m=load_runner(); source,_=authenticated_source(tmp_path/"source")
    root=tmp_path/"frozen"; substitute=tmp_path/"substitute"
    m.run_campaign(source,root,epochs=1,batch_size=32,hidden_dim=8,emb_dim=8,seed=11)
    m.run_campaign(source,substitute,epochs=1,batch_size=32,hidden_dim=8,emb_dim=8,seed=12)
    for name in ("model.pt","model_metadata.json"):
        (root/name).write_bytes((substitute/name).read_bytes())
    checkpoint_hash=m.sha256(root/"model.pt")
    calibration_scores=pd.read_csv(root/"calibration_prn_scores.csv"); calibration_scores["prn_node_rmse"] += .5
    m.atomic_csv(root/"calibration_prn_scores.csv",calibration_scores)
    calibration=m.derive_calibration(calibration_scores)
    calibration.update({"input_csv":"calibration_prn_scores.csv","input_sha256":m.sha256(root/"calibration_prn_scores.csv"),"checkpoint_sha256":checkpoint_hash})
    m.atomic_json(root/"calibration.json",calibration)
    held_scores=pd.read_csv(root/"held_clean_prn_scores.csv"); held_scores["prn_node_rmse"] += .5
    m.atomic_csv(root/"held_clean_prn_scores.csv",held_scores)
    events,report=m.held_clean_report(held_scores,calibration); m.atomic_csv(root/"held_clean_event_scores.csv",events)
    report.update({"score_input_sha256":m.sha256(root/"held_clean_prn_scores.csv"),"calibration_sha256":m.sha256(root/"calibration.json")})
    m.atomic_json(root/"held_clean_fpr.json",report)
    replaced=("model.pt","model_metadata.json","calibration_prn_scores.csv","calibration.json",
              "held_clean_prn_scores.csv","held_clean_event_scores.csv","held_clean_fpr.json")
    campaign=json.loads((root/"campaign_manifest.json").read_text())
    for name in replaced: campaign["artifacts"][name]=m.sha256(root/name)
    (root/"campaign_manifest.json").write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="score|checkpoint|rescor"):
        m.load_frozen_artifacts(root)
