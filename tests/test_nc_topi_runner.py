"""TDD contract tests for the NC-TOPI Stage-0 runner and independent verifier."""
from __future__ import annotations
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT=Path(__file__).parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name[:-3],ROOT/"scripts"/name)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module);return module
r=load("eval_texbat_nc_topi_stage0.py")
v=load("summarize_nc_topi_stage0.py")



def test_production_cli_and_campaign_entry_are_real_and_default(monkeypatch, tmp_path):
    """RED: the old smoke-only CLI is not an acceptable campaign entry."""
    assert hasattr(r, "run_campaign")
    called = []
    monkeypatch.setattr(r, "run_campaign", lambda config, out, **kw: called.append((config, out, kw)))
    assert r.main(["--out", str(tmp_path/"campaign"), "--stop-after-freeze"]) == 0
    assert called and called[0][2]["stop_after_freeze"] is True


def test_stage_gate_requires_typed_fit_objects_not_string_attestations():
    """RED: digesting arbitrary cleanStatic strings was the leakage blocker."""
    gate = r.StageGate()
    with pytest.raises(TypeError):
        gate.seal_fit("covariance", object(), ("cleanStatic/fake",))


def test_strict_verifier_rejects_incomplete_production_inventory(tmp_path):
    """RED: a plausible but incomplete artifact must fail closed."""
    artifact=tiny_artifact(tmp_path/"artifact")
    config=json.loads((artifact/"config.json").read_text())
    config["canonical_inputs"]={"cleanStatic":{"path":"x","sha256":"0"*64}}
    (artifact/"config.json").write_text(json.dumps(config))
    v.write_hash_inventory(artifact)
    result=v.verify_artifact(artifact,verify_source=False)
    assert not result["ok"]
    assert any("inventory" in error or "method" in error or "scenario" in error for error in result["errors"])


def test_source_role_rejects_holdout_mislabeled_as_calibration():
    node=r.NodeWindow("r","cleanStatic","G01",0,0,0,421.,422.,421.5,421.5,4,0,3,np.ones(9))
    assert r.source_role(node)=="normal_holdout"


def test_production_ast_wires_all_typed_apis_and_freeze_precedes_loader():
    import ast,inspect
    tree=ast.parse(Path(r.__file__).read_text())
    calls=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=="core":
            calls.add(node.func.attr)
    assert {"source_support_split","fit_shrinkage_covariance","primary_tangent_basis",
      "produce_topi_scores","condition_topi_scores","build_width_ablation_basis",
      "produce_width_ablation_scores","aggregate_prn_scores","shuffled_control_target",
      "calibrate_threshold","paired_pauc_delta_block_bootstrap","evaluate_stage0_decision"}<=calls
    source=inspect.getsource(r.run_campaign)
    assert source.index("gate.freeze()") < source.index("loader(cfg,b0)") < source.index("gate.load_attack_labels")
    assert "assert_same_epoch_mask" in Path(r.__file__).read_text()


def make_npz(path):
    rows=[]
    for seg,start,prn,ch in [(0,0.,2,1),(0,0.,1,0),(1,10.,1,0)]:
        for i in range(11):
            t=start+i*.1; mag=np.arange(1.,10.)*(1+i/100)
            rows.append((seg,ch,prn,t,int(t*1000),np.stack((mag,np.zeros(9)),axis=1)))
    rows=rows[::-1]
    np.savez(path,complex_iq=np.stack([x[5] for x in rows]).astype(np.float32),
        time_s=[x[3] for x in rows],prn=[x[2] for x in rows],channel=[x[1] for x in rows],
        segment_index=[x[0] for x in rows],sample_count=np.array([x[4] for x in rows],np.uint64))


def test_npz_to_windows_sort_normalization_and_source_support(tmp_path):
    path=tmp_path/"x.npz";make_npz(path)
    nodes,audit=r.build_node_windows(path,"cleanStatic")
    assert audit["stable_sort"]==["segment_index","channel","prn","time_s","sample_count","original_index"]
    assert audit["tap_coordinates_chips"]==[-.5,-.375,-.25,-.125,0.,.125,.25,.375,.5]
    node=next(x for x in nodes if x.segment_index==0 and x.prn=="G01")
    assert (node.source_start_s,node.source_end_s,node.epoch_count)==(0.,1.,11)
    assert node.actual_raw[4]==pytest.approx(1.,abs=1e-6)
    assert node.actual_raw[0]==pytest.approx(.2,rel=1e-5)
    assert (node.source_sample_min,node.source_sample_max)==(0,1000)


def test_sequence_reproduces_legacy_run_prn_grouping_without_cadence_runs():
    nodes=[]
    for seg,start in [(0,0.),(1,10.)]:
        for i in range(14):
            t=start+i*.5
            nodes.append(r.NodeWindow("rec","cleanStatic","G01",0,seg,i,t,t+1,t+.5,t+.5,4,i,i,np.ones(9)))
    examples,audit=r.build_sequence_examples(nodes)
    assert len(examples)==16 and audit["legacy_grouping"]==["run_id","prn"]
    assert audit["rejected_noncontiguous"]==0 and audit["cross_segment_examples"]>0
    assert [x.target_index for x in examples]==list(range(12,28))


def test_frozen_checkpoint_schema_no_train_and_deterministic_inference():
    config=json.loads((ROOT/"configs/nc_topi_stage0.json").read_text())
    model=r.FrozenB0(ROOT/"artifacts/ai_morph_gru_cleanStatic_q70_frame/prn_local_gru_predictor.pt",config["b0"],device="cpu")
    assert model.feature_order==("E4","E3","E2","E","P","L","L2","L3","L4")
    x=np.ones((3,12,9),np.float32);a=model.predict(x);b=model.predict(x)
    assert a.shape==(3,9) and np.array_equal(a,b) and not model.model.training
    with pytest.raises(RuntimeError,match="frozen"):model.train()


def test_legacy_ds7_actual_node_rows_replay_b0_with_defined_gpu_tolerance():
    config=json.loads((ROOT/"configs/nc_topi_stage0.json").read_text())
    model=r.FrozenB0(ROOT/"artifacts/ai_morph_gru_cleanStatic_q70_frame/prn_local_gru_predictor.pt",config["b0"],device="cpu")
    result=r.legacy_b0_positive_control(
      "/home/ubuntu/projects/gnss-doppler-lab/artifacts/texbat_ds7_9tap_eval_20260724/ds7/multi_prn_method_a_9tap_w1.0_s0.5_normalized_dmcpd/normal_prn_node_windows.csv",
      "/home/ubuntu/projects/gnss-doppler-lab/artifacts/ai_morph_gru_cleanStatic_q70_frame/scored/ds7/texbat_ds7_prn_local_scores.csv",model,
      node_sha256="d75a97433d5a4ff25fda7605de599054aed44db3ffa8fd4456baa5114fd771f7",
      score_sha256="d6a3480485d557a5710750ff65c09c836673339a1f23d90d1a41365e9fa91c08")
    assert result["positive_control_pass"] and result["key_sets_equal"]
    assert result["legacy_rows"]==result["covered_keys"]==5465
    assert result["max_abs_rmse_error"]<=result["defined_tolerance"]==3e-4
    assert result["primary_use"] is False


def test_prediction_inverse_alignment_pairs_and_common_mask():
    fake=object.__new__(r.FrozenB0);fake.mean=np.arange(9,dtype=np.float32);fake.std=np.arange(1,10,dtype=np.float32)
    targets=np.stack([np.arange(9.),np.arange(9.)+1]);pred=np.stack([np.ones(9),np.ones(9)*2])
    ids=[r.identity("rec","cleanStatic","G01",i,2+i) for i in range(2)]
    pairs=r.make_peak_pairs(targets,pred,fake,ids)
    assert np.allclose(pairs[0].predicted_raw,fake.mean+fake.std)
    assert np.allclose(pairs[0].residual_standardized,(targets[0]-pairs[0].predicted_raw)/fake.std)
    r.assert_same_epoch_mask({"B0":pairs,"TOPI":pairs,"NC_TOPI":pairs})
    with pytest.raises(ValueError,match="same exact"):r.assert_same_epoch_mask({"B0":pairs,"TOPI":pairs[:-1]})


def test_attack_label_loader_requires_all_typed_clean_fit_seals():
    gate=r.StageGate()
    with pytest.raises(RuntimeError,match="sealed"):gate.load_attack_labels({"DS1":100})
    with pytest.raises(TypeError):gate.seal_fit("covariance",object(),("cleanStatic:fake",))
    with pytest.raises(RuntimeError,match="must be sealed"):gate.freeze()
    assert gate.audit["attack_fit"] is False and gate.audit["attack_loader_calls"]==0


def test_iq_features_memmap_and_strict_history4_causality(tmp_path):
    path=tmp_path/"iq.bin";z=np.empty((2010,2),np.int16);z[:,0]=np.arange(len(z))%17;z[:,1]=np.arange(len(z))%11;z.tofile(path)
    blocks=r.extract_iq_blocks(path,"rec",sample_rate_hz=1000,block_duration_s=.01,block_stride_s=.5)
    assert len(blocks)==5 and blocks[0].sample_count==10 and np.isfinite(blocks[0].features).all()
    context,audit=r.causal_iq_context([1.51],["rec"],blocks,history=4,cadence=.5)
    assert context.shape==(1,4) and audit["strict_causal"]
    assert max(audit["selected_block_ends"][0])<=1.51
    with pytest.raises(ValueError,match="history"):r.causal_iq_context([.4],["rec"],blocks,history=4)


def test_variable_prn_aggregation_permutation_and_method_specific_thresholds():
    scores={"G03":3.,"G01":1.,"G02":2.,"G04":100.,"G05":4.}
    assert r.aggregate_event(scores,"median")==3
    assert r.aggregate_event(dict(reversed(list(scores.items()))),"top25_mean")==52
    thresholds=r.calibrate_all_thresholds({"B0":{"median":[1,2,3]},"TOPI":{"median":[10,20,30]}},[.99])
    assert thresholds["B0/median/q99"]["value"]==3
    assert thresholds["TOPI/median/q99"]["value"]==30
    assert thresholds["B0/median/q99"]["score_digest_sha256"]!=thresholds["TOPI/median/q99"]["score_digest_sha256"]


def tiny_artifact(root):
    root.mkdir();(root/"plots").mkdir()
    # Valid 1x1 RGBA PNG.
    (root/"plots"/"scores.png").write_bytes(bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c63606060f80f0001040100b51c0c020000000049454e44ae426082"))
    rows=[]
    base={"scenario":"cleanStatic","physical_recording_id":"rec","event_id":"e0","target_index":"1","availability_time_s":"321","source_start_s":"320","source_end_s":"321","role":"normal_calibration","phase":"normal","label":"","valid":"1","tracked_prn_count":"2"}
    rows.append({**base,"row_level":"prn","prn":"G01","B0":"1","TOPI":"2","NC_TOPI":"3"})
    rows.append({**base,"row_level":"prn","prn":"G02","B0":"3","TOPI":"4","NC_TOPI":"5"})
    rows.append({**base,"row_level":"event","prn":"","B0_median":"2","TOPI_median":"3","NC_TOPI_median":"4","B0_top25_mean":"3","TOPI_top25_mean":"4","NC_TOPI_top25_mean":"5"})
    fields=sorted(set().union(*(x.keys() for x in rows)))
    with (root/"per_epoch_scores.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    thresholds={"NC_TOPI/median/q99":{"detector":"NC_TOPI","aggregator":"median","quantile":.99,"method":"higher","comparison":"strict >","value":4,"clean_role":"normal_calibration"}}
    payloads={"config.json":{"schema":"gnss-doppler-lab.nc-topi-stage0.v1"},"data_manifest.json":{},"thresholds.json":thresholds,
      "synthetic_physics_tests.json":{"raw_trials":[],"criteria":{}},"bootstrap_results.json":{"comparisons":[]},
      "decision.json":{"status":"INCONCLUSIVE"},"fit_audit.json":{"attack_fit":False,"fit_scenarios":["cleanStatic"]},"provenance.json":{"attack_fit":False}}
    for name,data in payloads.items():(root/name).write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")
    for name in ("scenario_metrics.csv","ablation_metrics.csv"):(root/name).write_text("scenario,method,aggregator,value\n")
    # README depends only on reconstructed evidence; then hash everything except hashes.json.
    provisional=v.compute_summary(root,verify_hashes=False,verify_source=False);provisional["ok"]=True;provisional["errors"]=[]
    (root/"README.md").write_text(v.render_readme(provisional));v.write_hash_inventory(root)
    # hashed_files changes once inventory exists; regenerate README and inventory deterministically.
    summary=v.compute_summary(root,verify_hashes=True,verify_source=False);summary["ok"]=True;summary["errors"]=[]
    (root/"README.md").write_text(v.render_readme(summary));v.write_hash_inventory(root)
    return root


def test_independent_verifier_and_tamper_probes(tmp_path):
    root=tiny_artifact(tmp_path/"artifact")
    result=v.verify_artifact(root,verify_source=False)
    assert result["ok"],result["errors"]
    for filename in ("thresholds.json","scenario_metrics.csv","per_epoch_scores.csv","README.md","plots/scores.png"):
        bad=tmp_path/("bad_"+filename.replace("/","_"));shutil.copytree(root,bad)
        path=bad/filename;path.write_bytes(path.read_bytes()+b"x")
        assert not v.verify_artifact(bad,verify_source=False)["ok"]
    missing=tmp_path/"missing";shutil.copytree(root,missing);(missing/"ablation_metrics.csv").unlink()
    assert not v.verify_artifact(missing,verify_source=False)["ok"]


def test_artifact_stage_no_overwrite_and_failed_marker(tmp_path):
    final=tmp_path/"out"
    with r.ArtifactStage(final) as stage:
        (stage.path/"x").write_text("ok");stage.publish(lambda p:{"ok":True,"errors":[]})
    assert (final/"x").read_text()=="ok"
    with pytest.raises(FileExistsError):
        with r.ArtifactStage(final):pass
    with pytest.raises(RuntimeError):
        with r.ArtifactStage(tmp_path/"fail") as stage:raise RuntimeError("boom")
    markers=list(tmp_path.glob(".fail.tmp.*"));assert markers and (markers[0]/"FAILED.json").exists()


def test_synthetic_physics_raw_trials_are_deterministic_and_threshold_free():
    peak=np.exp(-4*r.core.CANONICAL_TAP_COORDS**2);std=np.linspace(.5,1.3,9);pairs=[]
    for i in range(30):
        ident=r.identity("rec","cleanStatic","G01",i,100+i)
        residual=np.linspace(-.2,.2,9)+i/100
        pairs.append(r.core.PeakPredictionPair(peak+residual,peak,residual/std,std,ident,r.core.RAW_SPACE,r.core.RAW_SPACE,r.core.STANDARDIZED_SPACE,r.core.CANONICAL_TAP_COORDS))
    provenance=r.core.FitProvenance("cleanStatic","normal_train",tuple(x.identity for x in pairs))
    covariance=r.core.fit_shrinkage_covariance(pairs,provenance=provenance)
    a=r.run_synthetic_physics(pairs,covariance);b=r.run_synthetic_physics(pairs,covariance)
    assert len(a["raw_trials"])==100 and len(a["second_peak_grid"])==25
    assert a["attack_thresholds_used"] is False and a["raw_trials"]==b["raw_trials"]
    assert max(x["b0_relative_difference"] for x in a["raw_trials"])<=1e-8



def test_iq_event_evidence_reuses_core_selected_indices(monkeypatch):
    blocks=[r.IQBlock("rec", i*.5, i*.5+.01, i*10, 10, np.full(4, i, float)) for i in range(8)]
    target=r.NodeWindow("rec","cleanStatic","G01",0,0,7,3.,3.,2.5,2.5,4,0,3,np.ones(9))
    example=r.SequenceExample(tuple([target]*12),target,7)
    calls=[]
    original=r.causal_iq_context
    def wrapped(*args,**kwargs):
        predictors,audit=original(*args,**kwargs);calls.append(audit["selected_block_indices"]);return predictors,audit
    monkeypatch.setattr(r,"causal_iq_context",wrapped)
    pair_x,rows,audit=r.build_event_iq_context([example],blocks,"cleanStatic")
    assert len(calls)==1 and calls[0]==[[2,3,4,5]]
    assert rows[0]["sample_offset"]=="20;30;40;50"
    assert np.array_equal(pair_x[0],np.full(4,3.5))
    assert audit["selection_algorithm"]=="group_index_searchsorted"


def test_runner_source_has_single_primary_geometry_and_clean_cache_reuse():
    import inspect
    fit=inspect.getsource(r._fit_geometry);score=inspect.getsource(r.score_scenario)
    assert '"geometry_cache"' in fit and 'state.get("geometry_cache")' in score
    assert score.count("produce_topi_scores(")==1
    assert "condition_topi_scores(" in score
    assert "workspace=workspace" in score



def test_score_scenario_calls_primary_geometry_once_for_attack_and_zero_for_clean_cache(monkeypatch):
    from types import SimpleNamespace
    peak=np.exp(-4*r.core.CANONICAL_TAP_COORDS**2);std=np.ones(9);pairs=[];examples=[]
    for i in range(2):
        residual=np.linspace(-.1,.1,9)+i*.01
        ident=r.identity("rec","DS1","G01",i,10.+i)
        pairs.append(r.core.PeakPredictionPair(peak+residual,peak,residual,std,ident,
          r.core.RAW_SPACE,r.core.RAW_SPACE,r.core.STANDARDIZED_SPACE,r.core.CANONICAL_TAP_COORDS))
        target=SimpleNamespace(source_start_s=10.+i,source_end_s=11.+i)
        examples.append(SimpleNamespace(target=target))
    projection=SimpleNamespace(tangent_energy=.2)
    top=SimpleNamespace(b0=.1,total=.3,tangent=.2,topi=.1)
    calls=[]
    monkeypatch.setattr(r.core,"primary_tangent_basis",lambda *a,**k:SimpleNamespace(matrix=np.ones((9,2))))
    def produce(*a,**k): calls.append(a[0].identity);return top
    monkeypatch.setattr(r.core,"produce_topi_scores",produce)
    monkeypatch.setattr(r.core,"condition_topi_scores",lambda *a,**k:SimpleNamespace(nc_topi=.05))
    monkeypatch.setattr(r.core,"build_width_ablation_basis",lambda *a,**k:object())
    monkeypatch.setattr(r.core,"produce_width_ablation_scores",lambda *a,**k:SimpleNamespace(score=SimpleNamespace(tangent=.25)))
    monkeypatch.setattr(r.core,"weighted_project",lambda *a,**k:projection)
    monkeypatch.setattr(r,"assert_same_epoch_mask",lambda x:None)
    state={"covariance":SimpleNamespace(W=np.eye(9)),"workspace":object(),"conditioner":object(),"shuffled":object()}
    r.score_scenario(examples,pairs,np.ones((2,4)),state,onsets={"DS1":0.})
    assert len(calls)==2
    state["geometry_cache"]=tuple((SimpleNamespace(matrix=np.ones((9,2))),top) for _ in pairs)
    r.score_scenario(examples,pairs,np.ones((2,4)),state,onsets={"DS1":0.})
    assert len(calls)==2
