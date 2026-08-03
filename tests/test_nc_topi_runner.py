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
    assert source.index("gate.freeze()") < source.index("gate.load_attack_labels") < source.index("run_legacy_positive_control") < source.index("loader(cfg,b0)")
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
    assert '"geometry_cache"' in fit and '_geometry_cache_key(pairs)' in score and 'caches.get(inventory_key)' in score
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
    monkeypatch.setattr(r.core.ProjectionWorkspace,"from_covariance",lambda x:object())
    state={"covariance":SimpleNamespace(W=np.eye(9)),"workspace":object(),"conditioner":object(),"shuffled":object()}
    r.score_scenario(examples,pairs,np.ones((2,4)),state)
    assert len(calls)==2
    # A differently sealed clean inventory of equal length is never reused.
    clean=[]
    for i,p in enumerate(pairs):
        ident=r.identity("rec","cleanStatic","G01",i,10.+i)
        clean.append(r.core.PeakPredictionPair(p.actual_raw,p.predicted_raw,p.residual_standardized,p.standardizer_std,ident,
          r.core.RAW_SPACE,r.core.RAW_SPACE,r.core.STANDARDIZED_SPACE,r.core.CANONICAL_TAP_COORDS))
    state["geometry_cache"]={r._geometry_cache_key(clean):tuple((SimpleNamespace(matrix=np.ones((9,2))),top) for _ in clean)}
    r.score_scenario(examples,pairs,np.ones((2,4)),state)
    assert len(calls)==4
    state["geometry_cache"][r._geometry_cache_key(pairs)]=tuple((SimpleNamespace(matrix=np.ones((9,2))),top) for _ in pairs)
    r.score_scenario(examples,pairs,np.ones((2,4)),state)
    assert len(calls)==4


def _sealed_score_state():
    peak=np.exp(-4*r.core.CANONICAL_TAP_COORDS**2);std=np.linspace(.8,1.2,9);clean=[]
    for i in range(12):
        residual=np.sin(np.arange(9)+i)*.02+np.linspace(-.03,.03,9)
        ident=r.identity("clean","cleanStatic",f"G{i%3+1:02d}",i,10.+i)
        clean.append(r.core.PeakPredictionPair(peak+residual,peak,residual/std,std,ident,
          r.core.RAW_SPACE,r.core.RAW_SPACE,r.core.STANDARDIZED_SPACE,r.core.CANONICAL_TAP_COORDS))
    train=r.core.FitProvenance("cleanStatic","normal_train",tuple(x.identity for x in clean[:8]))
    cov=r.core.fit_shrinkage_covariance(clean[:8],provenance=train);workspace=r.core.ProjectionWorkspace.from_covariance(cov)
    X=np.stack([np.array([i,.1+i/100,.5,.2]) for i in range(12)])
    top=[];cache=[]
    for pair in clean:
        basis=r.core.primary_tangent_basis(pair,r.core.CANONICAL_TAP_COORDS,cov)
        score=r.core.produce_topi_scores(pair,basis,cov,workspace=workspace);top.append(score.topi);cache.append((basis,score))
    actual=r.core.RobustConditioner().fit(X[:8],top[:8],provenance=train)
    shuffled=r.core.RobustConditioner().fit(X[:8],np.asarray(top[:8])[::-1],provenance=train)
    cap=r.core.FitProvenance("cleanStatic","normal_calibration",tuple(x.identity for x in clean[8:]))
    actual.calibrate_cap(X[8:],provenance=cap);shuffled.calibrate_cap(X[8:],provenance=cap)
    return peak,std,cov,{"covariance":cov,"workspace":workspace,"conditioner":actual,"shuffled":shuffled,
      "geometry_cache":{r._geometry_cache_key(clean):tuple(cache)}}


def test_sealed_attack_and_dynamic_pair_inventories_never_reuse_clean_cache():
    from types import SimpleNamespace
    peak,std,cov,state=_sealed_score_state();clean_keys=set(state["geometry_cache"])
    for scenario,count in (("DS1",2),("cleanDynamic",3)):
        pairs=[];examples=[]
        for i in range(count):
            residual=np.cos(np.arange(9)+i)*.03
            ident=r.identity(scenario,scenario,"G01",i,150.+i*.5)
            pairs.append(r.core.PeakPredictionPair(peak+residual,peak,residual/std,std,ident,
              r.core.RAW_SPACE,r.core.RAW_SPACE,r.core.STANDARDIZED_SPACE,r.core.CANONICAL_TAP_COORDS))
            examples.append(SimpleNamespace(target=SimpleNamespace(source_start_s=150.+i*.5,source_end_s=151.+i*.5)))
        scored=r.score_scenario(examples,pairs,np.ones((count,4)),state)
        assert len(scored)==count and all(np.isfinite(list(x["scores"].values())).all() for x in scored)
        assert set(state["geometry_cache"])==clean_keys


def test_stage_gate_rejects_partial_or_wrong_five_attack_onsets():
    gate=r.StageGate();gate._frozen=True
    with pytest.raises(ValueError): gate.load_attack_labels({"DS1":100.})
    with pytest.raises(ValueError): gate.load_attack_labels({"DS1":100,"DS2":100,"DS3":100,"DS7":110,"DS8":111})
    assert gate.load_attack_labels({"DS1":100,"DS2":100,"DS3":100,"DS7":110,"DS8":110})["DS8"]==110


def test_clean_dynamic_unavailable_reports_intended_runtime_error(monkeypatch):
    monkeypatch.setattr(r,"load_external_normal_nodes",lambda *a,**k:([],{"available":False,"reason":"missing fixture"}))
    with pytest.raises(RuntimeError,match="cleanDynamic nodes unavailable: missing fixture"):
        r._scenario_lineage({"clean_dynamic_nodes":{"path":"x","sha256":"0"}},"cleanDynamic",object())


def test_public_config_forbids_test_fixture_escape():
    config=json.loads((ROOT/"configs/nc_topi_stage0.json").read_text());config["test_fixture"]=True
    with pytest.raises(ValueError,match="test_fixture"):r.core.validate_config(config)
    assert "test_fixture" not in r.parse_args([]).__dict__


def test_iq_full_join_and_structural_tamper_probes(tmp_path):
    root=tmp_path/"iq";root.mkdir();cfg={"iq_conditioner":{"sample_rate_hz":1000,"block_duration_seconds":.01,"block_stride_seconds":.5}}
    (root/"config.json").write_text(json.dumps(cfg))
    base={"scenario":"cleanStatic","physical_recording_id":"rec","event_id":"rec@2","target_index":"0","availability_time_s":"2.6",
      "source_start_s":"1.6","source_end_s":"2.6","role":"normal_train","phase":"normal","label":"","valid":"1","tracked_prn_count":"2"}
    epoch=[{**base,"row_level":"event","prn":""},{**base,"row_level":"prn","prn":"G01","source_start_s":"1.6"},
           {**base,"row_level":"prn","prn":"G02","source_start_s":"1.7"}]
    with (root/"per_epoch_scores.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(x.keys() for x in epoch))));w.writeheader();w.writerows(epoch)
    features=np.arange(16,dtype=float).reshape(4,4);row={"scenario":"cleanStatic","physical_recording_id":"rec","event_id":"rec@2",
      "window_bin_s":"2","target_source_start_s":"1.6","history_blocks":"4","cadence_seconds":"0.5",
      "block_start_s":"0;0.5;1;1.5","block_end_s":"0.01;0.51;1.01;1.51","sample_offset":"0;500;1000;1500",
      "sample_count":"10;10;10;10","block_features_json":json.dumps(features.tolist()),
      "context_features_json":json.dumps(features.mean(axis=0).tolist()),"linked_prns":"G01;G02","linked_pair_count":"2",
      "history_reducer":"arithmetic_mean_per_feature"}
    fields=list(row)
    def write(value):
      with (root/"iq_context.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(value)
    write(row);errors=[];result=v.verify_iq_causality(root,errors);assert not errors and result["joined"]==1
    for field,value in (("sample_offset","1;500;1000;1500"),("target_source_start_s","1.7"),("linked_prns","G01"),
                        ("block_end_s","0.01;0.51;1.01;1.52")):
      bad=dict(row);bad[field]=value;write(bad);errors=[];v.verify_iq_causality(root,errors);assert errors,field


def test_full_sustained_metric_inventory_and_decision_uses_primary_finite_delays():
    from types import SimpleNamespace
    events=[]
    def add(scenario,start,phase,label,value,role="evaluation"):
      meta={"scenario":scenario,"physical_recording_id":scenario,"availability_time_s":start,
        "source_start_s":start,"source_end_s":start,"phase":phase,"label":label,"role":role}
      scores={m:{a:value for a in r.AGGREGATORS} for m in r.METHODS};events.append({"meta":meta,"scores":scores})
    add("cleanStatic",430.,"normal","",.1,"normal_holdout");add("cleanDynamic",50.,"normal","",.1)
    onsets={"DS1":100,"DS2":100,"DS3":100,"DS7":110,"DS8":110}
    for scenario,onset in onsets.items():
      for t in (40.,40.5,41.):add(scenario,t,"stable_pre",0,.1)
      for t in (onset,onset+.5,onset+1.,onset+40.,onset+40.5,onset+41.):add(scenario,t,"post",1,1.)
    thresholds={f"{m}/{a}/{q}":SimpleNamespace(value=.5) for m in r.METHODS for a in r.AGGREGATORS for q in ("q99","q995")}
    metrics,ab,bootstrap,_=r._campaign_statistics(events,thresholds,{"attacks":{"onsets_seconds":onsets}},None,None)
    assert len(metrics)==1480
    expected={"fpr","roc_auc","pr_auc","pauc","detection_rate","persistent_alarm_ratio","sustained_delay"}
    attack={x["metric"] for x in metrics if x["scenario"]=="DS1"};assert attack==expected
    delays=[x for x in metrics if x["scenario"]=="DS1" and x["metric"]=="sustained_delay"]
    assert delays and all(x["value"]==pytest.approx(1.) and x["censored"] is False and x["status"]=="detected" for x in delays)
    synthetic={"criteria":{"equal_rmse_pass":True,"nuisance_pass":True,"second_peak_pass":True}}
    decision=r._derive_decision(metrics,bootstrap,synthetic)
    assert decision["evidence"]["nc_delay"]["DS1"]==pytest.approx(1.)
    assert decision["evidence"]["b0_delay"]["DS1"]==pytest.approx(1.)


def test_physics_noise_equality_boundary_only_is_nonstrict():
    rows=[]
    for kind in ("amplitude","shift","noise"):
      rows.append({"kind":kind,"topi_normalized":1.,"b0_normalized":1.})
    assert r._nuisance_kind_pass(rows)=={"amplitude":False,"shift":False,"noise":True}
    assert v._nuisance_kind_pass_independent(rows)=={"amplitude":False,"shift":False,"noise":True}


def test_private_tiny_full_campaign_executes_all_stages_and_separate_verifier(tmp_path):
    import hashlib
    import torch
    config=json.loads((ROOT/"configs/nc_topi_stage0.json").read_text())
    # Tiny synthetic frozen checkpoint (private fixture only).
    model_cfg={"seq_len":12,"hidden_dim":8,"emb_dim":8,"dropout":0.0}
    torch.manual_seed(4);model=r._FrozenGRUModel.construct(torch,9,model_cfg)
    checkpoint=tmp_path/"synthetic_b0.pt"
    torch.save({"config":model_cfg,"node_feature_columns":list(r.FrozenB0.EXPECTED_FEATURE_COLUMNS),
      "standardizer":{"node_mean":np.exp(-4*r.core.CANONICAL_TAP_COORDS**2).astype(np.float32).tolist(),
                      "node_std":np.linspace(.2,.4,9,dtype=np.float32).tolist()},
      "model_state_dict":model.state_dict()},checkpoint)
    config["b0"]["checkpoint_sha256"]=r.sha256_file(checkpoint);config["checkpoint_path"]=str(checkpoint)
    config["iq_conditioner"]["sample_rate_hz"]=1000
    config["iq_conditioner"]["block_duration_seconds"]=.01
    config["iq_conditioner"]["block_stride_seconds"]=.5
    def canonical(path,scenario,start,end,onset=None):
      times=np.arange(start,end+.001,.25);coords=r.core.CANONICAL_TAP_COORDS
      peaks=[]
      for i,t in enumerate(times):
       shape=np.exp(-4*(coords-.015*np.sin(t/13))**2)*(1+.03*np.cos(t/9))
       if onset is not None and t>=onset: shape=shape+.12*np.exp(-18*(coords-.3)**2)
       peaks.append(np.stack((shape,np.zeros(9)),axis=1))
      np.savez(path,complex_iq=np.asarray(peaks,np.float32),time_s=times,prn=np.ones(len(times),int),
       channel=np.zeros(len(times),int),segment_index=np.zeros(len(times),int),sample_count=np.arange(len(times),dtype=np.uint64))
    canonical_inputs={}
    for scenario in ("cleanStatic","DS1","DS2","DS3","DS7","DS8"):
      path=tmp_path/f"{scenario}.npz";onset=config["attacks"]["onsets_seconds"].get(scenario)
      canonical(path,scenario,0. if scenario=="cleanStatic" else 70.,436. if scenario=="cleanStatic" else 156.,onset)
      canonical_inputs[scenario]={"path":str(path),"sha256":r.sha256_file(path)}
    config["canonical_inputs"]=canonical_inputs
    # Exact cleanDynamic node schema.
    dynamic=tmp_path/"cleanDynamic.csv";columns=[f"tap_{x}_rel_prompt_mean" for x in r.TAP_NAMES]
    fields=["run_id","source_fingerprint","prn","channel","segment_index","window_index","window_bin_s",
      "window_start_s","window_end_s","window_mid_s","epoch_count","tap_count","tap_layout",*columns]
    with dynamic.open("w",newline="") as f:
      w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
      for i,start in enumerate(np.arange(0.,61.,.5)):
       peak=np.exp(-4*(r.core.CANONICAL_TAP_COORDS-.01*np.sin(start/7))**2)
       row={"run_id":"cleanDynamic","source_fingerprint":"synthetic","prn":"G01","channel":0,"segment_index":0,
        "window_index":i,"window_bin_s":start+.5,"window_start_s":start,"window_end_s":start+1.,"window_mid_s":start+.5,
        "epoch_count":5,"tap_count":9,"tap_layout":",".join(r.TAP_NAMES)};row.update(dict(zip(columns,peak)));w.writerow(row)
    config["clean_dynamic_nodes"]={"path":str(dynamic),"sha256":r.sha256_file(dynamic),"evaluation_only":True,"required":True}
    # Small deterministic raw IQ files, with exact witness/full hashes.
    raw={}
    for si,scenario in enumerate(("cleanStatic","cleanDynamic","DS1","DS2","DS3","DS7","DS8")):
      seconds=438 if scenario=="cleanStatic" else (64 if scenario=="cleanDynamic" else 158)
      n=seconds*1000;index=np.arange(n);values=np.stack((((index+si*3)%97)-48,((index*3+si)%89)-44),axis=1).astype("<i2")
      path=tmp_path/f"{scenario}.bin";values.tofile(path);witness=r.raw_iq_witness(path)
      raw[scenario]={"path":str(path),"size_bytes":path.stat().st_size,"sha256":r.sha256_file(path),
       "first_1MiB_sha256":witness["first_1MiB_sha256"],"last_1MiB_sha256":witness["last_1MiB_sha256"],
       "default_full_hash_status":"expected_not_recomputed"}
    config["raw_iq_inputs"]=raw
    config["legacy_positive_control"].update({"node_path":str(tmp_path/"missing_nodes.csv"),
      "score_path":str(tmp_path/"missing_scores.csv")})
    config_path=tmp_path/"config.json";config_path.write_text(json.dumps(config))
    out=tmp_path/"campaign"
    result=r.run_campaign(config_path,out,checkpoint=checkpoint,device="cpu",_test_fixture=True)
    assert result==out and (out/"decision.json").is_file() and (out/"plots"/"roc.png").is_file()
    private=v.verify_test_artifact(out,verify_source=False);assert private["ok"],private["errors"]
    public=v.verify_artifact(out,verify_source=False);assert not public["ok"]
    assert any("test_fixture" in x for x in public["errors"])
    provenance=json.loads((out/"provenance.json").read_text());assert provenance["test_fixture"] is True
    metrics=list(csv.DictReader((out/"scenario_metrics.csv").open()));assert any(x["metric"]=="sustained_delay" for x in metrics)



def _phase_scored_row(prn, start, end, *, scenario="DS1", score=1.0):
    from types import SimpleNamespace
    target=SimpleNamespace(event_key=(scenario, 79.5), source_start_s=start, source_end_s=end)
    pair=SimpleNamespace(identity=SimpleNamespace(
        scenario=scenario, prn=prn, target_index=0, availability_time_s=end))
    return {"example":SimpleNamespace(target=target), "pair":pair,
            "scores":{method:score for method in r.METHODS},
            "source_start_s":start, "source_end_s":end, "pair_sequence_index":0}


def test_event_phase_is_classified_once_from_aggregate_prn_support(monkeypatch):
    """An event is excluded when any linked PRN extends across a phase boundary."""
    scored=[_phase_scored_row("G01",79.0,80.0),
            _phase_scored_row("G02",79.1,80.1,score=2.0)]
    calls=[]
    original=r.event_phase
    def observed(*args,**kwargs):
        calls.append((args,kwargs));return original(*args,**kwargs)
    monkeypatch.setattr(r,"event_phase",observed)
    rows,events=r.aggregate_scored(scored,onsets={"DS1":100.})
    assert calls==[(("DS1",79.0,80.1,{"DS1":100.}),{})]
    assert events[0]["meta"]["phase"]=="transition_excluded"
    assert events[0]["meta"]["label"]==""
    assert events[0]["meta"]["valid"] is False
    event=next(x for x in rows if x["row_level"]=="event")
    prns=[x for x in rows if x["row_level"]=="prn"]
    assert (event["source_start_s"],event["source_end_s"],event["availability_time_s"])==(79.0,80.1,80.1)
    assert {(x["prn"],x["source_start_s"],x["source_end_s"]) for x in prns}=={
        ("G01",79.0,80.0),("G02",79.1,80.1)}
    assert {(x["phase"],x["label"],x["valid"]) for x in rows}=={
        ("transition_excluded","",False)}


def test_score_scenario_carries_support_but_no_authoritative_phase():
    from types import SimpleNamespace
    peak,std,cov,state=_sealed_score_state()
    residual=np.cos(np.arange(9))*.03
    ident=r.identity("DS1","DS1","G01",0,101.)
    pair=r.core.PeakPredictionPair(peak+residual,peak,residual/std,std,ident,
      r.core.RAW_SPACE,r.core.RAW_SPACE,r.core.STANDARDIZED_SPACE,r.core.CANONICAL_TAP_COORDS)
    target=SimpleNamespace(source_start_s=100.,source_end_s=101.)
    scored=r.score_scenario([SimpleNamespace(target=target)],[pair],np.ones((1,4)),state)
    assert "phase" not in scored[0] and "label" not in scored[0]
    assert (scored[0]["source_start_s"],scored[0]["source_end_s"])==(100.,101.)


def _write_phase_rows(root, rows, onsets={"DS1":100.}):
    root.mkdir()
    (root/"config.json").write_text(json.dumps({"attacks":{"onsets_seconds":onsets}}))
    with (root/"per_epoch_scores.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(x.keys() for x in rows))))
        w.writeheader();w.writerows(rows)


def test_independent_verifier_recomputes_event_phase_and_rejects_prn_tamper(tmp_path):
    base={"scenario":"DS1","physical_recording_id":"DS1","event_id":"DS1@79.5","target_index":"0",
      "role":"evaluation","phase":"transition_excluded","label":"","valid":"false","tracked_prn_count":"2"}
    rows=[
      {**base,"row_level":"prn","prn":"G01","availability_time_s":"80.0","source_start_s":"79.0","source_end_s":"80.0","B0":"1"},
      {**base,"row_level":"prn","prn":"G02","availability_time_s":"80.1","source_start_s":"79.1","source_end_s":"80.1","B0":"2"},
      {**base,"row_level":"event","prn":"","availability_time_s":"80.1","source_start_s":"79.0","source_end_s":"80.1",
       "B0_median":"1.5","B0_top25_mean":"2"},
    ]
    good=tmp_path/"good";_write_phase_rows(good,rows)
    errors=[];result=v.verify_epoch_rows(good,errors)
    assert not errors and result["events_reaggregated"]==1
    mutations=[]
    tampered=[dict(x) for x in rows];tampered[0].update(phase="stable_pre",label="0",valid="true");mutations.append(tampered)
    tampered=[dict(x) for x in rows];tampered[2].update(phase="post",label="1",valid="true");mutations.append(tampered)
    tampered=[dict(x) for x in rows];tampered[2]["source_end_s"]="80.0";mutations.append(tampered)
    for i,tampered in enumerate(mutations):
        bad=tmp_path/f"bad{i}";_write_phase_rows(bad,tampered)
        errors=[];v.verify_epoch_rows(bad,errors)
        assert errors,i
